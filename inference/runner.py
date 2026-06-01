"""
SBI inference runner — compatible with sbi >= 0.26.

Supports NRE (binary classifier) and NPE (normalising flow), single-round
training with ensemble posteriors weighted by validation loss.
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.distributions import Distribution

try:
    from sbi.inference.posteriors import EnsemblePosterior
except ImportError:
    from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble as EnsemblePosterior

from .loader import _BaseLoader

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


_ENGINE_CLS = {
    "NRE":  "NRE_B",
    "NPE":  "NPE_C",
    "NLE":  "NLE_A",
    "SNRE": "NRE_B",
    "SNPE": "NPE_C",
}


def _get_sbi_cls(engine: str):
    import sbi.inference as sbi_inf
    name = _ENGINE_CLS.get(engine.upper(), engine)
    cls = getattr(sbi_inf, name, None)
    if cls is None:
        raise ValueError(f"Unknown sbi engine '{engine}'. Use NRE, NPE, or NLE.")
    return cls


class SBIRunner:
    """train a single-round SBI posterior ensemble."""

    def __init__(
        self,
        prior: Distribution,
        engine: str = "NRE",
        architecture: str = "mlp",
        n_ensemble: int = 3,
        train_args: Dict = {},
        out_dir: Union[str, Path, None] = None,
        device: str = "cpu",
        embedding_net: Optional[nn.Module] = None,
        name: str = "",
    ):
        self.prior = prior
        self.engine = engine.upper()
        self.architecture = architecture
        self.n_ensemble = n_ensemble
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.device = device
        self.embedding_net = embedding_net
        self.name = name

        self.train_args = dict(
            training_batch_size=256,
            learning_rate=5e-4,
            stop_after_epochs=20,
            validation_fraction=0.1,
            clip_max_norm=5.0,
        )
        self.train_args.update(train_args)

    def _make_trainer(self):
        cls = _get_sbi_cls(self.engine)
        kwargs = dict(prior=self.prior, device=self.device,
                      show_progress_bars=False)
        if "NRE" in self.engine:
            kwargs["classifier"] = self.architecture
        else:
            kwargs["density_estimator"] = self.architecture
        return cls(**kwargs)

    def _build_posterior(self, trainer, estimator):
        if "NRE" in self.engine:
            return trainer.build_posterior(
                estimator,
                sample_with="mcmc",
                mcmc_method="slice_np_vectorized",
                mcmc_parameters={"num_chains": 10, "warmup_steps": 100},
            )
        return trainer.build_posterior(estimator)

    def _fix_transform(self, posterior):
        """
        Rebuild theta_transform from the prior.
        In sbi >= 0.26 + PyTorch >= 2.0, _InverseTransform._inv becomes None
        after pickling; rebuilding from the live prior fixes it.
        """
        try:
            from sbi.utils import mcmc_transform
            t = mcmc_transform(self.prior, device=self.device)
            posterior.theta_transform = t
        except Exception:
            pass

    def _train_one(self, x: torch.Tensor, theta: torch.Tensor):
        """train a single model, returns (posterior, summary_dict)."""
        trainer = self._make_trainer()

        x_in = x
        if self.embedding_net is not None:
            with torch.no_grad():
                x_in = self.embedding_net(x_in.to(self.device))

        if "NRE" in self.engine:
            trainer.append_simulations(theta, x_in)
        else:
            trainer.append_simulations(theta, x_in, proposal=self.prior)

        train_kwargs = {k: v for k, v in self.train_args.items()
                        if k in ("training_batch_size", "learning_rate",
                                 "stop_after_epochs", "validation_fraction",
                                 "clip_max_norm", "max_num_epochs")}
        estimator = trainer.train(**train_kwargs)
        posterior  = self._build_posterior(trainer, estimator)
        self._fix_transform(posterior)
        summary    = dict(trainer._summary)
        return posterior, summary

    def _ensemble_weight(self, summary: dict) -> float:
        """higher val log-prob → higher weight."""
        if "best_validation_loss" in summary:
            return -float(summary["best_validation_loss"][-1])
        if "best_validation_log_prob" in summary:
            return float(summary["best_validation_log_prob"][-1])
        return 0.0

    def _save(self, ensemble, summaries):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        pkl = self.out_dir / f"{self.name}posterior.pkl"
        jsn = self.out_dir / f"{self.name}summary.json"
        with open(pkl, "wb") as f:
            pickle.dump(ensemble, f)
        with open(jsn, "w") as f:
            json.dump(summaries, f, default=str)
        log.info(f"Saved → {pkl}")

    def __call__(self, loader: _BaseLoader, seed: Optional[int] = None):
        if seed is not None:
            torch.manual_seed(seed)

        x     = torch.as_tensor(loader.get_all_data(),       dtype=torch.float32)
        theta = torch.as_tensor(loader.get_all_parameters(), dtype=torch.float32)

        posteriors, summaries = [], []
        t0 = time.time()
        for i in range(self.n_ensemble):
            log.info(f"Training model {i+1}/{self.n_ensemble} ({self.engine})")
            torch.manual_seed((seed or 0) + i)
            post, summ = self._train_one(x, theta)
            posteriors.append(post)
            summaries.append(summ)

        weights = torch.tensor([self._ensemble_weight(s) for s in summaries])
        weights = torch.softmax(weights, dim=0)

        ensemble = EnsemblePosterior(posteriors=posteriors, weights=weights)
        log.info(f"Ensemble trained in {time.time()-t0:.1f}s | weights={weights.numpy().round(3)}")

        if self.out_dir is not None:
            self._save(ensemble, summaries)

        return ensemble, summaries

    @staticmethod
    def load_posterior(path: Union[str, Path]):
        with open(path, "rb") as f:
            return pickle.load(f)


class SBIRunnerSequential(SBIRunner):
    """multi-round SNPE/SNRE — tightens the proposal each round."""

    def __init__(self, *args, num_rounds: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_rounds = num_rounds

    def __call__(self, loader: _BaseLoader, seed: Optional[int] = None):
        if not (hasattr(loader, "simulate") and hasattr(loader, "get_obs_data")):
            raise TypeError("Sequential runner requires loader.simulate() and .get_obs_data()")

        if seed is not None:
            torch.manual_seed(seed)

        x_obs     = torch.as_tensor(loader.get_obs_data(), dtype=torch.float32)
        proposal  = self.prior

        if len(loader) > 0:
            x     = torch.as_tensor(loader.get_all_data(),       dtype=torch.float32)
            theta = torch.as_tensor(loader.get_all_parameters(), dtype=torch.float32)
        else:
            theta_np, x_np = loader.simulate(proposal)
            theta = torch.as_tensor(theta_np, dtype=torch.float32)
            x     = torch.as_tensor(x_np,     dtype=torch.float32)

        ensemble, summaries = None, None
        t0 = time.time()
        for rnd in range(self.num_rounds):
            log.info(f"Sequential round {rnd+1}/{self.num_rounds}")
            posteriors, summaries = [], []
            for i in range(self.n_ensemble):
                torch.manual_seed((seed or 0) + rnd * 100 + i)
                post, summ = self._train_one(x, theta)
                posteriors.append(post)
                summaries.append(summ)
            weights  = torch.softmax(
                torch.tensor([self._ensemble_weight(s) for s in summaries]), 0
            )
            ensemble = EnsemblePosterior(posteriors=posteriors, weights=weights)
            proposal = ensemble.set_default_x(x_obs)

            if rnd < self.num_rounds - 1:
                theta_np, x_np = loader.simulate(proposal)
                theta = torch.as_tensor(theta_np, dtype=torch.float32)
                x     = torch.as_tensor(x_np,     dtype=torch.float32)

        log.info(f"Sequential training done in {time.time()-t0:.1f}s")
        if self.out_dir is not None:
            self._save(ensemble, summaries)
        return ensemble, summaries
