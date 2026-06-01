"""
Calibration tools for SBI posteriors: SBC, coverage test, and posterior predictive check.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

from .sampler import sample_nre_posterior, fix_posterior_transforms, draw_posterior_samples

import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)


class SBCRunner:
    """
    Rank-based SBC (Talts et al. 2018).

    For each test draw (theta*, x*): simulate x*, sample N_post draws from
    posterior(x*), record rank of theta*[j] in the marginal samples.
    A well-calibrated posterior produces uniform ranks on [0, N_post].
    """

    def __init__(
        self,
        posterior,
        simulator: Callable,
        prior,
        n_sbc: int = 500,
        n_post: int = 1000,
        n_candidates: int = 20000,
        batch_size: int = 200,
        param_names: Optional[List[str]] = None,
    ):
        self.posterior = posterior
        self.simulator = simulator
        self.prior = prior
        self.n_sbc = n_sbc
        self.n_post = n_post
        self.n_candidates = n_candidates
        self.batch_size = batch_size
        self.param_names = param_names

    def run(self, seed: int = 0) -> np.ndarray:
        """returns ranks : (n_sbc, d_theta) int array."""
        import torch
        torch.manual_seed(seed)
        fix_posterior_transforms(self.posterior, prior=self.prior)

        theta_star = self.prior.sample((self.n_sbc,)).cpu().numpy()
        d_theta = theta_star.shape[1]
        ranks = np.empty((self.n_sbc, d_theta), dtype=int)

        for i in range(self.n_sbc):
            if i % 10 == 0:
                log.info(f"SBC {i}/{self.n_sbc}")
            th = theta_star[i : i + 1]
            x_sim = np.asarray(self.simulator(th), dtype=np.float32)
            x_t = torch.as_tensor(x_sim[0], dtype=torch.float32)
            try:
                samples = draw_posterior_samples(
                    self.posterior, x_t, self.n_post, seed=seed + i
                ).numpy()
            except Exception:
                samples = sample_nre_posterior(
                    self.posterior, self.prior, x_t,
                    n_samples=self.n_post,
                    n_candidates=self.n_candidates,
                    batch_size=self.batch_size,
                    seed=seed + i,
                ).numpy()
            for j in range(d_theta):
                ranks[i, j] = int((samples[:, j] < th[0, j]).sum())

        return ranks

    def plot(
        self,
        ranks: np.ndarray,
        out_path: Optional[Union[str, Path]] = None,
        bins: int = 20,
    ) -> plt.Figure:
        d = ranks.shape[1]
        names = self.param_names or [f"θ_{j}" for j in range(d)]
        ncols = min(d, 4)
        nrows = (d + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                                  squeeze=False)
        uniform_level = self.n_sbc / bins
        for j, ax in enumerate(axes.flat):
            if j >= d:
                ax.set_visible(False)
                continue
            ax.hist(ranks[:, j], bins=bins, range=(0, self.n_post),
                    color="#4878CF", alpha=0.8, edgecolor="white")
            ax.axhline(uniform_level, color="k", ls="--", lw=1.5)
            ax.set_title(names[j])
            ax.set_xlabel("rank")
            ax.set_ylabel("count")
        fig.tight_layout()
        if out_path is not None:
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
        return fig


class CoverageTest:
    """
    Check that the α-credible region contains the true parameter α% of the time.

    For each test point, checks whether theta* falls inside the HPD at each
    credible level. Plots expected coverage vs. nominal level.
    """

    def __init__(
        self,
        posterior,
        simulator: Callable,
        prior,
        n_test: int = 500,
        n_post: int = 2000,
        n_candidates: int = 2000,
        batch_size: int = 200,
        levels: Optional[np.ndarray] = None,
        param_names: Optional[List[str]] = None,
    ):
        self.posterior = posterior
        self.simulator = simulator
        self.prior = prior
        self.n_test = n_test
        self.n_post = n_post
        self.n_candidates = n_candidates
        self.batch_size = batch_size
        self.levels = levels if levels is not None else np.linspace(0.05, 0.95, 19)
        self.param_names = param_names

    def run(self, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """returns (levels, coverage) where coverage is (n_levels, d_theta)."""
        import torch
        torch.manual_seed(seed)
        fix_posterior_transforms(self.posterior, prior=self.prior)
        theta_star = self.prior.sample((self.n_test,)).cpu().numpy()
        d_theta = theta_star.shape[1]
        inside = np.zeros((len(self.levels), d_theta), dtype=float)

        for i in range(self.n_test):
            th = theta_star[i : i + 1]
            x_sim = np.asarray(self.simulator(th), dtype=np.float32)
            x_t = torch.as_tensor(x_sim[0], dtype=torch.float32)
            try:
                samples = draw_posterior_samples(
                    self.posterior, x_t, self.n_post, seed=seed + i
                ).numpy()
            except Exception:
                samples = sample_nre_posterior(
                    self.posterior, self.prior, x_t,
                    n_samples=self.n_post,
                    n_candidates=self.n_candidates,
                    batch_size=self.batch_size,
                    seed=seed + i,
                ).numpy()
            for j in range(d_theta):
                s_j = samples[:, j]
                for k, lv in enumerate(self.levels):
                    lo_k = np.percentile(s_j, 50 * (1 - lv))
                    hi_k = np.percentile(s_j, 50 * (1 + lv))
                    inside[k, j] += float(lo_k <= th[0, j] <= hi_k)

        coverage = inside / self.n_test
        return self.levels, coverage

    def plot(
        self,
        levels: np.ndarray,
        coverage: np.ndarray,
        out_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        d = coverage.shape[1]
        names = self.param_names or [f"θ_{j}" for j in range(d)]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="ideal")
        for j in range(d):
            ax.plot(levels, coverage[:, j], marker="o", ms=4, label=names[j])
        ax.set_xlabel("Nominal credible level")
        ax.set_ylabel("Empirical coverage")
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        if out_path is not None:
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
        return fig


class PosteriorPredictiveCheck:
    """
    Draw theta from posterior(x_obs), simulate x_rep, compare to x_obs.

    Usage:
        ppc = PosteriorPredictiveCheck(posterior, simulator, x_obs, n_draws=200)
        x_rep = ppc.run()
        ppc.plot(x_rep, feature_names=["count_bin0", ...])
    """

    def __init__(
        self,
        posterior,
        simulator: Callable,
        x_obs: np.ndarray,
        prior=None,
        n_draws: int = 200,
        n_candidates: int = 20000,
    ):
        self.posterior = posterior
        self.simulator = simulator
        self.x_obs = np.asarray(x_obs, dtype=np.float32)
        self.prior = prior
        self.n_draws = n_draws
        self.n_candidates = n_candidates

    def run(self, seed: int = 0) -> np.ndarray:
        """returns x_rep : (n_draws, d_x) posterior predictive replicates."""
        import torch
        torch.manual_seed(seed)
        x_t = torch.as_tensor(self.x_obs, dtype=torch.float32)
        theta_draws = sample_nre_posterior(
            self.posterior, self.prior, x_t,
            n_samples=self.n_draws,
            n_candidates=self.n_candidates,
            seed=seed,
        ).numpy()
        x_rep = np.array([
            np.asarray(self.simulator(theta_draws[i : i + 1]), dtype=np.float32).ravel()
            for i in range(self.n_draws)
        ])
        return x_rep

    def plot(
        self,
        x_rep: np.ndarray,
        feature_names: Optional[List[str]] = None,
        out_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        d = x_rep.shape[1]
        names = feature_names or [f"x_{i}" for i in range(d)]
        ncols = min(d, 4)
        nrows = (d + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                                  squeeze=False)
        for i, ax in enumerate(axes.flat):
            if i >= d:
                ax.set_visible(False)
                continue
            ax.hist(x_rep[:, i], bins=30, color="#4878CF", alpha=0.7,
                    density=True, label="predictive")
            ax.axvline(self.x_obs[i], color="crimson", lw=2, label="observed")
            ax.set_title(names[i])
            ax.legend(fontsize=8)
        fig.tight_layout()
        if out_path is not None:
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
        return fig
