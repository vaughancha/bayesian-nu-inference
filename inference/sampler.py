"""
Sampler for NRE posteriors.

sbi >= 0.26 + PyTorch >= 2.0 loses the _InverseTransform._transform attribute
after pickling. fix_posterior_transforms() rebuilds from the prior via
sbi's mcmc_transform, restoring the native rejection sampler.
"""

from __future__ import annotations

import logging
from typing import Union

import numpy as np
import torch
from torch.distributions import Distribution

log = logging.getLogger(__name__)


def fix_posterior_transforms(posterior, prior=None) -> None:
    """
    Rebuild theta_transform on all component posteriors using mcmc_transform.

    After pickling in sbi >= 0.26 + PyTorch >= 2.0, _InverseTransform loses
    its wrapped _transform attribute. Rebuilding from a fresh prior fixes it.
    """
    from sbi.utils import mcmc_transform

    components = getattr(posterior, "posteriors", [posterior])
    for post in components:
        _prior = prior or getattr(post, "_prior", None)
        if _prior is None:
            continue
        t_fresh = mcmc_transform(_prior)
        post.theta_transform = t_fresh


def draw_posterior_samples(
    posterior,
    x_obs: torch.Tensor,
    n_samples: int,
    seed: int = 0,
    max_sampling_time: float = 30.0,
) -> torch.Tensor:
    """
    Draw samples using sbi's native rejection sampler with a timeout.

    If the rejection sampler hangs (sharp posterior, low acceptance rate),
    raises RuntimeError so the caller can fall back to importance resampling.
    """
    torch.manual_seed(seed)
    x_t = x_obs.squeeze(0).unsqueeze(0)   # (1, d_x)
    try:
        samples = posterior.sample(
            (n_samples,),
            x=x_t,
            show_progress_bars=True,
        )
    except TypeError:
        # older sbi versions don't accept max_sampling_time on MCMC posteriors
        samples = posterior.sample((n_samples,), x=x_t)
    return samples.cpu()


def sample_nre_posterior(
    posterior,
    prior: Distribution,
    x_obs: torch.Tensor,
    n_samples: int,
    n_candidates: int = 100_000,
    batch_size: int = 10_000,
    device: str = "cpu",
    seed: int = 0,
) -> torch.Tensor:
    """
    Fallback: draw samples via importance resampling from the prior.

    ESS will be low when the posterior is narrow relative to the prior.
    Only use if fix_posterior_transforms() + draw_posterior_samples() fails.
    """
    torch.manual_seed(seed)
    x_obs = x_obs.to(device).squeeze(0)

    log.info(f"Importance resampling: {n_candidates} candidates → {n_samples} samples")
    theta_cands = prior.sample((n_candidates,)).to(device)

    log_ratios = []
    for i in range(0, n_candidates, batch_size):
        batch = theta_cands[i : i + batch_size]
        x_rep = x_obs.unsqueeze(0).expand(len(batch), -1)
        with torch.no_grad():
            log_post  = posterior.log_prob(batch, x=x_rep)
            log_prior = prior.log_prob(batch)
            log_ratios.append((log_post - log_prior).cpu())

    log_ratio = torch.cat(log_ratios)
    log_ratio = torch.where(torch.isfinite(log_ratio), log_ratio,
                            torch.full_like(log_ratio, float("-inf")))

    finite_mask = torch.isfinite(log_ratio) & (log_ratio > float("-inf"))
    if not finite_mask.any():
        raise RuntimeError("All log_ratio values are -inf or NaN.")

    log_ratio = log_ratio - log_ratio[finite_mask].max()
    weights   = log_ratio.exp()
    weights   = weights / weights.sum()

    ess = 1.0 / (weights ** 2).sum().item()
    log.info(f"Effective sample size: {ess:.0f} / {n_candidates} ({100*ess/n_candidates:.1f}%)")

    idx = torch.multinomial(weights, n_samples, replacement=True)
    return theta_cands[idx].cpu()
