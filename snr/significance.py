"""
Significance tools: survival function, p-value calibration, and discovery potential.
"""

from __future__ import annotations

import math
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


@dataclass
class SurvivalResult:
    edges: np.ndarray          # (n_bins+1,) TS bin edges
    S: np.ndarray              # (n_bins,) empirical survival fraction S(λ) = P(>λ)
    S_lo: np.ndarray           # lower 68% Wilson band
    S_hi: np.ndarray           # upper 68% Wilson band
    N: int                     # total null samples
    tail_A: Optional[float]    # exponential tail amplitude
    tail_b: Optional[float]    # exponential tail rate
    tail_ts_min: Optional[float]


def _wilson_interval(k: np.ndarray, N: int, z: float = 1.0):
    """Wilson score interval for Binomial proportion k/N at z-sigma."""
    k = np.asarray(k, float)
    phat = k / float(N)
    denom = 1.0 + z * z / float(N)
    center = (phat + z * z / (2.0 * float(N))) / denom
    half = (z / denom) * np.sqrt(
        phat * (1.0 - phat) / float(N) + z * z / (4.0 * float(N) ** 2)
    )
    return np.clip(center - half, 0.0, 1.0), np.clip(center + half, 0.0, 1.0)


def survival_function(
    null_ts: np.ndarray,
    ts_max: Optional[float] = None,
    n_bins: int = 100,
) -> SurvivalResult:
    """empirical survival function S(λ) = P(>λ) from null TS samples."""
    null_ts = np.asarray(null_ts, float).ravel()
    N = len(null_ts)
    ts_max = ts_max if ts_max is not None else float(null_ts.max() * 1.05)
    edges = np.linspace(0.0, ts_max, n_bins + 1)

    tail_counts = np.array([(null_ts > e).sum() for e in edges[:-1]], dtype=float)
    S = tail_counts / N
    lo, hi = _wilson_interval(tail_counts, N)

    return SurvivalResult(
        edges=edges, S=S, S_lo=lo, S_hi=hi, N=N,
        tail_A=None, tail_b=None, tail_ts_min=None,
    )


def fit_exponential_tail(
    result: SurvivalResult,
    ts_fit_min: float,
    ts_fit_max: Optional[float] = None,
) -> SurvivalResult:
    """fit S(λ) ≈ A exp(-b λ) over [ts_fit_min, ts_fit_max] via log-linear OLS."""
    edges = result.edges[:-1]
    ts_fit_max = ts_fit_max if ts_fit_max is not None else float(edges[-1])

    mask = (edges >= ts_fit_min) & (edges <= ts_fit_max) & (result.S > 0)
    if mask.sum() < 3:
        raise ValueError(
            f"Only {mask.sum()} bins in [{ts_fit_min}, {ts_fit_max}] with S>0; "
            "need at least 3 for an exponential tail fit."
        )

    x = edges[mask]
    y = np.log(result.S[mask])
    coeffs = np.polyfit(x, y, 1)
    b = -float(coeffs[0])
    A = math.exp(float(coeffs[1]))

    return dataclasses.replace(result, tail_A=A, tail_b=b, tail_ts_min=ts_fit_min)


def ts_to_pvalue(
    ts: Union[float, np.ndarray],
    result: SurvivalResult,
    use_tail_fit: bool = True,
) -> np.ndarray:
    """TS → one-sided p-value via empirical survival (with optional tail fit)."""
    ts = np.atleast_1d(np.asarray(ts, float))
    p = np.interp(ts, result.edges[:-1], result.S, left=1.0, right=0.0)

    if use_tail_fit and result.tail_A is not None:
        in_tail = ts >= result.tail_ts_min
        p[in_tail] = result.tail_A * np.exp(-result.tail_b * ts[in_tail])

    return np.clip(p, 0.0, 1.0)


def pvalue_to_sigma(pvalue: Union[float, np.ndarray]) -> np.ndarray:
    """one-sided p-value → Gaussian significance in σ."""
    pvalue = np.asarray(pvalue, float)
    sigma = -norm.ppf(np.clip(pvalue, 1e-300, 1.0))
    return np.where(pvalue <= 0.0, np.inf, sigma)


def one_sided_p_from_sigma(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def ts_threshold(null_result: SurvivalResult, target_sigma: float) -> float:
    """TS value corresponding to target_sigma significance."""
    p_thresh = one_sided_p_from_sigma(target_sigma)
    S = null_result.S
    edges = null_result.edges[:-1]

    if null_result.tail_A is not None and p_thresh < null_result.S[null_result.edges[:-1] >= null_result.tail_ts_min][0]:
        return float(-math.log(p_thresh / null_result.tail_A) / null_result.tail_b)

    # invert S(λ): find λ s.t. S(λ) = p_thresh
    return float(np.interp(p_thresh, S[::-1], edges[::-1]))


def plot_survival(
    result: SurvivalResult,
    ax: Optional[plt.Axes] = None,
    sigma_lines: Tuple[float, ...] = (3.0, 5.0),
    label: str = "null",
    out_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    edges = result.edges[:-1]
    ax.step(edges, result.S, where="post", color="#2166ac", lw=1.8, label=label)
    ax.fill_between(edges, result.S_lo, result.S_hi,
                    step="post", alpha=0.25, color="#2166ac")

    if result.tail_A is not None:
        ts_tail = np.linspace(result.tail_ts_min, float(edges[-1]), 300)
        ax.plot(ts_tail, result.tail_A * np.exp(-result.tail_b * ts_tail),
                "k--", lw=1.5, label="exp. tail fit")

    for sig in sigma_lines:
        p = one_sided_p_from_sigma(sig)
        ax.axhline(p, color="gray", ls=":", lw=1.2)
        ax.text(float(edges[-1]) * 0.98, p * 1.3, f"{sig:.0f}σ",
                ha="right", fontsize=9, color="gray")

    ax.set_yscale("log")
    ax.set_xlabel("TS (λ)")
    ax.set_ylabel(r"$S(\lambda) = P(>\lambda)$")
    ax.legend()
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def discovery_potential(
    mu_grid: np.ndarray,
    ts_at_mu: np.ndarray,
    null_result: SurvivalResult,
    target_sigma: float = 5.0,
    fraction: float = 0.5,
) -> float:
    """
    Minimum signal strength μ such that `fraction` of signal trials exceed
    the null threshold at `target_sigma`.

    ts_at_mu can be (M,) medians or (M, K) for K signal trials per μ.
    """
    mu_grid = np.asarray(mu_grid, float)
    ts_at_mu = np.asarray(ts_at_mu, float)
    thresh = ts_threshold(null_result, target_sigma)

    if ts_at_mu.ndim == 1:
        frac_above = (ts_at_mu >= thresh).astype(float)
    else:
        frac_above = (ts_at_mu >= thresh).mean(axis=1)

    above = frac_above >= fraction
    if not above.any():
        return float("inf")
    return float(mu_grid[above][0])
