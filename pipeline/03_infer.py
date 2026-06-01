#!/usr/bin/env python3
"""
Bayesian inference and constraining plots.

Uses the trained NRE posterior to sample p(θ | x_obs) and produce:
  - constraining panel: prior vs posterior per parameter
  - likelihood ratio profiles: log r(θ, x_obs) per parameter
  - corner plot: joint posteriors

Outputs:
  out/inference/posterior_samples.npy     (N, 6)
  out/inference/constraining.png
  out/inference/likelihood_ratio.png
  out/inference/corner.png
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from inference.sampler import fix_posterior_transforms, draw_posterior_samples, sample_nre_posterior
import matplotlib.ticker as mticker
from scipy.stats import gaussian_kde

from population.physics import CosmologyGrid, ObservationParams
from population.simulator import PopulationSimulator, PopParamSpace, SummarySpec


def ic86_aeff(E: np.ndarray, delta: float) -> np.ndarray:
    E = np.asarray(E, float)
    return np.clip(1e4 * (E / 1e5) ** 0.5 * max(0.0, np.cos(float(delta))), 0.0, 1e6)


def _prior_bounds(param_space: PopParamSpace):
    attrs = [
        "log10_n0_bounds", "gamma_bounds", "log10_L_bounds",
        "p1_bounds", "p2_bounds", "zc_bounds",
    ]
    return [getattr(param_space, a) for a in attrs]


def plot_constraining(
    prior_samples: np.ndarray,
    post_samples: np.ndarray,
    names: list,
    bounds: list,
    truths=None,
    out_path=None,
):
    """one panel per parameter showing prior, posterior, 68%/95% CIs, and truth."""
    d = len(names)
    fig, axes = plt.subplots(1, d, figsize=(3.2 * d, 3.8), sharey=False)
    fig.suptitle(
        r"Bayesian constraining:  $p(\theta\,|\,x_{\rm obs})\;\propto\;p(\theta)\;\times\;r(\theta,\,x_{\rm obs})$",
        fontsize=11, y=1.02,
    )

    for j, (ax, name, bnd) in enumerate(zip(axes, names, bounds)):
        lo, hi = bnd
        prior_w = hi - lo

        s = post_samples[:, j]
        p16, p50, p84, p2, p98 = np.percentile(s, [16, 50, 84, 2.5, 97.5])
        post_68w = p84 - p16
        factor = prior_w / post_68w if post_68w > 0 else float("inf")

        xs = np.linspace(lo, hi, 400)
        try:
            kde = gaussian_kde(s, bw_method="scott")
            ys = kde(xs)
        except Exception:
            ys = np.zeros_like(xs)

        ax.axhspan(0, 0, alpha=0)
        ax.fill_between([lo, hi], [0, 0], [max(ys) * 1.15] * 2,
                        color="lightgray", alpha=0.4, label="Prior", zorder=1)
        ax.axvline(lo, color="gray", lw=0.8, ls="--")
        ax.axvline(hi, color="gray", lw=0.8, ls="--")

        mask95 = (xs >= p2) & (xs <= p98)
        ax.fill_between(xs[mask95], 0, ys[mask95],
                        color="#4878CF", alpha=0.20, zorder=2)
        mask68 = (xs >= p16) & (xs <= p84)
        ax.fill_between(xs[mask68], 0, ys[mask68],
                        color="#4878CF", alpha=0.45, zorder=3)

        ax.plot(xs, ys, color="#2155a0", lw=2.0, label="Posterior", zorder=4)
        ax.axvline(p50, color="#2155a0", lw=1.2, ls=":", zorder=5)

        if truths is not None:
            ax.axvline(truths[j], color="crimson", lw=1.8, zorder=6,
                       label=f"Truth = {truths[j]:.2f}")

        ax.set_xlabel(name, fontsize=10)
        ax.set_xlim(lo, hi)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_visible(False)

        ax.set_title(f"×{factor:.1f} constraint", fontsize=8.5, color="#333333")

        for val, label in [(p16, "16%"), (p84, "84%")]:
            ax.annotate("", xy=(val, 0), xytext=(val, max(ys) * 0.06),
                        arrowprops=dict(arrowstyle="-", color="#2155a0", lw=1.0))

        if j == 0:
            ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
    return fig


def plot_likelihood_ratio_profile(
    posterior,
    x_obs_t: torch.Tensor,
    post_samples: np.ndarray,
    names: list,
    bounds: list,
    prior=None,
    truths=None,
    n_grid: int = 80,
    out_path=None,
):
    """
    log r(θ, x_obs) profile per parameter.
    sweeps each parameter over the prior range while holding others at posterior median.
    """
    d = len(names)
    medians = np.median(post_samples, axis=0)

    fig, axes = plt.subplots(1, d, figsize=(3.2 * d, 3.2))
    fig.suptitle(
        r"Learned likelihood ratio  $\log\,r(\theta,\,x_{\rm obs})$  per parameter"
        "\n(all others held at posterior median)",
        fontsize=10, y=1.03,
    )

    for j, (ax, name, bnd) in enumerate(zip(axes, names, bounds)):
        lo, hi = bnd
        grid = np.linspace(lo, hi, n_grid)

        theta_grid = np.tile(medians, (n_grid, 1)).astype(np.float32)
        theta_grid[:, j] = grid.astype(np.float32)

        theta_t = torch.as_tensor(theta_grid)
        x_rep   = x_obs_t.expand(n_grid, -1)

        with torch.no_grad():
            log_post = posterior.log_prob(theta_t, x=x_rep).cpu().numpy()
            _prior = prior or getattr(posterior, "_prior", None) or getattr(posterior, "prior", None)
            log_prior = _prior.log_prob(theta_t).cpu().numpy()
            log_ratio = log_post - log_prior

        log_ratio -= log_ratio.max()

        ax.plot(grid, log_ratio, color="#c44e52", lw=2.0)
        ax.fill_between(grid, log_ratio, 0,
                        where=log_ratio > np.log(0.05),
                        color="#c44e52", alpha=0.15)
        ax.axhline(np.log(0.05), color="gray", lw=0.8, ls=":",
                   label="p=0.05 level")
        ax.axhline(0, color="gray", lw=0.5)

        if truths is not None:
            ax.axvline(truths[j], color="crimson", lw=1.6, ls="--",
                       label=f"True {name}={truths[j]:.2f}")

        ax.set_xlabel(name, fontsize=10)
        if j == 0:
            ax.set_ylabel(r"$\log\,r\,(\theta,\,x_{\rm obs})$  [normalised]", fontsize=9)
        ax.set_xlim(lo, hi)
        ax.set_title(name, fontsize=9)
        if j == 0:
            ax.legend(fontsize=7)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
    return fig


def plot_corner(
    samples: np.ndarray,
    names: list,
    bounds: list,
    truths=None,
    out_path=None,
):
    d = samples.shape[1]
    fig, axes = plt.subplots(d, d, figsize=(2.8 * d, 2.8 * d))
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue
            lo_x, hi_x = bounds[j]
            lo_y, hi_y = bounds[i]
            if i == j:
                xs = np.linspace(lo_x, hi_x, 300)
                try:
                    kde = gaussian_kde(samples[:, i], bw_method="scott")
                    ax.plot(xs, kde(xs), color="#2155a0", lw=1.8)
                    ax.fill_between(xs, kde(xs), alpha=0.25, color="#4878CF")
                except Exception:
                    ax.hist(samples[:, i], bins=30, density=True,
                            color="#4878CF", alpha=0.6)
                ax.axvline(lo_x, color="gray", lw=0.7, ls="--")
                ax.axvline(hi_x, color="gray", lw=0.7, ls="--")
                if truths is not None:
                    ax.axvline(truths[i], color="crimson", lw=1.5)
                ax.set_xlim(lo_x, hi_x)
                ax.set_yticklabels([])
            else:
                try:
                    k = gaussian_kde(
                        np.vstack([samples[:, j], samples[:, i]]),
                        bw_method="scott",
                    )
                    xg = np.linspace(lo_x, hi_x, 60)
                    yg = np.linspace(lo_y, hi_y, 60)
                    XG, YG = np.meshgrid(xg, yg)
                    ZG = k(np.vstack([XG.ravel(), YG.ravel()])).reshape(XG.shape)
                    levels = [np.percentile(ZG, p) for p in [95, 68, 50]]
                    ax.contourf(XG, YG, ZG, levels=sorted(set(levels)),
                                colors=["#c6d8f0", "#7eadd4", "#2155a0"],
                                alpha=0.7)
                    ax.contour(XG, YG, ZG, levels=sorted(set(levels)),
                               colors=["#2155a0"], linewidths=0.6, alpha=0.5)
                except Exception:
                    ax.scatter(samples[:, j], samples[:, i],
                               s=0.5, alpha=0.1, color="#4878CF")
                if truths is not None:
                    ax.axvline(truths[j], color="crimson", lw=1.0)
                    ax.axhline(truths[i], color="crimson", lw=1.0)
                ax.set_xlim(lo_x, hi_x)
                ax.set_ylim(lo_y, hi_y)

            if i == d - 1:
                ax.set_xlabel(names[j], fontsize=8)
            if j == 0 and i != 0:
                ax.set_ylabel(names[i], fontsize=8)
            ax.tick_params(labelsize=6)

    fig.tight_layout(h_pad=0.3, w_pad=0.3)
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posterior", type=str,
                    default="out/nre/nre_population_posterior.pkl")
    ap.add_argument("--out_dir",   type=str, default="out/inference")
    ap.add_argument("--n_samples",    type=int, default=4000)
    ap.add_argument("--n_candidates", type=int, default=5000,
                    help="importance resampling pool size — use 50k+ for real runs")
    ap.add_argument("--batch_size",   type=int, default=500,
                    help="log_prob batch size — reduce if OOM-killed")
    ap.add_argument("--x_obs",        type=str, default="",
                    help="path to .npy x_obs vector; empty = synthetic injection")
    ap.add_argument("--seed",         type=int, default=99)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.posterior, "rb") as f:
        posterior = pickle.load(f)

    param_space = PopParamSpace()
    names  = param_space.names
    bounds = _prior_bounds(param_space)
    truth  = None

    if args.x_obs:
        x_obs = np.load(args.x_obs).astype(np.float32).ravel()
        print(f"Loaded x_obs from {args.x_obs}  shape={x_obs.shape}")
    else:
        # synthetic injection at known theta* — lets us verify recovery
        cosmo = CosmologyGrid()
        obs   = ObservationParams(T=365.25 * 24 * 3600, A_eff=ic86_aeff)
        sim   = PopulationSimulator(cosmo, obs, param_space, SummarySpec.default())

        truth  = np.array([-6.0, 2.5, 44.0, 2.0, -1.0, 2.0], dtype=np.float32)
        x_obs  = sim(truth[None])[0]
        print("Synthetic injection:")
        for n, v in zip(names, truth):
            print(f"  {n:12s} = {v:.2f}")
        print(f"x_obs (expected counts per bin): {x_obs.round(3)}")

    x_obs_t = torch.as_tensor(x_obs[None], dtype=torch.float32)

    fresh_prior = param_space.make_torch_prior()
    fix_posterior_transforms(posterior, prior=fresh_prior)
    print(f"\nSampling {args.n_samples} posterior samples...")
    try:
        post_samples = draw_posterior_samples(
            posterior, x_obs_t.squeeze(0), args.n_samples, seed=args.seed
        ).numpy()
    except Exception as e:
        print(f"Native sampler failed ({e}), falling back to importance resampling...")
        post_samples = sample_nre_posterior(
            posterior, fresh_prior, x_obs_t.squeeze(0),
            n_samples=args.n_samples,
            n_candidates=args.n_candidates,
            batch_size=args.batch_size,
            seed=args.seed,
        ).numpy()
    np.save(out / "posterior_samples.npy", post_samples)

    prior_samples = param_space.make_torch_prior().sample(
        (args.n_samples,)
    ).numpy()

    print("\n" + "="*65)
    print(f"{'Parameter':<14} {'Prior range':>14} {'Post. 68% CI':>20} {'Factor':>8}")
    print("-"*65)
    for j, (name, bnd) in enumerate(zip(names, bounds)):
        lo, hi = bnd
        p16, p50, p84 = np.percentile(post_samples[:, j], [16, 50, 84])
        prior_w = hi - lo
        post_w  = p84 - p16
        factor  = prior_w / post_w if post_w > 0 else float("inf")
        truth_s = f"  [true={truth[j]:.2f}]" if truth is not None else ""
        print(
            f"{name:<14} [{lo:.1f}, {hi:.1f}]{'':<2}"
            f"{p50:.3f} [{p16:.3f}, {p84:.3f}]"
            f"  ×{factor:5.1f}{truth_s}"
        )
    print("="*65)
    print()

    print("Plotting constraining figure...")
    plot_constraining(
        prior_samples, post_samples, names, bounds,
        truths=truth,
        out_path=out / "constraining.png",
    )
    print(f"  → {out}/constraining.png")

    print("Plotting likelihood ratio profiles...")
    plot_likelihood_ratio_profile(
        posterior, x_obs_t, post_samples, names, bounds,
        prior=fresh_prior,
        truths=truth,
        out_path=out / "likelihood_ratio.png",
    )
    print(f"  → {out}/likelihood_ratio.png")

    print("Plotting corner...")
    plot_corner(post_samples, names, bounds, truths=truth,
                out_path=out / "corner.png")
    print(f"  → {out}/corner.png")

    print("\nDone. All outputs in", out)


if __name__ == "__main__":
    main()
