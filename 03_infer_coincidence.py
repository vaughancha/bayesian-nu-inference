#!/usr/bin/env python3
"""
Posterior inference on blazar population from neutrino coincidence SNR distribution.

Given observed coincidence TS values, infers the posterior over BL Lac parameters:
    p(theta | x_obs)  ∝  p(theta) × r(theta, x_obs)

Outputs:
  out/coincidence_inference/posterior_samples.npy   (N, 7)
  out/coincidence_inference/constraining.png
  out/coincidence_inference/corner.png
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from coincidence.simulator import CoincidenceSimulator, CoincidenceParamSpace
from coincidence.summary import CoincidenceSummary
from inference.sampler import fix_posterior_transforms, draw_posterior_samples, sample_nre_posterior


def _prior_bounds(param_space: CoincidenceParamSpace):
    attrs = [
        "log10_Lbreak_bounds", "alpha_bounds", "beta_bounds",
        "delta_bounds", "log10_Lambda_bounds",
        "variability_weight_bounds", "spectral_index_mu_bounds",
    ]
    return [getattr(param_space, a) for a in attrs]


def plot_constraining(prior_samples, post_samples, names, bounds, truths=None, out_path=None):
    d = len(names)
    fig, axes = plt.subplots(1, d, figsize=(3.2*d, 3.8))
    fig.suptitle(
        r"Bayesian constraining: $p(\theta|x_{\rm obs}) \propto p(\theta) \times r(\theta, x_{\rm obs})$",
        fontsize=10, y=1.02,
    )
    for j, (ax, name, bnd) in enumerate(zip(axes, names, bounds)):
        lo, hi = bnd
        s = post_samples[:, j]
        p16, p50, p84 = np.percentile(s, [16, 50, 84])
        post_68w = p84 - p16
        factor = (hi - lo) / post_68w if post_68w > 0 else float("inf")
        xs = np.linspace(lo, hi, 300)
        try:
            kde = gaussian_kde(s, bw_method="scott")
            ys  = kde(xs)
        except Exception:
            ys = np.zeros_like(xs)
        ax.fill_between([lo, hi], 0, max(ys)*1.15, color="lightgray", alpha=0.4, label="Prior")
        ax.axvline(lo, color="gray", lw=0.8, ls="--")
        ax.axvline(hi, color="gray", lw=0.8, ls="--")
        mask68 = (xs >= p16) & (xs <= p84)
        ax.fill_between(xs[mask68], 0, ys[mask68], color="#4878CF", alpha=0.45)
        ax.plot(xs, ys, color="#2155a0", lw=2.0, label="Posterior")
        if truths is not None:
            ax.axvline(truths[j], color="crimson", lw=1.8, label=f"Truth={truths[j]:.2f}")
        ax.set_xlabel(name, fontsize=9)
        ax.set_xlim(lo, hi)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_visible(False)
        ax.set_title(f"×{factor:.1f}", fontsize=8.5, color="#333333")
        if j == 0:
            ax.legend(fontsize=7)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_corner(samples, names, bounds, truths=None, out_path=None):
    d = samples.shape[1]
    fig, axes = plt.subplots(d, d, figsize=(2.5*d, 2.5*d))
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue
            lo_x, hi_x = bounds[j]
            lo_y, hi_y = bounds[i]
            if i == j:
                xs = np.linspace(lo_x, hi_x, 200)
                try:
                    kde = gaussian_kde(samples[:, i], bw_method="scott")
                    ax.plot(xs, kde(xs), color="#2155a0", lw=1.5)
                    ax.fill_between(xs, kde(xs), alpha=0.2, color="#4878CF")
                except Exception:
                    ax.hist(samples[:, i], bins=25, density=True, color="#4878CF", alpha=0.5)
                ax.axvline(lo_x, color="gray", lw=0.6, ls="--")
                ax.axvline(hi_x, color="gray", lw=0.6, ls="--")
                if truths is not None:
                    ax.axvline(truths[i], color="crimson", lw=1.3)
            else:
                ax.scatter(samples[:, j], samples[:, i], s=0.5, alpha=0.1, color="#4878CF")
                if truths is not None:
                    ax.axvline(truths[j], color="crimson", lw=0.8)
                    ax.axhline(truths[i], color="crimson", lw=0.8)
                ax.set_xlim(lo_x, hi_x)
                ax.set_ylim(lo_y, hi_y)
            if i == d-1:
                ax.set_xlabel(names[j], fontsize=7)
            if j == 0 and i != 0:
                ax.set_ylabel(names[i], fontsize=7)
            ax.tick_params(labelsize=5)
    fig.tight_layout(h_pad=0.2, w_pad=0.2)
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posterior",    type=str, default="out/nre_coincidence/nre_population_posterior.pkl")
    ap.add_argument("--out_dir",      type=str, default="out/coincidence_inference")
    ap.add_argument("--n_samples",    type=int, default=2000)
    ap.add_argument("--x_obs",        type=str, default="",
                    help="path to .npy x_obs (15-dim); empty = synthetic injection")
    ap.add_argument("--seed",         type=int, default=99)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    param_space = CoincidenceParamSpace()
    names  = param_space.names
    bounds = _prior_bounds(param_space)
    fresh_prior = param_space.make_torch_prior()
    truth = None

    with open(args.posterior, "rb") as f:
        posterior = pickle.load(f)

    fix_posterior_transforms(posterior, prior=fresh_prior)

    if args.x_obs:
        x_obs = np.load(args.x_obs).astype(np.float32).ravel()
        print(f"Loaded x_obs from {args.x_obs}")
    else:
        # synthetic injection at reference BL Lac parameters
        truth = np.array([47.0, -1.5, -2.5, -4.2, 3.71, 0.075, 2.1], dtype=np.float32)
        sim   = CoincidenceSimulator()
        x_obs = sim(truth[None], seed=args.seed)[0]
        print("Synthetic injection at reference BL Lac parameters:")
        for n, v in zip(names, truth):
            print(f"  {n:25s} = {v:.3f}")
        print(f"\nx_obs (TS summary): {x_obs.round(3)}")

    x_obs_t = torch.as_tensor(x_obs[None], dtype=torch.float32)

    print(f"\nSampling {args.n_samples} posterior samples...")
    try:
        post_samples = draw_posterior_samples(
            posterior, x_obs_t.squeeze(0), args.n_samples, seed=args.seed
        ).numpy()
    except Exception as e:
        print(f"Native sampler failed ({e}), falling back to importance resampling...")
        post_samples = sample_nre_posterior(
            posterior, fresh_prior, x_obs_t.squeeze(0),
            n_samples=args.n_samples, n_candidates=50000, batch_size=500, seed=args.seed,
        ).numpy()

    np.save(out / "posterior_samples.npy", post_samples)

    print("\n" + "="*70)
    print(f"{'Parameter':<25} {'Prior range':>15}  {'Post. 68% CI':>22}  {'Factor':>7}")
    print("-"*70)
    for j, (name, bnd) in enumerate(zip(names, bounds)):
        lo, hi = bnd
        p16, p50, p84 = np.percentile(post_samples[:, j], [16, 50, 84])
        pw = hi - lo
        factor = pw / (p84 - p16) if (p84 - p16) > 0 else float("inf")
        truth_s = f"  [true={truth[j]:.2f}]" if truth is not None else ""
        print(f"{name:<25} [{lo:.1f},{hi:.1f}]{'':<3} {p50:.3f} [{p16:.3f},{p84:.3f}]  "
              f"×{factor:5.1f}{truth_s}")
    print("="*70)

    prior_samples = fresh_prior.sample((args.n_samples,)).numpy()

    plot_constraining(prior_samples, post_samples, names, bounds,
                      truths=truth, out_path=out / "constraining.png")
    print(f"\nConstraining → {out}/constraining.png")

    plot_corner(post_samples, names, bounds, truths=truth,
                out_path=out / "corner.png")
    print(f"Corner       → {out}/corner.png")
    print(f"\nDone. Outputs in {out}")


if __name__ == "__main__":
    main()
