#!/usr/bin/env python3
"""
Train NRE (Neural Ratio Estimation).

NRE trains a binary classifier to distinguish joint pairs (θ, x) ~ p(θ,x)
from marginal pairs (θ, x) ~ p(θ)p(x). The classifier learns log r(θ, x).

Output:
  out/nre/nre_population_posterior.pkl
  out/nre/nre_population_summary.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from inference.loader import H5Loader
from inference.runner import SBIRunner
from population.simulator import PopParamSpace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims_h5",      type=str,   default="out/sims.h5")
    ap.add_argument("--out_dir",      type=str,   default="out/nre")
    ap.add_argument("--device",       type=str,   default="cpu")
    ap.add_argument("--architecture", type=str,   default="mlp",
                    choices=["mlp", "resnet"])
    ap.add_argument("--n_ensemble",   type=int,   default=3)
    ap.add_argument("--epochs",       type=int,   default=50)
    ap.add_argument("--batch_size",   type=int,   default=256)
    ap.add_argument("--lr",           type=float, default=5e-4)
    ap.add_argument("--seed",         type=int,   default=0)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    loader = H5Loader(args.sims_h5)
    d_theta = loader.get_all_parameters().shape[1]
    d_x     = loader.get_all_data().shape[1]
    print(f"Loaded {len(loader)} sims | d_theta={d_theta} d_x={d_x}")

    prior = PopParamSpace().make_torch_prior(device=args.device)

    runner = SBIRunner(
        prior=prior,
        engine="NRE",
        architecture=args.architecture,
        n_ensemble=args.n_ensemble,
        train_args=dict(
            training_batch_size=args.batch_size,
            learning_rate=args.lr,
            stop_after_epochs=args.epochs,
            validation_fraction=0.1,
            clip_max_norm=5.0,
        ),
        out_dir=args.out_dir,
        device=args.device,
        name="nre_population_",
    )

    posterior, summaries = runner(loader, seed=args.seed)
    best = min(min(s["best_validation_loss"]) for s in summaries)
    print(f"Training complete. Best val BCE: {best:.4f}")
    print(f"Saved → {args.out_dir}/nre_population_posterior.pkl")


if __name__ == "__main__":
    main()
