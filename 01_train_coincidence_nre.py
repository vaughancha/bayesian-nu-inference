#!/usr/bin/env python3
"""
Train NRE on blazar-neutrino coincidence simulations.

Uses the same SBIRunner as the population inference pipeline, with:
  theta = blazar population parameters (7 dims)
  x     = coincidence TS summary statistics (15 dims)

Note: the current simulator uses the diffuse background config (nu_diffuse_tracks.yml),
so neutrinos are NOT connected to individual sources. For real signal discrimination,
switch to nu_coincidence/config/nu_connected_*.yml.

Output:
  out/nre_coincidence/nre_population_posterior.pkl
  out/nre_coincidence/nre_population_summary.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from inference.loader import H5Loader
from inference.runner import SBIRunner
from coincidence.simulator import CoincidenceParamSpace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims_h5",      type=str,   default="out/coincidence_sims.h5")
    ap.add_argument("--out_dir",      type=str,   default="out/nre_coincidence")
    ap.add_argument("--device",       type=str,   default="cpu")
    ap.add_argument("--architecture", type=str,   default="mlp",
                    choices=["mlp", "resnet"])
    ap.add_argument("--n_ensemble",   type=int,   default=3)
    ap.add_argument("--epochs",       type=int,   default=30)
    ap.add_argument("--batch_size",   type=int,   default=128)
    ap.add_argument("--lr",           type=float, default=5e-4)
    ap.add_argument("--seed",         type=int,   default=0)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    loader = H5Loader(args.sims_h5)
    d_theta = loader.get_all_parameters().shape[1]
    d_x     = loader.get_all_data().shape[1]
    print(f"Loaded {len(loader)} coincidence sims | d_theta={d_theta} d_x={d_x}")

    prior = CoincidenceParamSpace().make_torch_prior(device=args.device)

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
