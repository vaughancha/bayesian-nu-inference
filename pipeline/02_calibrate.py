#!/usr/bin/env python3
"""
Calibrate the trained posterior.

Runs three tests in sequence:
  a) SBC (rank histograms) — checks posterior is not over/under-confident
  b) coverage test — expected coverage vs. nominal credible level
  c) posterior predictive check on a held-out x

Output:
  out/calibration/sbc_ranks.npy
  out/calibration/sbc_ranks.png
  out/calibration/coverage.png
  out/calibration/ppc.png
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from inference.calibration import SBCRunner, CoverageTest, PosteriorPredictiveCheck
from inference.loader import H5Loader
from population.physics import CosmologyGrid, ObservationParams
from population.simulator import PopulationSimulator, PopParamSpace, SummarySpec


def ic86_aeff(E: np.ndarray, delta: float) -> np.ndarray:
    E = np.asarray(E, float)
    return np.clip(1e4 * (E / 1e5) ** 0.5 * max(0.0, np.cos(float(delta))), 0.0, 1e6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posterior", type=str, default="out/npe/npe_population_posterior.pkl")
    ap.add_argument("--sims_h5", type=str, default="out/sims.h5")
    ap.add_argument("--out_dir", type=str, default="out/calibration")
    ap.add_argument("--n_sbc", type=int, default=200)
    ap.add_argument("--n_post", type=int, default=500)
    ap.add_argument("--n_coverage", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=200,
                    help="log_prob batch size — reduce if OOM-killed")
    ap.add_argument("--n_z", type=int, default=20,
                    help="redshift points for simulator inside calibration (keep low for speed)")
    ap.add_argument("--skip_ppc", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.posterior, "rb") as f:
        posterior = pickle.load(f)

    cosmo = CosmologyGrid()
    obs = ObservationParams(T=365.25 * 24 * 3600, A_eff=ic86_aeff)
    sim = PopulationSimulator(cosmo, obs, PopParamSpace(), SummarySpec.default(), n_z=args.n_z)
    prior = PopParamSpace().make_torch_prior()
    from inference.sampler import fix_posterior_transforms
    fix_posterior_transforms(posterior, prior=prior)
    names = PopParamSpace().names

    n_candidates = args.n_post * 10

    print("Running SBC...")
    sbc = SBCRunner(posterior, sim, prior,
                    n_sbc=args.n_sbc, n_post=args.n_post,
                    n_candidates=n_candidates, batch_size=args.batch_size,
                    param_names=names)
    ranks = sbc.run(seed=args.seed)
    np.save(out / "sbc_ranks.npy", ranks)
    sbc.plot(ranks, out_path=out / "sbc_ranks.png")
    print(f"  SBC ranks saved → {out}/sbc_ranks.png")

    print("Running coverage test...")
    cov = CoverageTest(posterior, sim, prior,
                       n_test=args.n_coverage, n_post=args.n_post,
                       n_candidates=n_candidates, batch_size=args.batch_size,
                       param_names=names)
    levels, coverage = cov.run(seed=args.seed + 1)
    cov.plot(levels, coverage, out_path=out / "coverage.png")
    print(f"  Coverage plot → {out}/coverage.png")

    if not args.skip_ppc:
        loader = H5Loader(args.sims_h5)
        x_obs = loader.get_all_data()[0]
        ppc = PosteriorPredictiveCheck(posterior, sim, x_obs, prior=prior,
                                       n_draws=20, n_candidates=n_candidates)
        x_rep = ppc.run(seed=args.seed + 2)
        ppc.plot(x_rep, out_path=out / "ppc.png")
        print(f"  PPC plot → {out}/ppc.png")
    else:
        print("  PPC skipped (--skip_ppc)")


if __name__ == "__main__":
    main()
