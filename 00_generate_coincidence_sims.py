#!/usr/bin/env python3
"""
Generate blazar-neutrino coincidence training set.

For each theta draw from the prior:
  1. simulate a BL Lac population (popsynth, nu_coincidence)
  2. simulate IceCube track events
  3. compute per-source coincidence TS
  4. compress TS distribution → summary statistic x (15 dims)

theta (7 dims): log10_Lbreak, alpha, beta, delta, log10_Lambda,
                variability_weight, spectral_index_mu

Timing: ~43s per sim single-core.
  50k sims / 64 workers ≈ 9 hours  (Polaris node)
  50k sims / 16 workers ≈ 37 hours (XPS local)

Output: out/coincidence_sims.h5
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import torch

from coincidence.simulator import (
    CoincidenceSimulator, CoincidenceParamSpace, _simulate_one,
)
from coincidence.summary import CoincidenceSummary


def _worker(theta_chunk: np.ndarray, start_idx: int,
            obs_time: float, seed: int) -> tuple:
    """simulate a chunk of theta values, returns (start_idx, x_chunk)."""
    from pathlib import Path
    sim = CoincidenceSimulator(obs_time_yr=obs_time)
    x_chunk = np.array([
        sim._simulate_one(theta_chunk[i], seed + i)
        for i in range(len(theta_chunk))
    ], dtype=np.float32)
    return start_idx, x_chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_sims",     type=int,   default=500)
    ap.add_argument("--out",        type=str,   default="out/coincidence_sims.h5")
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--chunk_size", type=int,   default=5,
                    help="sims per worker chunk — each sim ~43s")
    ap.add_argument("--n_workers",  type=int,   default=1)
    ap.add_argument("--obs_time",   type=float, default=10.0)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    param_space = CoincidenceParamSpace()
    prior  = param_space.make_torch_prior()
    thetas = prior.sample((args.n_sims,)).numpy().astype(np.float32)

    d_theta = thetas.shape[1]
    d_x     = CoincidenceSummary().d_x

    print(f"Coincidence sims: {args.n_sims} | "
          f"d_theta={d_theta} d_x={d_x} | "
          f"{args.n_workers} worker(s) | chunk={args.chunk_size}")
    print(f"Estimated time: "
          f"{args.n_sims * 43 / args.n_workers / 3600:.1f} h "
          f"({args.n_sims * 43 / args.n_workers / 60:.0f} min)")

    chunks = [
        (i, thetas[i : i + args.chunk_size])
        for i in range(0, args.n_sims, args.chunk_size)
    ]

    x_out = np.zeros((args.n_sims, d_x), dtype=np.float32)
    t0 = time.time()
    done = 0

    with ProcessPoolExecutor(max_workers=args.n_workers) as exe:
        futs = {
            exe.submit(_worker, chunk, start, args.obs_time, args.seed + start): start
            for start, chunk in chunks
        }
        for fut in as_completed(futs):
            start_idx, x_chunk = fut.result()
            end = min(start_idx + args.chunk_size, args.n_sims)
            x_out[start_idx:end] = x_chunk[:end - start_idx]
            done += end - start_idx
            elapsed = time.time() - t0
            rate = done / elapsed
            eta  = (args.n_sims - done) / rate if rate > 0 else 0
            print(f"  {done}/{args.n_sims} | "
                  f"{rate:.2f} sim/s | ETA {eta/3600:.1f}h", flush=True)

    with h5py.File(args.out, "w") as f:
        f.create_dataset("theta", data=thetas)
        f.create_dataset("x",     data=x_out)
        f.attrs["n_sims"]       = args.n_sims
        f.attrs["seed"]         = args.seed
        f.attrs["param_names"]  = param_space.names
        f.attrs["d_theta"]      = d_theta
        f.attrs["d_x"]          = d_x

    total = time.time() - t0
    print(f"\nSaved {args.n_sims} sims → {args.out}  ({total/3600:.2f}h total)")


if __name__ == "__main__":
    main()
