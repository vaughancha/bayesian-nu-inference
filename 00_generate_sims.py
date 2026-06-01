#!/usr/bin/env python3
"""
Generate simulation training set.

Draws N parameter vectors theta from the prior, runs the population
forward simulator for each, and saves (theta, x) to HDF5.

Timing (measured on this machine):
  n_z=50,  single core: ~2.1s/sim  → 50k sims ≈ 30h single / 1.5h on 22 cores
  n_z=100, single core: ~5.4s/sim  → 50k sims ≈ 74h single / 3.5h on 22 cores

Output: out/sims.h5
    /theta   (N, 6)  float32
    /x       (N, 60) float32  expected counts per (6 E-bins × 10 dec-bins)
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np

from population.physics import CosmologyGrid, ObservationParams
from population.simulator import PopulationSimulator, PopParamSpace, SummarySpec


def ic86_aeff(E: np.ndarray, delta: float) -> np.ndarray:
    """
    Placeholder IC86 effective area — replace with tabulated interpolation.
    Units: cm²
    """
    E = np.asarray(E, float)
    return np.clip(1e4 * (E / 1e5) ** 0.5 * max(0.0, np.cos(float(delta))), 0.0, 1e6)


def _worker(theta_chunk: np.ndarray, n_z: int, T_s: float, seed: int) -> np.ndarray:
    cosmo = CosmologyGrid()
    obs   = ObservationParams(T=T_s, A_eff=ic86_aeff)
    sim   = PopulationSimulator(cosmo, obs, PopParamSpace(), SummarySpec.default(), n_z=n_z)
    np.random.seed(seed)
    return sim(theta_chunk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_sims",    type=int,   default=1000)
    ap.add_argument("--out",       type=str,   default="out/sims.h5")
    ap.add_argument("--seed",      type=int,   default=42)
    ap.add_argument("--T_years",   type=float, default=1.0)
    ap.add_argument("--n_z",       type=int,   default=50,
                    help="redshift grid points. 50 = fast/accurate, 100 = more accurate")
    ap.add_argument("--n_workers", type=int,   default=1,
                    help="parallel worker processes")
    ap.add_argument("--chunk_size",type=int,   default=50,
                    help="sims per worker chunk — tune so each chunk takes ~60s")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    T_s = args.T_years * 365.25 * 24 * 3600

    import torch
    torch.manual_seed(args.seed)
    prior  = PopParamSpace().make_torch_prior()
    thetas = prior.sample((args.n_sims,)).numpy().astype(np.float32)

    d_theta = thetas.shape[1]
    d_x     = SummarySpec.default().d_x

    print(f"Generating {args.n_sims} sims | n_z={args.n_z} | "
          f"{args.n_workers} worker(s) | chunk={args.chunk_size}")
    print(f"  d_theta={d_theta}  d_x={d_x}")

    chunks = [
        (i, thetas[i : i + args.chunk_size])
        for i in range(0, args.n_sims, args.chunk_size)
    ]

    x_out = np.zeros((args.n_sims, d_x), dtype=np.float32)
    t0    = time.time()
    done  = 0

    with ProcessPoolExecutor(max_workers=args.n_workers) as exe:
        futs = {
            exe.submit(_worker, chunk, args.n_z, T_s, args.seed + idx): (idx, start)
            for idx, (start, chunk) in enumerate(chunks)
        }
        for fut in as_completed(futs):
            _, start = futs[fut]
            x_chunk  = fut.result()
            end      = min(start + args.chunk_size, args.n_sims)
            x_out[start:end] = x_chunk[:end - start]
            done    += end - start
            elapsed  = time.time() - t0
            rate     = done / elapsed
            eta      = (args.n_sims - done) / rate if rate > 0 else 0
            print(f"  {done}/{args.n_sims} done | "
                  f"{rate:.1f} sim/s | ETA {eta/60:.1f} min", flush=True)

    summary = SummarySpec.default()
    param_space = PopParamSpace()
    with h5py.File(args.out, "w") as f:
        f.create_dataset("theta", data=thetas)
        f.create_dataset("x",     data=x_out)
        f.attrs["n_sims"]       = args.n_sims
        f.attrs["seed"]         = args.seed
        f.attrs["param_names"]  = param_space.names
        f.attrs["d_theta"]      = d_theta
        f.attrs["d_x"]          = d_x
        f.create_dataset("E_edges",   data=summary.E_edges)
        f.create_dataset("dec_edges", data=summary.dec_edges)

    total = time.time() - t0
    print(f"\nSaved {args.n_sims} simulations → {args.out}  ({total/60:.1f} min total)")


if __name__ == "__main__":
    main()
