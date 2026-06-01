#!/usr/bin/env python3
"""
Generate HEALPix TS maps as training data for the skymap NRE mode.

For each simulation i:
  1. draw theta_i from the prior
  2. simulate IceCube events at that theta_i
  3. run the HEALPix TS scan
  4. store the map (N_pix, 3): channels (ts, ns_hat, gamma_hat)

Computationally expensive vs. 00_generate_sims.py.
~5–30 min per simulation depending on nside and event count.

Smoke test: --n_sims 20 --nside 16 --n_events 500

Output: out/ts_maps.h5
    /theta   (N, 6)         float32
    /maps    (N, N_pix, 3)  float32
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import healpy as hp
import numpy as np
import torch

from population.simulator import PopParamSpace
from snr.ts_scan import TSScan

BAYH = Path.home() / "BayH" / "nu_pop" / "1_detection_probability"
DEFAULT_ANGRES_E2    = BAYH / "input" / "angres_plot_E-2.csv"
DEFAULT_ANGRES_ATMOS = BAYH / "input" / "angres_plot_atmos.csv"
DEFAULT_BG_H5        = BAYH / "output" / "bg_5e5.h5"
DEFAULT_PL_H5        = BAYH / "output" / "pl_1e6.h5"


def _simulate_events_for_theta(theta: np.ndarray, n_events: int, seed: int):
    """
    Simulate IceCube events from population parameters theta.

    Placeholder — replace with a proper forward simulation that draws events
    from the population described by theta. Currently returns isotropic
    background events (correct for calibration; signal injection requires
    connecting to the IceCube event simulator).
    """
    rng = np.random.default_rng(seed)
    ra  = rng.uniform(0, 2 * np.pi, n_events)
    dec = np.arcsin(rng.uniform(-1, 1, n_events))
    gamma = float(theta[1])
    u = rng.uniform(0, 1, n_events)
    E_min, E_max = 1e2, 1e7
    if abs(gamma - 1.0) > 1e-6:
        reco_energy = (
            E_min ** (1 - gamma) + u * (E_max ** (1 - gamma) - E_min ** (1 - gamma))
        ) ** (1 / (1 - gamma))
    else:
        reco_energy = E_min * (E_max / E_min) ** u
    return ra.astype(np.float32), dec.astype(np.float32), reco_energy.astype(np.float32)


def _one_map(args):
    (i, theta, n_events, nside, seed,
     angres_e2, angres_atmos, bg_h5, pl_h5) = args

    ra, dec, energy = _simulate_events_for_theta(theta, n_events, seed)

    scanner = TSScan(
        events_ra=ra, events_dec=dec, events_reco_energy=energy,
        angres_E2_csv=angres_e2, angres_atmos_csv=angres_atmos,
        bg_h5=bg_h5, signal_pl_h5=pl_h5,
    )
    result = scanner.run(nside=nside, verbose=False)

    n_pix = hp.nside2npix(nside)
    m_ts    = np.full(n_pix, 0.0, dtype=np.float32)
    m_ns    = np.full(n_pix, 0.0, dtype=np.float32)
    m_gamma = np.full(n_pix, 2.19, dtype=np.float32)
    m_ts[result.pix]    = result.ts
    m_ns[result.pix]    = result.ns_hat
    m_gamma[result.pix] = result.gamma_hat

    sky = np.stack([m_ts, m_ns, m_gamma], axis=-1)
    return i, theta, sky


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_sims",       type=int,   default=500)
    ap.add_argument("--nside",        type=int,   default=32)
    ap.add_argument("--n_events",     type=int,   default=2000)
    ap.add_argument("--out",          type=str,   default="out/ts_maps.h5")
    ap.add_argument("--n_workers",    type=int,   default=4)
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--angres_e2",    type=str,   default=str(DEFAULT_ANGRES_E2))
    ap.add_argument("--angres_atmos", type=str,   default=str(DEFAULT_ANGRES_ATMOS))
    ap.add_argument("--bg_h5",        type=str,   default=str(DEFAULT_BG_H5))
    ap.add_argument("--pl_h5",        type=str,   default=str(DEFAULT_PL_H5))
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_pix = hp.nside2npix(args.nside)

    param_space = PopParamSpace()
    prior = param_space.make_torch_prior()
    torch.manual_seed(args.seed)
    thetas = prior.sample((args.n_sims,)).numpy().astype(np.float32)

    print(f"Generating {args.n_sims} TS maps | nside={args.nside} "
          f"({n_pix} pix) | {args.n_events} events/sim | "
          f"{args.n_workers} workers")

    job_args = [
        (i, thetas[i], args.n_events, args.nside,
         args.seed + i,
         args.angres_e2, args.angres_atmos,
         args.bg_h5, args.pl_h5)
        for i in range(args.n_sims)
    ]

    theta_out = np.zeros((args.n_sims, 6), dtype=np.float32)
    maps_out  = np.zeros((args.n_sims, n_pix, 3), dtype=np.float32)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.n_workers) as exe:
        futs = {exe.submit(_one_map, a): a[0] for a in job_args}
        done = 0
        for fut in as_completed(futs):
            i, theta, sky = fut.result()
            theta_out[i] = theta
            maps_out[i]  = sky
            done += 1
            if done % max(1, args.n_sims // 10) == 0:
                print(f"  {done}/{args.n_sims} maps done "
                      f"({time.time()-t0:.0f}s elapsed)")

    with h5py.File(args.out, "w") as f:
        f.create_dataset("theta", data=theta_out)
        f.create_dataset("maps",  data=maps_out)
        f.attrs["nside"]       = args.nside
        f.attrs["n_events"]    = args.n_events
        f.attrs["param_names"] = param_space.names

    print(f"\nSaved {args.n_sims} maps → {args.out}")
    print(f"  theta: {theta_out.shape}   maps: {maps_out.shape}")


if __name__ == "__main__":
    main()
