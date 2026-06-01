"""
Forward simulator for blazar-neutrino coincidence population inference.

Maps theta → blazar population + IceCube simulation → per-source coincidence TS → summary x.

theta = (log10_Lbreak, alpha, beta, delta, log10_Lambda,
         variability_weight, spectral_index_mu)
x = 15-dim summary of the coincidence TS distribution (see summary.py)
"""

from __future__ import annotations

import copy
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import yaml

from .ts import CoincidenceLikelihood, coincidence_ts
from .summary import CoincidenceSummary

_NC_ROOT = Path(os.environ.get(
    "NU_COINCIDENCE_ROOT",
    Path.home() / "Desktop" / "nu_coincidence" / "nu_coincidence"
))
_BLLAC_REF   = _NC_ROOT / "config" / "bllac_ref.yml"
_NU_TRACKS   = _NC_ROOT / "config" / "nu_diffuse_tracks.yml"


@dataclass
class CoincidenceParamSpace:
    """
    7-dimensional parameter space for the BL Lac population.

    theta = [log10_Lbreak, alpha, beta, delta, log10_Lambda,
             variability_weight, spectral_index_mu]
    """
    log10_Lbreak_bounds:      Tuple[float, float] = (46.0, 48.0)
    alpha_bounds:             Tuple[float, float] = (-2.0, -0.5)
    beta_bounds:              Tuple[float, float] = (-3.5, -2.0)
    delta_bounds:             Tuple[float, float] = (-6.0, -2.0)
    log10_Lambda_bounds:      Tuple[float, float] = (3.0,  4.5)
    variability_weight_bounds:Tuple[float, float] = (0.01, 0.2)
    spectral_index_mu_bounds: Tuple[float, float] = (1.8,  2.5)

    @property
    def d_theta(self) -> int:
        return 7

    @property
    def names(self) -> List[str]:
        return ["log10_Lbreak", "alpha", "beta", "delta",
                "log10_Lambda", "variability_weight", "spectral_index_mu"]

    def make_torch_prior(self, device: str = "cpu"):
        import torch
        from torch.distributions import Uniform, Independent
        bounds = [
            self.log10_Lbreak_bounds,
            self.alpha_bounds,
            self.beta_bounds,
            self.delta_bounds,
            self.log10_Lambda_bounds,
            self.variability_weight_bounds,
            self.spectral_index_mu_bounds,
        ]
        lows  = torch.tensor([b[0] for b in bounds], device=device)
        highs = torch.tensor([b[1] for b in bounds], device=device)
        return Independent(Uniform(lows, highs), 1)

    def unpack(self, theta: np.ndarray) -> dict:
        t = np.asarray(theta).ravel()
        return {
            "Lbreak":             10.0 ** t[0],
            "alpha":              float(t[1]),
            "beta":               float(t[2]),
            "delta":              float(t[3]),
            "Lambda":             10.0 ** t[4],
            "variability_weight": float(t[5]),
            "spectral_index_mu":  float(t[6]),
        }


def _to_python(obj):
    """recursively convert numpy scalars to native Python types for safe YAML dump."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    if hasattr(obj, "item"):      # numpy scalar
        return obj.item()
    return obj


def _build_bllac_config(params: dict, base_config: Path) -> str:
    """build a modified BL Lac YAML config and return its path."""
    with open(base_config) as f:
        cfg = yaml.safe_load(f)

    cfg["luminosity distribution"]["BPLDistribution"]["Lbreak"] = float(params["Lbreak"])
    cfg["luminosity distribution"]["BPLDistribution"]["alpha"]  = float(params["alpha"])
    cfg["luminosity distribution"]["BPLDistribution"]["beta"]   = float(params["beta"])

    cfg["spatial distribution"]["ZPowerCosmoDistribution"]["delta"]  = float(params["delta"])
    cfg["spatial distribution"]["ZPowerCosmoDistribution"]["Lambda"] = float(params["Lambda"])

    cfg["auxiliary samplers"]["variability"]["weight"] = float(params["variability_weight"])

    cfg["auxiliary samplers"]["spectral_index"]["mu"] = float(params["spectral_index_mu"])

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False, prefix="bllac_"
    )
    yaml.dump(_to_python(cfg), tmp)
    tmp.close()
    return tmp.name


def _simulate_one(
    params: dict,
    nu_config: Path,
    bllac_base: Path,
    seed: int,
    summary: CoincidenceSummary,
    obs_time_yr: float = 10.0,
) -> np.ndarray:
    """run one blazar population + IceCube simulation, return summary x."""
    from nu_coincidence.populations.popsynth_wrapper import PopsynthParams, PopsynthWrapper
    from nu_coincidence.neutrinos.icecube import IceCubeObsParams, IceCubeTracksWrapper

    bllac_cfg = _build_bllac_config(params, bllac_base)

    try:
        pop_ps      = PopsynthParams(bllac_cfg)
        pop_ps.seed = seed
        bllac       = PopsynthWrapper(pop_ps)
        survey      = bllac.survey

        src_ras  = np.asarray(survey.ra,  dtype=float)    # degrees
        src_decs = np.asarray(survey.dec, dtype=float)    # degrees
        n_src    = len(src_ras)

        src_gammas = np.asarray(survey.spectral_index, dtype=float) \
            if hasattr(survey, "spectral_index") \
            else np.full(n_src, params["spectral_index_mu"])

        # use first flare per source; default to 0.1 yr / t=0 if no flares
        if hasattr(survey, "flare_durations"):
            raw_dur = survey.flare_durations
            src_flare_durs = np.array(
                [float(d[0]) if hasattr(d, "__len__") and len(d) > 0 else 0.1
                 for d in raw_dur], dtype=float
            )
        else:
            src_flare_durs = np.full(n_src, 0.1)

        if hasattr(survey, "flare_times"):
            raw_t0 = survey.flare_times
            src_flare_t0s = np.array(
                [float(t[0]) if hasattr(t, "__len__") and len(t) > 0 else 0.0
                 for t in raw_t0], dtype=float
            )
        else:
            src_flare_t0s = np.zeros(n_src)

        nu_ps      = IceCubeObsParams.from_file(str(nu_config))
        nu_ps.seed = seed
        nu_wrap    = IceCubeTracksWrapper(nu_ps)
        obs        = nu_wrap.observation

        if obs is None or n_src == 0 or len(obs.ra) == 0:
            return summary.compute(np.array([0.0]))

        ev_ras    = np.asarray(obs.ra,       dtype=float)   # degrees
        ev_decs   = np.asarray(obs.dec,      dtype=float)   # degrees
        ev_sigmas = np.asarray(obs.ang_err,  dtype=float)   # degrees
        ev_Es     = np.asarray(obs.energies, dtype=float)   # GeV (reco)
        sel       = np.asarray(obs.selection, dtype=bool)
        ev_ts_yr  = np.asarray(obs.times,    dtype=float)[sel]   # years

        per_source_ts = []
        for j in range(n_src):
            llh = CoincidenceLikelihood(
                src_ra_deg        = float(src_ras[j]),
                src_dec_deg       = float(src_decs[j]),
                src_gamma         = float(src_gammas[j]),
                flare_duration_yr = float(src_flare_durs[j]),
                flare_t0_yr       = float(src_flare_t0s[j]),
                obs_time_yr       = obs_time_yr,
            )
            ts_vals = coincidence_ts(
                llh, ev_ras, ev_decs, ev_sigmas, ev_Es, ev_ts_yr
            )
            per_source_ts.append(float(ts_vals.max()) if len(ts_vals) > 0 else 0.0)

        return summary.compute(np.array(per_source_ts, dtype=np.float32))

    finally:
        os.unlink(bllac_cfg)


class CoincidenceSimulator:
    """
    Forward simulator: theta (N, 7) → x (N, 15).

    For each theta: draw a BL Lac population, simulate IceCube neutrinos,
    compute per-source coincidence TS, compress to summary statistic x.
    """

    def __init__(
        self,
        bllac_base: Optional[Path] = None,
        nu_config: Optional[Path] = None,
        obs_time_yr: float = 10.0,
    ):
        self.bllac_base  = Path(bllac_base  or _BLLAC_REF)
        self.nu_config   = Path(nu_config   or _NU_TRACKS)
        self.obs_time_yr = obs_time_yr
        self.summary     = CoincidenceSummary()
        self.param_space = CoincidenceParamSpace()

    @property
    def d_theta(self) -> int:
        return self.param_space.d_theta

    @property
    def d_x(self) -> int:
        return self.summary.d_x

    def _simulate_one(self, theta: np.ndarray, seed: int) -> np.ndarray:
        params = self.param_space.unpack(theta)
        return _simulate_one(
            params, self.nu_config, self.bllac_base,
            seed, self.summary, self.obs_time_yr
        )

    def __call__(self, theta: np.ndarray, seed: int = 42) -> np.ndarray:
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        return np.array([
            self._simulate_one(theta[i], seed + i)
            for i in range(len(theta))
        ], dtype=np.float32)


def generate_coincidence_training_set(
    simulator: CoincidenceSimulator,
    prior,
    n_sims: int,
    out_path: str,
    seed: int = 42,
    batch_size: int = 20,
):
    """draw theta from prior, simulate x, save to HDF5."""
    import h5py
    import torch

    torch.manual_seed(seed)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    d_theta = simulator.d_theta
    d_x     = simulator.d_x

    theta_all, x_all = [], []
    n_batches = (n_sims + batch_size - 1) // batch_size

    for b in range(n_batches):
        n_draw = min(batch_size, n_sims - b * batch_size)
        theta_np = prior.sample((n_draw,)).numpy().astype(np.float32)
        x_np     = simulator(theta_np, seed=seed + b * batch_size)
        theta_all.append(theta_np)
        x_all.append(x_np)
        print(f"Batch {b+1}/{n_batches}: {n_draw} sims done")

    theta_all = np.concatenate(theta_all)
    x_all     = np.concatenate(x_all)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("theta", data=theta_all)
        f.create_dataset("x",     data=x_all)
        f.attrs["n_sims"]      = n_sims
        f.attrs["seed"]        = seed
        f.attrs["param_names"] = simulator.param_space.names
        f.attrs["d_theta"]     = d_theta
        f.attrs["d_x"]         = d_x

    print(f"Saved {n_sims} coincidence simulations → {out_path}")
    return theta_all, x_all
