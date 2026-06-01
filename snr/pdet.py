"""
Detection probability p_det(N_src, sindec, gamma).

Loads the precomputed HDF5 grid produced by simulate_sources_dec.py, then
provides fast interpolation over (sindec, gamma, N_src).

Grid layout in the H5 file:
  Nsrc_list               : (K,) float
  dec_{sindec}/index_{gamma}/Pdet  : (K,) float
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator


@dataclass
class PdetGrid:
    """Precomputed detection probability grid: P[i_sindec, i_gamma, i_N]."""
    Nsrc_list: np.ndarray
    sindec_vals: np.ndarray
    gamma_vals: np.ndarray
    P: np.ndarray

    def __post_init__(self):
        assert self.P.shape == (
            len(self.sindec_vals), len(self.gamma_vals), len(self.Nsrc_list)
        ), (
            f"P shape {self.P.shape} does not match "
            f"({len(self.sindec_vals)}, {len(self.gamma_vals)}, {len(self.Nsrc_list)})"
        )

    def query(
        self,
        sindec: Union[float, np.ndarray],
        gamma: Union[float, np.ndarray],
        N_src: Union[float, np.ndarray],
    ) -> np.ndarray:
        """bilinearly interpolate p_det at arbitrary (sindec, gamma, N_src)."""
        sindec = np.atleast_1d(np.asarray(sindec, dtype=float))
        gamma = np.atleast_1d(np.asarray(gamma, dtype=float))
        N_src = np.atleast_1d(np.asarray(N_src, dtype=float))

        # clamp to grid
        sd = np.clip(sindec, self.sindec_vals[0], self.sindec_vals[-1])
        gm = np.clip(gamma, self.gamma_vals[0], self.gamma_vals[-1])
        ns = np.clip(N_src, self.Nsrc_list[0], self.Nsrc_list[-1])

        isd_lo, isd_hi, wsd = self._bracket(sd, self.sindec_vals)
        igm_lo, igm_hi, wgm = self._bracket(gm, self.gamma_vals)
        ins_lo, ins_hi, wns = self._bracket(ns, self.Nsrc_list)

        def _interp(isd, igm, ins):
            return self.P[isd, igm, ins]

        pdet = (
            (1 - wsd) * (1 - wgm) * ((1 - wns) * _interp(isd_lo, igm_lo, ins_lo) + wns * _interp(isd_lo, igm_lo, ins_hi))
            + (1 - wsd) * wgm     * ((1 - wns) * _interp(isd_lo, igm_hi, ins_lo) + wns * _interp(isd_lo, igm_hi, ins_hi))
            + wsd       * (1 - wgm) * ((1 - wns) * _interp(isd_hi, igm_lo, ins_lo) + wns * _interp(isd_hi, igm_lo, ins_hi))
            + wsd       * wgm       * ((1 - wns) * _interp(isd_hi, igm_hi, ins_lo) + wns * _interp(isd_hi, igm_hi, ins_hi))
        )
        return np.clip(pdet, 0.0, 1.0)

    @staticmethod
    def _bracket(v: np.ndarray, axis: np.ndarray):
        """return (lo_idx, hi_idx, weight_hi) for linear interpolation."""
        idx_f = np.interp(v, axis, np.arange(len(axis), dtype=float))
        lo = np.floor(idx_f).astype(int).clip(0, len(axis) - 2)
        hi = lo + 1
        w = idx_f - lo
        return lo, hi, w

    def marginal_Nsrc_for_pdet(
        self,
        target_pdet: float,
        sindec: float,
        gamma: float,
    ) -> float:
        """invert the p_det curve at (sindec, gamma) to get N_src for a target p_det."""
        curve = self.query(
            np.full_like(self.Nsrc_list, sindec, dtype=float),
            np.full_like(self.Nsrc_list, gamma, dtype=float),
            self.Nsrc_list,
        )
        return float(np.interp(target_pdet, curve, self.Nsrc_list))


def load_pdet_grid(path: Union[str, Path]) -> PdetGrid:
    """load a precomputed p_det HDF5 file (tree layout: dec_*/index_*/Pdet)."""
    path = Path(path)
    P_dict: Dict[Tuple[float, float], np.ndarray] = {}
    Nsrc_list = None

    with h5py.File(path, "r") as f:
        if "Nsrc_list" not in f:
            raise KeyError(f"{path} missing 'Nsrc_list' dataset.")
        Nsrc_list = f["Nsrc_list"][()].astype(float)

        def _visit(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            m = re.match(r"^dec_([^/]+)/index_([^/]+)/Pdet$", name)
            if not m:
                return
            sd = float(m.group(1))
            gm = float(m.group(2))
            P_dict[(sd, gm)] = obj[()].astype(float)

        f.visititems(_visit)

    if not P_dict:
        raise ValueError(f"No dec_*/index_*/Pdet datasets found in {path}")

    sindec_vals = np.array(sorted({k[0] for k in P_dict}))
    gamma_vals = np.array(sorted({k[1] for k in P_dict}))
    K = len(Nsrc_list)
    D = len(sindec_vals)
    G = len(gamma_vals)

    P = np.zeros((D, G, K), dtype=float)
    for (sd, gm), curve in P_dict.items():
        i = np.searchsorted(sindec_vals, sd)
        j = np.searchsorted(gamma_vals, gm)
        P[i, j, :] = curve[:K]

    return PdetGrid(Nsrc_list=Nsrc_list, sindec_vals=sindec_vals,
                    gamma_vals=gamma_vals, P=P)


def pdet_from_sim_h5(
    sim_h5_path: Union[str, Path],
    N_src_list: np.ndarray,
    n_sindec: int = 30,
    n_gamma: int = 6,
) -> PdetGrid:
    """
    Build a PdetGrid directly from a raw simulation H5 (simulate_sources_dec output).
    p_det is the fraction of simulated events passing the detector.
    """
    with h5py.File(sim_h5_path, "r") as f:
        sindec_grid = f["dec_to_sim"][()]
        gamma_grid = f["index_to_sim"][()]
        sindec_idx = f["sindec_index"][()]
        gamma_idx = f["gamma_index"][()]
        N_per_pair = int(f.attrs["N_per_pair"])

    D = len(sindec_grid)
    G = len(gamma_grid)
    K = len(N_src_list)
    P = np.zeros((D, G, K), dtype=float)

    for i_sd in range(D):
        for i_gm in range(G):
            mask = (sindec_idx == i_sd) & (gamma_idx == i_gm)
            n_detected = int(mask.sum())
            for k, N in enumerate(N_src_list):
                P[i_sd, i_gm, k] = min(n_detected / N_per_pair * (N / N_per_pair), 1.0)

    return PdetGrid(
        Nsrc_list=N_src_list,
        sindec_vals=sindec_grid,
        gamma_vals=gamma_grid,
        P=P,
    )
