"""
Coincidence test statistic (SNR layer).

For each neutrino-source pair, computes:
    TS_ij = 2 * log [ S(ΔΨ, E, Δt | source_j) / B(ΔΨ, E) ]

where:
    S = space_pdf(ΔΨ, σ_PSF) × energy_pdf(E, γ_src) × time_pdf(Δt, t_flare)
    B = 1/(4π) × energy_pdf(E, γ_atm)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


def space_pdf(dpsi_deg: torch.Tensor, sigma_deg: torch.Tensor) -> torch.Tensor:
    """2D Gaussian PSF on sphere."""
    coef = 1.0 / (2.0 * math.pi * sigma_deg ** 2)
    return coef * torch.exp(-0.5 * (dpsi_deg / sigma_deg) ** 2)


def energy_pdf(E_gev: torch.Tensor, gamma: float, E0_gev: float = 1e5) -> torch.Tensor:
    """power-law energy spectrum (E/E0)^{-gamma}."""
    return torch.pow(E_gev / E0_gev, -gamma)


def time_pdf(dt_yr: torch.Tensor, flare_duration_yr: float) -> torch.Tensor:
    """top-hat flare window: 1/dt_flare inside [0, dt_flare], 0 outside."""
    inside = (dt_yr >= 0) & (dt_yr <= flare_duration_yr)
    return inside.float() / max(flare_duration_yr, 1e-9)


def angular_separation_deg(
    ra1_deg: float, dec1_deg: float,
    ra2: torch.Tensor, dec2: torch.Tensor,
) -> torch.Tensor:
    """great-circle angular separation in degrees."""
    ra1  = math.radians(ra1_deg)
    dec1 = math.radians(dec1_deg)
    ra2r  = torch.deg2rad(ra2)
    dec2r = torch.deg2rad(dec2)
    cos_dpsi = (
        math.sin(dec1) * torch.sin(dec2r)
        + math.cos(dec1) * torch.cos(dec2r) * torch.cos(torch.tensor(ra1) - ra2r)
    )
    return torch.rad2deg(torch.acos(torch.clamp(cos_dpsi, -1.0, 1.0)))


@dataclass
class CoincidenceLikelihood:
    """per-event signal and background likelihoods for a neutrino-source pair."""
    src_ra_deg: float
    src_dec_deg: float
    src_gamma: float
    flare_duration_yr: float
    flare_t0_yr: float
    atm_gamma: float = 3.7
    obs_time_yr: float = 10.0

    def signal(
        self,
        ev_ra_deg: torch.Tensor,
        ev_dec_deg: torch.Tensor,
        ev_sigma_deg: torch.Tensor,
        ev_E_gev: torch.Tensor,
        ev_t_yr: torch.Tensor,
    ) -> torch.Tensor:
        """S_i = space × energy_signal × time  (unnormalised)."""
        dpsi  = angular_separation_deg(self.src_ra_deg, self.src_dec_deg,
                                        ev_ra_deg, ev_dec_deg)
        S_sp  = space_pdf(dpsi, ev_sigma_deg)
        S_en  = energy_pdf(ev_E_gev, self.src_gamma)
        dt    = ev_t_yr - self.flare_t0_yr
        S_t   = time_pdf(dt, self.flare_duration_yr)
        return S_sp * S_en * S_t

    def background(
        self,
        ev_E_gev: torch.Tensor,
    ) -> torch.Tensor:
        """B_i = isotropic sky × atmospheric energy spectrum."""
        B_iso = 1.0 / (4.0 * math.pi)
        B_en  = energy_pdf(ev_E_gev, self.atm_gamma)
        return torch.full_like(ev_E_gev, B_iso) * B_en


def coincidence_ts(
    llh: CoincidenceLikelihood,
    ev_ra_deg: np.ndarray,
    ev_dec_deg: np.ndarray,
    ev_sigma_deg: np.ndarray,
    ev_E_gev: np.ndarray,
    ev_t_yr: np.ndarray,
    eps: float = 1e-30,
) -> np.ndarray:
    """
    Per-event coincidence TS for all events against one source.

    TS_i = 2 * log( S_i / B_i ).
    Positive TS means the event is more likely from the source than background.
    """
    ra  = torch.as_tensor(ev_ra_deg,   dtype=torch.float32)
    dec = torch.as_tensor(ev_dec_deg,  dtype=torch.float32)
    sig = torch.as_tensor(ev_sigma_deg,dtype=torch.float32)
    E   = torch.as_tensor(ev_E_gev,    dtype=torch.float32)
    t   = torch.as_tensor(ev_t_yr,     dtype=torch.float32)

    with torch.no_grad():
        S = llh.signal(ra, dec, sig, E, t)
        B = llh.background(E)
        ts = 2.0 * torch.log((S + eps) / (B + eps))

    return ts.numpy()


def max_coincidence_ts(
    llh: CoincidenceLikelihood,
    ev_ra_deg: np.ndarray,
    ev_dec_deg: np.ndarray,
    ev_sigma_deg: np.ndarray,
    ev_E_gev: np.ndarray,
    ev_t_yr: np.ndarray,
) -> float:
    """maximum TS over all events — single-source detection statistic."""
    ts = coincidence_ts(llh, ev_ra_deg, ev_dec_deg, ev_sigma_deg, ev_E_gev, ev_t_yr)
    return float(ts.max()) if len(ts) > 0 else 0.0
