"""
Point-source TS scan over a HEALPix sky map.

The likelihood per pixel:
  TS = 2 [ log L(ns_hat, gamma_hat | events) - log L(0 | events) ]

maximised over (ns >= 0, gamma) using icecube_tools PointSourceLikelihood.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import healpy as hp

try:
    from icecube_tools.point_source_likelihood.energy_likelihood import (
        MarginalisedEnergyLikelihoodFromSim,
        MarginalisedEnergyLikelihoodFixed,
    )
    from icecube_tools.detector.angular_resolution import AngularResolution
    from icecube_tools.point_source_likelihood.spatial_likelihood import (
        EnergyDependentSpatialGaussianLikelihood,
    )
    from icecube_tools.point_source_likelihood.prior import GaussianPrior
    from icecube_tools.point_source_likelihood.point_source_likelihood import (
        PointSourceLikelihood,
    )
    _HAS_ICECUBE_TOOLS = True
except ImportError:
    _HAS_ICECUBE_TOOLS = False


@dataclass
class TSResult:
    """output of a TSScan run stored as HEALPix maps."""
    nside: int
    nest: bool
    pix: np.ndarray
    ts: np.ndarray
    ns_hat: np.ndarray
    gamma_hat: np.ndarray
    n_events: int
    elapsed_s: float

    def to_healpix_map(self, fill_value: float = hp.UNSEEN) -> np.ndarray:
        """full-sky HEALPix array with fill_value for unscanned pixels."""
        m = np.full(hp.nside2npix(self.nside), fill_value, dtype=np.float32)
        m[self.pix] = self.ts
        return m

    def best_fit(self) -> Tuple[float, float, float, float, float]:
        """(ra_rad, dec_rad, ts, ns_hat, gamma_hat) at the maximum TS pixel."""
        idx = int(np.argmax(self.ts))
        theta, phi = hp.pix2ang(self.nside, int(self.pix[idx]), nest=self.nest)
        dec = np.pi / 2.0 - theta
        return float(phi), float(dec), float(self.ts[idx]), float(self.ns_hat[idx]), float(self.gamma_hat[idx])

    def dec_of_pixels(self) -> np.ndarray:
        """declination (rad) of each scanned pixel."""
        theta, _ = hp.pix2ang(self.nside, self.pix, nest=self.nest)
        return np.pi / 2.0 - theta

    def ra_of_pixels(self) -> np.ndarray:
        """RA (rad) of each scanned pixel."""
        _, phi = hp.pix2ang(self.nside, self.pix, nest=self.nest)
        return phi

    def save(self, path: Union[str, Path]):
        np.savez_compressed(
            path,
            nside=self.nside,
            nest=self.nest,
            pix=self.pix,
            ts=self.ts,
            ns_hat=self.ns_hat,
            gamma_hat=self.gamma_hat,
            n_events=self.n_events,
            elapsed_s=self.elapsed_s,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TSResult":
        d = np.load(path)
        return cls(
            nside=int(d["nside"]),
            nest=bool(d["nest"]),
            pix=d["pix"],
            ts=d["ts"],
            ns_hat=d["ns_hat"],
            gamma_hat=d["gamma_hat"],
            n_events=int(d["n_events"]),
            elapsed_s=float(d["elapsed_s"]),
        )


def _patch_angres(ang):
    """normalise AngularResolution API across icecube_tools versions."""
    for attr in ("__call__", "get_angular_resolution", "_get_angular_resolution"):
        if hasattr(ang, attr):
            ang._get_angular_resolution = getattr(ang, attr)
            return ang
    raise AttributeError("Cannot find angular resolution method on AngularResolution.")


class TSScan:
    """IceCube point-source TS scan over a HEALPix pixelisation."""

    def __init__(
        self,
        events_ra: np.ndarray,
        events_dec: np.ndarray,
        events_reco_energy: np.ndarray,
        angres_E2_csv: Union[str, Path],
        angres_atmos_csv: Union[str, Path],
        bg_h5: Union[str, Path],
        signal_pl_h5: Union[str, Path],
        gamma_prior_mean: float = 2.19,
        gamma_prior_sigma: float = 0.1,
        band_width_factor: float = 3.0,
    ):
        if not _HAS_ICECUBE_TOOLS:
            raise ImportError(
                "icecube_tools is required for TSScan. "
                "Install from https://github.com/cescalara/icecube_tools"
            )

        self.events_ra = np.asarray(events_ra, dtype=float)
        self.events_dec = np.asarray(events_dec, dtype=float)
        self.events_reco_energy = np.asarray(events_reco_energy, dtype=float)
        self.band_width_factor = band_width_factor
        self.gamma_prior_mean = gamma_prior_mean

        ang_E2 = _patch_angres(AngularResolution(str(angres_E2_csv)))
        ang_atmos = _patch_angres(AngularResolution(str(angres_atmos_csv)))
        self._spatial = EnergyDependentSpatialGaussianLikelihood(ang_E2, ang_atmos)
        self._energy_bg = MarginalisedEnergyLikelihoodFixed(str(bg_h5))
        self._energy_sig = MarginalisedEnergyLikelihoodFromSim(str(signal_pl_h5))
        self._prior = GaussianPrior(gamma_prior_mean, gamma_prior_sigma)

    def _ts_at_pixel(self, ra_src: float, dec_src: float) -> Tuple[float, float, float]:
        llh = PointSourceLikelihood(
            ra_src=ra_src,
            dec_src=dec_src,
            ra=self.events_ra,
            dec=self.events_dec,
            reco_energy=self.events_reco_energy,
            spatial_likelihood=self._spatial,
            energy_likelihood_signal=self._energy_sig,
            energy_likelihood_background=self._energy_bg,
            prior=self._prior,
            band_width_factor=self.band_width_factor,
        )
        if llh.N == 0:
            return 0.0, 0.0, self.gamma_prior_mean
        ts, ns_hat, gamma_hat = llh.maximise()
        return max(0.0, float(ts)), max(0.0, float(ns_hat)), float(gamma_hat)

    def run(
        self,
        nside: int = 64,
        nest: bool = False,
        dec_min_rad: Optional[float] = None,
        dec_max_rad: Optional[float] = None,
        verbose: bool = True,
    ) -> TSResult:
        """
        Scan all HEALPix pixels at the given nside, optionally restricted to a
        declination band.

        nside=64 → ~0.9° pixel width; nside=128 → ~0.46°.
        Restricting the dec band gives a large speed-up.
        """
        n_pix_total = hp.nside2npix(nside)
        all_pix = np.arange(n_pix_total)
        theta_all, phi_all = hp.pix2ang(nside, all_pix, nest=nest)
        dec_all = np.pi / 2.0 - theta_all
        ra_all = phi_all

        pad = np.deg2rad(5.0)
        dec_lo = float(self.events_dec.min()) - pad if dec_min_rad is None else dec_min_rad
        dec_hi = float(self.events_dec.max()) + pad if dec_max_rad is None else dec_max_rad
        dec_lo = max(dec_lo, -np.pi / 2.0)
        dec_hi = min(dec_hi, np.pi / 2.0)

        band_mask = (dec_all >= dec_lo) & (dec_all <= dec_hi)
        scan_pix = all_pix[band_mask]
        scan_ra = ra_all[band_mask]
        scan_dec = dec_all[band_mask]

        n_scan = len(scan_pix)
        ts_arr = np.zeros(n_scan, dtype=np.float32)
        ns_arr = np.zeros(n_scan, dtype=np.float32)
        gm_arr = np.zeros(n_scan, dtype=np.float32)

        t0 = time.time()
        for i in range(n_scan):
            ts_arr[i], ns_arr[i], gm_arr[i] = self._ts_at_pixel(
                float(scan_ra[i]), float(scan_dec[i])
            )
            if verbose and (i + 1) % max(1, n_scan // 20) == 0:
                print(f"  TS scan {100*(i+1)/n_scan:.0f}% ({i+1}/{n_scan})", flush=True)

        elapsed = time.time() - t0
        if verbose:
            print(f"TS scan complete: {n_scan} pixels in {elapsed:.1f}s "
                  f"(nside={nside}, dec=[{np.degrees(dec_lo):.1f}°, {np.degrees(dec_hi):.1f}°])")

        return TSResult(
            nside=nside, nest=nest,
            pix=scan_pix, ts=ts_arr, ns_hat=ns_arr, gamma_hat=gm_arr,
            n_events=len(self.events_ra), elapsed_s=elapsed,
        )


def snr_vs_dec(
    result: TSResult,
    n_dec_bins: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """RA-averaged TS (proxy for SNR) as a function of declination."""
    dec_pix = result.dec_of_pixels()
    dec_lo = float(dec_pix.min())
    dec_hi = float(dec_pix.max())
    dec_edges = np.linspace(dec_lo, dec_hi, n_dec_bins + 1)
    dec_centers = 0.5 * (dec_edges[:-1] + dec_edges[1:])
    mean_ts = np.zeros(n_dec_bins)

    for k in range(n_dec_bins):
        mask = (dec_pix >= dec_edges[k]) & (dec_pix < dec_edges[k + 1])
        if mask.any():
            mean_ts[k] = float(result.ts[mask].max())

    return dec_centers, mean_ts


def plot_ts_map(
    result: TSResult,
    title: str = "TS map",
    out_path: Optional[Union[str, Path]] = None,
    coord: str = "C",
) -> None:
    """mollweide projection of the TS map using healpy."""
    m = result.to_healpix_map(fill_value=hp.UNSEEN)
    hp.mollview(m, title=title, coord=coord, unit="TS",
                min=0, cmap="inferno")
    hp.graticule()
    if out_path is not None:
        import matplotlib.pyplot as plt
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
