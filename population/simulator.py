"""
Population forward simulator for SBI.
Maps theta = (log10_n0, gamma, log10_L, p1, p2, zc) to expected event counts
per (energy bin × declination bin).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from .physics import (
    CosmologyGrid, SpectrumParams, PopulationParams, ObservationParams,
    expected_counts_from_one_source, fz_powerlaw,
)


@dataclass
class PopParamSpace:
    """
    6-parameter space: theta = [log10_n0, gamma, log10_L, p1, p2, zc].
    Bounds are used for prior construction.
    """
    log10_n0_bounds: Tuple[float, float] = (-10.0, -3.0)  # Mpc^-3
    gamma_bounds: Tuple[float, float] = (1.5, 3.5)
    log10_L_bounds: Tuple[float, float] = (40.0, 48.0)    # erg/s
    p1_bounds: Tuple[float, float] = (-3.0, 6.0)
    p2_bounds: Tuple[float, float] = (-6.0, 0.0)
    zc_bounds: Tuple[float, float] = (0.5, 5.0)

    @property
    def d_theta(self) -> int:
        return 6

    @property
    def names(self) -> List[str]:
        return ["log10_n0", "gamma", "log10_L", "p1", "p2", "zc"]

    def unpack(self, theta: np.ndarray):
        """theta (6,) -> (n0, gamma, L, p1, p2, zc)."""
        t = np.asarray(theta).ravel()
        return (
            10.0 ** t[0],   # n0
            float(t[1]),     # gamma
            10.0 ** t[2],   # L
            float(t[3]),     # p1
            float(t[4]),     # p2
            float(t[5]),     # zc
        )

    def make_torch_prior(self, device: str = "cpu"):
        """uniform prior over the parameter space."""
        import torch
        from torch.distributions import Uniform, Independent
        lows = torch.tensor([
            self.log10_n0_bounds[0], self.gamma_bounds[0], self.log10_L_bounds[0],
            self.p1_bounds[0], self.p2_bounds[0], self.zc_bounds[0],
        ], device=device)
        highs = torch.tensor([
            self.log10_n0_bounds[1], self.gamma_bounds[1], self.log10_L_bounds[1],
            self.p1_bounds[1], self.p2_bounds[1], self.zc_bounds[1],
        ], device=device)
        return Independent(Uniform(lows, highs), 1)


@dataclass
class SummarySpec:
    """
    Summary statistic x: expected counts per (E-bin, dec-bin).
    x is a flat vector of length n_E_bins × n_dec_bins.
    """
    E_edges: np.ndarray    # (n_E_bins+1,) in GeV
    dec_edges: np.ndarray  # (n_dec_bins+1,) in radians

    @property
    def n_E_bins(self) -> int:
        return len(self.E_edges) - 1

    @property
    def n_dec_bins(self) -> int:
        return len(self.dec_edges) - 1

    @property
    def d_x(self) -> int:
        return self.n_E_bins * self.n_dec_bins

    @property
    def E_centers(self) -> np.ndarray:
        return np.sqrt(self.E_edges[:-1] * self.E_edges[1:])

    @property
    def dec_centers(self) -> np.ndarray:
        return 0.5 * (self.dec_edges[:-1] + self.dec_edges[1:])

    @classmethod
    def default(cls) -> "SummarySpec":
        """6 log-spaced energy bins [1e2, 1e7] GeV × 10 dec bins."""
        return cls(
            E_edges=np.logspace(2, 7, 7),
            dec_edges=np.linspace(-np.pi / 2, np.pi / 2, 11),
        )


class PopulationSimulator:
    """
    Forward simulator: theta (N, 6) → x (N, d_x).

    Integrates expected neutrino counts via Eq. (10) of Capel+ 2020,
    summed over redshift, for each (theta, dec_bin) cell.
    """

    def __init__(
        self,
        cosmo: CosmologyGrid,
        obs: ObservationParams,
        param_space: Optional[PopParamSpace] = None,
        summary: Optional[SummarySpec] = None,
        n_z: int = 200,
    ):
        self.cosmo = cosmo
        self.obs = obs
        self.param_space = param_space or PopParamSpace()
        self.summary = summary or SummarySpec.default()
        self.n_z = n_z

        self._z_grid = np.linspace(0.01, cosmo.zmax, n_z)
        self._dVdz = cosmo.dVdz_of_z(self._z_grid)

    def _simulate_one(self, theta: np.ndarray) -> np.ndarray:
        n0, gamma, L, p1, p2, zc = self.param_space.unpack(theta)
        spec = SpectrumParams(gamma=gamma, Emin=self.summary.E_edges[0],
                               Emax=self.summary.E_edges[-1], L=L)

        fz = fz_powerlaw(self._z_grid, p1, p2, zc, normalize=True)
        pop_weight = n0 * fz * self._dVdz  # (n_z,)

        x = np.zeros(self.summary.d_x, dtype=np.float32)
        idx = 0
        for i_E in range(self.summary.n_E_bins):
            Emin_bin = self.summary.E_edges[i_E]
            Emax_bin = self.summary.E_edges[i_E + 1]
            spec_bin = SpectrumParams(gamma=gamma, Emin=Emin_bin, Emax=Emax_bin, L=L)

            for i_dec in range(self.summary.n_dec_bins):
                dec = float(self.summary.dec_centers[i_dec])

                counts_per_src = np.array([
                    expected_counts_from_one_source(
                        float(z), dec, spec_bin, self.obs, self.cosmo
                    )
                    for z in self._z_grid
                ])
                x[idx] = float(np.trapezoid(pop_weight * counts_per_src, self._z_grid))
                idx += 1

        return x

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        return np.array([self._simulate_one(th) for th in theta], dtype=np.float32)

    @property
    def d_theta(self) -> int:
        return self.param_space.d_theta

    @property
    def d_x(self) -> int:
        return self.summary.d_x


def generate_training_set(
    simulator: PopulationSimulator,
    prior,
    n_sims: int,
    out_path: str,
    seed: int = 42,
    batch_size: int = 100,
):
    """draw theta from prior, simulate x, save to HDF5."""
    import h5py
    import torch
    torch.manual_seed(seed)

    n_batches = (n_sims + batch_size - 1) // batch_size
    theta_all, x_all = [], []

    for b in range(n_batches):
        n_draw = min(batch_size, n_sims - b * batch_size)
        theta_t = prior.sample((n_draw,))
        theta_np = theta_t.cpu().numpy().astype(np.float32)
        x_np = simulator(theta_np)
        theta_all.append(theta_np)
        x_all.append(x_np)
        print(f"Batch {b+1}/{n_batches}: {n_draw} sims done")

    theta_all = np.concatenate(theta_all, axis=0)
    x_all = np.concatenate(x_all, axis=0)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("theta", data=theta_all)
        f.create_dataset("x", data=x_all)
        f.attrs["n_sims"] = n_sims
        f.attrs["seed"] = seed
        f.attrs["param_names"] = simulator.param_space.names
        f.attrs["d_theta"] = simulator.d_theta
        f.attrs["d_x"] = simulator.d_x
        f.create_dataset("E_edges", data=simulator.summary.E_edges)
        f.create_dataset("dec_edges", data=simulator.summary.dec_edges)

    print(f"Saved {n_sims} simulations → {out_path}")
    return theta_all, x_all
