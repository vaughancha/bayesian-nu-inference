# Core functions to reproduce Capel, Mortlock, and Finley (2020) Fig. 1
# Formal, normalized implementation of Eqs. (1)-(10)
# with absolute normalization (n0, L, Emin, Emax) and easy normalization for Fig. 1.

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple, Iterable, Dict, Any
import numpy as np
from scipy.integrate import cumulative_trapezoid, quad

# -------------------- Cosmology (flat ΛCDM) --------------------
_C_KM_S = 299_792.458  # km/s

@dataclass(frozen=True)
class CosmologyGrid:
    """Flat ΛCDM cosmology precomputed on a redshift grid (for Eqs. 5–9)."""
    H0: float = 70.0
    Om: float = 0.3
    Ol: float = 0.7
    zmax: float = 6.0
    nz: int = 4000

    def __post_init__(self):
        z = np.linspace(0.0, self.zmax, self.nz)
        object.__setattr__(self, "z", z)
        Ez = np.sqrt(self.Om * (1.0 + z) ** 3 + self.Ol)
        object.__setattr__(self, "Ez", Ez)
        invEz = 1.0 / Ez
        Dc = (_C_KM_S / self.H0) * cumulative_trapezoid(invEz, z, initial=0.0)        # comoving distance
        object.__setattr__(self, "Dc", Dc)
        Dl = (1.0 + z) * Dc                                                             # luminosity distance
        object.__setattr__(self, "Dl", Dl)
        # Eq. (5): dV/dz = 4π c D_L^2 / [H0 (1+z)^2 E(z)]
        dVdz = 4.0 * np.pi * (_C_KM_S / self.H0) * (Dl**2) / ((1.0 + z) ** 2 * Ez)
        object.__setattr__(self, "dVdz", dVdz)

    # Vectorized interpolators
    def luminosity_distance(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(np.asarray(z), 0.0, self.zmax)
        return np.interp(z, self.z, self.Dl)

    def dVdz_of_z(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(np.asarray(z), 0.0, self.zmax)
        return np.interp(z, self.z, self.dVdz)


# -------------------- Population evolution f(z) (Eq. 3) --------------------
def fz_powerlaw(z: np.ndarray, p1: float, p2: float, zc: float, normalize: bool = True) -> np.ndarray:
    """
    f(z) = (1+z)^p1 * (1 + z/zc)^p2  [Eq. (3) up to the overall n0]
    """
    z = np.asarray(z, dtype=float)
    f = (1.0 + z) ** p1 * (1.0 + z / zc) ** p2
    return f / f[0] if normalize else f

def fz_flat(z: np.ndarray) -> np.ndarray:
    return np.ones_like(z)

def fz_negative(z: np.ndarray) -> np.ndarray:
    return (1.0 + z) ** (-1.0)

def fz_sfr_md14(z: np.ndarray, normalize: bool = True) -> np.ndarray:
    # Madau & Dickinson (2014) SFR proxy used in the Fig. 1 caption; normalized to f(0)=1 for overlay.
    num = (1.0 + z) ** 2.7
    den = 1.0 + ((1.0 + z) / 2.9) ** 5.6
    f = num / den
    return f / f[0] if normalize else f


# -------------------- Single-source spectrum & normalization (Eqs. 1–2) --------------------
@dataclass(frozen=True)
class SpectrumParams:
    """
    Power-law spectrum normalized by energy luminosity L in [Emin, Emax]  (Eqs. 1-2).
    All energies must be in consistent units; L must use the same implied energy unit.
    """
    gamma: float
    Emin: float
    Emax: float
    L: float        # isotropic-equivalent luminosity (erg/s or matching units)

def k_gamma(gamma: float, Emin: float, Emax: float) -> float:
    """
    Energy-normalization constant k_γ s.t. L = ∫ E [L k_γ E^{-γ}] dE = L.
    For γ ≠ 2: k_γ = (2-γ) / (Emax^{2-γ} - Emin^{2-γ}); for γ = 2: k_2 = 1 / ln(Emax/Emin).
    """
    if gamma == 2.0:
        return 1.0 / np.log(Emax / Emin)
    return (2.0 - gamma) / (Emax ** (2.0 - gamma) - Emin ** (2.0 - gamma))

# nu_pop_core.py

_MPC_TO_CM = 3.085677581e24  # cm / Mpc

def dNsrc_dEdtdA_earth(E, z, cosmo, spec):
    E = np.asarray(E, dtype=float)

    Dl_mpc = cosmo.luminosity_distance(np.atleast_1d(z))[0]
    Dl_cm  = Dl_mpc * _MPC_TO_CM

    kg = k_gamma(spec.gamma, spec.Emin, spec.Emax)

    pref = spec.L * kg * (1.0 + z)**(2.0 - spec.gamma) / (4.0 * np.pi * Dl_cm**2)
    return pref * (E**(-spec.gamma))



# -------------------- Population flux (Eqs. 8–9) --------------------
@dataclass(frozen=True)
class PopulationParams:
    """Population parameters: local density n0 and evolution f(z) (Eq. 3)."""
    n0: float                      # Mpc^-3
    fz_fn: Callable[[np.ndarray], np.ndarray]

def population_differential_flux(E: np.ndarray, z: np.ndarray,
                                 cosmo: CosmologyGrid,
                                 spec: SpectrumParams,
                                 pop: PopulationParams) -> np.ndarray:
    """
    Differential *per-solid-angle* flux from the population (Eq. 8 integrand before z-integration):
      dN̄^tot_ν / dE dt dA dΩ = (1/4π) ∫ dz [ n0 f(z) dV/dz * dN̄^src_ν/dE dt dA ].
    Returns the z-integrand (array over z) for later integration/normalization.
    """
    z = np.asarray(z, dtype=float)
    dVdz = cosmo.dVdz_of_z(z)
    fz = pop.fz_fn(z)
    src = dNsrc_dEdtdA_earth(E, z, cosmo, spec)              # broadcast over z
    return (pop.n0 * fz * dVdz) * src / (4.0 * np.pi)

def population_number_flux_integrand(
    z: np.ndarray,
    cosmo: CosmologyGrid,
    spec: SpectrumParams,
    pop: PopulationParams,
) -> np.ndarray:
    """
    Robust Eq. (9) integrand without 0/0 at z=0:
      I(z) ∝ n0 * f(z) * (c/H0) * L * (1+z)^(-gamma) / E(z),
    where E(z)=sqrt(Om (1+z)^3 + Ol).
    """
    z = np.asarray(z, dtype=float)
    fz = pop.fz_fn(z)
    Ez = np.sqrt(cosmo.Om * (1.0 + z) ** 3 + cosmo.Ol)
    return pop.n0 * fz * ( _C_KM_S / cosmo.H0 ) * spec.L * (1.0 + z) ** (-spec.gamma) / Ez


def total_number_flux(spec: SpectrumParams, pop: PopulationParams, cosmo: CosmologyGrid) -> float:
    """Integrate Eq. (9) over z (all sky already accounted for in the formula)."""
    z = cosmo.z
    integrand = population_number_flux_integrand(z, cosmo, spec, pop)
    return np.trapezoid(integrand, z)


# -------------------- Expected counts (Eq. 10) --------------------
@dataclass(frozen=True)
class ObservationParams:
    """
    Observation configuration for expected counts (Eq. 10).
    A_eff can be a function A_eff(E, delta) or a function of E only (delta ignored).
    """
    T: float                                   # exposure time (s)
    A_eff: Callable[[np.ndarray, float], np.ndarray]  # effective area function

def expected_counts_from_one_source(z: float, delta: float,
                                    spec: SpectrumParams,
                                    obs: ObservationParams,
                                    cosmo: CosmologyGrid) -> float:
    """
    Eq. (10): N̄^src_nu = T ∫_{Emin}^{Emax} dE A_eff(E, δ) [dN̄^src_nu / dE dt dA].
    """
    Emin, Emax = spec.Emin, spec.Emax

    def integrand(E):
        return obs.A_eff(np.asarray(E, dtype=float), float(delta)) * dNsrc_dEdtdA_earth(np.asarray(E, dtype=float), z, cosmo, spec)

    val, _ = quad(lambda EE: float(integrand(EE)), Emin, Emax, epsabs=0, epsrel=1e-5, limit=200)
    return obs.T * val

def expected_counts_all_sky_average(spec: SpectrumParams,
                                    pop: PopulationParams,
                                    obs: ObservationParams,
                                    cosmo: CosmologyGrid,
                                    delta_sampler: Callable[[int], np.ndarray] | None = None,
                                    n_delta: int = 60) -> float:
    """
    Approximate total expected detected counts from the whole population:
      N̄_tot ≈ ∫ dz [ n0 f(z) dV/dz / (4 pi) ] * <N̄^src_nu(z, \delta)>_{sky}
    where the sky-average is approximated by sampling declinations (or using a flat A_eff(E) if \delta ignored).
    """
    z = cosmo.z
    dVdz = cosmo.dVdz_of_z(z)
    fz = pop.fz_fn(z)

    # sample declinations uniformly in sin(δ) if a sampler isn't provided
    if delta_sampler is None:
        u = np.linspace(-1.0, 1.0, n_delta)
        deltas = np.arcsin(u)
    else:
        deltas = delta_sampler(n_delta)

    # sky-averaged counts per source as a function of z
    per_z = np.zeros_like(z)
    for i, zi in enumerate(z):
        # average over sampled declinations
        counts = [expected_counts_from_one_source(zi, float(d), spec, obs, cosmo) for d in deltas]
        # weight ∝ dΩ ∝ cos(δ) dδ → uniform in sin(δ), so just mean over samples
        per_z[i] = np.mean(counts)

    integrand = (pop.n0 * fz * dVdz / (4.0 * np.pi)) * per_z
    return float(np.trapezoid(integrand, z))


# -------------------- Fig. 1 objects by normalization (from Eq. 9) --------------------
def fig1_Pz_and_cdf(z: np.ndarray,
                    cosmo: CosmologyGrid,
                    spec: SpectrumParams,
                    pop: PopulationParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build P(z) and P(<z) *by normalizing* the Eq. (9) integrand (Fig. 1 middle/bottom).
    """
    I = population_number_flux_integrand(z, cosmo, spec, pop)
    area = np.trapezoid(I, z)
    P = I / (area + 1e-300)
    C = np.cumsum(P) * (z[1] - z[0])
    C[-1] = 1.0
    return P, C


# -------------------- Families / loops --------------------
def loop_models_family(z: np.ndarray,
                       cosmo: CosmologyGrid,
                       spec: SpectrumParams,
                       families: Iterable[Dict[str, Any]]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Convenience helper: given a list of dicts like
      {"label":"SFR", "n0":1e-6, "fz": fz_sfr_md14(z)},
    compute P(z), C(z) for each and return label -> (P, C).
    """
    out = {}
    for m in families:
        pop = PopulationParams(n0=float(m["n0"]), fz_fn=(lambda zz, arr=np.asarray(m["fz"], float): np.interp(zz, z, arr)))
        P, C = fig1_Pz_and_cdf(z, cosmo, spec, pop)
        out[m.get("label", "model")] = (P, C)
    return out


# -------------------- Calibration to a target N_obs --------------------
def calibrate_L_for_target_counts(target_N: float,
                                  n0: float,
                                  fz_fn: Callable[[np.ndarray], np.ndarray],
                                  spec: SpectrumParams,
                                  obs: ObservationParams,
                                  cosmo: CosmologyGrid) -> float:
    """
    Solve for luminosity L such that expected_counts_all_sky_average(...) == target_N (with given n0).
    Useful to match an observed total event count or to set discovery scenarios.
    """
    from math import isfinite
    pop_template = PopulationParams(n0=n0, fz_fn=fz_fn)

    def f(Lval: float) -> float:
        sp = SpectrumParams(gamma=spec.gamma, Emin=spec.Emin, Emax=spec.Emax, L=Lval)
        return expected_counts_all_sky_average(sp, pop_template, obs, cosmo) - target_N

    # scalar root-finding on log L for stability
    lo, hi = 1e35, 1e48
    for _ in range(80):
        mid = 10 ** ((np.log10(lo) + np.log10(hi)) * 0.5)
        val = f(mid)
        if not isfinite(val):
            break
        if val > 0:
            hi = mid
        else:
            lo = mid
        if abs(val) / (target_N + 1e-12) < 1e-3:
            return mid
    return mid  #
