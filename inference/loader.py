"""
Data loaders for the SBI inference engine.
Supports numpy arrays, NPZ archives, and HDF5 files.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import h5py
import numpy as np


class _BaseLoader(ABC):
    """abstract base: anything that serves (theta, x) pairs to an SBIRunner."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def get_all_data(self) -> np.ndarray:
        """returns summary statistics x, shape (N, d_x)."""
        ...

    @abstractmethod
    def get_all_parameters(self) -> np.ndarray:
        """returns parameters theta, shape (N, d_theta)."""
        ...

    @abstractmethod
    def get_obs_data(self) -> np.ndarray:
        """returns observed data x_obs, shape (1, d_x) or (d_x,)."""
        ...

    @abstractmethod
    def get_fid_parameters(self) -> Optional[np.ndarray]:
        """returns fiducial theta corresponding to x_obs, or None."""
        ...


class NumpyLoader(_BaseLoader):
    """load pre-simulated (theta, x) pairs from numpy arrays or NPZ files."""

    def __init__(
        self,
        x: Union[np.ndarray, str, Path],
        theta: Union[np.ndarray, str, Path],
        x_obs: Optional[Union[np.ndarray, str, Path]] = None,
        theta_fid: Optional[Union[np.ndarray, str, Path]] = None,
    ):
        self._x = self._load(x)
        self._theta = self._load(theta)

        if len(self._x) != len(self._theta):
            raise ValueError(
                f"x and theta must have the same number of samples; "
                f"got {len(self._x)} and {len(self._theta)}"
            )

        self._x_obs = self._load(x_obs) if x_obs is not None else None
        self._theta_fid = self._load(theta_fid) if theta_fid is not None else None

    @staticmethod
    def _load(src: Union[np.ndarray, str, Path, None]) -> Optional[np.ndarray]:
        if src is None:
            return None
        if isinstance(src, np.ndarray):
            return src.astype(np.float32)
        p = Path(src)
        if p.suffix == ".npz":
            z = np.load(p)
            key = sorted(z.files)[0]
            return z[key].astype(np.float32)
        return np.load(p).astype(np.float32)

    def __len__(self) -> int:
        return len(self._x)

    def get_all_data(self) -> np.ndarray:
        return self._x

    def get_all_parameters(self) -> np.ndarray:
        return self._theta

    def get_obs_data(self) -> np.ndarray:
        if self._x_obs is None:
            raise RuntimeError("No x_obs was provided to NumpyLoader.")
        return self._x_obs

    def get_fid_parameters(self) -> Optional[np.ndarray]:
        return self._theta_fid


class H5Loader(_BaseLoader):
    """
    Load pre-simulated (theta, x) pairs from an HDF5 file.

    Expected datasets: x (N, d_x), theta (N, d_theta),
    and optionally x_obs and theta_fid.
    """

    def __init__(
        self,
        path: Union[str, Path],
        x_key: str = "x",
        theta_key: str = "theta",
        x_obs_key: str = "x_obs",
        theta_fid_key: str = "theta_fid",
    ):
        self._path = Path(path)
        self._x_key = x_key
        self._theta_key = theta_key
        self._x_obs_key = x_obs_key
        self._theta_fid_key = theta_fid_key
        self._cache: dict = {}
        self._len: int = self._read_len()

    def _read_len(self) -> int:
        with h5py.File(self._path, "r") as f:
            return f[self._x_key].shape[0]

    def _read(self, key: str) -> Optional[np.ndarray]:
        with h5py.File(self._path, "r") as f:
            if key not in f:
                return None
            return f[key][()].astype(np.float32)

    def __len__(self) -> int:
        return self._len

    def get_all_data(self) -> np.ndarray:
        if "x" not in self._cache:
            self._cache["x"] = self._read(self._x_key)
        return self._cache["x"]

    def get_all_parameters(self) -> np.ndarray:
        if "theta" not in self._cache:
            self._cache["theta"] = self._read(self._theta_key)
        return self._cache["theta"]

    def get_obs_data(self) -> np.ndarray:
        arr = self._read(self._x_obs_key)
        if arr is None:
            raise RuntimeError(f"No '{self._x_obs_key}' dataset in {self._path}")
        return arr

    def get_fid_parameters(self) -> Optional[np.ndarray]:
        return self._read(self._theta_fid_key)


class SimulatorLoader(_BaseLoader):
    """
    On-the-fly loader: draws theta from a prior and calls a simulator to get x.
    Suitable for sequential (multi-round) inference where the proposal is updated each round.
    """

    def __init__(
        self,
        simulator: Callable,
        prior: Any,
        n_sims: int,
        x_obs: np.ndarray,
        theta_fid: Optional[np.ndarray] = None,
    ):
        self._simulator = simulator
        self._prior = prior
        self._n_sims = n_sims
        self._x_obs = np.asarray(x_obs, dtype=np.float32)
        self._theta_fid = (
            np.asarray(theta_fid, dtype=np.float32) if theta_fid is not None else None
        )
        self._x: Optional[np.ndarray] = None
        self._theta: Optional[np.ndarray] = None

    def simulate(
        self, proposal=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """draw theta from proposal (or prior) and simulate x."""
        dist = proposal if proposal is not None else self._prior
        import torch
        theta_t = dist.sample((self._n_sims,))
        theta = theta_t.detach().cpu().numpy().astype(np.float32)
        x = np.asarray(self._simulator(theta), dtype=np.float32)
        self._theta = theta
        self._x = x
        return theta, x

    def __len__(self) -> int:
        return 0 if self._x is None else len(self._x)

    def get_all_data(self) -> np.ndarray:
        if self._x is None:
            raise RuntimeError("Call simulate() first.")
        return self._x

    def get_all_parameters(self) -> np.ndarray:
        if self._theta is None:
            raise RuntimeError("Call simulate() first.")
        return self._theta

    def get_obs_data(self) -> np.ndarray:
        return self._x_obs

    def get_fid_parameters(self) -> Optional[np.ndarray]:
        return self._theta_fid
