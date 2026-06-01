"""
Summary statistics for the coincidence TS distribution.

Compresses per-source coincidence TS values into a fixed-dim vector x
for the population-level NRE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


# chi2(2) Wilks' theorem approximation
_TS_THRESHOLDS = {
    "1sigma": 1.0,
    "2sigma": 4.0,
    "3sigma": 9.0,
    "5sigma": 25.0,
}

_TS_BINS = np.array([-50, -20, -10, -5, 0, 2, 5, 10, 20, 50, 200], dtype=np.float32)


@dataclass
class CoincidenceSummary:
    """
    Fixed-dim summary statistic x from a set of per-source TS values.

    x = concat([ts_histogram (10 bins), aggregate stats (5 values)]), total dim 15.
    """
    ts_bins: np.ndarray = None

    def __post_init__(self):
        if self.ts_bins is None:
            self.ts_bins = _TS_BINS

    @property
    def d_x(self) -> int:
        return len(self.ts_bins) - 1 + 5

    def compute(self, ts_values: np.ndarray) -> np.ndarray:
        """compress a set of per-source TS values to a summary vector x."""
        ts = np.asarray(ts_values, dtype=np.float32)
        N = len(ts) if len(ts) > 0 else 1

        hist, _ = np.histogram(ts, bins=self.ts_bins)
        hist_norm = hist.astype(np.float32) / N

        n_above_3sig  = float((ts > _TS_THRESHOLDS["3sigma"]).sum()) / N
        n_above_5sig  = float((ts > _TS_THRESHOLDS["5sigma"]).sum()) / N
        ts_max        = float(ts.max()) if len(ts) > 0 else 0.0
        ts_mean       = float(ts.mean()) if len(ts) > 0 else 0.0
        ts_median     = float(np.median(ts)) if len(ts) > 0 else 0.0

        agg = np.array([n_above_3sig, n_above_5sig,
                        ts_max, ts_mean, ts_median], dtype=np.float32)

        return np.concatenate([hist_norm, agg])

    def compute_batch(self, ts_list: List[np.ndarray]) -> np.ndarray:
        """compress multiple simulations, returns (N_sims, d_x)."""
        return np.stack([self.compute(ts) for ts in ts_list], axis=0)

    def feature_names(self) -> List[str]:
        edges = self.ts_bins
        names = [f"ts_bin_{edges[i]:.0f}_{edges[i+1]:.0f}"
                 for i in range(len(edges)-1)]
        names += ["n_3sig", "n_5sig", "ts_max", "ts_mean", "ts_median"]
        return names
