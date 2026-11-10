"""Robust statistics used for shot quality control."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["modified_z", "cube_root"]

#: Modified z-score above which a shot is treated as an outlier.
Z_THRESHOLD = 3.5


def modified_z(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Modified z-score (Iglewicz-Hoaglin), used to flag outlier shots.

    Built on the median and the median absolute deviation rather than the mean
    and standard deviation, because a single catastrophic shot inflates an SD
    enough to hide itself. Falls back to the mean absolute deviation when the
    MAD is degenerate (more than half the values identical), and returns zeros
    when every value is identical — the honest answer being that nothing
    stands out.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr

    median = np.median(arr)
    mad = np.median(np.abs(arr - median))
    if mad > 0:
        return 0.6745 * (arr - median) / mad

    mean_ad = np.mean(np.abs(arr - median))
    if mean_ad == 0:
        return np.zeros_like(arr)
    return (arr - median) / (1.253314 * mean_ad)


def cube_root(t, amplitude: float):
    """Curvature-driven coarsening law ``L = A t^(1/3)``, for curve fitting."""
    return amplitude * np.cbrt(t)
