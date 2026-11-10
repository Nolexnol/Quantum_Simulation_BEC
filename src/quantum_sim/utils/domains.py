"""
Domain-size and coarsening statistics for a 2D field.

There were three different spellings of "domain length" in the old scripts and
they did not agree: one applied a Hann window, one a Tukey window, one applied
no window at all because the line had been commented out. Two figures in the
same study could therefore report different lengths for the same shot. This
module is the single definition, with the window made an explicit argument so
a deviation has to be written down rather than drifting in silently.

Convention throughout: the first moment of the structure factor,

    L = 2 pi <S> / <k S>,

with the uniform (k = 0) mode dropped.
"""

from __future__ import annotations

import numpy as np
from scipy.signal.windows import tukey

__all__ = [
    "window_2d",
    "structure_factor",
    "domain_length_px",
    "domain_length_um",
    "radial_autocorr",
    "first_zero",
    "radial_spectrum",
    "local_exponents",
]


def window_2d(shape, alpha: float = 0.5, kind: str = "tukey") -> np.ndarray:
    """
    Separable 2D apodisation window.

    A field measured in a finite box is not periodic, so an un-windowed FFT
    sees a step discontinuity at the wrap-around and spreads power to high k,
    biasing ``L`` downwards. ``kind="tukey"`` with ``alpha=0.5`` is the
    project default: it tapers the outer half while leaving the interior
    untouched, unlike a Hann window, which attenuates the bulk of the field
    that carries the signal.
    """
    ny, nx = shape
    if kind == "tukey":
        return np.outer(tukey(ny, alpha), tukey(nx, alpha))
    if kind == "hann":
        return np.outer(np.hanning(ny), np.hanning(nx))
    if kind == "none":
        return np.ones((ny, nx))
    raise ValueError(f"Unknown window kind {kind!r}; use 'tukey', 'hann' or 'none'.")


def structure_factor(field) -> np.ndarray:
    """Power spectrum ``|FFT(field)|^2`` with the uniform mode zeroed."""
    spectrum = np.abs(np.fft.fft2(np.asarray(field, dtype=float))) ** 2
    spectrum[0, 0] = 0.0
    return spectrum


def _k_magnitude(shape, px_um: float = 1.0) -> np.ndarray:
    """Wavevector magnitude grid [rad/px, or rad/um when px_um is given]."""
    ny, nx = shape
    ky = 2 * np.pi * np.fft.fftfreq(ny, px_um)
    kx = 2 * np.pi * np.fft.fftfreq(nx, px_um)
    kxx, kyy = np.meshgrid(kx, ky, indexing="xy")
    return np.sqrt(kxx**2 + kyy**2)


def domain_length_px(field, window=None) -> float:
    """
    Characteristic domain length in pixels: ``L = 2 pi <S> / <k S>``.

    The field is de-meaned first, so the result measures fluctuations rather
    than the cloud's overall brightness. Pass ``window`` (see :func:`window_2d`)
    to apodise; ``None`` means no apodisation.
    """
    fluct = np.asarray(field, dtype=float)
    fluct = fluct - fluct.mean()
    if window is not None:
        fluct = fluct * window

    spectrum = structure_factor(fluct)
    k_mag = _k_magnitude(fluct.shape)
    return float(2 * np.pi * spectrum.sum() / (k_mag * spectrum).sum())


def domain_length_um(field, px_um: float, window=None) -> float:
    """:func:`domain_length_px` converted to micrometres."""
    return domain_length_px(field, window=window) * px_um


def radial_autocorr(field, px_um: float):
    """
    Azimuthally averaged aperiodic autocorrelation, normalised to ``g(0) = 1``.

    Zero-padding to twice the size makes the correlation aperiodic; dividing by
    the autocorrelation of the support mask removes the triangular overlap
    weighting, giving an unbiased estimate at every lag.

    Returns ``(r [um], g(r))``.
    """
    f = np.asarray(field, dtype=float)
    f = f - f.mean()
    ny, nx = f.shape

    padded = np.zeros((2 * ny, 2 * nx))
    padded[:ny, :nx] = f
    mask = np.zeros_like(padded)
    mask[:ny, :nx] = 1.0

    corr = np.fft.fftshift(np.fft.ifft2(np.abs(np.fft.fft2(padded)) ** 2).real)
    norm = np.fft.fftshift(np.fft.ifft2(np.abs(np.fft.fft2(mask)) ** 2).real)
    corr = corr / np.maximum(norm, 1.0)

    cy, cx = ny, nx  # zero lag after fftshift
    yy, xx = np.indices(corr.shape)
    r_px = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    r_max = min(ny, nx) - 1
    idx = np.clip(r_px.astype(int), 0, r_max)
    sel = r_px <= r_max
    g = np.bincount(idx[sel], weights=corr[sel], minlength=r_max + 1) / np.bincount(
        idx[sel], minlength=r_max + 1
    )
    return np.arange(r_max + 1) * px_um, g / g[0]


def first_zero(r, g) -> float:
    """
    Position of the first zero crossing of ``g(r)``, linearly interpolated.

    This is the length scale a coarsening study reports: the lag at which the
    field decorrelates. Returns NaN when ``g`` never crosses zero.
    """
    r = np.asarray(r, dtype=float)
    g = np.asarray(g, dtype=float)
    for i in range(1, len(g)):
        if g[i - 1] > 0 >= g[i]:
            return float(r[i - 1] + (r[i] - r[i - 1]) * g[i - 1] / (g[i - 1] - g[i]))
    return float("nan")


def radial_spectrum(field, px_um: float, alpha: float = 0.5):
    """
    Azimuthally averaged power spectrum of the de-meaned, Tukey-windowed field.

    Returns ``(q [1/um], S(q), bin counts)``. The counts are returned so a
    caller can drop under-populated high-q bins instead of plotting their noise.
    """
    f = np.asarray(field, dtype=float)
    ny, nx = f.shape
    f = (f - f.mean()) * window_2d(f.shape, alpha)

    power = np.abs(np.fft.fft2(f)) ** 2
    q_mag = _k_magnitude(f.shape, px_um)

    dq = 2 * np.pi / (min(ny, nx) * px_um)
    idx = np.round(q_mag / dq).astype(int)
    n_bins = idx.max() + 1

    counts = np.bincount(idx.ravel(), minlength=n_bins)
    spectrum = np.bincount(idx.ravel(), weights=power.ravel(), minlength=n_bins) / np.maximum(
        counts, 1
    )
    return np.arange(n_bins) * dq, spectrum, counts


def local_exponents(times, lengths, length_err=None, t_min: float = 30.0):
    """
    Local growth exponent ``1/z`` from pairs of times about a factor of two apart.

    Fitting a single power law across the whole series assumes the system is
    already in the scaling regime; comparing nearby pairs instead exposes
    whether ``1/z`` is still drifting, which is the actual question in a
    coarsening measurement. Pairs are restricted to ``t' / t`` in [1.6, 2.6]
    and to ``t >= t_min``, before which domains have not formed.

    Returns ``(t_mid, L_mid, one_over_z, err)`` as arrays.
    """
    times = np.asarray(times, dtype=float)
    lengths = np.asarray(lengths, dtype=float)

    out = []
    for i, t1 in enumerate(times):
        if t1 < t_min:
            continue
        ratios = times / t1
        candidates = np.where((ratios >= 1.6) & (ratios <= 2.6))[0]
        if len(candidates) == 0:
            continue

        j = candidates[np.argmin(np.abs(ratios[candidates] - 2.0))]
        t2 = times[j]
        if np.isnan(lengths[i]) or np.isnan(lengths[j]):
            continue

        z_inv = np.log(lengths[j] / lengths[i]) / np.log(t2 / t1)
        err = np.nan
        if length_err is not None:
            length_err = np.asarray(length_err, dtype=float)
            err = np.sqrt(
                (length_err[i] / lengths[i]) ** 2 + (length_err[j] / lengths[j]) ** 2
            ) / np.log(t2 / t1)

        out.append((np.sqrt(t1 * t2), np.sqrt(lengths[i] * lengths[j]), z_inv, err))

    if not out:
        empty = np.array([])
        return empty, empty, empty, empty
    return tuple(np.array(col) for col in zip(*out))
