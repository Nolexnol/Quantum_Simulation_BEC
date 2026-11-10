"""
Operations on optical-density frames: framing, rotation, registration.

These functions take and return plain arrays. They know nothing about HDF5 or
matplotlib, which keeps them trivially testable and lets ``io`` and ``viz``
both depend on them without a cycle.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, rotate

__all__ = [
    "rotate_and_crop",
    "threshold_centroid",
    "centroid_crop_box",
    "register_shifts",
    "optical_density",
]

#: Largest lag :func:`register_shifts` is allowed to return [px].
MAX_SHIFT_PX = 25


def optical_density(atoms, probe, background, floor: float = 1e-6) -> np.ndarray:
    """
    Absorption-imaging optical density from the three raw frames.

    ``OD = -ln[(atoms - bg) / (probe - bg)]``, with both differences floored at
    ``floor`` so read noise on a dark pixel cannot produce a negative argument
    to the logarithm.
    """
    atoms = np.asarray(atoms, dtype=float)
    probe = np.asarray(probe, dtype=float)
    background = np.asarray(background, dtype=float)
    signal = np.maximum(atoms - background, floor)
    reference = np.maximum(probe - background, floor)
    return -np.log(signal / reference)


def rotate_and_crop(od, angle: float = 45.0) -> np.ndarray:
    """
    Rotate a frame and cut the inscribed square, so no zero padding remains.

    The box sits diamond-wise in the raw camera frame. Rotating it upright
    introduces triangular zero-filled corners, and those corners are a hard
    edge that would dominate any Fourier-domain estimate of the domain size.
    Cropping to the largest square that fits inside the rotated frame removes
    them entirely.
    """
    rotated = rotate(np.asarray(od, dtype=float), angle, reshape=False)
    c0, c1 = rotated.shape[0] // 2, rotated.shape[1] // 2
    half = int(min(rotated.shape) / (2 * np.sqrt(2)))
    return rotated[c0 - half : c0 + half, c1 - half : c1 + half]


def threshold_centroid(img, sigma: float = 5.0, top_frac: float = 0.10):
    """
    Centroid of the brightest ``top_frac`` of a blurred frame, as ``(row, col)``.

    Blurring first stops single hot pixels from pulling the answer.
    """
    blurred = gaussian_filter(np.asarray(img, dtype=float), sigma=sigma)
    threshold = np.sort(blurred.ravel())[-int(blurred.size * top_frac)]
    weight = np.where(blurred < threshold, 0.0, blurred)
    rows, cols = weight.sum(axis=1), weight.sum(axis=0)
    return (
        (np.arange(weight.shape[0]) * rows).sum() / rows.sum(),
        (np.arange(weight.shape[1]) * cols).sum() / cols.sum(),
    )


def centroid_crop_box(frames, crop_half: int) -> list[int]:
    """
    Square crop box centred on the centroid of the mean of ``frames``.

    Used for runs where the cloud sits at a slightly different place each day,
    so a hard-coded rectangle would clip it. Returns ``[r0, r1, c0, c1]``.
    """
    mean_od = np.mean(np.asarray(frames, dtype=float), axis=0)
    row, col = (int(v) for v in threshold_centroid(mean_od))
    return [row - crop_half, row + crop_half, col - crop_half, col + crop_half]


def register_shifts(stack, max_shift_px: int = MAX_SHIFT_PX) -> np.ndarray:
    """
    Integer-pixel ``(dy, dx)`` of each frame against the stack mean.

    A thresholded centroid is the obvious way to ask "did the cloud move", but
    it does not work here: the crop keeps the bright box walls, the top-10%
    mask latches onto them, and the answer ends up tracking wall shot noise
    (~10 px of jitter at t = 0, where nothing has moved at all). The walls are
    the sharpest feature in the frame, so FFT cross-correlation of the whole
    frame against the time mean uses them as the fiducial instead of being
    confused by them.

    The lag search is capped at ``max_shift_px``. The box is a near-square of
    roughly uniform walls, so its autocorrelation has broad shoulders: on a
    low-contrast shot a lag of 45-67 px can beat zero lag by ~0.04 in
    normalized correlation purely by accident, which would be reported as a
    huge displacement that never happened. Anything genuinely shifted further
    than the cap is a gross fault that the correlation score flags instead.
    """
    stack = np.asarray(stack, dtype=float)
    reference = stack.mean(axis=0)
    reference_fft_conj = np.conj(np.fft.fft2(reference - reference.mean()))

    r = max_shift_px
    shifts = np.empty((len(stack), 2))
    for i, frame in enumerate(stack):
        correlation = np.fft.fftshift(
            np.fft.ifft2(np.fft.fft2(frame - frame.mean()) * reference_fft_conj).real
        )
        c0, c1 = correlation.shape[0] // 2, correlation.shape[1] // 2
        window = correlation[c0 - r : c0 + r + 1, c1 - r : c1 + r + 1]
        dy, dx = np.unravel_index(np.argmax(window), window.shape)
        shifts[i] = (dy - r, dx - r)
    return shifts
