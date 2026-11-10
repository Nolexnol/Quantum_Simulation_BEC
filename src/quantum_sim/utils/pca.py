"""
Principal-component decomposition of a stack of shots.

Three animation scripts carried byte-identical copies of everything here.
"""

from __future__ import annotations

import numpy as np

__all__ = ["pca_svd", "align_signs", "remove_components"]


def pca_svd(images):
    """
    Normalised eigenvalues and spatial PCA vectors for ``images[shot, x, y]``.

    The SVD is taken of the pixel-by-shot matrix, so the returned vectors are
    spatial modes shaped ``(x, y, n_components)`` and the eigenvalues are
    ``S**2`` normalised to sum to one — i.e. the fraction of total variance
    each mode explains.
    """
    images = np.asarray(images, dtype=float)
    if images.ndim != 3:
        raise ValueError("images must have shape [shot, x, y]")

    pixels_by_shot = images.reshape(images.shape[0], -1).T
    vectors, singular_values, _ = np.linalg.svd(pixels_by_shot, full_matrices=False)

    eigenvalues = singular_values**2
    total = eigenvalues.sum()
    if total == 0:
        raise ValueError("images must have non-zero variance")
    eigenvalues /= total

    return eigenvalues, vectors.reshape(*images.shape[1:], -1)


def _seed_sign(vec, uniform_min: float = 0.1) -> float:
    """
    Sign convention for the first frame of a component's time series.

    An SVD fixes each mode only up to an overall sign, so an animation of the
    same physical mode can flip arbitrarily from frame to frame. Seeding from
    the mode's own asymmetry — positive lobe heavier than negative — gives a
    deterministic starting orientation.
    """
    vec = np.asarray(vec, dtype=float)
    positive = vec[vec > 0].sum()
    negative = -vec[vec < 0].sum()
    if abs(positive - negative) < uniform_min * max(positive, negative, 1e-30):
        return 1.0
    return 1.0 if positive >= negative else -1.0


def align_signs(frames, chain_min: float = 0.05):
    """
    Remove the arbitrary per-frame sign of a PCA mode across a time series.

    Each frame is flipped to correlate positively with the previous one, so a
    mode that is physically continuous in time looks continuous in the movie.
    When consecutive frames are nearly uncorrelated (``|corr| < chain_min``)
    the chain carries no information, so the frame falls back to its own
    asymmetry seed rather than inheriting a meaningless sign.
    """
    frames = [np.asarray(f, dtype=float) for f in frames]
    if not frames:
        return []

    aligned = [frames[0] * _seed_sign(frames[0])]
    for frame in frames[1:]:
        previous = aligned[-1]
        denominator = np.linalg.norm(previous) * np.linalg.norm(frame)
        correlation = float(np.dot(previous.ravel(), frame.ravel()) / denominator) if denominator else 0.0
        if abs(correlation) < chain_min:
            aligned.append(frame * _seed_sign(frame))
        else:
            aligned.append(frame if correlation >= 0 else -frame)
    return aligned


def remove_components(image, eigenvectors, n_components: int):
    """
    Project the leading ``n_components`` out of ``image``.

    Returns ``(cleaned, removed)``. Used to strip common-mode imaging
    artefacts — probe fringes, box-wall structure — before measuring the
    domain field, so the measurement sees fluctuations rather than the
    apparatus.
    """
    image = np.asarray(image, dtype=float)
    eigenvectors = np.asarray(eigenvectors, dtype=float)

    if image.ndim != 2 or eigenvectors.shape[:2] != image.shape:
        raise ValueError("image and eigenvectors have incompatible shapes")
    if not 0 <= n_components <= eigenvectors.shape[-1]:
        raise ValueError("n_components is outside the available basis")

    basis = eigenvectors[..., :n_components].reshape(image.size, n_components)
    removed = (basis @ (basis.T @ image.ravel())).reshape(image.shape)
    return image - removed, removed
