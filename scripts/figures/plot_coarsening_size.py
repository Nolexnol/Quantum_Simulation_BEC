#!/usr/bin/env python3
"""
Average coarsening (domain) size vs. evolution time for the bec_box2 dataset.

For every shot we measure a characteristic domain length from the first moment
of the spin structure factor -- the same estimator used for the simulation in
scripts/simulations/run_binary_coarsening.py:

    L = 2 pi <S> / <k S>,     S(k) = |FFT2(fluctuation field)|^2,  S(0) = 0

Idea: the structure factor S(k) is the power spectrum of the density-contrast
field; it tells you how much of the pattern lives at each spatial wavenumber k
(k ~ 1/length). Big domains -> power piled up at small k; fine structure ->
power at large k. The ratio <S>/<k S> is the S-weighted mean of 1/k, i.e. a
characteristic length, and the 2 pi turns it into a real-space size. As domains
coarsen, spectral weight shifts to small k and L grows.

Because the experimental cloud has a smooth envelope that would dominate the
low-k power, we first high-pass each image (subtract a broad Gaussian blur) and
apodize before the FFT.

The figure is a log-log plot: every individual shot as a faint point, the
per-time mean of the ~5 repeats as a solid marker, and a curvature-driven
L = A t^(1/3) coarsening fit drawn as the reference power law.

    python scripts/figures/plot_coarsening_size.py

Note: the estimator is shared with every other analysis in this project via
quantum_sim.utils.domains, and uses a Tukey window (alpha from
config/experiments.yaml). An earlier copy of this script used a Hann window,
which attenuates the bulk of the field and biased L low relative to the
animation and scaling analyses.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

from quantum_sim.config import dataset
from quantum_sim.utils.plotting import use_headless_backend

use_headless_backend()
import matplotlib.pyplot as plt  # noqa: E402 — must follow the backend choice

from quantum_sim.utils.domains import domain_length_um, window_2d  # noqa: E402
from quantum_sim.utils.shots import group_by_hold_time  # noqa: E402
from quantum_sim.utils.stats import cube_root  # noqa: E402

DATASET = "coarsening"
BG_SIGMA = 20.0  # px; blur scale removed to kill the cloud envelope
FIT_MIN_MS = 10  # window used to anchor the t^(1/3) amplitude
FIT_MAX_MS = 100  # (the domain-growth regime before saturation)


def domain_size_um(od, px_um: float, window) -> float:
    """High-pass the envelope out, then take the structure-factor first moment."""
    fluctuation = od - gaussian_filter(od, sigma=BG_SIGMA)
    return domain_length_um(fluctuation, px_um, window)


def main():
    ds = dataset(DATASET)
    by_time = group_by_hold_time(ds, keep_repeats=False)
    times = sorted(by_time)
    print(
        f"{sum(len(v) for v in by_time.values())} shots across "
        f"{len(times)} evolution times: {times[0]}..{times[-1]} ms"
    )

    window = None
    shot_t, shot_L = [], []
    mean_t, mean_L, sem_L = [], [], []

    for t_ms in times:
        sizes = []
        for shot in by_time[t_ms]:
            od = shot.od
            if window is None or window.shape != od.shape:
                window = window_2d(od.shape, ds.calibration.window_alpha)
            sizes.append(domain_size_um(od, ds.pixel_um, window))
        sizes = np.asarray(sizes, dtype=float)

        shot_t.extend([t_ms] * len(sizes))
        shot_L.extend(sizes)
        mean_t.append(t_ms)
        mean_L.append(sizes.mean())
        sem_L.append(sizes.std(ddof=1) / np.sqrt(len(sizes)) if len(sizes) > 1 else 0.0)
        print(
            f"  t = {t_ms:4.0f} ms  n = {len(sizes)}  "
            f"L = {sizes.mean():5.2f} +/- {sem_L[-1]:.2f} um"
        )

    shot_t = np.asarray(shot_t, dtype=float)
    shot_L = np.asarray(shot_L, dtype=float)
    mean_t = np.asarray(mean_t, dtype=float)
    mean_L = np.asarray(mean_L, dtype=float)
    sem_L = np.asarray(sem_L, dtype=float)

    # L = A t^(1/3): fit the amplitude over the growth window, before saturation.
    fit_mask = (mean_t >= FIT_MIN_MS) & (mean_t <= FIT_MAX_MS)
    (amplitude,), _ = curve_fit(cube_root, mean_t[fit_mask], mean_L[fit_mask], p0=(4.0,))
    residual = mean_L[fit_mask] - cube_root(mean_t[fit_mask], amplitude)
    r_squared = 1 - np.sum(residual**2) / np.sum(
        (mean_L[fit_mask] - mean_L[fit_mask].mean()) ** 2
    )
    print(f"\ncube-root fit over {FIT_MIN_MS}-{FIT_MAX_MS} ms:")
    print(f"  L(t) = {amplitude:.3f} * t^(1/3)  (um, t in ms)   R^2 = {r_squared:.3f}")

    # -- log-log figure ----------------------------------------------------- #
    positive = shot_t > 0
    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.plot(
        shot_t[positive], shot_L[positive], "o", ms=3.5, color="C0", alpha=0.30,
        label="individual shots", zorder=2,
    )
    ax.errorbar(
        mean_t[mean_t > 0], mean_L[mean_t > 0], yerr=sem_L[mean_t > 0],
        fmt="o", ms=6, color="C0", capsize=3, lw=1.1,
        label="mean of repeats", zorder=3,
    )

    tt = np.linspace(mean_t[mean_t > 0].min(), mean_t.max(), 400)
    ax.plot(
        tt, cube_root(tt, amplitude), "--", color="0.25", lw=1.8,
        label=(
            rf"$L = {amplitude:.1f}\,t^{{1/3}}$   ($R^2$={r_squared:.2f} over "
            rf"{FIT_MIN_MS}-{FIT_MAX_MS} ms)"
        ),
        zorder=4,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("evolution time  t  (ms)")
    ax.set_ylabel(r"domain size  L  ($\mu$m)")
    ax.set_title("bec_box2 domain coarsening (structure-factor size)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()

    out = ds.figure("coarsening_size.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
