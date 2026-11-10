#!/usr/bin/env python3
"""
Atom number vs hold time, 700p (2026-08-19) against 2026-09-11.

The cloud is a flat-top box, not a Gaussian, so N is counted directly rather
than from a Gaussian fit:

    N = (A_px / sigma) * sum over the box of (OD - background)

  * box        the diamond |dr| + |dc| <= R around the t = 0 cloud centroid,
               R = half-diagonal of the measured box + MARGIN_PX
  * background median OD of a diamond ring just outside the box, per shot
  * sigma      resonant cross-section 3 lambda^2 / 2 pi of the Na D2 line

The two runs load different atom numbers, so the figure shows the fraction
left, N(t) / N(0), which is what compares their loss. Absolute N per time is
in the CSV.

Figure: results/figures/atom_number_vs_time.png
Table:  results/derived/atom_number.csv
"""

import csv

import h5py
import numpy as np

from quantum_sim.config import PATHS, apparatus, dataset
from quantum_sim.utils.plotting import use_headless_backend

use_headless_backend()
import matplotlib.pyplot as plt  # noqa: E402 — must follow the backend choice

from quantum_sim.io import ShotFile  # noqa: E402
from quantum_sim.utils.images import threshold_centroid  # noqa: E402

DATASETS = [("shots700p", "700p shots (2026-08-19)", "C0", "o"),
            ("sep11", "2026-09-11 shots", "C1", "s")]

LAMBDA_NA_D2 = 589.158e-9                          # [m]
SIGMA_0 = 3 * LAMBDA_NA_D2**2 / (2 * np.pi)        # resonant cross-section [m^2]
MARGIN_PX = 12      # box mask grows by this beyond the measured wall [px]
RING_PX = 40        # width of the background ring outside the mask [px]


def usable_shots(ds):
    """(path, t_ms) for every shot of the dataset that has camera frames."""
    out = []
    for path in ds.shot_paths():
        with h5py.File(path, "r") as f:
            if "images" not in f:
                continue
            t_ms = round(float(np.array(f["globals"].attrs["BEC_TIME_SEQ"])[ds.time_index])
                         * 1e3, 3)
        out.append((path, t_ms))
    return out


def get_ods(ds, shots):
    """Retrieve the optical densities for all shot samples."""
    ods = [ShotFile(p, cam=ds.camera).od for p, _ in shots]
    t0 = [od for od, (_, t) in zip(ods, shots) if t == 0.0]
    row, col = threshold_centroid(np.mean(t0, axis=0))

    # half_diag = apparatus().box_side_um / ds.pixel_um / np.sqrt(2)
    # rr, cc = np.indices(ods[0].shape)
    # l1 = np.abs(rr - row) + np.abs(cc - col)       # diamond distance: the box is rotated 45 deg
    # inside = l1 <= half_diag + MARGIN_PX
    # ring = (l1 > half_diag + MARGIN_PX + 5) & (l1 <= half_diag + MARGIN_PX + 5 + RING_PX)

    px_area = (ds.pixel_um * 1e-6) ** 2
    return row, col


def main():
    fig, ax = plt.subplots(figsize=(8, 5))
    rows = []
    for key, label, color, marker in DATASETS:
        ds = dataset(key)
        shots = usable_shots(ds)
        t_shots = np.array([t for _, t in shots])
        n = count_atoms(ds, shots)

        times = np.array(sorted(set(t_shots)))
        count = np.array([(t_shots == t).sum() for t in times])
        mean = np.array([n[t_shots == t].mean() for t in times])
        std = np.array([n[t_shots == t].std(ddof=1) for t in times])
        frac = mean / mean[0]
        frac_sem = std / np.sqrt(count) / mean[0]

        ax.errorbar(times, frac, yerr=frac_sem, fmt=f"{marker}-", ms=5, lw=1.2,
                    color=color, capsize=2, label=label)

        print(f"{label}: {len(shots)} shots, N(0) = {mean[0]:.3g}, "
              f"N(2000 ms)/N(0) = {frac[-1]:.2f}")
        rows += [[key, f"{t:g}", count[i], f"{mean[i]:.5g}", f"{std[i]:.4g}",
                  f"{frac[i]:.4f}", f"{frac_sem[i]:.4f}"] for i, t in enumerate(times)]

    ax.axhline(1, color="0.6", lw=0.8)
    ax.set_xlabel("evolution time  t  (ms)")
    ax.set_ylabel(r"fraction of atoms left  $N(t)\,/\,N(0)$")
    ax.set_title("Atom loss vs evolution time")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = PATHS.figure("atom_number_vs_time.png")
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")

    csv_path = PATHS.derived_file("atom_number.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "t_ms", "n_shots", "N_mean", "N_std", "N_over_N0",
                    "N_over_N0_sem"])
        w.writerows(rows)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
