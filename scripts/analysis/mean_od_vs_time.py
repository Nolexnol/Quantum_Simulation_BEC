#!/usr/bin/env python3
"""
Mean OD vs evolution time: the simplest proxy for atom number.

For every shot, take the mean of the full-frame OD (proportional to the number
of atoms in the picture) minus the frame's median OD, which removes the
background offset that drifts from run to run. Average the repeats at each time and plot one graph
per dataset. See atom_number.py for a calibrated atom count.

Figures: results/figures/mean_od_vs_time_<dataset>.png
"""

import h5py
import numpy as np

from quantum_sim.config import PATHS, dataset
from quantum_sim.utils.plotting import use_headless_backend

use_headless_backend()
import matplotlib.pyplot as plt  # noqa: E402 — must follow the backend choice

from quantum_sim.io import ShotFile  # noqa: E402

DATASETS = [("shots700p", "700p shots (2026-08-19)"),
            ("sep11", "2026-09-11 shots")]


def main():
    for key, label in DATASETS:
        ds = dataset(key)
        t_ms, mean_od, skipped = [], [], 0
        for path in ds.shot_paths():
            with h5py.File(path, "r") as f:
                if "images" not in f:          # shot aborted before imaging
                    skipped += 1
                    continue
            shot = ShotFile(path, cam=ds.camera)
            t_ms.append(round(float(shot.BEC_TIME_SEQ[ds.time_index]) * 1e3, 3))
            od = shot.od
            mean_od.append(od.mean() - np.median(od))   # median = background: atoms fill ~10% of frame
        t_ms, mean_od = np.array(t_ms), np.array(mean_od)

        times = np.unique(t_ms)
        avg = np.array([mean_od[t_ms == t].mean() for t in times])
        sem = np.array([mean_od[t_ms == t].std(ddof=1) / np.sqrt((t_ms == t).sum())
                        for t in times])

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(times, avg, yerr=sem, fmt="o", capsize=2)
        ax.set_xlabel("evolution time (ms)")
        ax.set_ylabel("mean OD")
        ax.set_title(label)
        fig.tight_layout()
        out = PATHS.figure(f"mean_od_vs_time_{key}.png")
        fig.savefig(out, dpi=130)
        print(f"{label}: {len(t_ms)} shots ({skipped} without images), mean OD {avg[0]:.4f} at t=0 -> "
              f"{avg[-1]:.4f} at {times[-1]:g} ms  |  wrote {out}")


if __name__ == "__main__":
    main()
