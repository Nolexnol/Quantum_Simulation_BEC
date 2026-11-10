#!/usr/bin/env python3
"""
Every nonlinear time-evolution shot in the dataset, plotted together.

The bec_box2 coarsening experiment was recorded across three sequences that
sample complementary evolution times (0022: 0-800 ms coarse, 0023: 10-40 ms
fine, 0024: 62-175 ms fill-in). Together they give 26 distinct evolution
times, each repeated ~5 times (base + rep00001..rep00004).

This lays out the full dataset as a grid: one row per evolution time (labelled
in ms), one column per repeat, each cell the optical density (OD) of that shot.

    python scripts/figures/plot_all_evolution.py
"""

import numpy as np

from quantum_sim.config import dataset
from quantum_sim.utils.plotting import use_headless_backend

use_headless_backend()
import matplotlib.pyplot as plt  # noqa: E402 — must follow the backend choice

from quantum_sim.utils.shots import group_by_hold_time  # noqa: E402

DATASET = "coarsening"
REP_ORDER = ["base", "00001", "00002", "00003", "00004"]


def select_times(available):
    """Everything up to 200 ms, then 200->400 by 100 ms, then 400->800 by 200 ms."""
    chosen = {t for t in available if t <= 200}
    chosen |= {t for t in available if t in range(200, 401, 100)}  # 200, 300, 400
    chosen |= {t for t in available if t in range(400, 801, 200)}  # 400, 600, 800
    return sorted(chosen)


def main():
    ds = dataset(DATASET)
    by_time = group_by_hold_time(ds)

    times = select_times(sorted(by_time))
    by_time = {t: by_time[t] for t in times}

    n_shots = sum(len(reps) for reps in by_time.values())
    print(
        f"{n_shots} shots across {len(times)} selected evolution times "
        f"({times[0]}..{times[-1]} ms): {times}"
    )

    # One colour scale across every panel, so brightness differences between
    # cells are physical rather than an artefact of per-panel autoscaling.
    all_od = [shot.od for reps in by_time.values() for shot in reps.values()]
    vmin, vmax = 0.0, np.percentile(np.concatenate([od.ravel() for od in all_od]), 99.5)

    nrows, ncols = len(times), len(REP_ORDER)
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.55 * ncols, 1.5 * nrows))
    axes = np.atleast_2d(axes)

    im = None
    for row, t_ms in enumerate(times):
        reps = by_time[t_ms]
        for col, rep_label in enumerate(REP_ORDER):
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])

            if rep_label in reps:
                im = ax.imshow(
                    reps[rep_label].od,
                    origin="lower",
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                )
            else:
                ax.axis("off")  # missing repeat (e.g. 750 ms rep1)

            if col == 0:  # row label = evolution time
                ax.set_ylabel(
                    f"{t_ms} ms", fontsize=10, rotation=0, ha="right",
                    va="center", labelpad=22,
                )
            if row == 0:  # column label = repeat number
                ax.set_title(f"rep {col + 1}", fontsize=10)

    fig.suptitle(
        "bec_box2 coarsening: nonlinear time evolution (all repeats)\n"
        "optical density vs. evolution time",
        fontsize=14,
        y=0.997,
    )
    fig.tight_layout(rect=[0.02, 0, 0.94, 0.985])
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.015)
    cbar.set_label("Optical density")

    out = ds.figure("time_evolution_selected.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
