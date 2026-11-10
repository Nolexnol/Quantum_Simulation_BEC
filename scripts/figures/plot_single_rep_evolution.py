#!/usr/bin/env python3
"""
One realization per evolution time, from 0 to 600 ms.

Picks a single repeat (the base shot) at each available evolution time in the
bec_box2 coarsening dataset and lays them out as a montage, so the demixing and
coarsening are visible in one clean row-by-row sweep -- the experimental
analogue of the simulated binary_coarsening.png.

    python scripts/figures/plot_single_rep_evolution.py
"""

from quantum_sim.config import dataset
from quantum_sim.utils.plotting import use_headless_backend

use_headless_backend()
import matplotlib.pyplot as plt  # noqa: E402 — must follow the backend choice

from quantum_sim.utils.plotting import montage  # noqa: E402
from quantum_sim.utils.shots import group_by_hold_time, one_shot_per_time  # noqa: E402

DATASET = "coarsening"
T_MAX_MS = 600  # keep evolution times in [0, T_MAX_MS]
REP_ORDER = ["base", "00001", "00002", "00003", "00004"]


def main():
    ds = dataset(DATASET)
    shots = one_shot_per_time(
        group_by_hold_time(ds, max_time_ms=T_MAX_MS), preferred=REP_ORDER
    )

    times = sorted(shots)
    print(
        f"{len(times)} evolution times (single rep) from "
        f"{times[0]} to {times[-1]} ms: {times}"
    )

    fig, _ = montage(
        [shots[t].od for t in times],
        [f"t = {t} ms" for t in times],
        suptitle=(
            "bec_box2 coarsening: single realization, 0-600 ms\n"
            "optical density vs. evolution time"
        ),
    )

    out = ds.figure("single_rep_evolution_0_600ms.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
