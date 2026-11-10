#!/usr/bin/env python3
"""
Montage of one coarsening pattern from the bec_box2 experiment.

Each pattern is a single sequence (fixed shot index) swept over 17 evolution
times, encoded in the global BEC_TIME_SEQ[6] (seconds). We show the optical
density (OD) of one realization at each time, laid out as a grid with the
evolution time (in ms) in each panel title -- the experimental analogue of the
simulated binary_coarsening.png.

    python scripts/figures/plot_coarsening_pattern.py [--sequence 2026-07-17_0022]
"""

import argparse

import numpy as np

from quantum_sim.config import dataset
from quantum_sim.utils.plotting import use_headless_backend

use_headless_backend()
import matplotlib.pyplot as plt  # noqa: E402 — must follow the backend choice

from quantum_sim.io import ShotFile  # noqa: E402
from quantum_sim.utils.plotting import montage  # noqa: E402
from quantum_sim.utils.shots import hold_time_ms  # noqa: E402

DATASET = "coarsening"
SEQUENCE = "2026-07-17_0022"  # one repeated pattern
N_INDICES = 17


def load_pattern(ds, sequence: str):
    """Return ``(times_ms, od_images)`` sorted by evolution time for one pattern."""
    shots = []
    for index in range(N_INDICES):
        path = ds.path / f"{sequence}_bec_box2_{index:02d}.h5"
        if not path.exists():
            print(f"  warning: missing {path.name}, skipping")
            continue
        shots.append(ShotFile(path, cam=ds.camera, crop_box=ds.crop_box))

    if not shots:
        raise SystemExit(
            f"No shots found for sequence {sequence!r} in {ds.path}. "
            f"Available datasets are declared in config/experiments.yaml."
        )

    print("swept evolution times [s]:", np.unique([s.BEC_TIME_SEQ[ds.time_index] for s in shots]))

    entries = sorted(
        ((hold_time_ms(s, ds.time_index), s.od) for s in shots), key=lambda e: e[0]
    )
    return [t for t, _ in entries], [od for _, od in entries]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default=SEQUENCE, help="sequence prefix to plot")
    args = parser.parse_args()

    ds = dataset(DATASET)
    times, images = load_pattern(ds, args.sequence)
    print(
        f"Loaded {len(images)} shots for pattern {args.sequence}: "
        f"t = {times[0]:.0f}..{times[-1]:.0f} ms"
    )

    fig, _ = montage(
        images,
        [f"t = {t:.0f} ms" for t in times],
        suptitle=(
            f"Coarsening pattern {args.sequence} (bec_box2): "
            "optical density vs. evolution time"
        ),
    )

    out = ds.figure("coarsening_pattern.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
