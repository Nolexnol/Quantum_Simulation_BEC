#!/usr/bin/env python3
"""
Animate the per-time principal components of a coarsening dataset.

At each evolution time the shots taken at that time are decomposed by PCA on
their own, so each frame answers "what does the shot-to-shot variation look
like *here*" rather than being projected onto a basis fitted elsewhere. One GIF
per leading component, plus main_image_evolution.gif showing the raw optical
density of one shot (same repeat slot) per time.

The eigenvector sign is a gauge freedom of the SVD, so the frames are
sign-aligned along the time chain; only the pattern is physical.

    python scripts/figures/animate_pc_evolution.py                    # coarsening
    python scripts/figures/animate_pc_evolution.py -d shots700p       # 700-shot run
    python scripts/figures/animate_pc_evolution.py -d shots700p --n-pc 6

This replaces the former animate_pc_evolution.py / animate_pc_evolution_700p.py
pair, which were identical from the PCA step onward and differed only in how
their dataset was framed -- something config/experiments.yaml now records.
"""

import argparse

import h5py
import numpy as np

from quantum_sim.config import dataset, dataset_names
from quantum_sim.utils.plotting import use_headless_backend

use_headless_backend()

from quantum_sim.io import ShotFile  # noqa: E402
from quantum_sim.utils.images import centroid_crop_box, rotate_and_crop  # noqa: E402
from quantum_sim.utils.pca import align_signs, pca_svd  # noqa: E402
from quantum_sim.utils.plotting import write_movie  # noqa: E402
from quantum_sim.utils.shots import hold_time_ms, repeat_label  # noqa: E402

REP_ORDER = ["base", "00001", "00002", "00003", "00004"]


def resolve_crop_box(ds, shots):
    """
    Crop box for this dataset: the fixed rectangle, or one found from the cloud.

    Runs where the cloud sits in a stable place declare ``crop_box`` in the
    config; runs where it drifts declare ``crop_half`` instead and get a square
    centred on the t = 0 centroid.
    """
    if ds.crop_box:
        return list(ds.crop_box)
    if ds.crop_half is None:
        raise SystemExit(
            f"Dataset '{ds.name}' declares neither crop_box nor crop_half "
            "in config/experiments.yaml."
        )

    t0_paths = [path for path, t_ms in shots if t_ms == 0.0]
    if not t0_paths:
        raise SystemExit("no t = 0 shots to locate the cloud with")

    frames = [ShotFile(p, cam=ds.camera).od for p in t0_paths]
    box = centroid_crop_box(frames, ds.crop_half)
    print(f"cloud centroid from {len(t0_paths)} t = 0 shots -> crop box {box}")
    return box


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-d", "--dataset", default="coarsening", choices=dataset_names(),
        help="dataset key from config/experiments.yaml",
    )
    parser.add_argument("--n-pc", type=int, default=4, help="leading components to animate")
    parser.add_argument("--fps", type=float, default=0.5,
                        help="frames per second (default 0.5: 2 s per frame)")
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()

    ds = dataset(args.dataset)
    clip = ds.calibration.clip_percentile

    # Read each shot's hold time once, before deciding on the crop.
    shots = []
    for path in ds.shot_paths():
        with h5py.File(path, "r") as f:
            if "images" not in f:       # aborted shot: globals only, no frames
                continue
        t_ms = hold_time_ms(ShotFile(path, cam=ds.camera), ds.time_index)
        if ds.max_time_ms is not None and t_ms > ds.max_time_ms:
            continue
        shots.append((path, t_ms))

    crop_box = resolve_crop_box(ds, shots)

    by_time: dict[float, list[str]] = {}
    for path, t_ms in shots:
        by_time.setdefault(t_ms, []).append(path)
    times = sorted(by_time)
    print(f"{len(shots)} shots across {len(times)} times: {times}")

    # One PCA per time, built from that time's shots alone.
    pca_by_time, main_frames = {}, []
    for t_ms in times:
        paths = sorted(by_time[t_ms], key=lambda p: REP_ORDER.index(repeat_label(p))
                       if repeat_label(p) in REP_ORDER else len(REP_ORDER))
        stack = np.stack(
            [
                rotate_and_crop(
                    ShotFile(p, cam=ds.camera, crop_box=crop_box).od, ds.rotation_deg
                )
                for p in paths
            ]
        )
        pca_by_time[t_ms] = pca_svd(stack)
        main_frames.append(stack[0])  # the base repeat where there is one
        print(f"  t = {t_ms:>8.1f} ms: {stack.shape[0]} shots")

    # -- one movie per principal component ---------------------------------- #
    for k in range(args.n_pc):
        have = [t for t in times if pca_by_time[t][1].shape[-1] > k]
        if len(have) < len(times):
            print(f"PC{k + 1}: skipping times with too few shots: "
                  f"{sorted(set(times) - set(have))}")
        if not have:
            continue

        frames = align_signs([pca_by_time[t][1][..., k] for t in have])
        limit = np.percentile(np.abs(np.stack(frames)), clip)
        write_movie(
            frames,
            have,
            [f"t = {t:g} ms      eigenvalue = {pca_by_time[t][0][k]:.1%}" for t in have],
            ds.figure(f"pc{k + 1}_evolution.gif"),
            suptitle=f"PC{k + 1} of the per-time PCA vs. evolution time",
            note=(
                "unit-norm eigenvector on a fixed scale; sign is gauge, "
                "only the pattern is physical"
            ),
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            cbar_label=f"PC{k + 1} amplitude",
            fps=args.fps,
            dpi=args.dpi,
        )

    # -- the main image, same repeat slot at every time ---------------------- #
    vmin, vmax = np.percentile(np.stack(main_frames), [100 - clip, clip])
    write_movie(
        main_frames,
        times,
        [f"t = {t:g} ms" for t in times],
        ds.figure("main_image_evolution.gif"),
        suptitle="Optical density vs. evolution time (one shot per time)",
        note="base repeat of each time where one exists",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        cbar_label="optical density",
        fps=args.fps,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
