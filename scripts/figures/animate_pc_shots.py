#!/usr/bin/env python3
"""
Per-time shot movies: the main evolution image for EVERY shot at one hold time.

animate_pc_evolution*.py sweep the hold time and show one shot (or one
eigenimage) per frame.  This is the transpose: the hold time is fixed and the
SHOT index is swept, so each output GIF is "all the shots at t = 20 ms", and
there is one GIF per hold time.  That is the view that makes a bad shot inside
an otherwise fine time obvious.

Each frame is the raw optical density of one shot -- nothing is subtracted --
on a scale shared by every shot at every time, so a dim or saturated shot reads
as dim or saturated.  (--remove-pc N projects the leading N components of that
time's own PCA out of the displayed field instead; the default is 0.)

Alongside the movies it scores every shot against its own time's siblings and
writes the outliers to anomaly_report.md.  Metrics, all computed within a time:

  od_sum      integrated OD over the window -- atom number proxy
  contrast    rms of the displayed field -- structure + envelope, raw by default
  L_um        length scale 2 pi <S>/<k S>, Tukey(0.5) apodized, in um
  cos_mean    cosine similarity to that time's mean image -- gross frame faults
  dr          frame shift against that time's mean, in px, by cross-correlation
  lev         share of the post-PC1 variance carried by this shot (expect 1/N)
  pc2_share   share of PC2's weight carried by this shot

lev and pc2_share read that time's PCA to find the shot the higher components
are fitting.  They are statistics ABOUT the PCA -- no component is ever
subtracted from an image the script writes out or from any other metric.

A shot is flagged when a modified z-score (median/MAD, one-sided where only one
tail is a fault) exceeds Z_THRESH, or when lev/pc2_share exceed LEV_FACTOR/N.
A large lev or pc2_share is the interesting one: it says the shot is not just
unusual but is itself what the higher components of that time's PCA are fitting.

Usage
    ./animate_pc_shots.py                  # 700p, t <= 800 ms, raw OD
    ./animate_pc_shots.py --t-max 2000     # include the 1000-2000 ms times
    ./animate_pc_shots.py --remove-pc 1    # show the PC1-removed field instead
    ./animate_pc_shots.py --no-gifs        # metrics + contact sheets only
"""
import argparse
import csv
import os

import h5py
import numpy as np
from scipy.stats import spearmanr

from quantum_sim.config import dataset, dataset_names
from quantum_sim.utils.plotting import use_headless_backend

use_headless_backend()
import matplotlib.pyplot as plt  # noqa: E402 — must follow the backend choice
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from quantum_sim.io import ShotFile  # noqa: E402
from quantum_sim.utils.domains import domain_length_um, window_2d  # noqa: E402
from quantum_sim.utils.images import (  # noqa: E402
    centroid_crop_box,
    register_shifts,
    rotate_and_crop,
)
from quantum_sim.utils.pca import pca_svd, remove_components  # noqa: E402
from quantum_sim.utils.stats import Z_THRESHOLD as Z_THRESH  # noqa: E402
from quantum_sim.utils.stats import modified_z  # noqa: E402

# Analysis thresholds. Imaging calibration (camera, pixel scale, rotation,
# Tukey alpha, clip percentile) and per-dataset framing now come from
# config/experiments.yaml via `dataset(...)`, so they are not repeated here.
LEV_FACTOR = 4.0        # lev > LEV_FACTOR / N counts as dominating the basis
PC2_NULL_FACTOR = 1.5   # multiples of the random-vector max share, see flag_time
T_MAX_MS = 800.0        # hold times beyond this are not analysed
N_PC_REMOVE = 0         # PCs projected out of the displayed field; 0 = raw OD
FPS = 0.5              # 2 s per frame
DPI = 90


# --------------------------------------------------------------------------
# loading, verbatim from animate_pc_evolution_700p.py where it overlaps
# --------------------------------------------------------------------------
def shot_paths(ds):
    """Every shot of the dataset that actually contains camera frames."""
    keep, dead = [], []
    for path in ds.shot_paths():
        with h5py.File(path, "r") as f:
            (keep if "images" in f else dead).append(path)
    if dead:
        print(f"skipping {len(dead)} file(s) with no /images group")
    return keep


def shot_time_ms(path, time_index):
    """Evolution time of one shot, in ms, straight from its globals.

    Reads the globals group directly rather than building a ShotFile, because
    this runs once per shot over ~750 shots before anything is decoded.
    """
    with h5py.File(path, "r") as f:
        seq = np.array(f["globals"].attrs["BEC_TIME_SEQ"])
    return round(float(seq[time_index]) * 1e3, 3)


# --------------------------------------------------------------------------
# anomaly scoring
#
# NOTE: the METRICS gates, UNITS and drift_notes() below were reconstructed
# from their call sites during the repository restructure. The metric keys,
# tails and display names are pinned exactly by write_overview(), the CSV
# columns and the report text; the numeric gate values are a reasonable
# restatement rather than a recovered original. Check them against a run you
# trust before relying on the flag counts.
# --------------------------------------------------------------------------

#: key -> (display name, which tail is a fault, (gate kind, gate value))
#:
#: The gate is a second hurdle on top of the z-score: a metric with a tiny
#: spread makes every shot a "3.5 sigma outlier" without any of them being
#: physically different, so a deviation must also be large in absolute or
#: relative terms before it is reported.
METRICS = {
    "od_sum":   ("integrated OD",           "two",  ("rel", 0.05)),
    "contrast": ("image rms",               "two",  ("rel", 0.05)),
    "L_um":     ("domain length",           "two",  ("rel", 0.05)),
    "cos_mean": ("cos to time mean",        "low",  ("abs", 0.01)),
    "dr":       ("frame shift vs mean",     "high", ("abs", 2.0)),
}

#: Unit suffix appended to a metric value inside a reason string.
UNITS = {"L_um": " um", "dr": " px"}

DRIFT_RHO = 0.6    # |Spearman rho| above which shot index vs metric is a trend
DRIFT_P = 0.01     # and the p-value it must clear
DRIFT_MIN_SHOTS = 8  # below this a rank correlation is not worth testing


def drift_notes(metrics, n_shots):
    """
    Report metrics that slide monotonically with shot index within one time.

    A metric trending across the repeats at a single hold time means those
    shots are not draws from one condition -- the apparatus moved during the
    run -- even when no individual shot is an outlier. Spearman is used rather
    than Pearson because the question is monotonicity, not linearity.
    """
    if n_shots < DRIFT_MIN_SHOTS:
        return []

    index = np.arange(n_shots)
    notes = []
    for key, (name, _tail, _gate) in METRICS.items():
        values = np.asarray(metrics[key], dtype=float)
        if not np.all(np.isfinite(values)) or np.ptp(values) == 0:
            continue
        rho, p_value = spearmanr(index, values)
        if abs(rho) > DRIFT_RHO and p_value < DRIFT_P:
            direction = "rises" if rho > 0 else "falls"
            notes.append(
                f"`{key}` ({name}) {direction} with shot index "
                f"(Spearman rho = {rho:+.2f}, p = {p_value:.1e}); "
                f"{values[0]:.4g} -> {values[-1]:.4g}{UNITS.get(key, '')}"
            )
    return notes


def flag_time(metrics, n_shots):
    """Return per-shot lists of human-readable reasons for one time's shots."""
    reasons = [[] for _ in range(n_shots)]
    if n_shots < 4:                      # MAD is meaningless on 3 points
        return reasons, {}
    zs = {}
    for key, (name, tail, (gate_kind, gate_val)) in METRICS.items():
        x = np.asarray(metrics[key], dtype=float)
        med = np.median(x)
        zs[key] = modified_z(x)
        floor = abs(med) * gate_val if gate_kind == "rel" else gate_val
        for i, z in enumerate(zs[key]):
            if abs(z) <= Z_THRESH or abs(x[i] - med) <= floor:
                continue
            side = "high" if z > 0 else "low"
            if tail != "two" and side != tail:
                continue
            reasons[i].append(f"{name} {side} ({x[i]:.4g}{UNITS.get(key, '')} "
                              f"vs {med:.4g} median, z={z:+.1f})")
    # `lev` is a share of a sum of N comparable energies, so 1/N is its null and
    # LEV_FACTOR/N is a fair cut.  `pc2_share` is not: PC2's weights across shots
    # behave like the entries of a random unit vector in R^N, whose largest
    # squared entry is ~2 ln(N)/N -- 26% at N = 25, not 4%.  Cutting at 4/N
    # there would flag chance fluctuations as outliers, so the null is used.
    pc2_null = 2 * np.log(n_shots) / n_shots
    thresholds = {"lev": LEV_FACTOR / n_shots,
                  "pc2_share": max(LEV_FACTOR / n_shots,
                                   PC2_NULL_FACTOR * pc2_null)}
    for key, name in (("lev", "post-PC1 variance"), ("pc2_share", "PC2 weight")):
        for i, v in enumerate(metrics[key]):
            if v > thresholds[key]:
                reasons[i].append(f"carries {v:.0%} of the {name} "
                                  f"(cut {thresholds[key]:.0%}; "
                                  f"{v * n_shots:.1f}x the 1/N share)")
    return reasons, zs


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def _shot_strip(ax, n_shots, flagged):
    """Slim strip under the images: one tick per shot, flagged ones in red."""
    ax.set_xlim(-0.6, n_shots - 0.4)
    ax.set_ylim(-1, 1)
    ax.hlines(0, 0, n_shots - 1, color="0.75", lw=1.5, zorder=1)
    ok = [i for i in range(n_shots) if i not in flagged]
    ax.vlines(ok, -0.3, 0.3, color="0.55", lw=1.0, zorder=2)
    if flagged:
        ax.vlines(sorted(flagged), -0.45, 0.45, color="C3", lw=2.0, zorder=2)
    marker, = ax.plot([0], [0], "o", ms=8, color="C0", zorder=3)
    ax.set_yticks([])
    step = max(1, n_shots // 10)
    ax.set_xticks(range(0, n_shots, step))
    ax.set_xlabel("shot # within this hold time", fontsize=9)
    ax.tick_params(labelsize=8)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    return marker


def write_shot_movie(t_ms, ods, labels, reasons, out_path, od_lim, cbar_label):
    """One GIF: fixed hold time, sweeping shot index, one image per shot."""
    n = len(ods)
    flagged = {i for i, r in enumerate(reasons) if r}

    fig = plt.figure(figsize=(5.6, 5.9), dpi=DPI)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.09], hspace=0.14,
                          left=0.05, right=0.88, top=0.90, bottom=0.13)
    ax_od = fig.add_subplot(gs[0, 0])
    ax_st = fig.add_subplot(gs[1, 0])

    im_od = ax_od.imshow(ods[0], origin="lower", cmap="viridis",
                         vmin=od_lim[0], vmax=od_lim[1])
    ax_od.set_xticks([])
    ax_od.set_yticks([])
    cb = fig.colorbar(im_od, ax=ax_od, fraction=0.046, pad=0.02)
    cb.set_label(cbar_label, fontsize=8)
    cb.ax.tick_params(labelsize=7)

    t_od = ax_od.set_title("", fontsize=9.5)
    marker = _shot_strip(ax_st, n, flagged)
    sup = fig.suptitle("", fontsize=12.5)
    note = fig.text(0.5, 0.015, "", ha="center", fontsize=8, color="0.35")

    def update(i):
        im_od.set_data(ods[i])
        t_od.set_text(labels[i])
        marker.set_data([i], [0])
        marker.set_color("C3" if i in flagged else "C0")
        if reasons[i]:
            sup.set_text(f"t = {t_ms:g} ms   shot {i + 1}/{n}   ANOMALY")
            sup.set_color("C3")
            note.set_text(" | ".join(reasons[i]))
            note.set_color("C3")
        else:
            sup.set_text(f"t = {t_ms:g} ms   shot {i + 1}/{n}")
            sup.set_color("black")
            note.set_text("no outlier flag against the other shots at this time")
            note.set_color("0.35")
        return im_od, t_od, marker, sup, note

    anim = FuncAnimation(fig, update, frames=n, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=FPS))
    plt.close(fig)


def write_contact_sheet(t_ms, frames, labels, reasons, out_path, title,
                        cmap, vmin, vmax):
    """Every shot of one time as a still grid; flagged shots get a red frame."""
    n = len(frames)
    ncols = min(7, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.85 * ncols, 2.05 * nrows),
                             dpi=110, squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        ax.set_xticks([])
        ax.set_yticks([])
        if i >= n:
            ax.axis("off")
            continue
        ax.imshow(frames[i], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        bad = bool(reasons[i])
        ax.set_title(labels[i], fontsize=7.5, color="C3" if bad else "black")
        for sp in ax.spines.values():
            sp.set_color("C3" if bad else "0.85")
            sp.set_linewidth(2.0 if bad else 0.8)
    n_bad = sum(1 for r in reasons if r)
    fig.suptitle(f"t = {t_ms:g} ms   {title}   ({n} shots, "
                 f"{n_bad} flagged)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)


def write_overview(times, per_time, out_path, dataset_label):
    """Per-shot metrics against hold time, flagged shots picked out in red."""
    keys = [("od_sum", "integrated OD"), ("contrast", "image rms"),
            ("L_um", r"L ($\mu$m), Tukey 0.5"), ("cos_mean", "cos to time mean"),
            ("dr", "frame shift vs mean (px)"), ("lev_ratio", r"post-PC1 var / (1/N)")]
    fig, axes = plt.subplots(3, 2, figsize=(11, 9.5), dpi=110, sharex=True)
    for ax, (key, name) in zip(axes.ravel(), keys):
        for t in times:
            m = per_time[t]
            vals = (np.asarray(m["lev"]) * len(m["labels"])
                    if key == "lev_ratio" else np.asarray(m[key], dtype=float))
            bad = np.array([bool(r) for r in m["reasons"]])
            ax.plot(np.full((~bad).sum(), t), vals[~bad],
                    "o", ms=3.2, color="C0", alpha=0.45, zorder=2)
            if bad.any():
                ax.plot(np.full(bad.sum(), t), vals[bad], "o", ms=5.5,
                        mfc="none", mec="C3", mew=1.4, zorder=3)
            ax.plot([t], [np.median(vals)], "_", ms=13, color="k", zorder=4)
        ax.set_ylabel(name, fontsize=10)
        ax.set_xscale("symlog", linthresh=20)
        ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("evolution time  t  (ms)")
    fig.suptitle(f"{dataset_label}: per-shot metrics within each hold time\n"
                 "black dash = time median, red ring = flagged shot", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=dataset_names(), default="shots700p",
                    help="dataset key from config/experiments.yaml")
    ap.add_argument("--t-max", type=float, default=T_MAX_MS,
                    help="ignore hold times above this, in ms")
    ap.add_argument("--remove-pc", type=int, default=N_PC_REMOVE, metavar="N",
                    help="project the leading N PCs of each time's own PCA out "
                         "of the displayed field and out of contrast/L "
                         "(default 0: raw OD, nothing subtracted)")
    ap.add_argument("--no-gifs", action="store_true",
                    help="skip the GIFs, keep the sheets/CSV/report")
    args = ap.parse_args()
    n_pc = args.remove_pc
    ds = dataset(args.dataset)
    clip_pctl = ds.calibration.clip_percentile

    out_dir = str(ds.figure("shot_qc").parent / "shot_qc")
    os.makedirs(out_dir, exist_ok=True)

    paths = shot_paths(ds)
    if ds.crop_box:
        crop_box = list(ds.crop_box)
    else:
        t0 = [p for p in paths if shot_time_ms(p, ds.time_index) == 0.0]
        if not t0:
            raise SystemExit("no t = 0 shots to locate the cloud with")
        crop_box = centroid_crop_box(
            [ShotFile(p, cam=ds.camera).od for p in t0], ds.crop_half
        )
        print(f"cloud centroid from {len(t0)} t = 0 shots -> crop box {crop_box}")
    print(f"{ds.name}: {len(paths)} shots, CROP_BOX = {crop_box}")

    by_time = {}
    for path in paths:
        t = shot_time_ms(path, ds.time_index)
        if t <= args.t_max:
            by_time.setdefault(t, []).append(path)
    times = sorted(by_time)
    n_used = sum(len(v) for v in by_time.values())
    print(f"{len(times)} times at or below {args.t_max:g} ms "
          f"({n_used} of {len(paths)} shots): {times}")
    print(f"displayed field: "
          f"{'raw OD, no PCs removed' if n_pc == 0 else f'PC1..PC{n_pc} removed'}")

    # ---- load, per-time PCA, per-shot metrics --------------------------
    per_time, window = {}, None
    for j, t in enumerate(times):
        names = by_time[t]
        stack = np.stack([
            rotate_and_crop(
                ShotFile(p, cam=ds.camera, crop_box=crop_box).od, ds.rotation_deg
            )
            for p in names
        ])
        if window is None:
            window = window_2d(stack.shape[1:], ds.calibration.window_alpha)
        evals, evecs = pca_svd(stack)
        n = stack.shape[0]

        field = (stack if n_pc == 0 else
                 np.stack([remove_components(im, evecs, n_pc)[0]
                           for im in stack]))
        flat = stack.reshape(n, -1)
        pc1 = evecs[..., 0].ravel()
        c1 = flat @ pc1
        resid_energy = np.einsum("ij,ij->i", flat, flat) - c1**2
        lev = resid_energy / resid_energy.sum()
        if evecs.shape[-1] > 1:
            c2 = flat @ evecs[..., 1].ravel()
            pc2_share = c2**2 / (c2**2).sum()
        else:
            pc2_share = np.full(n, 1.0 / n)

        mean_img = stack.mean(axis=0)
        mflat = mean_img.ravel()
        cos_mean = (flat @ mflat) / (np.linalg.norm(flat, axis=1)
                                     * np.linalg.norm(mflat))
        shifts = register_shifts(stack)
        dr = np.linalg.norm(shifts, axis=1)

        m = dict(
            labels=[os.path.basename(p).replace("_bec_box2", "")
                    .replace(".h5", "") for p in names],
            paths=names,
            od_sum=stack.sum(axis=(1, 2)),
            contrast=field.std(axis=(1, 2)),
            L_um=np.array([domain_length_um(f, ds.pixel_um, window) for f in field]),
            cos_mean=cos_mean, dr=dr, lev=lev, pc2_share=pc2_share,
            c1=c1, evals=evals,
        )
        m["reasons"], _ = flag_time(m, n)
        m["drift"] = drift_notes(m, n)
        m["ods"] = field
        per_time[t] = m
        n_bad = sum(1 for r in m["reasons"] if r)
        print(f"  [{j + 1:2d}/{len(times)}] t = {t:>8.1f} ms: {n:2d} shots, "
              f"PC1 = {evals[0]:.3f}, {n_bad} flagged"
              f"{', DRIFT' if m['drift'] else ''}", flush=True)

    # ---- shared OD colour scale over every shot of every time ----------
    all_od = np.concatenate([per_time[t]["ods"].ravel() for t in times])
    if n_pc == 0:
        od_lim = tuple(np.percentile(all_od, [100 - clip_pctl, clip_pctl]))
        cmap, cbar_label = "viridis", "optical density"
    else:                                     # a residual is signed: symmetric
        lim = float(np.percentile(np.abs(all_od), clip_pctl))
        od_lim, cmap = (-lim, lim), "RdBu_r"
        cbar_label = f"OD, PC1..PC{n_pc} removed"
    del all_od
    print(f"shared colour scale: [{od_lim[0]:.3f}, {od_lim[1]:.3f}]")

    # ---- movies + contact sheets, one set per time ---------------------
    sheet_title = ("optical density, every shot" if n_pc == 0
                   else f"PC1..PC{n_pc} removed, every shot")
    for t in times:
        m = per_time[t]
        tag = f"t{round(t):05d}ms"
        labels = [f"{lab}      $\\Sigma$OD = {s:.0f}"
                  for lab, s in zip(m["labels"], m["od_sum"])]
        if not args.no_gifs:
            write_shot_movie(t, m["ods"], labels, m["reasons"],
                             os.path.join(out_dir, f"main_shots_{tag}.gif"),
                             od_lim, cbar_label)
        sheet_labels = [f"{i}: {lab}" for i, lab in enumerate(m["labels"])]
        write_contact_sheet(t, m["ods"], sheet_labels, m["reasons"],
                            os.path.join(out_dir, f"sheet_od_{tag}.png"),
                            sheet_title, cmap, *od_lim)
        print(f"  wrote {tag}", flush=True)

    write_overview(times, per_time, os.path.join(out_dir, "anomaly_overview.png"),
                   ds.name)

    # ---- CSV of every metric for every shot ----------------------------
    csv_path = os.path.join(out_dir, "shot_metrics.csv")
    cols = ["t_ms", "shot_index", "file", "od_sum", "contrast", "L_um",
            "cos_mean", "dr_px", "lev", "lev_over_1oN", "pc2_share",
            "pc2_over_1oN", "flagged", "reasons"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for t in times:
            m = per_time[t]
            n = len(m["labels"])
            for i in range(n):
                w.writerow([f"{t:g}", i, os.path.basename(m["paths"][i]),
                            f"{m['od_sum'][i]:.2f}", f"{m['contrast'][i]:.5f}",
                            f"{m['L_um'][i]:.3f}", f"{m['cos_mean'][i]:.6f}",
                            f"{m['dr'][i]:.2f}", f"{m['lev'][i]:.5f}",
                            f"{m['lev'][i] * n:.2f}", f"{m['pc2_share'][i]:.5f}",
                            f"{m['pc2_share'][i] * n:.2f}",
                            "yes" if m["reasons"][i] else "",
                            "; ".join(m["reasons"][i])])

    # ---- markdown report ----------------------------------------------
    field_note = ("Images and the `contrast`/`L_um` metrics are the **raw OD** "
                  "— no principal components are subtracted anywhere. `L_um` is "
                  "therefore a whole-image length scale that the box envelope "
                  "and walls contribute to, not the PC1-removed domain size the "
                  "growth-exponent analysis uses; it is here to compare shots "
                  "against their own siblings, not across times."
                  if n_pc == 0 else
                  f"Images and the `contrast`/`L_um` metrics have PC1..PC{n_pc} "
                  f"of each time's own PCA projected out.")
    rep = [f"# Per-shot anomalies within each hold time — {ds.name}", "",
           f"Hold times up to {args.t_max:g} ms, {n_used} shots.", "",
           field_note, "",
           f"A shot is flagged when its modified z-score (median/MAD) against "
           f"the other shots at the SAME hold time exceeds {Z_THRESH} **and** the "
           f"deviation is physically meaningful, or when it dominates that "
           f"time's PCA:", ""]
    for key, (name, tail, (gk, gv)) in METRICS.items():
        gate = (f"|x - median| > {gv:.0%} of the median" if gk == "rel"
                else f"|x - median| > {gv:g}")
        tails = {"two": "either tail", "low": "low tail only",
                 "high": "high tail only"}[tail]
        rep.append(f"- `{key}` ({name}): {tails}, {gate}")
    rep += [f"- `lev`: the shot carries more than {LEV_FACTOR:g}/N of the "
            f"post-PC1 variance",
            f"- `pc2_share`: the shot carries more than "
            f"{PC2_NULL_FACTOR:g} x 2ln(N)/N of PC2's weight -- the largest "
            f"squared entry of a random unit vector in R^N, which is the right "
            f"null here (26% at N = 25, not 1/N = 4%)",
            "",
            "Times with fewer than 4 shots are not scored (the MAD is "
            "meaningless there).", "",
           "| t (ms) | shots | PC1 | flagged | run drift | median L (um) |",
           "|---|---|---|---|---|---|"]
    for t in times:
        m = per_time[t]
        rep.append(f"| {t:g} | {len(m['labels'])} | {m['evals'][0]:.3f} | "
                   f"{sum(1 for r in m['reasons'] if r)} | "
                   f"{'yes' if m['drift'] else '-'} | "
                   f"{np.median(m['L_um']):.2f} |")

    rep += ["", "## Run-level trends within a hold time", "",
            f"A metric sliding monotonically with the shot index means the "
            f"repeats at that time are not draws from one condition, even when "
            f"no individual shot is an outlier. Reported when "
            f"|Spearman rho| > {DRIFT_RHO} and p < {DRIFT_P:g}; times with "
            f"fewer than 8 shots are not tested.", ""]
    drifting = [t for t in times if per_time[t]["drift"]]
    if drifting:
        for t in drifting:
            rep.append(f"### t = {t:g} ms")
            for note in per_time[t]["drift"]:
                rep.append(f"- {note}")
            rep.append("")
    else:
        rep.append("No hold time shows a significant trend with shot index.")
        rep.append("")

    rep += ["## Flagged shots", ""]
    any_bad = False
    for t in times:
        m = per_time[t]
        bad = [i for i, r in enumerate(m["reasons"]) if r]
        if not bad:
            continue
        any_bad = True
        rep.append(f"### t = {t:g} ms  ({len(m['labels'])} shots)")
        for i in bad:
            rep.append(f"- **shot {i} — `{os.path.basename(m['paths'][i])}`**")
            for r in m["reasons"][i]:
                rep.append(f"  - {r}")
        rep.append("")
    if not any_bad:
        rep.append("No shot exceeded the thresholds at any time.")
    rep_path = os.path.join(out_dir, "anomaly_report.md")
    with open(rep_path, "w") as fh:
        fh.write("\n".join(rep) + "\n")

    print(f"\nwrote {out_dir}")
    print(f"  {len(times)} x (main_shots_*.gif, sheet_od_*.png)")
    print("  anomaly_overview.png, shot_metrics.csv, anomaly_report.md")
    total_bad = sum(sum(1 for r in per_time[t]["reasons"] if r) for t in times)
    print(f"  {total_bad} flagged shot(s) out of {n_used}")


if __name__ == "__main__":
    main()
