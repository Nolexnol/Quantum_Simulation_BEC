"""
Shared figure and animation helpers.

This is the only module in the package that imports matplotlib at module
scope, and even here the style is applied by an explicit :func:`apply_style`
call rather than on import. A library that restyles global ``rcParams`` the
moment it is imported silently changes every other plot in a notebook, which
is why the old ``file.py`` had to be imported before anything else to get
consistent figures.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

__all__ = [
    "apply_style",
    "use_headless_backend",
    "robust_limits",
    "shared_od_limits",
    "montage",
    "timeline_axis",
    "write_movie",
    "show_od",
]


def use_headless_backend() -> None:
    """
    Switch matplotlib to Agg, for scripts that only write files.

    Must be called before the first pyplot figure is created. Every batch
    script does this so it can run over SSH or in CI without a display.
    """
    import matplotlib

    matplotlib.use("Agg")


def apply_style() -> None:
    """Apply the project's figure defaults. Opt-in, never on import."""
    plt.rc("font", size=12)
    plt.rc("xtick", labelsize=10)
    plt.rc("ytick", labelsize=10)
    plt.rc("axes", titlesize=12, labelsize=11)
    plt.rc("figure", dpi=120)
    plt.rc("savefig", dpi=150, bbox_inches="tight")


def robust_limits(frames, percentile: float = 99.5) -> tuple[float, float]:
    """
    Symmetric colour limits that ignore the extreme tails.

    A single hot pixel sets the limits if you use min/max, washing out the
    structure you are trying to see; a high percentile keeps the scale on the
    signal.
    """
    values = np.abs(np.asarray(frames, dtype=float))
    limit = float(np.percentile(values, percentile))
    return -limit, limit


def shared_od_limits(frames, percentile: float = 99.5) -> tuple[float, float]:
    """
    Colour limits ``(0, robust max)`` shared across a series of OD frames.

    Every montage in this project needs one scale for all panels: with
    per-panel autoscaling, a visibly brighter cell means only that the panel
    was rescaled, not that the cloud was denser.
    """
    values = np.concatenate([np.asarray(f, dtype=float).ravel() for f in frames])
    return 0.0, float(np.percentile(values, percentile))


def montage(
    images,
    titles,
    *,
    ncols: int = 6,
    panel_size: tuple[float, float] = (2.1, 2.2),
    suptitle: str = "",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str = "Optical density",
    title_fontsize: float = 11,
):
    """
    Lay a series of frames out as a labelled grid sharing one colour bar.

    Returns ``(fig, axes)``; the caller saves. Unused cells in the last row are
    hidden rather than left as empty axes.
    """
    images = list(images)
    titles = list(titles)
    if vmin is None or vmax is None:
        auto_vmin, auto_vmax = shared_od_limits(images)
        vmin = auto_vmin if vmin is None else vmin
        vmax = auto_vmax if vmax is None else vmax

    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows)
    )
    axes = np.atleast_1d(axes).ravel()

    image = None
    for ax, title, frame in zip(axes, titles, images):
        image = ax.imshow(frame, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=title_fontsize)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(images) :]:
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95 if suptitle else 1.0])

    cbar = fig.colorbar(image, ax=axes.tolist(), fraction=0.02, pad=0.02)
    cbar.set_label(cbar_label)
    return fig, axes


def timeline_axis(ax, times: Sequence[float], ticks: Sequence[float] | None = None):
    """
    Slim strip under an animation: every measured time, marker on the current one.

    Returns the marker artist so the animation callback can move it.
    """
    times = np.asarray(times, dtype=float)
    span = times[-1] - times[0] if len(times) > 1 else 1.0

    ax.set_xlim(times[0] - 0.03 * span, times[-1] + 0.03 * span)
    ax.set_ylim(-1, 1)
    ax.hlines(0, times[0], times[-1], color="0.75", lw=1.5, zorder=1)
    ax.vlines(times, -0.3, 0.3, color="0.55", lw=1.0, zorder=2)
    (marker,) = ax.plot([times[0]], [0], "o", ms=9, color="C3", zorder=3)

    ax.set_yticks([])
    if ticks is not None:
        ax.set_xticks(list(ticks))
    ax.set_xlabel("evolution time  t  (ms)", fontsize=10)
    ax.tick_params(labelsize=9)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    return marker


def write_movie(
    frames,
    times,
    titles,
    out_path,
    *,
    suptitle: str = "",
    note: str = "",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str = "",
    fps: float = 1.5,
    dpi: int = 120,
    time_ticks: Sequence[float] | None = None,
) -> str:
    """
    Render a one-image-per-time animation to a looping GIF.

    Two scripts carried near-identical copies of this, differing only in the
    constants now exposed as keyword arguments.
    """
    out_path = os.fspath(out_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    fig, (ax, ax_time) = plt.subplots(
        2,
        1,
        figsize=(5.8, 6.0),
        dpi=dpi,
        gridspec_kw={"height_ratios": [1, 0.10], "hspace": 0.16},
    )

    img = ax.imshow(frames[0], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    title = ax.set_title(titles[0], fontsize=11)

    cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    marker = timeline_axis(ax_time, times, ticks=time_ticks)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12.5)
    if note:
        fig.text(0.5, 0.015, note, ha="center", fontsize=8, color="0.35")
    fig.subplots_adjust(top=0.91, bottom=0.11, left=0.06, right=0.87)

    def update(i):
        img.set_data(frames[i])
        title.set_text(titles[i])
        marker.set_data([times[i]], [0])
        return img, title, marker

    animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
    animation.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)

    print(f"Saved {os.path.abspath(out_path)}  ({len(frames)} frames)")
    return out_path


def show_od(od, ax=None, sigma: float = 10.0, **imshow_kwargs):
    """
    Display an OD frame with colour limits set from a blurred copy.

    Interactive convenience for notebooks; scripts should set their own limits
    so a figure series shares one scale.
    """
    from scipy.ndimage import gaussian_filter

    od = np.asarray(od, dtype=float)
    smooth = gaussian_filter(od, sigma=sigma)
    ax = ax or plt.gca()

    kwargs = {"vmin": smooth.min(), "vmax": smooth.max()}
    kwargs.update(imshow_kwargs)
    image = ax.imshow(od, **kwargs)
    plt.colorbar(image, ax=ax)
    return image
