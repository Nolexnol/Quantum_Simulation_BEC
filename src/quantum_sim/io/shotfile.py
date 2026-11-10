"""
Readers for the lab's HDF5 shot files.

Each shot stores three raw camera frames (atoms, probe, background) plus the
``globals`` group holding every sequence parameter. The classes here expose
them lazily: nothing is read from disk until a property is touched, and each
frame is cached afterwards, so iterating 750 shots to read one parameter does
not decompress 750 images.

The three variants differ only in where the atoms frame lives:

===================  =========================================================
:class:`ShotFile`    single absorption image under ``MOT``
:class:`ShotFileTOF` single image under a named label, e.g. ``PTAI``
:class:`ShotFileMultiImg`  several images per shot, indexed ``atoms_0``, ...
===================  =========================================================
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import h5py
import numpy as np

from ..utils.images import optical_density

__all__ = ["ShotFile", "ShotFileTOF", "ShotFileMultiImg", "max_od"]


class _ShotBase:
    """
    Shared loading, cropping and OD arithmetic for every shot-file flavour.

    Sequence parameters from the ``globals`` group are set as attributes, so a
    shot's hold time reads as ``shot.BEC_TIME_SEQ`` rather than a dictionary
    lookup with a string key.
    """

    #: Group under ``images/<cam>`` holding the frames.
    default_label = "MOT"

    def __init__(
        self,
        path: str | os.PathLike[str],
        cam: str,
        label: str | None = None,
        crop_box: Sequence[int] | None = None,
        od_masks: Sequence[Sequence[Sequence[int]]] | None = None,
    ):
        self.path = os.fspath(path)
        self.cam = cam
        self.img_label = label or self.default_label
        self.od_masks = od_masks

        self._crop_box = tuple(crop_box) if crop_box else None
        self._cache: dict[str, Any] = {}

        self.params: list[str] = []
        self.shot_num = 0
        self.shot_time = 0.0
        self.script = ""
        self.sequence_duration = 0.0

        with h5py.File(self.path, "r") as f:
            self.shot_num = f.attrs.get("sequence_index", 0)
            self.script = f.attrs.get("script_basename", "")
            run_time = f.attrs.get("run time")
            if run_time is not None:
                self.shot_time = datetime.strptime(run_time, "%Y%m%dT%H%M%S").timestamp()
            if "devices" in f and "pulseblaster_0" in f["devices"]:
                self.sequence_duration = np.array(
                    f["devices"]["pulseblaster_0"].attrs["stop_time"]
                )

            self.params = list(f["globals"].attrs.keys())
            for param in self.params:
                setattr(self, param, f["globals"].attrs[param])

    # -- representation ----------------------------------------------------- #
    def __repr__(self) -> str:
        return f"{type(self).__name__}({os.path.basename(self.path)!r}, cam={self.cam!r})"

    def as_dict(self) -> dict[str, Any]:
        """Sequence parameters plus provenance, for building a table of shots."""
        return {key: getattr(self, key) for key in [*self.params, "path", "cam"]}

    # -- cropping ----------------------------------------------------------- #
    @property
    def crop_box(self):
        """``[r0, r1, c0, c1]`` applied to every frame, or ``None`` for the full chip."""
        return self._crop_box

    @crop_box.setter
    def crop_box(self, value):
        self._cache.clear()  # cached frames were cropped with the old box
        self._crop_box = tuple(value) if value else None

    def _crop(self, img: np.ndarray) -> np.ndarray:
        if not self._crop_box:
            return img
        r0, r1, c0, c1 = self._crop_box
        return img[r0:r1, c0:c1]

    def _frame(self, name: str) -> np.ndarray:
        """Read, crop and cache one dataset from this shot's image group."""
        if name not in self._cache:
            with h5py.File(self.path, "r") as f:
                group = f["images"][self.cam][self.img_label]
                self._cache[name] = self._crop(np.array(group[name], dtype=int))
        return self._cache[name]

    # -- frames -------------------------------------------------------------- #
    @property
    def img_atoms(self) -> np.ndarray:
        return self._frame(self._atoms_key())

    @property
    def img_probe(self) -> np.ndarray:
        return self._frame("probe_beam")

    @property
    def img_background(self) -> np.ndarray:
        return self._frame("background")

    @property
    def od(self) -> np.ndarray:
        """Optical density, computed once and cached."""
        if "od" not in self._cache:
            od = optical_density(self.img_atoms, self.img_probe, self.img_background)
            self._cache["od"] = mask_regions(od, self.od_masks)
        return self._cache["od"]

    # -- subclass hook -------------------------------------------------------- #
    def _atoms_key(self) -> str:
        return "atoms"

    def rawfile(self) -> h5py.File:
        """Open the underlying HDF5 file; the caller is responsible for closing it."""
        return h5py.File(self.path, "r")


class ShotFile(_ShotBase):
    """A single absorption image stored under ``images/<cam>/MOT``."""


class ShotFileTOF(_ShotBase):
    """
    A single image under a named label, with the atoms frame suffixed.

    Time-of-flight sequences write ``atoms_<n>``; this picks the first one.
    """

    default_label = "PTAI"

    def _atoms_key(self) -> str:
        if "_atoms_key" not in self._cache:
            with h5py.File(self.path, "r") as f:
                keys = [
                    k
                    for k in f["images"][self.cam][self.img_label]
                    if k.startswith("atoms_")
                ]
            if not keys:
                raise KeyError(
                    f"No atoms_* dataset in {self.cam}/{self.img_label} of {self.path}"
                )
            self._cache["_atoms_key"] = sorted(keys)[0]
        return self._cache["_atoms_key"]


class ShotFileMultiImg(_ShotBase):
    """
    Several atoms frames per shot, addressed by index.

    ``img_atoms`` and ``od`` take an index here rather than being properties,
    since there is no single frame to return.
    """

    default_label = "PTAI"

    @property
    def n_img_atoms(self) -> int:
        """Number of ``atoms_*`` datasets in this shot."""
        with h5py.File(self.path, "r") as f:
            return sum(
                key.startswith("atoms")
                for key in f["images"][self.cam][self.img_label]
            )

    def img_atoms(self, index: int) -> np.ndarray:  # type: ignore[override]
        return self._frame(f"atoms_{index}")

    def od(self, index: int) -> np.ndarray:  # type: ignore[override]
        key = f"od_{index}"
        if key not in self._cache:
            od = optical_density(
                self.img_atoms(index), self.img_probe, self.img_background
            )
            self._cache[key] = mask_regions(od, self.od_masks)
        return self._cache[key]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def mask_regions(img: np.ndarray, masks) -> np.ndarray:
    """Zero out rectangular regions ``((x0, x1), (y0, y1))`` of a frame."""
    if not masks:
        return img
    out = np.array(img, copy=True)
    for mask_x, mask_y in masks:
        out[mask_y[0] : mask_y[1], mask_x[0] : mask_x[1]] = 0
    return out


def max_od(od, sigma: float = 10.0) -> float:
    """Peak OD of a blurred frame, so a single hot pixel cannot set it."""
    from scipy.ndimage import gaussian_filter

    return float(gaussian_filter(np.asarray(od, dtype=float), sigma=sigma).max())
