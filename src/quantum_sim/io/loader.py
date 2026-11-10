"""
Bulk loading of many shots into a single in-memory table.

The previous implementation kept its state in class attributes, so two
analyses in one notebook silently shared — and overwrote — each other's data,
and ``load_files_from_dir`` could not be called twice without clobbering the
first result. :class:`ShotCollection` holds that state per instance instead.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ..utils.images import optical_density
from .shotfile import mask_regions

__all__ = ["ShotCollection"]


class ShotCollection:
    """
    An accumulating table of shots: OD frames plus their sequence parameters.

    >>> shots = ShotCollection(cam="cam6", crop_box=[130, 400, 105, 380])
    >>> shots.load_dir("data/raw/coarsening_data")           # doctest: +SKIP
    >>> subset = shots.filter(BEC_TIME_SEQ=0.05)             # doctest: +SKIP
    """

    def __init__(
        self,
        cam: str = "cam3",
        label: str = "MOT",
        crop_box: Sequence[int] | None = None,
        od_masks: Sequence[Sequence[Sequence[int]]] | None = None,
    ):
        self.cam = cam
        self.label = label
        self.crop_box = tuple(crop_box) if crop_box else None
        self.od_masks = od_masks
        self.data: dict[str, list[Any]] = {"od": [], "probe": [], "atoms": []}

    # -- loading ------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.data["od"])

    def clear(self) -> None:
        """Drop everything loaded so far."""
        self.data = {"od": [], "probe": [], "atoms": []}

    def _crop(self, img: np.ndarray) -> np.ndarray:
        if not self.crop_box:
            return img
        r0, r1, c0, c1 = self.crop_box
        return img[r0:r1, c0:c1]

    def load_file(self, path: str | os.PathLike[str], params: Iterable[str] | None = None):
        """Append one shot. Missing parameters are recorded as NaN, not dropped."""
        with h5py.File(os.fspath(path), "r") as f:
            group = f["images"][self.cam][self.label]
            atoms = self._crop(np.array(group["atoms"], dtype=int))
            background = self._crop(np.array(group["background"], dtype=int))
            probe = self._crop(np.array(group["probe_beam"], dtype=int))

            od = optical_density(atoms, probe, background)

            if params is None:
                params = list(f["globals"].attrs.keys())

            for param in params:
                if param not in self.data:
                    if len(self) > 0:
                        raise ValueError(
                            f"Parameter {param!r} appeared after {len(self)} shots were "
                            "loaded; the table would have ragged columns. Load with an "
                            "explicit `params` list instead."
                        )
                    self.data[param] = []
                try:
                    self.data[param].append(f["globals"].attrs[param])
                except KeyError:
                    self.data[param].append(np.nan)

            self.data["od"].append(mask_regions(od, self.od_masks))
            self.data["probe"].append(np.maximum(probe - background, 1e-6))
            self.data["atoms"].append(np.maximum(atoms - background, 1e-6))

        return self.data

    def load_dir(
        self,
        dirname: str | os.PathLike[str],
        params: Iterable[str] | None = None,
        skip_errors: bool = True,
    ) -> ShotCollection:
        """Append every ``.h5`` in a directory, in sorted order."""
        paths = sorted(p for p in Path(dirname).iterdir() if p.is_file() and p.suffix == ".h5")
        return self._load_many(paths, params, skip_errors)

    def load_from_db(
        self,
        records,
        params: Iterable[str] | None = None,
        skip_errors: bool = False,
    ) -> ShotCollection:
        """Append the shots named by a sequence of database rows."""
        paths = [record.shot_path for record in records]
        return self._load_many(paths, params, skip_errors)

    def _load_many(self, paths, params, skip_errors: bool) -> ShotCollection:
        failures: list[tuple[str, str]] = []
        for path in paths:
            if not str(path).endswith(".h5"):
                continue
            try:
                self.load_file(path, params)
            except Exception as exc:  # noqa: BLE001 — one bad shot must not stop a run
                if not skip_errors:
                    raise
                failures.append((str(path), f"{type(exc).__name__}: {exc}"))

        # The old code swallowed these silently, so a half-loaded run looked complete.
        if failures:
            print(f"Skipped {len(failures)} unreadable shot(s):")
            for path, reason in failures[:5]:
                print(f"  {os.path.basename(path)}: {reason}")
            if len(failures) > 5:
                print(f"  ... and {len(failures) - 5} more")
        return self

    # -- querying ------------------------------------------------------------ #
    def filter(self, **queries) -> dict[str, np.ndarray]:
        """
        Select the shots whose parameters equal every given value.

        >>> shots.filter(BEC_TIME_SEQ=0.05, BOX_DEPTH=1.2)    # doctest: +SKIP
        """
        selector = np.ones(len(self), dtype=bool)
        for key, value in queries.items():
            if key not in self.data:
                raise KeyError(
                    f"Unknown parameter {key!r}. Loaded: {', '.join(sorted(self.data))}"
                )
            selector &= np.asarray(self.data[key]) == value
        return {key: np.asarray(value)[selector] for key, value in self.data.items()}
