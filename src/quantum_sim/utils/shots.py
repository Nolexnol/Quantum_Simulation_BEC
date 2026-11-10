"""
Shot-file bookkeeping: parsing names, grouping runs by hold time.

Seven scripts each carried their own copy of the filename regex and four had
their own ``collect_shots()``. They are here once, parameterised by the small
things that actually differed between them (whether repeats are kept, whether
a time cut applies).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable

import numpy as np

from ..config import SHOT_NAME_RE, Dataset

__all__ = [
    "parse_shot_name",
    "repeat_label",
    "hold_time_ms",
    "group_by_hold_time",
    "one_shot_per_time",
]


def parse_shot_name(path: str | os.PathLike[str]) -> dict[str, str] | None:
    """
    Split a shot filename into its sequence, index and repeat fields.

    Returns ``None`` when the name does not follow the lab convention
    ``<date>_<run>_bec_box2_<index>[_rep<nnnnn>].h5``, so a directory can be
    filtered with a single truthiness test.

    >>> parse_shot_name("2026-07-17_0022_bec_box2_00_rep00003.h5")["rep"]
    '00003'
    >>> parse_shot_name("notes.txt") is None
    True
    """
    match = SHOT_NAME_RE.match(os.path.basename(os.fspath(path)))
    return match.groupdict() if match else None


def repeat_label(path: str | os.PathLike[str]) -> str:
    """Repeat identifier of a shot; the un-suffixed shot is labelled ``"base"``."""
    parsed = parse_shot_name(path)
    if parsed is None:
        raise ValueError(f"Not a recognised shot filename: {path}")
    return parsed["rep"] or "base"


def hold_time_ms(shot, time_index: int) -> float:
    """
    Hold time of a shot in milliseconds, read from ``BEC_TIME_SEQ``.

    ``time_index`` selects the swept element of the sequence; it is a
    per-dataset acquisition setting and comes from ``config/experiments.yaml``
    rather than being spelled ``[6]`` inline.
    """
    sequence = np.asarray(shot.BEC_TIME_SEQ)
    return round(float(sequence[time_index]) * 1e3)


def group_by_hold_time(
    dataset: Dataset,
    *,
    loader: Callable[[str], object] | None = None,
    max_time_ms: float | None = None,
    keep_repeats: bool = True,
) -> dict[float, dict[str, object]] | dict[float, list[object]]:
    """
    Map hold time [ms] to the shots taken at it.

    With ``keep_repeats`` the value is ``{repeat_label: shot}``; without it the
    value is a plain list. ``max_time_ms`` defaults to the dataset's own cut.

    ``loader`` builds a shot object from a path and defaults to
    :class:`quantum_sim.io.ShotFile` with the dataset's camera and crop box.
    It is injected rather than imported at module scope so that this module
    does not drag in h5py for callers that only want the name parsing.
    """
    if loader is None:
        from ..io import ShotFile  # noqa: PLC0415 — keeps h5py optional here

        def loader(path):  # type: ignore[misc]
            return ShotFile(path, cam=dataset.camera, crop_box=dataset.crop_box)

    if max_time_ms is None:
        max_time_ms = dataset.max_time_ms

    grouped: dict[float, object] = {}
    for path in dataset.shot_paths():
        shot = loader(path)
        t_ms = hold_time_ms(shot, dataset.time_index)
        if max_time_ms is not None and t_ms > max_time_ms:
            continue
        if keep_repeats:
            grouped.setdefault(t_ms, {})[repeat_label(path)] = shot
        else:
            grouped.setdefault(t_ms, []).append(shot)
    return dict(sorted(grouped.items()))


def one_shot_per_time(
    grouped: dict[float, dict[str, object]],
    preferred: Iterable[str] = ("base",),
) -> dict[float, object]:
    """
    Collapse ``{time: {repeat: shot}}`` to one shot per time.

    Prefers the repeat labels in ``preferred`` (the un-suffixed shot first),
    then falls back to the earliest repeat, so a single-repeat time series is
    reproducible rather than dependent on directory ordering.
    """
    chosen: dict[float, object] = {}
    for t_ms, repeats in grouped.items():
        order = [label for label in preferred if label in repeats]
        order += sorted(label for label in repeats if label not in preferred)
        if order:
            chosen[t_ms] = repeats[order[0]]
    return chosen
