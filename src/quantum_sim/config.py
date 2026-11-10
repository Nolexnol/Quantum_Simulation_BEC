"""
Project paths and measured constants — the single source of truth.

Nothing else in this repository is allowed to hard-code a filesystem path or a
calibration number. Both live here so that:

* moving 1.4 GB of shots to an external drive is one environment variable, not
  a search-and-replace across every script, and
* re-measuring the camera scale is a one-line edit to ``config/experiments.yaml``
  instead of an error-prone hunt through five files that each kept their own
  copy of ``PIXEL_UM = 0.66``.

Every directory can be overridden from the environment, which is what makes the
code runnable on a cluster or a colleague's laptop:

===========================  ===================================================
``QUANTUM_SIM_ROOT``         project root (default: the repository containing
                             this file)
``QUANTUM_SIM_DATA_DIR``     raw + processed data (default: ``<root>/data``)
``QUANTUM_SIM_RESULTS_DIR``  figures + derived arrays (default: ``<root>/results``)
``QUANTUM_SIM_CONFIG``       experiment config file (default:
                             ``<root>/config/experiments.yaml``)
===========================  ===================================================

Typical use::

    from quantum_sim.config import PATHS, dataset

    ds = dataset("coarsening")
    for path in ds.shot_paths():
        ...
    fig_path = PATHS.figure("coarsening_size.png")
"""

from __future__ import annotations

import functools
import glob
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "PATHS",
    "Paths",
    "Dataset",
    "Calibration",
    "Apparatus",
    "dataset",
    "dataset_names",
    "calibration",
    "apparatus",
    "SHOT_NAME_RE",
]


# --------------------------------------------------------------------------- #
# Shot-file naming
#
# Lives here rather than in seven separate scripts, each of which had its own
# copy of this regex.
# --------------------------------------------------------------------------- #
SHOT_NAME_RE = re.compile(
    r"^(?P<seq>\d{4}-\d{2}-\d{2}_\d{4})_bec_box2_(?P<idx>\d+)"
    r"(?:_rep(?P<rep>\d+))?\.h5$"
)


def _find_root() -> Path:
    """Locate the project root: env override, then the repo containing this file."""
    env = os.environ.get("QUANTUM_SIM_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    # src/quantum_sim/config.py -> src/quantum_sim -> src -> <root>
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "pyproject.toml").is_file():
        return candidate

    # Installed non-editable (site-packages): fall back to the working directory.
    return Path.cwd()


def _env_dir(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value).expanduser().resolve() if value else default


@dataclass(frozen=True)
class Paths:
    """Every directory the project reads from or writes to."""

    root: Path
    data: Path
    results: Path
    config_file: Path

    # -- data ------------------------------------------------------------- #
    @property
    def raw(self) -> Path:
        """Untouched lab output (.h5 shots). Read-only by convention."""
        return self.data / "raw"

    @property
    def processed(self) -> Path:
        """Reduced arrays that are expensive to recompute but cheap to redo."""
        return self.data / "processed"

    @property
    def db(self) -> Path:
        """SQLite index over the raw shots."""
        return self.data / "db"

    @property
    def archive(self) -> Path:
        """Compressed originals kept for provenance."""
        return self.data / "archive"

    @property
    def database(self) -> Path:
        """Path of the shot-index database."""
        return self.db / "shots.db"

    # -- outputs ----------------------------------------------------------- #
    @property
    def figures(self) -> Path:
        """Publication and diagnostic figures."""
        return self.results / "figures"

    @property
    def derived(self) -> Path:
        """Intermediate .npz produced by one script and consumed by another."""
        return self.results / "derived"

    # -- helpers ----------------------------------------------------------- #
    def figure(self, *parts: str) -> Path:
        """Absolute path for a figure, creating its parent directory."""
        path = self.figures.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def derived_file(self, *parts: str) -> Path:
        """Absolute path for a derived array, creating its parent directory."""
        path = self.derived.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


_ROOT = _find_root()

PATHS = Paths(
    root=_ROOT,
    data=_env_dir("QUANTUM_SIM_DATA_DIR", _ROOT / "data"),
    results=_env_dir("QUANTUM_SIM_RESULTS_DIR", _ROOT / "results"),
    config_file=_env_dir("QUANTUM_SIM_CONFIG", _ROOT / "config" / "experiments.yaml"),
)


# --------------------------------------------------------------------------- #
# Configuration file
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def _load_config() -> dict[str, Any]:
    import yaml  # imported lazily so `import quantum_sim` stays cheap

    if not PATHS.config_file.is_file():
        raise FileNotFoundError(
            f"Experiment config not found at {PATHS.config_file}. "
            "Set QUANTUM_SIM_CONFIG or restore config/experiments.yaml."
        )
    with PATHS.config_file.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass(frozen=True)
class Calibration:
    """Imaging settings shared by every dataset unless overridden."""

    camera: str = "cam6"
    pixel_um: float = 0.66
    rotation_deg: float = 45.0
    time_index: int = 6
    window_alpha: float = 0.5
    clip_percentile: float = 99.5


@dataclass(frozen=True)
class Apparatus:
    """Measured properties of the trap and the condensate."""

    box_side_um: float = 112.9
    sheet_trap_hz: float = 2000.0
    xi_spin_um: float = 2.0
    n_areal_per_um2: float = 100.0

    @property
    def omega_z(self) -> float:
        """Sheet-trap angular frequency [rad/s]."""
        return 2.0 * np.pi * self.sheet_trap_hz

    @property
    def n_areal_si(self) -> float:
        """Areal density in SI [atoms/m^2]."""
        return self.n_areal_per_um2 * 1e12


@dataclass(frozen=True)
class Dataset:
    """One acquisition run: where its shots live and how they were framed."""

    name: str
    path: Path
    calibration: Calibration
    description: str = ""
    crop_box: tuple[int, int, int, int] | None = None
    crop_half: int | None = None
    max_time_ms: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    # -- convenience passthroughs ------------------------------------------ #
    @property
    def camera(self) -> str:
        return self.calibration.camera

    @property
    def pixel_um(self) -> float:
        return self.calibration.pixel_um

    @property
    def rotation_deg(self) -> float:
        return self.calibration.rotation_deg

    @property
    def time_index(self) -> int:
        return self.calibration.time_index

    def shot_paths(self, pattern: str = "*bec_box2*.h5") -> list[str]:
        """Sorted absolute paths of the shot files belonging to this dataset."""
        if not self.path.is_dir():
            raise FileNotFoundError(
                f"Dataset '{self.name}' expects shots in {self.path}, which does "
                "not exist. Point QUANTUM_SIM_DATA_DIR at the drive holding the "
                "raw data, or fix 'path' in config/experiments.yaml."
            )
        paths = sorted(glob.glob(str(self.path / pattern)))
        return [p for p in paths if SHOT_NAME_RE.match(os.path.basename(p))]

    def figure(self, *parts: str) -> Path:
        """Figure path namespaced under this dataset."""
        return PATHS.figure(self.name, *parts)


@functools.lru_cache(maxsize=1)
def calibration() -> Calibration:
    """Imaging calibration defaults from the config file."""
    return Calibration(**_load_config().get("calibration", {}))


@functools.lru_cache(maxsize=1)
def apparatus() -> Apparatus:
    """Measured apparatus constants from the config file."""
    return Apparatus(**_load_config().get("apparatus", {}))


def dataset_names() -> list[str]:
    """Names of every dataset declared in the config file."""
    return sorted(_load_config().get("datasets", {}))


@functools.cache
def dataset(name: str) -> Dataset:
    """Look up a dataset by the key it is given in ``config/experiments.yaml``."""
    entries = _load_config().get("datasets", {})
    if name not in entries:
        raise KeyError(
            f"Unknown dataset {name!r}. Declared datasets: {', '.join(sorted(entries)) or '(none)'}"
        )

    entry = dict(entries[name])
    rel = entry.pop("path", name)
    description = entry.pop("description", "").strip()
    crop_box = entry.pop("crop_box", None)
    crop_half = entry.pop("crop_half", None)
    max_time_ms = entry.pop("max_time_ms", None)

    # Dataset-level imaging overrides shadow the shared calibration block.
    overrides = {
        key: entry.pop(key)
        for key in list(entry)
        if key in Calibration.__dataclass_fields__
    }
    cal = Calibration(**{**calibration().__dict__, **overrides})

    return Dataset(
        name=name,
        path=(PATHS.raw / rel).resolve(),
        calibration=cal,
        description=description,
        crop_box=tuple(crop_box) if crop_box else None,
        crop_half=crop_half,
        max_time_ms=max_time_ms,
        extra=entry,
    )
