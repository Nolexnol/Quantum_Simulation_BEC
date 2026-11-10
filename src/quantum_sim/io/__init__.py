"""
Reading experimental data: HDF5 shot files and the SQLite index over them.

This is the only subpackage that knows the on-disk format. Solvers and the
helpers in :mod:`quantum_sim.utils` work on plain arrays, so they stay usable
on simulated fields that never came from a camera.

:class:`Database` is imported lazily by name — it needs ``peewee``, which is
an optional extra — so ``from quantum_sim.io import ShotFile`` works on an
install without the database dependencies.
"""

from .loader import ShotCollection
from .shotfile import ShotFile, ShotFileMultiImg, ShotFileTOF, max_od

__all__ = ["ShotFile", "ShotFileTOF", "ShotFileMultiImg", "ShotCollection", "max_od", "Database"]


def __getattr__(name):
    if name == "Database":
        from .database import Database

        return Database
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
