"""
Commonly reused functions, grouped by concern.

Each module here was, until recently, copy-pasted into several scripts:
``rotate_and_crop`` existed byte-identically in four files, ``pca_svd`` in
three, the shot-filename regex in seven. Pulling them into one place means a
fix lands everywhere at once, and means they can be unit-tested.

The split into themed files rather than one large ``utils.py`` keeps imports
honest: :mod:`~quantum_sim.utils.domains` needs only scipy, so it stays
importable on a machine with no HDF5 or notebook stack, and
:mod:`~quantum_sim.utils.plotting` is the only module that touches matplotlib.

===========================================  ==================================
:mod:`~quantum_sim.utils.shots`              shot filenames, grouping by hold time
:mod:`~quantum_sim.utils.images`             framing, rotation, registration, OD
:mod:`~quantum_sim.utils.pca`                principal-component decomposition
:mod:`~quantum_sim.utils.domains`            domain size, correlations, exponents
:mod:`~quantum_sim.utils.stats`              robust statistics, fit models
:mod:`~quantum_sim.utils.parallel`           multiprocess mapping over shots
:mod:`~quantum_sim.utils.plotting`           figure style, colour limits, movies
===========================================  ==================================

The most-used names are re-exported here, so ``from quantum_sim.utils import
rotate_and_crop`` works; import the submodule directly when you want the rest.
"""

from .domains import (
    domain_length_px,
    domain_length_um,
    first_zero,
    local_exponents,
    radial_autocorr,
    radial_spectrum,
    structure_factor,
    window_2d,
)
from .images import (
    centroid_crop_box,
    optical_density,
    register_shifts,
    rotate_and_crop,
    threshold_centroid,
)
from .parallel import run_parallel_tasks
from .pca import align_signs, pca_svd, remove_components
from .shots import (
    group_by_hold_time,
    hold_time_ms,
    one_shot_per_time,
    parse_shot_name,
    repeat_label,
)
from .stats import cube_root, modified_z

__all__ = [
    # shots
    "parse_shot_name",
    "repeat_label",
    "hold_time_ms",
    "group_by_hold_time",
    "one_shot_per_time",
    # images
    "rotate_and_crop",
    "threshold_centroid",
    "centroid_crop_box",
    "register_shifts",
    "optical_density",
    # pca
    "pca_svd",
    "align_signs",
    "remove_components",
    # domains
    "window_2d",
    "structure_factor",
    "domain_length_px",
    "domain_length_um",
    "radial_autocorr",
    "first_zero",
    "radial_spectrum",
    "local_exponents",
    # stats
    "modified_z",
    "cube_root",
    # parallel
    "run_parallel_tasks",
]
