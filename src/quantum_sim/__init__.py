"""
quantum_sim
===========

A numerical toolkit for simulating spinor Bose-Einstein condensates governed
by the Gross-Pitaevskii Equation (GPE), using the Split-Step Fourier Method
(SSFM), together with the analysis pipeline for the absorption images that the
simulations are compared against.

Layout
------
Dependencies point one way: ``config`` -> ``utils`` -> ``io`` -> scripts.
Nothing to the left imports anything to its right.

=============================  =============================================
:mod:`quantum_sim.config`      project paths and measured constants — the one
                               place a path or calibration number is defined
:mod:`quantum_sim.units`       CODATA constants and SI <-> solver conversions
:mod:`quantum_sim.solvers`     the GPE solvers; pure array in, array out
:mod:`quantum_sim.utils`       commonly reused functions, grouped by concern
:mod:`quantum_sim.io`          HDF5 shot files and the SQLite index
=============================  =============================================

The solvers are re-exported here for convenience. The heavier subpackages are
resolved on first access, so ``import quantum_sim`` does not pull in h5py,
scipy or matplotlib unless you actually use them.

Public API
----------
SpinorGPESolver
    Two-component (spin-1/2) 1D solver supporting real- and imaginary-time
    propagation.
Spin1GPE2D
    Three-component (spin-1) quasi-2D solver used for spin-domain / coarsening
    dynamics of a sodium (antiferromagnetic) or rubidium (ferromagnetic)
    condensate.
BinaryGPE2D
    Two-component 2D solver for an immiscible binary condensate (phase
    separation and domain coarsening).
"""

from .solvers import BinaryGPE2D, Spin1GPE2D, SpinorGPESolver

__all__ = [
    "SpinorGPESolver",
    "Spin1GPE2D",
    "BinaryGPE2D",
    "config",
    "units",
    "utils",
    "io",
    "solvers",
]
__version__ = "0.3.0"


def __getattr__(name):
    """Resolve subpackages lazily so the top-level import stays cheap."""
    if name in {"config", "units", "utils", "io", "solvers"}:
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
