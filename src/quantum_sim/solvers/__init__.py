"""
Gross-Pitaevskii solvers, integrated with the split-step Fourier method.

All three work in dimensionless units and are pure: they take arrays and
parameters, and return arrays. They never read a file, never plot, and never
consult the project configuration, which is what makes them testable against
analytic results in :mod:`tests.test_gpe_solver`.

Use :mod:`quantum_sim.units` to convert laboratory parameters into the
dimensionless quantities these expect.

======================  ====================================================
:class:`SpinorGPESolver`  two-component (spin-1/2) condensate in a 1D trap
:class:`Spin1GPE2D`       three-component (spin-1) quasi-2D condensate
:class:`BinaryGPE2D`      immiscible two-component quasi-2D condensate
======================  ====================================================
"""

from .binary_gpe_2d import BinaryGPE2D
from .gpe_solver import SpinorGPESolver
from .spinor1_gpe_2d import Spin1GPE2D

__all__ = ["SpinorGPESolver", "Spin1GPE2D", "BinaryGPE2D"]
