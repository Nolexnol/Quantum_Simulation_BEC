"""
quantum_sim
===========

A numerical toolkit for simulating a two-component (spin-1/2) Bose-Einstein
condensate governed by the coupled Gross-Pitaevskii Equation (GPE) in a 1D
harmonic trap, using the Split-Step Fourier Method (SSFM).

Public API
----------
SpinorGPESolver
    The two-component (spin-1/2) 1D solver supporting real- and imaginary-time
    propagation.
Spin1GPE2D
    The three-component (spin-1) quasi-2D solver used for spin-domain / coarsening
    dynamics of a sodium (antiferromagnetic) or rubidium (ferromagnetic) condensate.
"""

from .gpe_solver import SpinorGPESolver
from .spinor1_gpe_2d import Spin1GPE2D

__all__ = ["SpinorGPESolver", "Spin1GPE2D"]
__version__ = "0.2.0"
