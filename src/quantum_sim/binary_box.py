"""
Derived physical scales for the measured-box binary-mixture simulations.

These are the numbers that turn the laboratory box into solver units, and
both ``run_binary_box_match.py`` and ``run_binary_box_ensemble.py`` need the
same values -- the ensemble must reproduce exactly the setup the match run
was validated against.

Previously the ensemble script imported them from the match script, so one
entry point depended on another entry point: it only worked because the two
files happened to sit in the same directory, and it dragged matplotlib and
the whole experimental-analysis path into a job meant to run headless. The
shared setup belongs here, where both can import it and neither depends on
the other.

Everything is derived from :mod:`quantum_sim.units` and the measured values
in ``config/experiments.yaml``; nothing here is hand-copied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import apparatus
from .units import (
    A_NA23_F0,
    A_NA23_F2,
    HBAR,
    M_NA23,
    g_quasi_2d,
    spin1_couplings_3d,
)

__all__ = ["BinaryBoxSetup", "box_setup"]


@dataclass(frozen=True)
class BinaryBoxSetup:
    """
    The measured box expressed in the solver's dimensionless units.

    In the solver ``hbar = m = g n0 = 1``; this carries the conversions back
    to micrometres and milliseconds, plus the interaction strengths that set
    the immiscibility.
    """

    box_side_um: float
    wall_1090_um: float
    v0_over_mu: float
    n0_areal: float
    omega_z: float
    xi_spin_um: float

    # -- derived ------------------------------------------------------------ #
    @property
    def g_2d(self) -> float:
        """Quasi-2D contact coupling [J m^2]."""
        c0_3d, _ = spin1_couplings_3d(A_NA23_F0, A_NA23_F2, M_NA23)
        return g_quasi_2d(c0_3d, M_NA23, self.omega_z)

    @property
    def mu(self) -> float:
        """Chemical potential [J]."""
        return self.g_2d * self.n0_areal

    @property
    def xi_um(self) -> float:
        """Density healing length [um]."""
        return HBAR / np.sqrt(M_NA23 * self.mu) * 1e6

    @property
    def t_unit_ms(self) -> float:
        """One dimensionless time unit, in milliseconds."""
        return HBAR / self.mu * 1e3

    @property
    def g(self) -> float:
        """Intra-species coupling in solver units (unity by construction)."""
        return 1.0

    @property
    def g_ud(self) -> float:
        """
        Inter-species coupling in solver units.

        Fixed by requiring the domain-wall width to match the measured spin
        healing length: ``g_ud / g = 1 + (xi / xi_s)^2``.
        """
        return self.g * (1.0 + (self.xi_um / self.xi_spin_um) ** 2)

    @property
    def mu_eff(self) -> float:
        """Mixture chemical potential [g n0]."""
        return 0.5 * (self.g + self.g_ud)


def box_setup(
    wall_1090_um: float = 4.2,
    v0_over_mu: float = 5.0,
    n0_areal: float = 1.0e14,
) -> BinaryBoxSetup:
    """
    Build the standard box setup from the measured apparatus constants.

    ``wall_1090_um`` is the measured 10-90% wall width, ``v0_over_mu`` the
    barrier height in units of mu (the data only bound it from below, so 5 is
    "effectively hard"), and ``n0_areal`` the PTAI-corrected areal density
    [m^-2].
    """
    app = apparatus()
    return BinaryBoxSetup(
        box_side_um=app.box_side_um,
        wall_1090_um=wall_1090_um,
        v0_over_mu=v0_over_mu,
        n0_areal=n0_areal,
        omega_z=app.omega_z,
        xi_spin_um=app.xi_spin_um,
    )
