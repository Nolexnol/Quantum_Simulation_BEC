"""
Physical constants (SI) and unit-conversion helpers.

The solvers in this package work in *dimensionless* units — that is standard
practice: it keeps the numerics well-conditioned (all quantities O(1)) and one
dimensionless run describes a whole family of experiments. This module is the
bridge to real laboratory numbers: it converts measured scattering lengths,
trap frequencies, atom numbers and densities into the dimensionless couplings
the solvers take, and back again for plotting in micrometres/milliseconds.

Two unit systems are provided, one per solver:

* ``HarmonicUnits`` (for ``SpinorGPESolver``): lengths in the oscillator
  length a_ho = sqrt(hbar / (m omega_x)), times in 1/omega_x, energies in
  hbar omega_x.
* ``SpinorUnits2D`` (for ``Spin1GPE2D``): the natural scales of spin-mixing
  physics — energies in the spin-interaction energy g_s n0, times in
  hbar / (g_s n0), lengths in hbar / sqrt(m g_s n0) (~ the spin healing
  length).
"""

from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- #
# Fundamental constants (CODATA 2018, SI units)
# --------------------------------------------------------------------------- #
HBAR = 1.054571817e-34        # reduced Planck constant [J s]
H_PLANCK = 6.62607015e-34     # Planck constant [J s]
A_BOHR = 5.29177210903e-11    # Bohr radius [m]
AMU = 1.66053906660e-27       # atomic mass unit [kg]
MU_B = 9.2740100783e-24       # Bohr magneton [J/T]

# Atomic masses
M_RB87 = 86.909180527 * AMU   # rubidium-87 [kg]
M_NA23 = 22.98976928 * AMU    # sodium-23 [kg]

# s-wave scattering lengths [m]
# 87Rb hyperfine states |F=1,mF=-1> ("up") and |F=2,mF=+1> ("down");
# Egorov et al., PRA 87, 053614 (2013).
A_RB87_UP_UP = 100.40 * A_BOHR      # a_{11} [m]
A_RB87_DOWN_DOWN = 95.44 * A_BOHR   # a_{22} [m]
A_RB87_UP_DOWN = 98.01 * A_BOHR     # a_{12} [m]

# 23Na F=1 spin-1 collision channels; Knoop et al., PRA 83, 042704 (2011).
A_NA23_F0 = 48.91 * A_BOHR    # total spin F=0 channel a_0 [m]
A_NA23_F2 = 54.54 * A_BOHR    # total spin F=2 channel a_2 [m]

# Quadratic Zeeman shift of 23Na F=1: q = Q_NA23_ZEEMAN * B^2 (B in gauss).
Q_NA23_ZEEMAN_HZ_PER_G2 = 276.8    # [Hz/G^2] (from the Breit-Rabi formula)


# --------------------------------------------------------------------------- #
# 3D coupling constants from scattering lengths
# --------------------------------------------------------------------------- #
def g_contact_3d(a_s: float, mass: float) -> float:
    """3D contact coupling g = 4 pi hbar^2 a_s / m  [J m^3]."""
    return 4.0 * np.pi * HBAR**2 * a_s / mass


def spin1_couplings_3d(a_f0: float, a_f2: float, mass: float):
    """
    Spin-1 density and spin couplings from the F=0,2 scattering lengths [m]:
        c0 = 4 pi hbar^2 (a0 + 2 a2) / (3 m)   [J m^3]  (density channel)
        c2 = 4 pi hbar^2 (a2 - a0) / (3 m)     [J m^3]  (spin channel)
    c2 > 0 is antiferromagnetic/polar (23Na); c2 < 0 ferromagnetic (87Rb).
    """
    pref = 4.0 * np.pi * HBAR**2 / (3.0 * mass)
    return pref * (a_f0 + 2.0 * a_f2), pref * (a_f2 - a_f0)


# --------------------------------------------------------------------------- #
# Dimensional reduction (tight transverse confinement)
# --------------------------------------------------------------------------- #
def oscillator_length(mass: float, omega: float) -> float:
    """Harmonic-oscillator length a_ho = sqrt(hbar / (m omega))  [m]."""
    return np.sqrt(HBAR / (mass * omega))


def g_quasi_1d(a_s: float, omega_perp: float) -> float:
    """
    Effective 1D coupling for a gas frozen in the transverse ground state of a
    radial trap omega_perp [rad/s]:  g_1D = 2 hbar omega_perp a_s  [J m].
    (Valid away from confinement-induced resonances, a_s << a_perp.)
    """
    return 2.0 * HBAR * omega_perp * a_s


def g_quasi_2d(g_3d: float, mass: float, omega_z: float) -> float:
    """
    Effective 2D coupling for a gas frozen in the ground state of a tight
    z-trap omega_z [rad/s]:  g_2D = g_3D / (sqrt(2 pi) a_z)  [J m^2].
    """
    return g_3d / (np.sqrt(2.0 * np.pi) * oscillator_length(mass, omega_z))


# --------------------------------------------------------------------------- #
# Unit systems
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HarmonicUnits:
    """
    Harmonic-oscillator units for a particle of ``mass`` [kg] in a trap of
    frequency ``omega`` [rad/s]: hbar = m = omega = 1.
    """
    mass: float    # [kg]
    omega: float   # [rad/s]

    @property
    def length(self) -> float:
        """Unit of length a_ho = sqrt(hbar / (m omega))  [m]."""
        return oscillator_length(self.mass, self.omega)

    @property
    def time(self) -> float:
        """Unit of time 1/omega  [s]."""
        return 1.0 / self.omega

    @property
    def energy(self) -> float:
        """Unit of energy hbar omega  [J]."""
        return HBAR * self.omega

    def g1d_dimensionless(self, a_s: float, omega_perp: float,
                          n_atoms: float) -> float:
        """
        Dimensionless 1D coupling for a wavefunction normalized to 1:
        g~ = g_1D N / (hbar omega a_ho), with g_1D = 2 hbar omega_perp a_s.
        """
        return g_quasi_1d(a_s, omega_perp) * n_atoms / (self.energy * self.length)


@dataclass(frozen=True)
class SpinorUnits2D:
    """
    Natural units of 2D spin-1 spin-mixing dynamics, built from the quasi-2D
    spin coupling ``g_s_2d`` [J m^2] and the areal density ``n0`` [m^-2]:
    hbar = m = g_s n0 = 1. In these units the solver takes g_s = 1, n0 = 1,
    g_n = c0/c2, and q measured in units of g_s n0.
    """
    mass: float      # [kg]
    g_s_2d: float    # [J m^2]
    n0: float        # [m^-2]

    @property
    def energy(self) -> float:
        """Unit of energy: spin-interaction energy g_s n0  [J]."""
        return self.g_s_2d * self.n0

    @property
    def time(self) -> float:
        """Unit of time: spin time t_s = hbar / (g_s n0)  [s]."""
        return HBAR / self.energy

    @property
    def length(self) -> float:
        """Unit of length hbar / sqrt(m g_s n0)  [m] (~ spin healing length)."""
        return HBAR / np.sqrt(self.mass * self.energy)
