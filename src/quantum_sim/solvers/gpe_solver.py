"""
Split-step Fourier solver for a two-component (spin-1/2) Gross-Pitaevskii
equation in one dimension.

The coupled GPE solved here (dimensionless harmonic-oscillator units,
hbar = m = omega_x = 1; see ``units.py`` for the mapping to SI) is

    i d_t psi_up   = [ -1/2 d_xx + V(x) + g_uu |psi_up|^2 + g_ud |psi_down|^2 ] psi_up
                     - Omega psi_down
    i d_t psi_down = [ -1/2 d_xx + V(x) + g_dd |psi_down|^2 + g_ud |psi_up|^2 ] psi_down
                     - Omega psi_up

i.e. the Rabi coupling enters as H_c = -Omega sigma_x. With the wavefunction
normalized to 1, the couplings g_ij absorb the atom number
(g_ij = g_1D^ij N_atoms / (hbar omega_x a_ho); see ``units.py``).

Time stepping is a symmetric (Strang) split step,

    exp(-i H dt) ~ K(dt/2) * D(dt/2) * C(dt) * D(dt/2) * K(dt/2),

with K the kinetic propagator applied in Fourier space, D the diagonal
potential + interaction phase (density frozen over the step), and C the exact
2x2 Rabi rotation exp(+i Omega dt sigma_x). Passing an imaginary dt = -i dtau
performs imaginary-time propagation (gradient flow toward the ground state);
the norm is restored after every imaginary-time step.
"""

from collections.abc import Callable

import numpy as np


class SpinorGPESolver:
    """Two-component 1D GPE solver on a periodic grid of N points, length L."""

    def __init__(
        self,
        N: int,
        L: float,
        g_up_up: float,
        g_down_down: float,
        g_up_down: float,
        #TODO: Remove magnetic field term
        Omega: float,
        V_ext: Callable[[np.ndarray], np.ndarray] | np.ndarray | None = None,
    ):
        """
        Args:
            N: number of grid points (powers of two are best for the FFT).
            L: box length; the periodic grid covers [-L/2, L/2).
            g_up_up, g_down_down: intra-component interaction strengths.
            g_up_down: inter-component interaction strength.
            Omega: Rabi coupling; populations transfer as sin^2(Omega t).
            V_ext: external potential, either a callable V(x) or an array of
                length N. Defaults to the harmonic trap V = x^2 / 2.
        """
        self.N = int(N)
        self.L = float(L)
        self.dx = self.L / self.N
        self.x = np.linspace(-self.L / 2, self.L / 2, self.N, endpoint=False)

        self.g_up_up = g_up_up
        self.g_down_down = g_down_down
        self.g_up_down = g_up_down
        self.Omega = Omega

        if V_ext is None:
            self.V_ext = 0.5 * self.x**2
        elif callable(V_ext):
            self.V_ext = np.asarray(V_ext(self.x), dtype=float)
        else:
            self.V_ext = np.asarray(V_ext, dtype=float)

        k = 2 * np.pi * np.fft.fftfreq(self.N, self.dx)
        self.k2 = k**2

        # Spinor wavefunction: index 0 -> spin-up, index 1 -> spin-down.
        self.psi = np.zeros((2, self.N), dtype=complex)

        # Cached kinetic phase, reused while dt is unchanged (the loop-invariant
        # exp() over the grid is otherwise the dominant per-step cost).
        self._kin_dt: complex | None = None
        self._kin_phase: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # State preparation
    # ------------------------------------------------------------------ #
    def initialize_gaussian(self, w: float, spin_mix: tuple[float, float]):
        """
        Prepare both components as a Gaussian envelope exp(-x^2 / (2 w^2))
        weighted by ``spin_mix`` and normalize the total norm to 1.

        The resulting density |psi|^2 ~ exp(-x^2 / w^2) has variance w^2 / 2.
        """
        envelope = np.exp(-self.x**2 / (2.0 * w**2))
        self.psi[0] = spin_mix[0] * envelope
        self.psi[1] = spin_mix[1] * envelope
        self._normalize()

    def set_wavefunction(self, psi_data: np.ndarray):
        """Set the spinor directly; must have shape (2, N)."""
        psi_data = np.asarray(psi_data)
        if psi_data.shape != (2, self.N):
            raise ValueError(
                f"Wavefunction must have shape (2, {self.N}), got {psi_data.shape}"
            )
        self.psi = psi_data.astype(complex)

    def _normalize(self):
        self.psi /= np.sqrt(self.get_norm())

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #
    def get_densities(self) -> tuple[np.ndarray, np.ndarray]:
        """Component densities (n_up(x), n_down(x))."""
        return np.abs(self.psi[0]) ** 2, np.abs(self.psi[1]) ** 2

    def get_norm(self) -> float:
        """Total norm integral (|psi_up|^2 + |psi_down|^2) dx."""
        return float(np.sum(np.abs(self.psi) ** 2) * self.dx)

    def get_populations(self) -> tuple[float, float]:
        """Fractional populations (p_up, p_down), p_up + p_down = 1."""
        n_up, n_down = self.get_densities()
        tot = np.sum(n_up) + np.sum(n_down)
        return float(np.sum(n_up) / tot), float(np.sum(n_down) / tot)

    def calculate_energies(self) -> dict[str, float]:
        """Total energy and its kinetic/potential/interaction/coupling parts."""
        # Kinetic energy via Parseval: sum over k of (k^2/2)|psihat|^2 * dx/N.
        psihat = np.fft.fft(self.psi, axis=1)
        E_kin = 0.5 * (self.dx / self.N) * np.sum(self.k2 * np.abs(psihat) ** 2)

        n_up, n_down = self.get_densities()
        E_pot = np.sum(self.V_ext * (n_up + n_down)) * self.dx
        E_int = (
            0.5 * self.g_up_up * np.sum(n_up**2)
            + 0.5 * self.g_down_down * np.sum(n_down**2)
            + self.g_up_down * np.sum(n_up * n_down)
        ) * self.dx
        # H_c = -Omega sigma_x  =>  E_c = -2 Omega Re integral psi_up* psi_down.
        E_cpl = -2.0 * self.Omega * float(
            np.real(np.sum(np.conjugate(self.psi[0]) * self.psi[1]))
        ) * self.dx

        return {
            "kinetic": float(E_kin),
            "potential": float(E_pot),
            "interaction": float(E_int),
            "coupling": float(E_cpl),
            "total": float(E_kin + E_pot + E_int + E_cpl),
        }

    # ------------------------------------------------------------------ #
    # Time stepping
    # ------------------------------------------------------------------ #
    def _kinetic_half_step(self, dt: complex):
        """Apply exp(-i (k^2/2) dt/2) to both components in Fourier space."""
        if dt != self._kin_dt:
            self._kin_phase = np.exp(-0.25j * self.k2 * dt)
            self._kin_dt = dt
        psihat = np.fft.fft(self.psi, axis=1)
        psihat *= self._kin_phase
        self.psi = np.fft.ifft(psihat, axis=1)

    def _local_step(self, dt: complex):
        """
        Full-dt on-site evolution with the density frozen at the step input:
        half diagonal phase, exact Rabi rotation, half diagonal phase.
        """
        n_up, n_down = self.get_densities()
        half_up = np.exp(
            -0.5j * dt * (self.V_ext + self.g_up_up * n_up + self.g_up_down * n_down)
        )
        half_down = np.exp(
            -0.5j * dt * (self.V_ext + self.g_down_down * n_down + self.g_up_down * n_up)
        )

        self.psi[0] *= half_up
        self.psi[1] *= half_down

        # Exact rotation exp(+i Omega dt sigma_x); cos/sin handle complex dt.
        c = np.cos(self.Omega * dt)
        s = 1j * np.sin(self.Omega * dt)
        up = c * self.psi[0] + s * self.psi[1]
        down = s * self.psi[0] + c * self.psi[1]
        self.psi[0], self.psi[1] = up, down

        self.psi[0] *= half_up
        self.psi[1] *= half_down

    def step(self, dt: complex):
        """
        Advance one symmetric split step of size dt.

        Real dt: unitary real-time propagation. Imaginary dt = -i dtau:
        imaginary-time propagation; the norm is restored after the step.
        """
        dt = complex(dt)
        self._kinetic_half_step(dt)
        self._local_step(dt)
        self._kinetic_half_step(dt)
        if dt.imag != 0.0:
            self._normalize()
