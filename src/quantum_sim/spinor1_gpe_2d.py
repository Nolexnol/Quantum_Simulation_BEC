import numpy as np
from typing import Optional, Tuple


class Spin1GPE2D:
    """
    Split-step Fourier solver for the quasi-2D spin-1 (three-component) Gross-Pitaevskii
    equations, following the energy functional of Williamson & Blakie,
    PRL 116, 025301 (2016):

        E = \\int d^2x [ psi^dag (p^2 / 2M) psi
                        + (g_n / 2) n^2
                        + (g_s / 2) |F|^2
                        + q  psi^dag f_z^2 psi ]

    The order parameter is the spinor psi = (psi_+1, psi_0, psi_-1), stored as an
    array of shape (3, Ny, Nx). The equations of motion i hbar d_t psi = dE/dpsi* are

        i hbar d_t psi = [ -hbar^2 nabla^2 / 2M + g_n n + q f_z^2 ] psi + g_s (F . f) psi

    where n = psi^dag psi is the total density and F = (F_x, F_y, F_z) is the spin
    density, F_mu = psi^dag f_mu psi, with f_mu the spin-1 matrices.

    Units are dimensionless: hbar = M = 1. The sign of g_s selects the magnetic
    physics:
        g_s < 0  -> ferromagnetic (e.g. 87Rb), the regime of the PRL.
        g_s > 0  -> antiferromagnetic / polar (e.g. 23Na, sodium). The polar state
                    (0, 1, 0) is the ground state for q > 0; quenching q into the
                    window -2 g_s n < q < 0 makes it dynamically unstable and drives
                    spin-domain formation via spin-mixing 2 psi_0 -> psi_+1 + psi_-1.

    The local (interaction + Zeeman) evolution is applied by holding the nonlinear
    coefficients (n, F) fixed over the step and Strang-splitting the on-site
    Hamiltonian as
        exp(-i g_n n dt) * exp(-i q f_z^2 dt/2) * exp(-i g_s F.f dt) * exp(-i q f_z^2 dt/2).
    The spin rotation exp(-i g_s F.f dt) is evaluated in closed form using the spin-1
    identity (n_hat . f)^3 = (n_hat . f), which avoids a per-grid-point diagonalisation.
    """

    INV_SQRT2 = 1.0 / np.sqrt(2.0)

    def __init__(
        self,
        Nx: int,
        Ny: int,
        Lx: float,
        Ly: float,
        g_n: float,
        g_s: float,
        q: float,
    ):
        """
        Args:
            Nx, Ny: grid points along x and y (powers of two are best for the FFT).
            Lx, Ly: physical box lengths; the grid is periodic on [-L/2, L/2).
            g_n: density (spin-independent) interaction strength.
            g_s: spin interaction strength. g_s > 0 for sodium (antiferromagnetic).
            q: quadratic Zeeman energy (tunable; the quench parameter).
        """
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        self.dx = Lx / Nx
        self.dy = Ly / Ny
        self.dA = self.dx * self.dy

        self.g_n = g_n
        self.g_s = g_s
        self.q = q

        # Real-space grid (periodic, endpoint excluded)
        x = np.linspace(-Lx / 2, Lx / 2, Nx, endpoint=False)
        y = np.linspace(-Ly / 2, Ly / 2, Ny, endpoint=False)
        self.X, self.Y = np.meshgrid(x, y, indexing="xy")  # shape (Ny, Nx)

        # Fourier grid and kinetic factor k^2 / 2  (hbar = M = 1)
        kx = 2 * np.pi * np.fft.fftfreq(Nx, self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(Ny, self.dy)
        KX, KY = np.meshgrid(kx, ky, indexing="xy")
        self.k2 = KX ** 2 + KY ** 2  # shape (Ny, Nx)

        # Spinor: index 0 -> m = +1, index 1 -> m = 0, index 2 -> m = -1
        self.psi = np.zeros((3, Ny, Nx), dtype=complex)

    # ------------------------------------------------------------------ #
    # State preparation
    # ------------------------------------------------------------------ #
    def initialize_polar(self, n0: float = 1.0, noise: float = 0.0,
                         seed: Optional[int] = None):
        """
        Prepare the polar condensate psi = sqrt(n0) (0, 1, 0) and optionally seed it
        with small complex Gaussian noise on every component (a simple stand-in for the
        vacuum fluctuations that trigger the dynamical instability).

        Args:
            n0: mean areal density of the m = 0 component.
            noise: standard deviation of the seeded field noise (per component, per
                   quadrature). A few percent of sqrt(n0) is enough to seed domains.
            seed: optional RNG seed for reproducibility.
        """
        self.psi[:] = 0.0
        self.psi[1] = np.sqrt(n0)

        if noise > 0.0:
            rng = np.random.default_rng(seed)
            shape = self.psi.shape
            fluct = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
            self.psi += noise * self.INV_SQRT2 * fluct

    def set_state(self, psi_data: np.ndarray):
        """Set the spinor directly; must have shape (3, Ny, Nx)."""
        if psi_data.shape != (3, self.Ny, self.Nx):
            raise ValueError(
                f"State must have shape (3, {self.Ny}, {self.Nx}), got {psi_data.shape}"
            )
        self.psi = psi_data.astype(complex)

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #
    def total_density(self) -> np.ndarray:
        """Total areal density n = |psi_+1|^2 + |psi_0|^2 + |psi_-1|^2."""
        return np.sum(np.abs(self.psi) ** 2, axis=0)

    def component_densities(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Densities (n_+1, n_0, n_-1) of the three magnetic components."""
        return (
            np.abs(self.psi[0]) ** 2,
            np.abs(self.psi[1]) ** 2,
            np.abs(self.psi[2]) ** 2,
        )

    def spin_density(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Cartesian spin density (F_x, F_y, F_z) with
            F_z = |psi_+1|^2 - |psi_-1|^2
            F_+ = F_x + i F_y = sqrt(2) (psi_+1^* psi_0 + psi_0^* psi_-1).
        """
        p1, p0, m1 = self.psi[0], self.psi[1], self.psi[2]
        Fz = np.abs(p1) ** 2 - np.abs(m1) ** 2
        Fp = np.sqrt(2.0) * (np.conjugate(p1) * p0 + np.conjugate(p0) * m1)
        return np.real(Fp), np.imag(Fp), Fz

    def total_norm(self) -> float:
        """Total atom number N = integral of n."""
        return float(np.sum(self.total_density()) * self.dA)

    def total_magnetization(self) -> float:
        """Longitudinal magnetization M_z = integral of F_z (conserved by the dynamics)."""
        _, _, Fz = self.spin_density()
        return float(np.sum(Fz) * self.dA)

    def energy(self) -> dict:
        """Total energy and its kinetic / density / spin / Zeeman contributions."""
        # Kinetic energy via Parseval: integral |grad psi|^2 / 2 = sum (k^2/2)|psihat|^2 * dA/(Nx Ny)
        E_kin = 0.0
        norm_fft = self.dA / (self.Nx * self.Ny)
        for m in range(3):
            psihat = np.fft.fft2(self.psi[m])
            E_kin += 0.5 * norm_fft * np.sum(self.k2 * np.abs(psihat) ** 2)

        n = self.total_density()
        Fx, Fy, Fz = self.spin_density()
        F2 = Fx ** 2 + Fy ** 2 + Fz ** 2

        E_den = 0.5 * self.g_n * np.sum(n ** 2) * self.dA
        E_spin = 0.5 * self.g_s * np.sum(F2) * self.dA
        n1, _, nm1 = self.component_densities()
        E_zee = self.q * np.sum(n1 + nm1) * self.dA

        E_total = E_kin + E_den + E_spin + E_zee
        return {
            "kinetic": float(E_kin),
            "density": float(E_den),
            "spin": float(E_spin),
            "zeeman": float(E_zee),
            "total": float(E_total),
        }

    # ------------------------------------------------------------------ #
    # Time stepping
    # ------------------------------------------------------------------ #
    def _kinetic_half_step(self, dt: complex):
        """Apply exp(-i (k^2/2) dt/2) to every component in Fourier space."""
        phase = np.exp(-0.5j * (self.k2 / 2.0) * dt)
        for m in range(3):
            self.psi[m] = np.fft.ifft2(np.fft.fft2(self.psi[m]) * phase)

    def _apply_spin_f(self, p1, p0, m1, Fz, Fp, Fm):
        """
        Return the three components of (F . f) acting on (p1, p0, m1), using the
        supplied (frozen) spin-density fields Fz, Fp = F_+, Fm = F_-:
            (F.f psi)_+1 = F_z psi_+1 + (1/sqrt2) F_- psi_0
            (F.f psi)_0  = (1/sqrt2) (F_+ psi_+1 + F_- psi_-1)
            (F.f psi)_-1 = -F_z psi_-1 + (1/sqrt2) F_+ psi_0
        """
        out_p1 = Fz * p1 + self.INV_SQRT2 * Fm * p0
        out_0 = self.INV_SQRT2 * (Fp * p1 + Fm * m1)
        out_m1 = -Fz * m1 + self.INV_SQRT2 * Fp * p0
        return out_p1, out_0, out_m1

    def _spin_rotation(self, dt: complex):
        """
        Apply exp(-i g_s (F . f) dt) in closed form, with F frozen at the substep input.
        For spin 1, M = (F.f)/|F| has eigenvalues {-1, 0, 1} and M^3 = M, so
            exp(-i theta M) = I - i sin(theta) M + (cos(theta) - 1) M^2 ,   theta = g_s |F| dt.
        Written via A1 = (F.f)psi and A2 = (F.f)^2 psi this is
            psi -> psi - i (sin theta / |F|) A1 + ((cos theta - 1)/|F|^2) A2,
        with both coefficients having finite |F| -> 0 limits (g_s dt and -(g_s dt)^2/2).
        """
        p1, p0, m1 = self.psi[0], self.psi[1], self.psi[2]

        Fz = np.abs(p1) ** 2 - np.abs(m1) ** 2
        Fp = np.sqrt(2.0) * (np.conjugate(p1) * p0 + np.conjugate(p0) * m1)
        Fm = np.conjugate(Fp)
        Fmag = np.sqrt(Fz ** 2 + np.abs(Fp) ** 2)

        # A1 = (F.f) psi ,  A2 = (F.f)^2 psi  (same frozen F for both applications)
        a1_p1, a1_0, a1_m1 = self._apply_spin_f(p1, p0, m1, Fz, Fp, Fm)
        a2_p1, a2_0, a2_m1 = self._apply_spin_f(a1_p1, a1_0, a1_m1, Fz, Fp, Fm)

        theta = self.g_s * Fmag * dt
        eps = 1e-12
        safe = Fmag > eps

        # c1 = sin(theta)/|F|  -> g_s dt ;  c2 = (cos(theta)-1)/|F|^2 -> -(g_s dt)^2/2
        c1 = np.where(safe, np.sin(theta) / np.where(safe, Fmag, 1.0), self.g_s * dt)
        c2 = np.where(
            safe,
            (np.cos(theta) - 1.0) / np.where(safe, Fmag ** 2, 1.0),
            -0.5 * (self.g_s * dt) ** 2,
        )

        self.psi[0] = p1 - 1j * c1 * a1_p1 + c2 * a2_p1
        self.psi[1] = p0 - 1j * c1 * a1_0 + c2 * a2_0
        self.psi[2] = m1 - 1j * c1 * a1_m1 + c2 * a2_m1

    def _local_step(self, dt: complex):
        """
        Full-dt on-site evolution exp(-i H_loc dt) with H_loc frozen at the input state:
            exp(-i g_n n dt) * exp(-i q f_z^2 dt/2) * exp(-i g_s F.f dt) * exp(-i q f_z^2 dt/2).
        The density phase is a global scalar (commutes with everything); the q term is
        diagonal, acting only on the m = +-1 components (f_z^2 = diag(1, 0, 1)).
        """
        # Density phase exp(-i g_n n dt) on all components
        n = self.total_density()
        phase_n = np.exp(-1j * self.g_n * n * dt)
        self.psi[0] *= phase_n
        self.psi[1] *= phase_n
        self.psi[2] *= phase_n

        # Half quadratic-Zeeman phase on m = +-1
        phase_q = np.exp(-0.5j * self.q * dt)
        self.psi[0] *= phase_q
        self.psi[2] *= phase_q

        # Spin rotation exp(-i g_s F.f dt)
        self._spin_rotation(dt)

        # Second half quadratic-Zeeman phase
        self.psi[0] *= phase_q
        self.psi[2] *= phase_q

    def step(self, dt: complex):
        """
        Advance one symmetric split-step of size dt (kinetic / local / kinetic).
        Pass a real dt for real-time quench dynamics.
        """
        self._kinetic_half_step(dt)
        self._local_step(dt)
        self._kinetic_half_step(dt)
