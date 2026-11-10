"""
Two-component (binary) GPE solver in 2D, split-step Fourier method.

The model is the coupled equations (S6)-(S7) for a binary condensate,
in dimensionless units (hbar = m = 1):

    i d_t psi_up   = [-1/2 nabla^2 + V_up   + g_uu |psi_up|^2   + g_ud |psi_down|^2] psi_up
    i d_t psi_down = [-1/2 nabla^2 + V_down + g_dd |psi_down|^2 + g_ud |psi_up|^2  ] psi_down

The chemical potential terms eps_up/down are dropped: there is no coherent
coupling between the components, so they only rotate each component's global
phase and don't affect any observable.
"""


import numpy as np


class BinaryGPE2D:
    """Binary GPE on a periodic 2D box. psi has shape (2, Ny, Nx)."""

    def __init__(
        self,
        Nx: int,
        Ny: int,
        Lx: float,
        Ly: float,
        g_uu: float,
        g_dd: float,
        g_ud: float,
        V_up: np.ndarray | None = None,
        V_down: np.ndarray | None = None,
        smooth_w: float | None = None,
    ):
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        self.dx = Lx / Nx
        self.dy = Ly / Ny
        self.dA = self.dx * self.dy

        self.g_uu = g_uu
        self.g_dd = g_dd
        self.g_ud = g_ud

        x = np.linspace(-Lx / 2, Lx / 2, Nx, endpoint=False)
        y = np.linspace(-Ly / 2, Ly / 2, Ny, endpoint=False)
        self.X, self.Y = np.meshgrid(x, y, indexing="xy")

        kx = 2 * np.pi * np.fft.fftfreq(Nx, self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(Ny, self.dy)
        KX, KY = np.meshgrid(kx, ky, indexing="xy")
        self.k2 = KX**2 + KY**2

        # zero potential = uniform box with periodic edges
        self.V_up = np.zeros((Ny, Nx)) if V_up is None else np.asarray(V_up, dtype=float)
        self.V_down = np.zeros((Ny, Nx)) if V_down is None else np.asarray(V_down, dtype=float)

        # optionally smooth the potentials with a Gaussian of width smooth_w
        if smooth_w is not None and smooth_w > 0.0:
            self.V_up = self._gaussian_smooth(self.V_up, smooth_w)
            self.V_down = self._gaussian_smooth(self.V_down, smooth_w)

        self.psi = np.zeros((2, Ny, Nx), dtype=complex)

        # cache the kinetic phases so we don't redo the exp() every step
        self._kin_dt = None
        self._kin_half = None
        self._kin_full = None

    def _gaussian_smooth(self, V: np.ndarray, w: float) -> np.ndarray:
        """Convolve V with a unit-normalized Gaussian of width w, respecting
        the periodic boundaries (the kernel in Fourier space is exp(-k^2 w^2 / 2))."""
        return np.fft.ifft2(np.fft.fft2(V) * np.exp(-0.5 * self.k2 * w**2)).real

    def initialize_mixture(self, n0: float, noise: float = 0.0,
                           seed: int | None = None, n_modes: int = 64,
                           max_mode: int = 32):
        """Uniform 50/50 mixture at total density n0, plus small noise to
        seed the demixing instability.

        The noise is a superposition of n_modes cosine waves with random
        wavevectors (integer mode numbers up to max_mode, so the perturbation
        respects the periodic box) and random phases, normalized to unit
        standard deviation. seed=None draws a fresh realization each run.
        """
        self.psi[:] = np.sqrt(n0 / 2)
        if noise > 0.0:
            rng = np.random.default_rng(seed)
            for c in range(2):
                fluct = np.zeros((self.Ny, self.Nx))
                mx = rng.integers(-max_mode, max_mode + 1, n_modes)
                my = rng.integers(-max_mode, max_mode + 1, n_modes)
                phase = 2 * np.pi * rng.random(n_modes)
                for mxi, myi, ph in zip(mx, my, phase):
                    if mxi == 0 and myi == 0:
                        continue  # skip the uniform mode
                    kx = 2 * np.pi * mxi / self.Lx
                    ky = 2 * np.pi * myi / self.Ly
                    fluct += np.cos(kx * self.X + ky * self.Y + ph)
                fluct /= fluct.std()
                self.psi[c] += noise * fluct

    def set_state(self, psi_data: np.ndarray):
        if psi_data.shape != (2, self.Ny, self.Nx):
            raise ValueError(f"State must have shape (2, {self.Ny}, {self.Nx})")
        self.psi = psi_data.astype(complex)


    def densities(self) -> tuple[np.ndarray, np.ndarray]:
        return np.abs(self.psi[0])**2, np.abs(self.psi[1])**2

    def norms(self) -> tuple[float, float]:
        """Atom number in each component (both conserved separately)."""
        n_up, n_down = self.densities()
        return float(n_up.sum() * self.dA), float(n_down.sum() * self.dA)

    def magnetization(self) -> np.ndarray:
        """Local magnetization z = (n_up - n_down) / n, in [-1, 1]."""
        n_up, n_down = self.densities()
        return (n_up - n_down) / (n_up + n_down + 1e-12)

    def energy(self) -> dict:
        psihat = np.fft.fft2(self.psi, axes=(1, 2))
        norm_fft = self.dA / (self.Nx * self.Ny)
        E_kin = 0.5 * norm_fft * np.sum(self.k2 * np.abs(psihat)**2)

        n_up, n_down = self.densities()
        E_pot = np.sum(self.V_up * n_up + self.V_down * n_down) * self.dA
        E_int = (0.5 * self.g_uu * np.sum(n_up**2)
                 + 0.5 * self.g_dd * np.sum(n_down**2)
                 + self.g_ud * np.sum(n_up * n_down)) * self.dA

        return {
            "kinetic": float(E_kin),
            "potential": float(E_pot),
            "interaction": float(E_int),
            "total": float(E_kin + E_pot + E_int),
        }

    def _kinetic_phases(self, dt: float):
        if dt != self._kin_dt:
            self._kin_half = np.exp(-0.25j * self.k2 * dt)
            self._kin_full = self._kin_half**2
            self._kin_dt = dt
        return self._kin_half, self._kin_full

    def _apply_kinetic(self, phase: np.ndarray):
        psihat = np.fft.fft2(self.psi, axes=(1, 2))
        psihat *= phase
        self.psi = np.fft.ifft2(psihat, axes=(1, 2))

    def _local_step(self, dt: float):
        # potential + mean-field phase, densities frozen over the step
        n_up, n_down = self.densities()
        self.psi[0] *= np.exp(-1j * dt * (self.V_up + self.g_uu * n_up + self.g_ud * n_down))
        self.psi[1] *= np.exp(-1j * dt * (self.V_down + self.g_dd * n_down + self.g_ud * n_up))

    def step(self, dt: float):
        """One symmetric split step: kinetic half / local full / kinetic half."""
        half, _ = self._kinetic_phases(dt)
        self._apply_kinetic(half)
        self._local_step(dt)
        self._apply_kinetic(half)

    def evolve(self, dt: float, n_steps: int):
        """Same as calling step() n_steps times, but merges the back-to-back
        kinetic half steps so it does about half the FFT work."""
        if n_steps <= 0:
            return
        half, full = self._kinetic_phases(dt)
        self._apply_kinetic(half)
        self._local_step(dt)
        for _ in range(n_steps - 1):
            self._apply_kinetic(full)
            self._local_step(dt)
        self._apply_kinetic(half)
