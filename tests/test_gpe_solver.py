"""
Correctness tests for the SpinorGPESolver.

Each test targets a regime where an independent analytical result is known, so
that a passing suite is real evidence the numerics are right (not just stable):

* Imaginary-time propagation -> exact harmonic-oscillator ground state.
* Real-time propagation -> norm (unitarity) and energy conservation.
* Pure Rabi coupling -> analytic sin^2 population transfer.
* Free Gaussian wavepacket -> analytic dispersive spreading of the variance.
* Variational principle -> energy decreases monotonically under ITP.
* Input validation and basic bookkeeping (normalization, populations).
"""

import numpy as np
import pytest

from quantum_sim import SpinorGPESolver


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def zero_potential(x):
    return np.zeros_like(x)


def variance(x, density, dx):
    """Position variance <x^2> - <x>^2 of a (normalized) density."""
    norm = np.sum(density) * dx
    mean = np.sum(x * density) * dx / norm
    mean_sq = np.sum(x ** 2 * density) * dx / norm
    return mean_sq - mean ** 2


# --------------------------------------------------------------------------- #
# 1. Ground state via imaginary-time propagation
# --------------------------------------------------------------------------- #
def test_itp_recovers_harmonic_ground_state():
    """Non-interacting ITP must reproduce psi(x) = pi^-1/4 exp(-x^2/2), E = 1/2."""
    solver = SpinorGPESolver(
        N=512, L=16.0,
        g_up_up=0.0, g_down_down=0.0, g_up_down=0.0, Omega=0.0,
    )
    solver.initialize_gaussian(w=2.0, spin_mix=(1.0, 0.0))

    for _ in range(500):
        solver.step(-1j * 0.02)

    x = solver.x
    analytic = np.pi ** (-0.25) * np.exp(-x ** 2 / 2.0)

    psi_up = solver.psi[0].copy()
    psi_up *= np.exp(-1j * np.angle(psi_up[solver.N // 2]))  # fix global phase
    l2_error = np.sqrt(np.sum(np.abs(psi_up - analytic) ** 2) * solver.dx)

    assert l2_error < 1e-4
    assert solver.calculate_energies()["total"] == pytest.approx(0.5, abs=1e-4)


# --------------------------------------------------------------------------- #
# 2. Unitarity: real-time propagation conserves the norm
# --------------------------------------------------------------------------- #
def test_rtp_conserves_norm():
    solver = SpinorGPESolver(
        N=256, L=20.0,
        g_up_up=15.0, g_down_down=12.0, g_up_down=8.0, Omega=1.3,
    )
    solver.initialize_gaussian(w=1.5, spin_mix=(1.0, 0.7))

    initial_norm = solver.get_norm()
    for _ in range(1000):
        solver.step(0.005)

    assert solver.get_norm() == pytest.approx(initial_norm, rel=1e-8)
    assert initial_norm == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# 3. Energy conservation under real-time evolution
# --------------------------------------------------------------------------- #
def test_rtp_conserves_energy():
    """Split-step is symplectic; total energy drift should be O(dt^2) and tiny."""
    solver = SpinorGPESolver(
        N=256, L=20.0,
        g_up_up=10.0, g_down_down=10.0, g_up_down=5.0, Omega=0.8,
    )
    solver.initialize_gaussian(w=1.2, spin_mix=(1.0, 1.0))

    e0 = solver.calculate_energies()["total"]
    for _ in range(2000):
        solver.step(0.004)
    e1 = solver.calculate_energies()["total"]

    assert abs(e1 - e0) / abs(e0) < 1e-3


# --------------------------------------------------------------------------- #
# 4. Rabi oscillations: analytic sin^2 population transfer
# --------------------------------------------------------------------------- #
def test_rabi_population_transfer():
    """With g=0, V=0 and only Rabi coupling, |psi_down|^2 = sin^2(Omega t) exactly.

    Kinetic evolution is spin-independent, so it cannot change the relative
    populations; the transfer is set purely by the coupling angle Omega*t.
    """
    Omega = 1.5
    dt = 0.002
    solver = SpinorGPESolver(
        N=256, L=20.0,
        g_up_up=0.0, g_down_down=0.0, g_up_down=0.0, Omega=Omega,
        V_ext=zero_potential,
    )
    solver.initialize_gaussian(w=1.5, spin_mix=(1.0, 0.0))  # start fully spin-up

    for n in range(1, 401):
        solver.step(dt)
        t = n * dt
        _, p_down = solver.get_populations()
        assert p_down == pytest.approx(np.sin(Omega * t) ** 2, abs=1e-6)


# --------------------------------------------------------------------------- #
# 5. Free Gaussian wavepacket: analytic dispersive spreading
# --------------------------------------------------------------------------- #
def test_free_gaussian_spreading():
    """A free Gaussian's position variance grows as sigma0^2 + t^2/(4 sigma0^2)."""
    w = 1.0
    solver = SpinorGPESolver(
        N=1024, L=60.0,
        g_up_up=0.0, g_down_down=0.0, g_up_down=0.0, Omega=0.0,
        V_ext=zero_potential,
    )
    solver.initialize_gaussian(w=w, spin_mix=(1.0, 0.0))

    sigma0_sq = w ** 2 / 2.0  # |psi|^2 ~ exp(-x^2/w^2) => variance = w^2/2
    dt, steps = 0.005, 400
    for _ in range(steps):
        solver.step(dt)
    t = dt * steps

    n_up, _ = solver.get_densities()
    numeric_var = variance(solver.x, n_up, solver.dx)
    analytic_var = sigma0_sq + t ** 2 / (4.0 * sigma0_sq)

    assert numeric_var == pytest.approx(analytic_var, rel=1e-3)


# --------------------------------------------------------------------------- #
# 6. Variational principle: ITP decreases energy monotonically
# --------------------------------------------------------------------------- #
def test_itp_energy_decreases_monotonically():
    solver = SpinorGPESolver(
        N=256, L=20.0,
        g_up_up=15.0, g_down_down=15.0, g_up_down=10.0, Omega=0.5,
    )
    solver.initialize_gaussian(w=1.5, spin_mix=(1.0, 1.0))

    energies = []
    for step in range(600):
        solver.step(-1j * 0.01)
        if step % 20 == 0:
            energies.append(solver.calculate_energies()["total"])

    diffs = np.diff(energies)
    assert np.all(diffs < 1e-9)  # non-increasing to within round-off


# --------------------------------------------------------------------------- #
# 7. Bookkeeping and input validation
# --------------------------------------------------------------------------- #
def test_normalize_gives_unit_norm():
    solver = SpinorGPESolver(N=128, L=10.0, g_up_up=0, g_down_down=0,
                             g_up_down=0, Omega=0)
    solver.initialize_gaussian(w=1.0, spin_mix=(0.6, 0.8))
    assert solver.get_norm() == pytest.approx(1.0, abs=1e-12)


def test_populations_sum_to_total_norm_squared():
    solver = SpinorGPESolver(N=128, L=10.0, g_up_up=1, g_down_down=1,
                             g_up_down=1, Omega=0.4)
    solver.initialize_gaussian(w=1.0, spin_mix=(1.0, 1.0))
    p_up, p_down = solver.get_populations()
    assert p_up + p_down == pytest.approx(1.0, abs=1e-12)
    assert p_up == pytest.approx(p_down, abs=1e-12)  # symmetric superposition


def test_set_wavefunction_rejects_wrong_shape():
    solver = SpinorGPESolver(N=128, L=10.0, g_up_up=0, g_down_down=0,
                             g_up_down=0, Omega=0)
    with pytest.raises(ValueError):
        solver.set_wavefunction(np.zeros((2, 64), dtype=complex))
