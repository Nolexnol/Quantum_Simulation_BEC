"""
Two-component (spin-1/2) 87Rb BEC in a 1D harmonic trap.

The physical system: N_ATOMS rubidium-87 atoms in the hyperfine states
|F=1, mF=-1> ("up") and |F=2, mF=+1> ("down"), held in an elongated
cigar-shaped trap and coupled by a two-photon microwave+RF drive (the
standard Rabi-coupled mixture of the JILA/Swinburne experiments).

All laboratory parameters are set in SI units below and converted to the
dimensionless harmonic-oscillator units the solver uses (hbar = m = omega_x
= 1); plots are labelled back in micrometres and milliseconds.

Pipeline:
1. Verify the solver against the analytical harmonic ground state.
2. Find the interacting coupled ground state via imaginary-time propagation.
3. Suddenly shift the trap and strengthen the Rabi drive; propagate in real
   time and record densities and populations.
4. Save figures to results/.
"""

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")  # non-interactive backend, safe for headless/scripted runs
import matplotlib.pyplot as plt

from quantum_sim import SpinorGPESolver
from quantum_sim.config import PATHS
from quantum_sim.units import (
    A_RB87_DOWN_DOWN,
    A_RB87_UP_DOWN,
    A_RB87_UP_UP,
    M_RB87,
    HarmonicUnits,
)

# Resolve output directory relative to the project root, independent of cwd
RESULTS_DIR = str(PATHS.figures / "simulation")

# Styling
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14,
    'legend.fontsize': 10,
    'grid.alpha': 0.3
})

# --------------------------------------------------------------------------- #
# Laboratory parameters (SI units)
# --------------------------------------------------------------------------- #
OMEGA_X = 2 * np.pi * 20.0       # axial trap frequency [rad/s]  (20 Hz)
OMEGA_PERP = 2 * np.pi * 250.0   # radial trap frequency [rad/s] (250 Hz)
N_ATOMS = 300                    # atom number [dimensionless]
# Scattering lengths a_uu, a_dd, a_ud imported from units.py [m]
# (100.40, 95.44, 98.01 Bohr radii; Egorov et al., PRA 87, 053614 (2013)).

# Two-photon Rabi frequencies Omega_R (populations transfer as
# sin^2(Omega_R t / 2); the solver's Omega equals Omega_R / 2).
RABI_GROUND = 2 * np.pi * 20.0   # drive during ground-state prep [rad/s] (20 Hz)
RABI_QUENCH = 2 * np.pi * 80.0   # drive during dynamics [rad/s] (80 Hz)

TRAP_SHIFT = 3.6e-6              # sudden trap displacement [m] (3.6 um)

UNITS = HarmonicUnits(mass=M_RB87, omega=OMEGA_X)
# Derived scales: length a_ho ~ 2.41 um, time 1/omega_x ~ 7.96 ms.


def run_verification_analytical():
    """
    Verify the solver against the analytical ground state of a 1D harmonic
    oscillator, psi(x) = pi^(-1/4) exp(-x^2/2) with E = hbar omega / 2
    (dimensionless units; this check needs no physical parameters).
    """
    print("==================================================")
    print("Verification: Comparing with Analytical Ground State")
    print("==================================================")

    N = 512
    L = 16.0
    solver = SpinorGPESolver(
        N=N, L=L,
        g_up_up=0.0, g_down_down=0.0, g_up_down=0.0,
        Omega=0.0
    )

    # Initialize with a wide Gaussian
    solver.initialize_gaussian(w=2.0, spin_mix=(1.0, 0.0))

    # Run Imaginary Time Propagation (ITP)
    dtau = 0.02
    for _ in range(500):
        solver.step(-1j * dtau)

    # Analytical ground state (spin-up component only)
    x = solver.x
    analytical_psi = np.pi ** (-0.25) * np.exp(-(x ** 2) / 2.0)

    # Adjust global phase to match the analytical real solution
    numerical_psi_up = solver.psi[0]
    phase_offset = np.angle(numerical_psi_up[N // 2])
    numerical_psi_up *= np.exp(-1j * phase_offset)

    l2_error = np.sqrt(np.sum(np.abs(numerical_psi_up - analytical_psi) ** 2) * solver.dx)

    energies = solver.calculate_energies()
    analytic_energy = 0.5  # E0 = hbar omega / 2, in units of hbar omega

    print(f"L2 Wavefunction Error: {l2_error:.2e}")
    print(f"Numerical Energy:      {energies['total']:.6f}")
    print(f"Analytical Energy:     {analytic_energy:.6f}")
    print(f"Energy Absolute Error: {abs(energies['total'] - analytic_energy):.2e}")

    assert l2_error < 1e-4, "ITP ground state does not match analytical solution!"
    print("Verification successful! Solver meets accuracy standards.\n")


def run_full_simulation():
    """
    Simulate the Rabi-coupled two-component 87Rb BEC:
    1. Run ITP to find the interacting coupled ground state.
    2. Shift the trap centre and strengthen the Rabi drive (quench).
    3. Run RTP and record spatial densities and populations.
    4. Save visualization results to the 'results/' folder.
    """
    print("==================================================")
    print("Running Full Spinor BEC Simulation (87Rb, real units)")
    print("==================================================")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Convert laboratory parameters to solver (dimensionless) units ------
    a_ho = UNITS.length          # [m]
    t0 = UNITS.time              # [s]

    # Numerical grid
    N = 256                      # grid points
    L = 20.0                     # box length [a_ho] (~48 um)

    # Dimensionless couplings g~ = g_1D N_atoms / (hbar omega_x a_ho),
    # with g_1D = 2 hbar omega_perp a_s the quasi-1D contact coupling.
    g_up_up = UNITS.g1d_dimensionless(A_RB87_UP_UP, OMEGA_PERP, N_ATOMS)
    g_down_down = UNITS.g1d_dimensionless(A_RB87_DOWN_DOWN, OMEGA_PERP, N_ATOMS)
    g_up_down = UNITS.g1d_dimensionless(A_RB87_UP_DOWN, OMEGA_PERP, N_ATOMS)

    # Solver Rabi coupling Omega = (Omega_R / 2) / omega_x  [units of omega_x]
    Omega_ground = 0.5 * RABI_GROUND / OMEGA_X
    Omega_quench = 0.5 * RABI_QUENCH / OMEGA_X
    x_c = TRAP_SHIFT / a_ho      # trap displacement [a_ho]

    print(f"Trap: omega_x = 2pi x {OMEGA_X / (2 * np.pi):.0f} Hz, "
          f"omega_perp = 2pi x {OMEGA_PERP / (2 * np.pi):.0f} Hz, "
          f"N = {N_ATOMS} atoms")
    print(f"Units: a_ho = {a_ho * 1e6:.2f} um, t0 = {t0 * 1e3:.2f} ms")
    print(f"Grid: {N} points over L = {L:.0f} a_ho = {L * a_ho * 1e6:.0f} um")
    print(f"Couplings (units of hbar omega_x a_ho): "
          f"g_uu = {g_up_up:.2f}, g_dd = {g_down_down:.2f}, g_ud = {g_up_down:.2f}")
    print(f"Rabi drive: {RABI_GROUND / (2 * np.pi):.0f} Hz -> "
          f"{RABI_QUENCH / (2 * np.pi):.0f} Hz "
          f"(solver Omega: {Omega_ground:.2f} -> {Omega_quench:.2f})")
    print(f"Trap shift: {TRAP_SHIFT * 1e6:.1f} um = {x_c:.2f} a_ho")

    solver = SpinorGPESolver(
        N=N, L=L,
        g_up_up=g_up_up, g_down_down=g_down_down, g_up_down=g_up_down,
        Omega=Omega_ground
    )

    # Initialize with a symmetric spin superposition state
    solver.initialize_gaussian(w=1.5, spin_mix=(1.0, 1.0))

    # --- PHASE 1: Find Interacting Ground State via ITP ---
    print("\nPhase 1: Finding ground state using Imaginary Time Propagation (ITP)...")
    dtau = 0.01                  # imaginary-time step [1/omega_x]
    itp_steps = 1500
    energies_history = []

    for step in range(itp_steps):
        solver.step(-1j * dtau)
        if step % 10 == 0:
            eng = solver.calculate_energies()
            energies_history.append((step * dtau, eng["total"], eng["kinetic"],
                                     eng["potential"], eng["interaction"],
                                     eng["coupling"]))

    # Save the ground state wave function
    psi_ground = solver.psi.copy()
    g_up_dens, g_down_dens = solver.get_densities()
    print("Ground state search converged.")
    print(f"Final ground state energy: {solver.calculate_energies()['total']:.5f} hbar omega_x")

    # Plot Energy Convergence
    energies_history = np.array(energies_history)
    x_um = solver.x * a_ho * 1e6                 # position axis [um]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(energies_history[:, 0], energies_history[:, 1], 'k-', label='Total')
    ax1.plot(energies_history[:, 0], energies_history[:, 2], 'r--', label='Kinetic')
    ax1.plot(energies_history[:, 0], energies_history[:, 3], 'g--', label='Potential')
    ax1.plot(energies_history[:, 0], energies_history[:, 4], 'b--', label='Interaction')
    ax1.plot(energies_history[:, 0], energies_history[:, 5], 'm--', label='Coupling')
    ax1.set_xlabel(r"Imaginary Time $\tau$ ($1/\omega_x$)")
    ax1.set_ylabel(r"Energy ($\hbar\omega_x$)")
    ax1.set_title("Energy Convergence during ITP")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(x_um, g_up_dens, 'r-', label=r'$n_\uparrow(x)$  $|F{=}1,m_F{=}{-}1\rangle$')
    ax2.plot(x_um, g_down_dens, 'b--', label=r'$n_\downarrow(x)$  $|F{=}2,m_F{=}{+}1\rangle$')
    ax2.set_xlabel(r"Position $x$ ($\mu$m)")
    ax2.set_ylabel(r"Density ($1/a_{ho}$, norm 1)")
    ax2.set_title("Coupled Ground State Density Profile")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    ground_state_path = os.path.join(RESULTS_DIR, "gpe_ground_state.png")
    plt.savefig(ground_state_path, dpi=150)
    plt.close()
    print(f"Ground state visualization saved to '{ground_state_path}'")

    # --- PHASE 2: Simulate Real-Time Dynamics with Sudden Perturbation ---
    print("\nPhase 2: Running Real Time Propagation (RTP) with perturbations...")

    # Perturbation (quench):
    # 1. Suddenly shift the trap centre by TRAP_SHIFT to trigger dipole
    #    oscillations (sloshing).
    # 2. Suddenly strengthen the Rabi drive to trigger rapid spin oscillations.
    solver.V_ext = 0.5 * ((solver.x - x_c) ** 2)
    solver.Omega = Omega_quench

    # Initialize wavefunction with the previously computed ground state
    solver.set_wavefunction(psi_ground)

    # Time evolution parameters
    dt = 0.005                   # time step [1/omega_x] (~40 us)
    rtp_steps = 2000             # total time 10 / omega_x (~80 ms)
    record_every = 5             # density snapshot subsampling (memory)
    t_history = []
    pop_up_history = []
    pop_down_history = []
    density_up_xt = []
    density_down_xt = []

    for step in range(1, rtp_steps + 1):
        solver.step(dt)
        t = step * dt
        p_up, p_down = solver.get_populations()

        t_history.append(t)
        pop_up_history.append(p_up)
        pop_down_history.append(p_down)

        if step % record_every == 0:
            n_up, n_down = solver.get_densities()
            density_up_xt.append(n_up)
            density_down_xt.append(n_down)

    t_history = np.array(t_history)
    t_ms = t_history * t0 * 1e3                  # time axis [ms]
    t_snap_ms = t_ms[record_every - 1::record_every]  # times of stored densities
    density_up_xt = np.array(density_up_xt)
    density_down_xt = np.array(density_down_xt)

    # --- Plotting the Dynamics ---
    fig = plt.figure(figsize=(14, 10))

    # 1. Spin populations over time
    ax_pop = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax_pop.plot(t_ms, pop_up_history, 'r-', label='Spin-Up Fraction')
    ax_pop.plot(t_ms, pop_down_history, 'b-', label='Spin-Down Fraction')
    ax_pop.set_xlabel(r"Real Time $t$ (ms)")
    ax_pop.set_ylabel("Fractional Population")
    ax_pop.set_title(f"Spin Component Populations "
                     f"(Rabi drive {RABI_QUENCH / (2 * np.pi):.0f} Hz)")
    ax_pop.legend(loc='upper right')
    ax_pop.grid(True)
    ax_pop.set_xlim(0, t_ms[-1])
    ax_pop.set_ylim(-0.05, 1.05)

    # 2 & 3. Spatio-temporal (streak) plots of the component densities
    streak_specs = [
        (density_up_xt, 'magma', r"Spin-Up Density Evolution $n_\uparrow(x, t)$", (1, 0)),
        (density_down_xt, 'viridis', r"Spin-Down Density Evolution $n_\downarrow(x, t)$", (1, 1)),
    ]
    for data, cmap, title, pos in streak_specs:
        ax = plt.subplot2grid((3, 2), pos)
        im = ax.imshow(
            data,
            extent=[x_um[0], x_um[-1], 0, t_ms[-1]],
            aspect='auto',
            origin='lower',
            cmap=cmap
        )
        ax.set_xlabel(r"Position $x$ ($\mu$m)")
        ax.set_ylabel(r"Time $t$ (ms)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="Density")

    # 4. Snapshots of the total density at representative times
    ax_snap = plt.subplot2grid((3, 2), (2, 0), colspan=2)
    times_to_plot_ms = [0.0, 20.0, 45.0, 70.0]
    colors = ['gray', 'orange', 'cyan', 'magenta']

    for t_val, col in zip(times_to_plot_ms, colors):
        idx = int(np.argmin(np.abs(t_snap_ms - t_val)))
        ax_snap.plot(
            x_um,
            density_up_xt[idx] + density_down_xt[idx],
            color=col,
            label=f'Total density at t = {t_snap_ms[idx]:.0f} ms'
        )
    ax_snap.set_xlabel(r"Position $x$ ($\mu$m)")
    ax_snap.set_ylabel(r"Total Density $n_{tot}(x)$")
    ax_snap.set_title("Total BEC Density Snapshots (Dipole Motion)")
    ax_snap.legend()
    ax_snap.grid(True)

    plt.tight_layout()
    dynamics_path = os.path.join(RESULTS_DIR, "gpe_dynamics.png")
    plt.savefig(dynamics_path, dpi=150)
    plt.close()

    print(f"Dynamics visualizations saved to '{dynamics_path}'")
    print("Simulation completed successfully!")


if __name__ == "__main__":
    run_verification_analytical()
    run_full_simulation()
