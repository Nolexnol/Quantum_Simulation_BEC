import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for headless/scripted runs
import matplotlib.pyplot as plt

# Allow running directly (``python scripts/run_simulation.py``) without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from quantum_sim import SpinorGPESolver

# Resolve output directory relative to the project root, independent of cwd
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

#Styling
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

def run_verification_analytical():
    """
    Verify the solver against the analytical ground state of a 1D harmonic oscillator:
    psi(x) = pi^(-1/4) * exp(-x^2 / 2)
    This corresponds to g = 0 and Omega = 0.
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
    steps = 500
    for _ in range(steps):
        solver.step(-1j * dtau)
        
    # Analytical ground state (spin-up component only)
    x = solver.x
    analytical_psi = np.pi**(-0.25) * np.exp(- (x ** 2) / 2.0)
    
    # Calculate difference
    numerical_psi_up = solver.psi[0]
    
    # Adjust global phase to match the analytical real solution
    phase_offset = np.angle(numerical_psi_up[N // 2])
    numerical_psi_up *= np.exp(-1j * phase_offset)
    
    l2_error = np.sqrt(np.sum(np.abs(numerical_psi_up - analytical_psi) ** 2) * solver.dx)
    
    # Verify kinetic energy
    energies = solver.calculate_energies()
    analytic_energy = 0.5  # E0 = 1/2 hbar * omega (in dimensionless units)
    
    print(f"L2 Wavefunction Error: {l2_error:.2e}")
    print(f"Numerical Energy:      {energies['total']:.6f}")
    print(f"Analytical Energy:     {analytic_energy:.6f}")
    print(f"Energy Absolute Error: {abs(energies['total'] - analytic_energy):.2e}")
    
    assert l2_error < 1e-4, "ITP ground state does not match analytical solution!"
    print("Verification successful! Solver meets accuracy standards.\n")

def run_full_simulation():
    """
    Simulate a coupled two-component BEC in a 1D harmonic trap.
    1. Run ITP to find the interacting coupled ground state with Rabi coupling.
    2. Shift the trap center and change Rabi coupling to trigger complex dynamics.
    3. Run RTP and record spatial densities and populations.
    4. Save visualization results to the 'results/' folder.
    """
    print("==================================================")
    print("Running Full Spinor BEC Simulation")
    print("==================================================")
    
    # Create results folder if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Parameters
    N = 256e6
    L = 20.0
    g_up_up = 15.0
    g_down_down = 20.0
    g_up_down = 10.0
    Omega_ground = 0.5
    
    print(f"Grid size: N={N}, Domain L={L}")
    print(f"Interaction Strengths: g_uu={g_up_up}, g_dd={g_down_down}, g_ud={g_up_down}")
    print(f"Initial Rabi coupling: Omega={Omega_ground}")
    
    # Initialize solver
    solver = SpinorGPESolver(
        N=N, L=L,
        g_up_up=g_up_up, g_down_down=g_down_down, g_up_down=g_up_down,
        Omega=Omega_ground
    )
    
    # Initialize with a symmetric spin superposition state
    solver.initialize_gaussian(w=1.5, spin_mix=(1.0, 1.0))
    
    # --- PHASE 1: Find Interacting Ground State via ITP ---
    print("\nPhase 1: Finding ground state using Imaginary Time Propagation (ITP)...")
    dtau = 0.01
    itp_steps = 1500
    energies_history = []
    
    for step in range(itp_steps):
        solver.step(-1j * dtau)
        if step % 10 == 0:
            eng = solver.calculate_energies()
            energies_history.append((step * dtau, eng["total"], eng["kinetic"], eng["potential"], eng["interaction"], eng["coupling"]))
            
    # Save the ground state wave function
    psi_ground = solver.psi.copy()
    g_up_dens, g_down_dens = solver.get_densities()
    print("Ground state search converged.")
    print(f"Final ground state energy: {solver.calculate_energies()['total']:.5f}")
    
    # Plot Energy Convergence
    energies_history = np.array(energies_history)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(energies_history[:, 0], energies_history[:, 1], 'k-', label='Total')
    ax1.plot(energies_history[:, 0], energies_history[:, 2], 'r--', label='Kinetic')
    ax1.plot(energies_history[:, 0], energies_history[:, 3], 'g--', label='Potential')
    ax1.plot(energies_history[:, 0], energies_history[:, 4], 'b--', label='Interaction')
    ax1.plot(energies_history[:, 0], energies_history[:, 5], 'm--', label='Coupling')
    ax1.set_xlabel(r"Imaginary Time $\tau$")
    ax1.set_ylabel(r"Energy ($\hbar\omega$)")
    ax1.set_title("Energy Convergence during ITP")
    ax1.legend()
    ax1.grid(True)
    
    # Plot Ground State Wavefunctions and Densities
    ax2.plot(solver.x, g_up_dens, 'r-', label=r'Spin-Up Density $n_\uparrow(x)$')
    ax2.plot(solver.x, g_down_dens, 'b--', label=r'Spin-Down Density $n_\downarrow(x)$')
    ax2.set_xlabel("Position $x$ ($a_{ho}$)")
    ax2.set_ylabel("Density")
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
    
    # Perturbation details:
    # 1. Suddenly shift the center of the harmonic trap from 0 to x_c = 1.5 to trigger spatial dipole oscillations (sloshing).
    # 2. Suddenly increase Rabi coupling to Omega = 2.0 to trigger rapid spin oscillations.
    x_c = 1.5
    solver.V_ext = 0.5 * ((solver.x - x_c) ** 2)
    solver.Omega = 2.0
    
    # Initialize wavefunction with the previously computed ground state
    solver.set_wavefunction(psi_ground)
    
    # Time evolution parameters
    dt = 0.005
    rtp_steps = 2000
    t_history = []
    pop_up_history = []
    pop_down_history = []
    
    # Matrices to store densities for 2D spatio-temporal plotting (streak plots)
    density_up_xt = []
    density_down_xt = []
    
    for step in range(rtp_steps):
        solver.step(dt)
        
        # Record population and densities
        t = step * dt
        p_up, p_down = solver.get_populations()
        n_up, n_down = solver.get_densities()
        
        t_history.append(t)
        pop_up_history.append(p_up)
        pop_down_history.append(p_down)
        
        if step % 5 == 0:  # sub-sample in time to optimize memory
            density_up_xt.append(n_up)
            density_down_xt.append(n_down)
            
    t_history = np.array(t_history)
    density_up_xt = np.array(density_up_xt)
    density_down_xt = np.array(density_down_xt)
    
    # --- Plotting the Dynamics ---
    fig = plt.figure(figsize=(14, 10))
    
    # 1. Spin populations over time
    ax_pop = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax_pop.plot(t_history, pop_up_history, 'r-', label='Spin-Up Fraction')
    ax_pop.plot(t_history, pop_down_history, 'b-', label='Spin-Down Fraction')
    ax_pop.set_xlabel(r"Real Time $t$ ($1/\omega$)")
    ax_pop.set_ylabel("Fractional Population")
    ax_pop.set_title("Spin Component Populations (Rabi Oscillations)")
    ax_pop.legend(loc='upper right')
    ax_pop.grid(True)
    ax_pop.set_xlim(0, t_history[-1])
    ax_pop.set_ylim(-0.05, 1.05)
    
    # 2. Spatio-temporal plot (Streak Plot) of Spin Up Density
    ax_up_xt = plt.subplot2grid((3, 2), (1, 0))
    im_up = ax_up_xt.imshow(
        density_up_xt, 
        extent=[solver.x[0], solver.x[-1], 0, t_history[-1]],
        aspect='auto', 
        origin='lower', 
        cmap='magma'
    )
    ax_up_xt.set_xlabel("Position $x$ ($a_{ho}$)")
    ax_up_xt.set_ylabel(r"Time $t$ ($1/\omega$)")
    ax_up_xt.set_title(r"Spin-Up Density Evolution $n_\uparrow(x, t)$")
    fig.colorbar(im_up, ax=ax_up_xt, label="Density")
    
    # 3. Spatio-temporal plot (Streak Plot) of Spin Down Density
    ax_down_xt = plt.subplot2grid((3, 2), (1, 1))
    im_down = ax_down_xt.imshow(
        density_down_xt, 
        extent=[solver.x[0], solver.x[-1], 0, t_history[-1]],
        aspect='auto', 
        origin='lower', 
        cmap='viridis'
    )
    ax_down_xt.set_xlabel("Position $x$ ($a_{ho}$)")
    ax_down_xt.set_ylabel(r"Time $t$ ($1/\omega$)")
    ax_down_xt.set_title(r"Spin-Down Density Evolution $n_\downarrow(x, t)$")
    fig.colorbar(im_down, ax=ax_down_xt, label="Density")
    
    # 4. Snapshots of the density at representative times
    ax_snap = plt.subplot2grid((3, 2), (2, 0), colspan=2)
    # Get snapshot indexes corresponding to t = 0, 3, 6, 9
    times_to_plot = [0.0, 3.0, 6.0, 9.0]
    colors = ['gray', 'orange', 'cyan', 'magenta']
    
    for t_val, col in zip(times_to_plot, colors):
        idx = np.argmin(np.abs(t_history[::5] - t_val))
        t_actual = t_history[::5][idx]
        ax_snap.plot(
            solver.x, 
            density_up_xt[idx] + density_down_xt[idx], 
            color=col, 
            label=f'Total density at t = {t_actual:.1f}'
        )
    ax_snap.set_xlabel("Position $x$ ($a_{ho}$)")
    ax_snap.set_ylabel("Total Density $n_{tot}(x)$")
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
