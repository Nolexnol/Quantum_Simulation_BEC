"""
Spin-domain formation in a quasi-2D antiferromagnetic 23Na spin-1 BEC.

The physical system: a uniform (box-trapped) sodium condensate, tightly
confined along z so the dynamics is two-dimensional, prepared in the polar
state (all atoms in mF = 0) and quenched to a *negative* quadratic Zeeman
energy q (experimentally reached with microwave dressing; a bare magnetic
field only gives q > 0, q/h = +276.8 Hz/G^2 x B^2 for Na). Inside the window
-2 g_s n0 < q < 0 the polar state is dynamically unstable and spin-mixing
collisions 2|mF=0> -> |mF=+1> + |mF=-1> amplify vacuum noise into magnetic
domains, which then coarsen (Williamson & Blakie, PRL 116, 025301 (2016)).

All laboratory parameters are set in SI units below and converted to the
dimensionless spin units the solver uses (hbar = m = g_s n0 = 1); plot axes
are labelled back in micrometres and milliseconds.
"""

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")  # non-interactive backend, safe for headless/scripted runs
import matplotlib.pyplot as plt

from quantum_sim import Spin1GPE2D
from quantum_sim.config import PATHS
from quantum_sim.units import (
    A_NA23_F0,
    A_NA23_F2,
    H_PLANCK,
    M_NA23,
    SpinorUnits2D,
    g_quasi_2d,
    spin1_couplings_3d,
)

RESULTS_DIR = str(PATHS.figures / "simulation")

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "figure.titlesize": 14,
})

# --------------------------------------------------------------------------- #
# Laboratory parameters (SI units)
# --------------------------------------------------------------------------- #
OMEGA_Z = 2 * np.pi * 2000.0    # tight z-confinement frequency [rad/s] (2 kHz)
N0_AREAL = 1.0e14               # mean areal density [atoms/m^2] (100 /um^2)
# Scattering lengths a_F=0 = 48.91 a_B, a_F=2 = 54.54 a_B imported from
# units.py [m] (Knoop et al., PRA 83, 042704 (2011)); a2 > a0 makes sodium
# antiferromagnetic (c2 > 0, polar ground state for q > 0).
Q_QUENCH_UNITS = -1.5           # quench target q [units of g_s n0]; must lie in
                                # (-2, 0) for the polar state to be unstable


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Convert laboratory parameters to solver (dimensionless) units ------
    c0_3d, c2_3d = spin1_couplings_3d(A_NA23_F0, A_NA23_F2, M_NA23)  # [J m^3]
    g_n_2d = g_quasi_2d(c0_3d, M_NA23, OMEGA_Z)   # density coupling [J m^2]
    g_s_2d = g_quasi_2d(c2_3d, M_NA23, OMEGA_Z)   # spin coupling [J m^2]

    units = SpinorUnits2D(mass=M_NA23, g_s_2d=g_s_2d, n0=N0_AREAL)
    spin_energy_hz = units.energy / H_PLANCK      # g_s n0 / h [Hz]
    t_s_ms = units.time * 1e3                     # spin time [ms]
    length_um = units.length * 1e6                # length unit [um]
    q_quench_hz = Q_QUENCH_UNITS * spin_energy_hz # quench target q/h [Hz]

    # Dimensionless solver parameters (hbar = m = g_s n0 = 1)
    n0 = 1.0                     # mean density [n0]
    g_s = 1.0                    # spin coupling [g_s n0 / n0]
    g_n = g_n_2d / g_s_2d        # density coupling: c0/c2 ~ 28 for sodium
    q_quench = Q_QUENCH_UNITS

    # Numerical grid
    N = 128                      # grid points per side
    L = 128.0                    # box length [units.length]; dx = 1.0 is fine
                                 # for the fastest-growing mode (k* ~ 1, i.e.
                                 # ~6 grid points per emergent domain)

    # Instability / domain scales (dimensionless)
    k_star = np.sqrt(2.0 * (-q_quench - g_s * n0))   # fastest-growing k
    domain_size = 2.0 * np.pi / k_star               # emergent domain scale

    print("=" * 60)
    print("Quasi-2D spin-1 SODIUM BEC: spin-domain formation")
    print("=" * 60)
    print(f"Confinement: omega_z = 2pi x {OMEGA_Z / (2 * np.pi):.0f} Hz,  "
          f"areal density n0 = {N0_AREAL * 1e-12:.0f} /um^2")
    print(f"Couplings: g_n = {g_n_2d:.3e} J m^2,  g_s = {g_s_2d:.3e} J m^2  "
          f"(g_n/g_s = {g_n:.1f})")
    print(f"Spin energy g_s n0 / h = {spin_energy_hz:.1f} Hz  ->  "
          f"time unit t_s = {t_s_ms:.2f} ms,  length unit = {length_um:.2f} um")
    print(f"Quench: q/h -> {q_quench_hz:.0f} Hz  ({q_quench} g_s n0; "
          f"unstable window {-2 * spin_energy_hz:.0f} Hz < q/h < 0)")
    print(f"Grid: {N}x{N},  box {L * length_um:.0f} um,  "
          f"dx = {L / N * length_um:.2f} um")
    print(f"Fastest-growing mode: domain size ~ {domain_size * length_um:.0f} um\n")

    solver = Spin1GPE2D(Nx=N, Ny=N, Lx=L, Ly=L, g_n=g_n, g_s=g_s, q=q_quench)
    solver.initialize_polar(n0=n0, noise=0.02, seed=12345)

    # --- Time evolution -----------------------------------------------------
    dt = 0.005                   # time step [t_s] (~17 us); keeps the stiff
                                 # density phase g_n n dt well below 1 rad
    t_end = 200.0                # total time [t_s] (~0.68 s)
    record_every = 20            # steps between population records (0.1 t_s)
    n_blocks = int(round(t_end / (dt * record_every)))
    snap_times = [0.0, 1.0, 2.0, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0,
                  10.0, 15.0, 25.0, 30.0, 50.0, 80.0, 90.0, 200.0]  # [t_s]
    snap_set = {round(t, 6) for t in snap_times}

    E0 = solver.energy()["total"]
    N0 = solver.total_norm()
    Mz0 = solver.total_magnetization()

    snapshots = {}   # t -> component-density RGB image
    times, frac_p1, frac_0, frac_m1 = [], [], [], []

    def record_snapshot(t):
        n1, ncomp0, nm1 = solver.component_densities()
        rgb = np.stack([n1, ncomp0, nm1], axis=-1)
        snapshots[t] = rgb / (rgb.max() + 1e-12)

    def record_populations(t):
        n1, ncomp0, nm1 = solver.component_densities()
        tot = (n1 + ncomp0 + nm1).sum()
        times.append(t)
        frac_p1.append(n1.sum() / tot)
        frac_0.append(ncomp0.sum() / tot)
        frac_m1.append(nm1.sum() / tot)

    record_snapshot(0.0)
    record_populations(0.0)

    # evolve() fuses adjacent kinetic half-steps between observation points,
    # halving the FFT work relative to per-step stepping.
    for block in range(1, n_blocks + 1):
        solver.evolve(dt, record_every)
        t = round(block * record_every * dt, 6)
        record_populations(t)
        if t in snap_set:
            record_snapshot(t)

    E1 = solver.energy()["total"]
    N1 = solver.total_norm()
    Mz1 = solver.total_magnetization()

    print("Conservation check (should be ~constant):")
    print(f"  Norm N:  {N0:.6f} -> {N1:.6f}   (rel. drift {abs(N1 - N0) / N0:.2e})")
    print(f"  Energy:  {E0:.5f} -> {E1:.5f}   (rel. drift {abs(E1 - E0) / abs(E0):.2e})")
    print(f"  M_z:     {Mz0:.3e} -> {Mz1:.3e}\n")

    # --- Plot: component-density domain snapshots (RGB) --------------------
    ordered = [t for t in snap_times if t in snapshots]
    n_snap = len(ordered)
    ncols = 5
    nrows = int(np.ceil(n_snap / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.3 * nrows))
    axes = np.atleast_2d(axes)
    half_box_um = L / 2 * length_um
    extent = [-half_box_um, half_box_um, -half_box_um, half_box_um]

    for idx, t in enumerate(ordered):
        ax = axes.ravel()[idx]
        ax.imshow(snapshots[t], extent=extent, origin="lower")
        ax.set_title(f"t = {t * t_s_ms:.0f} ms  ($t/t_s$ = {t:.1f})")
        ax.set_xticks([])
        ax.set_yticks([])

    # Blank any unused panels in the last row
    for ax in axes.ravel()[n_snap:]:
        ax.axis("off")

    axes.ravel()[0].set_ylabel(r"R = $m{+}1$   G = $m0$   B = $m{-}1$")

    fig.suptitle("Spin-domain formation & coarsening in a 2D spin-1 sodium BEC "
                 f"(quench $q/h \\to {q_quench_hz:.0f}$ Hz)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    domains_path = os.path.join(RESULTS_DIR, "sodium_spin_domains.png")
    fig.savefig(domains_path, dpi=140)
    plt.close(fig)
    print(f"Domain snapshots saved to '{domains_path}'")

    # --- Plot: component populations over time -----------------------------
    times_ms = np.array(times) * t_s_ms
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(times_ms, frac_0, "g-", label=r"$m_F=0$ (polar)")
    ax.plot(times_ms, frac_p1, "r-", label=r"$m_F=+1$")
    ax.plot(times_ms, frac_m1, "b--", label=r"$m_F=-1$")
    ax.set_xlabel(r"Time $t$ (ms)")
    ax.set_ylabel("Fractional population")
    ax.set_title("Spin-mixing: population transfer "
                 r"$2\,|m_F{=}0\rangle \to |m_F{=}{+}1\rangle + |m_F{=}{-}1\rangle$")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    pops_path = os.path.join(RESULTS_DIR, "sodium_populations.png")
    fig.savefig(pops_path, dpi=140)
    plt.close(fig)
    print(f"Population dynamics saved to '{pops_path}'")
    print("Done.")


if __name__ == "__main__":
    run()
