"""
Phase separation and coarsening in an immiscible binary condensate.

A uniform 50/50 mixture of two components is dynamically unstable when
g_ud > sqrt(g_uu g_dd). Small noise gets amplified into magnetic domains
(demixing), and the domains then grow in time (coarsening).

Everything is dimensionless: hbar = m = 1, total density n0 = 1, and
g n0 = 1 so the density healing length xi = 1 and time is in units of
hbar / (g n0). The interspecies coupling is picked to match the length
scale ratio of the experiment, xi_s / xi ~ 3.3 (0.6 um vs 2 um).
"""

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantum_sim import BinaryGPE2D
from quantum_sim.config import PATHS
from quantum_sim.units import HBAR, M_RB87

RESULTS_DIR = str(PATHS.figures / "simulation")

# Physical time unit hbar / (g n0) = m xi^2 / hbar for the 87Rb experiment
# the couplings are modelled on (density healing length xi = 0.6 um).
XI_METERS = 0.6e-6
TIME_UNIT_S = M_RB87 * XI_METERS**2 / HBAR   # [s] per dimensionless time unit


def domain_size(z, dA, k_mag):
    # mean domain size from the spin structure factor: L = 2 pi <S> / <k S>
    S = np.abs(np.fft.fft2(z))**2
    S[0, 0] = 0.0  # drop the uniform mode
    return 2 * np.pi * S.sum() / (k_mag * S).sum()


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # couplings: g = 1 in both components, g_ud set by the interface width,
    # xi_s = xi / sqrt(g_ud/g - 1)  ->  g_ud = 1 + (xi/xi_s)^2
    g = 1.0
    xi_s_over_xi = 3.3
    g_ud = g * (1.0 + 1.0 / xi_s_over_xi**2)
    n0 = 1.0

    N = 256
    L = 256        # box side in units of xi
    dt = 0.05
    t_end = 2000.0
    record_every = 20            # steps between records (2 time units)
    snap_times = [100.0, 200.0, 300.0, 400.0, 500.0,
                  600.0, 700.0, 800.0, 900.0, 1000.0,
                  1100.0, 1200.0, 1300.0, 1400.0, 1500.0,
                  1600,1800,2000.0]

    # fastest growing mode of the demixing instability
    dg = g_ud - g
    k_star = np.sqrt(dg * n0 / 2)
    print(f"g_ud/g = {g_ud:.3f}  (immiscible, since > 1)")
    print(f"fastest-growing mode k* = {k_star:.2f} -> initial domain scale ~ "
          f"{2 * np.pi / k_star:.0f} xi, growth time ~ {4 / (dg * n0):.0f}")
    print(f"grid {N}x{N}, box {L:.0f} xi, dt = {dt}, t_end = {t_end:.0f}\n")

    solver = BinaryGPE2D(Nx=N, Ny=N, Lx=L, Ly=L, g_uu=g, g_dd=g, g_ud=g_ud, smooth_w=2.0)
    # fresh random-phase cosine noise each run; print the seed so a
    # particular run can be reproduced with initialize_mixture(..., seed=...)
    seed = int(np.random.SeedSequence().entropy % 2**32)
    print(f"noise seed = {seed}")
    solver.initialize_mixture(n0=n0, noise=0.01, seed=seed)

    kx = 2 * np.pi * np.fft.fftfreq(N, solver.dx)
    KX, KY = np.meshgrid(kx, kx, indexing="xy")
    k_mag = np.sqrt(KX**2 + KY**2)

    E0 = solver.energy()["total"]
    Nup0, Ndn0 = solver.norms()

    snapshots = {}
    times, mean_abs_z, L_domain = [], [], []

    def record(t):
        z = solver.magnetization()
        times.append(t)
        mean_abs_z.append(np.abs(z).mean())
        L_domain.append(domain_size(z, solver.dA, k_mag))
        if t in snap_times:
            snapshots[t] = z.copy()

    record(0.0)
    n_blocks = int(round(t_end / (dt * record_every)))
    for block in range(1, n_blocks + 1):
        solver.evolve(dt, record_every)
        record(round(block * record_every * dt, 6))
        if block % (n_blocks // 10) == 0:
            print(f"  t = {times[-1]:6.0f}   <|z|> = {mean_abs_z[-1]:.3f}   "
                  f"L_domain = {L_domain[-1]:.1f} xi")

    E1 = solver.energy()["total"]
    Nup1, Ndn1 = solver.norms()
    print("\nConservation check:")
    print(f"  N_up:   {Nup0:.4f} -> {Nup1:.4f}")
    print(f"  N_down: {Ndn0:.4f} -> {Ndn1:.4f}")
    print(f"  Energy: {E0:.4f} -> {E1:.4f}  (rel. drift {abs(E1 - E0) / abs(E0):.2e})")

    # magnetization snapshots, blue = down domains, red = up domains
    fig, axes = plt.subplots(3, 4, figsize=(13, 10))
    extent = [-L / 2, L / 2, -L / 2, L / 2]
    for ax, t in zip(axes.ravel(), snap_times):
        ax.imshow(snapshots[t], extent=extent, origin="lower",
                  cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(f"t = {t * TIME_UNIT_S:.2f} s")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Binary condensate: demixing and domain coarsening "
                 r"(magnetization $z = (n_\uparrow - n_\downarrow)/n$)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(RESULTS_DIR, "binary_coarsening.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"\nSnapshots saved to '{path}'")

    # left: mean domain size vs time; right: same data on log-log axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    t_arr = np.array(times)
    t_arr_s = t_arr * TIME_UNIT_S               # time axis [s]
    L_arr = np.array(L_domain)

    ax1.plot(t_arr_s, L_arr, color="#2166ac")
    ax1.set_xlabel(r"Time $t$ (s)")
    ax1.set_ylabel(r"$L_{\rm domain}$ [$\xi$]")
    ax1.set_title("Mean domain size")
    ax1.grid(alpha=0.3)

    # diffusive t^(1/3) growth is what we expect this close to miscibility
    late = t_arr > 300  # after demixing saturates (dimensionless units)
    ax2.loglog(t_arr_s[late], L_arr[late], color="#2166ac",
               label="simulation")
    t_ref = t_arr_s[late]
    ax2.loglog(t_ref, L_arr[np.argmax(late)] * (t_ref / t_ref[0])**(1 / 3),
               "k--", alpha=0.6, label=r"$t^{1/3}$")
    ax2.set_xlabel(r"Time $t$ (s)")
    ax2.set_ylabel(r"$L_{\rm domain}$ [$\xi$]")
    ax2.set_title("Domain coarsening (log-log)")
    ax2.legend()
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "binary_coarsening_growth.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"Growth curves saved to '{path}'")


if __name__ == "__main__":
    run()
