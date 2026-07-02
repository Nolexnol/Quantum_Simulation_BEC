import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for headless/scripted runs
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from quantum_sim import Spin1GPE2D

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "figure.titlesize": 14,
})


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    #Params
    N = 128              # grid points per side
    L = 128.0            # box length -> dx = 0.5 resolves the spin healing length
    n0 = 1.0             # mean density; sets the spin energy scale g_s * n0
    g_s = 1.0            # spin interaction > 0  => sodium (antiferromagnetic)
    g_n = 3.0            # density interaction (keeps total density stiff/uniform)
    q_quench = -1.5      # final quadratic Zeeman: in (-2 g_s n0, -g_s n0) window

    # Instability / domain scales
    k_star = np.sqrt(2.0 * (-q_quench - g_s * n0))          # fastest-growing k
    domain_size = 2.0 * np.pi / k_star
    t_s = 1.0 / (g_s * n0)                                  # spin time scale

    print("=" * 55)
    print("Quasi-2D spin-1 SODIUM BEC: spin-domain formation")
    print("=" * 55)
    print(f"Grid: {N}x{N},  box L={L},  dx={L / N:.3f}")
    print(f"g_s={g_s} (>0, antiferromagnetic),  g_n={g_n},  n0={n0}")
    print(f"Quench:  q -> {q_quench}   (unstable window: {-2 * g_s * n0} < q < 0)")
    print(f"Fastest-growing k* = {k_star:.3f} -> domain size ~ {domain_size:.1f}")
    print(f"Spin time t_s = {t_s:.2f}\n")

    solver = Spin1GPE2D(Nx=N, Ny=N, Lx=L, Ly=L, g_n=g_n, g_s=g_s, q=q_quench)
    solver.initialize_polar(n0=n0, noise=0.02, seed=12345)

    # --- Time evolution -----------------------------------------------------
    dt = 0.01
    t_end = 200.0
    n_steps = int(round(t_end / dt))
    snap_times = [0.0,1.0,2.0,5.0, 5.5,6.0,6.5, 7.0,7.5, 8.0,8.5, 9.0, 10.0,15.0,25.0,30.0,50.0,80.0,90.0,200.0]
    snap_steps = {int(round(t / dt)): t for t in snap_times}

    E0 = solver.energy()["total"]
    N0 = solver.total_norm()
    Mz0 = solver.total_magnetization()

    snapshots = {}   # t -> (Fz/n, n0_fraction, rgb)
    times, frac_p1, frac_0, frac_m1 = [], [], [], []

    def record_snapshot(t):
        n1, ncomp0, nm1 = solver.component_densities()
        n = n1 + ncomp0 + nm1
        _, _, Fz = solver.spin_density()
        rgb = np.stack([n1, ncomp0, nm1], axis=-1)
        rgb = rgb / (rgb.max() + 1e-12)
        snapshots[t] = (Fz / n, ncomp0 / n, rgb)

    if 0 in snap_steps:
        record_snapshot(snap_steps[0])

    for step in range(1, n_steps + 1):
        solver.step(dt)

        if step % 20 == 0:
            n1, ncomp0, nm1 = solver.component_densities()
            tot = (n1 + ncomp0 + nm1).sum()
            times.append(step * dt)
            frac_p1.append(n1.sum() / tot)
            frac_0.append(ncomp0.sum() / tot)
            frac_m1.append(nm1.sum() / tot)

        if step in snap_steps:
            record_snapshot(snap_steps[step])

    E1 = solver.energy()["total"]
    N1 = solver.total_norm()
    Mz1 = solver.total_magnetization()

    print("Conservation check (should be ~constant):")
    print(f"  Norm N:  {N0:.6f} -> {N1:.6f}   (rel. drift {abs(N1 - N0) / N0:.2e})")
    print(f"  Energy:  {E0:.5f} -> {E1:.5f}   (rel. drift {abs(E1 - E0) / abs(E0):.2e})")
    print(f"  M_z:     {Mz0:.3e} -> {Mz1:.3e}\n")

    # --- Plot: component-density domain snapshots (RGB), 4 per row ---------
    ordered = [t for t in snap_times if t in snapshots]
    n_snap = len(ordered)
    ncols = 5
    nrows = int(np.ceil(n_snap / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.3 * nrows))
    axes = np.atleast_2d(axes)
    extent = [-L / 2, L / 2, -L / 2, L / 2]

    for idx, t in enumerate(ordered):
        ax = axes.ravel()[idx]
        _, _, rgb = snapshots[t]
        ax.imshow(rgb, extent=extent, origin="lower")
        ax.set_title(f"t = {t:.1f}  ($t/t_s$={t / t_s:.1f})")
        ax.set_xticks([])
        ax.set_yticks([])

    # Blank any unused panels in the last row
    for ax in axes.ravel()[n_snap:]:
        ax.axis("off")

    axes.ravel()[0].set_ylabel(r"R = $m{+}1$   G = $m0$   B = $m{-}1$")

    fig.suptitle("Spin-domain formation & coarsening in a 2D spin-1 sodium BEC "
                 f"(quench $q\\to{q_quench}$)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    domains_path = os.path.join(RESULTS_DIR, "sodium_spin_domains.png")
    fig.savefig(domains_path, dpi=140)
    plt.close(fig)
    print(f"Domain snapshots saved to '{domains_path}'")

    # --- Plot: component populations over time -----------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(times, frac_0, "g-", label=r"$m=0$ (polar)")
    ax.plot(times, frac_p1, "r-", label=r"$m=+1$")
    ax.plot(times, frac_m1, "b--", label=r"$m=-1$")
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel("Fractional population")
    ax.set_title("Spin-mixing: population transfer $2\\,m0 \\to m{+}1 + m{-}1$")
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
