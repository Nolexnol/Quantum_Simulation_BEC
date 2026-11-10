"""
Binary-mixture coarsening in the MEASURED box trap, matched to the experiment.

The experiment (scripts/coarsening_data/): a sodium condensate in a
DMD-projected square box; a microwave pulse at the Na F=1 <-> F=2 splitting
(1.7716 GHz in the shot globals) creates a 50/50 two-state mixture, the
immiscible mixture demixes into magnetic domains, and absorption imaging of
ONE hyperfine state records the domain pattern at evolution times 0-800 ms.

Trap parameters measured from the t = 0 images (Thomas-Fermi inversion of the
wall profile, 0.66 um/px):
    box side (half-max)   112.9 um
    wall 10-90% width     4.2 um   (resolution/healing limited -> hard wall)
    barrier height        >= mu; modelled as 5 mu (steepness is what matters)

Physical scales assume areal density n0 = 1e14 m^-2 (the raw OD undercounts
by the PTAI transfer fraction; same calibration assumption as
run_sodium_domains.py). That gives mu = g_2D n0 ~ h x 1.3 kHz, healing length
xi ~ 0.58 um and time unit hbar/mu ~ 0.12 ms. The interspecies coupling is
set by the observed domain-wall width, xi_s = 2 um: g_ud/g = 1 + (xi/xi_s)^2.

The simulation is recorded at exactly the experimental evolution times.
Experimental domain sizes use the raw notebook method (debugs.ipynb):
rotate each image 45 deg, crop to the inscribed square, then
L = 2 pi <S> / <k S> with only the uniform mode dropped — no window, no PCs.
Simulation domain sizes are computed directly on the simulated images
(imaged-component density, box interior). Outputs: domain size vs time
(linear and log-log with a power-law line of best fit to the experiment)
and separate time-evolution image galleries for experiment and simulation.
"""

import os
import time

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantum_sim import BinaryGPE2D
from quantum_sim.binary_box import box_setup
from quantum_sim.config import PATHS, dataset
from quantum_sim.units import H_PLANCK
from quantum_sim.utils.domains import domain_length_um
from quantum_sim.utils.images import rotate_and_crop

DATASET = "coarsening"
RESULTS_DIR = str(PATHS.figures / "binary_box_match")

# --------------------------------------------------------------------------- #
# Experimental parameters and derived scales
#
# The box setup lives in quantum_sim.binary_box so that run_binary_box_ensemble
# reproduces exactly this configuration without importing this script.
# --------------------------------------------------------------------------- #
SETUP = box_setup()

BOX_SIDE_UM = SETUP.box_side_um
WALL_1090_UM = SETUP.wall_1090_um
V0_OVER_MU = SETUP.v0_over_mu
XI_UM = SETUP.xi_um
T_UNIT_MS = SETUP.t_unit_ms
G = SETUP.g
G_UD = SETUP.g_ud
MU_EFF = SETUP.mu_eff
MU = SETUP.mu                      # chemical potential [J]
XI_S_UM = SETUP.xi_spin_um         # spin healing length [um]

SEED = 12345               # noise seed for the demixing instability
FIT_MIN_MS = 30            # log-log best fit uses t >= this (domains formed)

# --------------------------------------------------------------------------- #
# Notebook estimator: raw structure factor, no window / no PCs / no mean sub
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Experimental analysis
# --------------------------------------------------------------------------- #
def load_od(path, ds):
    with h5py.File(path, "r") as f:
        t_ms = f["globals"].attrs["BEC_TIME_SEQ"][ds.time_index] * 1e3
        g = f["images"][ds.camera]["MOT"]
        atoms = np.array(g["atoms"], dtype=int)
        bg = np.array(g["background"], dtype=int)
        probe = np.array(g["probe_beam"], dtype=int)
    r0, r1, c0, c1 = ds.crop_box
    sl = np.s_[r0:r1, c0:c1]
    atoms, bg, probe = atoms[sl], bg[sl], probe[sl]
    return t_ms, -np.log(np.maximum(atoms - bg, 1e-6)
                         / np.maximum(probe - bg, 1e-6))


def analyze_experiment(ds):
    """Per-shot and per-time L(t) with the raw (un-windowed) estimator."""
    by_time = {}
    for path in ds.shot_paths("*.h5"):
        t_ms, od = load_od(path, ds)
        by_time.setdefault(round(float(t_ms), 3), []).append(
            rotate_and_crop(od, ds.rotation_deg))

    times = np.array(sorted(by_time))
    t_shots, L_shots = [], []
    mean_ods = {}
    for t in times:
        for crop in by_time[t]:
            t_shots.append(t)
            L_shots.append(domain_length_um(crop, ds.pixel_um))
        mean_ods[t] = np.mean(by_time[t], axis=0)
    t_shots = np.array(t_shots)
    L_shots = np.array(L_shots)
    L_mean = np.array([L_shots[t_shots == t].mean() for t in times])
    L_sem = np.array([L_shots[t_shots == t].std(ddof=1)
                      / np.sqrt((t_shots == t).sum()) for t in times])
    return times, t_shots, L_shots, L_mean, L_sem, mean_ods


# --------------------------------------------------------------------------- #
# Simulation in the measured box
# --------------------------------------------------------------------------- #
def simulate(seed, exp_times_ms, N=256, dt=0.05):
    side = BOX_SIDE_UM / XI_UM               # box side [xi]
    L = side + 32.0                          # margin ~ 9 um per side
    dx = L / N
    wall_w = (WALL_1090_UM / XI_UM) / (2 * np.arctanh(0.8))

    solver = BinaryGPE2D(Nx=N, Ny=N, Lx=L, Ly=L, g_uu=G, g_dd=G, g_ud=G_UD)
    box_x = 0.5 * (np.tanh((solver.X + side / 2) / wall_w)
                   - np.tanh((solver.X - side / 2) / wall_w))
    box_y = 0.5 * (np.tanh((solver.Y + side / 2) / wall_w)
                   - np.tanh((solver.Y - side / 2) / wall_w))
    V = V0_OVER_MU * MU_EFF * (1.0 - box_x * box_y)
    solver.V_up = V
    solver.V_down = V.copy()

    # Thomas-Fermi mixture, briefly relaxed in imaginary time (components
    # stay identical during relaxation, so no premature demixing)
    n_tf = np.clip(1.0 - V / MU_EFF, 0.0, None)
    solver.psi[0] = np.sqrt(n_tf / 2)
    solver.psi[1] = np.sqrt(n_tf / 2)
    norms0 = solver.norms()
    for _ in range(150):
        solver.step(-0.02j)
        for c, N_c in enumerate(solver.norms()):
            solver.psi[c] *= np.sqrt(norms0[c] / N_c)
    solver._kin_dt = None   # drop the cached imaginary-time kinetic phase

    # seed the demixing instability (random-phase cosines, as
    # BinaryGPE2D.initialize_mixture, enveloped by the box density)
    rng = np.random.default_rng(seed)
    envelope = np.sqrt(np.clip(n_tf, 0.0, None))
    for c in range(2):
        fluct = np.zeros((N, N))
        mx = rng.integers(-32, 33, 64)
        my = rng.integers(-32, 33, 64)
        phase = 2 * np.pi * rng.random(64)
        for mxi, myi, ph in zip(mx, my, phase):
            if mxi == 0 and myi == 0:
                continue
            fluct += np.cos(2 * np.pi * (mxi * solver.X + myi * solver.Y) / L
                            + ph)
        fluct /= fluct.std()
        solver.psi[c] += 0.01 * fluct * envelope

    E0 = solver.energy()["total"]

    snaps = {}
    steps_done = 0
    t_wall = time.time()
    for t_ms in exp_times_ms:
        target = int(round(t_ms / T_UNIT_MS / dt))
        solver.evolve(dt, target - steps_done)
        steps_done = target
        n_up, _ = solver.densities()
        snaps[float(t_ms)] = n_up.astype(np.float32)   # full field
        print(f"  t = {t_ms:6.1f} ms ({steps_done:6d} steps)  "
              f"[{time.time() - t_wall:5.0f} s]", flush=True)

    E1 = solver.energy()["total"]
    print(f"  energy drift {abs(E1 - E0) / abs(E0):.2e}")
    return snaps, dx * XI_UM, side


# --------------------------------------------------------------------------- #
def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    exp_times_ms, t_shots, L_shots, L_exp, L_exp_sem, exp_ods = \
        analyze_experiment(dataset(DATASET))
    print(f"experimental L(t): {len(exp_times_ms)} times, "
          f"{len(L_shots)} shots")

    print("=" * 60)
    print("Binary coarsening in the measured box trap (23Na)")
    print("=" * 60)
    print(f"mu = g_2D n0 = h x {MU / H_PLANCK:.0f} Hz,  xi = {XI_UM:.2f} um,  "
          f"time unit = {T_UNIT_MS:.3f} ms")
    print(f"g_ud/g = {G_UD:.4f}  (xi_s = {XI_S_UM} um)")
    print(f"box {BOX_SIDE_UM:.0f} um, wall 10-90% {WALL_1090_UM} um, "
          f"V0 = {V0_OVER_MU} mu,  seed = {SEED}\n")

    snaps, dx_um, side = simulate(SEED, exp_times_ms)

    # domain size directly on the simulated images (box interior, ~92 um)
    N = snaps[0.0].shape[0]
    half_px = int(0.41 * side / (dx_um / XI_UM))
    sl = slice(N // 2 - half_px, N // 2 + half_px + 1)
    L_sim = np.array([domain_length_um(snaps[t][sl, sl], dx_um)
                      for t in exp_times_ms])

    print("\n   t (ms)   L_exp    L_sim")
    for i, t in enumerate(exp_times_ms):
        print(f"  {t:7.1f}   {L_exp[i]:5.1f}    {L_sim[i]:5.1f}")

    np.savez_compressed(
        os.path.join(RESULTS_DIR, "binary_box_match.npz"),
        exp_times_ms=exp_times_ms, t_shots=t_shots, L_shots=L_shots,
        L_exp=L_exp, L_exp_sem=L_exp_sem, L_sim=L_sim,
        xi_um=XI_UM, t_unit_ms=T_UNIT_MS, dx_um=dx_um,
        **{f"nup_{t:g}ms": s for t, s in snaps.items()})

    # --- figure 1: domain size vs time (linear) ---------------------------- #
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_shots, L_shots, "o", ms=3, color="C0", alpha=0.3,
            label="individual shots")
    ax.errorbar(exp_times_ms, L_exp, yerr=L_exp_sem, fmt="o", ms=6,
                color="C0", capsize=3, label="experiment (mean of repeats)",
                zorder=3)
    ax.plot(exp_times_ms, L_sim, "s-", ms=4, color="C1", label="simulation")
    ax.set_xlabel("evolution time  t  (ms)")
    ax.set_ylabel(r"domain size  L  ($\mu$m)")
    ax.set_title("domain size vs evolution time")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "domain_size_vs_time.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"\nsaved '{path}'")

    # --- figure 2: log-log with power-law line of best fit ----------------- #
    fit_sel = exp_times_ms >= FIT_MIN_MS
    b, a = np.polyfit(np.log10(exp_times_ms[fit_sel]),
                      np.log10(L_exp[fit_sel]), 1)
    t_fit = np.array([exp_times_ms[fit_sel][0], exp_times_ms[-1]])
    print(f"best-fit exponent (experiment, t >= {FIT_MIN_MS} ms): {b:.3f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    pos = exp_times_ms > 0
    ax.errorbar(exp_times_ms[pos], L_exp[pos], yerr=L_exp_sem[pos], fmt="o",
                ms=6, color="C0", capsize=3,
                label="experiment (mean of repeats)", zorder=3)
    ax.plot(exp_times_ms[pos], L_sim[pos], "s-", ms=4, color="C1",
            label="simulation")
    ax.plot(t_fit, 10**a * t_fit**b, "k--", alpha=0.8,
            label=rf"best fit:  $L \propto t^{{{b:.2f}}}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("evolution time  t  (ms)")
    ax.set_ylabel(r"domain size  L  ($\mu$m)")
    ax.set_title("domain size vs evolution time (log-log)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "domain_size_vs_time_loglog.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"saved '{path}'")

    # --- figures 3 & 4: separate time-evolution galleries ------------------ #
    def evolution_gallery(images, title, path, ncols=6):
        nrows = int(np.ceil(len(exp_times_ms) / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(2.3 * ncols, 2.5 * nrows))
        for ax in axes.ravel():
            ax.axis("off")
        vmax = max(im.max() for im in images.values())
        for ax, t in zip(axes.ravel(), exp_times_ms):
            ax.imshow(images[t], cmap="viridis", vmin=0, vmax=vmax)
            ax.set_title(f"t = {t:g} ms", fontsize=9)
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(path, dpi=140)
        plt.close(fig)
        print(f"saved '{path}'")

    evolution_gallery(
        {t: exp_ods[t] for t in exp_times_ms},
        "Experiment: imaged-component OD (mean of repeats)",
        os.path.join(RESULTS_DIR, "binary_box_evolution_experiment.png"))
    evolution_gallery(
        {t: snaps[t] for t in exp_times_ms},
        "Simulation: imaged-component density",
        os.path.join(RESULTS_DIR, "binary_box_evolution_simulation.png"))


if __name__ == "__main__":
    run()
