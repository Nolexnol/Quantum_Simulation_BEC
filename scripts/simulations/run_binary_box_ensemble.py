"""
Ensemble of binary-box simulations for the scaling analysis
(Hofmann, Natu & Das Sarma, PRL 113, 095702 (2014) methodology).

Runs the measured-box simulation of run_binary_box_match.py for several
independent noise seeds SEQUENTIALLY (macOS demotes multiprocessing workers
to efficiency cores; top-level processes keep full speed — launch several
invocations with disjoint seed lists for parallelism) and stores, at every
experimental evolution time, the magnetization m = (n_up - n_dn)/(n_up + n_dn)
and the imaged-component density on the box interior (~92 um square).

Usage: python3 run_binary_box_ensemble.py <seed> [<seed> ...]
Output: results/derived/binary_box_ensemble_part<firstseed>.npz (one per
invocation; the analysis script merges every part file it finds)
"""

import sys
import time

import numpy as np

from quantum_sim import BinaryGPE2D
from quantum_sim.binary_box import box_setup
from quantum_sim.config import PATHS

# The same box setup run_binary_box_match.py was validated against. Imported
# from the package rather than from that script, so this job neither depends
# on a sibling entry point nor drags its plotting stack into a headless run.
SETUP = box_setup()
BOX_SIDE_UM = SETUP.box_side_um
WALL_1090_UM = SETUP.wall_1090_um
V0_OVER_MU = SETUP.v0_over_mu
XI_UM = SETUP.xi_um
T_UNIT_MS = SETUP.t_unit_ms
G = SETUP.g
G_UD = SETUP.g_ud
MU_EFF = SETUP.mu_eff

TIMES_MS = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 62.5, 75.0, 87.5, 100.0,
            125.0, 150.0, 175.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0,
            500.0, 550.0, 600.0, 650.0, 700.0, 750.0, 800.0]
SEEDS = [12345, 98765, 24680, 13579, 31415, 27182, 16180, 46368]
N_WORKERS = 4


def run_seed(seed, N=256, dt=0.05):
    side = BOX_SIDE_UM / XI_UM
    L = side + 32.0
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

    n_tf = np.clip(1.0 - V / MU_EFF, 0.0, None)
    solver.psi[0] = np.sqrt(n_tf / 2)
    solver.psi[1] = np.sqrt(n_tf / 2)
    norms0 = solver.norms()
    for _ in range(150):
        solver.step(-0.02j)
        for c, N_c in enumerate(solver.norms()):
            solver.psi[c] *= np.sqrt(norms0[c] / N_c)
    solver._kin_dt = None

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

    half_px = int(0.41 * side / dx)
    sl = slice(N // 2 - half_px, N // 2 + half_px + 1)

    m_snaps, nup_snaps = [], []
    steps_done = 0
    t_wall = time.time()
    for i, t_ms in enumerate(TIMES_MS):
        target = int(round(t_ms / T_UNIT_MS / dt))
        solver.evolve(dt, target - steps_done)
        steps_done = target
        n_up, n_dn = solver.densities()
        m = (n_up - n_dn) / (n_up + n_dn + 1e-12)
        m_snaps.append(m[sl, sl].astype(np.float32))
        nup_snaps.append(n_up[sl, sl].astype(np.float32))
        if i % 5 == 0 or t_ms == TIMES_MS[-1]:
            print(f"  seed {seed}: t = {t_ms:6.1f} ms  "
                  f"[{time.time() - t_wall:5.0f} s]", flush=True)
    return seed, np.array(m_snaps), np.array(nup_snaps), dx * XI_UM


def main():
    seeds = [int(s) for s in sys.argv[1:]] or SEEDS
    print(f"ensemble part: seeds {seeds}, {len(TIMES_MS)} times to "
          f"{TIMES_MS[-1]:.0f} ms", flush=True)
    t0 = time.time()
    results = [run_seed(s) for s in seeds]
    print(f"all seeds done in {(time.time() - t0) / 60:.1f} min")

    out = {"times_ms": np.array(TIMES_MS), "seeds": np.array(seeds),
           "dx_um": results[0][3], "xi_um": XI_UM}
    for seed, m_snaps, nup_snaps, _ in results:
        out[f"m_seed{seed}"] = m_snaps
        out[f"nup_seed{seed}"] = nup_snaps
    path = PATHS.derived_file(f"binary_box_ensemble_part{seeds[0]}.npz")
    np.savez_compressed(path, **out)
    print(f"saved '{path}'")


if __name__ == "__main__":
    main()
