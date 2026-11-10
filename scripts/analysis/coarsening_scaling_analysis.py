"""
Universal-scaling analysis of binary-mixture coarsening, following
Hofmann, Natu & Das Sarma, PRL 113, 095702 (2014).

Experiment (the `coarsening` dataset, box interior, imaged-component OD
fluctuations) and simulation (results/derived/binary_box_ensemble_part*.npz,
magnetization
of the 8-seed ensemble in the measured box) are analyzed identically:

  * pair correlation g(r, t): azimuthally averaged, aperiodic (zero-padded)
    autocorrelation of the mean-subtracted field, normalized to g(0) = 1
  * domain size L(t): FIRST ZERO of g(r, t)  [insensitive to the additive
    noise/envelope power that biases structure-factor moments]
  * static structure factor S(q): azimuthally averaged power spectrum
    (Tukey-windowed); the experimental S(q) is noise-floor-subtracted using
    the t = 0 shots
  * local dynamical exponent 1/z(t) = log(L'/L)/log(t'/t) on time pairs with
    t'/t ~ 2, extrapolated linearly in xi_s/L(t) -> 0 (finite-size scaling)

Figures (results/figures/coarsening/):
  scaling_Lt_firstzero.png        L(t) experiment vs ensemble, linear + log-log
  scaling_correlation_collapse.png g(r/L) collapse, experiment | simulation
  scaling_structure_factor.png    L q^3 S(q) vs qL Porod collapse, exp | sim
  scaling_exponent.png            1/z(t) vs xi_s/L with extrapolation
"""


import h5py
import numpy as np
from scipy.ndimage import gaussian_filter, rotate

from quantum_sim.config import PATHS, apparatus, dataset
from quantum_sim.utils.plotting import use_headless_backend

use_headless_backend()
import matplotlib.pyplot as plt  # noqa: E402 — must follow the backend choice

from quantum_sim.utils.domains import (  # noqa: E402
    first_zero,
    local_exponents,
    radial_autocorr,
    radial_spectrum,
)

DATASET = "coarsening"
XI_S_UM = apparatus().xi_spin_um          # spin healing length [um]
COLLAPSE_MS = [50.0, 75.0, 100.0, 150.0, 200.0, 300.0]
FIT_WINDOW_MS = (40.0, 400.0)             # pairs used for the 1/z extrapolation fit


# --------------------------------------------------------------------------- #
# Experimental data
#
# The estimators (radial_autocorr, first_zero, radial_spectrum,
# local_exponents) are imported from quantum_sim.utils.domains, which is the
# single definition shared with the figure scripts.
# --------------------------------------------------------------------------- #
def load_experiment(ds):
    """Interior-cropped, per-shot OD fields keyed by time [ms]."""
    r0, r1, c0, c1 = ds.crop_box
    by_time = {}
    for path in ds.shot_paths("*.h5"):
        with h5py.File(path, "r") as f:
            t = round(float(f["globals"].attrs["BEC_TIME_SEQ"][ds.time_index] * 1e3), 3)
            g = f["images"][ds.camera]["MOT"]
            a = np.array(g["atoms"], int)[r0:r1, c0:c1]
            b = np.array(g["background"], int)[r0:r1, c0:c1]
            pr = np.array(g["probe_beam"], int)[r0:r1, c0:c1]
        od = -np.log(np.maximum(a - b, 1e-6) / np.maximum(pr - b, 1e-6))
        by_time.setdefault(t, []).append(rotate(od, ds.rotation_deg, reshape=False))

    times = sorted(by_time)
    sm = gaussian_filter(np.mean(by_time[times[0]], axis=0), 2)
    mask = sm > 0.5 * sm.max()
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    inset = 15
    ri = slice(rows[0] + inset, rows[-1] - inset)
    ci = slice(cols[0] + inset, cols[-1] - inset)
    return {t: [od[ri, ci] for od in by_time[t]] for t in times}


# --------------------------------------------------------------------------- #
def main():
    ds = dataset(DATASET)

    # ---- experiment ------------------------------------------------------- #
    exp = load_experiment(ds)
    exp_times = np.array(sorted(exp))
    exp_g = {}          # t -> shot-averaged g(r)
    exp_L, exp_L_sem = [], []
    exp_S = {}          # t -> shot-averaged S(q)
    for t in exp_times:
        gs, Ls, Ss = [], [], []
        for od in exp[t]:
            r_um, g = radial_autocorr(od, ds.pixel_um)
            gs.append(g)
            Ls.append(first_zero(r_um, g))
            q_um, S, _ = radial_spectrum(od, ds.pixel_um)
            Ss.append(S)
        exp_g[t] = (r_um, np.mean(gs, axis=0))
        Ls = np.array(Ls)
        exp_L.append(np.nanmean(Ls))
        exp_L_sem.append(np.nanstd(Ls, ddof=1) / np.sqrt(len(Ls)))
        exp_S[t] = (q_um, np.mean(Ss, axis=0))
    exp_L = np.array(exp_L)
    exp_L_sem = np.array(exp_L_sem)
    print("experiment: first-zero L(t):")
    for t, L, s in zip(exp_times, exp_L, exp_L_sem):
        print(f"  t = {t:6.1f} ms   L = {L:5.2f} +/- {s:4.2f} um")

    # ---- simulation ensemble (merge all part files) ----------------------- #
    part_files = sorted(PATHS.derived.glob("binary_box_ensemble_part*.npz"))
    if not part_files:
        raise SystemExit(
            "no binary_box_ensemble_part*.npz found in "
            f"{PATHS.derived} — run scripts/simulations/run_binary_box_ensemble.py first"
        )
    parts = [np.load(f) for f in part_files]
    sim_times = parts[0]["times_ms"]
    dx_um = float(parts[0]["dx_um"])
    seeds = np.concatenate([p["seeds"] for p in parts])
    ens = {}
    for p in parts:
        for s in p["seeds"]:
            ens[f"m_seed{s}"] = p[f"m_seed{s}"]
    print(f"\nensemble: {len(seeds)} seeds from {len(parts)} part files")
    sim_g = {}
    sim_L_seeds = np.full((len(seeds), len(sim_times)), np.nan)
    sim_S = {}
    for it, t in enumerate(sim_times):
        gs, Ss = [], []
        for isd, seed in enumerate(seeds):
            m = ens[f"m_seed{seed}"][it]
            r_um, g = radial_autocorr(m, dx_um)
            gs.append(g)
            sim_L_seeds[isd, it] = first_zero(r_um, g)
            q_um_s, S, _ = radial_spectrum(m, dx_um)
            Ss.append(S)
        sim_g[float(t)] = (r_um, np.mean(gs, axis=0))
        sim_S[float(t)] = (q_um_s, np.mean(Ss, axis=0))
    sim_L = np.nanmean(sim_L_seeds, axis=0)
    sim_L_sem = np.nanstd(sim_L_seeds, axis=0, ddof=1) / np.sqrt(len(seeds))
    print("\nsimulation ensemble: first-zero L(t):")
    for t, L, s in zip(sim_times, sim_L, sim_L_sem):
        print(f"  t = {t:6.1f} ms   L = {L:5.2f} +/- {s:4.2f} um")

    np.savez_compressed(
        PATHS.derived_file("scaling_analysis.npz"),
        exp_times=exp_times, exp_L=exp_L, exp_L_sem=exp_L_sem,
        sim_times=sim_times, sim_L=sim_L, sim_L_sem=sim_L_sem,
        sim_L_seeds=sim_L_seeds)

    # ---- figure 1: L(t) --------------------------------------------------- #
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))
    ax1.errorbar(exp_times, exp_L, yerr=exp_L_sem, fmt="o", ms=6, color="C0",
                 capsize=3, label="experiment")
    ax1.errorbar(sim_times, sim_L, yerr=sim_L_sem, fmt="s-", ms=4, color="C1",
                 capsize=2, label="simulation (8-seed ensemble)")
    ax1.set_xlabel("evolution time  t  (ms)")
    ax1.set_ylabel(r"domain size  L  ($\mu$m)  [first zero of g(r)]")
    ax1.set_title("domain size vs evolution time")
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.25)

    pos_e = exp_times >= 20
    pos_s = sim_times >= 20
    ax2.errorbar(exp_times[pos_e], exp_L[pos_e], yerr=exp_L_sem[pos_e],
                 fmt="o", ms=6, color="C0", capsize=3, label="experiment")
    ax2.errorbar(sim_times[pos_s], sim_L[pos_s], yerr=sim_L_sem[pos_s],
                 fmt="s-", ms=4, color="C1", capsize=2,
                 label="simulation (8-seed ensemble)")
    t_ref = np.array([30.0, 800.0])
    L30 = exp_L[exp_times == 30.0][0]
    ax2.plot(t_ref, L30 * (t_ref / 30.0) ** (1 / 3), "k--", alpha=0.6,
             label=r"$t^{1/3}$")
    ax2.plot(t_ref, L30 * (t_ref / 30.0) ** (2 / 3), "k:", alpha=0.6,
             label=r"$t^{2/3}$")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("evolution time  t  (ms)")
    ax2.set_ylabel(r"domain size  L  ($\mu$m)")
    ax2.set_title("log-log")
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(alpha=0.25, which="both")
    fig.tight_layout()
    path = str(ds.figure("scaling_Lt_firstzero.png"))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"\nsaved '{path}'")

    # ---- figure 2: correlation-function collapse -------------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    for ax, (label, gdict, times, Ls) in zip(axes, [
            ("experiment", exp_g, exp_times, exp_L),
            ("simulation (ensemble mean)", sim_g, sim_times, sim_L)]):
        for t in COLLAPSE_MS:
            i = np.where(times == t)[0][0]
            r_um, g = gdict[t]
            ax.plot(r_um / Ls[i], g, lw=1.4, label=f"t = {t:g} ms")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlim(0, 5)
        ax.set_ylim(-0.45, 1.02)
        ax.set_xlabel("r / L(t)")
        ax.set_ylabel("g(r)")
        ax.set_title(label)
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.25)
    fig.suptitle("dynamic scaling: pair-correlation collapse")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = str(ds.figure("scaling_correlation_collapse.png"))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"saved '{path}'")

    # ---- figure 3: structure-factor collapse (Porod tail) ----------------- #
    q0, S0 = exp_S[exp_times[0]]     # experimental noise floor from t = 0
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    for t in COLLAPSE_MS:
        i = np.where(exp_times == t)[0][0]
        q_um, S = exp_S[t]
        S_sub = S - S0               # noise-floor subtraction
        L = exp_L[i]
        sel = q_um > 0
        axes[0].plot(q_um[sel] * L, L * q_um[sel] ** 3 * S_sub[sel] / 1e4,
                     lw=1.4, label=f"t = {t:g} ms")
    for t in COLLAPSE_MS:
        i = np.where(sim_times == t)[0][0]
        q_um, S = sim_S[t]
        L = sim_L[i]
        sel = q_um > 0
        axes[1].plot(q_um[sel] * L, L * q_um[sel] ** 3 * S[sel] / 1e4,
                     lw=1.4, label=f"t = {t:g} ms")
    for ax, label in zip(axes, ["experiment (noise-floor subtracted)",
                                "simulation (ensemble mean)"]):
        ax.set_xlim(0, 20)
        ax.set_xlabel("q L(t)")
        ax.set_ylabel(r"$L\,q^3\,S(q)$  (arb. u.)")
        ax.set_title(label)
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.25)
    fig.suptitle("structure-factor collapse: plateau at high qL = Porod "
                 r"$q^{-3}$ tail (domain walls)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = str(ds.figure("scaling_structure_factor.png"))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"saved '{path}'")

    # ---- figure 4: local exponent extrapolation --------------------------- #
    tmid_s, Lmid_s, zinv_s, _ = local_exponents(sim_times, sim_L)
    tmid_e, Lmid_e, zinv_e, zerr_e = local_exponents(
        exp_times, exp_L, exp_L_sem)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x_s = XI_S_UM / Lmid_s
    fit_sel = (tmid_s >= FIT_WINDOW_MS[0]) & (tmid_s <= FIT_WINDOW_MS[1])
    b, a = np.polyfit(x_s[fit_sel], zinv_s[fit_sel], 1)
    x_line = np.linspace(0, x_s.max() * 1.05, 50)
    ax.plot(x_s, zinv_s, "s", ms=6, color="C1",
            label="simulation (8-seed ensemble)")
    ax.plot(x_line, a + b * x_line, "-", color="C1", alpha=0.7,
            label=rf"linear fit: $1/z(\infty)$ = {a:.2f}")
    ax.plot(0, a, "*", ms=14, color="C1")
    x_e = XI_S_UM / Lmid_e
    ax.errorbar(x_e, zinv_e, yerr=zerr_e, fmt="o", ms=6, color="C0",
                capsize=3, label="experiment")
    ax.axhline(2 / 3, color="k", ls=":", alpha=0.6)
    ax.axhline(1 / 3, color="k", ls="--", alpha=0.6)
    ax.text(0.965, 2 / 3, r"$2/3$ inertial hydrodynamic", ha="right",
            va="bottom", transform=ax.get_yaxis_transform(), fontsize=9)
    ax.text(0.965, 1 / 3, r"$1/3$ diffusive", ha="right", va="bottom",
            transform=ax.get_yaxis_transform(), fontsize=9)
    ax.set_xlabel(r"$\xi_s / L(t)$")
    ax.set_ylabel(r"local exponent  $1/z(t)$")
    ax.set_title("dynamical exponent with finite-size extrapolation")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = str(ds.figure("scaling_exponent.png"))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"saved '{path}'")
    print(f"\nsimulation extrapolated exponent 1/z(infinity) = {a:.3f}")


if __name__ == "__main__":
    main()
