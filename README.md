# quantum_sim — Spinor Bose–Einstein Condensate Solver

A numerical toolkit for simulating a **two-component (spin-½) Bose–Einstein
condensate** in a 1D harmonic trap, governed by the coupled **Gross–Pitaevskii
Equation (GPE)** and integrated with the **Split-Step Fourier Method (SSFM)**.

The solver models:

- intra-species contact interactions (`g_up_up`, `g_down_down`),
- inter-species contact interaction (`g_up_down`),
- coherent **Rabi coupling** between the two spin components (`Omega`),
- an arbitrary external potential (default: harmonic, `V = ½x²`).

It supports both **real-time propagation** (dynamics) and **imaginary-time
propagation** (ground-state relaxation via the variational principle).

## Project layout

```
quantum_sim/
├── src/quantum_sim/
│   ├── __init__.py          # package API: SpinorGPESolver
│   └── gpe_solver.py        # the core solver
├── scripts/
│   └── run_simulation.py    # end-to-end demo -> results/*.png
├── tests/
│   └── test_gpe_solver.py   # analytic correctness tests
├── results/                 # generated plots (git-ignored)
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Or install the package (editable) with its dev extras:

```bash
pip install -e ".[dev]"
```

## Running the tests

The suite validates the solver against independent analytic results
(harmonic ground state, unitarity, energy conservation, Rabi oscillations,
free-wavepacket dispersion, the variational principle):

```bash
pytest
```

## Running the simulation

```bash
python scripts/run_simulation.py
```

This runs an analytical verification, finds the interacting coupled ground
state via imaginary-time propagation, then quenches the trap and Rabi coupling
to drive real-time dynamics. Figures are written to `results/`:

- `gpe_ground_state.png` — ITP energy convergence and coupled ground-state density.
- `gpe_dynamics.png` — Rabi oscillations, density streak plots, and dipole snapshots.

## Physics conventions

Dimensionless units with `ħ = m = ω = 1`; lengths in harmonic-oscillator units
`a_ho`, energies in `ħω`. The kinetic operator is `T = k²/2`, and the Rabi term
is `H_c = -Ω σ_x`.
