# quantum_sim — Spinor BEC Solvers & Coarsening Analysis

A numerical toolkit for simulating spinor **Bose–Einstein condensates**
governed by the **Gross–Pitaevskii Equation (GPE)** with the **Split-Step
Fourier Method (SSFM)**, together with the analysis pipeline for the
absorption images the simulations are compared against.

Three solvers are included:

- **`SpinorGPESolver`** — a two-component (spin-½) condensate in a 1D
  harmonic trap, with intra-/inter-species contact interactions
  (`g_up_up`, `g_down_down`, `g_up_down`) and coherent **Rabi coupling**
  (`Omega`, entering as `H_c = -Ω σ_x`). Supports **real-time propagation**
  (dynamics) and **imaginary-time propagation** (ground-state relaxation).
- **`Spin1GPE2D`** — a three-component (spin-1) quasi-2D condensate with
  density (`g_n`) and spin (`g_s`) interactions and a quadratic Zeeman
  energy (`q`), used for spin-domain formation and coarsening after a
  quench (Williamson & Blakie, PRL 116, 025301 (2016)).
- **`BinaryGPE2D`** — an immiscible two-component quasi-2D condensate, for
  phase separation and domain coarsening in the measured box.

New to this? Read **`TUTORIAL.md`** — a from-scratch guide to Hamiltonian
simulation with this codebase.

## Project layout

Dependencies point one way — `config` → `utils` → `io` → `scripts` → `notebooks`
— and nothing to the left imports anything to its right. Data and outputs sit
outside that chain entirely, reached only through `quantum_sim.config`.

```
quantum_sim/
├── config/
│   └── experiments.yaml        # measured calibration + per-dataset settings
├── src/quantum_sim/
│   ├── config.py               # THE source of paths and constants
│   ├── units.py                # CODATA constants + SI ↔ solver conversions
│   ├── solvers/                # the GPE solvers — pure, array in / array out
│   │   ├── gpe_solver.py       #   SpinorGPESolver  (spin-½, 1D)
│   │   ├── spinor1_gpe_2d.py   #   Spin1GPE2D       (spin-1, quasi-2D)
│   │   └── binary_gpe_2d.py    #   BinaryGPE2D      (binary, quasi-2D)
│   ├── utils/                  # commonly reused functions, by concern
│   │   ├── shots.py            #   filenames, grouping by hold time
│   │   ├── images.py           #   OD, rotation, framing, registration
│   │   ├── pca.py              #   principal-component decomposition
│   │   ├── domains.py          #   domain size, correlations, exponents
│   │   ├── stats.py            #   robust statistics, fit models
│   │   ├── parallel.py         #   multiprocess mapping over shots
│   │   └── plotting.py         #   style, colour limits, montages, movies
│   └── io/                     # the only code that knows the on-disk format
│       ├── shotfile.py         #   ShotFile, ShotFileTOF, ShotFileMultiImg
│       ├── loader.py           #   ShotCollection — bulk load into one table
│       └── database.py         #   SQLite index over the raw shots
├── scripts/                    # thin entry points; no reusable logic here
│   ├── simulations/            #   run a solver
│   ├── analysis/               #   reduce data to numbers
│   └── figures/                #   produce plots and movies
├── notebooks/                  # exploration only
├── data/                       # ALL data — git-ignored (see data/README.md)
│   ├── raw/                    #   measured shots; treat as read-only
│   ├── processed/  db/  archive/
├── results/                    # ALL outputs — git-ignored
│   ├── figures/                #   PNGs and GIFs, namespaced by dataset
│   └── derived/                #   intermediate .npz between script stages
├── tests/
├── Makefile
└── pyproject.toml
```

## Setup

```bash
make setup                       # venv + editable install with all extras
```

or manually:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[all]"          # or ".[dev]" for just the core + tests
```

Extras: `db` (peewee/tqdm, for the shot index), `notebook` (jupyter,
ipywidgets, multiprocess), `dev` (pytest, ruff), `all`.

Because the package is installed editable, **no script needs a `sys.path`
hack** — `from quantum_sim import ...` works from anywhere.

## Configuration

Every measured constant lives in `config/experiments.yaml`, and every path is
resolved by `quantum_sim.config`. Re-measuring the camera scale is a one-line
edit there, not a hunt through five scripts that each kept their own copy.

```python
from quantum_sim.config import PATHS, dataset, apparatus

ds = dataset("coarsening")       # declared in config/experiments.yaml
ds.pixel_um, ds.camera, ds.crop_box
paths = ds.shot_paths()          # every shot file of that run
fig = ds.figure("my_plot.png")   # results/figures/coarsening/my_plot.png
```

Relocate the data without touching any code:

```bash
export QUANTUM_SIM_DATA_DIR=/Volumes/lab_drive/quantum_sim_data
export QUANTUM_SIM_RESULTS_DIR=/tmp/scratch_results
```

Declared datasets: `coarsening`, `box40ms`, `shots700p`.

## Running things

```bash
make test        # pytest
make lint        # ruff
make check       # both
make figures     # regenerate the experimental figures
make sim         # run the simulation demos
make index       # build the SQLite shot index
make help        # everything available
```

Individual scripts:

```bash
python scripts/simulations/run_simulation_ho.py       # 87Rb spin-½ mixture (~2 s)
python scripts/simulations/run_sodium_domains.py      # 23Na spin-1 quench (~1 min)
python scripts/simulations/run_binary_coarsening.py   # binary demixing
python scripts/simulations/run_binary_box_match.py    # match the measured box
python scripts/simulations/run_binary_box_ensemble.py 12345 98765

python scripts/analysis/coarsening_scaling_analysis.py
python scripts/analysis/index_shots.py --build coarsening

python scripts/figures/plot_all_evolution.py
python scripts/figures/plot_single_rep_evolution.py
python scripts/figures/plot_coarsening_pattern.py
python scripts/figures/plot_coarsening_size.py
python scripts/figures/animate_pc_evolution.py -d coarsening
python scripts/figures/animate_pc_shots.py --dataset shots700p
```

Figures land in `results/figures/<dataset>/`.

## Tests

The suite validates the solvers against independent analytic results
(harmonic ground state, unitarity, energy conservation, Rabi oscillations,
free-wavepacket dispersion, the variational principle), and pins the shared
analysis helpers against properties that can be checked by hand — a stripe
pattern of known wavelength, an SVD's orthonormality, a pure `t^(1/3)` power
law, a translation of known size.

```bash
make test
```

## Physics conventions and units

The **solvers are dimensionless** (standard practice — it keeps every number
O(1) and one run describes a family of experiments):

- `SpinorGPESolver`: harmonic-oscillator units, `ħ = m = ω_x = 1`; lengths
  in `a_ho = sqrt(ħ/(m ω_x))`, times in `1/ω_x`, energies in `ħω_x`. Kinetic
  operator `T = k²/2`, Rabi term `H_c = -Ω σ_x`.
- `Spin1GPE2D`: spin units, `ħ = m = g_s n₀ = 1`; times in the spin time
  `t_s = ħ/(g_s n₀)`, lengths in `ħ/sqrt(m g_s n₀)` (≈ the spin healing
  length).

The **scripts carry the real units**: laboratory parameters (trap frequencies
in Hz, scattering lengths in Bohr radii, atom numbers, areal densities) are
declared in SI with units in comments, or read from
`config/experiments.yaml`, and converted through `quantum_sim.units`, which
holds CODATA constants, measured ⁸⁷Rb/²³Na scattering lengths, and the
quasi-1D/2D dimensional-reduction formulas. Plot axes are labelled in
micrometres and milliseconds.

### Domain size

There is **one** definition, in `quantum_sim.utils.domains`: the first moment
of the structure factor, `L = 2π⟨S⟩/⟨kS⟩`, with the uniform mode dropped and
a Tukey window (α from the config). Use `domain_length_um`; do not re-derive
it in a script. `first_zero(radial_autocorr(...))` is the alternative
estimator used by the scaling analysis, and is insensitive to the additive
noise that biases spectral moments.

## Working with notebooks

Notebooks under `notebooks/` are for exploration. Anything reused belongs in
`src/quantum_sim/utils/`.

Notebook **outputs** are embedded images and inflate git history permanently
(`make nbsize` shows the current cost). To commit source without outputs:

```bash
make nbstrip-on     # enable; make nbstrip-off to undo
```
