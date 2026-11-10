# Hamiltonian Simulation from Scratch: How This Codebase Works

This guide assumes you have never done a classical simulation of a quantum
system. By the end you should understand every design decision in this
repository and be able to rebuild it yourself. Read it top to bottom once,
then again with the source files open next to it.

---

## 1. What "Hamiltonian simulation" means here

A quantum system evolves by the Schrödinger equation

```
i ħ ∂ψ/∂t = H ψ
```

where `H` (the Hamiltonian) is the energy operator. "Hamiltonian simulation"
means: given `H` and an initial state `ψ(x, 0)`, compute `ψ(x, t)` on a
computer. Formally the answer is `ψ(t) = exp(-i H t / ħ) ψ(0)`, so the whole
game is **applying the exponential of an operator to a wavefunction,
efficiently and accurately**.

For a Bose–Einstein condensate (BEC) — millions of atoms cooled into the same
quantum state — we don't track every atom. In the mean-field approximation all
atoms share one wavefunction `ψ(x, t)`, and collisions between atoms show up
as a *nonlinear* term. The result is the **Gross–Pitaevskii equation (GPE)**:

```
i ħ ∂ψ/∂t = [ -ħ²∇²/2m  +  V(x)  +  g |ψ|² ] ψ
              kinetic      trap     interactions
```

- `V(x)` is the trapping potential (usually harmonic, `V = ½ m ω² x²`).
- `g |ψ|²` is the mean-field interaction: each atom feels the density of the
  others. `g` is fixed by the measured s-wave scattering length `a_s` via
  `g = 4πħ²a_s/m` in 3D.

A *spinor* BEC has atoms with internal (spin) states, so `ψ` becomes a small
vector of wavefunctions — 2 components in `gpe_solver.py`, 3 in
`spinor1_gpe_2d.py` — with extra terms coupling the components (Rabi drive,
spin-exchange collisions, Zeeman shifts). The machinery below is identical;
only the "local" part of `H` gets richer.

---

## 2. Step 0 before any code: choose your units

Never put SI numbers directly into a solver. `ħ = 1.05e-34` and
`x ~ 1e-6` differ by 28 orders of magnitude; floating point hates that, and
every formula grows constant clutter. Instead, **nondimensionalize**: pick a
natural length, time, and energy scale for your problem, divide them out, and
simulate equations where everything is O(1).

For a harmonic trap of frequency `ω_x` the natural choices are

| quantity | unit                          | value for ⁸⁷Rb at ω_x = 2π·20 Hz |
|----------|-------------------------------|----------------------------------|
| length   | `a_ho = sqrt(ħ/(m ω_x))`      | 2.41 μm                          |
| time     | `1/ω_x`                       | 7.96 ms                          |
| energy   | `ħ ω_x`                       | h × 20 Hz                        |

Substituting `x → a_ho·x̃`, `t → t̃/ω_x` into the GPE turns it into the clean
dimensionless form the solver actually integrates:

```
i ∂ψ̃/∂t̃ = [ -½ ∂²/∂x̃² + ½ x̃² + g̃ |ψ̃|² ] ψ̃
```

All the physics is now inside one dimensionless number
`g̃ = g_1D N_atoms / (ħ ω_x a_ho)` (for a wavefunction normalized to 1).

**Where to look:** `src/quantum_sim/units.py` holds the real constants
(scattering lengths in Bohr radii, atomic masses, trap frequencies) and the
conversion helpers. The scripts declare laboratory parameters in SI — e.g.
`OMEGA_X = 2π × 20 Hz`, `A_RB87_UP_UP = 100.40 a_Bohr` — convert them to
dimensionless numbers for the solver, and convert back to μm/ms for the plot
axes. That's the pattern to copy: **SI at the edges, dimensionless in the
core.**

Two more conversions appear because our simulations are lower-dimensional
than reality. A cigar-shaped trap freezes radial motion, giving an effective
1D coupling `g_1D = 2ħω_⊥ a_s`; a pancake trap freezes the z-motion, giving
`g_2D = g_3D / (√(2π) a_z)`. Both are in `units.py`.

---

## 3. Discretize: the wavefunction becomes an array

Put the system in a box of length `L` with `N` grid points, spacing
`dx = L/N` (see `SpinorGPESolver.__init__`):

```python
x = np.linspace(-L/2, L/2, N, endpoint=False)   # periodic grid
psi = np.zeros((2, N), dtype=complex)            # 2 spin components
```

Because we will use FFTs, the grid is **periodic**: the right edge wraps to
the left. That imposes two rules:

1. `L` must be big enough that the wavefunction is ~0 at the edges
   (otherwise the cloud "leaks" around and interferes with itself).
2. `dx` must be small enough to resolve the shortest length in the problem
   (healing length, the fastest-growing instability wavelength, ...).
   Equivalently, in Fourier space you can only represent momenta up to
   `k_max = π/dx`.

The Fourier grid comes for free:

```python
k = 2 * np.pi * np.fft.fftfreq(N, dx)   # momenta of each FFT mode
```

Derivatives are trivial in Fourier space: `∂²/∂x² ψ ↔ -k² ψ̂`. That is the
whole reason for the method that follows.

---

## 4. The split-step Fourier method (the heart of everything)

The Hamiltonian has two parts that are each *easy* in the right basis:

- **Kinetic** `T = k²/2` — diagonal in momentum space. Applying
  `exp(-i T dt)` is just multiplying `ψ̂(k)` by a phase `exp(-i k²/2 dt)`.
- **Local** `U = V(x) + g|ψ|² (+ spin terms)` — diagonal (or a small matrix)
  at each point in position space. Applying `exp(-i U dt)` is multiplying
  `ψ(x)` by a phase at each grid point.

They don't commute, so `exp(-i(T+U)dt) ≠ exp(-iTdt)exp(-iUdt)` exactly — but
for small `dt` the **Strang (symmetric) splitting** is accurate to O(dt³) per
step:

```
exp(-i(T+U)dt) ≈ exp(-iT dt/2) · exp(-iU dt) · exp(-iT dt/2)
```

One time step is therefore (see `SpinorGPESolver.step`):

```
1. FFT ψ           -> ψ̂(k)
2. ψ̂ *= exp(-i k²/2 · dt/2)          # half kinetic
3. inverse FFT      -> ψ(x)
4. ψ  *= exp(-i [V + g|ψ|²] · dt)     # full local step, density frozen
5. FFT, half kinetic again, inverse FFT
```

Why this method is beloved:

- **Exactly unitary** — every factor is a pure phase, so the norm is
  conserved to machine precision (test 2 in the suite checks 10⁻⁸ over 1000
  steps).
- **Spectrally accurate in space** — derivatives via FFT are exact for
  band-limited functions; error drops exponentially with `N`.
- **Stable** — no CFL-style blow-up if `dt` is too big; you just lose
  accuracy, which the energy-conservation check catches.

Two practical refinements used in this codebase:

- The nonlinear density `|ψ|²` is **frozen during each local step** (it's
  part of the Hamiltonian, so it can't consistently change mid-exponential).
- Adjacent steps end and begin with half kinetic steps; `Spin1GPE2D.evolve`
  **fuses** them into full steps between observation times, halving the FFT
  count. The kinetic phase array is also cached while `dt` is unchanged —
  recomputing `exp` over the grid each step is pure waste.

**The spinor extension.** When `H_local` mixes components it's a small matrix
at each grid point, and you exponentiate it *exactly*:

- Spin-½ Rabi coupling `-Ω σ_x`: `exp(iΩdt σ_x) = cos(Ωdt) I + i sin(Ωdt) σ_x`
  (see `_local_step` in `gpe_solver.py`).
- Spin-1 spin-exchange `g_s F·f`: a 3×3 rotation done in closed form using
  the identity `(n̂·f)³ = n̂·f` (see `_spin_rotation` in `spinor1_gpe_2d.py`) —
  no per-pixel matrix diagonalization needed.

Non-commuting pieces *within* the local step are themselves Strang-split
(look at the `q/2 — spin rotation — q/2` sandwich in `_local_step`).

---

## 5. Finding ground states: imaginary time

Real-time evolution needs an initial state. Usually you want the *ground
state* of the interacting system, which has no closed form. The trick:
replace `dt → -i dτ`. Then

```
ψ(τ) = exp(-H τ) ψ(0) = Σₙ cₙ e^(-Eₙ τ) φₙ
```

Every energy eigenstate decays, but **excited states decay faster**, so any
initial guess relaxes exponentially toward the ground state. The evolution is
no longer unitary, so you **renormalize after every step** (that's the
`if dt.imag != 0: self._normalize()` in `step`). You've converged when the
energy stops changing; because this is a gradient flow, the energy decreases
*monotonically* (test 6 verifies that).

The workflow in `run_simulation_ho.py` is the canonical experiment shape:

1. **Imaginary time** with the "before" Hamiltonian → ground state.
2. **Quench**: suddenly change a parameter (shift the trap, crank the Rabi
   drive, flip the Zeeman energy `q` negative).
3. **Real time** with the "after" Hamiltonian → watch the dynamics.

---

## 6. Trust but verify: the validation ladder

A simulation that runs is not a simulation that's right. Every check in
`tests/test_gpe_solver.py` compares against something known *independently*:

| check | analytic truth |
|---|---|
| ITP, no interactions | harmonic ground state `π^(-1/4) e^(-x²/2)`, `E = ½ħω` |
| real-time norm | exactly 1 forever (unitarity) |
| real-time energy | conserved (drift is O(dt²), should be tiny) |
| pure Rabi drive | populations transfer as `sin²(Ωt)` exactly |
| free wavepacket | variance grows as `σ₀² + t²/(4σ₀²)` |
| ITP energy trace | monotonically decreasing (variational principle) |

The scripts also print conservation checks after long runs (the sodium run
holds the norm to 10⁻¹² and magnetization to machine precision over 40,000
steps). **Adopt this habit**: for every new Hamiltonian term, find a limit
where the answer is known and write the test before trusting any plot.

Convergence checks you should also run once per new problem: halve `dt` and
double `N` — if the observables move, your resolution was inadequate.

---

## 7. The full recipe (what to do when you build your own)

1. **Write down the Hamiltonian** and its energy functional. Identify the
   kinetic part and the local part (potential + interactions + spin terms).
2. **Pick real experimental parameters** (atom, scattering lengths, trap
   frequencies, atom number, densities) from the literature, with units.
3. **Nondimensionalize.** Choose length/time/energy units natural to the
   physics; compute the handful of dimensionless numbers the solver needs.
   Keep the conversion code separate (`units.py`) and print both forms.
4. **Choose the grid.** Estimate the smallest length scale (healing length
   `ξ = ħ/√(2mgn)`, instability wavelength `2π/k*`) → sets `dx`. Estimate
   the largest (cloud size) → sets `L`. Use powers of two for `N`.
5. **Choose `dt`.** Every phase applied per step should stay well under
   1 radian: `dt < 1/max(k²_max/2, |V+gn|_max)`. Then verify by halving it.
6. **Implement the split step**: FFT → half kinetic phase → inverse FFT →
   local phase / small exact matrix exponential → half kinetic. Vectorize
   with numpy (no Python loops over grid points); cache loop-invariant
   phase arrays.
7. **Implement diagnostics**: norm, energy (kinetic via Parseval in k-space),
   populations, and whatever your physics conserves (magnetization here).
8. **Ground state via imaginary time**, renormalizing each step, until the
   energy plateaus.
9. **Validate** against every analytic limit you can find (Section 6).
10. **Only then run the real experiment**: quench, propagate, record
    observables at intervals (not every step — memory), plot with axes
    converted back to μm and ms.

---

## 8. Guided tour of the code

- `src/quantum_sim/units.py` — SI constants, measured ⁸⁷Rb/²³Na scattering
  lengths, quasi-1D/2D reductions, the two unit systems. *Read first.*
- `src/quantum_sim/gpe_solver.py` — the simplest full solver (spin-½, 1D).
  ~200 lines; everything in Sections 3–5 maps line-by-line onto it.
- `src/quantum_sim/spinor1_gpe_2d.py` — the same skeleton scaled up: 2D
  grid, 3 components, exact spin-1 rotation, fused `evolve()` stepping.
- `scripts/run_simulation_ho.py` — verify → ground state → quench → dynamics
  for a Rabi-coupled ⁸⁷Rb mixture, real units in and out.
- `scripts/run_sodium_domains.py` — a genuine physics experiment: quench a
  polar ²³Na condensate to `q/h ≈ -70 Hz` and watch spin domains form and
  coarsen (compare Williamson & Blakie, PRL 116, 025301 (2016)).
- `tests/test_gpe_solver.py` — the validation ladder.

Common pitfalls to remember: wavefunction touching the periodic box edge;
forgetting to renormalize in imaginary time (norm explodes/vanishes);
recording every step (memory); reporting dimensionless numbers as if they
were SI; trusting a pretty plot that no analytic limit ever checked.

## 9. Where to read more

- C. J. Pethick & H. Smith, *Bose–Einstein Condensation in Dilute Gases* —
  the standard textbook for the GPE and its physics.
- W. Bao, D. Jaksch, P. A. Markowich, J. Comput. Phys. 187, 318 (2003) —
  the split-step method for the GPE, with error analysis.
- Y. Kawaguchi & M. Ueda, Phys. Rep. 520, 253 (2012) — spinor BEC review
  (all the spin-1 terms used here).
- L. A. Williamson & P. B. Blakie, PRL 116, 025301 (2016) — the coarsening
  physics the sodium script reproduces.
