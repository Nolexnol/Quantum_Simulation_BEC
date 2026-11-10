# data/

**Nothing in this directory is version-controlled.** It holds lab output and
derived caches, both of which are too large and too opaque for git. The
directory structure is tracked (via these README files); the contents are not.

```
data/
├── raw/        measured shots, exactly as the lab wrote them — treat as READ-ONLY
├── processed/  reduced arrays that are expensive to recompute
├── db/         SQLite index over raw/ (a cache: delete and rebuild freely)
└── archive/    compressed originals kept for provenance
```

## raw/

One directory per acquisition run, named in `config/experiments.yaml`:

| directory | dataset key | contents |
|---|---|---|
| `coarsening_data/` | `coarsening` | 23Na binary mixture in the 113 um box, 2026-07-17 |
| `2026-07-27_40ms/` | `box40ms` | 40 ms hold-time series, 2026-07-27 |
| `700p_shots/2026_08_19/` | `shots700p` | ~750-shot ensemble, 2026-08-19 |
| `2026-09-11/` | `sep11` | ~730-shot ensemble, 2026-09-11, 0-2000 ms |

Never edit a file under `raw/`. Every analysis must be reproducible from it.

## Pointing the code somewhere else

Paths are resolved by `quantum_sim.config`, so moving this directory to an
external drive needs no code change:

```bash
export QUANTUM_SIM_DATA_DIR=/Volumes/lab_drive/quantum_sim_data
```

## Backups

`raw/` and `archive/` are the only irreplaceable things in this repository —
`processed/` and `db/` regenerate, and so does everything in `results/`. Back
those two up somewhere off this machine.
