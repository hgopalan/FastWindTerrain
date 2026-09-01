# FastWindTerrain

Mass-consistent wind solver on a Cartesian AMReX mesh, with terrain
represented as an immersed boundary. This repo ports only the
**mass-consistent solver** core (not buildings, canopy, turbine wakes,
dispersion, EnKF, etc. from the broader `massconsistent_amr` project).

Velocities are stored cell-centered and the pressure/potential (`lambda`)
nodal, so a future fractional-step solver can build directly on this
layout.

## Status

- **Phase 1 (this PR): grid & data layout scaffolding.**
  Builds the AMReX `BoxArray`/`DistributionMapping`/`Geometry` for a
  Cartesian mesh with uniform x,y spacing and a geometrically-stretched
  z spacing (finer near the surface, coarsening upward -- useful for
  resolving the ABL surface layer). See `Source/Grid.H`/`Grid.cpp`.

Later phases (terrain/IB masking, inflow profiles, directional BCs,
the variational Poisson solve, anisotropy + O'Brien adjustment,
diagnostics/output) are tracked separately and build on this scaffolding.

## Building

Requires an AMReX source checkout. Point `AMREX_HOME` at it (or place
it at `../amrex` relative to this repo):

```
export AMREX_HOME=/path/to/amrex
make -j4
```

This produces an executable named like `main3d.gnu.ex`.

## Grid stretching

The vertical grid is geometric: `dz(k) = dz0 * r^k` for `k = 0..nz-1`,
where `dz0` is the surface-adjacent cell thickness and `r` is
`grid.stretching_ratio` (default `1.0`, i.e. uniform). `nz`, `dz0`, `r`,
and the requested domain height are all independent inputs; the code
validates them after summing the geometric series to the actual height
`H_computed`:

- `H_computed` matches the requested height (within tolerance): proceeds normally.
- `H_computed` **exceeds** the requested height: **non-fatal warning**,
  and `grid.prob_hi[2]` is overridden to `H_computed` so the grid and
  domain agree exactly.
- `H_computed` **falls short** of the requested height: **fatal abort**
  (increase `n_cell[2]`, `dz0`, or `stretching_ratio`).

Example `inputs`:

```
grid.n_cell           = 40 40 66
grid.prob_lo          = 0.0 0.0 0.0
grid.prob_hi          = 1000.0 1000.0 961.2758234855
grid.dz0              = 2.0
grid.stretching_ratio = 1.05
grid.max_grid_size    = 32
grid.report_file      = grid_report.txt
```

## Regtests

`regtests/` holds one folder per phase, each with its own `inputs*`
files and a standalone `check.py`. There is no separate `tests_example`
tier -- regtests are the only test suite for now.

```
python3 run_regtests.py /path/to/main3d.gnu.ex
```

or to run a single phase:

```
python3 run_regtests.py /path/to/main3d.gnu.ex phase1_grid
```

### phase1_grid

- `inputs_nominal` -- stretched grid, exact height match (no warning)
- `inputs_uniform` -- `stretching_ratio=1.0` regression case (must
  reproduce a plain uniform grid exactly)
- `inputs_overshoot` -- computed height exceeds requested height
  (expects non-fatal warning + `prob_hi[2]` override)
- `inputs_undershoot` -- computed height falls short of requested
  height (expects fatal abort, nonzero exit code)
