# FastWindTerrain

Mass-consistent wind solver on a Cartesian AMReX mesh, with terrain
represented as an immersed boundary. This repo ports only the
**mass-consistent solver** core (not buildings, canopy, turbine wakes,
dispersion, EnKF, etc. from the broader `massconsistent_amr` project).

Velocities are stored cell-centered and the pressure/potential (`lambda`)
nodal, so a future fractional-step solver can build directly on this
layout.

## Status

- **Phase 1 (this PR): grid & data layout scaffolding**, plus the AMReX
  submodule and the CMake build.
  Builds the AMReX `BoxArray`/`DistributionMapping`/`Geometry` for a
  Cartesian mesh with uniform x,y spacing and a geometrically-stretched
  z spacing (finer near the surface, coarsening upward -- useful for
  resolving the ABL surface layer). See `Source/Grid.H`/`Grid.cpp`.

Later phases (terrain/IB masking, inflow profiles, directional BCs,
the variational Poisson solve, anisotropy + O'Brien adjustment,
diagnostics/output) are tracked separately and build on this scaffolding.

## Building

AMReX is bundled as a git submodule at `external/amrex` (pinned to
release `26.08`), so a fresh clone needs:

```
git submodule update --init --recursive
```

Two build systems are supported and kept configured the same way
(3D, double precision, `Src/Base` only, MPI/OpenMP off by default).

### CMake (recommended)

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8
ctest --test-dir build --output-on-failure
```

This produces `build/fastwindterrain`. Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `FWT_MPI` | `OFF` | Build with MPI |
| `FWT_OMP` | `OFF` | Build with OpenMP |
| `FWT_USE_INTERNAL_AMREX` | `ON` | Use the submodule; `OFF` uses `find_package(AMReX)` |
| `FWT_ENABLE_TESTS` | `ON` | Register the regtests with CTest |

### GNUmake (AMReX native)

```
make -j8
```

This produces `main3d.gnu.ex`. `AMREX_HOME` defaults to the submodule;
override it to build against a different checkout:

```
make AMREX_HOME=/path/to/amrex
```

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

## Output

`grid.output_format` selects how the grid is written:

| Value | Effect |
| --- | --- |
| `ascii` (default) | Plain-text grid report to `grid.report_file` (default `grid_report.txt`) |
| `plt` | AMReX native plotfile to `grid.plot_file` (default `plt_grid`) |
| `both` | Both of the above |

Any other value is a fatal error. Because AMReX's `Geometry` is uniform
in z, the plotfile's own vertical coordinate is only nominal -- the true
stretched grid is carried in the cell-centered fields `z_cc` and `dz`.

## Debugging

`fwt.debug = 1` turns on verbose diagnostics for the whole run:

- every input that was parsed, with the ones that fell back to a
  **default** marked as such
- the domain-height arithmetic (`H_requested`, `H_computed`, relative
  difference, and which of the three branches was taken)
- the full `k, z_face, dz, z_cc` table
- the index domain, `dx/dy`, periodicity, box list with owning rank,
  and cells per rank
- every file written

Default is off, and with it off the output is byte-for-byte what it was
before the switch existed. Debug lines carry a `[debug]` prefix and never
contain the words `WARNING`/`ERROR`, so they cannot confuse the regtest
checkers that key on those strings. Tables longer than 200 rows are
elided in the middle.

```
./build/fastwindterrain inputs fwt.debug=1
```

## Tools

`tools/make_terrain.py` generates synthetic terrain files in the format
the solver reads (`x,y,z` points, comma or whitespace separated, `#`
comments, optional header line -- the same format as
`massconsistent_amr`). Standard library only.

```
python3 tools/make_terrain.py --shape hill --peak 100 --sigma 150 \
    --xhi 1000 --yhi 1000 --nx 51 --ny 51 -o terrain.csv
```

Shapes: `flat`, `hill` (Gaussian), `valley`, `ridge` (Gaussian in x,
uniform in y), `slope` (constant gradient). `--jitter` displaces the
sample points off the lattice, so the output is genuinely scattered and
exercises the IDW interpolation rather than landing on grid nodes.

The shape functions are importable, so a checker can compute the
expected terrain height independently of the file:

```python
from make_terrain import elevation
z = elevation("hill", x, y, peak=100.0, sigma=150.0, xc=500.0, yc=500.0)
```

## Regtests

`regtests/` holds one folder per phase, each with its own `inputs*`
files and a standalone `check.py`. There is no separate `tests_example`
tier -- regtests are the only test suite for now.

Cases run in a scratch work directory (`build/regtests/<phase>` by
default), so running the tests leaves nothing behind in the source tree.

```
python3 run_regtests.py build/fastwindterrain
```

or to run a single phase:

```
python3 run_regtests.py build/fastwindterrain phase1_grid
```

The same tests are registered with CTest (`ctest --test-dir build`).

### phase1_grid

- `inputs_nominal` -- stretched grid, exact height match (no warning)
- `inputs_uniform` -- `stretching_ratio=1.0` regression case (must
  reproduce a plain uniform grid exactly)
- `inputs_overshoot` -- computed height exceeds requested height
  (expects non-fatal warning + `prob_hi[2]` override)
- `inputs_undershoot` -- computed height falls short of requested
  height (expects fatal abort, nonzero exit code)
- `inputs_plt` -- `output_format=both` writes both the ascii report and
  a well-formed plotfile (`z_cc`, `dz`)
- `inputs_badformat` -- an unrecognized `output_format` aborts fatally
- `inputs_debug` -- `fwt.debug=1` prints the full diagnostics, agrees
  with the ascii report, and changes no result; the default stays silent
