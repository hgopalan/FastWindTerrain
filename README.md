# FastWindTerrain

Mass-consistent wind solver on a Cartesian AMReX mesh, with terrain
represented as an immersed boundary.

Velocities are stored cell-centered and the pressure/potential (`lambda`)
nodal, so a future fractional-step solver can build directly on this
layout.

## Quick start

AMReX and pybind11 are bundled as submodules, so a fresh clone needs:

```
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8
ctest --test-dir build --output-on-failure
```

Then run a case:

```
./build/fastwindterrain inputs
```

A GNUmake build is also supported (`make -j8`, producing `main3d.gnu.ex`).
It builds the C++ solver only — the Python bindings are CMake-only.

## Python bindings

Install them:

```
pip install .
```

A conda environment with everything the build needs is in
`environment.yml`:

```
conda env create -f environment.yml && conda activate fastwindterrain
```

Or build them in tree — off by default, turned on with `FWT_PYTHON`:

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFWT_PYTHON=ON
cmake --build build -j 8
```

That lays the package out under `build/python`, so an import needs only
`PYTHONPATH`:

```
PYTHONPATH=build/python python3 -c "import fastwindterrain as fwt; print(fwt.__version__)"
```

Run a case from Python, or equivalently straight from a shell:

```
PYTHONPATH=build/python python3 -m fastwindterrain inputs poisson.alpha_v=0.3
```

The bindings link the **same compiled library** as the executable rather
than recompiling the solver, so the two paths share object files and not
merely source. That is what makes their results bit-for-bit identical —
a claim the test suite checks rather than assumes.
`build/fastwindterrain-py` is an argv-compatible stand-in for the
executable, so the entire regtest suite runs through Python unchanged:

```
python3 run_regtests.py build/fastwindterrain-py
```

Components can also be built directly from Python, with no inputs file:

```python
import fastwindterrain as fwt

with fwt.session():
    g = fwt.Grid({"n_cell": (24, 24, 40),
                  "prob_lo": (0.0, 0.0, 0.0),
                  "prob_hi": (1000.0, 1000.0, 483.19909696997223),
                  "dz0": 4.0, "stretching_ratio": 1.05})
    print(g.z_cc)          # numpy, (nz,)
```

A whole case can be described in Python, with no inputs file anywhere —
the terrain point cloud and the profile go through the same
interpolation the file readers feed, so the result is bit-for-bit what
the files would have produced:

```python
with fwt.session():                       # no arguments: ParmParse is empty
    g = fwt.Grid({"n_cell": (24, 24, 40), ...})
    t = fwt.Terrain(g, {"points": pts})   # (n, 3) numpy array
    inf = fwt.Inflow(g, t, {"mode": "powerlaw", "u_ref": 8.0, "v_ref": 6.0})
```

A whole case can be handed to the solver as one nested dict, and driven
a step at a time:

```python
with fwt.session():
    s = fwt.Solver({"grid": {...}, "terrain": {"points": pts},
                    "inflow": {"u_ref": 8.0, "v_ref": 6.0},
                    "poisson": {"alpha_v": 0.5, "n_projections": 4}})
    s.setup()
    for _ in range(4):
        s.project_once()
        print(s.max_divergence_fe, s.solve_iterations)
    s.diagnose()
```

The variational weights can also be built on their own, which is what
choosing `slope_scale` and `decay_height` actually needs:

```python
a = fwt.Anisotropy(grid, terrain, {"enable": True, "slope_scale": 0.5,
                                   "alpha_v_base": 0.5})
a.alpha_v          # (nz, ny, nx)
```

Output comes back in memory, no file round-trip needed — the same object
the plotfile and ascii backends are handed:

```python
s.setup(); s.solve(); s.diagnose()
f = s.fields()                    # {name: numpy array}, 17 fields
s.write_plotfile("plt_case")      # still there, for VisIt/ParaView/yt
```

Fields come back as numpy, channels-first, so they hand straight to
PyTorch:

```python
with fwt.session(["inputs"]):
    s = fwt.Solver()
    s.setup()
    u = s.velocity[0]           # (nz, ny, nx)
    s.set_velocity(new_field)   # ghosts refilled through the BCs
```

That path never touches ParmParse, which matters when driving many cases
in one process: ParmParse persists for the life of an AMReX
initialization, so a case that omits a parameter would otherwise inherit
whatever an earlier case set. An unknown key raises rather than being
ignored, a bad input raises instead of aborting the interpreter, and a
domain-height overshoot is a `UserWarning`.

### Generating a dataset

Many cases in one process, on one fixed grid, stacked into an array a
neural operator can train on — the reason the bindings exist:

```python
from fastwindterrain import dataset

configs = dataset.sweep(base, {"inflow.u_ref": [4.0, 8.0, 12.0],
                               "inflow.v_ref": [0.0, 6.0]})

dataset.generate(configs, "wind.npz", fields=["u", "v", "w", "mask"],
                 dtype="float32", seed=0)
```

The grid is checked rather than assumed: sweeping a grid parameter
raises, and so does a config whose grid section differs from the first
one's. A ragged dataset does not fail at generation time — it fails hours
into a training run, or never, because the loader padded.

`examples/quickstart.ipynb` is the tour, and CI executes it.

### Real terrain

`cases/` holds eight 5 × 5 km domains over real ground, at wildfire
locations — Creek, Dixie, Rim, August Complex, Tubbs, Thomas, Woolsey and
Bootleg:

```
pip install ".[cases]"
python3 cases/creek_fire/prepare.py    # download SRTM, derive the grid
python3 cases/creek_fire/run.py        # solve it, with sanity checks
```

Cell counts are the same for every case, but the vertical extent follows
each case's own relief — the floor sits on the ground rather than at sea
level, which matters when the ground is at 2000 m. The solver does not
check that terrain fits inside the domain, so `cases/` does.

See [Real terrain cases](docs/cases.rst).

### Tests

```
pytest tests
```

The bindings' own suite. The C++ regtest groups stay on
`run_regtests.py`; see [Regtests](docs/regtests.rst) for why the two are
split.

numpy is a runtime requirement of the bindings. On a system Python that
refuses installs (PEP 668), point CMake at a virtual environment:

```
python3 -m venv build/venv && build/venv/bin/pip install numpy
cmake -S . -B build -DFWT_PYTHON=ON -DPython3_EXECUTABLE=$PWD/build/venv/bin/python
```

See [Python bindings](docs/python.rst) for the API and what parity does
and does not cover.

## Documentation

Rendered at **<https://hgopalan.github.io/FastWindTerrain/>**. The sources
live in [`docs/`](docs/):

| Topic | |
| --- | --- |
| [Building](docs/building.rst) | CMake and GNUmake, build options |
| [Grid](docs/grid.rst) | Vertical stretching and the domain-height policy |
| [Terrain](docs/terrain.rst) | Terrain files, interpolation, the immersed-boundary mask |
| [Inflow](docs/inflow.rst) | Wind profiles, AGL anchoring, boundary mass flux |
| [Boundary conditions](docs/boundary_conditions.rst) | Face classification, ghost values, lambda conditions |
| [Poisson solver](docs/poisson.rst) | The variational solve, anisotropy, and the stretched grid |
| [Anisotropy](docs/anisotropy.rst) | Cell-local weights and the O'Brien adjustment |
| [Numerics](docs/numerics.rst) | Derivative schemes and their verification |
| [Convergence study](docs/convergence.rst) | Scheme order of accuracy, measured through the solver |
| [Python bindings](docs/python.rst) | Building, using, and the C++/Python parity guarantee |
| [Input reference](docs/parmparse_reference.rst) | Every input in one place; `regtests/inputs_master` is the runnable version |
| [References](docs/references.rst) | The methods, and where each is used |
| [Output](docs/output.rst) | Diagnostics, the report, and the plt/ascii field backends |
| [Debugging](docs/debugging.rst) | The `fwt.debug` diagnostics switch |
| [Tools](docs/tools.rst) | Synthetic terrain generation |
| [Real terrain cases](docs/cases.rst) | The eight-case SRTM catalogue in `cases/` |
| [Regtests](docs/regtests.rst) | Test suite and how to run it |

To build the docs as HTML:

```
make -C docs html          # output in docs/_build/html
```

## Continuous integration

Every pull request builds with both build systems and runs the whole
regtest suite:

- **CMake** on Linux (Release and Debug) and macOS (Release), with CTest
- **GNUmake** on Linux, which nothing else exercises and would otherwise
  rot unnoticed
- **Wheel** on Linux and macOS: `pip install .`, then the bindings test
  suite against the *installed* package, plus the example notebook
  executed top to bottom
- **Python bindings** on Linux, running the entire regtest suite a second
  time through the bindings and asserting the outputs are byte-identical
- **CUDA, HIP and SYCL** builds, compile-only: hosted runners have no
  GPU, so these prove the code still compiles for each backend and stop
  there
- a check that a full build and test run leaves the working tree clean
- the documentation, built with warnings as errors, and published to
  GitHub Pages from `main`

## Layout

```
Source/      solver source (built as one library, fwt_core)
python/      pybind11 bindings and the Python package
docs/        documentation
regtests/    one directory per C++ test group, each self-contained
cases/       real-terrain case catalogue (SRTM, 5 x 5 km)
tests/       pytest suite for the Python bindings
examples/    quickstart notebook
convergence/ the scheme convergence sweep
tools/       helper scripts
external/    AMReX and pybind11 submodules
```
