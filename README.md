# FastWindTerrain

Mass-consistent wind solver on a Cartesian AMReX mesh, with terrain
represented as an immersed boundary.

Velocities are stored cell-centered and the pressure/potential (`lambda`)
nodal, so a future fractional-step solver can build directly on this
layout.

## Quick start

AMReX is bundled as a submodule, so a fresh clone needs:

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

## Documentation

Full documentation lives in [`doc/`](doc/):

| Topic | |
| --- | --- |
| [Building](doc/building.rst) | CMake and GNUmake, build options |
| [Grid](doc/grid.rst) | Vertical stretching and the domain-height policy |
| [Terrain](doc/terrain.rst) | Terrain files, interpolation, the immersed-boundary mask |
| [Inflow](doc/inflow.rst) | Wind profiles, AGL anchoring, boundary mass flux |
| [Boundary conditions](doc/boundary_conditions.rst) | Face classification, ghost values, lambda conditions |
| [Poisson solver](doc/poisson.rst) | The variational solve, anisotropy, and the stretched grid |
| [Anisotropy](doc/anisotropy.rst) | Cell-local weights and the O'Brien adjustment |
| [Numerics](doc/numerics.rst) | Derivative schemes and their verification |
| [Convergence study](doc/convergence.rst) | Scheme order of accuracy, measured through the solver |
| [Input reference](doc/parmparse_reference.rst) | Every input in one place; `regtests/inputs_master` is the runnable version |
| [References](doc/references.rst) | The methods, and where each is used |
| [Output](doc/output.rst) | Diagnostics, the report, and the plt/ascii field backends |
| [Debugging](doc/debugging.rst) | The `fwt.debug` diagnostics switch |
| [Tools](doc/tools.rst) | Synthetic terrain generation |
| [Regtests](doc/regtests.rst) | Test suite and how to run it |

To build the docs as HTML:

```
sphinx-build -b html doc doc/_build
```

## Continuous integration

Every pull request builds with both build systems and runs the whole
regtest suite:

- **CMake** on Linux (Release and Debug) and macOS (Release), with CTest
- **GNUmake** on Linux, which nothing else exercises and would otherwise
  rot unnoticed
- **CUDA, HIP and SYCL** builds, compile-only: hosted runners have no
  GPU, so these prove the code still compiles for each backend and stop
  there
- a check that a full build and test run leaves the working tree clean
- the documentation, built with warnings as errors

## Layout

```
Source/      solver source
doc/         documentation
regtests/    one directory per test group, each self-contained
tools/       helper scripts
external/    AMReX submodule
```
