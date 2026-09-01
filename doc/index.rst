===============
FastWindTerrain
===============

Mass-consistent wind solver on a Cartesian AMReX mesh, with terrain
represented as an immersed boundary.

Velocities are stored cell-centered and the pressure/potential
(``lambda``) nodal, so a future fractional-step solver can build directly
on this layout.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   building
   grid
   terrain
   inflow
   boundary_conditions
   poisson
   anisotropy
   numerics
   output
   debugging
   parmparse_reference
   tools
   regtests
   references

Input file conventions
======================

Inputs are read with AMReX's ``ParmParse``, grouped by prefix:

===============  ==========================================================
Prefix           Covers
===============  ==========================================================
``grid.``        Mesh extents, vertical stretching, output selection
``terrain.``     Terrain file and the interpolation onto grid columns
``inflow.``      Wind profile and its parameters
``bc.``          Boundary-condition options (currently a test aid)
``numerics.``    Derivative scheme selection
``poisson.``     Variational Poisson solve and its coefficients
``anisotropy.``  Cell-local variational weights
``obrien.``      Vertical-velocity adjustment
``fwt.``         Whole-run switches, currently ``fwt.debug``
===============  ==========================================================

Any parameter can be overridden on the command line, which is how the
regtests point a case at an absolute file path::

    ./build/fastwindterrain inputs terrain.file=/abs/path/terrain.csv

An unrecognized value for an enumerated input (``grid.output_format``,
``inflow.mode``) is a fatal error rather than a silent fallback.

:doc:`parmparse_reference` lists every input in one place.

GPU support
===========

The compute kernels are written for the GPU -- every ``ParallelFor``
takes its data through device containers -- and CI compiles the CUDA, HIP
and SYCL backends on every pull request. Those jobs are **compile-only**:
hosted runners have no GPU, so nothing is executed there.

Several setup and diagnostic paths (the boundary ghost fill, the nodal
RHS averaging, the divergence diagnostics, the O'Brien column pass) are
host loops. They are correct, and under a GPU build they run on the host
through managed memory rather than being accelerated. They are not in the
solve, which is AMReX's own multigrid.
