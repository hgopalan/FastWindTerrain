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
   output
   debugging
   tools
   regtests

Input file conventions
======================

Inputs are read with AMReX's ``ParmParse``, grouped by prefix:

===============  ==========================================================
Prefix           Covers
===============  ==========================================================
``grid.``        Mesh extents, vertical stretching, output selection
``terrain.``     Terrain file and the interpolation onto grid columns
``inflow.``      Wind profile and its parameters
``fwt.``         Whole-run switches, currently ``fwt.debug``
===============  ==========================================================

Any parameter can be overridden on the command line, which is how the
regtests point a case at an absolute file path::

    ./build/fastwindterrain inputs terrain.file=/abs/path/terrain.csv

An unrecognized value for an enumerated input (``grid.output_format``,
``inflow.mode``) is a fatal error rather than a silent fallback.
