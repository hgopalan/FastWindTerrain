=======
Terrain
=======

The terrain surface is read from a scattered point file, interpolated
onto each grid column, and used to mark cells below the surface as
solid.

Inputs
======

.. list-table::
   :widths: 30 12 45
   :header-rows: 1

   * - Input
     - Default
     - Meaning
   * - ``terrain.file``
     - *(none)*
     - ``x,y,z`` point file. Absent means flat ground
   * - ``terrain.flat_elevation``
     - ``0.0``
     - Ground elevation when no file is given [m]
   * - ``terrain.idw_n_neighbors``
     - ``6``
     - Nearest points used by the interpolation
   * - ``terrain.idw_exponent``
     - ``2.0``
     - IDW power ``p``; weight ``= d^-p``
   * - ``terrain.extrapolation``
     - ``idw``
     - What a column outside the point cloud gets: ``idw`` or ``nearest``

File format
===========

One ``x y z`` point per line, comma **or** whitespace separated, ``#``
comments stripped, and any line that does not parse as three numbers
skipped -- so a leading ``x,y,z`` header is fine.

.. code-block:: none

    # a small terrain file
    x,y,z
    0.0,0.0,0.0
    20.0,0.0,1.4
    40.0,0.0,5.6

This is the format ``massconsistent_amr``'s ``read_terrain_file``
accepts, so files are interchangeable between the two codes.

A named file that cannot be opened, or that yields no points, is a fatal
error rather than a silent fall back to flat ground.

Generate synthetic terrain with ``tools/make_terrain.py``; see
:doc:`tools`.

Interpolation
=============

Terrain height at a column center is the inverse-distance-weighted
average of the ``k`` nearest input points:

.. code-block:: none

    z_terrain(x,y) = sum_i w_i z_i / sum_i w_i,   w_i = d_i^-p

over the ``k`` nearest points by squared distance. A query landing on an
input point (within ``1e-12`` in squared distance) returns that point's
elevation exactly. This is a port of ``massconsistent_amr``'s
``idw_terrain``, so the two codes agree point for point.

``z_terrain`` is a two-dimensional field but is stored in a normal
cell-centered ``MultiFab`` replicated along ``k``. That costs ``nz``
times more memory than a column array and buys two things worth more at
this scale: direct plotfile output, and uniform ``(i,j,k)`` indexing in
every kernel with no special-cased ``k``.

Columns the point cloud does not cover
======================================

IDW is an interpolation. Asked for a height at a column that no input
point brackets it still answers, with a distance-weighted average of
points that all lie to one side -- a smooth surface, and an arbitrary
one. Because the mask below is only ``z_cc <= z_terrain``, a wrong
elevation out there does not look wrong: the column simply comes out all
fluid or all solid, with nothing to say why.

``terrain.extrapolation`` chooses what those columns get:

.. list-table::
   :widths: 16 74
   :header-rows: 1

   * - Value
     - Height at a column outside the cloud
   * - ``idw``
     - The same ``k``-nearest average as everywhere else. The
       **default**, so existing cases are bit-for-bit unchanged
   * - ``nearest``
     - The elevation of the single nearest input point

"Outside" means outside the **axis-aligned extent** of the points --
``x < x_min``, ``x > x_max``, ``y < y_min`` or ``y > y_max``, computed
from the cloud itself. That test, rather than "no point within radius
``R``", for three reasons:

* it needs no length scale from the user, and so cannot be set wrong;
* the bounding box contains the convex hull, so a column it calls
  outside really is being extrapolated -- the fallback never replaces a
  genuine interpolation;
* a radius test would fire wherever the cloud is merely *sparse*,
  quietly turning good interior data into a nearest-point staircase,
  while still missing the column just past the edge of a dense cloud,
  which is where extrapolation goes worst.

The comparison is strict, so a column sitting exactly on the extent is
interpolated.

Either way the run reports how many columns fell outside, as
``terrain_n_columns_outside`` in the report and as
``Terrain.n_columns_outside`` from Python. Under ``idw`` a non-zero count
also raises a warning naming this input; under ``nearest`` it prints a
note. A case whose terrain data carries a margin past the domain has no
such column and is unaffected by any of this -- ``cases/casegen.py`` and
``cases/corpus.py`` keep 250 m of points beyond the domain for exactly
that reason.

``nearest`` extends the surface as a piecewise-constant (Voronoi) fill
of the boundary elevations, so it is not smooth out there. It is not
meant to be: it is the nearest thing actually measured, which is a
defensible answer where an average of one-sided points is not. Terrain
data that reaches past the domain is still the right fix; this is the
guard for when it does not.

The immersed-boundary mask
==========================

The mask is **binary** -- there are no partial volume fractions:

.. code-block:: none

    mask(i,j,k) = 1 (solid)  if  z_cc(k) <= z_terrain(i,j)
                  0 (fluid)  otherwise

matching ``massconsistent_amr``'s ``is_solid = (z_cc - z_terrain <= 0)``,
so a cell center sitting exactly on the surface is solid. ``z_cc`` is the
true stretched cell-center height (see :doc:`grid`), not
``geom().CellSize(2)``.

Terrain may intersect any lateral boundary, which leaves that face only
partly open. That is a supported configuration, not an error; see
:doc:`inflow` for what it means for the boundary mass flux.
