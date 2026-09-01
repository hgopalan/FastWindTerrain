=====
Tools
=====

``make_terrain.py``
===================

Generates synthetic terrain files in the format the solver reads (see
:doc:`terrain`). Standard library only::

    python3 tools/make_terrain.py --shape hill --peak 100 --sigma 150 \
        --xhi 1000 --yhi 1000 --nx 51 --ny 51 -o terrain.csv

Shapes
------

.. list-table::
   :widths: 14 60
   :header-rows: 1

   * - Shape
     - Description
   * - ``flat``
     - Constant elevation
   * - ``hill``
     - Gaussian hill
   * - ``valley``
     - Gaussian valley
   * - ``ridge``
     - Gaussian in x, uniform in y
   * - ``slope``
     - Constant gradient plane

``--jitter`` displaces the sample points off the lattice, so the output
is genuinely scattered and exercises the interpolation rather than
landing on grid nodes. Points are clamped to stay inside the domain.

Two details matter when generating test data:

* Make the terrain point spacing differ from the grid spacing, so cell
  centers never coincide with a terrain point. Otherwise the
  interpolation degenerates into a lookup and is not really tested.
* The default output precision is 10 decimals, which keeps the write
  error far below the tolerances the regtest checkers use.

Importable shape functions
--------------------------

The shape functions are plain, side-effect-free callables, so a checker
can compute an expected elevation analytically rather than trusting the
same code path the solver read:

.. code-block:: python

    from make_terrain import elevation
    z = elevation("hill", x, y, peak=100.0, sigma=150.0, xc=500.0, yc=500.0)
