======
Output
======

``grid.output_format`` selects how a run is written:

.. list-table::
   :widths: 22 60
   :header-rows: 1

   * - Value
     - Effect
   * - ``ascii`` (default)
     - Plain-text report to ``grid.report_file`` (default ``grid_report.txt``)
   * - ``plt``
     - AMReX native plotfile to ``grid.plot_file`` (default ``plt_grid``)
   * - ``both``
     - Both of the above

Any other value is a fatal error.

Plotfile
========

The plotfile carries these cell-centered fields, in order:

.. list-table::
   :widths: 16 60
   :header-rows: 1

   * - Field
     - Meaning
   * - ``z_cc``
     - True (stretched) cell-center height [m]
   * - ``dz``
     - True cell thickness [m]
   * - ``terrain_z``
     - Terrain surface height for the column [m]
   * - ``mask``
     - 1 solid, 0 fluid
   * - ``u``
     - Velocity x-component [m/s]
   * - ``v``
     - Velocity y-component [m/s]
   * - ``w``
     - Velocity z-component [m/s]

``z_cc`` and ``dz`` are written because AMReX's ``Geometry`` is uniform
in z, so the plotfile's own vertical coordinate is only nominal. A
visualization tool needs these fields to place cells at their true
heights.

Fields are appended to the end of this list as the solver grows, so
anything reading a plotfile should look components up **by name** rather
than assuming a count.

Report
======

The ascii report is a plain-text ``key value`` listing: grid parameters,
the full ``z_face`` and ``z_cc`` arrays, then a terrain summary and an
inflow summary. Every value is written at full round-trip precision, so
it can be compared against analytic formulas without loss.

It is written by the regtest checkers and is the simplest way to see
what a run actually resolved::

    grep terrain_ grid_report.txt
    grep inflow_ grid_report.txt
