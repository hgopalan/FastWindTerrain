======
Output
======

A run produces two things, and they are selected separately.

``grid.output_format`` -- which outputs
=======================================

.. list-table::
   :widths: 26 60
   :header-rows: 1

   * - Value
     - Effect
   * - ``report`` (default)
     - Plain-text report to ``grid.report_file`` (default
       ``grid_report.txt``)
   * - ``fields``
     - The per-cell field output, in whichever backend ``output.format``
       selects
   * - ``both``
     - Both of the above

``ascii`` and ``plt`` are accepted as aliases for ``report`` and
``fields``. That is what this switch was called before the field output
had two formats of its own, so every input file written earlier keeps
working. Any other value is a fatal error.

``output.format`` -- which backend
==================================

.. list-table::
   :widths: 26 60
   :header-rows: 1

   * - Value
     - Effect
   * - ``plt`` (default)
     - AMReX native plotfile to ``grid.plot_file`` (default ``plt_grid``)
   * - ``ascii``
     - One gathered plain-text file to ``output.ascii_file`` (default
       ``fields.txt``)
   * - ``both``
     - Both

Both backends are handed the **same** object -- one MultiFab and one list
of names, assembled once by ``CollectOutputFields``. Neither gathers its
own idea of what the fields are. Two backends that each did would drift:
one gains a component, the other does not, and the plain-text file the
regtests read stops being the file a user looks at. A regtest compares
the two outputs cell by cell across every component, and on the reference
case they are bit-identical.

Fields
======

Both backends carry these cell-centered fields, in this order:

.. list-table::
   :widths: 20 60
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
   * - ``u``, ``v``, ``w``
     - Velocity after the projection [m/s]
   * - ``sigma_x``, ``sigma_y``, ``sigma_z``
     - Poisson operator coefficients, vertical metric included
   * - ``u0``, ``v0``, ``w0``
     - The initial field, before the projection [m/s]
   * - ``alpha_h``, ``alpha_v``
     - The cell-local variational weights
   * - ``lambda``
     - The potential, averaged from its eight surrounding nodes to the
       cell centre [m\ :sup:`2`/s]
   * - ``divergence``
     - Discrete ``div(u)`` after the projection [1/s], zero in solid cells

``z_cc`` and ``dz`` are written because AMReX's ``Geometry`` is uniform
in z, so the plotfile's own vertical coordinate is only nominal. A
visualization tool needs these fields to place cells at their true
heights.

``lambda`` is nodal in the solver and is averaged to cell centres here so
that one output format can carry it alongside everything else. The solve
itself never sees this averaged copy.

Fields are appended to the end of this list as the solver grows, so
anything reading the output should look components up **by name** rather
than assuming a count.

The ascii backend
=================

One file, gathered to rank 0, one row per cell::

    # FastWindTerrain ascii field output
    # one row per cell; i fastest, k slowest
    # n_cell 24 24 40
    # n_rows 23040
    # ncomp 17
    # x, y are cell centers [m]; the cell-center height is the z_cc column
    # columns: i j k x y z_cc dz terrain_z mask u v w ...
    0 0 0 20.8333333 20.8333333 2 4 0.00358579 0 6.27304357 ...

Never per-rank or per-box files: reassembling those would push the
format into every consumer, which is where one format quietly becomes
several. Values are written at 17 significant digits, so a row
round-trips a double exactly and can be compared against the binary
plotfile without the comparison degrading into a test of the formatting.

**This is a regtest aid.** It is serial, it is roughly 450 bytes per
cell, and it has no place in a production case configuration -- it
exists so a checker can read the fields with nothing but the Python
standard library. A run past two million cells says so on stdout.

Diagnostics
===========

After the projection the run reports what it achieved, in the report and
on stdout:

.. list-table::
   :widths: 34 52
   :header-rows: 1

   * - Report key
     - Meaning
   * - ``diag_n_fluid_cells``
     - Cells the diagnostics cover; solid cells are excluded throughout
   * - ``diag_div_max`` / ``diag_div_min``
     - Extrema of the discrete ``div(u)`` over fluid cells [1/s]
   * - ``diag_div_l2``
     - Volume-weighted L2 of ``div(u)`` [1/s]
   * - ``diag_flux_in`` / ``diag_flux_out``
     - Volumetric flux through the open boundaries [m\ :sup:`3`/s]
   * - ``diag_flux_net``
     - ``out - in``
   * - ``diag_flux_imbalance``
     - ``|net| / max(in, out)``
   * - ``diag_flux_xlo`` ... ``diag_flux_top``
     - Net flux per face, positive outward
   * - ``diag_flux_tolerance``
     - ``diagnostics.flux_tolerance`` (default ``1e-3``)
   * - ``diag_flux_within_tolerance``
     - 1 or 0

The divergence **field** in the output and the ``diag_div_*`` numbers in
the report are the same array: the scalars are reductions of the field
the file carries, not a second computation of it. The same applies to the
flux -- :doc:`inflow` reports the boundary flux of the initial field by
calling this routine, so the before and after numbers cannot come from
two different definitions.

The L2 is volume weighted, so on a stretched grid the thin near-surface
cells do not dominate a norm they occupy little of.

Exceeding ``diagnostics.flux_tolerance`` prints a warning and **nothing
else**. The imbalance is a measurement: a mass-consistent adjustment with
Dirichlet ``lambda`` on the laterals does not enforce a closed budget,
and quietly forcing one would hide the very thing the number is there to
show. In practice the projection closes it anyway -- the reference
regtest case reaches a relative imbalance around 10\ :sup:`-14`, because
the net boundary flux is the volume integral of a divergence the solve
has driven to zero.

Report
======

The report is a plain-text ``key value`` listing: grid parameters, the
full ``z_face`` and ``z_cc`` arrays, then one section per module --
terrain, inflow, boundary conditions, numerics, anisotropy, O'Brien, the
Poisson solve, and the diagnostics above. Every value is written at full
round-trip precision, so it can be compared against analytic formulas
without loss.

It is what the regtest checkers read, and the simplest way to see what a
run actually resolved::

    grep terrain_ grid_report.txt
    grep diag_ grid_report.txt
