====================
ParmParse reference
====================

Every input, in one place. Each is also documented in context on the page
listed, which is where the reasoning lives; this page is for looking one
up.

``regtests/inputs_master`` is the same list as a runnable input file,
with the defaults and permitted values as comments. It is kept honest by
a regtest that greps every ParmParse call out of ``Source/``.

Any input can be overridden on the command line::

    ./build/fastwindterrain inputs poisson.alpha_v=0.3 fwt.debug=1

Unrecognised values for an enumerated input are fatal, never a silent
fallback.

``grid.`` -- mesh and output selection
======================================

See :doc:`grid` and :doc:`output`.

.. list-table::
   :widths: 30 14 44
   :header-rows: 1

   * - Input
     - Default
     - Meaning
   * - ``grid.n_cell``
     - --
     - Cell counts ``nx ny nz``
   * - ``grid.prob_lo``
     - --
     - Domain lower corner [m]
   * - ``grid.prob_hi``
     - --
     - Domain upper corner [m]
   * - ``grid.dz0``
     - --
     - Surface-adjacent cell thickness [m]
   * - ``grid.stretching_ratio``
     - ``1.0``
     - Geometric ratio ``r``; ``1.0`` is a uniform grid
   * - ``grid.max_grid_size``
     - ``32``
     - Box size in x and y. The grid is never split in z
   * - ``grid.output_format``
     - ``report``
     - ``report``, ``fields`` or ``both`` (``ascii``/``plt`` are aliases
       for the first two)
   * - ``grid.report_file``
     - ``grid_report.txt``
     - Plain-text report
   * - ``grid.plot_file``
     - ``plt_grid``
     - AMReX plotfile

``output.`` and ``diagnostics.``
================================

See :doc:`output`.

.. list-table::
   :widths: 30 14 44
   :header-rows: 1

   * - Input
     - Default
     - Meaning
   * - ``output.format``
     - ``plt``
     - Which backend writes the field output: ``plt``, ``ascii`` or
       ``both``
   * - ``output.ascii_file``
     - ``fields.txt``
     - One gathered plain-text file. A regtest aid
   * - ``diagnostics.flux_tolerance``
     - ``1e-3``
     - Relative boundary-flux imbalance above which the run warns. Never
       corrected, never fatal

``terrain.`` -- surface and immersed boundary
=============================================

See :doc:`terrain`.

.. list-table::
   :widths: 30 14 44
   :header-rows: 1

   * - Input
     - Default
     - Meaning
   * - ``terrain.file``
     - *(none)*
     - ``x,y,z`` point file; absent means flat ground
   * - ``terrain.flat_elevation``
     - ``0.0``
     - Ground elevation with no file [m]
   * - ``terrain.idw_n_neighbors``
     - ``6``
     - Nearest points used by the interpolation
   * - ``terrain.idw_exponent``
     - ``2.0``
     - IDW power
   * - ``terrain.extrapolation``
     - ``idw``
     - Height of a column outside the point cloud's extent: ``idw`` or
       ``nearest``

``inflow.`` -- the initial wind field
=====================================

See :doc:`inflow`.

.. list-table::
   :widths: 30 14 44
   :header-rows: 1

   * - Input
     - Default
     - Meaning
   * - ``inflow.mode``
     - ``powerlaw``
     - ``powerlaw``, ``loglaw`` or ``userfile``
   * - ``inflow.u_ref``, ``inflow.v_ref``
     - ``0.0``
     - Reference velocity components [m/s]. Both zero is fatal
   * - ``inflow.z_ref``
     - ``10.0``
     - Reference height, AGL [m]
   * - ``inflow.powerlaw_exponent``
     - ``0.14``
     - Power-law exponent
   * - ``inflow.z0``
     - ``0.1``
     - Roughness length [m]
   * - ``inflow.file``
     - *(none)*
     - Six-column ``x y z u v w`` file
   * - ``inflow.idw_n_neighbors``
     - ``6``
     - Nearest points for ``userfile``
   * - ``inflow.idw_exponent``
     - ``2.0``
     - IDW power for ``userfile``
   * - ``inflow.z_agl_min``
     - ``z0``
     - Floor on height above ground [m]

``bc.`` -- boundary conditions
==============================

See :doc:`boundary_conditions`.

.. list-table::
   :widths: 30 14 44
   :header-rows: 1

   * - Input
     - Default
     - Meaning
   * - ``bc.dump_file``
     - *(none)*
     - One row per boundary ghost cell. A single-rank test aid

``anisotropy.`` and ``obrien.``
===============================

See :doc:`anisotropy`.

.. list-table::
   :widths: 30 14 44
   :header-rows: 1

   * - Input
     - Default
     - Meaning
   * - ``anisotropy.enable``
     - ``0``
     - Cell-local variational weights
   * - ``anisotropy.source``
     - ``slope``
     - ``slope`` or ``none``
   * - ``anisotropy.alpha_h_mode``
     - ``base``
     - Whether the slope factor reaches ``alpha_h``
   * - ``anisotropy.slope_scale``
     - ``0.5``
     - Slope at which the factor falls to ``1/e``
   * - ``anisotropy.decay_height``
     - ``500.0``
     - Decay height for the suppression [m]
   * - ``anisotropy.min_factor``
     - ``0.05``
     - Lower clamp, as a factor on the base
   * - ``anisotropy.max_factor``
     - ``2.0``
     - Upper clamp
   * - ``obrien.enable``
     - ``0``
     - Vertical-velocity adjustment

``surface.`` -- the first fluid cell above terrain
==================================================

See :doc:`terrain`.

Velocity is zeroed *inside* the terrain, but without this nothing
constrains the fluid cell just above it: the flow there has a component
running into the surface, and its speed comes from the inflow profile
evaluated a metre or two above ground, inside the roughness sublayer.

This is an **immersed** boundary, not a body-fitted wall, so the distance
in the log law is perpendicular to the sloped surface --
``(z_cc - h) * n_z`` -- and the speed it acts on is the surface-parallel
one, not the horizontal one.

.. list-table::
   :widths: 30 18 40
   :header-rows: 1

   * - Input
     - Default
     - Meaning
   * - ``surface.type``
     - ``wall_function``
     - ``wall_function``, ``slip``, ``noslip`` or ``none``
   * - ``surface.apply``
     - ``initial``
     - ``initial``, or ``both`` to re-impose after every projection pass
   * - ``surface.z0``
     - ``inflow.z0``
     - Roughness length for the wall function
   * - ``surface.kappa``
     - ``0.41``
     - von Karman constant

``poisson.`` -- the solve
=========================

See :doc:`poisson`.

.. list-table::
   :widths: 30 14 44
   :header-rows: 1

   * - Input
     - Default
     - Meaning
   * - ``poisson.alpha_h``
     - ``1.0``
     - Horizontal transmissivity (base value)
   * - ``poisson.alpha_v``
     - ``1.0``
     - Vertical transmissivity (base value)
   * - ``poisson.lambda_bc``
     - ``flowthrough``
     - ``flowthrough`` or ``directional``
   * - ``poisson.rhs_operator``
     - ``fe``
     - ``fe`` (AMReX's divergence) or ``scheme``
   * - ``poisson.gradient_operator``
     - ``amrex``
     - ``amrex`` (AMReX's gradient) or ``scheme``
   * - ``poisson.n_projections``
     - ``4``
     - Projection passes
   * - ``poisson.max_iter``
     - ``200``
     - MLMG iteration cap
   * - ``poisson.reltol``
     - ``1e-11``
     - MLMG relative tolerance
   * - ``poisson.abstol``
     - ``0.0``
     - MLMG absolute tolerance; ``0`` leaves it unused
   * - ``poisson.num_pre_smooth``
     - *(from the grid)*
     - Smoothing sweeps; chosen from the cell aspect ratio
   * - ``poisson.num_post_smooth``
     - *(from the grid)*
     - As above
   * - ``poisson.verbose``
     - ``0``
     - MLMG verbosity
   * - ``poisson.manufactured``
     - ``0``
     - Solve a known analytic problem instead
   * - ``poisson.force_all_dirichlet``
     - ``0``
     - Dirichlet on all six faces. Only meaningful with
       ``manufactured = 1``, whose solution is posed that way
   * - ``poisson.rhs_dump_file``
     - *(none)*
     - Nodal RHS. A single-rank test aid

``verify.`` -- verification aids
================================

See :doc:`convergence`.

.. list-table::
   :widths: 30 14 44
   :header-rows: 1

   * - Input
     - Default
     - Meaning
   * - ``verify.gradient_dump_file``
     - *(none)*
     - Writes ``d(u)/dz`` and ``d(v)/dz`` per level, as the solver
       computes them, so their order can be measured end to end
   * - ``verify.gradient_advect``
     - ``1.0``
     - Which upwind branch that dump measures. Nonzero. Read only when
       the dump is on

``numerics.`` and ``fwt.``
==========================

See :doc:`numerics` and :doc:`debugging`.

.. list-table::
   :widths: 30 14 44
   :header-rows: 1

   * - Input
     - Default
     - Meaning
   * - ``numerics.gradient_scheme``
     - ``weno3js``
     - ``weno3js``, ``upwind2`` or ``central2``
   * - ``numerics.selftest_file``
     - *(none)*
     - Scheme convergence study. A test aid
   * - ``fwt.debug``
     - ``0``
     - Verbose run diagnostics
