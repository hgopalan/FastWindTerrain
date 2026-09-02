========
Regtests
========

``regtests/`` holds one directory per test group, each with its own
``inputs*`` files and a standalone ``check.py``. A directory counts as a
test group as soon as it holds a ``check.py``; the names follow no
required pattern. There is no separate example tier -- these are the
whole test suite.

Running them
============

::

    python3 run_regtests.py build/fastwindterrain

or one group::

    python3 run_regtests.py build/fastwindterrain phase1_grid

The same tests are registered with CTest::

    ctest --test-dir build --output-on-failure

The full four-resolution convergence sweep is not part of the suite --
the ``nz = 512`` runs are slower than a regtest should be. The
``profile_convergence`` group runs a reduced version of it; the full one
is::

    python3 convergence/run_convergence.py build/fastwindterrain

Cases run in a scratch work directory (``build/regtests/<group>`` by
default, overridable with ``--workdir``), so running the tests leaves
nothing behind in the source tree.

How the checks are written
==========================

Two rules keep these tests meaningful rather than self-confirming:

**Recompute, do not read back.** Expected values are derived in Python
from the same inputs the solver was given -- the analytic profile laws,
an independent IDW implementation, an independent flux integration --
rather than read out of the solver's own output. A checker that only
compares the solver against itself would pass through most real bugs.

**Test the invariant, not a sample.** Where an invariant is cheap to
check exhaustively, it is: the mask is verified against
``z_cc <= z_terrain`` in every cell of the domain, not at sampled
points.

From Phase 6 the plotfile carries the velocity **after** the projection
in ``u``/``v``/``w``, and the initial field in ``u0``/``v0``/``w0``.
Checkers that test the profile or the assembled RHS read the ``u0``
fields.

``regtests/plotfile.py`` is a small standard-library reader for AMReX
single-level plotfiles, shared by the checkers. It exists so the tests
can inspect field values without depending on ``yt``. Each FAB is
self-describing, so only the ``FabOnDisk`` offsets have to be parsed out
of ``Cell_H``. It also runs standalone as a summary dumper::

    python3 regtests/plotfile.py build/regtests/phase2_terrain_ib/plt_hill

Test groups
===========

``phase5_poisson_assembly``
---------------------------

The anisotropic Poisson operator (see :doc:`poisson`).

* ``inputs_mms`` -- a manufactured solution with a known analytic
  lambda, solved at three resolutions for each of several fixed total
  stretch ratios, requiring second-order convergence throughout. The
  true cell heights reach the operator only through sigma, so a wrong
  metric factor costs an order here -- and costs more of one the more
  the grid is stretched, which is why several ratios are swept
* ``inputs_rhs`` -- a hill makes ``u0`` genuinely non-solenoidal, since
  the profile is anchored to height above ground. Checks the assembled
  nodal RHS against an independent Python divergence of the plotfile's
  own velocity, and sigma against its definition in every sampled cell
  including the ``sigma = 0`` branch inside terrain

``phase6_solve_correction``
---------------------------

The linear solve and the velocity correction (see :doc:`poisson`).

* ``inputs_flat`` -- a uniform profile over flat ground is already
  solenoidal, so lambda must come out ~0 and the velocity must be
  untouched. This is the case that catches a projection which "corrects"
  a field that was already fine
* ``inputs_bump`` -- over a 100 m hill the projection has real work to
  do. Checks that the divergence in the norm the solve **controls** goes
  down, that it keeps going down as passes are added, and that the
  corrected wind stays physical
* the same case across all three derivative schemes, which must leave
  the corrected field identical -- the scheme does not reach the
  projection

The velocity-extrema assertion is not decoration. An earlier version of
this solver reduced divergence fifteen-fold while turning a 10 m/s
inflow into a 35 m/s corrected wind, and no divergence number showed it.

``phase7_anisotropy_obrien``
----------------------------

Cell-local anisotropy and the O'Brien adjustment (see :doc:`anisotropy`).

* ``inputs_slope`` -- over a steep hill, ``alpha_v`` must follow a slope
  factor recomputed independently in the checker, be suppressed on the
  flanks and sit at base over flat ground; and after the adjustment
  ``w`` must be **exactly** zero at the domain top, held to round-off
  rather than a tolerance, since making it exact is the whole point
* with ``anisotropy.enable = 0`` both weights must hold their base
  values, so the feature cannot change results when switched off
* ``alpha_h_mode = slope`` must apply the same factor to both weights

``phase8_diagnostics_output``
-----------------------------

The post-solve diagnostics and the two output backends (see
:doc:`output`).

* ``inputs_both`` -- a Phase 6/7 case with ``output.format = both``, so
  the plotfile and the plain-text file are written from the same run.
  Every component of every cell is then compared between them; on this
  case they are bit-identical. This is the check that keeps the shared
  collect routine honest -- two backends that each gathered their own
  fields would pass everything else here and still drift apart
* well-formedness: one file, a parseable header, exactly ``nx*ny*nz``
  rows, the column count the header itself declares, every cell present
  exactly once, no NaN or infinity, and coordinate columns that really
  are the cell centres
* the reported diagnostics are **recomputed from the output rows**:
  ``max|div|``, the volume-weighted L2, and each of the five boundary
  face fluxes integrated independently in Python from the velocity and
  the true ``dz``. The divergence must be exactly zero in every solid
  cell
* ``diagnostics.flux_tolerance = 0`` must warn and change nothing -- the
  imbalance is a measurement, not something the code may quietly fix
* ``output.format`` really selects: ``plt`` writes no plain-text file,
  ``ascii`` writes no plotfile, ``grid.output_format = report``
  suppresses both, the legacy ``ascii``/``plt`` spellings still work,
  and an unknown value aborts

``phase9_bindings_parity``
--------------------------

The Python bindings, and C++/Python parity (see :doc:`python`).

* the module imports, reports its version and the AMReX it was built
  against, and does **not** initialize AMReX as a side effect of import
* every lifecycle misuse **raises** rather than aborting: a second
  ``initialize()``, a stray ``finalize()``, and ``run()`` inside an
  existing initialization. A guard that segfaults instead of raising
  loses a notebook session's work, which is the failure this API exists
  to prevent
* ``session()`` finalizes even when its block raises
* five cases -- a flat solve, a hill with anisotropy and O'Brien, the
  manufactured solution, and the ascii backend -- run through both the
  executable and the bindings, with every output file compared **byte
  for byte**, plus stdout. An AMReX plotfile carries no timestamp, so it
  compares like everything else
* command-line ``name=value`` overrides reach the solver through the
  shim and give an identical report

The group SKIPS when the bindings are not built, since a C++-only build
is supported; CI builds one job with them on.

Beyond this group, the whole suite runs a second time through the
bindings::

    python3 run_regtests.py build/fastwindterrain-py

``build/fastwindterrain-py`` is argv-compatible with the executable, so
no checker knows the difference. CI runs both.

``mpi_parity``
--------------

The one group that runs on more than one rank. It is the **only** thing
in the suite that does, so an MPI build is exercised here or nowhere.

* three cases borrowed from the other groups -- a flat solve, a hill
  with a real projection to do, and the anisotropy and O'Brien path --
  are run serially and again under ``mpirun -n 2``, and every entry of
  the two reports is compared. The decomposition is a distribution
  detail, not physics, so they must agree to round-off. They are not
  compared byte for byte: a global sum lands in a different order on a
  different decomposition, and that difference is real. The tolerance is
  ``1e-9`` relative, six orders of magnitude looser than what is
  observed and still far tighter than any genuine parallel bug
* the near-zero entries -- a solve residual, a flux net, the flux
  through a closed lid -- are *values that are round-off*, so their
  relative difference between decompositions is O(1) however right they
  both are. Those are compared absolutely, and separately held to the
  physical bound they actually have, so parity cannot be satisfied by
  two equally broken runs
* every run is under a **wall-clock timeout**, and expiry is reported as
  its own failure

That last one is the reason the group exists. A collective call reached
by only some ranks does not give a wrong answer, it gives no answer: the
ranks that skipped it walk on, the ranks that entered it wait forever.
``Poisson::AppendReport`` did exactly that -- it returned early on
non-IO ranks and then called ``amrex::MultiFab::min`` and ``max``, which
are collective. Every number the solver produced was correct and none of
them were ever written. Checking numbers cannot catch that, because the
failure is the absence of numbers.

The pattern to look for, if this group ever times out again, is a
rank-conditional early return followed by a collective -- particularly
the collectives that do not look like one: ``MultiFab::min``, ``max``,
``norm0``, ``norm2``, ``sum``, and anything built on
``ParallelDescriptor::Reduce*`` or ``ParallelAllReduce``. Take them
**before** the IO rank is singled out, then write the results.

The group SKIPS when the executable was not built with MPI (the
default), or when ``mpirun`` is not on PATH -- it asks by running the
binary and looking for AMReX's communicator-size line, since that is the
only thing that actually knows. See :doc:`building`.

``tests/`` -- the bindings' own suite
-------------------------------------

Everything the bindings do beyond that parity guarantee -- ``Grid`` from
a dict, fields as numpy, terrain from a point cloud, the stepwise solver,
anisotropy, in-memory output, dataset generation -- is tested with
**pytest**, in ``tests/``, not here::

    pytest tests

Those were ``regtests/phase10_*`` through ``regtests/phase15_*`` and moved
in Phase 16. The reason is what each suite is testing. A group in
``regtests/`` runs a binary on an inputs file and reads its output files
back, and ``run_regtests.py`` is the right driver for that. The bindings
tests are testing a Python API, where fixtures, parametrization and
assertion rewriting are worth having -- and where the old checkers had to
shell out to a subprocess per case and parse ``::KEY value`` lines back
out of stdout to say anything at all.

The assertions came across unchanged. ``phase9_bindings_parity`` stays
here, because byte-for-byte parity between the executable and the
bindings is a property of a build tree rather than of the Python package.
:doc:`python` has the details.

``profile_convergence``
-----------------------

The order of accuracy of the derivative schemes, measured end to end
through the solver (see :doc:`convergence`).

* a reduced sweep of the convergence driver -- two profiles, three
  schemes, three resolutions -- asserting the observed L2 order on the
  finest pair: 2 for ``central2`` and ``upwind2``, 3 for ``weno3js``
* WENO must also be more *accurate*, not merely higher order: at
  ``nz = 256`` it is ~25x better than either second-order scheme on the
  same grid. Order is a claim about the limit; this is the claim a user
  cares about
* over flat ground the vertical gradient must not vary horizontally. The
  runs are decomposed into several boxes, so a spread above round-off
  would be a decomposition or ghost bug -- exactly what
  ``numerics.selftest_file`` cannot see, since it never builds a box. It
  measures **exactly** zero
* the negative upwind branch converges too. A sign error hides in the
  branch that is never measured

``master_inputs``
-----------------

``regtests/inputs_master`` is a runnable input file naming **every**
input the solver reads, with its default and permitted values.

A reference like that is worth exactly what its accuracy is worth, and
the usual failure is silent: someone adds a ParmParse query, the
reference never mentions it, and a year later a user trusts a file that
is quietly wrong. So the checker does not read it and nod:

* it greps every ParmParse prefix and key out of ``Source/``,
  reconstructs the full input names, and asserts each appears in the
  master file -- commented out is fine, absent is not
* and the reverse, so a renamed or removed input cannot leave a stale
  entry behind
* it **runs** the file, so the reference is a working case, and fails if
  AMReX reports any input the run never read
* it runs it again under ``fwt.debug = 1`` and requires the report to be
  byte-identical: documenting a run must not change it

``terrain_extrapolation``
-------------------------

What a grid column outside the terrain point cloud gets (see
:doc:`terrain`). The point of the group is that the failure it guards
against is invisible: IDW answers every query, and the mask is only
``z_cc <= z_terrain``, so a wrong height out there produces a column
that is all fluid or all solid rather than anything that looks like an
error.

* ``inputs_covered`` -- the cloud spans the whole domain, so no column
  is outside it and the two modes must produce **bit-identical**
  terrain. That is the guarantee that makes ``idw`` safe to leave as the
  default: turning the option on cannot disturb a case that did not need
  it
* ``inputs_partial`` under ``idw`` -- the cloud covers
  ``[0, 400] x [0, 600]`` of a 1000 x 1000 m domain, leaving 1216 of the
  1600 columns outside it. Every column, covered or not, must still be
  the plain IDW, so the historical behaviour is unchanged -- and the run
  must now *say* the cloud is short
* ``inputs_partial`` under ``nearest`` -- the outside columns take the
  nearest input point's elevation exactly, the inside ones do not move
  at all, and the surface must actually shift, so a fallback that
  quietly returned the IDW value could not pass
* an unrecognized ``terrain.extrapolation`` aborts, naming the value

Both references -- the IDW and the nearest-point lookup -- are recomputed
in the checker from the point file, and the count of outside columns is
derived from the cloud's extent rather than read back from the report.

``gradient_schemes``
--------------------

The directional-derivative schemes (see :doc:`numerics`).

* ``inputs_selftest`` -- a grid-refinement study of every scheme on both
  a uniform and a stretched grid, in both norms and for both upwind
  branches, checking the observed order of accuracy. Order is the check
  that separates a working scheme from a plausible-looking one: a sign
  error or a mis-shifted stencil still produces smooth output, but will
  not converge at the right rate. Also checks that the WENO
  reconstruction stays bounded across a step while the unlimited linear
  combination does not
* ``inputs_scheme`` -- the scheme name round-trips into the report, the
  default is ``weno3js``, and an unrecognized name aborts

``phase1_grid``
---------------

Grid construction and the domain-height policy.

* ``inputs_nominal`` -- exact height match, no warning, report matches
  the analytic geometric-stretching formula
* ``inputs_uniform`` -- ``stretching_ratio = 1.0`` reproduces a plain
  uniform grid exactly
* ``inputs_overshoot`` -- non-fatal warning plus ``prob_hi[2]`` override
* ``inputs_undershoot`` -- fatal abort, nonzero exit, no report written
* ``inputs_plt`` -- ``output_format = both`` writes the ascii report and
  a well-formed plotfile
* ``inputs_badformat`` -- an unrecognized ``output_format`` aborts
* ``inputs_debug`` -- ``fwt.debug=1`` prints the full diagnostics and
  changes no result; the default stays silent

``phase2_terrain_ib``
---------------------

Terrain interpolation and the immersed-boundary mask.

* ``inputs_flat`` -- no terrain file: ``z_terrain == 0`` everywhere and
  every cell fluid
* ``inputs_hill`` -- Gaussian hill sampled on a 20 m lattice against a
  25 m grid, so cell centers never coincide with a terrain point.
  Checks ``z_terrain`` against an independent Python IDW, that the mask
  is exactly ``z_cc <= z_terrain`` in every cell, that solid cells are
  contiguous from the ground up, that the mask boundary lands in the
  right cell in every column, and that the interpolated surface tracks
  the analytic Gaussian it was sampled from
* ``inputs_scattered`` -- the same hill sampled off-lattice, so the
  k-nearest search faces irregular spacing
* a missing terrain file must abort rather than fall back to flat ground

``phase3_inflow_profile``
-------------------------

Wind profiles and their terrain awareness.

* ``inputs_powerlaw``, ``inputs_loglaw`` -- ``u0`` matches the analytic
  law at every sampled height and points along ``(u_ref, v_ref)``;
  solid cells hold zero; nothing is non-finite
* ``inputs_userfile`` -- ``u0`` matches an independent Python 3D IDW of
  the same six-column file, the sixth column is genuinely used, and an
  exact hit on a table point returns that point's values
* ``inputs_powerlaw_bump`` -- AGL anchoring over a hill. The profile is
  compared against the flat run interpolated to the same height above
  ground, with the tolerance calibrated from the interpolation error
  itself rather than guessed, plus a check that the profile moved at all
  relative to the same absolute height
* ``inputs_boundary_terrain`` -- terrain intersecting the lateral
  boundaries. The reported boundary flux must match an independent
  integration, the resulting imbalance must be surfaced rather than
  hidden, and the interior profile must be unchanged
* calm wind and an unknown mode must both abort

``phase4_bc_direction``
-----------------------

Directional boundary conditions. Every boundary cell is checked, not a
sample: the solver writes one row per boundary cell to ``bc.dump_file``,
and the expected ghost value is recomputed from the profile law.

* ``inputs_sw`` / ``inputs_ne`` -- opposite winds. Each face must be
  classified correctly, and every lateral face must flip between the two
* ``inputs_edge`` -- an axis-aligned wind, so one inflow face, one
  outflow face, and two tangential faces treated as open. Also the
  lower bound of the "one or two prescribed faces" rule
* ``inputs_terrain`` -- an inflow face partly buried in terrain: the
  buried ghost cells are shut off and the rest still carry the profile
* ``inputs_userfile`` -- ``userfile`` mode has no reference wind vector,
  so this pins that the classification comes from the field's own face
  fluxes
