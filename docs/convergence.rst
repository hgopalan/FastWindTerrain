=================
Convergence study
=================

``convergence/run_convergence.py`` sweeps **inflow profile × derivative
scheme × vertical resolution**, generates an input file for each
combination, runs the solver, and reports the observed order of
accuracy::

    python3 convergence/run_convergence.py build/fastwindterrain

That is 24 runs -- two profiles, three schemes, four resolutions -- and
writes ``convergence_results.csv`` plus a summary table to
``build/convergence``.

Why the vertical derivative
===========================

Over flat ground a powerlaw or loglaw profile has ``div(u) = 0``
**identically**. ``u`` and ``v`` depend only on ``z`` and ``w`` is zero,
so ``du/dx``, ``dv/dy`` and ``dw/dz`` each vanish for every scheme on
every grid. A divergence-based study on flat ground measures nothing.

The only nontrivial derivative in that problem is ``dU/dz`` -- which is
also where the three schemes actually differ, and where a wrong vertical
metric on a stretched grid would show up. So the study measures the
solver's own ``d(u)/dz`` and ``d(v)/dz``, written by
``verify.gradient_dump_file``, against the analytic derivative of the
configured profile law.

The analytic law is implemented in the driver, independently of the C++,
so the solver is not grading its own work.

What this covers that the self-test does not
============================================

``numerics.selftest_file`` (see :doc:`numerics`) measures the same three
schemes and reports the same orders -- but on a ``std::vector`` holding
``sin(2 pi x)``. It never builds a Grid, a MultiFab or a ghost cell. It
is a unit test of the stencil.

Everything between that stencil and the number the solver uses is
untested there:

* the column metric ``dz/dk``
* the box decomposition -- the study runs ``max_grid_size`` below the
  domain width on purpose, so several boxes are in play
* the index clamping at the top and bottom of the domain
* the profile as ``Inflow`` actually evaluated it

Any of those can cost an order without the self-test noticing.

The fixed measurement band
==========================

The error is measured over a **fixed physical band of height**
(``--z-window``, default 20 m to 900 m), not over whatever levels happen
to exist.

This is the part that makes the study work at all. Both profile laws have
``dU/dz ~ z^(alpha-1)`` or ``~ 1/z``, so the derivative is near-singular
as ``z -> 0``. On a stretched grid the first cell thins as ``nz`` grows,
so an "all interior levels" norm would march its own lower limit toward
that singularity. Measured that way the error **increases** under
refinement, and the first version of this driver duly reported an order
of ``-0.4``. Holding the band fixed measures the same physical region at
every resolution, which is what a grid-convergence study means.

Three exclusions, all deliberate:

* levels outside the band
* the two levels at each end of the column, where the stencil index is
  clamped to the domain and the derivative is one-sided by design
* any level at or below ``inflow.z_agl_min``, plus a stencil radius above
  it, where the profile is floored and its derivative has a kink

Refinement holds the **total** stretch fixed: the ratio is recomputed at
each ``nz`` so the last cell is always the same multiple of the first.
Holding the ratio itself fixed would change the underlying mapping as the
grid refines, and the study would stop being a convergence study.

Results
=======

Stretched grid, total stretch 10, upwind branch ``+1``, L2 over the band.
This is what the sweep produces:

.. list-table::
   :widths: 16 14 12 12 12 12
   :header-rows: 1

   * - Scheme
     - Profile
     - nz=64
     - nz=128
     - nz=256
     - nz=512
   * - ``central2``
     - powerlaw
     - 4.88e-05
     - 1.58e-05
     - 3.96e-06
     - 9.27e-07
   * -
     - *order*
     -
     - 1.62
     - 2.00
     - **2.10**
   * - ``upwind2``
     - powerlaw
     - 1.53e-04
     - 4.09e-05
     - 9.19e-06
     - 2.05e-06
   * -
     - *order*
     -
     - 1.90
     - 2.15
     - **2.17**
   * - ``weno3js``
     - powerlaw
     - 1.59e-05
     - 2.63e-06
     - 2.89e-07
     - 3.77e-08
   * -
     - *order*
     -
     - 2.60
     - 3.19
     - **2.94**
   * - ``central2``
     - loglaw
     - 7.36e-05
     - 2.42e-05
     - 6.04e-06
     - 1.41e-06
   * -
     - *order*
     -
     - 1.61
     - 2.00
     - **2.10**
   * - ``upwind2``
     - loglaw
     - 2.32e-04
     - 6.23e-05
     - 1.39e-05
     - 3.08e-06
   * -
     - *order*
     -
     - 1.90
     - 2.16
     - **2.18**
   * - ``weno3js``
     - loglaw
     - 2.99e-05
     - 5.15e-06
     - 5.71e-07
     - 5.60e-08
   * -
     - *order*
     -
     - 2.54
     - 3.17
     - **3.35**

Both second-order schemes reach 2, and WENO3-JS reaches 3 -- and unlike
the ``sin(2 pi x)`` self-test, its L-infinity order matches its L2 order
here, because a monotone wind profile has no critical points for the
Jiang-Shu weights to degrade at. At ``nz = 256`` WENO is roughly 25 times
more accurate than either second-order scheme on the same grid.

The coarse end is pre-asymptotic: the band contains a discrete set of
levels, and which levels those are changes with ``nz``, so the first
order estimate jitters. The sweep starts at ``nz = 64`` for that reason,
and the assertion is made on the finest pair.

Options
=======

.. list-table::
   :widths: 30 54
   :header-rows: 1

   * - Flag
     - Effect
   * - ``--grids 64,128,256``
     - Vertical resolutions to sweep
   * - ``--profiles``, ``--schemes``
     - Restrict either axis
   * - ``--uniform``
     - Uniform vertical grid instead of stretched
   * - ``--both-grids``
     - Run the uniform and the stretched grid
   * - ``--advect plus|minus|both``
     - Which upwind branch to measure. A sign error hides in the branch
       that is never measured
   * - ``--z-window lo,hi``
     - The fixed measurement band [m]
   * - ``--check``
     - Assert the observed orders and exit nonzero on failure

Horizontal uniformity is asserted on every run regardless of ``--check``:
over flat ground the gradient must not vary in x or y, and since the runs
are decomposed into several boxes, a spread above round-off would mean a
decomposition or ghost bug. It measures exactly zero.

In CI
=====

The regtest group ``profile_convergence`` runs a reduced version of the
same sweep -- three resolutions rather than four -- and asserts the
orders, so the schemes are protected on every pull request. The full
four-resolution sweep, and the ``nz = 512`` runs in particular, are left
to be run by hand; see :doc:`regtests`.
