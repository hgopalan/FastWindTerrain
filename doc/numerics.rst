========
Numerics
========

Derivative schemes
==================

``numerics.gradient_scheme`` selects the scheme used for directional
derivatives:

.. list-table::
   :widths: 14 60
   :header-rows: 1

   * - Value
     - Scheme
   * - ``weno3js`` (default)
     - Third-order WENO with Jiang-Shu smoothness weights
   * - ``upwind2``
     - Second-order one-sided difference, leaning into the flow
   * - ``central2``
     - Second-order central difference, no upwind bias

Any other value is a fatal error.

All three take the same five-point stencil centred on the point, plus the
advecting velocity whose sign selects the upwind side. Passing a zero
advecting velocity falls back to the central difference, since there is
then no upwind side to lean toward.

What they are for
=================

WENO3-JS and second-order upwind are upwind-biased reconstructions built
for hyperbolic transport: the advection term of a fractional-step solver,
and scalar transport. Central differencing is not the right tool there.

**The projection is a separate question.** A discrete projection is exact
only when its divergence ``D`` and its gradient ``G`` satisfy
``G = -D^T``, and ``D G`` is the very operator the linear solve inverted.
Central ``D`` and central ``G`` are such a pair, which is why a central
projection drives ``div(u)`` down to solver tolerance.

Using the *same* scheme on both sides does not recover that. WENO is
nonlinear -- its weights depend on the solution -- so ``D_weno G_weno``
is not a fixed linear operator, and in particular is not the Laplacian
the solve inverted. The corrected field then carries ``div(u) = O(h^p)``
rather than machine zero. That is an approximate projection: a legitimate
published approach, but it changes what a divergence-free check can
assert.

So this setting governs directional derivatives. Which operators the
projection is built from is a decision for where the projection is
assembled.

Stretched directions
====================

The schemes are written on a uniformly spaced index coordinate, and the
spacing argument is the local metric ``d(coordinate)/d(index)``. For
``x`` and ``y`` that is simply ``dx`` or ``dy``. For the stretched ``z``
it is ``dz/dk``, so passing ``(z_cc[k+1] - z_cc[k-1]) / 2`` gives the
mapped-coordinate form.

Formal order is retained for a smoothly stretched grid, which the
geometric stretching used here is. It is *not* retained if the grid
spacing varies abruptly.

Stencil width
=============

Both upwind schemes reach two cells to the upwind side, so any field they
are applied to needs a **two-cell halo**. The velocity field carries two
ghost layers for exactly this reason, and the boundary conditions fill
both.

Verification
============

``numerics.selftest_file`` runs a grid-refinement study of every scheme
and writes the observed orders of accuracy -- on a uniform periodic grid
and on a stretched grid, in both the L-infinity and L2 norms, for both
upwind branches. It also measures how far each reconstruction strays
outside the local data range across a step.

Measured on the finest grid pair:

.. list-table::
   :widths: 14 16 16 20
   :header-rows: 1

   * - Scheme
     - Uniform
     - Stretched
     - Step overshoot
   * - ``central2``
     - 2.00
     - ~1.95
     - --
   * - ``upwind2``
     - 2.00
     - ~1.90
     - --
   * - ``weno3js``
     - 3.0 and above
     - ~3.0
     - 2.5e-13
   * - unlimited linear
     - --
     - --
     - 0.167

Two things are worth reading out of that table. WENO3-JS reaches third
order where the field is smooth, but its L-infinity order is set by
critical points, where the derivative and the smoothness indicators
vanish together and the Jiang-Shu weights lose an order -- which is why
both norms are reported. And the overshoot column is the other half of
the point: the same third-order accuracy from the unlimited linear
combination of the two candidate stencils comes with a 1/6 overshoot at
a discontinuity, which the nonlinear weights remove.

The study is a test aid, written only when the input is set.
