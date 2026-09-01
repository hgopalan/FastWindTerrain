==============
Poisson solver
==============

The variational adjustment solves an anisotropic Poisson problem for the
potential ``lambda`` on nodes:

.. code-block:: none

    div( sigma grad(lambda) ) = div(u0)
    sigma = (alpha_h^2, alpha_h^2, alpha_v^2)

with the velocity correction

.. code-block:: none

    u = u0 - alpha_h^2 dlambda/dx        (and likewise v)
    w = w0 - alpha_v^2 dlambda/dz

So **alpha is a transmissivity, not a penalty**: a smaller ``alpha_v``
means less vertical adjustment. That is the sense the slope-suppression
term needs, and it is why this convention was chosen over the classical
Sasaki form, where the same symbol is a cost weight and the relationship
inverts. The operator coefficient and the correction multiplier are the
same object, and a regtest checks they stay that way.

Inputs
======

.. list-table::
   :widths: 24 12 45
   :header-rows: 1

   * - Input
     - Default
     - Meaning
   * - ``poisson.alpha_h``
     - ``1.0``
     - Horizontal transmissivity
   * - ``poisson.alpha_v``
     - ``1.0``
     - Vertical transmissivity
   * - ``poisson.max_iter``
     - ``200``
     - MLMG iteration cap
   * - ``poisson.reltol``
     - ``1e-11``
     - MLMG relative tolerance
   * - ``poisson.verbose``
     - ``0``
     - MLMG verbosity

The stretched grid
==================

AMReX's ``Geometry`` is uniform in z, and the nodal operator
(``mlndlap_adotx_ha``) builds its stencil from that uniform spacing with
a per-cell, per-direction sigma. What AMReX calls *mapped* mode is **not**
a coordinate mapping and carries no Jacobian -- it is exactly a
spatially varying diagonal coefficient. That is the hook the true cell
heights ride in on.

Writing ``zeta`` for the computational vertical coordinate, whose spacing
is the nominal ``Geometry`` ``dz``, and

.. code-block:: none

    J(k) = dz(k) / dz_nominal

for the dimensionless metric, the problem becomes, in computational
space:

.. code-block:: none

    sigma_x = alpha_h^2 * J      sigma_y = alpha_h^2 * J
    sigma_z = alpha_v^2 / J
    rhs     = J * (du/dx + dv/dy) + dw/dzeta

The ``J`` weighting is the cell volume in computational space, which is
what keeps the assembled matrix symmetric -- MLMG still sees an SPD
operator, and no AMReX modification is needed.

This is derived rather than borrowed. ``massconsistent_amr`` runs a
uniform ``dz``; its terrain-following option puts metric terms in the
divergence while leaving the operator coefficients untouched, which
would leave the two disagreeing about the coordinate system.

Because it is derived, it is verified rather than assumed. See
**Verification** below.

The immersed boundary
=====================

No-flux inside the terrain is imposed as ``sigma = 0`` in solid cells:
zero conductivity carries no flux, which is the Neumann condition. It
falls straight out of sigma being cell-centered, so no separate
treatment is needed.

A node surrounded entirely by solid cells then has an empty row. Those
nodes are pinned through the overset mask rather than left singular, and
the count is reported as ``poisson_n_pinned_nodes``.

Boundary conditions
===================

The ``lambda`` conditions come from the face classification (see
:doc:`boundary_conditions`): Neumann where velocity is prescribed or
there is no flow, Dirichlet where it is free. At least one face must be
Dirichlet or the operator is singular, which the boundary-condition code
asserts before the solve is ever built.

The projection
==============

The solve inverts a **linear** operator -- that is what MLMG does. The
selected derivative scheme (see :doc:`numerics`) enters at the two ends
instead: the divergence that forms the RHS, and the gradient in the
velocity correction.

With WENO at those ends the result is an **approximate projection**: the
corrected field carries ``div(u) = O(h^p)`` rather than machine zero,
because the nonlinear ``D`` and ``G`` do not compose into the linear
Laplacian that was inverted. This is a legitimate and published
approach, but it means a divergence-free check has to assert a
*convergence rate* under refinement rather than a fixed threshold.

Verification
============

``poisson.manufactured = 1`` replaces the RHS with an analytic source
for a known solution,

.. code-block:: none

    lambda = sin(pi x/Lx) sin(pi y/Ly) sin(pi z/Lz)

which vanishes on every face, so homogeneous Dirichlet is exact and the
boundary treatment cannot mask an error in the interior discretization.
The run reports the L2 and L-infinity error against that solution.

The regtest sweeps three resolutions at each of several fixed total
stretch ratios and requires second-order convergence throughout.
Measured:

.. list-table::
   :widths: 20 20
   :header-rows: 1

   * - Total stretch
     - L2 order
   * - 1x (uniform)
     - 1.97
   * - 4x
     - 1.97
   * - 20x
     - 1.99

Holding the *total* stretch fixed while refining is what makes this a
convergence study: keeping the stretching ratio fixed instead would
distort the grid further at every refinement and measure nothing. A
wrong metric factor in sigma or in the RHS weighting costs an order
here, and costs more of one the more the grid is stretched -- which is
why the sweep runs at several ratios rather than one.

Test aids
=========

``poisson.rhs_dump_file`` writes one row per node (``i j k rhs``) so a
checker can compare the assembled RHS against an independently computed
divergence. Like the boundary dump it is a single-rank test aid, not a
production output path.

The plotfile also carries ``sigma_x``, ``sigma_y`` and ``sigma_z``, so
the coefficients including the metric can be inspected directly.
