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

The immersed boundary is imposed on the **field**, never on the operator
coefficients:

* the divergence is cleared at nodes buried in terrain when the RHS is
  built (reported as ``poisson_rhs_nodes_zeroed``)
* the velocity is re-zeroed in solid cells after the correction
* ``sigma`` stays elliptic everywhere, terrain included

This follows ``massconsistent_amr``, whose face coefficients carry no
terrain masking at all. The reason is practical and was learned the hard
way: setting ``sigma = 0`` in solid cells makes no-flux exact in the
operator, but leaves any node buried in terrain with a **zero diagonal**,
which the multigrid smoother divides by. The result is NaN — and a NaN
that hides itself, because ``max(x, NaN)`` returns ``x``, so every
diagnostic reports a clean-looking zero: zero residual, zero lambda, zero
divergence.

Lambda is therefore solved inside the terrain as well, where it is
meaningless but harmless.

Boundary conditions
===================

``poisson.lambda_bc`` selects them, defaulting to ``flowthrough``: the
classical mass-consistent convention of ``lambda = 0`` on every
flow-through boundary, with Neumann where nothing flows through (ground
and domain top). It is **fixed, not derived from the wind**.

That is deliberate, and it supersedes the mapping the face
classification suggests (see :doc:`boundary_conditions`). Deriving the
lambda conditions from the wind -- Neumann on whichever faces the flow
enters -- interacts badly with the *nodal* operator. ``mlndlap_divu``
deliberately does not see the tangential velocity at a face it treats as
inflow, and with an oblique wind those faces carry a large tangential
component, so zeroing it manufactures an enormous artificial divergence.
Measured on the same case:

.. list-table::
   :widths: 30 20 20
   :header-rows: 1

   * -
     - directional
     - flowthrough
   * - initial ``max|div u|`` (controlled norm)
     - 6.92
     - **0.113**
   * - corrected ``|U|max`` from a 10 m/s inflow
     - 34.8 m/s
     - **18.9 m/s**

``massconsistent_amr`` never meets this, because its operator is
cell-centered; it also fixes its lambda conditions rather than deriving
them (``x`` Dirichlet, ``y`` and ``z`` Neumann).

The face classification still governs the **velocity** boundary
conditions, which is where the wind direction genuinely belongs. Setting
``poisson.lambda_bc = directional`` restores the derived mapping, for
comparison.

The projection
==============

``MLMG`` inverts a linear operator, so the derivative scheme (see
:doc:`numerics`) does not enter the solve at all. Two inputs choose the
operators:

.. list-table::
   :widths: 28 16 46
   :header-rows: 1

   * - Input
     - Default
     - Meaning
   * - ``poisson.rhs_operator``
     - ``fe``
     - ``fe`` uses AMReX's own nodal divergence; ``scheme`` uses the
       configured derivative scheme averaged to nodes
   * - ``poisson.gradient_operator``
     - ``amrex``
     - ``amrex`` uses AMReX's own nodal gradient; ``scheme`` averages
       lambda to cell centres and differentiates it with the configured
       scheme, as ``massconsistent_amr`` does
   * - ``poisson.n_projections``
     - ``4``
     - How many times to repeat the projection

The defaults pair AMReX's own divergence and gradient, because those are
the operators the solve is assembled from. The ``scheme`` variants are
offered because they are the familiar formulation, but they are looser:
on the Phase 6 hill case the divergence falls to 0.043 with ``amrex``
against 0.076 with ``scheme``.

Because the scheme never reaches the projection, ``weno3js``,
``upwind2`` and ``central2`` give an **identical** corrected field. Only
the reported diagnostic divergence differs, and a regtest pins exactly
that.

Repeating the projection
------------------------

AMReX's nodal projection is *approximate*: its divergence and gradient
are not an exact factorisation of the operator, so one pass removes only
part of the divergence. The remainder shrinks monotonically with
``poisson.n_projections``. On the Phase 6 hill case:

.. list-table::
   :widths: 20 20
   :header-rows: 1

   * - Passes
     - ``max|div u|``
   * - 0
     - 0.1133
   * - 1
     - 0.0935
   * - 2
     - 0.0728
   * - 4
     - 0.0431
   * - 8
     - 0.0234

Multigrid and cell aspect ratio
===============================

Multigrid convergence degrades as cells get more anisotropic, and the
cure is more smoothing sweeps -- roughly twice the aspect ratio, so 8
sweeps at 4:1 and 16 at 8:1. The sweeps are chosen from the grid unless
``poisson.num_pre_smooth`` / ``poisson.num_post_smooth`` are set, and the
ratio actually used is reported.

The surface layer is the worst case, being the thinnest: a 25 m
horizontal spacing over a 2 m first cell is 12.5:1.

Diagnostics
===========

Two divergence numbers are reported, and they measure different things:

``poisson_div_controlled_before`` / ``_after``
    The divergence in the norm the solve controls, using AMReX's own
    nodal operator. **This is the one that says whether the projection
    worked.**

``poisson_div_before`` / ``_after``
    The same field measured with the configured derivative scheme. A
    different, wider operator that nothing drives to zero, so it is a
    physics-facing diagnostic rather than a target.

Velocity extrema are reported too -- per component, plus ``|U|max``,
either side of the projection. They earn their place: a projection can
reduce the divergence handsomely while wrecking the field, and no
divergence number shows it. An earlier version of this solver reduced
divergence fifteen-fold while turning a 10 m/s inflow into a 35 m/s
corrected wind; only the extrema exposed it.

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
