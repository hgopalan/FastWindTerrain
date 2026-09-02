======
Inflow
======

The initial velocity field ``u0`` is built by mapping a 1D wind profile
onto every grid column in **height above ground level (AGL)**.

Inputs
======

.. list-table::
   :widths: 30 14 40
   :header-rows: 1

   * - Input
     - Default
     - Meaning
   * - ``inflow.mode``
     - ``powerlaw``
     - ``powerlaw`` | ``loglaw`` | ``userfile``
   * - ``inflow.u_ref``
     - ``0.0``
     - Reference velocity x-component [m/s]
   * - ``inflow.v_ref``
     - ``0.0``
     - Reference velocity y-component [m/s]
   * - ``inflow.z_ref``
     - ``10.0``
     - Reference height, AGL [m]
   * - ``inflow.powerlaw_exponent``
     - ``0.14``
     - Power-law ``alpha``
   * - ``inflow.z0``
     - ``0.1``
     - Roughness length [m]
   * - ``inflow.file``
     - *(none)*
     - Velocity file for ``userfile``
   * - ``inflow.idw_n_neighbors``
     - ``6``
     - Nearest points for ``userfile``
   * - ``inflow.idw_exponent``
     - ``2.0``
     - IDW power for ``userfile``
   * - ``inflow.z_agl_min``
     - ``z0``
     - Floor applied to ``z_agl`` [m]

A calm reference wind (``u_ref`` and ``v_ref`` both zero) is a fatal
error: there is no wind, and no boundary face can be classified as
inflow.

Profile laws
============

Reference speed and direction come from ``(u_ref, v_ref)``:
``speed_ref = hypot(u_ref, v_ref)``, with the horizontal direction taken
from the same vector.

**Power law**

.. code-block:: none

    speed(z_agl) = speed_ref * (z_agl / z_ref)^alpha

**Log law**

.. code-block:: none

    speed(z_agl) = speed_ref * ln((z_agl + z0)/z0) / ln((z_ref + z0)/z0)

The ``z + z0`` numerator is ``massconsistent_amr``'s form: the profile
reaches exactly zero at the ground instead of diverging.

**User file**

Six columns, ``x y z u v w``, read with the same tolerant parser as
terrain files -- commas or whitespace, ``#`` comments, header line
skipped. A five-column file (``x y z u v``, ``massconsistent_amr``'s
``read_velocity_file`` format) is also accepted, with ``w`` taken as
zero. Values are interpolated with a 3D inverse-distance weighting over
the ``k`` nearest points, with the same exact-hit behaviour as the
terrain interpolation.

The file's ``z`` is read as **height above ground**, so the user profile
is terrain-following like the analytic laws.

Terrain awareness
=================

The profile is a function of

.. code-block:: none

    z_agl = z_cc(k) - z_terrain(i,j)

evaluated per column, so every column's profile starts at its own
ground. This holds on the lateral boundaries too: terrain may intersect
any of ``xlo``/``xhi``/``ylo``/``yhi``, leaving that face only partly
open.

Two consequences are handled explicitly:

* ``z_agl`` is negative inside the terrain, and the log law diverges as
  ``z_agl -> 0``. It is floored at ``inflow.z_agl_min`` before either law
  is evaluated, so a near-surface cell can never produce a huge speed or
  a NaN.
* Velocity is zeroed in solid cells, so the immersed boundary and the
  profile agree from the outset rather than only after the projection.

Boundary mass flux
==================

Because terrain blocks part of a face, the open areas of the inflow and
outflow faces generally differ, so the raw profile has a nonzero net
boundary flux. The outward flux is integrated over open (fluid) faces
only, using the stretched ``dz(k)``, and reported:

.. code-block:: none

    inflow_flux_in         inflow_flux_out
    inflow_flux_net        inflow_flux_imbalance

where ``flux_imbalance = |net| / max(|in|, |out|)``.

**This is a diagnostic, not an error, and nothing is rescaled.** The
mass-consistent solve is itself the mass correction, and it stays well
posed as long as at least one boundary face carries a Dirichlet
condition on ``lambda``.

Two further reasons no correction is applied here:

* Scaling the whole field cannot fix it. Multiplying ``u0`` by ``s``
  scales inflow and outflow alike, so the net scales by ``s`` and only
  reaches zero if it was already zero.
* Any correction that *could* work adjusts the outflow faces alone, and
  that requires knowing which faces those are -- a classification that
  belongs with the directional boundary conditions, not here.

The magnitude is still worth watching. Whatever imbalance remains leaves
through the ``lambda = 0`` faces, so a large one means a large
correction concentrated there.
