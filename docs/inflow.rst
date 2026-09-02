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
   * - ``inflow.balance_flux``
     - ``0``
     - Redistribute the net boundary flux over the lateral faces

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

**By default this is a diagnostic, not an error, and nothing is
rescaled.** The mass-consistent solve is itself the mass correction, and
it stays well posed as long as at least one boundary face carries a
Dirichlet condition on ``lambda``.

The magnitude is worth watching either way. Whatever imbalance remains
leaves through the ``lambda = 0`` faces, so a large one means a large
correction concentrated there.

Redistributing it
-----------------

``inflow.balance_flux = 1`` spreads the net over the boundary instead of
only reporting it. One uniform outward-normal velocity

.. code-block:: none

    shift = -flux_net / (open area of xlo + xhi + ylo + yhi)

is added to every open cell of the four lateral faces, so ``in`` and
``out`` match to round-off before the Poisson solve runs. A corner cell
belongs to two faces and takes one shift in each of those two
components. The top is left alone -- it carries ``w = 0`` by boundary
condition, and pushing flow through it would contradict that -- and the
ground is closed.

A whole-field rescale would not work: multiplying ``u0`` by ``s`` scales
inflow and outflow alike, so the net scales by ``s`` and only reaches
zero if it was already zero. The correction has to be additive and it
has to live on the boundary.

**It is a cleaner starting point, not an accuracy fix.** On the terrain
corpus the pre-solve imbalance runs 0.7% to 27%, but the projection
already drives it to 2e-4 -- 3.5e-3 by itself, and as a bulk divergence
(net flux over fluid volume) the initial imbalance is 6e-7 to 1.4e-5
1/s against a local ``div_l2`` of 3e-3 to 1e-2 -- three orders smaller.
So it is off by default, and turning it on should not be expected to
move an answer. It does not: see the measurement below. What it buys:

* the initial field is exactly conservative, which is standard practice
  in urban CFD and a cleaner thing to hand a projection;
* it removes a warning that fires on most real terrain windows and is
  expected to.

The report carries both states, so one run says what the profile
carried and what was done about it:

.. code-block:: none

    inflow_balance_flux          0 or 1
    inflow_flux_balance_shift    the shift [m/s], 0 when off
    inflow_flux_imbalance_raw    before redistribution
    inflow_flux_net_raw          before redistribution

Two consequences, both handled in the code rather than left to chance:

* **Face classification does not move.** The boundary conditions
  classify each lateral face from the flux the *raw* profile carries
  (``Inflow::flux_prebalance``). Classifying from the redistributed
  field would call a tangential face -- one the wind does not blow
  through at all -- inflow or outflow, and where the shift points inward
  it would produce three inflow faces and trip the assertion that keeps
  the Poisson operator non-singular. Which face the wind enters through
  is a property of the wind, which is the same reason the projection
  never reclassifies anything either.
* **The inflow ghost cells carry the shift too.** They are prescribed
  from the profile rather than copied from the interior, so without it
  the ghost and the interior cell it faces would differ by exactly the
  shift and the redistribution would show up as a divergence step across
  the face. Outflow and tangential ghosts are zero-gradient copies and
  inherit it already.

What it does to a solve
-----------------------

The shift lands entirely in the first cell layer, so on a smooth case
the *initial* field's divergence rises where the correction meets the
interior. On the ``inputs_boundary_terrain`` regtest case -- an 11%
imbalance, a 0.53 m/s shift, no terrain roughness to speak of -- the
controlled divergence norm of the initial field goes 0.062 to 0.137,
while after four passes it is 0.0370 against 0.0369. There the
post-solve flux imbalance improves (3.0e-3 to 1.8e-3) and so does
``div_l2`` (9.7e-3 to 9.1e-3).

On real terrain none of that is visible, because the boundary layer is
nowhere near the largest source of divergence. Two corpus windows at
10 m/s and 225 deg, ``max_divergence_fe`` after each projection pass:

============================  =====  =====  =====  =====  =====  =====
``ditch_fire:20`` (63.6% s.)  0      1      2      4      8      16
============================  =====  =====  =====  =====  =====  =====
off                           0.194  0.227  0.250  0.259  0.251  0.202
on (shift -0.665 m/s)         0.194  0.227  0.249  0.258  0.251  0.202
============================  =====  =====  =====  =====  =====  =====

============================  =====  =====  =====  =====  =====  =====
``kincade_fire:20`` (59.5%)   0      1      2      4      8      16
============================  =====  =====  =====  =====  =====  =====
off                           0.152  0.175  0.177  0.161  0.140  0.089
on (shift +0.861 m/s)         0.152  0.175  0.177  0.161  0.140  0.089
============================  =====  =====  =====  =====  =====  =====

The raw imbalance on these is 12.5% and 17.1%, and the option drives it
to 5e-16 and 0 exactly -- the field the Poisson solve sees is balanced
to round-off, O'Brien's vertical adjustment included, which runs after
it and does not disturb it. And the projection does the same thing
anyway. In particular **the non-monotone rise over the first passes
(:doc:`corpus`) is not caused by the boundary flux imbalance**: it is
there, unchanged, with the initial field exactly conservative.

The post-solve flux imbalance comes out slightly worse with the option
on -- 1.7e-3 to 2.2e-3 on Ditch, 1.1e-3 to 1.5e-3 on Kincade -- and
``div_l2`` moves either way (9.05e-3 to 9.40e-3 on Ditch, 9.66e-3 to
8.87e-3 on Kincade). So the case for turning it on is the conservative
starting point itself, not a better answer.
