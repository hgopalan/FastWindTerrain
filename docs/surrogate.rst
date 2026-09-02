==========================
Surrogate groundwork
==========================

Phases 17-19 of the U-FNO programme: freeze the operator the training data
will come from, build the two operators that turn a 3D field into a stack
of 2D levels and back, and measure what that reconstruction costs.

**No machine learning appears anywhere here.** That is the point. The
reconstruction error measured below is a ceiling: a surrogate predicting
those levels cannot beat it, so it is worth knowing before any network is
designed.

The frozen operator
===================

The dataset is generated at ``poisson.n_projections = 4`` with anisotropy
on and the catalogue's defaults. Four passes is **not** a converged
projection -- it is a stated and consistent one, and the distinction
matters enough to record.

Measured on the Creek Fire case, comparing each pass count against the
24-pass field:

.. list-table::
   :widths: 16 16 22 22
   :header-rows: 1

   * - passes
     - wall clock
     - ``max|div|``
     - distance to 24 passes
   * - 1
     - 13 s
     - 0.1153
     - 67 %
   * - 4
     - 51 s
     - 0.1018
     - 34 %
   * - 8
     - 93 s
     - 0.0776
     - 21 %
   * - 16
     - 176 s
     - 0.0460
     - 7 %
   * - 24
     - 260 s
     - 0.0293
     - --

The outer iteration converges geometrically at about 0.87 per pass, and
**that rate is not a property of the transmissivity**: running the same
case at ``alpha_h/alpha_v`` ratios of 40:1, 8:1, 2:1 and 1:1 gives factors
of 0.872, 0.873, 0.906 and 0.881. MLMG itself converges in 4-6 iterations
throughout, so the cost is in the outer loop, not the linear solve. The
isotropic case is the slowest per pass, not the fastest.

This is known behaviour of mass-consistent solvers and is the motivation
for a surrogate rather than a defect to fix. A surrogate trained on this
data learns "FastWindTerrain at four passes", which is a defensible thing
to name in a methods section and an indefensible thing to leave implicit.

Levels
======

``fastwindterrain.levels`` extracts horizontal slices from a 3D field and
stitches them back.

.. list-table::
   :widths: 34 66
   :header-rows: 1

   * - Call
     - Meaning
   * - ``extract_levels(field, ...)``
     - ``(nz, ny, nx)`` to ``(nlev, ny, nx)``
   * - ``stitch_levels(values, ...)``
     - and back again
   * - ``obrien_w(u, v, w, ...)``
     - ``w`` from column-integrated continuity
   * - ``surface_kinematic_w(...)``
     - ``w = u.grad(h)`` at the first fluid cell, to seed the integration
   * - ``height_above_ground``, ``first_fluid_k``, ``log_law``
     - the geometry underneath

The level set is ``ENGINEERING_LEVELS`` -- 10 m for a met mast, 80 to 160 m
for hub heights -- plus ``ALOFT_LEVELS`` at 300, 600 and 1200 m. The aloft
levels are not decoration; see the results below. ``DIAGNOSTIC_LEVEL`` is
2 m, kept separate because with ``dz0 = 4 m`` the first fluid cell above
terrain sits 0 to 4 m above the surface, so 2 m is sub-grid in most
columns and comes from a log law rather than from the mesh.

O'Brien in numpy
----------------

``obrien_w`` is a transcription of ``Obrien::Apply``
(``Source/Obrien.cpp:74``). It exists because the C++ operator cannot go
where it is needed -- inside a training loop, where the reconstruction has
to be differentiable.

It is validated against the C++ one without any binding change:
``Solver.cpp:113`` copies ``velocity0`` *after* the adjustment has run, so
running a case with ``obrien.enable`` off and on gives two fields
differing by exactly one application of it. The numpy version applied to
the first must reproduce the second.

Agreement is to **at most 2 units in the last place**, averaging a quarter
of one, not accumulating up the column, and exactly zero at the domain
top. That is one rounding per step: clang contracts ``w -= Dh * dz[k]``
into a fused multiply-subtract, rounding once where numpy rounds twice,
and numpy has no FMA to match it with. In float32 training this is eight
orders of magnitude below the noise.

What reconstruction costs
=========================

``cases/stitching_study.py`` takes a solved field, keeps only ``u`` and
``v`` on K levels, rebuilds the 3D field, and compares against what it
deleted. Errors are RMSE over fluid cells, scaled by the true ``|U|``
maximum.

.. list-table:: Whole-column error, and error within the 10-160 m band
   :widths: 22 20 20 19 19
   :header-rows: 1

   * - levels
     - Creek, column
     - Bootleg, column
     - Creek, 10-160 m
     - Bootleg, 10-160 m
   * - 5 (engineering only)
     - 0.198
     - 0.116
     - 0.018
     - 0.004
   * - 6 (+600 m)
     - 0.053
     - 0.028
     - 0.018
     - 0.004
   * - 8 (+300, 600, 1200 m)
     - **0.016**
     - **0.011**
     - 0.018
     - 0.004
   * - 12 (dense)
     - 0.012
     - 0.010
     - 0.007
     - 0.002

Three things fall out of that table.

**The engineering levels alone are not enough for a 3D field.** Five levels
covering 10-160 m leave 20 % error over the column on Creek, because on
this grid they span only its bottom third and everything above is being
held constant. One extra level at 600 m cuts that to 5 %; three cut it to
1.6 %.

**But they are enough for the band people ask about.** Error inside
10-160 m is flat at 0.018 (Creek) and 0.004 (Bootleg) whether you predict
5 levels or 8. So the answer depends entirely on what the deliverable is:
hub-height wind needs five levels, a 3D field needs eight.

**Diminishing returns arrive early.** Twelve levels buy little over eight
in the column, though they do help inside the band.

Ablations, at 8 levels
----------------------

.. list-table::
   :widths: 34 22 22 22
   :header-rows: 1

   * - variant
     - Creek, column
     - Bootleg, column
     - note
   * - baseline (agl, loglinear)
     - 0.016
     - 0.011
     -
   * - **frame = cartesian**
     - **0.792**
     - **0.842**
     - catastrophic
   * - method = linear
     - 0.019
     - 0.017
     - and 0.024 vs 0.018 in the band
   * - w interpolated, not O'Brien
     - ``rmse_w`` 0.006
     - 0.002
     - vs 0.045 and 0.008
   * - w seeded at 0, not kinematic
     - ``rmse_w`` 0.031
     - 0.014
     - vs 0.045 and 0.008

**Terrain-following levels win by a factor of fifty.** A slice at constant
elevation over 1100 m of relief is underground across much of the domain,
so most of what it samples is not flow. This is the clearest result in the
study and it settles the frame question. (The divergence reported for the
Cartesian variant is not meaningful -- the field is degenerate.)

**Log-linear interpolation beats linear**, modestly over the column and by
four-fold inside the band on Bootleg, which is where the profile really is
logarithmic.

**Interpolating w beats deriving it from continuity**, by roughly eight
times. That is not a free win: interpolating w requires the network to
predict three fields instead of two, and this study gave it *perfect* w at
the levels. What the number does establish is a floor -- O'Brien's w
carries 4.5 % error on Creek even from perfect ``u`` and ``v``, because it
inherits the reconstruction error in those and then differentiates and
integrates it.

**Seeding the integration is terrain-dependent**, and my prediction was
wrong. The kinematic condition ``w = u.grad(h)`` helps on Bootleg (0.008
against 0.014) and *hurts* on Creek (0.045 against 0.031). The plausible
reason is that on slopes reaching 1.89, ``u.grad(h)`` implies a surface
``w`` far larger than the flow actually has -- because there the solver's
suppressed ``alpha_v`` is pushing air around the obstacle rather than over
it, which is exactly what the kinematic condition assumes it does not do.

Reproducing it
--------------

::

    python3 cases/creek_fire/prepare.py
    python3 cases/stitching_study.py --case creek_fire --figure study.png

Each case costs one solve, and every reconstruction after that is numpy.

What transfers
==============

The mass-consistent solver is a stand-in: it is cheap enough to afford
abundant ground truth while the method is established, and the intended
target is a fractional-step solver where running it is expensive enough
for a surrogate to pay for itself. So it is worth being explicit about
which of these results are about geometry and which are about this
particular physics.

**Transfers.** Terrain-following extraction and stitching, the level
counts, the frame result and the interpolation-method result are geometry
and vertical structure. They say nothing about how the wind was computed.

**Does not transfer cleanly.** Everything about ``w``. In a
mass-consistent solve ``w`` is already a derived quantity -- O'Brien
computed it from ``u`` and ``v`` before the projection ever ran -- so
rebuilding it from continuity here is partly circular. In a
fractional-step solver ``w`` is an independent dynamical variable and the
comparison would have to be repeated. Column-integrated continuity itself
does carry over, since incompressibility is not specific to this
formulation, but the quadratic top-boundary redistribution is a
mass-consistent convention.

**Irrelevant later.** ``n_projections``, the transmissivity ``alpha``, and
the convergence behaviour above are properties of this solver only.
