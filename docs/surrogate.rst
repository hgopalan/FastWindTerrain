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

Where to put the levels
=======================

The count is what a paper usually reports; the placement is what has to be
reproduced. At fixed k, four distribution rules across 10-1600 m above
ground, measured on three terrains (``--placement``):

.. list-table:: Column RMSE at fixed level count
   :widths: 10 26 21 21 21
   :header-rows: 1

   * - k
     - rule
     - Creek
     - Bootleg
     - Thomas
   * - 5
     - uniform
     - 0.0353
     - 0.0164
     - 0.0203
   * - 5
     - **log**
     - **0.0170**
     - **0.0113**
     - **0.0127**
   * - 5
     - engineering heights only
     - 0.1977
     - 0.1164
     - 0.1184
   * - 8
     - log
     - 0.0129
     - 0.0105
     - 0.0115
   * - 12
     - uniform
     - 0.0187
     - 0.0116
     - 0.0144
   * - 12
     - log
     - 0.0119
     - 0.0104
     - 0.0113

**Five log-spaced levels beat twelve uniform ones, on every terrain
tested.** Placement matters more than count, which is the useful form of
the result: it transfers to a grid that is not this one.

Sample where the shear is, not where the answer is
--------------------------------------------------

Anchoring the level set on the heights people ask for -- 10, 80, 100, 120,
160 m -- is worse than log spacing **even for accuracy at those heights**:

.. list-table:: RMSE inside the 10-160 m band, k = 8
   :widths: 34 22 22 22
   :header-rows: 1

   * - level set
     - Creek
     - Bootleg
     - Thomas
   * - log-spaced
     - 0.0099
     - 0.0014
     - 0.0039
   * - anchored on 10/80/100/120/160
     - 0.0183
     - 0.0040
     - 0.0097

Two to three times worse, consistently, and the reason is visible in the
level lists: anchoring leaves **nothing between 10 and 80 m**, which is
where the shear actually is. Log spacing puts levels at 21, 43 and 88 m in
that gap and reconstructs the whole band better as a result -- including
at 100 and 120 m, which it never samples.

So the heights an answer is wanted at are not the heights samples should
be taken at. Sample logarithmically and interpolate out to the engineering
heights.

Dividing the budget
-------------------

A fixed number of levels has to be shared between the band, where the
answer is wanted, and the column above, which has to be spanned for a 3D
reconstruction. Sweeping that split at k = 8 (``--split``):

.. list-table:: Column RMSE by split
   :widths: 28 24 24 24
   :header-rows: 1

   * - split
     - Creek
     - Bootleg
     - Thomas
   * - 2 band + 6 aloft
     - 0.0189
     - 0.0117
     - 0.0145
   * - 3 + 5
     - 0.0137
     - 0.0105
     - 0.0121
   * - **4 + 4**
     - **0.0127**
     - **0.0104**
     - 0.0116
   * - **5 + 3**
     - 0.0130
     - 0.0105
     - **0.0115**
   * - 6 + 2
     - 0.0150
     - 0.0110
     - 0.0119
   * - 7 + 1
     - 0.0239
     - 0.0150
     - 0.0152

Both extremes fail for opposite reasons -- too few band levels
under-resolve the shear, too few aloft fail to span the column -- and
everything between 4+4 and 5+3 is within 2 % of the best on all three
terrains. A flat optimum is what a recipe wants.

The recommended set
-------------------

**Five levels octave-spaced across the band, three log-spaced above it:**

::

    10, 20, 40, 80, 160, 345, 743, 1600 m above ground

``geomspace(10, 160, 5)`` has a ratio of exactly 2, so the band levels are
octaves, and the set contains 10, 80 and 160 m outright. Measured: column
RMSE 0.0130 / 0.0105 / 0.0115 and band RMSE 0.0093 / 0.0013 / 0.0037 on
Creek / Bootleg / Thomas -- within a few per cent of the best achievable
at any split, on both metrics at once.

At k = 12 the pattern holds at 7+5 or 8+4. The column stops improving past
about six band levels while the band keeps improving, so the extra levels
are worth buying only when the band is the deliverable.

How many levels are worth having
--------------------------------

**Levels cost nothing in solver time.** All k are extracted from the same
solved field, so a sample is 51 s whether five levels are kept or twenty.
The budget trades against network size, not against how much data can be
afforded.

And the ceiling adds in quadrature with the surrogate's own error, which
is what settles it:

.. list-table:: Total error, ceiling combined with network error
   :widths: 22 20 20 20 18
   :header-rows: 1

   * - network error
     - k = 5
     - k = 8
     - k = 12
     - gain, 5 to 12
   * - 10 %
     - 10.09 %
     - 10.07 %
     - 10.06 %
     - 0.3 %
   * - 5 %
     - 5.18 %
     - 5.13 %
     - 5.12 %
     - 1.1 %
   * - 2 %
     - 2.42 %
     - 2.31 %
     - 2.29 %
     - 5.4 %
   * - 1 %
     - 1.69 %
     - 1.53 %
     - 1.50 %
     - 11.3 %

At 5 % network error, going from five levels to twelve buys 1.1 % of total
accuracy for 2.4 times the output channels. The level count only begins to
matter below about 2 %.

The rule, which is more use than the number: **choose the smallest k whose
ceiling is under a third of the network's own error.** The recommended
eight-level set has a ceiling near 1.2 % **on Creek**, so it is
comfortable for anything above about 3 %. See the revalidation below
before carrying that number to other terrain -- the ceiling runs 0.4 % to
1.5 % across the corpus and grows with relief.

Spend the real budget on terrains instead. At 51 s a solve, eight terrains
by sixty wind conditions is 480 solves -- under an hour on eight cores.

Reproducing it
--------------

::

    python3 cases/creek_fire/prepare.py
    python3 cases/stitching_study.py --case creek_fire --figure study.png
    python3 cases/stitching_study.py --placement --case creek_fire
    python3 cases/stitching_study.py --split --case creek_fire

Each case costs one solve, and every reconstruction after that is numpy --
which is why the placement and split sweeps are nearly free once the field
exists.

Warm starting: a negative result
================================

If the surrogate's output is an initial condition rather than an answer,
the figure of merit is not RMSE but **iterations saved**. The projection
is a stationary iteration, so that is directly measurable:
``cases/warmstart_study.py`` seeds it from a reconstruction instead of
from the solver's own initial field and counts passes to a fixed target.

**It does not work well here, and the number is worth recording so that
nobody re-derives the idea and re-runs the experiment.**

.. list-table:: Passes to reach the divergence the cold start reaches at 12
   :widths: 30 18 12 18 12
   :header-rows: 1

   * - start
     - Creek
     - saved
     - Bootleg
     - saved
   * - cold (solver default)
     - 12
     - --
     - 12
     - --
   * - warm, perfect levels
     - 7
     - 5
     - 11
     - 1
   * - warm, 2 % level noise
     - 8
     - 4
     - 11
     - 1
   * - warm, 5 % level noise
     - 9
     - 3
     - 13
     - **-1**
   * - warm, 20 % level noise
     - 16
     - **-4**
     - 18
     - **-6**

Even a *perfect* reconstruction saves 5 passes of 12 on Creek and 1 of 12
on Bootleg. At a plausible 5 % surrogate error it saves 3 and nothing, and
by 20 % it is worse than the initial condition the solver builds for
itself.

Why, and why it is not a bug
----------------------------

The mechanism checks out, which is how the number is known to be real. The
perfect reconstruction sits 0.447 from the fixed point in max norm against
the cold start's 0.915 -- exactly twice as close. At a convergence factor
of 0.87 a two-fold reduction predicts ``ln 2 / ln(1/0.87)`` = 5 passes.
Measured: 5.

So the reconstruction is simply not much closer to the answer than the
solver's own guess **in the norm the iteration converges in**. Its RMSE is
excellent -- about 1.3 % on speed -- and its max-norm error is about 44 %,
and the projection converges in a max norm.

The obvious hope is that those worst cells sit near the terrain, where
stitching is hardest and a better near-surface treatment would fix them.
They do not:

.. list-table:: Max-norm error by height above the surface, Creek
   :widths: 30 20 20 30
   :header-rows: 1

   * - cells above the ground
     - max
     - rmse
     - share of fluid cells
   * - 1st fluid cell
     - 0.441
     - 0.041
     - 6 %
   * - 2-3
     - 0.296
     - 0.048
     - 12 %
   * - 4-10
     - 0.335
     - 0.045
     - 40 %
   * - 11+
     - 0.331
     - 0.028
     - 42 %

The error is spread through the column, so there is no localised fix to
be had.

What to conclude
----------------

Do not claim warm-start value on this evidence. The stitching recipe above
stands on its own and is the stronger result.

The idea is not dead for a fractional-step solver -- each step there costs
far more, so a given fractional reduction in iterations is worth much more
wall clock, and both the iteration and the norm it converges in are
different. But that has to be measured on that solver rather than inferred
from this one.

**Caveat on the measurement.** One random seed per noise level, and the
non-monotonicity -- 10 % noise saved more than 5 % on Creek -- shows it is
noisy. Several seeds would tighten it. The gap between "5 passes saved
from a perfect reconstruction" and "12 passes to beat" is wide enough that
this is unlikely to change the conclusion.

::

    python3 cases/warmstart_study.py --case creek_fire --case bootleg_fire

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

Revalidation on the corpus, after the surface condition
========================================================

Everything above was measured on three catalogue cases with **nothing
constraining the first fluid cell above terrain**. The surface condition
(:doc:`terrain`) now puts the kinematic ``w`` there by construction, which
changes ``w`` everywhere the projection carries it, so the results were
re-measured against the operator that actually runs --
``cases/revalidate_levels.py``, eight corpus windows spanning 99 m to
1813 m of relief.

Errors here are RMSE over fluid cells scaled by ``|U_h|max``, and quoted
in m/s as well, because a dimensionless residual is not something a
tolerance can be applied to. ``|U_h|max`` runs 14.5-18.7 m/s across these
windows.

What held
---------

**Placement.** Inside the 10-160 m band, at eight levels:

=====================  ============  ==========  ======================
window                  recommended     uniform   engineering-anchored
=====================  ============  ==========  ======================
``bootleg_fire:22``          0.0017      0.0086                 0.0043
``carr_fire:12``             0.0062      0.0195                 0.0134
``slinkard_fire:22``         0.0170      0.0408                 0.0278
``ditch_fire:20``            0.0227      0.0526                 0.0306
=====================  ============  ==========  ======================

Uniform is 2.3-5x worse and engineering-anchored 1.3-2.5x worse, on every
window. The phase 19 claim survives on terrain it never saw.

**Interpolating** ``w`` **rather than deriving it**, by more than before:
12-17x against O'Brien, up from 8x. On ``ditch_fire:20`` O'Brien's ``w``
carries **2.30 m/s** of error against 0.18 m/s interpolated. The kinematic
seed is now built in, and on steep ground it is large, so reconstruction
error in ``u`` and ``v`` propagates through it hard.

What did not hold
-----------------

**The ceiling is not one number.** It scales with relief:

=====================  ==========  ==============  ==========
window                    relief     column RMSE        m/s
=====================  ==========  ==============  ==========
``bootleg_fire:22``          99 m          0.0040       0.058
``carr_fire:12``            311 m          0.0075       0.114
``slinkard_fire:22``       1069 m          0.0107       0.185
``ditch_fire:20``          1813 m          0.0153       0.286
=====================  ==========  ==============  ==========

**0.4 % to 1.5 %**, or 0.06 to 0.29 m/s -- not the ~1.2 % measured on
Creek and quoted above as though it were general. The budget rule still
applies; the number it takes must come from terrain of comparable
complexity.

**``recommended`` and plain ``log`` are a tie.** Log wins the column on 4
of 8 windows and recommended wins the band on 6 of 8, with gaps of about
2 %. Pinning the band to exact octaves is not doing measurable work.

Near the surface
----------------

The 0-50 m band is the worst one on every window, and it grows with
relief -- 0.149, 0.308, 0.518, 0.374, 0.416 m/s. Past the ~0.25 m/s that
CFD practice runs at, from 311 m of relief upward.

**That is expected rather than wrong.** Near-surface flow follows the
terrain, so it varies on the terrain's own length scale; aloft it is
smooth and terrain-blind. Both the reconstruction and, later, the network
will be worst there, and it is where the answer is wanted.

A level at 5 m recovers part of it:

=====================  ===========  =============  ==============
window                 recommended   + 5 m (k+1)   + 5 m (same k)
=====================  ===========  =============  ==============
``carr_fire:12``             0.308          0.220           0.220
``delta_fire:20``            0.518          0.402           0.402
``ditch_fire:20``            0.416          0.365           0.365
=====================  ===========  =============  ==============

12-29 % better, and **only as an extra level**. Swapping it in at fixed
``k`` costs the aloft level it displaces and wrecks the column -- 0.114 to
0.392 m/s on ``carr_fire:12``. So the near-surface error is partly
vertical resolution and mostly not.

What a surrogate has to beat
=============================

``cases/baseline_study.py`` measures the cheap analytical fields in
:mod:`fastwindterrain.baseline` on the same windows and metric.

**The undisturbed profile is the baseline.** Terrain-following but
terrain-blind, it is what the solver starts from:

=====================  ==========  ==========
window                    relief         m/s
=====================  ==========  ==========
``bootleg_fire:22``          99 m       0.251
``carr_fire:12``            311 m       0.367
``delta_fire:20``           521 m       0.843
``slinkard_fire:22``       1069 m       1.460
``ditch_fire:20``          1813 m       1.462
=====================  ==========  ==========

Doing nothing costs 0.25 m/s on gentle ground and 1.46 m/s on complex
ground. **That is the value proposition, quantified**: near nothing on
easy terrain and roughly six times the tolerance on hard terrain. Against
it, the reconstruction floor on ``ditch_fire:20`` is 0.29 m/s, so
stitching preserves about 80 % of the terrain effect and leaves a factor
of five of headroom for a network.

**The other two baselines are broken and are not quotable.**
``continuity_speedup`` and ``slope_speedup`` reach ``|U|max`` of 41.7 and
36.8 m/s on ``ditch_fire:20`` against the solver's 18.65. Both assume a
shallow hill -- linearised theory, and a column speed-up with no decay in
height -- and the corpus reaches slopes near 2. They are kept because a
baseline that fails loudly is more useful than one quietly omitted, but
any comparison should use ``undisturbed``.

Imposing the log law near the surface: a negative result
=========================================================

If the reconstruction is good aloft and poor near the ground, an obvious
repair is to take a level it handles well, invert the log law there for a
friction velocity, impose the resulting profile below, and taper the
correction out with height. That is
:func:`fastwindterrain.levels.log_blend_correction`, and it does not work.

============================  ==============  ==============
``carr_fire:12`` (311 m)      0-50 m           10-160 m
============================  ==============  ==============
no correction                  0.740 m/s        0.444 m/s
anchor 160 m, 10 % noise       **0.676**        **0.361**
============================  ==============  ==============

On the gentlest window tested it repairs 10 % noise usefully. On
``slinkard_fire:22`` and ``ditch_fire:20`` it is worse at every anchor and
every taper, noisy or not. With *perfect* levels it is worse everywhere,
which is expected -- ``stitch_levels`` reproduces the level values exactly
at the levels, so there is nothing to repair and a correction can only
move a right answer.

**The mechanism is the same physics that makes the band hard.** Near the
surface the flow follows the terrain rather than a universal profile, so
imposing a log law there imposes the wrong shape exactly where it matters
most. It works on gentle ground because gentle ground is where a profile
is a good description.

Two consequences:

* **do not use it as a decoder** inside the surrogate, and in particular
  do not drop the sub-40 m levels and derive them -- the log law cannot
  supply what they carry on complex terrain;
* the correction is kept, tested and documented so the result is
  reproducible rather than folklore.

Two implementation notes, if it is ever revisited. Anchor on a level the
network predicts, not an arbitrary height: interpolating the anchor speed
from cell centres 20 m apart on a logarithmic profile injects 0.003 m/s
of bias that propagates down the column. And taper linearly in height --
the log taper is elegant and far too aggressive, weighting 0.31 at 10 m
and 0.10 at 40 m for an 80 m anchor, and it repaired only 12 % of a
deliberate 50 % near-surface error.

What a surrogate has to beat, over the corpus
=============================================

Phase 22a. Everything above measured a handful of windows at one
direction. This is the whole test fold -- 216 independent solves, 54
windows at four directions -- scored with
:mod:`fastwindterrain.evaluate` by ``cases/eval_harness.py``.

The metric is **vector RMSE in metres per second**, over fluid cells,
against the solver's own field. Vector, not speed: a field with the right
magnitude and the wrong direction is wrong, and a speed metric scores it
perfectly.

Two fields bracket what a surrogate can do. The **baseline** is the
undisturbed profile -- terrain-following but terrain-blind, the field
available for nothing. The **floor** is the dataset's own nine levels
stitched back into the sixty-layer grid: a perfect network reproducing
the stored levels exactly still lands there, because nine levels do not
carry sixty.

==========  ===========  ====  ============  =========  ==========
group       relief (m)      n      baseline      floor    headroom
==========  ===========  ====  ============  =========  ==========
gentle           0-200     84         0.542      0.070       0.473
moderate       200-500     28         0.755      0.075       0.680
complex        500-900     64         2.069      0.197       1.872
extreme           900+     40         2.552      0.259       2.292
==========  ===========  ====  ============  =========  ==========

Means, in m/s. The worst single sample in each bin runs to 1.23, 1.54,
3.02 and 3.22 m/s of baseline error and 0.17, 0.15, 0.28 and 0.34 m/s of
floor.

**The headroom is much larger than the eight-window studies suggested.**
Those measured 0.25 to 1.46 m/s of baseline error; over the corpus it
reaches 3.2. The earlier windows simply did not include the steepest
ground. The floor moved far less -- 0.06-0.29 became 0.05-0.34 -- so the
gap a surrogate is being asked to close is wider, not narrower, than the
groundwork implied.

**Relief predicts both, and predicts them differently.** From gentle to
extreme the baseline error grows 4.7x and the floor 3.7x, so the ratio
between them is roughly flat: the floor is one part in 7.8 of the
baseline on gentle ground and one in 9.8 to 10.5 on everything steeper,
so a fixed tenth or so of the terrain effect is unrecoverable from nine
levels whatever the terrain. That is the useful invariant -- it means the
level placement is not quietly failing on steep ground, it is failing
proportionately. If anything it does slightly *better* there.

**Do not report one aggregate number.** The mean over all 216 samples is
1.394 m/s of baseline and 0.143 m/s of floor, and it describes no bin: it
is 2.6x too high for gentle terrain and 1.8x too low for extreme. Every
table in the paper should be grouped.

The demonstration sites, scored the same way, land where their relief
says they should:

==========  ===========  ====  ============  =========  ==========
group       relief (m)      n      baseline      floor    headroom
==========  ===========  ====  ============  =========  ==========
complex        500-900     36         1.991      0.177       1.814
extreme           900+     36         2.446      0.229       2.217
==========  ===========  ====  ============  =========  ==========

Both bins sit slightly *better* than the matching test-fold bins (0.177
against 0.197, 0.229 against 0.259). Unseen terrain is not pathological
terrain -- the difficulty is set by relief, not by whether the corpus has
seen the site. A model that transfers should therefore land near its
test-fold numbers on these, and a large gap would be evidence of
memorisation rather than of hard ground.

Reproducing it
--------------

.. code-block:: console

   $ python3 cases/eval_harness.py
   $ python3 cases/eval_harness.py --data data/demo --fold demo

About thirty seconds for the test fold. It scores only the solved half:
the derived samples are exact negations and both the baseline and the
stitch are odd in the wind direction, so they measure identically.

Where in the column the floor error lives
-----------------------------------------

``--by-height`` breaks the same samples down by height above ground. The
question is not how error varies with height but whether it lands where
the answer is wanted.

==========  ========  ========  ==========  ==========
band (AGL)      RMSE       p95    max cell    baseline
==========  ========  ========  ==========  ==========
0-10 m         0.372     0.760       8.749       2.179
10-50 m        0.187     0.397       4.351       1.593
50-160 m       0.128     0.258       5.576       1.644
160-500 m      0.048     0.096       2.134       1.585
500+ m         0.093     0.189       2.482       0.984
==========  ========  ========  ==========  ==========

**The error is worst at the surface, on every statistic.** It is not
concentrated aloft, which would have been the comfortable answer: the
floor falls monotonically from 0.372 m/s at the surface to 0.048 m/s at
160-500 m, then rises again above 500 m where the aloft levels are widely
spaced.

**But the cause is extrapolation, not resolution.** Splitting the column
at the lowest level, 5 m AGL:

==============================================  ==========  ============
region                                                RMSE   fluid cells
==============================================  ==========  ============
below the lowest level -- log-law FILL               0.467         1.4 %
at or above it -- INTERPOLATED                       0.112        98.6 %
==============================================  ==========  ============

The fill is **4.2x** the interpolated error. Nothing between the levels is
struggling; the whole near-surface penalty is the log law imposed below
the lowest one -- the same wrong-shape-near-the-surface mechanism that
made ``log_blend_correction`` a negative result, confirmed here from a
different direction.

**In relative terms it is inside tolerance.** Below 5 m the mean speed is
4.25 m/s against 0.467 m/s of error: **11.7 %**, worst sample 20.7 %.
Judged in m/s alone the surface band looks like a failure; judged against
the 20-30 % that turbulent flow carries in the field, it is not. Convert
before concluding.

**Name the statistic every time.** The single worst cell runs 23-44x the
RMSE in every band. That is the same phenomenon that killed warm starting
-- ~1.3 % RMSE against ~44 % max-norm error -- so a max-norm criterion
will always look alarming here while an RMSE one looks fine. Neither is
wrong; they measure different things.

One experiment this suggests for the training phase: **add a level below
5 m**, so the bottom cell is interpolated rather than extrapolated. One
extra output channel, aimed at the only band outside tolerance in
absolute terms.

How much of the domain is over half a metre per second
------------------------------------------------------

An RMSE of 0.19 m/s can be a uniform 0.19 or a quiet field with two per
cent of it at 1.5 m/s, and those are different problems with different
fixes. ``cases/error_maps.py --histogram`` bins the absolute vector error
per level over the whole test fold.

========  ==========  ==========  ============
level           slab    over 0.5      over 1.0
========  ==========  ==========  ============
5 m            0-7 m     13.78 %        2.81 %
10 m          7-14 m      2.65 %        0.13 %
20 m         14-28 m      3.68 %        0.66 %
40 m         28-57 m      4.86 %        1.13 %
80 m        57-113 m      2.13 %        0.40 %
160 m      113-261 m      0.64 %        0.09 %
312 m      261-691 m      0.10 %        0.01 %
607 m     691-1831 m      1.00 %        0.07 %
1184 m      1831-top      2.18 %        0.12 %
all                       1.60 %        0.24 %
========  ==========  ==========  ============

**98.4 % of all fluid cells are under 0.5 m/s.** The reconstruction is
not marginal; it is good nearly everywhere and bad in a small, locatable
minority.

**The tail tracks slab thickness, not height.** 40 m is the second-worst
level, worse than 10 or 20 m, because it owns 28-57 m -- the widest gap
inside the band, so its cells sit farthest from any stored level. The
same effect appears at 607 and 1184 m aloft. Only the 5 m level breaks
the pattern, and that one is the log-law fill rather than interpolation.

For contrast, the undisturbed baseline has just **19.1 %** of cells under
0.5 m/s at 5 m and 30.9 % over 2.0. The terrain-blind profile is not
merely worse on average near the ground; it is wrong almost everywhere.

Maps rather than numbers
------------------------

``cases/error_maps.py`` draws the same error as one panel per level. The
stitch reproduces the stored values exactly AT a level, so a map at 80 m
would be zero by construction; each level is instead given the slab
between the geometric midpoints either side of it -- geometric because
the levels are octaves, so the midpoint between 40 and 80 m belongs at
57 m, not 60.

What they show, on ``ditch_fire:10`` (1970 m relief, the steepest window
in the corpus):

* the near-surface panels are **speckled, not patterned** -- those slabs
  are one or two cells deep, and the error is isolated cells rather than
  a coherent field. That is the same thing the 23-44x max-to-RMSE ratio
  says, and it is why a max-norm criterion is the wrong one here;
* from 40 m upward the error becomes clearly **terrain-following**,
  concentrating on the steepest quadrant and along ridge lines. That part
  is structured, which is what makes it learnable;
* a column with no cell in a slab is drawn flat grey, not white. The two
  must not be confused: no data is not zero error.

Does the error follow the slope?
--------------------------------

``cases/slope_error.py`` correlates the per-column error against three
descriptors of the ground beneath it, over the whole test fold. Pearson
r, with the mean error on near-flat columns for scale. "slope" is the
slope magnitude ``sqrt(dh/dx^2 + dh/dy^2)``:

========  ==========  ============  ===========  ===========
level         slope     along-wind    curvature  flat-ground
========  ==========  ============  ===========  ===========
5 m            0.241        -0.003       -0.035        0.230
10 m           0.133         0.008        0.081        0.121
20 m           0.498        -0.010        0.179        0.042
40 m           0.563        -0.027        0.108        0.031
80 m           0.547        -0.045        0.031        0.022
160 m          0.549        -0.060       -0.005        0.014
312 m          0.716        -0.074        0.022        0.011
607 m          0.706        -0.065        0.008        0.022
1184 m         0.654        -0.050        0.108        0.036
========  ==========  ============  ===========  ===========

**Slope magnitude predicts the error, and predicts it strongly** --
r = 0.50 to 0.72 at every level from 20 m up. That is the single most
useful feature a network could be given.

**The along-wind slope correlates at essentially zero, and that is not
the absence of a pattern.** The dependence is a clean symmetric V with
its minimum at zero slope: 0.031 m/s on flat ground at 40 m rising to
0.62 m/s at a slope of 1.3, and the same on both sides. Pearson r reads a
U-shape as no correlation. This is why the binned means are the primary
output of that script and r is only a summary -- read alone, r would have
said the wind direction does not matter, when what it actually shows is
that only the *magnitude* does.

**Lee and windward faces are indistinguishable**, where the along-wind
slope exceeds 0.2 in magnitude:

========  =========  ==========  ==========
level           lee    windward    lee/wind
========  =========  ==========  ==========
5 m           0.414       0.421        0.98
20 m          0.293       0.302        0.97
40 m          0.407       0.403        1.01
80 m          0.294       0.283        1.04
160 m         0.170       0.161        1.06
1184 m        0.314       0.304        1.03
========  =========  ==========  ==========

That deserves to be stated rather than passed over. A momentum solver
would separate in the lee and the two columns would differ strongly. A
mass-consistent solver has no momentum equation and cannot separate, so
orientation-independence is what its physics predicts, and the
measurement confirms it. A surrogate trained on this data is learning an
orientation-independent operator because the operator is one.

**The two lowest levels break the pattern in the established way.** Their
slope correlation is weak (0.24 and 0.13) and they carry a large error
floor on flat ground -- 0.230 and 0.121 m/s where every level above sits
at 0.011 to 0.042. That is the log-law fill: a systematic offset, not a
terrain effect. It is the third independent confirmation of the same
mechanism, after the by-height split and the negative
``log_blend_correction`` result.

**Curvature adds nothing** (``|r|`` at most 0.18 everywhere).

Two consequences for the training phase: feed the network **slope
magnitude**, not a signed or directional slope; and do not spend a
channel on curvature.

Training a surrogate on it
==========================

Phase 22b. ``fastwindterrain.training`` turns the dataset into what a
network sees, ``fastwindterrain.models`` holds three architectures behind
one signature, and ``cases/train_surrogate.py`` runs them.

The pipeline was written before any architecture, for the reason the
scoring was: a data bug does not announce itself. A terrain channel
negated along with the velocity turns every ridge into a valley and still
produces a falling loss curve, so ``tests/test_training.py`` asserts it
rather than hoping.

Four pipeline decisions, each resting on something already measured:

* **normalise by** ``u_ref``. The solve is exactly linear in it, so wind
  speed is not an input and not an axis of the dataset.
* **direction as sin/cos**, not eight classes. The operator is odd, so
  negating the encoding must negate the target. Two continuous channels
  express that; a one-hot cannot, and the network would have to
  rediscover from data a symmetry it could be handed.
* **terrain scaled by a CONSTANT**, not by its own relief. Per-sample
  scaling divides away the amplitude that decides how much the flow
  deflects, so a 50 m hill and a 1500 m ridge would arrive identical.
* **targets scaled per channel, by RMS, with no mean removed.** ``w`` is
  six times smaller than ``u`` and ``v``, and the 5 m level 2.7 times
  smaller than the top; unweighted, the loss is dominated by the aloft
  horizontal channels, which are the easiest part of the column and the
  part nobody asked for. No mean is subtracted because the dataset
  contains every sample with its exact negation, so the mean is
  identically zero -- subtracting an estimate would break the oddness.

Two runs died before the cause was found
----------------------------------------

Worth recording because the first fix was the wrong one. Run one diverged
at epoch 11. I blamed the learning rate, lowered it, added gradient
clipping and the per-channel scaling. Run two died at epoch 5 -- earlier.
Both learned to about 2.1 m/s and then collapsed to the zero solution:
loss exactly 1.0 on unit-variance targets, weights frozen, gradients an
order of magnitude below normal.

Instrumenting per step found the moment: a gradient norm of 390 against a
typical 0.3. Clipping did not prevent the death, because by then the
model was already in a dead region -- the gradient had been ten times
normal for fifty steps before the spike.

**The defect was the architecture, not the optimiser or the data.** The
Fourier blocks had no normalisation at all, so their output was
unbounded. Over the same 1400 steps, everything else held fixed:

==============  ==============  ==================  ==============
blocks              final loss     worst step loss      worst grad
==============  ==============  ==================  ==============
no norm                 1.0005               23.30           390.0
GroupNorm               0.2945                1.17             2.5
==============  ==============  ==================  ==============

It was not an outlier sample: 648 solved training samples, maximum
23.8 m/s, every one finite. Diagnosing that first would have saved a run.

First results
-------------

Sixty epochs each, published defaults, nothing tuned. Validation is
vector RMSE at the levels, in m/s; the skill columns are against the
undisturbed baseline, where 0 is no better than doing nothing:

========  ============  ==========  =============  ==========  ==========
arch        parameters     s/epoch   best val m/s     complex     extreme
========  ============  ==========  =============  ==========  ==========
unet         4,898,171         8.7          1.050       +0.49       +0.51
ufno         2,235,003        24.3          1.666       +0.19       +0.22
fno          2,105,659        23.2          2.009       -0.10       +0.05
========  ============  ==========  =============  ==========  ==========

**The convolutional baseline wins, clearly, and it is three times faster
per epoch.** That is not the expected result for an FNO paper and it is
reported as measured.

The ORDERING is more informative than the winner. Going ``fno`` to
``ufno`` adds the U-Net branch and improves things substantially; going
``ufno`` to ``unet`` removes the spectral path entirely and improves them
again. Every step that moves weight from the spectral path to local
convolution helps.

**A mechanism that fits, and is testable.** These runs truncate at 16
modes on a 100-cell grid, which keeps features coarser than about six
cells -- 300 m and up -- and discards everything below. But the error was
measured to be controlled by LOCAL slope (r = 0.50-0.72) and to live at
the surface, and 50 m SRTM terrain carries most of its power below 300 m.
On that reading the spectral path is discarding exactly the scales that
decide the answer, and the U-Net branch is what puts some of them back.

Two caveats before this becomes a claim: the U-Net carries 2.2 times the
parameters, and none of the three is tuned. The mode count and a
parameter-matched comparison are the two experiments that would settle
it.

**And none of them is close to useful yet.** The best model is 1.05 m/s
against a floor of 0.20 and a baseline of 2.07 -- about half the terrain
effect explained, where the reconstruction allows 90 %. This is a working
harness and a first number, not a result.

The whole chain: terrain in, 3D field out
=========================================

Phase 23, and the paper's second contribution. Everything before this
measured one link. ``eval_harness.py`` measured what stitching costs from
PERFECT levels; ``train_surrogate.py`` measured what the network costs AT
the levels. ``cases/end_to_end.py`` composes them:

    terrain + direction  ->  9 levels  ->  60-layer 3D field

The question is whether stitching AMPLIFIES the model's error. It is not
a rhetorical one: the floor was measured from level values the solver
produced, which are smooth and mutually consistent, while a predicted
level field is neither, and interpolating between two independently wrong
levels can be worse than either. If the errors merely combine, the
pipeline is sound and only the model needs work. ``amplification`` is the
end-to-end error over ``sqrt(floor^2 + levels^2)``: 1.0 means the two
simply combined.

Test fold, 216 samples, the U-Net of phase 22b. Vector RMSE in m/s
against the solver's own 3D field:

==========  =====  =========  ========  ========  ===========  =========  ========
group           n   baseline     floor    levels   end to end    amplif.     skill
==========  =====  =========  ========  ========  ===========  =========  ========
gentle         84      0.542     0.070     0.439        0.380      0.858    +0.241
moderate       28      0.755     0.075     0.491        0.386      0.783    +0.463
complex        64      2.069     0.197     1.124        0.949      0.826    +0.539
extreme        40      2.552     0.259     1.553        1.367      0.867    +0.463
all           216      1.394     0.143     0.855        0.732      0.841    +0.399
==========  =====  =========  ========  ========  ===========  =========  ========

**Stitching does not amplify the model's error -- it reduces it.**
Amplification is 0.84, consistently, in every relief bin and on both
folds. Two reasons, and both are properties of the geometry rather than
luck: most of the 3D column lies between and above the levels, where the
model is more accurate than it is near the surface, while the level
metric weights all nine equally; and interpolating between levels whose
errors are independent averages some of them away.

So the 2D-to-3D step is not a tax on the surrogate. That is the
contribution-2 result, and it is a stronger version of it than expected:
the composition is favourable, not merely neutral.

Unseen terrain
--------------

The demonstration sites, never in any fold, scored identically:

==========  =====  =========  ========  ========  ===========  =========  ========
group           n   baseline     floor    levels   end to end    amplif.     skill
==========  =====  =========  ========  ========  ===========  =========  ========
complex        36      1.991     0.177     1.120        0.940      0.822    +0.520
extreme        36      2.446     0.229     1.341        1.169      0.857    +0.517
all            72      2.219     0.203     1.231        1.054      0.839    +0.518
==========  =====  =========  ========  ========  ===========  =========  ========

**The transfer prediction holds.** Phase 22a measured the demo terrain
and predicted that a model which generalises should land near its
test-fold numbers there, because difficulty is set by relief and not by
familiarity. It does: 0.940 against 0.949 on complex ground, and 1.169
against 1.367 on extreme -- the unseen sites score BETTER than the
matching test bin. Skill against the baseline is +0.52 on unseen terrain
against +0.46 to +0.54 on the test fold.

That prediction was registered before the model existed, which is what
makes it worth something. A gap would have been evidence of memorisation;
its absence is evidence against.

A fix that failed: replicate padding
------------------------------------

The prediction maps show a bright frame around the domain edge, and
trimming three border cells cuts the RMSE by 4.6 %, so the artefact is
real. The obvious cause is zero padding: a zero border tells the network
the terrain drops to the mean elevation just outside the window, which is
an artificial cliff around every domain. Replicate padding should fix it.

It does, and it loses anyway:

============  =========  ==========  =========  ============
padding             all    interior     border    border/int
============  =========  ==========  =========  ============
zeros            0.7817      0.7608     0.9232         1.21x
replicate        0.7953      0.7803     0.8996         1.15x
============  =========  ==========  =========  ============

The border penalty falls from 1.21x to 1.15x -- the intended effect -- but
the interior gets worse and the net is worse. The likely reason is that
zero padding leaks absolute position into a convolutional network, which
networks are known to exploit; replicate padding removes that cue along
with the cliff.

**Reverted.** Recorded because the reasoning was sound and the result was
still negative, and because the principled fix is now obvious: supply
coordinate channels explicitly, so position is available without the
artificial border, and then replicate padding costs nothing. Untested.

How much data does this actually need?
======================================

The question the corpus was built without an answer to. Five fractions of
the training fold, with and without D4 augmentation, at a fixed 30 000
gradient steps -- steps rather than epochs, because at fixed epochs a
smaller set gets fewer updates and the comparison would measure training
amount as much as data amount. Whole windows are held in or out; splitting
the four directions of one window would overstate how much ground the
model saw. Validation error in m/s:

=========  ========  =========  =========  ========
windows      solves      no D4    with D4      gain
=========  ========  =========  =========  ========
10               40     1.1348     0.8947   -21.2 %
20               81     0.9788     0.8248   -15.7 %
40              162     0.8660     0.7709   -11.0 %
81              324     0.8181     0.7568    -7.5 %
162             648     0.8211     0.7616    -7.2 %
=========  ========  =========  =========  ========

**Doubling from 81 to 162 windows bought nothing** -- marginally negative
on both curves. 324 solves reached what 648 reached, and with D4
augmentation 162 solves came within 2 % of the full corpus. The overnight
generation run could have been ninety minutes.

**D4 augmentation is worth about four times the data**, at every scale:
20 windows augmented (0.825) beats 40 plain (0.866); 40 augmented (0.771)
beats 81 plain (0.818). The gain decays smoothly -- 21, 16, 11, 7.5,
7.2 % -- which is the signature of genuine augmentation rather than a
regulariser that happens to help. Eight solves of diagnostic bought it.

**And 7 % of it survives at the plateau.** The model never fully learns
the symmetry from data alone, even with the whole corpus. An architecture
with D4 equivariance built in rather than taught should recover that for
free, which is a motivated next step in a way the architecture sweep was
not.

One seed per point. The plateau is far larger than the +/-0.005 wobble at
the top of the curve, but each point wants three seeds before this is
quoted.

Global spectral descriptors: a negative result
----------------------------------------------

Chetco Bar's gentle cells carry 0.750 m/s against 0.205 at Flatirons at
identical LOCAL slope, so the region's overall ruggedness clearly matters
and a bounded receptive field cannot see it. Six D4-invariant scalars --
spectral slope, power in three wavelength bands, spectral anisotropy,
detrended RMS height -- were added as constant planes to supply it.
Scored on the unseen sites, where the hypothesis lives:

=========  =====  ==========  =========  =========  =========  =========  ========
spectral      D4    perdigao      gorge    cameron     chetco        ALL     ratio
=========  =====  ==========  =========  =========  =========  =========  ========
no            no      0.3835     0.4397     0.5800     0.7959     0.5498      2.08
yes           no      0.3897     0.4357     0.5318     0.7715     0.5321      1.98
no           yes      0.3448     0.3859     0.4745     0.6709     0.4690      1.95
yes          yes      0.3716     0.4109     0.5050     0.7062     0.4984      1.90
=========  =====  ==========  =========  =========  =========  =========  ========

The last column is Chetco over Perdigao, the hard-to-easy spread the
descriptors were built to narrow. It does narrow, monotonically, 2.08 to
1.98 and 1.95 to 1.90. But the mean improves 3 % without augmentation and
gets 6 % WORSE with it, and augmented is the configuration that matters.

**Verdict: no.** The reasoning was sound and the prediction was specific;
the measurement declines it. Kept and documented, because the alternative
is somebody having the same good idea again in a year. For scale, D4
augmentation improves the same unseen mean by 14.7 %.

What terrain explains, per scale and height
===========================================

``cases/coherence_study.py``. The field's cross-spectrum with terrain,
ensemble-averaged per mode over 162 windows and then binned radially --
the question linearised flow theory has answered analytically since
Jackson and Hunt (1975), and the basis of models that transform the
terrain, multiply by a transfer function and transform back.

Coherence is the fraction of wind variance at wavenumber k explained by
terrain at the SAME wavenumber:

========  ========  ========  ========  ========  ========  ========  ========
level         3536      1768      1000       632       400       253       141
========  ========  ========  ========  ========  ========  ========  ========
5 m          0.288     0.238     0.210     0.264     0.308     0.297     0.187
40 m         0.452     0.387     0.307     0.291     0.242     0.168     0.067
160 m        0.618     0.660     0.667     0.655     0.570     0.503     0.184
308 m        0.668     0.773     0.815     0.842     0.814     0.806     0.539
592 m        0.664     0.748     0.777     0.854     0.860     0.854     0.734
========  ========  ========  ========  ========  ========  ========  ========

Columns are terrain wavelength in metres. Direction 45 degrees agrees with
direction 0 throughout.

**Terrain explains about a quarter of the near-surface wind variance and
about 85 % of it aloft.**

THE CONTROL RULES OUT THE OBVIOUS OBJECTION. The same machinery, the same
levels, the same samples give ``w`` a coherence of 0.61-0.89 at 5 m with
an admittance slope of +0.94 against the kinematic theory's +1. So
windowing and non-periodicity are not suppressing coherence: near the
surface ``w`` is a clean diagonal linear function of terrain and ``u`` is
not.

That control also caught a real error. The first implementation formed the
coherence AFTER radially averaging the cross-spectrum and returned 0.000
with a slope of +0.13. The kinematic response flips phase with the sign of
u.k, so a radial bin sums modes of opposite phase and they cancel.
Coherence must be ensemble-averaged per mode and binned afterwards.

Influence decays with height as potential flow predicts in scaling, but
deeper:

============  =========  ==================
wavelength        L/2pi     measured e-fold
============  =========  ==================
632 m               101                 240
400 m                64                 101
253 m                40                  64
141 m                23                  46
============  =========  ==================

A factor of 1.6 to 2.4 deeper than exp(-kz), which is what suppressed
vertical transmissivity should do -- the solver pushes air around
obstacles rather than over them, and the influence spreads further.

Why the spectral architecture lost
----------------------------------

An FNO's spectral layer multiplies each mode independently: it is
diagonal in wavenumber by construction. The coherence says the operator
IS diagonal and near-linear above about 300 m, and is neither below it.

So the architecture is well matched to the part of the column that is easy
and that nobody asked for, and structurally mismatched to the 5-160 m band
where the deliverable is and where the error lives. That turns the
architecture result from "a U-Net beat a U-FNO in our runs" into a
statement about the operator, measured before and independently of any
training.

It also bounds the linear-spectral-baseline idea usefully: a transfer
function would reproduce most of the field above 300 m for nothing, and
almost none of what matters below it.

Exact equivariance, for free
----------------------------

The learning curve left a loose end: D4 augmentation still bought 7 % at
the plateau, so the model never fully learns the symmetry from data even
with the whole corpus. ``models.d4_average`` closes that by construction
rather than by teaching.

Frame averaging. For a finite group, running the model on all eight
symmetries of the input, mapping each output back and averaging makes ANY
network exactly equivariant:

    f(x) = (1/|G|) sum_g  g^-1 . model(g . x)

Measured on an untrained U-Net: equivariance error 6e-08 wrapped against
0.704 bare, on a field of order one. Twelve million times better, and no
weight was touched.

It costs eight forward passes and needs no retraining, which is the point
-- it sizes the prize before anyone builds a group-equivariant
architecture to get the same thing at 1x cost. On the unseen sites, end
to end in m/s:

==================  ============  =========  =============  =========
model               trained w/ D4      plain    D4-averaged       gain
==================  ============  =========  =============  =========
lc_f10                        no     0.5498         0.4569    -16.9 %
lc_f10augmentd4              yes     0.4690         0.4215    -10.1 %
unet_conv                     no     0.4931         0.4204    -14.7 %
==================  ============  =========  =============  =========

**A model trained WITH augmentation still gains 10 %**, which settles what
the learning curve only hinted at: augmentation teaches the symmetry
approximately and never completely. The 7 % residual there and the 10 %
here are the same gap seen from two directions.

0.4215 is the best unseen-terrain result in this work, and it came from a
model that was already trained.

CHANNEL SEMANTICS ARE NOT OPTIONAL. Under a rotation the terrain and
slope planes merely move; the direction planes rotate as a vector, and so
does (u, v) at each output level, while w is a scalar. Treating a vector
as a scalar gives a field that looks right and points the wrong way, which
no loss curve would reveal -- so the wrapper is told which channels are
which rather than guessing.

The obvious next step is a group-equivariant convolution: the same
guarantee with tied weights instead of averaged outputs, at one forward
pass and roughly eight times fewer effective parameters. Untested.
