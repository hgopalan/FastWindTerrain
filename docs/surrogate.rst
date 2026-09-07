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

Rescored on the best model, and on matched support
--------------------------------------------------

The numbers above are the 60-epoch U-Net of phase 22b. The best model is
now ``dcnn w96 +FiLM`` at 0.6572 at the levels, 15 % better, so the whole
chain was rescored. Four unseen sites, 144 samples, plus the measurement
site separately:

================  =======  ========  ===========  ========
site                floor    levels   end to end      skill
================  =======  ========  ===========  ========
cameron_peak       0.168     0.658        0.362
chetco_bar         0.238     0.779        0.495
columbia_gorge     0.131     0.461        0.280
perdigao           0.120     0.430        0.250
ALL UNSEEN                                0.347     +0.801
nrel_flatirons     0.064     0.295        0.137     +0.767
TEST FOLD                                 0.313     +0.761
+ D4 averaging                            0.317     +0.818
================  =======  ========  ===========  ========

Test fold 0.732 -> 0.313 and unseen 1.054 -> 0.347 against phase 22b.
Every unseen site except Chetco Bar is now at or inside the 0.25 m/s
tolerance in the full 3D field, and frame averaging buys a further 9 %
at inference for no training.

**But the amplification metric above mixes spatial supports, and the
claim built on it was too strong.** ``levels`` is scored AT the nine
levels, which sit where the error is largest; ``floor`` and ``end to
end`` are scored over the whole fluid volume, most of which lies above
160 m where the model is accurate. Part of the sub-1.0 ratio is
therefore structural. On this model the mixed-support figure is 0.569,
which would read as a 43 % reduction.

``scratchpad/bandamp.py`` rescores band by band, between consecutive
levels, so every quantity describes the same cells. ``L_b`` is the rms
model error at the two levels bracketing the band. The null matters:
linear interpolation between endpoints whose errors have correlation
``rho`` gives ``var = sigma^2 (2/3 + rho/3)`` averaged across the band,
so ``amp = sqrt(2/3 + rho/3)`` on arithmetic alone -- 0.82 for
independent endpoints, 1.00 for perfectly correlated ones. That
predicted value is the column ``pred``:

===========  ==========  =======  =======  =======  ======  ======  ======
band (AGL)        cells    floor      L_b      e2e     amp    pred     rho
===========  ==========  =======  =======  =======  ======  ======  ======
0-5 m           784,676   0.5471   0.9479   0.8743   0.799      --      --
5-10 m          781,848   0.1855   0.9192   0.8186   0.873   0.977   0.862
10-20 m       1,525,152   0.2657   0.8040   0.6821   0.806   0.954   0.733
20-40 m       2,710,348   0.3066   0.6034   0.5431   0.802   0.878   0.315
40-80 m       4,849,776   0.2250   0.4017   0.3506   0.762   0.843   0.132
80-160 m      7,851,316   0.1057   0.2846   0.2474   0.815   0.917   0.524
160-308 m    12,439,744   0.0556   0.2459   0.2277   0.903   0.921   0.545
308-594 m    16,946,152   0.0716   0.2270   0.2262   0.950   0.966   0.799
594-1144 m   17,448,692   0.1391   0.2416   0.2453   0.880   0.931   0.599
===========  ==========  =======  =======  =======  ======  ======  ======

**On matched support amplification is 0.76-0.95, not 0.569.** The
mixed-support metric was inflating the effect roughly twofold, and "a
43 % reduction" does not survive. What does survive is better founded:
**measured amplification is below the arithmetic null in all eight bands
that have one**, by 0.02 to 0.10. Interpolation between correlated
endpoints would already reduce the error; stitching consistently beats
what that alone predicts.

So the contribution-2 sentence is that composition is favourable at
every height -- amplification 0.76-0.95 against a null of 0.84-0.98 --
rather than that stitching cuts the error by two fifths. Support-
consistent, and it survives a reader who knows what linear interpolation
does to variance.

Two things the band view exposes that the aggregate hid.

**The 0-5 m band has a floor of 0.5471**, three times any other band,
because it is EXTRAPOLATED below the lowest level rather than
interpolated. That is the near-surface extrapolation result reappearing
in the stitching metric, and it is the one place where a level below 5 m
still has a live argument -- narrower than the case rejected earlier,
since it concerns the floor in one band rather than the model's error
overall.

**Endpoint error correlation collapses with height** -- 0.862 at 5-10 m,
0.315 at 20-40 m, 0.132 at 40-80 m -- then rises again aloft. Where the
bracketing errors are correlated, interpolation cannot average them away
and the null sits near 1.0; where they are independent it can. That is
why the mid-column bands show the largest gains, and it is the same
correlation structure that made the log-law residual useless as an error
indicator.

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

    f(x) = (1/N) sum over the eight g of  g^-1 . model(g . x)

Measured on an untrained U-Net: equivariance error 6e-08 wrapped against
0.704 bare, on a field of order one. Twelve million times better, and no
weight was touched.

It costs eight forward passes and needs no retraining, which is the point
-- it sizes the prize before anyone builds a group-equivariant
architecture to get the same thing at 1x cost. On the unseen sites, end
to end in m/s:

==================  ============  =========  =============  =========
model               trained w/D4      plain    D4-averaged       gain
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

A prediction that failed
========================

The coherence study measured terrain explaining about a quarter of the
near-surface wind variance and about 85 % of it aloft, and from that I
drew a mechanism: an FNO's spectral layer is diagonal in wavenumber, so a
basis that is GLOBAL in space is matched to the easy part of the column
and mismatched to the 5-160 m band where the deliverable lives.

That mechanism made a specific, falsifiable prediction. A wavelet
neural operator uses a basis localised in space AND scale, so if the
diagnosis were right, WNO's advantage over FNO should appear **below
160 m** and largely vanish aloft.

It was recorded before the runs and it is wrong. Per level on the 180
unseen windows, vector RMSE in m/s:

=========  =========  =========  =========  =========  ==============
height         U-FNO        WNO      U-Net      G-CNN    WNO vs U-FNO
=========  =========  =========  =========  =========  ==============
5 m           1.3362     1.2247     1.0378     0.9934          -8.3 %
10 m          1.3711     1.2160     0.9705     0.9270         -11.3 %
20 m          1.3463     1.1212     0.8155     0.7408         -16.7 %
40 m          1.2747     0.9409     0.5803     0.4961         -26.2 %
80 m          1.2929     0.8698     0.4316     0.3129         -32.7 %
160 m         1.3212     0.8559     0.4139     0.2732         -35.2 %
355 m         1.2895     0.8356     0.4113     0.2551         -35.2 %
787 m         1.0957     0.7619     0.4049     0.2350         -30.5 %
1744 m        1.0689     0.8573     0.5202     0.3021         -19.8 %
column        1.2704     0.9789     0.6652     0.5798         -22.9 %
=========  =========  =========  =========  =========  ==============

**WNO's gain over U-FNO is smallest at the surface and largest aloft** --
8 % at 5 m against 35 % at 160 m. The prediction was not merely
unsupported; the effect runs the other way.

What survives and what does not
-------------------------------

The coherence measurement stands. It is a property of the operator,
measured with a control that reproduces a known analytic result -- w
coherent with terrain at 0.61-0.89 near the surface with an admittance
slope of +0.94 against the kinematic +1.

The INFERENCE drawn from it does not. "Spectral models fail near the
surface because their basis is global" predicted an outcome that did not
occur, and no amount of restating the coherence numbers repairs that.

A better reading, and one the same data supports
------------------------------------------------

The problem is not global-versus-local basis. It is that a fixed linear
transform followed by pointwise multiplication -- Fourier or wavelet --
is the wrong operator class for this map, and the choice of transform is
second order.

* a localised basis genuinely helps: WNO beats U-FNO by 23 % overall;
* it does not close the gap: WNO still loses to a plain U-Net by 47 %
  and to the group-equivariant CNN by 69 %;
* both spectral variants are nearly FLAT with height, 1.07-1.37 for
  U-FNO and 0.76-1.22 for WNO, while both convolutional models improve
  three- to fourfold from the surface upward.

That last row is the strongest form of it. The spectral models are not
losing only where the coherence is low -- they fail to exploit the part
of the column where the operator is nearly linear and diagonal, which is
the regime their inductive bias is supposed to suit.

This is recorded rather than quietly dropped because the prediction was
specific and registered in advance, and a negative result on one's own
mechanism is worth more than an unfalsifiable story that happens to sit
beside the right answer.

Would a level below 5 m help? Not for the reason expected
=========================================================

The near-surface band is the only one outside tolerance, and every
architecture lands within 4.5 % of every other there, so it is not an
architecture problem. ``--by-height`` had traced the FLOOR's near-surface
error to extrapolation: below the lowest level the field is filled from a
log law carrying 0.467 m/s against 0.112 for anything interpolated. The
obvious fix is a level underneath, so the bottom cells are interpolated.

That would cost a regenerated corpus, because levels are extracted at
generation time. So the ceiling was measured first, from 3D fields
already stored -- no solving, no training. If the floor does not improve,
nothing built on the new level set can.

``cases/low_level_study.py``, 60 test samples, m/s:

============  ==============  ==============  ===============
band (AGL)        9 (corpus)     10 (+2.5 m)    10 (+1 aloft)
============  ==============  ==============  ===============
0-10 m                0.3426          0.2989           0.3426
10-50 m               0.1794          0.1794           0.1794
50-160 m              0.1302          0.1302           0.1302
160+ m                0.0938          0.0938           0.0579
column                0.1479          0.1448           0.1266
============  ==============  ==============  ===============

**The 2.5 m level does what it was designed to do** -- 12.7 % off the
0-10 m floor. **And the control kills it anyway.** Spending the same
tenth level ALOFT improves the column floor by 14.4 %, seven times more,
while doing nothing at the surface. Adding any tenth level adds capacity;
a gain at 2.5 m only means something if the same level spent elsewhere
does not buy more, and it does.

The number that actually decides it
-----------------------------------

Neither of those. The floor is not what the models are up against:

============  ============  ============  ==========
band (AGL)           floor    best model       ratio
============  ============  ============  ==========
0-10 m              0.3426        0.9934        2.9x
160+ m              0.0938        0.2350        2.5x
============  ============  ============  ==========

**Every model sits two and a half to three times above the reconstruction
floor, at every height.** Lowering a ceiling from 0.34 to 0.30 buys
nothing when the model is at 0.99. The level set is not the binding
constraint anywhere -- the model is.

A correction, and what it changes
---------------------------------

This experiment had been ranked above the architecture work, on the
strength of the earlier finding that the near-surface FLOOR error is
extrapolation rather than resolution. That finding stands. The inference
drawn from it -- that the model's near-surface error was therefore about
the level set -- does not. They are different quantities and were
conflated.

What it leaves is a sharper target. There is a factor of about three
available at every height before level placement matters at all, and the
error fields of architectures as different as a group-equivariant CNN and
a plain U-Net are correlated at 0.82, so roughly seventy per cent of that
error is shared. It will not come from another architecture either.

That points at the inputs rather than the model or the level set: the
network is given terrain, slope and direction, and if the residual is
systematic across every architecture and well clear of the
representational floor, the most likely explanation is that the input
does not determine the answer. The slope ablation and the larger-context
run test exactly that.

Recorded because it cost nothing and killed a planned corpus
regeneration -- which is the whole argument for measuring ceilings before
paying for experiments.

The architecture table, and what a single seed can carry
========================================================

Every architecture at 30 000 gradient steps on the full corpus, so the
rows compare. Validation is vector RMSE at the levels, in m/s:

===================  =========  =======  =======
model                   params  val m/s  s/epoch
===================  =========  =======  =======
fno (60 ep)          2,235,003   2.0093       --
ufno (400 ep)        2,235,003   1.3551       --
wno w48                     --   1.1543       --
wno w64              2,945,627   1.0804       --
unet w32             4,898,171   0.8211       --
unet w32 + D4 aug    4,898,171   0.7616       --
gcnn w12             1,649,751   0.7148       --
gcnn w20             4,581,247   0.6791      118
gcnn w26             7,741,501   0.6755      187
dcnn w96 --no-slope  1,021,947   0.6691       36
dcnn w96             1,021,947   0.6626       36
dcnn w96 + FiLM      1,060,539   0.6572       41
===================  =========  =======  =======

**The dilated CNN leads at a fifth of the group-equivariant model's
parameters and a fifth of its epoch time.** No symmetry machinery, no
group convolutions: full resolution and a 6.5 km receptive field. That is
an awkward result for a paper whose strongest single measurement is the
solver's exact D4 equivariance, and it is the right one. Built-in
equivariance produced the best UNDERSTANDING -- an exactly verified
symmetry group, and the demonstration that augmentation never fully
teaches it -- while a simpler model that preserves resolution and sees
far is the practical recommendation.

**The G-CNN saturates rather than stops.** 4.58M to 7.74M parameters buys
0.5 %, against the 5.0 % that 1.65M to 4.58M bought.

**FiLM is worth 0.8 %**, and the slope channel 1.0 %. The latter is the
more interesting number: slope was supplied in all 31 runs because the
correlation study found it predicting the error, but a 3x3 convolution
takes a finite difference of the terrain in its first layer, so the
network was already computing it. Cheap enough to keep; not load-bearing.

Per level and per site, at 80 m -- hub height, and a level the model
predicts directly, so no reconstruction is involved:

===============  =======  ======  ======  =========  ========
model            cameron  chetco   gorge  flatirons  perdigao
===============  =======  ======  ======  =========  ========
fno               2.2849  3.0100  1.6735     0.7157    1.7321
ufno              1.5865  1.8925  1.1064     0.4634    0.9055
wno w64           0.9982  1.3036  0.7328     0.3177    0.6702
unet w32          0.5848  0.7288  0.4294     0.1622    0.3363
unet w32 + D4     0.4533  0.5994  0.3516     0.1411    0.2928
gcnn w20          0.3564  0.4909  0.2600     0.0847    0.2165
dcnn w96          0.3764  0.4827  0.2557     0.0908    0.2225
dcnn w96 + FiLM   0.3682  0.4773  0.2524     0.0884    0.2125
===============  =======  ======  ======  =========  ========

Site ordering is identical for every model including the ones five times
worse overall, so it is the terrain's difficulty showing through and not
an architecture effect.

Every number here is ONE SEED
------------------------------

The top four models span 2.8 % and FiLM is 0.8 %. Neither gap is
defensible at n = 1, and this is the cheapest remaining threat to any
claim in this document.

A free check first, since the checkpoints already exist. If A genuinely
beats B it should win most of the 45 independent cells of five unseen
sites by nine levels, not merely the aggregate, which one site could
carry:

======================================  ==========  =============
pair                                     cells won        verdict
======================================  ==========  =============
dcnn w96 vs gcnn w20                        35/45     consistent
dcnn w96 + FiLM vs dcnn w96                 42/45     consistent
dcnn w96 vs dcnn w96 --no-slope             43/45     consistent
gcnn w26 vs gcnn w20                        38/45     consistent
======================================  ==========  =============

FiLM winning 42 of 45 is stronger evidence than its 0.8 % aggregate
suggests, and gcnn w26 winning 38 of 45 says the G-CNN is saturating
rather than finished.

**WHAT THIS TEST CANNOT DO, and it is the thing that matters.** The 45
cells are not independent -- levels within a site correlate at up to
0.897 -- so the effective sample is nearer 15 than 45. Worse, a lucky
initialisation produces a model that is better EVERYWHERE, winning most
cells while saying nothing about reproducibility. The sign test
establishes that a ranking is consistent across terrain and height. That
is a different and weaker claim than being outside seed noise.

The seed work queued is deliberately not three seeds of everything.
Two extra seeds of ``dcnn w96`` give one sigma at the top of the table,
which applied to every row says which gaps are real -- assuming sigma is
comparable across models trained identically, which is stated rather
than assumed silently. And two extra seeds each at ``frac 0.5`` and
``frac 1.0`` with augmentation defend the plateau directly, because that
claim rests on 0.7568 against 0.7616, a gap of 0.6 %, and it is the
spine of the paper. U-Net at 30 000 steps is about 35 minutes, so the
claim that matters most is also the cheapest to protect.

A prediction, registered before the runs
-----------------------------------------

``--surface-weight`` tests whether the surface layer is capacity-starved
rather than information-starved. **The expectation is a small gain at
best, under 5 % at 5 m for W = 4**, for three reasons: the dilated CNN
and the group-equivariant CNN make the same surface error at r = 0.945
despite a 4.5x difference in capacity; ``channel_rms`` already
normalises every channel to unit rms, so an equal-weighted loss is
already balanced and this is deliberate over-weighting rather than a
correction; and a reweighting redistributes effort without adding
information, while nothing computable from the terrain predicts the
error mode's sign.

More than about 15 % at 5 m would contradict the cross-architecture
result and be worth chasing. The risk to watch is the aloft levels,
which sit at 0.32 and 0.26 m/s in vector terms against a 0.25 tolerance
and have no headroom to give.

The error has one vertical shape, and it is the surface layer
=============================================================

The solver distrusts its own first fluid cell: ``Source/Surface.H``
rebuilds its speed from the friction velocity of the SECOND cell, on the
grounds that the first is the one being corrected and below it the next
known value is the roughness length, not a cell. The obvious question is
whether the surrogate should do the same to its lowest level.

Composing the wall function's two equations cancels u* and leaves a fixed
ratio between two heights, ``c = ln((5+z0)/z0) / ln((10+z0)/z0)``, which
is 0.8519 at z0 = 0.1 m. So "set 5 m from the friction velocity implied
by 10 m" is the claim that two output channels are proportional with a
known constant, and it can be tested without training anything.

Two forms of it were tried, and both failed.

**As an overwrite** -- replace the predicted 5 m speed by ``c`` times the
predicted 10 m speed, direction untouched, exactly as the solver does --
it costs 2.7 %, on all four architectures tried. But with the TRUE 10 m
as the anchor it is 26 % better than the network's own 5 m prediction, so
the log law is a better model of that level than anything the network
learned. What is missing is the solver's trust hierarchy: its second cell
really is more reliable than its first, whereas the surrogate's 10 m
prediction is about as wrong as its 5 m one.

**As a residual** -- ``R = | |U(5)| - c |U(10)| |``, which needs no
ground truth and is therefore available on unseen terrain at inference --
it carries no information about the error at all:

==================  =============  =============
correlation with        residual          slope
==================  =============  =============
error, pointwise           +0.032         +0.344
rmse, per window           -0.603         +0.911
==================  =============  =============

Terrain slope, already free at inference, beats it on both. The residual
is not merely weak but U-shaped in the error, and negatively correlated
per window, because it tracks a real property of the flow -- distance
from log-law equilibrium, which the solver's own fields show is largest
in GENTLE terrain -- and that is anti-correlated with model difficulty.
Mean residual is 0.18 m/s against a mean error of 0.79: **the surrogate
is four times more self-consistent than it is accurate.** It satisfies
the log law between two levels that are wrong together.

Why they failed, and what the failure exposes
---------------------------------------------

An SVD of the speed error over the level axis, 104 040 columns of unseen
terrain, ``scratchpad/vertmode.py``:

=========  =========  =========  ==================
   z AGL       bias        rms     corr w/ 5 m err
=========  =========  =========  ==================
      5.0    -0.0780     0.8315              +1.000
     10.0    -0.0494     0.7926              +0.897
     20.0    -0.0330     0.6144              +0.582
     40.0    -0.0209     0.3878              +0.122
     80.0    +0.0099     0.2261              +0.031
    160.0    +0.0144     0.1784              +0.036
    326.1    +0.0076     0.1680              +0.014
    664.7    -0.0022     0.1757              +0.012
   1354.9    +0.0170     0.2196              +0.005
=========  =========  =========  ==================

**Seventy-two per cent of the error variance is one vertical mode**, 83 %
in two, with profile +0.65, +0.64, +0.40, +0.07 and zero above -- and no
sign change, so it is an amplitude error rather than a displacement.
``gcnn w20`` reproduces this at 71.2 % with the same profile, so it is a
property of the problem and not of one model.

Three things follow.

The mode is **confined to 5-20 m and decoupled above 40 m**. A correction
estimated at the surface can legitimately be carried to 10 and 20 m,
where the error correlations are 0.90 and 0.58, and says nothing about
80 m, where it is 0.03.

**Its vertical extent is the surface layer.** The log law is accurate
over roughly the lowest 10-20 % of the boundary layer; for a neutral PBL
of about 500 m that is 50-100 m, and the mode has decayed to nothing
between 40 and 80 m. The agreement is why this reads as a surface-layer
modelling error rather than as an artefact of the level spacing.

And it explains the residual's failure directly: R is a difference
between two levels whose errors correlate at 0.897, so the differencing
cancels most of the signal. The indicator was blind by construction.

Where the remaining error actually is
-------------------------------------

**Which metric, and it matters here.** The mode analysis above is on
horizontal SPEED error, the quantity a magnitude correction can act on.
Everything else in this document reports VECTOR RMSE over three
components, which is larger because it also carries w and direction.
The two must not be mixed, and the tolerance applies to the vector one:

============  ==========  ==========
z AGL              speed      vector
============  ==========  ==========
5.0               0.8315      0.9501
10.0              0.7926      0.8933
20.0              0.6144      0.7146
40.0              0.3878      0.4860
80.0              0.2261      0.3155
160.0             0.1784      0.2646
============  ==========  ==========

So the aloft levels are NOT finished: 80 m is 0.316 and 160 m 0.265
against a 0.25 m/s tolerance, 6-26 % over, even though their speed error
is comfortably inside it. Most of the remaining budget is in the bottom
40 m, and that is the only part with a known one-mode structure, but the
column above is not solved.

The same distinction bounds the prize. Removing mode 1 exactly takes
SPEED error at 5 m from 0.832 to 0.280. A magnitude correction leaves w
and direction untouched, so in vector terms 5 m would go from 0.950 to
about 0.538 -- **43 %, not the 66 % the speed numbers alone suggest.**
Still the largest single gain available anywhere in this study, and
larger than the whole architecture sweep, but the honest figure is 43 %.

It also makes the task smaller than "predict the wind better": the shape
is known, so what is wanted is a single scalar amplitude per column.
Slope predicts its MAGNITUDE -- 2.4x across slope quintiles -- but
nothing tested predicts its SIGN, including the direction-aware
along-wind gradient, whose own effect on the true field is symmetric
because a mass-consistent solver has no separation physics to break
lee from windward. So an uncertainty map is buildable today and a
correction is not, in this solver.

The cheapest test of whether the surface layer is capacity-starved rather
than information-starved is ``--surface-weight``, which weights the
lowest three levels in the loss. An equal weighting spends most of the
network on six levels with nothing left to win. If concentrating it moves
the surface layer, that is capacity; if it does not, the inputs do not
determine the surface field, which is where the slope ablation and the
larger-context run were already pointing.

Where this sits: two papers with larger datasets
================================================

Two recent papers cover unseen complex terrain with neural operators, and
a reader will hold both up against this work. Their numbers are recorded
here so the comparison is made deliberately rather than reconstructed at
submission time.

Zhang et al., *Transformer-based Neural Operators for 3D Wind Field
Prediction over Complex Mountainous Terrain* (arXiv 2605.25679,
Communications Physics). Ground truth is OpenFOAM RANS on SRTM terrain at
30 m, with a reference velocity of 10 m/s at 10 m -- the same convention
used here. They report MSE, relative L2 and MAE for speed magnitude; the
RMSE column below is the square root of their MSE, not a number they
print. Test set:

============  ======  ======  ========  ======
model            MSE    RMSE    L2 (%)     MAE
============  ======  ======  ========  ======
Patch-solver   1.032   1.016     8.290   0.635
AeroGTO        1.191   1.091     8.912   0.676
Transolver     1.264   1.124     9.257   0.692
Geo-FNO        3.798   1.949    16.252   1.267
============  ======  ======  ========  ======

Zero-shot on four held-out mountain sites, their best model:

===========  ======  ======  ========
site            MSE    RMSE    L2 (%)
===========  ======  ======  ========
Chatou-1      0.828   0.910     7.408
Chatou-2      0.662   0.814     6.444
Daguping      1.515   1.231    11.391
Hengdong      0.238   0.488     3.776
===========  ======  ======  ========

Lian et al., *Reconstructing fine-scale 3D wind fields with
terrain-informed machine learning* (Nature Communications 2026).
12 000-plus RANS simulations over 300 000 CPU-hours, 12 x 12 km domains
at 30 m with a 9 x 9 km output region, 27 levels to 214 m, trained in
southeastern China and validated against three European tall towers
(OPE, Torfhaus, Ispra) at roughly 10, 50 and 120 m. **They report MAE
and relative L2 graphically -- parity plots and bar charts -- and no
numeric error value appears in the accessible main text.** Nothing is
quoted from them here for that reason.

The dataset sizes, corrected
----------------------------

Zhang's corpus is smaller than its framing suggests, and the correction
matters because the obvious argument is built on it. Their introduction
says "500 complex terrain regions"; the methods say **45 distinct terrain
geometries** at up to 16 inflow angles, giving 467 terrain-angle
combinations at about one core-hour each.

=================  ===========  ============  ==============
source                terrains       samples     core-hours
=================  ===========  ============  ==============
this corpus                252          1008           46.0
Zhang et al.                45           467            ~467
Lian et al.                  -        12 000        300 000
=================  ===========  ============  ==============

So the cost ratio is about **6 500x against Lian and only 11x against
Zhang**, and Zhang trains on *fewer* distinct terrains than the 162
windows in this training fold -- they buy coverage with 16 directions
where this corpus buys it with four plus an exact symmetry. Any claim of
"orders of magnitude less data" applies to Lian and would be wrong about
Zhang. Their per-sample cost is roughly 20x higher because RANS with
momentum is a harder solve than a mass-consistent projection, which is a
difference in physics, not in discipline.

Error against height, and the one place these agree
---------------------------------------------------

Zhang resolve error against altitude for two representative cases, and
find the same vertical structure reported above: error falls
monotonically with height, and the near-surface dominates.

===========================  ==============
quantity, complex case          relative L2
===========================  ==============
near-surface, all models          0.22-0.30
by 200 m, all models              0.03-0.06
altitude-integrated, best             0.078
altitude-integrated, simple           0.044
===========================  ==============

Their headline 8.29 % is therefore a volume average over a near-surface
layer running three to four times worse. Pixel-wise error at the 10 m
plane runs below 5.6 m/s in the simple case and below 8.00 m/s in the
complex one, against a field spanning 0 to 23 m/s; by 300 m all models
agree within 1 m/s.

**This is the useful part of the comparison.** A different group, a
different solver with momentum and separation, a different architecture
and a different continent recover the vertical structure measured in
``--by-height`` here. The near-surface band being the hard one is a
property of the problem rather than an artefact of a mass-consistent
reference, and the decomposition by height is on firmer ground for it.

Both papers decompose by height. Neither decomposes by terrain property
-- slope, curvature, relief, lee versus windward. Zhang's near-surface
figures come from two hand-picked cases labelled simple and complex, not
from binning a test set. That gap is what ``cases/slope_error.py``
fills.

Why the accuracy numbers must not be tabulated together
--------------------------------------------------------

Putting their RMSE beside this work's would be wrong three times over:

* **Different ground truth.** They emulate RANS; this emulates a
  mass-consistent solver with no momentum equation, which is a
  materially easier operator to learn. A lower error here is not
  evidence of a better method.
* **Different quantity.** Theirs is speed magnitude; the metric used
  throughout this document is vector RMSE over three components, which
  is stricter.
* **Different support.** Their 1.016 averages the full 3D volume
  including the near-surface layer; the 80 m figures here are one level.

The defensible sentence is that Zhang report 0.49-1.23 m/s speed RMSE
zero-shot on unseen RANS terrain while this work reports 0.31 m/s vector
RMSE at 80 m against a mass-consistent reference, that the targets
differ, and that the comparison being made is one of data cost and
method rather than accuracy.

What is left unclaimed
----------------------

Neither paper reports a learning curve. Neither states how much of its
data was necessary, and the question of which part of 12 000 runs
mattered is unanswered in this literature. That is the gap the
measurement above fills, and it is the one place where being smaller is
the contribution rather than the limitation.

Bigger domains by overset tiling -- a plan, not a result
========================================================

**NOTHING IN THIS SECTION IS BUILT.** It is recorded because the design
follows from measurements already made, and because the obstacle has a
number attached rather than being a guess.

Five kilometres is a demonstration. Fifty is what someone siting a wind
farm or staging a fire response actually needs, and the model cannot be
handed a bigger domain: it was trained on 5 km windows and its receptive
field is sized to them. The way out is to run it on overlapping tiles
and assemble the pieces.

The obstacle, measured
----------------------

The model is worse at the edge of its window than in the middle.
Trimming three border cells cuts RMSE by 4.6 %, so the contamination is
about **150 m deep at 50 m resolution** -- shallow, which is the good
news. Zero padding is the cause: a constant border tells the network
where the edge is, and it uses it.

**And the obvious fix failed.** Replicate padding cut the border penalty
from 1.21x to 1.15x as intended and made the INTERIOR worse, netting
0.795 against 0.782. That result is what forces the tiling design: the
edge effect cannot be trained away cheaply, so it has to be handled at
assembly instead.

The assembly
------------

Four steps, in this order:

1. **Blend at the LEVEL stage, not after stitching.** The model emits
   nine 2D levels and the vertical reconstruction is deterministic, so
   blend the 27 level fields and stitch once over the whole domain.
   Blending stitched 3D fields would average each tile's vertical
   reconstruction against its neighbour's for no benefit.

2. **Partition of unity, normalised.** With ``d_i(x)`` the distance from
   x to the edge of tile i,

   ``w_i(x) = f(d_i(x)) / sum_j f(d_j(x))``

   The normalisation makes the weights sum to exactly 1 at every point
   for ANY tile layout and any shape of ``f``. Tapers that only
   approximately sum to 1 leave faint grid artefacts that read as
   physics, which is the failure to design out rather than debug later.

3. **Inverse-variance weights**, ``f(d) = 1/sigma^2(d)``, with
   ``sigma(d)`` the measured error against distance from the edge --
   the measurement above rather than a guess. Since the penalty is
   confined to about three cells, a raised-cosine ramp to full weight by
   ten cells behaves almost identically. Blend the vector components
   ``u, v, w`` linearly, never speed and direction separately.

   TWO CAUTIONS. Inverse-variance weighting is optimal for INDEPENDENT
   estimates; neighbouring tiles see overlapping terrain through one
   model, and same-terrain error correlations elsewhere in this document
   run to 0.945, so the variance reduction will be far smaller than
   theory promises. And vector averaging SHRINKS magnitude where tiles
   disagree on direction, which would appear as slow bands along seams.
   That is the specific artefact to look for.

4. **One mass-consistent projection over the assembled domain.** This is
   what makes the result physical rather than merely smooth, and it is
   the cheapest part of the solver.

Why overset, and why the vocabulary matters
-------------------------------------------

The pieces map onto overset (Chimera) grids almost one to one: hole
cutting is the terrain mask already in use; a grid's outer FRINGE, where
it receives from a neighbour's interior DONOR cells, is exactly the
"trust the middle, not the edge" rule the border measurement produced.
Overset reached that principle for a different reason -- its outer
boundary conditions are artificial -- and it means the vocabulary for
this already exists.

What overset buys beyond a regular blend is **non-matching tiles**: a
fine tile over a complex ridge inside a coarse background, tiles rotated
onto a valley axis, coverage following a valley network instead of a
square. Partition-of-unity blending on a regular grid is the degenerate
case of it.

One caution of its own: classical overset interpolation is one-way and
sharp -- continuous in value, not in derivative -- so the blended
variant is the one to use.

Conservation across tiles: two routes
--------------------------------------

Conventional overset interpolation is NON-CONSERVATIVE, which is the
whole reason conservative overset schemes exist. There are two ways out
here and they are not equivalent.

**Repair it.** The projection in step 4 enforces mass consistency
globally after assembly, so a conservative interpolation scheme is not
strictly needed. Owning the solver that generated the training data is
what makes that available -- a group without one would have to impose
interface continuity in the loss, as cPINNs do. The cost is that the
projection is then doing two jobs at once: repairing the ML tiles'
disagreement AND repairing the interpolation.

**Or avoid it.** Devlin, Chandar and Quinlan, *Computers and Fluids* 267
(2023) 106072, give a conservative overset scheme -- DFVF-overset,
Direct Flux via Virtual Faces -- that needs no interpolation at all.
Fluxes pass between overlapping cells through virtual faces of
rigorously defined area, derived by generalising the finite volume
method to overlapping control volumes. No donors, no acceptors, no
overset assembly, no external connectivity library, and conservation
exact by construction rather than patched afterwards. Their multiphase
cases make the difference concrete: conventional overset LOSES liquid
mass across grid boundaries where DFVF conserves it strictly, and their
pressure fields are smooth where interpolation shows a discontinuity as
fluid crosses grids.

**And they arrive at a partition of unity too, independently.** Cell
volume is distributed by a top-hat weight function normalised to form a
partition of unity over the domain, and the intercell areas emerge from
that. The construction proposed above uses the same object to blend
FIELD VALUES; theirs uses it to distribute CONTROL VOLUME. Same
mathematics, different role, which means the two compose -- one weight
function could serve both the blend and the discretisation.

TWO REASONS NOT TO REACH FOR IT YET. DFVF is a solver discretisation,
not a merging rule: it governs how the finite-volume operator is
assembled across overlapping cells and says nothing about how to combine
two disagreeing ML predictions, which remains the blend's job. And for
same-resolution, axis-aligned tiles on a regular lattice, assembling
into one contiguous grid is trivial and the projection is then an
ordinary single-grid MLMG solve -- DFVF earns its keep only for
NON-MATCHING grids.

The stack matters as well. DFVF is implemented as a foam-extend
preprocessor; this solver is AMReX with a nodal MLMG Poisson solve, and
**AMReX's native answer to variable resolution is AMR with refluxing,
already conservative and already in the library.** So the nearer path to
multi-resolution tiling here is AMR, and DFVF is the right tool
specifically for arbitrarily overlapping or rotated grids that nested
refinement cannot express.

Names, so this is legible to a reader from either field: partition of
unity (Babuska and Melenk), overlapping domain decomposition of the
Schwarz family, the overlap-tile strategy of the U-Net paper,
sliding-window inference with Gaussian weighting, feathering and
multi-band blending from image stitching, FBPINNs and POU-PINNs for the
partition-of-unity-over-subdomains construction in a learned setting,
inverse-variance weighting or BLUE for the weights, DFVF-overset for
interpolation-free conservative overlapping, and
Helmholtz-Hodge -- here the mass-consistent adjustment of Sherman
(1978) -- for the projection. Every ingredient is standard; the assembly
is the contribution, and the part that is unusual is repairing the seams
with the same cheap solver that generated the training data.

The validation, and it needs one solve per site
------------------------------------------------

The test geometry already exists. **Each demonstration site is a 3 x 3
grid of 5 km windows -- 15 x 15 km.** So: solve one 15 x 15 km domain
directly as the reference, predict the nine tiles, blend, stitch,
project, and compare on two numbers -- error in the tile interiors,
which should match the single-tile result, and error in a narrow band
along the seams, which should not be visibly worse.

That is one reference solve per site plus a blending routine, and it is
the smallest honest test of whether 50 km is a wrapper or a research
problem. Until it is run, nothing here should be quoted as a capability.
