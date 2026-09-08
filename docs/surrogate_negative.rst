==================================
Surrogate: what did not work
==================================

Every construction tried against the surrogate's error that failed, with
the measurement that killed it. Kept in full rather than summarised: a
negative result is only useful if the number and the reason survive with
it, and several of these were re-derived twice because the first reading
was wrong.

The summary table and the thread running through them are in
:doc:`surrogate`.

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


A fix that failed: replicate padding
====================================

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


Global spectral descriptors: a negative result
==============================================

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


A prediction, registered before the runs
=========================================

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

The outcome: 0.9 %, and it costs the rest of the column
--------------------------------------------------------

W = 4 landed at 0.9 % at 5 m, inside the predicted bound and far inside
the 15 % that would have overturned the cross-architecture result. Per
level on the unseen sites, vector RMSE in m/s:

=========  ==========  ==========  ==========
z AGL        dcnn w96      W = 4       W = 8
=========  ==========  ==========  ==========
5.0            0.9501     -0.9 %      -1.3 %
10.0           0.8933     -1.1 %      -1.5 %
20.0           0.7146     -1.5 %      -2.0 %
40.0           0.4860     +1.3 %      +2.5 %
80.0           0.3155     +5.5 %      +9.4 %
160.0          0.2646    +10.0 %     +16.9 %
593.9          0.2276    +24.7 %     +39.8 %
1144.2         0.2713    +26.3 %     +47.8 %
=========  ==========  ==========  ==========

**Surface band 5-40 m: -0.9 % at W = 4 and -1.1 % at W = 8. Aloft:
+15.7 % and +27.4 %.** Doubling the weight bought four tenths of a
percentage point at the surface and cost nearly twelve aloft, so the
surface gain is saturating near one or two per cent while the damage is
not. Five levels are over tolerance at W = 8 against three at W = 1.
The trade is a curve, not a point, and extrapolating it says that no
weight recovers the surface layer.

The friction velocity: a correlation worth nothing
===================================================

If the surface layer is information-limited, the next question is which
information. The manifest records the solver's own per-case diagnostics,
none of which the network receives, so the hypothesis costs nothing to
test. Correlated against the SIGNED surface error over 180 unseen cases,
one of them stands out:

================  ===========  ========
diagnostic        r magnitude  r signed
================  ===========  ========
solid fraction         +0.894    +0.245
O'Brien residual       +0.862    -0.289
div L2                 +0.845    -0.065
flux imbalance         +0.568    -0.024
max u*                 -0.266    -0.682
================  ===========  ========

**The maximum friction velocity predicts the SIGN at -0.68**, where
slope, relief, elevation, curvature, the along-wind gradient and the
log-law residual had all returned essentially zero. It survives a
within-site control -- four of the five sites give -0.54 to -0.73, and
pooling after removing each site's own mean leaves -0.50 -- so it is not
an artefact of one hard site having both a high u* and a large error.
u* is exactly the quantity the wall function uses to set the first fluid
cell, and it is an output of the solve rather than a function of the
terrain the network is given.

**Supplying it as an input channel changed nothing**: 0.6655 against
0.6626, marginally worse. The reason is arithmetic and should have been
checked first:

=================================  ==========
surface 5-20 m speed error            m/s
=================================  ==========
total rms                              0.7588
case-mean bias rms                     0.0870
=================================  ==========

The per-case mean is **1.3 % of the squared error**. Removing it
perfectly would cut the root-mean-square by 0.7 %. A constant input
plane can only shift a whole field, and u* correlates with a per-case
MEAN, so the most it could ever have bought was under one per cent --
whatever the correlation.

RECORDED AS A METHOD FAILURE AS MUCH AS A RESULT. The ceiling was
computable in two minutes from fields already on disk, exactly as in the
level-below-5 m study, and it was not computed because the correlation
looked convincing. A correlation identifies a relationship; it says
nothing about the share of the error that relationship governs. The same
argument kills the deployable version before it is run: the frontal and
plan area indices are also domain scalars and inherit the same 0.7 %
ceiling, however good a drag proxy they are. That run was queued,
started, and cancelled on this reasoning rather than on its result.

What survives is the diagnosis, not the remedy. The surface error is
dominated by spatial structure within each case, not by a per-case
offset, so any correction must vary in space. A local drag or
frontal-area map computed in a moving window would qualify; a single
number per case cannot. Two further levels are
pushed over the 0.25 m/s tolerance, five in place of three. Quadrupling
the weight on the lowest three levels bought one per cent there and
damaged everything else, which is not what capacity starvation looks
like: had the network been spending its capacity aloft, redirecting four
times the weight would have moved the surface substantially.

WHAT THIS CLOSES. Two independent lines of evidence now say the same
thing. The error is the same field across architectures 4.5x apart in
capacity, at r = 0.945; and reweighting the loss toward it changes it by
one per cent. **The surface layer is information-limited, not
capacity-limited** -- terrain, slope and direction do not determine the
near-surface field, which is consistent with the sign of the error mode
being unpredictable from every terrain feature tested, the direction-aware
one included.

That is a more useful conclusion than a gain would have been. It converts
"the near surface is hard" into "the near surface is not the model's
fault", which is what makes a CFD delta the right next step rather than
another architecture. The prediction and its outcome are recorded
together because the prediction was specific, registered in advance, and
correct -- which is the only circumstance in which a null result carries
weight.


A column model below 20 m: the reference has no 1D physics to recover
=====================================================================

The residual error is one vertical shape confined to the lowest 20 m, so
the obvious remedy is a vertical overset: predict above 20 m with the
network, and obtain 5 and 10 m from a one-dimensional surface-layer
model anchored on the 20 m value. A 1D solver for exactly this is
available in ``hgopalan/onedterrainsolver``.

It does not work here, and the reason belongs to the reference rather
than to the column model.

**The deciding test needs no network.** Take the reference field's own
value at 20 m and obtain the reference at 5 and 10 m from it by the log
law. Horizontal components, unseen terrain, m/s:

=================================  ========  ========
source of the 5 and 10 m values       5 m      10 m
=================================  ========  ========
log law from the exact 20 m          0.967     0.719
the network                          0.934     0.874
=================================  ========  ========

At 10 m the log law wins. At 5 m it loses, despite having been handed
the exact value above it. The reference is not logarithmic across this
band.

Why: the wall function acts on the first fluid cell alone, whose centre
sits at 2 m with ``dz0 = 4 m`` and is therefore below the lowest
reported level. Everything between 5 and 20 m is set by the projection,
which enforces mass conservation over the whole domain. That band is the
output of a global constraint, not of a local balance between the
surface and the flow above it.

**The result is not specific to the log law.** Every neutral column
model supplies a factor multiplying the horizontal wind and they differ
only in how it is obtained, so bound them by how much the factor is
allowed to vary, fitting each against the truth. An oracle fit cannot be
beaten by a closure that has to derive the factor:

=========================  ========  ========
factor fitted against the     5 m      10 m
truth, m/s
=========================  ========  ========
network (no factor)          0.934     0.874
one factor everywhere        0.925     0.702
factor linear in slope       0.924     0.693
factor per column            0.485     0.370
field RMS                    5.601     6.678
=========================  ========  ========

The best single factor is 0.705 at 5 m against the log law's 0.741, so
the log law is already close to the best uniform choice. The gap to the
per-column oracle is large. That factor has a standard deviation of
0.119 and correlates with the local terrain slope at **-0.011**, which
I first read as "not obtainable from the terrain" -- see the correction
in the next section, which is what a slope correlation of zero does and
does not establish.

**And all of the above assumes a perfect anchor.** With the network's own
20 m value, which is what the arrangement would actually have, the
overset is 13.8 % worse than the baseline across 5-10 m and 8.7 % worse
over all nine levels. The anchor's error is carried downwards and
replaces predictions that were better.

Where the idea is right
-----------------------

Against a RANS or LES reference the 5-20 m layer is a surface layer in
the ordinary sense, its profile is set locally, and a column model is the
correct instrument. What kills it here is the mass-consistent operator,
not the concept. The first table above is the test that decides which
case applies: it uses the reference alone, needs no trained model, and
costs one solve. That makes it a cheap addition to the list of things
worth measuring before committing to an expensive reference.

Scripts: ``scratchpad/oneD_overset.py`` (the log-law variants and the
deployable configuration) and ``scratchpad/oneD_ceiling.py`` (the
closure-agnostic bound).

One correction worth recording: the first run of ``oneD_overset.py``
labelled a column "5-20 m" when the stored 20 m level is 20.000...04 and
fell outside a ``<= 20.0`` cut. The band is 5-10 m, which is the right
set anyway, since those are the levels a column model would supply.

Learning the factor instead: a correction, and a ceiling already reached
========================================================================

The section above measured what a *closure* can supply below 20 m. The
next question is whether the factor could be learned during training
instead, to account for the 20 m error propagating downwards.

**First, a correction to the section above.** A correlation of -0.011
between the per-column factor and the local terrain slope is a
correlation with *one local feature*. It does not establish that the
factor is unobtainable from the terrain, and it is not. The network's
own implied factor -- the ratio of the levels it already predicts --
correlates with the true factor at **+0.56**. The factor is
substantially obtainable from terrain. The earlier phrasing was too
strong and is fixed above.

**The premise turns out to be inverted.** Writing the horizontal wind as
a complex number and ``c = (level below) / (20 m)``, the error splits
exactly into the anchor error carried down and the factor error itself::

    P_low - Y_low = c_p (P_20 - Y_20)  +  (c_p - c_t) Y_20

=======  ==========  =============  =============
level     total       anchor term    factor term
=======  ==========  =============  =============
5 m        0.934         0.491          0.780
10 m       0.874         0.575          0.595
=======  ==========  =============  =============

At 5 m the factor term is the larger, so the low-level error is mostly a
wrong ratio rather than propagation from above. A perfect factor with
the network's own anchor would remove **47 % at 5 m** and 34 % at 10 m.

**But a change of parameterisation cannot collect it.** A least-squares
predictor with correlation ``r`` against its target is at its best when
its own standard deviation is ``r*sigma`` and it leaves ``sigma*sqrt(1 -
r^2)``:

=====================  ==========  =========================
quantity                network     MSE optimum at r = 0.56
=====================  ==========  =========================
sd of the factor          0.064              0.067
residual rms              0.098              0.098
=====================  ==========  =========================

The network sits on both. It is already extracting the factor as well as
anything trained on the same loss with the same inputs can, and
``u_5 = f * u_20`` is representable by predicting ``u_5`` directly, so
the reparameterisation adds no information. Closing the gap needs the
inputs to say more about the factor, not a different output form.

**The vertical mode is worth something, but not during training.** The
mode shape carries a correction from one level to the others. If the
error at 5 m were known exactly and regressed onto the levels above:

========  ============  =========
level      r with 5 m    change
========  ============  =========
10 m          0.90        -56 %
20 m          0.58        -19 %
40 m          0.12         -1 %
========  ============  =========

That is a real gain and it is an assimilation result, not a training
one: it needs one measured value per column, which a surrogate running
on terrain alone does not have. It matters if this is ever coupled to a
mast or a sensor network, and not before.

Script: ``scratchpad/ratio_split.py``.

Tracking the surface error: worth a great deal, and only per column
====================================================================

The previous section ended by saying the vertical mode is worth
something with a measurement rather than during training. Here is the
measurement, done properly. Give the scheme the exact error at 5 m and
correct every level above it with a complex least-squares coefficient
per level, ``corrected(z) = P(z) - beta(z) * (P(5) - Y(5))``, so beta
carries both a magnitude and a turning of the wind.

Horizontal RMSE over unseen terrain, m/s:

========  ===============  ==============  ===============
level      no correction    beta fitted     beta from the
                            on this data    validation fold
========  ===============  ==============  ===============
5 m            0.9340          0.0000           0.0000
10 m           0.8735          0.4173           0.4381
20 m           0.6871          0.5672           0.5944
40 m           0.4457          0.4403           0.4408
80 m           0.2613          0.2612           0.2613
160 m          0.1985          0.1983           0.1983
all            0.5324          0.3222           0.3307
========  ===============  ==============  ===============

**Two things stand out.** The gain is large -- 39.5 % over all levels,
or **25.4 % excluding the 5 m level itself**, which is given rather than
corrected and would otherwise flatter the result. And the coefficient
transfers: fitted on the corpus validation fold and applied unchanged to
unseen sites it gives 37.9 % against the oracle's 39.5 %, so beta is a
property of the operator and not of the terrain it was fitted on.

**And then the catch.** All of that assumes the 5 m error is known in
every column. Replace it by one number per case -- which is what a mast
actually gives -- and the whole thing collapses:

===========================================  ==========  ==========
correction                                     before      after
===========================================  ==========  ==========
5 m error known in every column                 0.5324      0.3222
5 m error known once per case (one mast)        0.5324      0.5308
===========================================  ==========  ==========

**-0.3 %.** This is the friction-velocity wall again, arriving from a
completely different direction: the surface error is spatial structure
within each case, not a per-case offset, and a single scalar cannot
touch it however exactly it is known. The u* experiment measured that
share as 1.3 % of squared error; this measures the same thing with a
perfect instrument instead of a proxy and gets the same answer.

So: dense near-surface observation would be worth a quarter of the error
above 5 m, and one mast is worth nothing. That is a statement about
instrument density, and it is the useful form of the result -- it says
what a campaign would have to look like before assimilation is worth
building.

Script: ``scratchpad/track_surface.py``.

A speed-up factor from a mast: the method is backwards for this error
======================================================================

The mast test above applied an ADDITIVE correction and was worth -0.3 %
per case. Turbine siting does not do that. It measures at a mast, forms
a speed-up factor against a reference, and applies the ratio as a
transfer function: the model supplies the SHAPE of the field and the
mast fixes its MAGNITUDE. That is a different repair -- an offset
removes a bias, a ratio removes a gain error -- and it was worth
testing, especially since this solver is exactly linear in the inflow
speed, so a gain error is a mode the operator genuinely has.

The ceiling settles it in one line. Horizontal RMSE, unseen terrain:

============================  =============  ============
variant                        all levels     5 and 10 m
============================  =============  ============
as predicted                      0.5323        0.9038
best real k (oracle)              0.5315        0.9033
best complex k (oracle)           0.5311        0.9032
mast at centre, 5 m               1.8389        1.3328
mast at centre, 10 m              1.5059        1.1910
mast at centre, 80 m              0.5819        0.9121
mast at centre, per level         0.7229        1.2507
============================  =============  ============

**The best possible single multiplier per case is worth -0.1 %.** Over
180 cases the oracle multiplier has mean 1.0003, sd 0.0027, range 0.990
to 1.008. There is no gain error to remove: the surrogate's absolute
magnitude is already right to about a quarter of a per cent per case.
No mast, however sited or however accurate, can beat that ceiling.

The mast rows are worse than doing nothing, and the reason is
instructive. The speed ratio at a single column at 5 m has mean 1.012
and **sd 0.140**, ranging 0.54 to 1.45 -- it is a 14 % noisy estimate of
a quantity whose true value is 1.000 +/- 0.003. Multiplying the whole
field by it injects far more error than it removes. Higher up the
denominator is quieter and the damage falls (80 m: +9.3 %), but it never
becomes a gain.

Why the analogy does not carry
------------------------------

WAsP works because of a division of labour: the mast supplies the
absolute level, which a linearised flow model cannot know because it has
no access to the regional wind climate, and the model supplies the ratio
between locations, which is what it is good at. Both halves invert here:

* the surrogate was trained on the same operator it predicts, so its
  absolute level is already calibrated -- there is nothing for a mast to
  contribute;
* the error that remains IS the spatial pattern, which is precisely the
  speed-up ratio that the transfer function asks the model to supply.

So the method hands over the part already correct and leans on the part
that is wrong. This is not a criticism of WAsP, whose failure mode is
the opposite one; it is a statement that the two situations need
different remedies.

The salvageable version is a transfer function that VARIES IN SPACE
rather than a scalar, and that is the per-column experiment in the
previous section: -25 % above 5 m, and it needs a sensor in every
column.

Script: ``scratchpad/wrg_transfer.py``.

Fitting the 10 m residual from local quantities
=================================================

The proposal: take u* from the 20 m wind through the log law, form the
log-law estimate at 10 m, and fit ``U10_true = f(u*, z0, dh/dx, dh/dy,
U10_loglaw, U10_ML)``.

**The feature set collapses before any fitting.** ``z0`` is constant at
0.1 m over the whole corpus and carries nothing; ``u* = kappa * U20 /
ln((20+z0)/z0)`` is U20 rescaled; and ``U10_loglaw`` is U20 rescaled
again. Three of the six inputs are one input. What remains is ``U20_ML,
U10_ML, dh/dx, dh/dy``, and every one of those is already available to
the network -- two are its own outputs and two are input channels. A
post-hoc fit could still win, but only by exploiting a local relation
that a whole-field squared-error loss has averaged away. It cannot add
information.

Added beyond the proposal, to be fair to it: the wind-aligned and
cross-wind slope, which are the rotation-covariant pair (``dh/dx`` and
``dh/dy`` alone make the fit depend on the grid axes, a defect for an
operator with exact D4 equivariance), and **neighbours** -- the same
quantities box-averaged over 3x3 and 9x9 stencils, so the fit can see
upwind and downwind ground rather than a single column. Sixteen features
in all, fitted on the corpus validation fold and evaluated on the unseen
sites. 10 m speed residual, m/s:

=========  =========  =======  ======  =======
fit        in sample  vs       unseen  vs
=========  =========  =======  ======  =======
(no fit)   0.9116              0.7114
linear     0.9099     -0.2 %   0.7135  +0.3 %
quadratic  0.9094     -0.2 %   0.7136  +0.3 %
MLP 2x128  0.8088     -11.3 %  0.8181  +15.0 %
=========  =========  =======  ======  =======

**Linear and quadratic find nothing even in sample.** That is the
decisive row: where the fit is free to see the answer, no linear or
low-order relation exists. The MLP finds 11.3 % in sample and is 15 %
worse on unseen terrain, which is memorisation.

An MLP that fails could be undertrained rather than starved, so the
identical network was trained on a target which IS a function of the
features -- ``U10_ML`` itself. It recovers it to rms 0.0105 against a
field rms of 6.32. The harness trains; the features do not contain the
residual.

An earlier run at 250 steps gave the MLP -0.5 % in sample and +0.4 %
unseen, which looked reassuringly flat. It was undertrained. Trained
properly it overfits instead. Both readings point the same way, but only
the second earns it.

One numerical detail: the linear fit puts weights of -7754, +5289 and
+2465 on ``U20_ML``, ``U10_loglaw`` and ``ustar``. Those are the three
collinear copies of one quantity and they cancel -- a degenerate
direction in the normal equations, not a signal. Every genuinely
distinct feature carries a weight below 0.04.

Script: ``scratchpad/local_fit.py``.

The slab mass budget: there is no violation to correct
=========================================================

The lateral inflow is prescribed and therefore known exactly, the
surface is impenetrable, and the reference is divergence free, so the
mass budget over a slab between two heights must close. Measure the
imbalance slab by slab and use it as a correction.

A budget over a slab is an integral: one number per slab per case. So
the ceiling for the whole family is the best multiplier per level, which
is measurable without building any flux machinery:

=================================  =========
correction                          change
=================================  =========
global k(z), fitted on validation    +0.21 %
k(z) per case (oracle)               -0.58 %
bias per case (oracle)               -0.73 %
=================================  =========

Even the oracle per-level multiplier is worth -0.58 %, and the
multiplier at 5 m is 1.007 +/- 0.015, so there is no gain error by
height either.

The budget residual itself settles it. Normalised by the lateral
throughflow, for the slab between 5 and 10 m, the REFERENCE gives 0.0500
and the prediction 0.0474; at 10-20 m, 0.0285 against 0.0275. **The
prediction's residual is smaller than the reference's.** Both are
dominated by the discretisation of the budget on nine levels. The
surrogate already closes the slab budget as well as the solver does, so
the signal the scheme would drive on does not exist.

This does not contradict the projection result, which uses the same
boundary conditions and is worth about 29 %. A projection is a global
elliptic solve with a degree of freedom in every cell; a slab budget has
one per slab. That difference is the difference between 29 % and
0.58 %.

Script: ``scratchpad/flux_levels.py``.

How many 10 m masts? None of them help
=========================================

One measurement per case, used as an offset, was worth -0.3 %. A
measurement in every column was worth -39.5 %. The counts a siting study
actually runs lie in between, so: 1, 2, 5, 10 masts at 10 m, error
interpolated by inverse-distance weighting and carried to other heights
by a coefficient beta(z) fitted on the validation fold and never on the
unseen sites. Random placement averaged over draws, with a regular grid
alongside.

=====  ==========  =======  ==========  =======  =========
masts  all levels  vs       5 and 10 m  vs       grid, all
=====  ==========  =======  ==========  =======  =========
0      0.5323      0.0 %    0.9038      0.0 %    0.5323
1      0.6961      +30.8 %  1.2385      +37.0 %  0.6900
2      0.6366      +19.6 %  1.1185      +23.8 %  0.6388
5      0.6004      +12.8 %  1.0451      +15.6 %  0.5975
10     0.5884      +10.5 %  1.0205      +12.9 %  0.5806
25     0.5743      +7.9 %   0.9916      +9.7 %   0.5742
50     0.5654      +6.2 %   0.9733      +7.7 %   0.5651
100    0.5587      +5.0 %   0.9597      +6.2 %   0.5616
=====  ==========  =======  ==========  =======  =========

**Every count is worse than no masts at all**, improving monotonically
toward the baseline without ever reaching it. Placing the masts on a
regular grid rather than at random changes nothing.

The residual is coherent in the vertical and white in the horizontal
-----------------------------------------------------------------------

The reason is one table. Lagged correlation of the 10 m error along a
row, against the same quantity for the wind field itself:

=====  =========  =========
lag    the ERROR  the FIELD
=====  =========  =========
0      1.000      1.000
50 m   0.029      0.449
100 m  0.024      0.358
250 m  0.023      0.247
500 m  0.014      0.164
=====  =========  =========

**The error decorrelates in one grid cell.** The field does not: it
still holds 0.45 at 50 m and decays over hundreds of metres, as
terrain-driven flow should. A mast reports the error in its own column
and says essentially nothing about its neighbour, so interpolating it
spreads a number uncorrelated with the target and injects more error
than it removes. More masts help only by averaging that injected noise
back toward zero.

Set against the vertical behaviour this is a sharp description of what
is left. The propagation coefficient from 10 m is 0.94 at 5 m, 0.69 at
20 m and 0.12 at 40 m, so the error is strongly coherent through the
surface layer. Horizontally, at the same heights, it is white at the
grid scale.

That single fact accounts for the whole run of negative results recorded
above. Loss reweighting, friction velocity, the frontal-area channels,
the column model, the additive mast correction, the WAsP-style
multiplicative transfer and the slab mass budget are all corrections
which are smooth in the horizontal, and none of them can touch a
horizontally white residual. The two constructions that did move it, a
per-column measurement and the elliptic projection, are the two with a
degree of freedom in every cell.

It also says where the error comes from: the network reproduces the
smooth part of the field and misses the grid-scale part, and the missing
part is what the error consists of. That is a statement about what the
terrain raster determines, not about the size of the network, and it is
the conclusion the architecture sweep and the reweighting experiment
reached by other routes.

Script: ``scratchpad/mast_count.py``.

A CALMET-style kernel stops the harm but cannot make a gain
==============================================================

The mast experiment above spread each measurement by inverse-distance
weighting NORMALISED to a partition of unity, which is what CALMET does
with its 1/r^2 weights, and every count came out worse than doing
nothing. That normalisation is the problem: far from every mast the
weights are all tiny but their RATIO is not, so a full-magnitude
correction is still applied, drawn from data that has no bearing on that
location.

Two other families do not have that property. An UNNORMALISED kernel
with k(0) = 1 decays to zero away from the masts, so an unobserved
location is simply left alone. A CORRELATION-WEIGHTED scheme multiplies
the nearest mast's error by the measured spatial correlation at that
distance, which for a single informative observation is the optimal
linear analysis -- the increment shrunk by exactly the correlation,
which is what optimal interpolation and kriging do and what the
normalised form omits.

Gaussian kernels, bandwidth in grid cells, one cell being 50 m. Change
against a baseline of 0.5253 m/s over all levels; negative is better:

=====  ============  ========  ========  ========  ========  ========
masts  scheme        sig=0.5   sig=1     sig=2     sig=4     sig=8
=====  ============  ========  ========  ========  ========  ========
5      normalised    +1.51 %   +5.70 %   +14.72 %  +26.86 %  +25.23 %
5      unnormalised  -0.02 %   +0.01 %   +0.18 %   +0.66 %   +3.05 %
10     normalised    +3.39 %   +10.87 %  +23.44 %  +27.53 %  +23.81 %
10     unnormalised  -0.03 %   +0.03 %   +0.37 %   +1.39 %   +5.36 %
50     normalised    +12.48 %  +24.71 %  +26.74 %  +19.95 %  +9.05 %
50     unnormalised  -0.17 %   +0.16 %   +1.56 %   +7.39 %   +27.16 %
100    normalised    +18.84 %  +27.05 %  +24.05 %  +14.21 %  +4.52 %
100    unnormalised  -0.32 %   +0.33 %   +3.29 %   +14.04 %  +51.59 %
=====  ============  ========  ========  ========  ========  ========

**The normalised family is worse everywhere and worse the wider the
kernel.** The unnormalised family is essentially neutral, and turns
slightly positive only at the narrowest bandwidth.

The correlation-weighted scheme -- the ceiling for any linear analysis
driven by these observations -- gives -0.02 %, -0.04 %, -0.18 % and
-0.35 % for 5, 10, 50 and 100 masts. It coincides with the unnormalised
kernel at sigma = 0.5, as it must: the measured correlation is 0.017 at
one cell, so the optimal weight is a spike on the mast column and
nothing beyond it.

The gain is exactly the fraction of columns instrumented
-----------------------------------------------------------

=====  ===================  ============  ==============
masts  fraction of columns  optimal gain  N/10000 x 40 %
=====  ===================  ============  ==============
5      0.05 %               -0.02 %       0.02 %
10     0.10 %               -0.04 %       0.04 %
50     0.50 %               -0.18 %       0.20 %
100    1.00 %               -0.35 %       0.40 %
=====  ===================  ============  ==============

The optimal gain tracks ``(masts / columns) x 40 %`` across a factor of
twenty in mast count. That is the arithmetic of a correction which fixes
the columns it measures and no others, and it is the practical form of
the white-in-the-horizontal result: **a mast is worth exactly one
column.**

For scale, 100 masts in a 5 x 5 km box is one every 500 m, a density no
campaign has ever fielded, and it buys a third of a per cent. So a
Gaussian or CALMET-style spreading is worth using in place of a
normalised one -- it removes a 27 % penalty -- but the reason to use it
is to avoid harm, not to obtain a gain.

Script: ``scratchpad/mast_kernel.py``.

