==================================
The terrain corpus and its splits
==================================

:doc:`cases` is a catalogue: eight domains to look at and solve. This is
the other thing -- a corpus to train on, and, more to the point, to hold
terrain **out** of.

The surrogate's headline claim (:doc:`surrogate`) is a demonstration on
unseen terrain. Whether that is a demonstration or a self-portrait comes
down entirely to whether the held-out terrain was really held out, so the
split logic here gets more care than the size of the corpus does.

::

    python3 cases/build_corpus.py --survey     # download; needs network
    python3 cases/build_corpus.py --split      # cluster and split; offline
    python3 cases/build_corpus.py --summary
    python3 cases/build_corpus.py --check      # solve a sample

What is in it
=============

Twenty-nine wildfire locations from the ``wildfire_levelset`` reference
list, one 10 × 10 km SRTM tile each, cut into a 3 × 3 grid of 5 × 5 km
windows at a 2500 m stride. A window is a phase 16A domain exactly --
same 100 × 100 × 60 cells, same 50 m horizontal resolution -- and the
middle window of each site *is* that site's catalogue case.

After screening: **28 sites, 252 windows, 20 clusters**.

Two stages, and why they are separate
=====================================

``--survey`` downloads and **measures**. ``--split`` **judges**. Every
threshold lives in the second stage and reads from committed JSON, so a
threshold can be argued about, changed and re-run without downloading
eleven megabytes of SRTM twenty-nine times over. It also means the split
is reproducible on a machine with no network, no geo stack and no tiles,
which is what makes it a property of the corpus rather than something
that happened once on a laptop.

Each site's ``corpus/<slug>/survey.json`` is committed and carries every
window's extremes, derived grid and terrain descriptors. The ``tile.csv``
it came from -- about ten megabytes -- is not.

Three ways terrain leaks
========================

A window and its neighbour
--------------------------

Adjacent windows share half their ground. A test window whose neighbour
is in train is not held out at all, so a site's windows are never split:
the fold is chosen for the site, not for the window.

Two fires on the same mountain
------------------------------

This one is easy to miss, and the reference list is full of it.

============================  ==========
Pair                          Separation
============================  ==========
Tubbs and Kincade                10.3 km
Easy and Woolsey                 15.1 km
Detwiler and Yosemite            15.1 km
Carr and Delta                   22.4 km
**Thomas and Woolsey**           28.3 km
============================  ==========

Thomas and Woolsey are **both in the phase 16A catalogue**. Treating
fires as independent because they have different names would put the
same ridgeline on both sides of the split.

So sites are single-linkage clustered by great-circle distance first, and
the fold is chosen for the cluster. Single linkage rather than a radius
search around each site, because the property has to be transitive: if A
is near B and B is near C, all three share ground even when A and C are
far apart.

The radius comes out of the list rather than out of a preference for
round numbers. Thirteen pairs fall under 55 km and the next pair up is
80 km, so **50 km** sits in a gap -- moving it to 40 or 70 changes
nothing, which is what you want of a threshold.

A window that reaches into another site
---------------------------------------

Not implied by the radius. A window centre can sit a corner offset --
3536 m for a 3 × 3 at a 2500 m stride -- from its site, so two clusters
just over the radius apart could still have physically overlapping
windows. ``assert_no_leakage`` computes that geometry rather than
assuming the radius covers it, so changing the stride is caught here
instead of in a result.

What is **not** a leak
======================

Every window has its own vertical grid, because the floor follows its own
relief, so no two samples share a ``z_cc``. Phase 16A flagged this as
making samples unstackable. It does not matter: the surrogate trains on
levels at fixed heights **above ground**
(:mod:`fastwindterrain.levels`), which have the same shape and the same
physical meaning on every window whatever the absolute grid is. The
per-window grid is needed only to reconstruct 3D, which happens one
sample at a time anyway.

Screening
=========

The reference list is taken verbatim, including rows that read oddly as
fire records. A hand-picked eight could sidestep that; a corpus cannot,
so it has to be able to say no.

Screening is **per window**, and the granularity is the finding. Erskine
Fire spans 81 m across its 10 km tile, which reads as terrain --
while eight of its nine 5 km windows hold 10 to 12 m and are plates. A
site-level screen on tile relief would have admitted it and trained on
eight flat samples. A window is what gets trained on, so a window is what
gets judged; a site whose windows are mostly plates is rejected outright.

At a 60 m minimum -- a 1.2% grade over 5 km -- exactly one site is
affected, and the threshold is not on a knife edge: raising it to 80 m
drops three more windows out of 261.

======================  =======================================
Rejected                Why
======================  =======================================
``erskine_fire``        1 of 9 windows clear 60 m; a plate with
                        a corner
======================  =======================================

The split
=========

Balanced by **site count**, not cluster count -- the clusters are
lopsided, three holding three fires each and eighteen holding one, so
splitting on cluster count would put a third more terrain in one fold
than intended. Clusters go largest-first to whichever fold is furthest
below its target share.

=========  =======  =========  ==========
Fold       Sites    Windows    Share
=========  =======  =========  ==========
train           18        162       64.3%
val              4         36       14.3%
test             6         54       21.4%
=========  =======  =========  ==========

The closest cross-fold pair is **Rim and Slinkard, 53.1 km apart** --
which is the sentence the unseen-terrain claim actually rests on, and the
reason it is computed and committed rather than asserted in prose.

Held out: Bootleg, Coastal, Ditch, Marshall, Park and Soberanes.

Does the held-out terrain look like the training terrain?
=========================================================

The question a reviewer asks about any unseen-terrain claim, and the
honest answer is a range per fold rather than a sentence. Six descriptors
are computed per window from a plain binning of the point cloud --
deliberately from the ground rather than from the solver's interpolation,
so they do not depend on solver settings.

==============  =========================================================
``relief``      max minus min; it sets the vertical grid
``std``         elevation spread -- one big slope versus broken country
``slope_mean``  everyday steepness
``slope_p95``   the steep tail, which pushes flow around rather than over
``tri``         ruggedness: the mean absolute difference between a cell
                and the mean of its eight neighbours. Small-scale, and
                largely independent of relief -- a smooth 1000 m ramp and
                a boulder field can share a relief and never share a TRI
``aniso``       eigenvalue ratio of the gradient covariance. 1 is
                isotropic hummocks; large is parallel ridges, which
                channel wind
==============  =========================================================

Min – median – max over each fold's windows:

===========  =====================  =====================  =====================
Descriptor   train                  val                    test
===========  =====================  =====================  =====================
relief       52 – 524 – 1328        412 – 739 – 1158       86 – 364 – 1970
std          9.8 – 99.8 – 300       67 – 141 – 244         18 – 64 – 383
slope_mean   0.030 – 0.216 – 0.483  0.118 – 0.319 – 0.428  0.032 – 0.165 – 0.455
slope_p95    0.084 – 0.475 – 0.965  0.316 – 0.605 – 0.708  0.081 – 0.488 – 0.880
tri          0.59 – 2.75 – 6.11     1.23 – 3.41 – 6.32     0.52 – 2.66 – 6.11
aniso        1.01 – 1.21 – 1.91     1.06 – 1.21 – 1.55     1.06 – 1.25 – 2.26
===========  =====================  =====================  =====================

**Read this before quoting a test score.** The test fold is not a subset
of the training range, and saying so is more useful than a reassuring
sentence:

* Test **relief reaches 1970 m against training's 1328 m**, and test
  ``std`` reaches 383 against 300. Ditch Fire's steepest windows are
  outside anything trained on, so results there are extrapolation, not
  interpolation.
* Test ``aniso`` reaches 2.26 against training's 1.91 -- more strongly
  ridged ground than the network will have seen.
* At the other end the test fold's median relief (364 m) is *below*
  training's (524 m), because Bootleg, Marshall and Coastal are the
  gentle sites. So the test fold is harder at one end and easier at the
  other, and a single aggregate score over it will hide both.

The useful way to report a result on this corpus is therefore **against
the descriptors**, not as one number: error versus relief, versus
``slope_p95``. That also turns the extrapolation from an awkward
admission into the more interesting half of the result.

Wind
====

Many terrain shapes, few wind directions: **eight**, the conventional
rose, identical for every window so that what differs between samples is
the ground.

Speed is **not** a dataset axis, and that is measured rather than
assumed. ``cases/linearity_study.py`` solves one window at 2, 4, 16 and
30 m/s and compares each against the 10 m/s result scaled:

.. code-block:: text

       speed             u             v             w
         2.0     5.044e-15     5.773e-15     1.689e-14
         4.0     5.044e-15     5.773e-15     1.689e-14
        16.0     5.044e-15     5.773e-15     1.689e-14
        30.0     5.864e-15     8.758e-15     2.716e-14

Linear to round-off, as the algebra says it must be: the inflow scales
with ``u_ref``, O'Brien integrates a divergence that scales with it, the
Poisson right-hand side scales with it, and ``alpha`` depends on terrain
slope rather than on the flow. So a speed is a free normalisation and a
second one would double the compute for nothing.

That is a property of *this* operator. ``Anisotropy``'s ``f_Ri`` and
``f_Fr`` hooks return 1 today; a stability or Froude correction puts the
speed back into the problem, and the study says so and should be re-run.

The reference is **10 m/s at 80 m** -- hub height rather than the
solver's 10 m default, which is a met-mast height inside the steepest
part of the profile. ``Inflow``'s profile is terrain-following
(``z_agl = z_cc[k] - h[j][i]``, ``Source/Inflow.cpp:349``), so this
anchors 10 m/s at 80 m above ground in every column, which in the lowest
column is 80 m above the lowest terrain.

Counting samples honestly
-------------------------

252 windows × 8 directions is 2016 solves. It is **not** 2016
independent samples. On a square domain the physics is very nearly
rotation-equivariant, so a terrain at 90° and a quarter-turn of that
terrain at 180° are close to the same problem: the four axis-aligned
directions act partly as a rotation augmentation of the terrain rather
than as four independent conditions. Worth saying in a methods section
before a reviewer says it.

Checking the corpus solves
==========================

``--check`` solves a seeded sample from each fold and runs the five
checks that catch the ways a real-terrain case goes quietly wrong --
``straddles``, ``finite``, ``divergence``, ``vertical``, ``speed-up``
(see :doc:`cases`). The manifest's arithmetic can be perfect and the
corpus still broken: the solver never checks terrain against the domain,
and 252 windows is 252 chances at the silent all-fluid or all-solid
failure.

One note on reading its output: the cell-centred ``max|div(u)|`` can
*rise* across the projection while the norm the solve actually controls
falls. ``Source/Solver.cpp:180`` reports both, and the check reads
``max_divergence_fe``, which is the controlled one.

Four passes is not enough on heavily occupied windows
-----------------------------------------------------

A seeded sample of twelve windows, four projection passes, 10 m/s at
225°. Mean **50.7 s** a solve. Every window straddles its terrain, every
field is finite, every one accelerates over its ridges -- and **four
fail the divergence check**:

========================  =======  =====================  ======
window                    solid    ``max_div_fe``         result
========================  =======  =====================  ======
``bootleg_fire:22``         13.0%  0.0367 → 0.0165        ok
``marshall_fire:01``        17.5%  0.0478 → 0.0157        ok
``carr_fire:12``            23.4%  0.0748 → 0.0396        ok
``delta_fire:20``           25.7%  0.133 → 0.108          ok
``easy_fire:01``            31.2%  0.115 → 0.106          ok
``apple_fire:11``           51.6%  0.138 → 0.131          ok
``black_summer_fire:10``    52.0%  0.151 → 0.147          ok
``slinkard_fire:22``        57.9%  0.128 → **0.165**      FAIL
``kincade_fire:20``         59.5%  0.152 → **0.161**      FAIL
``apple_fire:10``           59.9%  0.108 → **0.170**      FAIL
``ditch_fire:20``           63.6%  0.194 → **0.259**      FAIL
========================  =======  =====================  ======

The ordering is the finding. **The failures are exactly the windows
above about 55% solid**, and the margin narrows steadily on the way there
-- Bootleg's divergence falls by a factor of 2.2, Apple's 51.6% window by
5%, and then it inverts. On these windows four passes leave the flow with
*more* divergence than the terrain-following inflow it started from, in
the norm the solve controls.

This is not the phase 17 convergence table contradicting itself. That was
measured on Creek at 71% solid and converged at 0.87 a pass, so solid
fraction alone does not predict it; something about these particular
windows -- narrow passages between solid blocks, most likely -- makes the
first few passes non-monotone. What it does mean is concrete:

* the dataset target is "FastWindTerrain at four passes", and on roughly
  a third of the corpus that field is **not** closer to mass-consistent
  than the initial condition. That is still a well-defined target for a
  surrogate to emulate, but it cannot be described as an approximately
  converged solution;
* ``--check`` should be run and its failures counted before generating a
  dataset, not after;
* whether more passes recover monotonicity on these windows is open, and
  is the first thing to settle in phase 21 -- if they need 16, the
  corpus's cost is four times the 28 core-hours quoted above.
