======================================
Beyond the 5 km window
======================================

Every window in the corpus is 5 x 5 km. The question this page answers
is whether a model trained on those windows can be carried to a 10 or
15 km domain, and if so what that costs.

A word on terms first. Elsewhere in these notes **stitching** means the
VERTICAL reconstruction -- a few horizontal levels reassembled into a 3D
field, measured by ``cases/stitching_study.py``. Nothing on this page is
about that. Here the question is horizontal: tiles of ground laid side by
side, and whether the wind over one of them can be computed without
knowing what the wind over its neighbours is doing.

Two things have to be true for the answer to be yes. The 5 km SOLVE has
to be a good approximation to the same ground inside a larger domain,
and the NETWORK has to behave itself on a grid larger than the one it
trained on. They fail in different ways and are measured separately.

.. contents::
   :local:
   :depth: 1


What a 5 km box costs, from data already on disk
=================================================

The corpus cuts nine windows per site on a 3 x 3 lattice at a 2500 m
stride, so adjacent windows **overlap by half**. Each was solved
independently as its own 5 km domain, with its own vertical grid and its
own uniform inflow on its own four boundaries
(``cases/corpus.py:window_config``). The same ground is therefore solved
twice, under two different truncations, and the two answers can be
differenced cell for cell.

If two independent 5 km solves disagree over the ground they share, then
the wind at a point is not determined by a 5 km neighbourhood: it depends
on where the box was drawn. That is a ceiling on any scheme which
reconstructs a larger domain out of 5 km tiles, and it needs no new
solves to measure.

The index mapping is checked rather than assumed. Both windows grid their
terrain from the same point cloud onto coinciding cell centres, so the
terrain in the shared region must agree exactly; it does, to
**0.0000 m rms** over all 9.0 M cells compared. Had the mapping been
wrong -- an x/y transpose being the obvious way -- that check would have
failed and every number below with it.

Only the six band levels are compared. They are terrain-following AGL at
5, 10, 20, 40, 80 and 160 m and are identical in every window. The three
aloft levels are geomspaced to each window's own column top, sit at
different physical heights in the two members of a pair, and would
measure the level placement rather than the flow.

Reverse directions are skipped: the reverse of a solve is its exact
negation, so it differences to the same magnitude and would only double
the run time.

Over all 2240 overlapping pair-cases, 8 960 000 cell-comparisons per
level:

.. list-table::
   :widths: 12 18 18 16 14
   :header-rows: 1

   * - level
     - 3-component
     - horizontal
     - field rms
     - ratio
   * - 5 m
     - 1.3604
     - 1.3412
     - 5.6807
     - 23.9 %
   * - 10 m
     - 1.3571
     - 1.3361
     - 6.7729
     - 20.0 %
   * - 20 m
     - 1.1836
     - 1.1566
     - 7.9060
     - 15.0 %
   * - 40 m
     - 0.9197
     - 0.8842
     - 9.0802
     - 10.1 %
   * - 80 m
     - 0.7960
     - 0.7542
     - 10.1064
     - 7.9 %
   * - 160 m
     - 0.7728
     - 0.7309
     - 11.1764
     - 6.9 %
   * - **band**
     - **1.0932**
     - **1.0651**
     - 8.6617
     -


The vertical grid is part of that number
-----------------------------------------

``casegen.grid_from_relief`` sets each window's floor at its own ``z_min``
and its stretching ratio from its own relief, so two windows with
different relief have cell centres at different heights and the band
fields are interpolated out of different columns. Some of the
disagreement above is that interpolation rather than the truncation the
experiment is about.

Binning the pairs by how different the two grids are separates it:

.. list-table::
   :widths: 20 20 20
   :header-rows: 1

   * - difference
     - binned by relief
     - binned by floor
   * - 0-25 m
     - 0.8510
     - 0.8293
   * - 25-50 m
     - 1.0509
     - 0.9671
   * - 50-100 m
     - 1.1035
     - 1.2563
   * - 100-200 m
     - 1.1958
     - 1.4130
   * - 200-400 m
     - 1.2288
     - 1.5496
   * - over 400 m
     - 1.5926
     - 1.5645

So it is part of it, and not all of it. Read this carefully in one
respect: relief and floor differences are also proxies for how different
the two pieces of GROUND are, so the trend mixes the grid with the
terrain and cannot be attributed to the grid alone.

Taking the tightest subset -- both floor and relief within 25 m, so the
ratio and the cell heights effectively coincide -- leaves 380 pair-cases
and 1 700 000 cells per level. Against it, the model's own error on the
same six levels:

.. list-table::
   :widths: 10 16 12 14 14 10
   :header-rows: 1

   * - level
     - box ambiguity
     - / sqrt 2
     - model, test
     - model, demo
     - ratio
   * - 5 m
     - 0.896
     - 0.633
     - 0.901
     - 1.014
     - 0.62
   * - 10 m
     - 0.789
     - 0.558
     - 0.889
     - 0.977
     - 0.57
   * - 20 m
     - 0.666
     - 0.471
     - 0.766
     - 0.786
     - 0.60
   * - 40 m
     - 0.565
     - 0.399
     - 0.544
     - 0.530
     - 0.75
   * - 80 m
     - 0.538
     - 0.380
     - 0.367
     - 0.344
     - 1.11
   * - 160 m
     - 0.537
     - 0.379
     - 0.304
     - 0.284
     - 1.34
   * - **band**
     - **0.679**
     - **0.480**
     - **0.672**
     - **0.717**
     -

The ``/ sqrt 2`` column turns a disagreement into a per-solve estimate on
the assumption that the two boxes' truncation errors are independent and
of equal size. **It is a lower bound.** If both 5 km boxes are wrong in
the same direction relative to a true 10 km solve -- which is what a
common truncation would do -- that shared error cancels in the difference
and is invisible here. Only a 10 km solve can see it.

Note also which comparator this is. The published ``e_3D`` of 0.347 m/s
is the reconstructed VOLUME, most of which is well-predicted air aloft;
``e_lev`` over all nine levels is 0.582 m/s. On the six band levels alone
-- the like-for-like number, since that is what the ceiling is measured
on -- the model is at 0.672 m/s on the corpus test fold and 0.717 m/s on
the demo sites.


Fetch says it is not a boundary transient
------------------------------------------

If the disagreement were the flow adjusting to an artificial inflow face,
it would decay downwind of that face and a buffer strip would fix it. It
does not decay. Binned by the distance back to the nearer box's own
upwind boundary along the flow direction, on the matched-grid subset:

.. list-table::
   :widths: 18 14 14
   :header-rows: 1

   * - fetch
     - rms m/s
     - cells
   * - 0-500 m
     - 0.7514
     - 308 690
   * - 500-1000 m
     - 0.6397
     - 290 070
   * - 1000-1500 m
     - 0.6124
     - 271 450
   * - 1500-2000 m
     - 0.6141
     - 252 830
   * - 2000-2500 m
     - 0.6866
     - 234 210
   * - 2500-3500 m
     - 0.7346
     - 222 560
   * - over 3500 m
     - 0.7225
     - 120 190

Flat, with a shallow minimum around 1-2 km and a rise beyond it. The same
shape appears on the full set at a higher level (1.19 falling to 1.03 and
back to 1.08). Whatever a 5 km box is missing, it is not something the
flow forgets a kilometre downwind of the inlet.

Scripts: ``scratchpad/overlap_ceiling.py`` (the ceiling and the geometry
check), ``scratchpad/grid_control.py`` (the vertical-grid confound) and
``scratchpad/fetch_decay.py`` (fetch, and the band-level model error).


The network side
================

Measured on ``dcnn_w96_film``, the best run, whose inputs are terrain,
slope magnitude and two constant direction planes. It carries no
window-global spectral channels, so nothing in its input convention is
tied to the window size.

Script: ``scratchpad/tier1_extent.py``.

**A uniform shift of the ground changes nothing.**
``training.terrain_channels`` subtracts the window's own mean, so raising
the terrain by 500 m moves the output by **0.000e+00 m/s** -- exact. Tiles
need no shared vertical datum, and each may be centred on its own mean
exactly as in training.

**Reach.** One backward pass gives the influence of every input cell on
one output cell, which is the receptive field as the network weights it
rather than as the dilations bound it. For ``u`` at 10 m in the centre
cell:

.. list-table::
   :widths: 14 28
   :header-rows: 1

   * - radius
     - share of total influence
   * - 100 m
     - 52.52 %
   * - 200 m
     - 62.99 %
   * - 400 m
     - 69.06 %
   * - 800 m
     - 77.68 %
   * - 1600 m
     - 88.56 %
   * - 2400 m
     - 97.75 %
   * - 3200 m
     - 99.86 %

The nominal radius from dilations 1, 2, 4, 8, 16, 1 is
2 x (1+2+4+8+16+1) = 64 cells = 3.2 km, and only **0.14 %** of the
influence lies beyond it. That residue is not a numerical accident:
``DilatedCNN`` normalises with ``GroupNorm``, which takes its statistics
over the whole spatial extent and so couples every cell to every other
one however distant. It is weak, but it is not zero, and it is why the
next result comes out as it does.

Two things follow that matter for planning. The field is genuinely
non-local -- barely half the influence is within 100 m -- and its radius
is **wider than the 2.5 km half-window it trained in**, so every training
sample had part of its receptive field filled with zero padding rather
than ground.

**Tiled against full-domain.** On a real 10 x 10 km field assembled from
one site's nine windows, running the weights over the whole 200 x 200
grid in one pass is compared against tiling at the training geometry and
keeping each tile's valid interior:

.. list-table::
   :widths: 16 14 16
   :header-rows: 1

   * - crop margin
     - overlap
     - difference
   * - 0 cells
     - 0 m
     - 0.6623
   * - 8 cells
     - 800 m
     - 0.5120
   * - 16 cells
     - 1600 m
     - 0.3886
   * - 24 cells
     - 2400 m
     - 0.3616
   * - 32 cells
     - 3200 m
     - 0.3501
   * - 40 cells
     - 4000 m
     - 0.3630
   * - 48 cells
     - 4800 m
     - 0.3467

**It plateaus rather than converging to zero.** A pure convolution with a
bounded receptive field would agree exactly once the overlap exceeded
that bound; this one settles around 0.35 m/s, which is half the model's
own error. A control which replicates a single window 2 x 2 -- identical
GroupNorm statistics by construction, so only the receptive field sees
anything new -- gives 0.2459 m/s, so both mechanisms contribute and
neither explains it alone.

It is tempting to read this as an argument for tiling: every tile then
gets exactly the size and exactly the normalisation it trained with,
while one big pass is a different computation from the validated one.
**That reading is wrong, and the next section is why.** This experiment
can only measure the two against each OTHER; neither is scored against a
reference, because at this stage there was no 10 km solve to score
against. A disagreement does not say which one is closer to the truth.

That leaves a real tension. The influence profile wants a 2.4 km margin,
but a 5 km tile which gives up 2.4 km on each side has 200 m of usable
output left. Affording the margin means larger tiles, which is precisely
the direction the plateau above lives in.

**Cost is not the constraint.** A single forward pass takes 0.04 s at
5 km, 0.18 s at 10 km and 0.44 s at 15 km on one CPU thread. Even the
maximal-overlap tiling -- 676 passes for a 10 km domain -- is under
thirty seconds.


A real 10 km solve
==================

Everything above is an estimate because both members of every pair are
5 km boxes: whatever they get wrong TOGETHER cancels in the difference.
One 10 km solve removes the guesswork, and it turns out to be cheap.

**The terrain costs nothing.** A site's nine window terrain arrays are
gridded on their own 100 x 100 cell centres at 50 m, and at a 2500 m
stride those centres coincide exactly with a 200 x 200 grid over the same
10 km. ``Terrain::InterpolateIDW`` returns an input point's own value on
an exact hit, so the surface the 10 km solve sees is identical, cell for
cell, to the surface the window solves saw. No download, no
re-interpolation, and no terrain difference to confound anything.

**The vertical grid is controlled, not hoped away.** Every window is
re-solved here on the 10 km domain's own vertical grid, so the only thing
left different between a window solve and the 10 km solve is the
horizontal extent.

On ``park_fire`` at 045 degrees -- a test-fold site, 1001 m of relief
over the 10 km field -- the 10 km solve took about a quarter of an hour
and each window about three minutes. What truncating to 5 km costs,
measured rather than inferred:

.. list-table::
   :widths: 18 12 14 14 14
   :header-rows: 1

   * - window
     - place
     - box error
     - on own grid
     - grid share
   * - park_fire:11
     - centre
     - **0.5654**
     - 1.3234
     - 1.1625
   * - park_fire:00
     - corner
     - 0.6131
     - 1.1916
     - 0.9762
   * - park_fire:20
     - corner
     - 0.6309
     - 1.1138
     - 0.9164
   * - park_fire:10
     - edge
     - 0.7086
     - 1.3840
     - 1.2061
   * - park_fire:21
     - edge
     - 0.7491
     - 1.4882
     - 1.2867
   * - park_fire:01
     - edge
     - 0.7518
     - 1.3578
     - 1.0959
   * - park_fire:02
     - corner
     - 0.7961
     - 1.5444
     - 1.3112
   * - park_fire:12
     - edge
     - 0.8040
     - 1.5328
     - 1.3495
   * - park_fire:22
     - corner
     - 0.8176
     - 1.6750
     - 1.4413

The centre window is the one to read: all four of its boundaries are
artificial, while a corner window shares two of them with the real 10 km
boundary and is flattered by it. Its **0.5654 m/s** sits inside the
0.48-0.77 bracket the overlap estimate predicted, so the reasoning there
held up. Per level it is 0.492, 0.554, 0.593, 0.601, 0.581, 0.566 --
nearly flat in height, where the model's own error falls by a factor of
three from 5 m to 160 m.

The unexpected column is the last one. Putting a window on a different
VERTICAL grid moves the answer by 0.92-1.44 m/s, which is larger than
truncating its horizontal extent. A 5 km window sized from its own relief
resolves a much shorter column with the same sixty cells than the 10 km
domain does, and that discretisation change dominates. It vindicates the
matched-grid filter used above, and it is worth remembering before
anyone compares two solves whose grids were derived independently.


Blending the overlap, which is the overset question
----------------------------------------------------

Two windows cover the same ground and each is wrong against the 10 km
solve. Averaging them helps only if those two errors are INDEPENDENT, in
which case the mean has 1/sqrt(2) of the error and blending buys -29 %.
If both boxes are missing the same upwind terrain they are wrong the same
way, the errors correlate, and blending is cosmetic -- a seam remover
rather than an error remover.

Over the twenty overlapping pairs the mean error correlation is
**0.497** and averaging buys **-12.2 %**, about two fifths of what
independence would give. So it is worth doing and it is not a rescue. The
pattern within it is consistent: pairs involving the centre window
correlate least (0.20-0.48) and gain most (-13 % to -38 %), because the
centre box is the one whose four artificial boundaries are in completely
different places from its neighbours'.

This is the interpolation half of an overset scheme, and it is the half
available today. The other half -- re-solving each tile with its
neighbour's values on the fringe, iterated, which is what makes overset
converge for an elliptic operator -- has no path here: the network takes
terrain, slope and two constant direction planes, and there is no channel
through which a neighbour's boundary field could enter. Giving it one
means retraining with edge-condition inputs.


What the surrogate actually scores at 10 km
--------------------------------------------

The end-to-end number, all three scored on the centre window's footprint:

.. list-table::
   :widths: 34 12 12 12 12 12 12 14
   :header-rows: 1

   * - scored against
     - 5 m
     - 10 m
     - 20 m
     - 40 m
     - 80 m
     - 160 m
     - band
   * - its own 5 km solve (the usual metric)
     - 1.335
     - 1.470
     - 1.331
     - 0.697
     - 0.389
     - 0.277
     - 1.0351
   * - the 10 km solve, tiled over nine windows
     - 1.325
     - 1.453
     - 1.333
     - 0.886
     - 0.728
     - 0.684
     - 1.1123
   * - the 10 km solve, one full-domain pass
     - 1.239
     - 1.330
     - 1.159
     - 0.755
     - 0.571
     - 0.541
     - **0.9862**

Two things here, and the second is the one that matters.

Moving the target from a 5 km solve to a 10 km solve costs the tiled
model only **7 %** (1.035 to 1.112). That is far less than the 0.565 m/s
box error alone would suggest, because the two errors partly cancel
rather than adding: in quadrature and independent they would give 1.179.

And **the full-domain pass beats tiling** -- 0.986 against 1.112, and
better even than the model's score against the 5 km solve it was trained
to reproduce. This inverts the conclusion the previous section invites.
Tiling faithfully reproduces the 5 km answer, and the 5 km answer is
itself wrong by 0.565 m/s against the truth; letting the receptive field
see the real surrounding ground recovers more than the out-of-distribution
GroupNorm statistics cost. Fidelity to training turns out to be the wrong
objective when the training target is itself truncated.

**One site, one direction.** This is the result most in need of
replication before it is relied on, and it is the cheapest to replicate.

Script: ``scratchpad/tier2_tenkm.py``.


What to conclude
================

The network side is ready and the data side is the question. Nothing
about running these weights on a 200 x 200 or 300 x 300 grid is unsound:
the offset invariance is exact, the receptive field is almost entirely
inside the nominal bound, and inference is free. Tile rather than run one
big pass, and the scheme reproduces the validated computation.

What a 5 km box costs is now measured rather than estimated:
**0.5654 m/s** for a window with four artificial boundaries, against a
model error of 0.672-1.035 m/s on the same levels depending on the site.
It is the same order as the model's own error, and its height dependence
runs the opposite way -- nearly flat, where the model's falls threefold
from 5 m to 160 m. A larger domain therefore degrades a reconstruction
least where the surrogate is already worst, and most where it is
currently good.

The cost of moving to 10 km is **7 %** if the model is tiled, and
negative if it is run full-domain. That is a smaller price than the
overlap study on its own suggested, because the box error and the model
error partly cancel rather than adding in quadrature.

**Run it full-domain, not tiled.** The pull towards tiling is that it
reproduces the validated computation exactly; the objection is that the
validated computation targets a 5 km solve which is itself wrong by
0.565 m/s. Scored against the truth, one full-domain pass wins by 11 %
over tiling and beats the model's own score against its training target.
This conclusion is drawn from one site at one direction and should be
replicated before it is leaned on.

Three things are worth carrying forward. Averaging the overlap buys
-12 %, not the -29 % independence would give, because the two boxes'
errors correlate at 0.50; that is the whole of what an overset scheme can
offer without a way to feed boundary values into the network. The
vertical grid matters more than the horizontal extent, so two solves
whose grids were derived independently should not be differenced without
saying so. And nothing here is limited by compute: the 10 km solve is a
quarter of an hour and inference is under a second.


Re-measured with the anchor channel
====================================

Everything above used the baseline model, and two of its numbers carried
a caveat: the 10 km reference has 60 cells over a 2001 m column while the
training windows had much shorter ones, and that difference in vertical
spacing was afterwards measured to be worth about 1 m/s. The anchor
channel (see :doc:`surrogate`) is a fix for exactly that confound -- it
tells the network the grid it is predicting on -- so the question is
asked again here with a model that can see the thing which was
contaminating it.

Both variants take their anchor from the 10 km grid, since that is the
field being predicted; the tiled variant gets the corresponding crop of
the same plane, so the two differ only in horizontal treatment. The grid
reconstruction was checked against the corpus's own stored ``z_cc`` and
``k_first`` first: agreement to 1.4e-12 m and zero mismatches.

============================  ==========  ==========
scored against                  baseline      anchor
============================  ==========  ==========
its own 5 km solve                1.0351      0.4515
the 10 km solve, tiled            1.1123      0.7652
the 10 km solve, full-domain      0.9862      0.6100
============================  ==========  ==========

Three things change and one does not.

**The anchor model is much better at 10 km**, by 38.1 % full-domain. The
fix carries across domain sizes rather than being confined to the size it
was trained at.

**Full-domain's margin over tiling grows**, from 11.3 % to 20.3 %. The
recommendation in the previous section is strengthened, not weakened.

**The cost of extension is now visible.** Going from the 5 km target to
the 10 km one cost the baseline -4.7 % full-domain, which looked like
extension being free. It was not: the baseline's 5 km error was inflated
by the phase artefact, and that inflation was masking the extent penalty.
With the artefact removed the same step costs +35.1 % full-domain and
+69.5 % tiled. The truncation error measured on the solver alone -- 0.5654
m/s, no model involved -- has not moved; it has simply stopped being
hidden. In quadrature with the anchor model's own 0.4515 it predicts
0.723 against the 0.765 measured for tiling, so the two errors are now
close to independent and additive.

What does not change is the ranking, which is what the earlier section
claimed was safe.

Blending improves too, and for a reason worth stating. The tile errors
now correlate at 0.625-0.787 rather than 0.765-0.864, because the part of
the error that was the same network making the same mistake has been
removed and what is left is truncation, which differs between boxes. The
limit rises to 11-21 % and four tiles deliver -15.2 %. It is still
dominated: one full-domain pass beats tiling by 20.3 %, which is what
infinitely many blended tiles would achieve.

**The binding constraint has moved.** Before, the network's own error
dominated at 10 km. Now the 5 km truncation of the training data does. If
prediction over larger terrain is the goal, the next thing to change is
the window the corpus is built from, not the architecture and not the
tiling scheme.

Script: ``scratchpad/extent_anchor.py``.
