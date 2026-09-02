---
name: level-placement
description: Choose where to sample 2D slices of a 3D field, and measure the reconstruction ceiling, BEFORE training any surrogate. Use when a project plans to predict a few 2D levels/layers/slices and reassemble a 3D volume from them — wind over terrain, ocean depth layers, tomography, any dimension-reduced surrogate. Covers the ceiling experiment, placement rules that beat level count, and the traps that make a decimation study lie.
---

# Choosing levels before there is a model

A surrogate that predicts K two-dimensional levels and reconstructs a 3D
volume has **two** error sources: what the network gets wrong, and what
the reconstruction loses even from perfect input. Only the second is
knowable before any training, it is cheap to measure, and it bounds the
first — if stitching from ground-truth levels loses 20%, no network beats
20%.

Measure it first. It costs an afternoon, it needs no ML framework, and it
decides the architecture's output shape, which is expensive to change
later.

## The procedure

### 1. Build extraction and stitching as ONE pair of operators

Write `extract(volume) -> levels` and `stitch(levels) -> volume` once, in
the numerical library the training loop will use (numpy/torch), not in the
simulation code.

**The same `stitch` must run in the ceiling experiment and in production.**
If they are two implementations, the measured ceiling does not bound the
real thing and the whole study is decoration.

If the reconstruction involves an operator that already exists in
compiled simulation code, transcribe it rather than calling it — a
training loop needs it differentiable. Then **validate the transcription
against the original as an oracle**, and report the agreement as a number.
Expect a few ULP, not bit-exactness: compiled code contracts `a - b*c`
into an FMA and rounds once where numpy rounds twice. Look for the
signature of that — bounded, non-accumulating, and exactly zero wherever
the operator is exactly constrained — rather than assuming any small
difference is fine.

Look for a way to validate **without modifying the simulation code**.
There is often an existing switch that isolates the operator: running with
it off and on gives two fields differing by exactly one application.

### 2. Run the ceiling experiment

Take a solved volume. Keep only the levels. Stitch. Compare against what
you deleted.

- Mask to valid cells (fluid, unmasked, in-domain). Including masked cells
  makes any method look good.
- Normalise by a physical scale (max magnitude), not by the mean, so the
  number transfers between cases.
- Report **error in the deliverable band separately from whole-volume
  error**. They answer different questions and frequently disagree — in
  the wind case, 5 levels gave 20% over the column and 1.8% inside the
  band people actually asked about. Quoting one number would have been
  wrong either way.
- Run on at least three cases spanning the hard parameter (relief,
  gradient, roughness). A single case cannot distinguish a property of the
  method from a property of that case.

### 3. Sweep placement, not just count

**This is the finding that transfers.** Count is what papers report;
placement is what has to be reproduced, and placement usually wins.

Sweep at least: uniform, log-spaced, and "anchored on the heights the
deliverable is quoted at". Measured on wind over terrain:

| k | rule | column RMSE |
|---|---|---|
| 5 | uniform | 0.0353 |
| 5 | **log** | **0.0170** |
| 12 | uniform | 0.0187 |
| 12 | log | 0.0119 |

**Five log-spaced levels beat twelve uniform ones**, on every terrain
tested. If that holds in your problem, say so in the form "placement beats
count" — it transfers to grids that are not yours, where "use 8 levels"
does not.

### 4. Do not anchor levels on the heights the answer is wanted at

The most counter-intuitive result, and the one worth checking in any new
problem. Anchoring on the requested heights (10, 80, 100, 120, 160 m) was
**2–3× worse inside that very band** than log spacing:

| level set | band RMSE |
|---|---|
| log-spaced | 0.0099 |
| anchored on the requested heights | 0.0183 |

Because clustering samples at 80–160 m leaves nothing between 10 and 80 m,
which is where the gradient lives. **Sample where the field changes
fastest; interpolate out to where the answer is wanted.** Those are
different places, and conflating them is the default mistake.

### 5. Ablate the frame

If levels can be taken in more than one coordinate frame — terrain-
following vs constant-elevation, isopycnal vs fixed-depth — this is the
single highest-leverage ablation and it is one line of code.

It was worth a factor of **fifty** (0.016 vs 0.792 column RMSE). A
constant-elevation slice over 1100 m of relief is underground across most
of the domain, so most of what it samples is not the field at all. Check
this before tuning anything else.

### 6. Pick k from a budget rule, not from the table's knee

Establish whether levels cost anything. In the wind case they did not —
the same solve, extracted differently — so k trades against **network
size**, not dataset size, and the rule becomes:

> pick the smallest k whose ceiling is under a third of the expected
> network error.

A ceiling comparable to network error doubles total error for nothing; a
ceiling far below it is spent capacity. Work out which resource k actually
consumes in your problem before copying either the rule or the number.

## Traps

**Interpolate the field, or derive it from a constraint?** Deriving a
component from a physical constraint (continuity, incompressibility) looks
free but inherits the reconstruction error of its inputs and then
differentiates and integrates it. Measured: deriving was ~8× worse than
interpolating. That is not a verdict — interpolating means the network
predicts three fields instead of two — but it establishes a floor, and the
comparison must be made rather than assumed.

**Seeding a boundary integration is case-dependent.** A physically
motivated seed (the kinematic condition `w = u·∇h`) *helped* on gentle
terrain and *hurt* on steep, because on steep slopes the flow is being
pushed around the obstacle rather than over it and the seed assumes
otherwise. Predictions about which seed wins are unreliable; measure per
regime.

**Below the lowest level is not interpolation.** Extrapolating a straight
line down from the first level puts a sign error in the near-boundary
cells. Use the physical asymptotic law (log law, no-slip, whatever the
problem has). Same for above the highest level — holding the top value is
only defensible if the highest level is somewhere the geometry has stopped
mattering, which is exactly what the ceiling experiment tells you.

**Sub-grid levels are diagnostics, not predictions.** If a requested level
sits below the first cell centre in most columns, it cannot be
interpolated and must come from a law. Keep it out of the set the network
predicts, and say why.

**A "warm start" claim needs its own measurement.** Seeding an iterative
solver from a reconstruction is an obvious-looking win that was measured
and found **negative**: the reconstruction had ~1.3% RMSE but ~44% max-norm
error, and the iteration converges in a max norm. Check which norm your
consumer converges in before claiming the reconstruction is a good initial
condition.

## What to write down

Record the ceiling, the placement rule, and the frame result as
**transferable** findings, separately from the ones that are properties of
your particular operator. A reader wants to know which results survive a
change of solver.

Record negative results with their mechanism. "Warm starting did not help"
invites someone to retry it; "warm starting did not help *because* the
error is max-norm and spread through the column, so there is no localised
fix" closes it.
