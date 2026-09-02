---
name: flow-error-tolerances
description: What counts as an acceptable error in atmospheric/turbulent flow work, and how to avoid chasing numerical precision far below the physical noise floor. Use when judging CFD or wind-solver results, choosing convergence criteria, deciding whether to discard simulation data, or building datasets for flow surrogates — especially if you come from an ML or software background where tight numerical tolerances feel like rigour.
---

# Error tolerances in turbulent flow

If you come to this from software or ML, your instinct is that smaller
error is always better and that a residual which fails to decrease is a
bug. In turbulent atmospheric flow that instinct is wrong often enough to
waste days and, worse, to make you throw away good data.

**These are turbulent flows, not laminar ones.** The physical variability,
the measurement uncertainty and the terrain representation all swamp the
tolerances a solver can be driven to.

## The numbers

| Quantity | Typical | Notes |
|---|---|---|
| Wind speed error, CFD practice | **~0.25 m/s** | An *average*, not a ceiling |
| Wind speed error, complex terrain | **up to ~1 m/s** | Steep sites (e.g. Réunion: 3000 m peaks, near-vertical cirque walls in 50 km) run higher |
| Acceptable relative error, turbulent atmospheric flow | **20–30%** | This is the operative tolerance, not a stretch target |
| Terrain-induced speed-up | **20–30%** | Also a sanity bound: far past this and you are looking at a failed solve, not flow |

**The tolerance is terrain-dependent and widens with complexity.** So the
steepest, most complex cases are the *last* ones to judge harshly, not the
first — which is the opposite of the instinct that they must be the broken
ones.

## Grid convergence does not exist in this regime

This is the part that most surprises people arriving from software, and it
is the *reason* behind the tolerances above. "Refine the mesh until the
answer stops changing" is the standard verification move, and it does not
work here.

| Regime | Grid convergence? |
|---|---|
| Laminar | **Yes.** Refine and it converges. |
| Turbulent, **wall-resolved** | **Yes.** Achievable. |
| Turbulent, **wall-functioned** (wall-modelled LES) | **No** — even on a flat surface. |
| Atmospheric boundary layer over **complex terrain** | Not remotely. |

Wall-modelled LES for the ABL struggles to converge on a **flat** surface —
the wall model imposes a grid-dependent behaviour that does not vanish
under refinement (the log-layer mismatch is a long-standing open problem).
Over complex terrain the situation is worse still.

**Consequences for how you verify:**

- **A mesh-refinement study is not a verification strategy here.** Do not
  run one expecting asymptotic behaviour and do not treat a non-monotone
  refinement sequence as a bug. You will find one, and it will not mean
  what it means in a laminar solver.
- **Choose the mesh on cost and adequacy, then validate against data** —
  measurements, or a trusted reference solution. Validation replaces
  convergence as the standard of evidence.
- **Do not treat "the finer mesh disagrees" as the finer mesh being right.**
  In this regime it is a different model, not a better-resolved one.
- Verification of the *code* (manufactured solutions, exact operator
  parity, conservation identities) is still valid and still worth doing.
  It just does not extend to convergence of the *physics* under
  refinement.

## What this changes about how you work

### Convert to physical units before judging anything

A dimensionless residual or a divergence in 1/s means nothing on its own.
Multiply divergence by the cell size to get the local velocity imbalance:

> at dx = 50 m, an L2 divergence of 0.007 1/s is **0.35 m/s** — the same
> order as the tolerance CFD already runs at.

Do this *first*. A number that looks alarming as `0.007` and unremarkable
as `0.35 m/s` should be judged as the second.

### Divergence is not mesh-independent

`div ~ Δu/Δx`, so halving the cell size roughly doubles the reported
divergence for the same flow. Measured on one real case:

```
uniform dx = 200 m    max|div| after 16 passes   0.0086
uniform dx = 100 m    max|div| after 16 passes   0.0219
```

The finer mesh looks worse and is not. **Never compare absolute divergence
across meshes of different resolution** — that comparison measures the
mesh. If you must compare discretisations, hold resolution fixed and change
only the thing under test.

### Do not drop data on criteria finer than the tolerance

The specific trap: "the residual went up instead of down, so this case
failed." Before acting on that, check three things.

1. **Is the difference physically small?** If it is inside ~0.25–1 m/s,
   it is not a failure.
2. **Do other norms agree?** L∞ and L2 routinely disagree, and can move in
   opposite directions on different cases. A max norm is dominated by a
   handful of cells — typically at an immersed boundary or a cut cell —
   and is not a statement about the field.
3. **Does the criterion track anything real?** If the "failures" do not
   correlate with any physical descriptor (relief, slope, roughness,
   curvature, occupancy), the criterion is measuring noise.

I once had all three warning signs at once and was about to discard a third
of a terrain corpus. It was caught by a domain expert asking, in effect,
"compared to what?"

**A dataset trimmed on a too-fine criterion is trimmed to what you
expected**, which is fatal for a held-out generalisation test.

### Separate correctness from accuracy

This is not a licence for sloppiness. Two different standards apply:

- **Correctness** — bit-for-bit parity between two code paths, a numerical
  transcription agreeing with its reference to a few ULP, a guard that a
  domain actually contains the terrain. These are exact and must hold.
  Tolerate nothing here.
- **Accuracy** — how close the field is to reality. Governed by the
  physical tolerance above. Tightening it past the noise floor buys
  nothing.

Confusing the two in either direction causes trouble: exact checks get
waived as "close enough", and physical tolerances get chased to round-off.

### Stretched meshes are a trade, not a defect

Geometric vertical stretching makes grid metric terms first-order accurate
where a uniform grid is second-order, so a stretched grid genuinely is a
coarser discretisation. **This is deliberate.** People use it to cut cost
and accept a small accuracy loss, and if the loss is inside the tolerance
above, that is the trade working as intended. Do not "fix" it.

### Report against complexity, not as one number

Since the acceptable band widens with terrain complexity, a single
aggregate error over a heterogeneous set hides both ends. Report error
against a descriptor — relief, 95th-percentile slope — so a reader can see
where the method holds and where it strains. It also turns an awkward
admission into the more interesting half of a result.

## Questions worth asking before you act

- What is this error **in m/s**?
- Compared to **what measurement**, at what tolerance?
- Would a **domain expert** call this a failure, or a Tuesday?
- Is the criterion I am about to discard data on **finer than the physics**?
