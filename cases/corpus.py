#!/usr/bin/env python3
"""
corpus.py -- the terrain corpus, and the splits that make a held-out claim
mean something.

Phase 16A built eight 5 x 5 km cases, one per fire. That is a catalogue: a
handful of domains to look at and solve. This is the other thing -- a corpus
to TRAIN on, and, more importantly, to hold terrain OUT of.

The paper's headline is a demonstration on unseen terrain. Whether it is a
demonstration or a self-portrait comes down entirely to whether the held-out
terrain was really held out, so the split logic gets more care here than the
size of the corpus does.

THREE WAYS TERRAIN LEAKS, and what is done about each.

1.  A WINDOW AND ITS NEIGHBOUR. Each site's tile is cut into overlapping
    5 km windows, which is how a corpus of a few dozen fires becomes a few
    hundred samples. Adjacent windows share half their ground, so a window
    in test whose neighbour is in train is not held out at all. Windows are
    therefore never split: a site's windows all land in the same fold, and
    the fold is chosen for the site, not the window.

2.  TWO FIRES ON THE SAME MOUNTAIN. This one is easy to miss, and the
    reference list is full of it -- Tubbs and Kincade are 10.3 km apart,
    Easy and Woolsey 15.1 km, Detwiler and Yosemite 15.1 km. Thomas and
    Woolsey, BOTH in the phase 16A catalogue, are 28.3 km apart in the same
    range. Treating fires as independent because they have different names
    would put the same ridgeline on both sides of the split. So sites are
    single-linkage clustered by great-circle distance first, and the fold
    is chosen for the CLUSTER.

3.  A WINDOW THAT REACHES INTO ANOTHER SITE. Two clusters could sit just
    over the clustering radius apart and still have windows that physically
    overlap, since a window centre can be a corner offset away from its
    site. assert_no_leakage computes that geometry rather than assuming the
    radius covers it, so changing the stride cannot quietly break the split.

WHAT IS NOT A LEAK, and would look like one. Every window has its own
vertical grid, because the floor follows its own relief -- so no two samples
share a z_cc, and phase 16A worried that this made them unstackable. It does
not matter here: the surrogate trains on levels at fixed heights ABOVE
GROUND (see python/fastwindterrain/levels.py), which have the same shape and
the same physical meaning on every window whatever the absolute grid is. The
per-window grid is needed only to reconstruct 3D, which happens one sample
at a time anyway.

Usage:

    import corpus
    sites  = corpus.candidate_sites()              # from the reference CSV
    groups = corpus.cluster_sites(sites, 50.0)     # leakage-safe groups
    folds  = corpus.split_clusters(groups, seed=20)
    corpus.assert_no_leakage(sites, folds)         # the guard, not a comment

Building it needs the download (pip install ".[cases]"); everything above is
numpy and the standard library, so the split is testable offline.
"""

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import casegen                                              # noqa: E402

CORPUS_DIR = os.path.join(HERE, "corpus")
MANIFEST = os.path.join(HERE, "corpus_manifest.json")

# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

WINDOW_M = casegen.DOMAIN_M      # a window IS a phase 16A domain, 5 x 5 km
N_WINDOWS_PER_SIDE = 3           # 3 x 3 = 9 windows per site
WINDOW_STRIDE_M = 2500.0         # half a window, so neighbours share half

#: Width of the square each site's tile covers: the windows' own span.
CORPUS_EXTENT_M = (N_WINDOWS_PER_SIDE - 1) * WINDOW_STRIDE_M + WINDOW_M

# ---------------------------------------------------------------------------
# Wind
#
# The dataset is many terrain shapes crossed with a few wind directions, and
# NOT with several speeds -- see cases/linearity_study.py, which measures
# what the pipeline's algebra promises: with the terrain fixed, every field
# scales exactly with the reference speed. inflow scales with u_ref, O'Brien
# integrates a divergence that scales with it, the Poisson right-hand side
# scales with it, and alpha depends on terrain slope rather than on the flow.
# So a second speed would cost another full pass over the corpus and add a
# multiple of what is already there.
#
# That is a property of THIS operator, not of wind. Anisotropy's f_Ri and
# f_Fr hooks return 1 today; a stability or Froude correction would put the
# speed back into the problem and the study would have to be re-run.
# ---------------------------------------------------------------------------

#: 10 m/s at 80 m -- a hub-height anchor rather than the solver's own 10 m
#: default, which is a met-mast height and sits inside the steepest part of
#: the profile, where a small error in z moves the speed a long way.
#:
#: WHAT THE HEIGHT IS MEASURED FROM. Inflow's profile is terrain-following:
#: it evaluates the power law at ``z_agl = z_cc[k] - h[j][i]``, per column
#: (Source/Inflow.cpp:349). So this anchors 10 m/s at 80 m ABOVE GROUND
#: everywhere, which in the domain's lowest column is 80 m above the lowest
#: terrain. An inflow anchored to an absolute elevation instead -- the same
#: speed at a given height ASL whatever the ground below does -- is a
#: different profile and the solver has no mode for it.
REFERENCE_SPEED_MS = 10.0
REFERENCE_HEIGHT_M = 80.0

#: Eight directions, the conventional wind rose, as degrees the wind comes
#: FROM. Held identical for every window so that direction is a controlled
#: variable and what differs between samples is the ground.
#:
#: Worth knowing when counting samples: on a square domain the physics is
#: very nearly rotation-equivariant, so (terrain, 90 deg) and (terrain
#: rotated a quarter turn, 180 deg) are close to the same problem. The four
#: axis-aligned directions therefore act partly as a rotation augmentation
#: of the terrain rather than as four independent conditions. Eight
#: directions over 252 windows is 2016 solves but not 2016 independent
#: samples, and a paper should not claim otherwise.
WIND_DIRECTIONS = tuple(float(45 * k) for k in range(8))

#: The directions actually worth SOLVING. Half the rose, because the other
#: half is free.
#:
#: THE SOLVER IS EXACTLY ODD IN THE INFLOW. Reversing the wind negates the
#: entire field -- measured on ditch_fire:20, the corpus's steepest window:
#:
#:     u(  0 deg) vs -u(180 deg)   relative 8.58e-16
#:     u( 90 deg) vs -u(270 deg)   relative 1.33e-15
#:
#: which is round-off, not approximation. Every operator in the chain is
#: odd or invariant: the profile scales with u_ref, O'Brien integrates a
#: divergence linear in (u, v), alpha depends on |terrain slope| and not on
#: direction, the Poisson right-hand side and its correction both flip
#: sign, and the surface condition's rescale is a RATIO of speeds and so
#: leaves the sign alone.
#:
#: TWO CONSEQUENCES, and the second matters more than the first.
#:
#: 1. Generate four directions and derive the reverse by negation. Half the
#:    compute for the same information.
#: 2. **A dataset of 252 windows x 8 directions is 1008 independent
#:    samples, not 2016.** Quoting 2016 would be counting each sample
#:    twice, and it is the kind of claim a reader checks.
#:
#: This is a property of a mass-consistent operator with no advection and
#: no stability dependence. A fractional-step solver has neither of those
#: properties and will NOT be odd; re-measure before assuming it there.
INDEPENDENT_DIRECTIONS = tuple(float(45 * k) for k in range(4))


def reverse_of(field):
    """The field for the opposite wind direction, exactly.

    See INDEPENDENT_DIRECTIONS: the operator is odd in the inflow, so the
    reverse is the negation and needs no solve.
    """
    import numpy as np

    return -np.asarray(field)

#: Projection passes the corpus is generated at.
#:
#: Phase 17 froze this at 4 on Creek, where the outer iteration falls
#: geometrically at ~0.87 a pass from the first one. Across the corpus the
#: L-infinity divergence instead RISES for the first passes on some windows
#: and only then falls -- over 0, 1, 2, 4, 8, 16 and 24 passes:
#:
#:     ditch_fire:20    64% solid   .194 .227 .250 .259 .251 .203 .159
#:     kincade_fire:20  59% solid   .152 .175 .177 .161 .140 .089 .057
#:     bootleg_fire:02  18% solid   .056 .042 .037 .029 .019 .012 .009
#:
#: Four passes lands on ditch_fire's hump exactly. Nothing there is
#: divergent: both are below their starting value by 24 passes. 16 is past
#: the hump on the windows measured and is cheap enough to take.
#:
#: WHAT THIS IS NOT, since two plausible explanations were tested and both
#: are wrong:
#:
#: * Not the domain top. Raising it from 1000 m above the highest ground to
#:   4000 m -- atmosphere/relief 0.54 to 2.16 on ditch_fire:20 -- moves the
#:   final divergence 0.0086 to 0.0079, 8% for a 4x change, with every
#:   window monotone at every height.
#: * Not terrain occupancy. marshall_fire, at 137-223 m of relief and 17%
#:   solid, has windows that fail the same test.
#:
#: WHAT IT PROBABLY IS: the vertical stretching. Geometric grading makes
#: the grid metric terms first-order accurate where a uniform grid is
#: second-order, so a stretched column is simply a coarser discretisation
#: of the same problem. That is a deliberate trade -- stretching buys a lot
#: of cost for a little accuracy, and is why it is here -- not a defect.
#:
#: AND IT DOES NOT MATTER MUCH. Divergence is 1/s; multiplied by the cell
#: size it is a local velocity imbalance, and at dx = 50 m an L2 divergence
#: of 0.007 is 0.35 m/s. CFD practice runs at ~0.25 m/s and 20-30% error is
#: acceptable for turbulent atmospheric flow, so these differences sit at
#: the edge of the noise. NOTHING IS DROPPED FROM THE CORPUS ON THIS BASIS;
#: screening is for gross failure only -- see screen_window.
N_PROJECTIONS = 16

#: How wide to download. The vendored reader smooths the outer 20% of every
#: side (srtm_terrain_reader.py:281), so only 60% of a tile is untouched, and
#: that 60% has to contain the whole window span plus its halo. Solved for
#: below rather than guessed at, and asserted at import.
CORPUS_HALF_WIDTH_M = 9200.0


def untouched_interior_m(half_width=CORPUS_HALF_WIDTH_M):
    """Width of a tile the vendored border smoothing does not reach."""
    return 2.0 * half_width * (1.0 - 2.0 * casegen._SMOOTHED_FRACTION)


def required_interior_m(n_side=N_WINDOWS_PER_SIDE, stride=WINDOW_STRIDE_M,
                        window=WINDOW_M, halo=casegen.HALO_M):
    """How much untouched tile the windows and their halo actually need."""
    return (n_side - 1) * stride + window + 2.0 * halo


def assert_download_is_wide_enough(half_width=CORPUS_HALF_WIDTH_M,
                                   n_side=N_WINDOWS_PER_SIDE,
                                   stride=WINDOW_STRIDE_M):
    """Refuse a tile whose smoothed border would reach into a window.

    Failing this is not loud. _smooth_terrain_border blends a Gaussian into
    the outer fifth of the tile, so the result is terrain that is real in
    the middle and quietly flattened at the edges -- which is precisely
    where terrain-driven flow separates. Nothing downstream would notice.
    """
    have = untouched_interior_m(half_width)
    need = required_interior_m(n_side, stride)
    if have < need:
        raise ValueError(
            f"a tile of half-width {half_width:.0f} m leaves {have:.0f} m "
            f"untouched by the border smoothing, but {n_side} x {n_side} "
            f"windows of {WINDOW_M:.0f} m at a {stride:.0f} m stride plus "
            f"{casegen.HALO_M:.0f} m of halo need {need:.0f} m. Raise "
            f"CORPUS_HALF_WIDTH_M to at least "
            f"{need / (2.0 * (1.0 - 2.0 * casegen._SMOOTHED_FRACTION)):.0f} "
            f"m, or the outer windows would be built on smoothed terrain "
            f"without anything saying so.")
    return have, need


assert_download_is_wide_enough()


def window_offsets(n_side=N_WINDOWS_PER_SIDE, stride=WINDOW_STRIDE_M,
                   window=WINDOW_M, extent=CORPUS_EXTENT_M):
    """Where each window's lower-left corner sits in tile coordinates.

    Yields ``(iw, jw, x0, y0)``. The grid is centred on the tile, so window
    (1, 1) of a 3 x 3 is the phase 16A domain for that fire exactly.
    """
    span = (n_side - 1) * stride + window
    start = 0.5 * (extent - span)
    for jw in range(n_side):
        for iw in range(n_side):
            yield iw, jw, start + iw * stride, start + jw * stride


def window_id(slug, iw, jw):
    return f"{slug}:{iw}{jw}"


def max_window_offset_m(n_side=N_WINDOWS_PER_SIDE, stride=WINDOW_STRIDE_M):
    """Furthest a window CENTRE can sit from its site centre.

    The corner window, so the diagonal. Used by assert_no_leakage to turn a
    distance between two fires into a distance between their closest
    windows.
    """
    half = 0.5 * (n_side - 1) * stride
    return math.hypot(half, half)


def window_points(points, x0, y0, window=WINDOW_M, halo=casegen.HALO_M):
    """Cut one window out of a site's tile, in the window's own coordinates.

    Keeps `halo` metres beyond the window on each side, exactly as
    Case.download_points does for a single domain, so the IDW at the window
    boundary averages real neighbours rather than extrapolating from one
    side. Returns an (n, 3) array with x, y on [-halo, window + halo] and z
    still absolute metres above sea level.
    """
    import numpy as np

    p = np.asarray(points, dtype=np.float64)
    x, y = p[:, 0] - x0, p[:, 1] - y0
    keep = ((x >= -halo) & (x <= window + halo)
            & (y >= -halo) & (y <= window + halo))
    if not keep.any():
        raise ValueError(
            f"no terrain points inside the window at ({x0:.0f}, {y0:.0f}); "
            f"the tile spans x {p[:, 0].min():.0f}..{p[:, 0].max():.0f}, "
            f"y {p[:, 1].min():.0f}..{p[:, 1].max():.0f}")
    return np.column_stack([x[keep], y[keep], p[keep, 2]])


# ---------------------------------------------------------------------------
# Gridding and terrain descriptors
#
# Descriptors exist to answer one question a reviewer will ask: is the
# held-out terrain actually LIKE the training terrain, or is the test set an
# extrapolation dressed up as a generalisation test? Reporting the range each
# fold covers answers it in a table.
#
# Deliberately computed from a plain binning of the point cloud rather than
# from the solver's IDW field. They are a property of the ground, not of the
# interpolation, and this way they can be computed without building a solver
# -- which keeps the manifest independent of solver settings.
# ---------------------------------------------------------------------------

def grid_terrain(points, n=100, window=WINDOW_M):
    """Bin a window's point cloud onto its own (n, n) cell centres.

    SRTM1 is a 30 m raster and the cells are 50 m, so every cell holds two
    or three points and the mean is a fair cell value. Cells with no points
    -- which happens only at a corner where the projection leaves a gap --
    are filled from the global mean rather than left as NaN, since a
    descriptor is a summary and a hole in one would poison the lot.
    """
    import numpy as np

    p = np.asarray(points, dtype=np.float64)
    inside = ((p[:, 0] >= 0.0) & (p[:, 0] < window)
              & (p[:, 1] >= 0.0) & (p[:, 1] < window))
    p = p[inside]
    if p.shape[0] == 0:
        raise ValueError("no terrain points inside the window")

    dx = window / n
    i = np.clip((p[:, 0] / dx).astype(np.int64), 0, n - 1)
    j = np.clip((p[:, 1] / dx).astype(np.int64), 0, n - 1)
    flat = j * n + i

    total = np.zeros(n * n)
    count = np.zeros(n * n)
    np.add.at(total, flat, p[:, 2])
    np.add.at(count, flat, 1.0)

    z = np.where(count > 0, total / np.maximum(count, 1.0), np.nan)
    z = np.where(np.isnan(z), np.nanmean(z), z)
    return z.reshape(n, n)


def descriptors(z, dx=WINDOW_M / 100.0):
    """Six numbers summarising one window's ground.

    ``relief``      max minus min. The crudest and the most load-bearing:
                    it sets the vertical grid.
    ``std``         elevation spread. Separates one big slope (large relief,
                    modest std) from broken country.
    ``slope_mean``  mean |grad z|, the everyday steepness.
    ``slope_p95``   the steep tail, which is what pushes flow around rather
                    than over and is where a mass-consistent solve differs
                    most from a uniform one.
    ``tri``         terrain ruggedness index -- mean |z - mean(neighbours)|.
                    Small-scale roughness, largely independent of relief:
                    a smooth 1000 m ramp and a boulder field can share a
                    relief and never share a TRI.
    ``aniso``       eigenvalue ratio of the gradient covariance. 1 is
                    isotropic hummocks; large is parallel ridges, which
                    channel wind and are the interesting case.
    ``dslope_p95``  slope CHANGE per cell -- the Hessian norm times dx, so
                    a dimensionless "how much does the slope turn between
                    one cell and the next". Steepness and curvature are
                    different things and only this one is about the mesh:
                    a uniform 30-degree ramp is steep and perfectly
                    resolved, while a cliff edge turns the surface through
                    a large angle inside one 50 m cell and cannot be. That
                    is where a coarse mesh makes the wind behave oddly,
                    independently of how steep or how rough the ground is.
    ``dslope_max``  the worst cell, which is the cliff edge itself.
    """
    import numpy as np

    z = np.asarray(z, dtype=np.float64)
    gy, gx = np.gradient(z, dx)
    slope = np.hypot(gx, gy)

    # TRI over the eight neighbours, edges clamped.
    ny, nx = z.shape
    ip1 = np.minimum(np.arange(nx) + 1, nx - 1)
    im1 = np.maximum(np.arange(nx) - 1, 0)
    jp1 = np.minimum(np.arange(ny) + 1, ny - 1)
    jm1 = np.maximum(np.arange(ny) - 1, 0)
    neigh = (z[np.ix_(jm1, im1)] + z[np.ix_(jm1, range(nx))]
             + z[np.ix_(jm1, ip1)] + z[:, im1] + z[:, ip1]
             + z[np.ix_(jp1, im1)] + z[np.ix_(jp1, range(nx))]
             + z[np.ix_(jp1, ip1)]) / 8.0

    cov = np.array([[float(np.mean(gx * gx)), float(np.mean(gx * gy))],
                    [float(np.mean(gx * gy)), float(np.mean(gy * gy))]])
    eig = np.linalg.eigvalsh(cov)
    aniso = float(np.sqrt(max(eig[1], 0.0) / max(eig[0], 1e-30)))

    # Slope change per cell: the Hessian's Frobenius norm scaled by dx, so
    # the number is the angle-ish amount the surface turns from one cell to
    # the next. Dimensionless, and therefore a statement about the MESH as
    # much as about the ground -- which is the point of carrying it
    # alongside slope_p95.
    gxy, gxx = np.gradient(gx, dx)
    gyy, gyx = np.gradient(gy, dx)
    dslope = dx * np.sqrt(gxx ** 2 + 2.0 * (0.5 * (gxy + gyx)) ** 2
                          + gyy ** 2)

    return {
        "relief": float(z.max() - z.min()),
        "std": float(z.std()),
        "slope_mean": float(slope.mean()),
        "slope_p95": float(np.percentile(slope, 95.0)),
        "tri": float(np.mean(np.abs(z - neigh))),
        "aniso": aniso,
        "dslope_p95": float(np.percentile(dslope, 95.0)),
        "dslope_max": float(dslope.max()),
    }


DESCRIPTOR_KEYS = ("relief", "std", "slope_mean", "slope_p95", "tri", "aniso",
                   "dslope_p95", "dslope_max")


# ---------------------------------------------------------------------------
# Screening
#
# The reference CSV is taken verbatim, including rows that read oddly as fire
# records -- "Black Summer Fire, New Mexico" (Black Summer was Australian),
# and "Coastal Fire" at 32.73/-117.13, which is inside San Diego. Phase 16A
# chose eight by hand and sidestepped the question. A corpus cannot: it takes
# what the list gives it, so it has to be able to say no.
#
# The rejections below are about the GROUND, not about the fire record. A
# site is unusable if its terrain would not exercise the solver or would not
# be real.
# ---------------------------------------------------------------------------

# SCREENING IS A SPLIT-STAGE DECISION, NOT A SURVEY-STAGE ONE. The survey
# measures and records; every judgement about what is usable happens here,
# from the committed numbers. So a threshold can be argued about, changed and
# re-run without downloading eleven megabytes of SRTM twenty-nine times.

MIN_RELIEF_M = 60.0        # per WINDOW; below this the sample is a plate
MIN_WINDOWS = 5            # a site needs a majority of its nine to survive
MAX_SEA_FRACTION = 0.02    # ocean reads as exactly 0 m after _fix_srtm_zeros

#: Largest terrain-induced amplification a window may show before it is
#: treated as a failed solve rather than as flow.
#:
#: Speed-up over complex terrain does not normally exceed 20-30%, so 2.0 --
#: a doubling -- is far outside anything physical and catches only gross
#: failure. It is deliberately not set near the physical rule: the screen
#: is here to remove fields that are wrong, not fields that are unusual,
#: and a corpus trimmed to what was expected teaches a surrogate what was
#: expected.
#:
#: `Poisson.cpp:218` records the failure this is shaped for -- a 34.8 m/s
#: corrected wind from a 10 m/s inflow, from a lambda boundary condition
#: that interacted badly with the nodal divergence. No divergence norm
#: would have caught it.
MAX_SPEEDUP = 2.0


def screen_window(entry, solvability=None):
    """``(ok, reason)`` for one window: enough relief, and it must solve.

    RELIEF, and the granularity is the point. Screening a SITE on its tile's
    relief passes anything with one steep corner: Erskine Fire spans 81 m
    across its 10 km tile, which sounds like terrain, while eight of its
    nine 5 km windows hold 10 to 12 m and are plates. A window is what gets
    trained on, so a window is what gets judged.

    A sample with no relief produces flow barely distinguishable from the
    inflow profile. It teaches a surrogate nothing, and worse, it inflates
    every skill score measured against a flat baseline -- so a corpus full
    of them reports success it has not earned. 60 m over 5 km is a 1.2%
    grade: generous, and still enough to exclude a plate.

    THE SOLVE, when `solvability` is given -- that window's entry from
    build_corpus.py --solvability. GROSS FAILURE ONLY: a non-finite field,
    or an amplification past MAX_SPEEDUP that means the solve produced
    something other than flow.

    IT IS DELIBERATELY NOT A CONVERGENCE TEST, and the history matters
    because the obvious screen is the wrong one. "Did the L-infinity
    divergence fall across the projection" looked like a clean criterion
    and would have removed about a third of the corpus. It is a bad
    criterion for three independently checked reasons:

    * the differences are physically tiny. Divergence is 1/s; times the
      cell size it is a local velocity imbalance, and at dx = 50 m the
      worst windows sit at 0.3-0.4 m/s -- the same order as the ~0.25 m/s
      CFD practice runs at, and well inside the 20-30% that is acceptable
      for turbulent atmospheric flow. On genuinely steep ground the
      accepted band is wider still.
    * the norms disagree with each other. Over 1 to 16 passes ditch_fire's
      L-infinity humps while its L2 falls monotonically, and bootleg_fire
      does the exact opposite. Picking one and calling it convergence
      would bake in a choice nobody made.
    * it does not track anything real. The failures are not the steep
      windows (marshall_fire, 137 m of relief, has them), not the occupied
      ones, and no terrain descriptor separates them -- relief, slope,
      ruggedness and curvature all overlap between passing and failing.

    A corpus trimmed on a criterion that fine is a corpus trimmed to what
    was expected, which is the one thing a held-out terrain set must not
    be.
    """
    relief = float(entry["z_max"] - entry["z_min"])
    if relief <= 0.0:
        return False, "perfectly flat -- missing or nodata SRTM"
    if relief < MIN_RELIEF_M:
        return False, f"relief {relief:.0f} m is under {MIN_RELIEF_M:.0f} m"

    if solvability is not None:
        import math

        after = solvability.get("div_fe_after") or []
        n_bad = sum(1 for v in after if not math.isfinite(v))
        if n_bad:
            return (False,
                    f"the solve produced a non-finite field at {n_bad} of "
                    f"{len(after)} wind directions")

        speedup = [v for v in (solvability.get("speedup_max") or [])
                   if math.isfinite(v)]
        if speedup and max(speedup) > MAX_SPEEDUP:
            return (False,
                    f"wind speed reaches {max(speedup):.1f}x the "
                    f"undisturbed profile at the same height; terrain "
                    f"speed-up does not exceed about 1.3x, so this is a "
                    f"failed solve rather than flow")
    return True, "ok"


def screen_site(survey, solvability=None):
    """``(ok, reason, kept_window_ids)`` for one surveyed site.

    `solvability` is ``{window_id: entry}`` from build_corpus.py
    --solvability, or None to screen on geometry alone.

    Three ways a site is unusable, all of which the reference list contains
    or nearly contains:

    * **flat.** Zero relief is the nodata signature that cost a day in
      phase 16A: a window of missing SRTM becomes a perfect sea-level
      plain, smooth and plausible and entirely fictional.
    * **sea.** A coastal site is partly ocean, and the vendored reader
      clamps negatives to zero (``z = where(z < 0, 0, z)``), so open water
      arrives as an immaculate plane at exactly 0 m rather than as anything
      that announces itself. A domain that is one third sea is a domain
      where a third of the answer is trivial.
    * **too few windows left.** A site whose windows are mostly plates is
      not a held-out fire worth the name -- it would enter the test fold as
      one or two samples and be reported as if it were nine.
    """
    relief = float(survey.get("relief", 0.0))
    if relief <= 0.0:
        return False, "terrain is perfectly flat -- missing or nodata SRTM", []
    sea = float(survey.get("sea_fraction", 0.0))
    if sea > MAX_SEA_FRACTION:
        return (False,
                f"{100.0 * sea:.0f}% of the tile is at or below sea level; "
                f"the reader clamps water to a flat 0 m plane", [])

    kept, flat, stuck = [], 0, 0
    for w in survey.get("windows", []):
        solv = (solvability or {}).get(w["id"])
        ok, reason = screen_window(w, solvability=solv)
        if ok:
            kept.append(w["id"])
        elif "non-finite" in reason or "failed solve" in reason:
            stuck += 1
        else:
            flat += 1
    if len(kept) < MIN_WINDOWS:
        n = len(kept) + flat + stuck
        # Name the cause: a site of plates and a site the solver cannot
        # handle are different problems and want different responses.
        if flat >= stuck:
            why = (f"only {len(kept)} of {n} windows clear the "
                   f"{MIN_RELIEF_M:.0f} m relief minimum; the site is a "
                   f"plate with a corner")
        else:
            why = (f"only {len(kept)} of {n} windows solve -- {stuck} "
                   f"produced a non-finite or unphysical field")
        return False, why, []
    return True, "ok", kept


# ---------------------------------------------------------------------------
# Geography
# ---------------------------------------------------------------------------

EARTH_R_KM = 6371.0088

#: Two fires closer than this are treated as one piece of ground.
#:
#: Chosen from the list itself rather than as a round number. The reference
#: fires have a clear gap in nearest-neighbour distance: thirteen pairs fall
#: under 55 km (down to 10.3 km, Tubbs and Kincade in the same Mayacamas
#: range), and the next pair up is 80 km. 50 km sits in that gap, so the
#: clustering is not balanced on a knife edge -- moving it to 40 or 70
#: changes nothing.
CLUSTER_RADIUS_KM = 50.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance. Metres would be false precision at this range."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    h = (math.sin(0.5 * dp) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(0.5 * dl) ** 2)
    return 2.0 * EARTH_R_KM * math.asin(math.sqrt(min(1.0, h)))


def candidate_sites(path=casegen.REFERENCE_CSV):
    """Every row of the reference CSV as a Case, in file order.

    The corpus starts from all of them and screens, rather than starting
    from the phase 16A eight. Eight fires cannot support a claim about
    unseen terrain: with one held-out fire the result is an anecdote and
    with two it is two anecdotes.
    """
    ref = casegen.read_reference(path)
    return [casegen.Case(**row) for row in ref.values()]


def cluster_sites(sites, radius_km=CLUSTER_RADIUS_KM):
    """Single-linkage clusters of sites within `radius_km` of each other.

    Single linkage, not a fixed grid or a k-means: the property that has to
    hold is transitive. If A is near B and B is near C then A, B and C share
    ground even when A and C are far apart, and only single linkage puts all
    three in one group.

    Returns a list of lists of slugs, each inner list sorted, the outer list
    sorted by its first member -- so the result depends on the sites and the
    radius and not on the order they arrived in.
    """
    n = len(sites)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_km(sites[i].lat, sites[i].lon,
                             sites[j].lat, sites[j].lon)
            if d <= radius_km:
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(sites[i].slug)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])


# ---------------------------------------------------------------------------
# The split
# ---------------------------------------------------------------------------

FOLDS = ("train", "val", "test")
DEFAULT_FRACTIONS = {"train": 0.65, "val": 0.15, "test": 0.20}
DEFAULT_SEED = 20            # the phase number, so it is not a magic constant


def split_clusters(clusters, fractions=None, seed=DEFAULT_SEED):
    """Assign whole clusters to train / val / test.

    Clusters, never sites -- that is the entire point, and it is why this
    cannot be `sklearn.train_test_split`.

    Balanced by SITE count rather than cluster count, because the clusters
    are lopsided: three of them hold three fires each and eighteen hold one,
    so splitting on cluster count would put a third more terrain in one fold
    than intended. Each cluster goes, largest first, to whichever fold is
    furthest below its target share. Largest-first matters: assigning the
    three-site clusters last leaves no room to place them without
    overshooting.

    Deterministic given the seed, which shuffles ties. Returns
    ``{fold: [cluster, ...]}``.
    """
    import random

    fractions = dict(fractions or DEFAULT_FRACTIONS)
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    if set(fractions) != set(FOLDS):
        raise ValueError(f"fractions must name exactly {FOLDS}")

    total = sum(len(c) for c in clusters)
    if total == 0:
        raise ValueError("no sites to split")

    rng = random.Random(seed)
    order = list(clusters)
    rng.shuffle(order)                                  # break ties
    order.sort(key=len, reverse=True)                   # stable: largest first

    out = {f: [] for f in FOLDS}
    have = {f: 0 for f in FOLDS}
    for cluster in order:
        # Deficit relative to target, in sites. The fold that is furthest
        # behind takes the next cluster; ties go to the larger target, which
        # keeps train from starving on a small corpus.
        fold = min(FOLDS, key=lambda f: (have[f] - fractions[f] * total,
                                         -fractions[f]))
        out[fold].append(cluster)
        have[fold] += len(cluster)

    for fold in FOLDS:
        out[fold].sort(key=lambda c: c[0])
        if not out[fold]:
            raise ValueError(
                f"the {fold} fold came out empty: {len(clusters)} clusters "
                f"cannot be split {fractions}. Add sites or widen the "
                f"clustering radius.")
    return out


def fold_of(folds):
    """``{slug: fold}`` from ``{fold: [cluster, ...]}``."""
    out = {}
    for fold, clusters in folds.items():
        for cluster in clusters:
            for slug in cluster:
                if slug in out:
                    raise ValueError(
                        f"{slug} appears in both {out[slug]} and {fold}")
                out[slug] = fold
    return out


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------

def assert_no_leakage(sites, folds, radius_km=CLUSTER_RADIUS_KM,
                      n_side=N_WINDOWS_PER_SIDE, stride=WINDOW_STRIDE_M):
    """Refuse a split whose folds share ground. Four checks, all of them real.

    Written as an assertion rather than as a paragraph in the paper because
    the claim it protects -- that the demonstration terrain was unseen -- is
    the one a reader cannot verify and has to take on trust.

    1. every site lands in exactly one fold;
    2. no site is missing from the split;
    3. every cross-fold pair of sites is at least `radius_km` apart, which
       is what the clustering was supposed to guarantee and is cheap to
       confirm;
    4. no two windows in different folds physically overlap. This is the
       one that is not implied by (3): a window centre sits up to a corner
       offset from its site, so two sites just over the radius apart can
       still have overlapping windows. Computed from the geometry, so a
       change to the stride is caught here rather than in a result.

    Returns the closest cross-fold pair, as ``(km, slug, slug)`` -- worth
    printing, since "the nearest held-out fire is 83 km from anything
    trained on" is the sentence the claim rests on.
    """
    by_slug = {s.slug: s for s in sites}
    assigned = fold_of(folds)

    missing = sorted(set(by_slug) - set(assigned))
    if missing:
        raise ValueError(f"these sites are in no fold: {missing}")
    unknown = sorted(set(assigned) - set(by_slug))
    if unknown:
        raise ValueError(f"the split names sites that do not exist: {unknown}")

    closest = None
    for a in by_slug.values():
        for b in by_slug.values():
            if a.slug >= b.slug or assigned[a.slug] == assigned[b.slug]:
                continue
            d = haversine_km(a.lat, a.lon, b.lat, b.lon)
            if closest is None or d < closest[0]:
                closest = (d, a.slug, b.slug)

    if closest is None:
        raise ValueError("every site is in the same fold; nothing is held out")

    d, sa, sb = closest
    if d < radius_km:
        raise ValueError(
            f"{sa} ({assigned[sa]}) and {sb} ({assigned[sb]}) are "
            f"{d:.1f} km apart, inside the {radius_km:.0f} km clustering "
            f"radius, so they are in different folds and on the same "
            f"ground. The clustering did not do its job.")

    # (4) The window geometry. Two windows overlap when their centres are
    # closer than one window width in BOTH x and y; the safe separation is
    # therefore a window width, and the worst case puts each centre a corner
    # offset from its site.
    reach_km = 0.001 * (2.0 * max_window_offset_m(n_side, stride) + WINDOW_M)
    if d < reach_km:
        raise ValueError(
            f"{sa} ({assigned[sa]}) and {sb} ({assigned[sb]}) are "
            f"{d:.1f} km apart, and with {n_side} x {n_side} windows at a "
            f"{stride:.0f} m stride a window can sit "
            f"{max_window_offset_m(n_side, stride):.0f} m from its site "
            f"centre -- so windows in different folds can end up "
            f"{d - 0.002 * max_window_offset_m(n_side, stride):.1f} km "
            f"apart, closer than the {0.001 * WINDOW_M:.1f} km they are "
            f"wide, and OVERLAP. Reduce the stride, or raise the clustering "
            f"radius above {reach_km:.1f} km.")

    return closest


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def site_dir(slug):
    return os.path.join(CORPUS_DIR, slug)


def tile_path(slug):
    return os.path.join(site_dir(slug), "tile.csv")


def site_survey_path(slug):
    return os.path.join(site_dir(slug), "survey.json")


def read_site_surveys(corpus_dir=CORPUS_DIR):
    """Every surveyed site under `corpus_dir`, keyed by slug. No network."""
    out = {}
    if not os.path.isdir(corpus_dir):
        return out
    for slug in sorted(os.listdir(corpus_dir)):
        path = os.path.join(corpus_dir, slug, "survey.json")
        if os.path.isfile(path):
            with open(path) as f:
                out[slug] = json.load(f)
    return out


def load_manifest(path=MANIFEST):
    with open(path) as f:
        return json.load(f)


def windows_in(manifest, fold):
    """Every window id in one fold, in manifest order."""
    return [w["id"] for w in manifest["windows"] if w["fold"] == fold]


def window_config(manifest, window_id_, wind_speed=REFERENCE_SPEED_MS,
                  wind_direction=225.0, z_ref=REFERENCE_HEIGHT_M, **kwargs):
    """A ready-to-solve fwt.Solver config for one window of the corpus.

    Reads the site's tile, cuts the window out of it, and derives the grid
    from the window's OWN relief -- not the site's, since a 3 x 3 grid of
    windows over broken country can differ by hundreds of metres corner to
    corner and a floor set from the wrong one is the silent all-fluid
    failure that casegen.assert_fits exists to catch.
    """
    import numpy as np

    entry = next((w for w in manifest["windows"] if w["id"] == window_id_),
                 None)
    if entry is None:
        raise KeyError(f"{window_id_!r} is not in the manifest")

    slug = entry["site"]
    path = tile_path(slug)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} does not exist. Tiles are large and are not committed "
            f"-- run\n    python3 cases/build_corpus.py --survey "
            f"--only {slug}\nto download it.")

    points = window_points(casegen.read_terrain(path),
                           entry["x0"], entry["y0"])
    grid = casegen.grid_from_relief(entry["z_min"], entry["z_max"])
    casegen.assert_fits(np.asarray(points)[:, 2],
                        grid["prob_lo"][2], grid["prob_hi"][2],
                        what=f"{window_id_} terrain")

    theta = math.radians(wind_direction)
    cfg = {
        "grid": grid,
        "terrain": {"points": np.asarray(points, dtype=np.float64)},
        "inflow": {"mode": "powerlaw",
                   "u_ref": -wind_speed * math.sin(theta),
                   "v_ref": -wind_speed * math.cos(theta),
                   "z_ref": z_ref},
        "anisotropy": {"enable": True},
        "obrien": {"enable": True},
        "poisson": {"alpha_v": 0.5, "n_projections": N_PROJECTIONS},
    }
    for name, section in kwargs.items():
        cfg.setdefault(name, {}).update(section)
    return cfg


def summarise(manifest, stream=None):
    """Print what the split holds out, in the terms the claim needs."""
    out = stream if stream is not None else sys.stdout
    print(f"corpus: {len(manifest['sites'])} sites, "
          f"{len(manifest['windows'])} windows, "
          f"{len(manifest['clusters'])} clusters at "
          f"{manifest['cluster_radius_km']:.0f} km", file=out)
    if manifest.get("rejected"):
        print(f"  rejected {len(manifest['rejected'])} site(s):", file=out)
        for slug, reason in sorted(manifest["rejected"].items()):
            print(f"    {slug:32s} {reason}", file=out)
    if manifest.get("dropped_windows"):
        print(f"  dropped {len(manifest['dropped_windows'])} window(s) from "
              f"otherwise usable sites:", file=out)
        for wid, reason in sorted(manifest["dropped_windows"].items()):
            print(f"    {wid:32s} {reason}", file=out)
    print(file=out)
    for fold in FOLDS:
        sites = [s for s, f in manifest["fold_of"].items() if f == fold]
        wins = windows_in(manifest, fold)
        print(f"  {fold:6s} {len(sites):3d} sites  {len(wins):4d} windows  "
              f"({100.0 * len(wins) / max(1, len(manifest['windows'])):4.1f}%)",
              file=out)
        print(f"         {', '.join(sorted(sites))}", file=out)
    if manifest.get("closest_cross_fold"):
        d, a, b = manifest["closest_cross_fold"]
        print(f"\n  closest cross-fold pair: {a} and {b}, {d:.1f} km apart",
              file=out)
