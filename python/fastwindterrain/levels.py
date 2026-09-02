"""
Levels: extracting 2D horizontal slices, and stitching them back into 3D.

The surrogate this exists for predicts wind on a handful of horizontal
levels and reconstructs the full 3D field from them, because training in
3D is expensive. This module is the two halves of that -- the extraction
that builds the training targets, and the stitching that puts a
prediction back together -- with no machine learning anywhere in it.

Keeping them here rather than in the training code matters: the SAME
stitching operator has to run when measuring what reconstruction costs on
known-perfect data and when reconstructing a prediction. If those were two
implementations, the measured ceiling would not bound the real thing.

TWO FRAMES, AND THE CHOICE MATTERS.

    agl        levels at a fixed height ABOVE GROUND. Every column has the
               same number of values and none of them are underground, so
               the slice is dense and smooth.
    cartesian  levels at a fixed absolute elevation. The slice cuts
               through terrain, so part of it is inside the rock and the
               holes move as the terrain changes.

The solver's grid is Cartesian either way -- this is only about where the
slices are taken. Engineering heights (10 m for a met mast, 80-160 m for
hub heights) are quoted above ground, which is an argument for ``agl``,
but the comparison is an experiment rather than an opinion and both are
implemented so it can be run.

WHAT IS NOT RESOLVED. With ``dz0 = 4 m`` the first fluid cell above
terrain sits somewhere between 0 and 4 m above the surface, so a 2 m
value is sub-grid in most columns: it is diagnosed from a log law rather
than interpolated, and ``DIAGNOSTIC_LEVEL`` is kept separate from
``ENGINEERING_LEVELS`` for that reason.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ALOFT_LEVELS",
    "BAND_BASE_M",
    "BAND_TOP_M",
    "DEFAULT_LEVELS",
    "RECOMMENDED_LEVELS",
    "DIAGNOSTIC_LEVEL",
    "ENGINEERING_LEVELS",
    "extract_levels",
    "first_fluid_k",
    "height_above_ground",
    "log_law",
    "max_agl",
    "obrien_w",
    "recommended_levels",
    "stitch_levels",
    "surface_kinematic_w",
]

#: Heights people ask for, in metres above ground: a met mast at 10 m and
#: the hub-height band at 80-160 m.
ENGINEERING_LEVELS = (10.0, 80.0, 100.0, 120.0, 160.0)

#: Levels aloft. Without them the column above 160 m is unconstrained --
#: on the catalogue's grid the engineering levels occupy only the bottom
#: third of it -- and the reconstruction there would be an extrapolation
#: rather than an interpolation.
ALOFT_LEVELS = (300.0, 600.0, 1200.0)

#: The obvious set: the requested heights plus something aloft. Kept
#: because the placement study compares against it, and because it is what
#: anyone would reach for first -- but see RECOMMENDED_LEVELS, which beats
#: it on both the column and the band.
DEFAULT_LEVELS = ENGINEERING_LEVELS + ALOFT_LEVELS

#: What the placement and split studies actually land on, and the set to
#: use: five levels octave-spaced across 10-160 m, three log-spaced above.
#:
#: The band levels are exact octaves -- ``geomspace(10, 160, 5)`` has a
#: ratio of 2 -- and the set still contains 10, 80 and 160 m outright.
#:
#: It beats DEFAULT_LEVELS on the band by roughly two to one, because
#: anchoring on 80/100/120/160 leaves nothing between 10 and 80 m, which
#: is where the shear is. The heights an answer is wanted at are not the
#: heights samples should be taken at. See docs/surrogate.rst for the
#: measurements behind that.
RECOMMENDED_LEVELS = (10.0, 20.0, 40.0, 80.0, 160.0, 345.0, 743.0, 1600.0)

#: The band the engineering levels live in, and the base of the set.
BAND_BASE_M = 10.0
BAND_TOP_M = 160.0


def recommended_levels(top_agl, n_band=5, n_aloft=3, base=BAND_BASE_M,
                       band_top=BAND_TOP_M):
    """The recommended set, with its TOP LEVEL SCALED TO THE COLUMN.

    ``RECOMMENDED_LEVELS`` is this with ``top_agl = 1600`` and is kept for
    the studies that produced it -- but as a fixed tuple it is a bug on any
    domain taller than the one it was tuned on.

    WHY IT HAS TO SCALE. The set was measured on Creek (1128 m of relief),
    where a column of relief + 1000 m is about 2100 m and a top level at
    1600 m leaves little above it. The terrain corpus reaches 1970 m of
    relief, so its tallest columns are near 3000 m and a fixed 1600 m top
    leaves 1200 m of column reconstructed by holding the top value
    constant.

    That is not a small effect. On ``ditch_fire:20`` (1850 m relief), error
    by height band, with and without two more levels at 2400 and 3200 m:

        band          top 1600 m    + 2400, 3200
        0-50 m           0.0214        0.0214
        50-200 m         0.0105        0.0105
        200-500 m        0.0080        0.0080
        500-1000 m       0.0042        0.0042
        1000-1600 m      0.0029        0.0029
        1600 m +         0.0197        0.0013     <- 15x

    Every other band is untouched, which is what makes the diagnosis
    certain: the aloft levels are the only thing that changed.

    ``top_agl`` should be the largest height above ground in the domain --
    ``prob_hi[2]`` minus the LOWEST terrain, since that is the deepest
    column and the one with the most to reconstruct.

    Structure, preserved from the tuned set: ``n_band`` levels geometrically
    spaced across 10-160 m (exact octaves at the default of 5, and the set
    still contains 10, 80 and 160 m outright), then ``n_aloft`` continuing
    geometrically from 160 m to ``top_agl``.
    """
    if top_agl <= band_top:
        raise ValueError(
            f"top_agl must be above the {band_top:.0f} m band top, got "
            f"{top_agl}. A domain that shallow does not need aloft levels; "
            f"pass the engineering band directly.")
    band = np.geomspace(base, band_top, n_band)
    aloft = np.geomspace(band_top, top_agl, n_aloft + 1)[1:]
    return tuple(float(v) for v in np.concatenate([band, aloft]))


def max_agl(z_cc, z_terrain):
    """The deepest column's height above ground -- what to pass as top_agl."""
    zt = np.asarray(z_terrain)
    if zt.ndim == 3:
        zt = zt[0]
    return float(np.asarray(z_cc)[-1] - zt.min())

#: Pedestrian height. Sub-grid, so it is diagnosed from a log law and is
#: deliberately not part of the level set the surrogate predicts.
DIAGNOSTIC_LEVEL = 2.0


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def height_above_ground(z_cc, z_terrain):
    """``(nz, ny, nx)`` height of each cell centre above the ground below it.

    Negative inside the terrain, which is how a caller can tell without
    consulting the mask.
    """
    z_cc = np.asarray(z_cc)
    zt = np.asarray(z_terrain)
    if zt.ndim == 3:
        zt = zt[0]                      # the solver replicates it along k
    return z_cc[:, None, None] - zt[None, :, :]


def first_fluid_k(mask):
    """``(ny, nx)`` index of the lowest fluid cell in each column.

    ``nz`` where a column is solid all the way up, which cannot happen on
    a well-posed case but is what an all-solid mask would mean -- see the
    domain-fit guards in ``cases/casegen.py``.
    """
    mask = np.asarray(mask)
    fluid = mask == 0
    nz = mask.shape[0]
    return np.where(fluid.any(axis=0), fluid.argmax(axis=0), nz)


def log_law(u_ref, z_ref, z, z0=0.1):
    """Neutral surface-layer scaling of a speed from one height to another.

    ``u(z) = u_ref * ln((z + z0)/z0) / ln((z_ref + z0)/z0)``

    The ``+ z0`` keeps it finite at the ground instead of diverging to
    minus infinity, which matters because this is evaluated at cell
    centres that can sit a few centimetres above the surface.
    """
    z = np.asarray(z, dtype=np.float64)
    num = np.log(np.maximum(z, 0.0) / z0 + 1.0)
    den = np.log(z_ref / z0 + 1.0)
    return u_ref * (num / den)


# ---------------------------------------------------------------------------
# Extraction: 3D -> a stack of 2D levels
# ---------------------------------------------------------------------------

def extract_levels(field, z_cc, z_terrain, levels, mask=None, frame="agl",
                   method="linear", z0=0.1):
    """Sample a ``(nz, ny, nx)`` field on ``levels``, giving ``(nlev, ny, nx)``.

    ``frame='agl'`` reads ``levels`` as heights above ground and
    ``frame='cartesian'`` as absolute elevations.

    Interpolation is between cell centres in ``z``. Where the target falls
    below the lowest FLUID cell centre in a column -- which is common for
    a 10 m level on a 4 m grid, since the first fluid centre can sit
    higher than that -- the value is taken from the log law anchored at
    that cell rather than by extrapolating a straight line into the
    ground.

    Passing ``mask`` is what makes that possible; without it the lowest
    cell centre in the column is used, which is inside the terrain over
    real ground.
    """
    field = np.asarray(field, dtype=np.float64)
    z_cc = np.asarray(z_cc, dtype=np.float64)
    nz, ny, nx = field.shape
    levels = np.atleast_1d(np.asarray(levels, dtype=np.float64))

    zt = np.asarray(z_terrain, dtype=np.float64)
    if zt.ndim == 3:
        zt = zt[0]

    k0 = (first_fluid_k(mask) if mask is not None
          else np.zeros((ny, nx), dtype=np.int64))
    k0 = np.minimum(k0, nz - 1)

    out = np.empty((levels.size, ny, nx), dtype=np.float64)
    for n, level in enumerate(levels):
        z_target = (zt + level) if frame == "agl" else np.full((ny, nx), level)
        out[n] = _sample_column(field, z_cc, z_target, zt, k0, method, z0)
    return out


def _sample_column(field, z_cc, z_target, zt, k0, method, z0):
    """One level, interpolated in z per column."""
    nz, ny, nx = field.shape

    # searchsorted over the shared 1-D z_cc, with a per-column target --
    # the whole slice in one vectorised call rather than a Python loop
    # over 10 000 columns.
    ku = np.searchsorted(z_cc, z_target.ravel()).reshape(ny, nx)
    ku = np.clip(ku, 1, nz - 1)
    kl = ku - 1

    zl = z_cc[kl]
    zu = z_cc[ku]
    fl = np.take_along_axis(field, kl[None], axis=0)[0]
    fu = np.take_along_axis(field, ku[None], axis=0)[0]

    if method == "loglinear":
        agl_l = np.maximum(zl - zt, 0.0)
        agl_u = np.maximum(zu - zt, 0.0)
        agl_t = np.maximum(z_target - zt, 0.0)
        a = np.log(agl_l / z0 + 1.0)
        b = np.log(agl_u / z0 + 1.0)
        t = np.log(agl_t / z0 + 1.0)
        wgt = np.where(b > a, (t - a) / np.where(b > a, b - a, 1.0), 0.0)
    else:
        wgt = np.where(zu > zl, (z_target - zl) / np.where(zu > zl, zu - zl, 1.0),
                       0.0)

    value = fl + wgt * (fu - fl)

    # Below the lowest fluid cell centre: the log law from that cell,
    # rather than a straight line extrapolated into the rock.
    below = kl < k0
    if below.any():
        k_anchor = np.minimum(k0, nz - 1)
        anchor = np.take_along_axis(field, k_anchor[None], axis=0)[0]
        z_anchor = np.maximum(z_cc[k_anchor] - zt, 1e-6)
        z_want = np.maximum(z_target - zt, 0.0)
        value = np.where(below, log_law(anchor, z_anchor, z_want, z0), value)

    return value


# ---------------------------------------------------------------------------
# Stitching: a stack of 2D levels -> 3D
# ---------------------------------------------------------------------------

def stitch_levels(values, levels, z_cc, z_terrain, mask=None, frame="agl",
                  method="loglinear", z0=0.1, fill=0.0):
    """Rebuild a ``(nz, ny, nx)`` field from ``(nlev, ny, nx)`` level values.

    The inverse of :func:`extract_levels`, and the operator a surrogate's
    output passes through. Three regions, each handled differently
    because each is a different question:

    * **between levels** -- interpolate, linearly in ``z`` or linearly in
      ``log(z_agl/z0 + 1)``. The second is the one to beat near the
      surface, where the profile really is logarithmic.
    * **below the lowest level** -- the log law anchored at that level.
      Extrapolating a straight line down from 10 m puts a sign error in
      the bottom cells.
    * **above the highest level** -- hold the top value. Defensible only
      because ``ALOFT_LEVELS`` reaches 1200 m, where the terrain's
      influence has largely gone; with engineering levels alone this is
      an extrapolation over two thirds of the column.

    Solid cells are set to ``fill``. They are excluded from every error
    metric, and leaving stale values there invites reading them by
    accident.
    """
    values = np.asarray(values, dtype=np.float64)
    levels = np.atleast_1d(np.asarray(levels, dtype=np.float64))
    z_cc = np.asarray(z_cc, dtype=np.float64)
    if values.shape[0] != levels.size:
        raise ValueError(f"{values.shape[0]} level fields for {levels.size} "
                         f"levels")
    if not np.all(np.diff(levels) > 0):
        raise ValueError("levels must be strictly increasing")

    zt = np.asarray(z_terrain, dtype=np.float64)
    if zt.ndim == 3:
        zt = zt[0]
    nz = z_cc.size
    ny, nx = values.shape[1:]

    # The height of every cell, in the frame the levels are quoted in.
    if frame == "agl":
        coord = z_cc[:, None, None] - zt[None]
    else:
        coord = np.broadcast_to(z_cc[:, None, None], (nz, ny, nx)).copy()

    ku = np.clip(np.searchsorted(levels, coord.ravel()).reshape(coord.shape),
                 1, levels.size - 1)
    kl = ku - 1

    if method == "loglinear" and frame == "agl":
        a = np.log(np.maximum(levels[kl], 0.0) / z0 + 1.0)
        b = np.log(np.maximum(levels[ku], 0.0) / z0 + 1.0)
        t = np.log(np.maximum(coord, 0.0) / z0 + 1.0)
    else:
        a, b, t = levels[kl], levels[ku], coord
    wgt = np.clip((t - a) / np.where(b > a, b - a, 1.0), 0.0, 1.0)

    fl = np.take_along_axis(values, kl.reshape(nz, ny, nx), axis=0)
    fu = np.take_along_axis(values, ku.reshape(nz, ny, nx), axis=0)
    out = fl + wgt * (fu - fl)

    # Below the lowest level: the log law, anchored there.
    low = coord < levels[0]
    if low.any():
        out = np.where(low, log_law(values[0][None], levels[0],
                                    np.maximum(coord, 0.0), z0), out)

    # Above the highest: hold.
    high = coord > levels[-1]
    if high.any():
        out = np.where(high, values[-1][None], out)

    if mask is not None:
        out = np.where(np.asarray(mask) == 1, fill, out)
    return out


def surface_kinematic_w(u, v, z_terrain, dx, dy, mask):
    """``w = u.grad(h)`` in each column's first fluid cell, zero elsewhere.

    The kinematic condition for flow that follows the ground: air climbing
    a slope at speed u acquires a vertical velocity set by how steep the
    slope is. It is the natural seed for :func:`obrien_w`, whose
    integration has to start from *some* value at the surface.

    Whether it beats seeding with zero is terrain-dependent and measured
    rather than assumed -- see docs/surrogate.rst. On gentle ground it
    helps; on very steep ground it overestimates, because there the flow
    is being pushed AROUND the obstacle rather than over it, which is
    precisely what the solver's suppressed ``alpha_v`` encodes.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    zt = np.asarray(z_terrain, dtype=np.float64)
    if zt.ndim == 3:
        zt = zt[0]
    ny, nx = zt.shape

    ip1 = np.minimum(np.arange(nx) + 1, nx - 1)
    im1 = np.maximum(np.arange(nx) - 1, 0)
    jp1 = np.minimum(np.arange(ny) + 1, ny - 1)
    jm1 = np.maximum(np.arange(ny) - 1, 0)
    dhdx = (zt[:, ip1] - zt[:, im1]) / (2.0 * dx)
    dhdy = (zt[jp1, :] - zt[jm1, :]) / (2.0 * dy)

    k0 = np.minimum(first_fluid_k(mask), u.shape[0] - 1)
    u0 = np.take_along_axis(u, k0[None], axis=0)[0]
    v0 = np.take_along_axis(v, k0[None], axis=0)[0]

    out = np.zeros_like(u)
    np.put_along_axis(out, k0[None], (u0 * dhdx + v0 * dhdy)[None], axis=0)
    return out


# ---------------------------------------------------------------------------
# w from continuity: the O'Brien adjustment, in numpy
# ---------------------------------------------------------------------------

def obrien_w(u, v, w, dz, dx, dy, mask):
    """Rebuild ``w`` from ``u`` and ``v`` by column-integrated continuity.

    A numpy transcription of ``Obrien::Apply`` (``Source/Obrien.cpp:74``),
    exact rather than approximate, so it can be checked against the C++
    operator and then used where the C++ one cannot go -- inside a
    training loop, where it has to be differentiable.

    Two passes per column, from its first fluid cell ``k0`` to the top:

    1. integrate ``-div_h(u, v)`` upward and keep the residual ``E`` left
       at the top;
    2. integrate again, subtracting ``frac**2 * E`` as it goes, where
       ``frac`` runs 0 to 1 over the column.

    At the top ``frac = 1`` so ``w`` is exactly zero, and the quadratic
    weight puts the correction aloft where the divergence estimate is
    least trustworthy, leaving the near-surface values nearly untouched.

    ``w`` supplies the value in each column's first fluid cell, which the
    adjustment does not modify; everything above it is overwritten.

    The horizontal divergence is a plain central difference with indices
    clamped at the domain edge. That is not a simplification: with no
    advecting velocity every scheme in ``Derivatives.H`` falls through to
    ``Central2`` (``Derivatives.H:148``), so this matches whatever
    ``numerics.gradient_scheme`` is set to.

    AGREEMENT WITH THE C++ OPERATOR IS TO A FEW ULP, NOT BIT-EXACT. The
    two do the same operations in the same order, and the residual
    difference is at most 2 units in the last place, averages a quarter
    of one, does not accumulate up the column, and is exactly zero at the
    domain top. That is the signature of one rounding per step: clang
    contracts ``w -= Dh * dz[k]`` into a fused multiply-subtract, which
    rounds once where numpy rounds twice. numpy has no FMA, so it cannot
    be reproduced -- and it does not matter, since training runs in
    float32 where this is 8 orders of magnitude below the noise.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    out = np.array(w, dtype=np.float64, copy=True)
    dz = np.asarray(dz, dtype=np.float64)
    nz, ny, nx = u.shape

    ip1 = np.minimum(np.arange(nx) + 1, nx - 1)
    im1 = np.maximum(np.arange(nx) - 1, 0)
    jp1 = np.minimum(np.arange(ny) + 1, ny - 1)
    jm1 = np.maximum(np.arange(ny) - 1, 0)
    div_h = ((u[:, :, ip1] - u[:, :, im1]) / (2.0 * dx)
             + (v[:, jp1, :] - v[:, jm1, :]) / (2.0 * dy))

    k0 = first_fluid_k(mask)
    khi = nz - 1
    active = k0 < khi                      # C++ skips a column with no room

    kk = np.arange(nz)[:, None, None]
    above = (kk > k0[None]) & active[None]

    # Sequential accumulation, one k at a time, rather than a cumulative
    # sum: it is the order Obrien.cpp adds in, and matching it is what
    # makes the two agree to the last bit.
    start = np.take_along_axis(out, np.minimum(k0, khi)[None], axis=0)[0]

    acc = start.copy()
    for k in range(nz):
        acc = np.where(above[k], acc - div_h[k] * dz[k], acc)
    E = acc

    span = np.maximum((khi - k0).astype(np.float64), 1.0)
    acc = start.copy()
    for k in range(nz):
        step = acc - div_h[k] * dz[k]
        acc = np.where(above[k], step, acc)
        frac = (k - k0) / span
        out[k] = np.where(above[k], acc - frac * frac * E, out[k])

    return out
