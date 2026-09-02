"""
``terrain.extrapolation`` from Python.

IDW answers every query, including ones outside the data it was given.
Over a point cloud that lies entirely to one side that answer is a
distance-weighted average of far-away points -- smooth, and arbitrary --
and since the mask is only ``z_cc <= z_terrain``, a wrong height there
shows up as a column that is all fluid or all solid rather than as
anything resembling an error.

The dataset generators keep a margin of points beyond the domain
(``cases/casegen.py``'s ``HALO_M``), so in practice no column is outside
the cloud. These tests cover the case where that guarantee does not
hold: the count is exposed as ``Terrain.n_columns_outside``, and
``{"extrapolation": "nearest"}`` gives those columns the nearest input
point's elevation.

The reference heights are recomputed here in numpy, so the bindings are
not being checked against themselves.
"""

import numpy as np
import pytest

import fastwindterrain as fwt

# Most cases here deliberately build a terrain the cloud does not cover,
# so the coverage warning is the expected outcome and not a signal. It is
# silenced at module scope to keep the run's warning summary meaningful;
# ``pytest.warns`` overrides the filter, so the one test that asserts the
# warning still sees it.
pytestmark = pytest.mark.filterwarnings("ignore:WARNING .Terrain.:UserWarning")

NX, NY, NZ = 24, 24, 40
GRID = {
    "n_cell": (NX, NY, NZ),
    "prob_lo": (0.0, 0.0, 0.0),
    "prob_hi": (1000.0, 1000.0, 483.19909696997223),
    "dz0": 4.0,
    "stretching_ratio": 1.05,
    "max_grid_size": 16,
}

DX = 1000.0 / NX
DY = 1000.0 / NY
XC = (np.arange(NX) + 0.5) * DX          # column centres
YC = (np.arange(NY) + 0.5) * DY

IDW_K = 6
IDW_P = 2.0
DISTANCE_EPSILON = 1.0e-12


# ---------------------------------------------------------------------------
# Point clouds and the reference interpolations
# ---------------------------------------------------------------------------

def plane(x, y):
    """A constant-gradient surface. Steep enough that the six nearest
    points to a far column span a real range of heights, so the k-nearest
    average and the single nearest point cannot coincide by accident."""
    return 5.0 + 0.25 * x + 0.15 * y


def cloud(xhi, yhi, n=21):
    """``(n*n, 3)`` samples of `plane` on ``[0, xhi] x [0, yhi]``."""
    xs, ys = np.meshgrid(np.linspace(0.0, xhi, n),
                         np.linspace(0.0, yhi, n), indexing="ij")
    xs, ys = xs.ravel(), ys.ravel()
    return np.column_stack([xs, ys, plane(xs, ys)])


def idw(xq, yq, pts, k=IDW_K, exponent=IDW_P):
    """Terrain::InterpolateIDW, re-derived."""
    d2 = (pts[:, 0] - xq) ** 2 + (pts[:, 1] - yq) ** 2
    order = np.argsort(d2, kind="stable")[:min(k, len(pts))]
    if d2[order[0]] < DISTANCE_EPSILON:
        return pts[order[0], 2]
    w = d2[order] ** (-exponent / 2.0)
    return float((w * pts[order, 2]).sum() / w.sum())


def nearest(xq, yq, pts):
    """Terrain::NearestElevation, re-derived. ``argmin`` takes the first
    of any tie, which is the same rule."""
    d2 = (pts[:, 0] - xq) ** 2 + (pts[:, 1] - yq) ** 2
    return float(pts[int(np.argmin(d2)), 2])


def outside_mask(pts):
    """``(NY, NX)`` -- which columns lie strictly outside the cloud's
    axis-aligned extent. Strictly, so a column exactly on the extent is
    bracketed by data and is interpolated."""
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    ox = (XC < x_min) | (XC > x_max)
    oy = (YC < y_min) | (YC > y_max)
    return oy[:, None] | ox[None, :]


def heights(terrain):
    """``(NY, NX)`` column heights. ``z_terrain`` is replicated along k,
    so any k does."""
    return np.asarray(terrain.z_terrain)[0]


# ---------------------------------------------------------------------------
# A cloud that covers the domain: the option must change nothing
# ---------------------------------------------------------------------------

def test_covered_terrain_is_untouched_by_the_option(amrex):
    """The guarantee that makes the default safe to leave alone: where
    the data covers the domain, `nearest` is not merely close to `idw`,
    it is the same array."""
    g = fwt.Grid(GRID)
    pts = cloud(1000.0, 1000.0)
    assert not outside_mask(pts).any()

    t_idw = fwt.Terrain(g, {"points": pts})
    t_near = fwt.Terrain(g, {"points": pts, "extrapolation": "nearest"})

    assert t_idw.n_columns_outside == 0
    assert t_near.n_columns_outside == 0

    # Bit for bit -- the fallback never ran, so there is no round-off to
    # allow for.
    assert np.array_equal(heights(t_idw), heights(t_near))
    assert np.array_equal(np.asarray(t_idw.mask), np.asarray(t_near.mask))
    assert t_idw.n_solid == t_near.n_solid > 0


def test_the_default_is_idw(amrex):
    """Not asking for a mode must be the same as asking for the one the
    code has always used."""
    g = fwt.Grid(GRID)
    pts = cloud(400.0, 600.0)

    t_default = fwt.Terrain(g, {"points": pts})
    t_explicit = fwt.Terrain(g, {"points": pts, "extrapolation": "idw"})

    assert np.array_equal(heights(t_default), heights(t_explicit))


# ---------------------------------------------------------------------------
# A cloud smaller than the domain
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def partial_points():
    """A plane sampled on [0, 400] x [0, 600] of a 1000 x 1000 m
    domain."""
    return cloud(400.0, 600.0)


def test_uncovered_columns_are_counted(amrex, partial_points):
    """The count is a property of the data, not of the mode: it is what
    tells you the cloud is short, whichever fallback is in force."""
    g = fwt.Grid(GRID)
    expect = int(outside_mask(partial_points).sum())
    assert 0 < expect < NX * NY

    for mode in ("idw", "nearest"):
        t = fwt.Terrain(g, {"points": partial_points, "extrapolation": mode})
        assert t.n_columns_outside == expect, mode


def test_idw_still_extrapolates_everywhere(amrex, partial_points):
    """The default is bit-for-bit what it was before the option existed:
    a plain IDW at every column, covered or not."""
    g = fwt.Grid(GRID)
    h = heights(fwt.Terrain(g, {"points": partial_points}))

    expect = np.array([[idw(x, y, partial_points) for x in XC] for y in YC])
    assert np.allclose(h, expect, rtol=0.0, atol=1.0e-9)


def test_nearest_replaces_only_the_uncovered_columns(amrex, partial_points):
    """Outside the cloud, the nearest point's elevation exactly. Inside
    it, the IDW, unchanged -- the fallback is for extrapolation, not a
    different interpolation."""
    g = fwt.Grid(GRID)
    out = outside_mask(partial_points)

    h_idw = heights(fwt.Terrain(g, {"points": partial_points}))
    h_near = heights(fwt.Terrain(
        g, {"points": partial_points, "extrapolation": "nearest"}))

    # Inside: not merely close, identical.
    assert np.array_equal(h_near[~out], h_idw[~out])

    expect = np.array([[nearest(x, y, partial_points) for x in XC]
                       for y in YC])
    assert np.array_equal(h_near[out], expect[out])

    # And the fallback must actually do something: a rule that returned
    # the IDW value would satisfy everything above by accident.
    moved = np.abs(h_near[out] - h_idw[out])
    assert (moved > 1.0e-9).all()
    assert moved.max() > 1.0


def test_the_mask_follows_the_replaced_heights(amrex, partial_points):
    """Why any of this matters: the mask is only z_cc <= z_terrain, so a
    height chosen out there decides whether the column is open."""
    g = fwt.Grid(GRID)
    t_idw = fwt.Terrain(g, {"points": partial_points})
    t_near = fwt.Terrain(
        g, {"points": partial_points, "extrapolation": "nearest"})

    assert t_idw.n_solid != t_near.n_solid


def test_a_short_cloud_warns_under_the_default(amrex, partial_points):
    """The failure this option exists for used to be silent. Under the
    default it now reaches Python's warnings machinery, where a notebook
    can promote it to an error."""
    g = fwt.Grid(GRID)

    with pytest.warns(UserWarning, match="does not cover the domain"):
        fwt.Terrain(g, {"points": partial_points})


def test_flat_ground_has_no_columns_outside(amrex):
    """No point cloud, nothing to be outside of."""
    g = fwt.Grid(GRID)
    t = fwt.Terrain(g, {"flat_elevation": 12.0, "extrapolation": "nearest"})
    assert t.n_columns_outside == 0
    assert heights(t).min() == heights(t).max() == 12.0


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def test_an_unknown_mode_raises(amrex, partial_points):
    """Rejected by name rather than falling back to the default, and
    before any interpolation has run."""
    g = fwt.Grid(GRID)
    with pytest.raises((ValueError, TypeError), match="corner_average"):
        fwt.Terrain(g, {"points": partial_points,
                        "extrapolation": "corner_average"})
