"""
Level extraction and 2D->3D stitching (fastwindterrain.levels).

The surrogate predicts wind on a few horizontal levels and reconstructs
the 3D field from them. These are the two operators that makes possible,
and they are tested here without any machine learning -- which is the
point, since the reconstruction error they carry is a ceiling on anything
built on top of them.

The load-bearing test is test_obrien_matches_the_cpp_operator. The C++
O'Brien is not callable from Python on an arbitrary field, so the numpy
transcription is checked a different way: run the same case with the
adjustment off and on, and require the numpy version applied to the first
to reproduce the second EXACTLY. Solver.cpp:113 copies velocity0 after
O'Brien has run, which is what makes that comparison possible.
"""

import numpy as np
import pytest

import fastwindterrain as fwt
from fastwindterrain import levels as lv


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def test_first_fluid_k_finds_the_lowest_fluid_cell():
    mask = np.ones((6, 2, 3), dtype=np.int32)
    mask[3:, 0, 0] = 0
    mask[1:, 1, 2] = 0
    assert lv.first_fluid_k(mask)[0, 0] == 3
    assert lv.first_fluid_k(mask)[1, 2] == 1
    # A column that is solid all the way up reports nz, not 0.
    assert lv.first_fluid_k(mask)[0, 1] == 6


def test_height_above_ground_is_negative_inside_the_terrain():
    z_cc = np.array([1.0, 3.0, 5.0])
    zt = np.array([[2.0]])
    agl = lv.height_above_ground(z_cc, zt)
    assert agl.shape == (3, 1, 1)
    assert agl[0, 0, 0] == -1.0 and agl[2, 0, 0] == 3.0


def test_log_law_is_finite_at_the_ground_and_matches_at_the_reference():
    assert lv.log_law(8.0, 10.0, 10.0) == pytest.approx(8.0)
    assert np.isfinite(lv.log_law(8.0, 10.0, 0.0))
    assert lv.log_law(8.0, 10.0, 0.0) == 0.0
    # Monotone in height, as a neutral surface layer must be.
    z = np.array([0.5, 2.0, 10.0, 50.0])
    assert np.all(np.diff(lv.log_law(8.0, 10.0, z)) > 0)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

@pytest.fixture
def column():
    """A single column, 20 stretched cells, ground at 100 m."""
    dz = 4.0 * 1.05 ** np.arange(20)
    z_face = np.concatenate([[100.0], 100.0 + np.cumsum(dz)])
    z_cc = 0.5 * (z_face[:-1] + z_face[1:])
    zt = np.array([[100.0]])
    mask = np.zeros((20, 1, 1), dtype=np.int32)
    return z_cc, zt, mask


def test_extract_at_a_cell_centre_returns_that_cell(column):
    z_cc, zt, mask = column
    field = np.arange(20, dtype=np.float64)[:, None, None]
    agl = z_cc[7] - zt[0, 0]
    got = lv.extract_levels(field, z_cc, zt, [agl], mask=mask)
    assert got[0, 0, 0] == pytest.approx(7.0, abs=1e-12)


def test_extract_recovers_a_linear_profile_exactly(column):
    z_cc, zt, mask = column
    field = (2.5 * (z_cc - zt[0, 0]) + 1.0)[:, None, None]
    got = lv.extract_levels(field, z_cc, zt, [10.0, 40.0, 80.0], mask=mask,
                            method="linear")
    want = 2.5 * np.array([10.0, 40.0, 80.0]) + 1.0
    assert np.allclose(got[:, 0, 0], want, rtol=0, atol=1e-9)


def test_extract_below_the_first_fluid_cell_uses_the_log_law(column):
    """A 10 m level on a 4 m grid can sit below the lowest fluid cell
    centre once terrain cuts the column. Extrapolating a straight line
    down from two cells above puts a sign error in the bottom cells; the
    log law cannot."""
    z_cc, zt, _ = column
    mask = np.zeros((20, 1, 1), dtype=np.int32)
    mask[:4] = 1                                # ground up to the 5th cell
    field = np.linspace(5.0, 15.0, 20)[:, None, None]
    got = lv.extract_levels(field, z_cc, zt, [2.0], mask=mask)
    anchor = field[4, 0, 0]
    assert 0.0 < got[0, 0, 0] < anchor          # slower than at the anchor
    assert got[0, 0, 0] > 0.0                   # and not extrapolated negative


def test_cartesian_and_agl_frames_differ_over_terrain():
    """The two frames are the experiment; if they agreed there would be
    nothing to choose between them."""
    z_cc = np.linspace(5.0, 195.0, 20)
    zt = np.array([[0.0, 50.0]])
    mask = np.zeros((20, 1, 2), dtype=np.int32)
    mask[:5, 0, 1] = 1
    field = np.broadcast_to(z_cc[:, None, None], (20, 1, 2)).copy()

    agl = lv.extract_levels(field, z_cc, zt, [80.0], mask=mask, frame="agl")
    cart = lv.extract_levels(field, z_cc, zt, [80.0], mask=mask,
                             frame="cartesian")
    assert agl[0, 0, 0] == pytest.approx(cart[0, 0, 0], abs=1e-9)   # flat column
    assert abs(agl[0, 0, 1] - cart[0, 0, 1]) > 40.0                 # raised one


# ---------------------------------------------------------------------------
# Stitching
# ---------------------------------------------------------------------------

def test_stitch_reproduces_a_log_profile_it_was_sampled_from(column):
    """The case the loglinear method exists for: a neutral surface layer
    sampled at the engineering heights and put back together."""
    z_cc, zt, mask = column
    agl = z_cc - zt[0, 0]
    truth = lv.log_law(8.0, 10.0, agl)[:, None, None]

    lev = list(lv.ENGINEERING_LEVELS)
    sampled = lv.extract_levels(truth, z_cc, zt, lev, mask=mask)
    back = lv.stitch_levels(sampled, lev, z_cc, zt, mask=mask,
                            method="loglinear")

    err = np.abs(back - truth).max() / np.abs(truth).max()
    assert err < 0.02, f"loglinear stitching lost {100*err:.1f}% of a log profile"


def test_loglinear_beats_linear_on_a_log_profile(column):
    z_cc, zt, mask = column
    agl = z_cc - zt[0, 0]
    truth = lv.log_law(8.0, 10.0, agl)[:, None, None]
    lev = list(lv.ENGINEERING_LEVELS)
    sampled = lv.extract_levels(truth, z_cc, zt, lev, mask=mask)

    e_log = np.abs(lv.stitch_levels(sampled, lev, z_cc, zt, mask=mask,
                                    method="loglinear") - truth).max()
    e_lin = np.abs(lv.stitch_levels(sampled, lev, z_cc, zt, mask=mask,
                                    method="linear") - truth).max()
    assert e_log < e_lin


def test_stitch_holds_the_top_level_above_it(column):
    z_cc, zt, mask = column
    lev = [10.0, 20.0]
    values = np.ones((2, 1, 1)) * np.array([3.0, 7.0])[:, None, None]
    back = lv.stitch_levels(values, lev, z_cc, zt, mask=mask)
    top_cells = (z_cc - zt[0, 0]) > 20.0
    assert np.allclose(back[top_cells, 0, 0], 7.0)


def test_stitch_zeroes_solid_cells(column):
    z_cc, zt, _ = column
    mask = np.zeros((20, 1, 1), dtype=np.int32)
    mask[:3] = 1
    values = np.ones((2, 1, 1))
    back = lv.stitch_levels(values, [10.0, 20.0], z_cc, zt, mask=mask)
    assert np.all(back[:3] == 0.0)


def test_stitch_rejects_a_mismatched_level_count(column):
    z_cc, zt, mask = column
    with pytest.raises(ValueError):
        lv.stitch_levels(np.ones((3, 1, 1)), [10.0, 20.0], z_cc, zt, mask=mask)


def test_stitch_rejects_unsorted_levels(column):
    z_cc, zt, mask = column
    with pytest.raises(ValueError, match="increasing"):
        lv.stitch_levels(np.ones((2, 1, 1)), [20.0, 10.0], z_cc, zt, mask=mask)


# ---------------------------------------------------------------------------
# O'Brien, against the C++ operator
# ---------------------------------------------------------------------------

@pytest.fixture
def obrien_pair(amrex, case):
    """The same case with the adjustment off and on.

    Solver.cpp:113 copies velocity0 AFTER O'Brien runs, so the two
    velocity0 fields differ by exactly one application of it -- which is
    the oracle, with no binding change needed.
    """
    off = fwt.Solver(case(obrien={"enable": False}))
    off.setup()
    on = fwt.Solver(case(obrien={"enable": True}))
    on.setup()
    return off, on


def test_the_adjustment_actually_does_something(obrien_pair):
    """Otherwise the comparison below passes for the wrong reason."""
    off, on = obrien_pair
    assert not np.array_equal(off.velocity0[2], on.velocity0[2])
    assert np.array_equal(off.velocity0[0], on.velocity0[0])   # u untouched
    assert np.array_equal(off.velocity0[1], on.velocity0[1])   # v untouched


def test_obrien_matches_the_cpp_operator(obrien_pair):
    """To a few ULP -- and the bound is what carries the meaning.

    Not bit-exact, and the reason is understood rather than tolerated:
    clang contracts ``w -= Dh * dz[k]`` into a fused multiply-subtract,
    rounding once where numpy rounds twice, and numpy has no FMA to match
    it with. What that permits is ONE rounding per step, so the test
    asserts the difference stays within a few ULP and does not accumulate.
    A reassociated column sum -- a genuinely different operator, which is
    what this test exists to catch -- would drift far past that.
    """
    off, on = obrien_pair
    g = off.grid
    dz = np.diff(np.asarray(g.z_face))
    dx = (g.prob_hi[0] - g.prob_lo[0]) / g.nx
    dy = (g.prob_hi[1] - g.prob_lo[1]) / g.ny

    u, v, w = off.velocity0
    got = lv.obrien_w(u, v, w, dz, dx, dy, off.mask)
    ref = on.velocity0[2]

    diff = np.abs(got - ref)
    ulps = diff / np.maximum(np.spacing(np.abs(ref)), np.spacing(1.0))

    assert ulps.max() <= 4.0, (
        f"the numpy O'Brien is {ulps.max():.1f} ULP from Obrien::Apply; "
        f"more than a rounding apart means the algorithm differs, not the "
        f"arithmetic")
    assert np.mean(ulps) < 1.0
    # Non-accumulating: the top of the column is where a drifting sum
    # would show, and O'Brien pins it to zero on both sides.
    assert diff[-1].max() == 0.0


def test_obrien_leaves_w_exactly_zero_at_the_top(obrien_pair):
    off, _ = obrien_pair
    g = off.grid
    dz = np.diff(np.asarray(g.z_face))
    u, v, w = off.velocity0
    got = lv.obrien_w(u, v, w, dz,
                      (g.prob_hi[0] - g.prob_lo[0]) / g.nx,
                      (g.prob_hi[1] - g.prob_lo[1]) / g.ny, off.mask)
    fluid_top = off.mask[-1] == 0
    assert np.abs(got[-1][fluid_top]).max() == 0.0


def test_obrien_does_not_touch_the_first_fluid_cell(obrien_pair):
    """The scheme's whole point is that the correction is quadratic in
    height, so the near-surface values -- where the divergence estimate is
    most trustworthy -- are left alone."""
    off, _ = obrien_pair
    g = off.grid
    dz = np.diff(np.asarray(g.z_face))
    u, v, w = off.velocity0
    got = lv.obrien_w(u, v, w, dz,
                      (g.prob_hi[0] - g.prob_lo[0]) / g.nx,
                      (g.prob_hi[1] - g.prob_lo[1]) / g.ny, off.mask)
    k0 = lv.first_fluid_k(off.mask)
    jj, ii = np.meshgrid(np.arange(g.ny), np.arange(g.nx), indexing="ij")
    assert np.array_equal(got[k0, jj, ii], w[k0, jj, ii])


def test_surface_kinematic_w_follows_the_slope():
    """w = u.grad(h): flow climbing a slope acquires vertical velocity in
    proportion to how fast it is climbing."""
    nz, ny, nx = 5, 3, 4
    u = np.ones((nz, ny, nx)) * 10.0
    v = np.zeros((nz, ny, nx))
    # Ground rising 10 m per cell in x, cells 100 m wide -> slope 0.1.
    zt = np.tile(np.arange(nx) * 10.0, (ny, 1))
    mask = np.zeros((nz, ny, nx), dtype=np.int32)
    mask[0] = 1                                  # first fluid cell is k = 1

    w = lv.surface_kinematic_w(u, v, zt, 100.0, 100.0, mask)
    assert np.all(w[0] == 0.0)                   # solid cell untouched
    assert np.all(w[2:] == 0.0)                  # only the first fluid cell
    assert w[1, 1, 1] == pytest.approx(10.0 * 0.1)


def test_surface_kinematic_w_is_zero_over_flat_ground():
    u = np.ones((4, 2, 2)) * 7.0
    v = np.ones((4, 2, 2)) * 3.0
    zt = np.zeros((2, 2))
    mask = np.zeros((4, 2, 2), dtype=np.int32)
    assert np.all(lv.surface_kinematic_w(u, v, zt, 50.0, 50.0, mask) == 0.0)


def test_recommended_band_levels_are_octaves():
    """The placement study's outcome, asserted so it cannot drift: five
    levels octave-spaced across 10-160 m, then log-spaced aloft."""
    band = np.array(lv.RECOMMENDED_LEVELS[:5])
    assert band[0] == 10.0 and band[-1] == 160.0
    assert np.allclose(band[1:] / band[:-1], 2.0)
    # And it still contains the engineering heights that survive octave
    # spacing, which is why it is usable without interpolating to them.
    assert {10.0, 80.0, 160.0}.issubset(set(lv.RECOMMENDED_LEVELS))


def test_recommended_levels_span_more_than_the_engineering_band():
    """The failure mode the study found: five levels covering only
    10-160 m leave 20% error over the column, because on this grid that
    is its bottom third."""
    # Exactly a decade above the top engineering height, as it happens.
    assert max(lv.RECOMMENDED_LEVELS) >= 10.0 * max(lv.ENGINEERING_LEVELS)
    assert sorted(lv.RECOMMENDED_LEVELS) == list(lv.RECOMMENDED_LEVELS)


def test_recommended_levels_reproduces_the_tuned_set_at_its_own_column():
    """The scaling function must agree with the constant it generalises.

    To a tenth of a percent, not exactly: the constant carries the aloft
    levels rounded to whole metres (345 and 743 against 344.71 and
    742.654), which is how they were written down.
    """
    got = lv.recommended_levels(1600.0)
    assert np.allclose(got, lv.RECOMMENDED_LEVELS, rtol=1e-3)


def test_the_top_level_follows_the_column():
    """The defect this exists to fix.

    A fixed 1600 m top was tuned on Creek at 1128 m relief. The corpus
    reaches 1970 m, whose column is near 3000 m, and holding the top value
    constant over the last 1200 m of it cost 15x in that band.
    """
    for top in (1200.0, 1600.0, 3000.0):
        assert lv.recommended_levels(top)[-1] == pytest.approx(top)


def test_the_band_is_unchanged_whatever_the_column():
    """Only the aloft levels may move: the band is what the placement
    study fixed, and the measurement showed every band below 1600 m was
    identical with and without the extra aloft levels."""
    a = lv.recommended_levels(1600.0)
    b = lv.recommended_levels(3200.0)
    assert np.allclose(a[:5], b[:5], rtol=0, atol=1e-12)
    assert np.allclose(a[:5], lv.RECOMMENDED_LEVELS[:5], rtol=1e-12)


def test_levels_stay_sorted_and_positive():
    for top in (200.0, 1000.0, 5000.0):
        got = lv.recommended_levels(top)
        assert list(got) == sorted(got)
        assert got[0] > 0.0
        assert len(set(got)) == len(got)


def test_a_column_shallower_than_the_band_is_refused():
    """Silently returning a degenerate set would be worse than an error."""
    with pytest.raises(ValueError, match="band top"):
        lv.recommended_levels(100.0)


def test_max_agl_measures_the_deepest_column():
    """top_agl is the LOWEST ground's column, since that is the one with
    the most to reconstruct above the highest level."""
    z_cc = np.linspace(1000.0, 3000.0, 50)
    zt = np.array([[1000.0, 1500.0], [1200.0, 1800.0]])
    assert lv.max_agl(z_cc, zt) == pytest.approx(3000.0 - 1000.0)
    # And it accepts the solver's k-replicated terrain field.
    assert lv.max_agl(z_cc, np.broadcast_to(zt, (50, 2, 2))) == \
        pytest.approx(2000.0)
