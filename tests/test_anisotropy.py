"""
Anisotropy and O'Brien from Python (was
regtests/phase14_anisotropy_python).

Phase 7's assertions, recomputed in numpy rather than read back out of a
plotfile. That is the difference these bindings make to this particular
group: ``alpha_v`` can be checked cell by cell against an independent
implementation of the same formula, instead of being spot-checked against
values the solver itself wrote out.
"""

import numpy as np
import pytest

import fastwindterrain as fwt

NX, NY, NZ = 40, 40, 66
ALPHA_H_BASE, ALPHA_V_BASE = 1.0, 0.5
SLOPE_SCALE, DECAY_HEIGHT = 0.5, 500.0
MIN_FACTOR, MAX_FACTOR = 0.05, 2.0

GRID = {"n_cell": (NX, NY, NZ), "prob_lo": (0.0, 0.0, 0.0),
        "prob_hi": (1000.0, 1000.0, 961.2758234855), "dz0": 2.0,
        "stretching_ratio": 1.05, "max_grid_size": 32}

ANISO = {"enable": True, "source": "slope", "alpha_h_mode": "base",
         "slope_scale": SLOPE_SCALE, "decay_height": DECAY_HEIGHT,
         "alpha_h_base": ALPHA_H_BASE, "alpha_v_base": ALPHA_V_BASE}


@pytest.fixture
def aniso_case(terrain_points):
    def _case(**kw):
        cfg = {
            "grid": GRID,
            "terrain": {"points": terrain_points},
            "inflow": {"mode": "powerlaw", "u_ref": 8.0, "v_ref": 6.0},
            "poisson": {"alpha_h": ALPHA_H_BASE, "alpha_v": ALPHA_V_BASE,
                        "n_projections": 1},
        }
        cfg.update(kw)
        return cfg
    return _case


@pytest.fixture
def slope_terrain(amrex, terrain_points):
    g = fwt.Grid(GRID)
    t = fwt.Terrain(g, {"points": terrain_points})
    return g, t


def column_slopes(z_terrain):
    """``|grad z_terrain|`` per column, from the same central differences
    the solver uses -- written out here independently rather than read
    back, so the two can disagree."""
    h = z_terrain[0]                    # constant along k
    ny, nx = h.shape
    dx, dy = 1000.0 / NX, 1000.0 / NY
    i, j = np.arange(nx), np.arange(ny)
    ip1, im1 = np.minimum(i + 1, nx - 1), np.maximum(i - 1, 0)
    jp1, jm1 = np.minimum(j + 1, ny - 1), np.maximum(j - 1, 0)
    dhdx = (h[:, ip1] - h[:, im1]) / ((ip1 - im1)[None, :] * dx)
    dhdy = (h[jp1, :] - h[jm1, :]) / ((jp1 - jm1)[:, None] * dy)
    return np.hypot(dhdx, dhdy)


# ---------------------------------------------------------------------------
# The slope factor, cell by cell
# ---------------------------------------------------------------------------

@pytest.fixture
def slope_expectation(slope_terrain):
    """``alpha_v`` as the documented formula says it should be, and the
    ``Anisotropy`` that claims to implement it."""
    g, t = slope_terrain
    a = fwt.Anisotropy(g, t, ANISO)

    zt = t.z_terrain
    slope = column_slopes(zt)
    z_agl = np.maximum(np.asarray(g.z_cc)[:, None, None] - zt, 0.0)
    slope_3d = slope[None, :, :] * np.exp(-z_agl / DECAY_HEIGHT)
    expected = np.clip(ALPHA_V_BASE * np.exp(-slope_3d / SLOPE_SCALE),
                       MIN_FACTOR * ALPHA_V_BASE, MAX_FACTOR * ALPHA_V_BASE)
    return a, slope, expected


def test_the_hill_is_steep_enough_to_test_anything(slope_expectation):
    _, slope, _ = slope_expectation
    assert float(slope.max()) > 0.3, (
        "the hill is too gentle to exercise the slope factor")


def test_reported_slope_max_matches_an_independent_gradient(slope_expectation):
    a, slope, _ = slope_expectation
    assert float(a.slope_max) == pytest.approx(float(slope.max()), abs=1e-12)


def test_alpha_v_matches_the_formula_cell_by_cell(slope_expectation):
    a, _, expected = slope_expectation
    assert float(np.abs(a.alpha_v - expected).max()) < 1.0e-12


def test_the_suppression_actually_bites(slope_expectation):
    a, _, _ = slope_expectation
    assert int((a.alpha_v < 0.99 * ALPHA_V_BASE).sum()) > 100
    assert float(a.alpha_v.min()) < 0.5 * ALPHA_V_BASE


def test_alpha_h_mode_base_leaves_the_horizontal_weight_alone(
        slope_expectation):
    """As massconsistent_amr does."""
    a, _, _ = slope_expectation
    assert np.all(a.alpha_h == ALPHA_H_BASE)


def test_suppression_weakens_with_height(slope_expectation):
    """``slope_3d`` carries ``exp(-z_agl / decay_height)``, so the
    per-level minimum can only rise with k.

    It does not reach base by the domain top -- 961 m over a 500 m decay
    height is under two e-foldings -- so the assertion is that it decays,
    not that it has finished.
    """
    a, _, _ = slope_expectation
    per_level = a.alpha_v.min(axis=(1, 2))
    assert np.all(np.diff(per_level) >= 0.0)
    assert per_level[-1] > 2.0 * per_level[0]


# ---------------------------------------------------------------------------
# Disabled is inert; alpha_h_mode
# ---------------------------------------------------------------------------

def test_disabled_holds_both_weights_at_base(slope_terrain):
    """A feature that cannot change results when it is switched off."""
    g, t = slope_terrain
    off = fwt.Anisotropy(g, t, dict(ANISO, enable=False))
    assert np.all(off.alpha_h == ALPHA_H_BASE)
    assert np.all(off.alpha_v == ALPHA_V_BASE)


def test_source_none_is_also_inert(slope_terrain):
    g, t = slope_terrain
    none = fwt.Anisotropy(g, t, dict(ANISO, source="none"))
    assert np.all(none.alpha_v == ALPHA_V_BASE)


def test_alpha_h_mode_slope_applies_the_same_factor_to_both(slope_terrain):
    g, t = slope_terrain
    both = fwt.Anisotropy(g, t, dict(ANISO, alpha_h_mode="slope"))

    ratio_h = both.alpha_h_min / ALPHA_H_BASE
    ratio_v = both.alpha_v_min / ALPHA_V_BASE
    assert ratio_h == pytest.approx(ratio_v, abs=1e-9)
    assert ratio_h < 0.5, "the factor barely moved, so this compared nothing"
    # Cell by cell, not just at the minimum.
    assert np.allclose(both.alpha_h / ALPHA_H_BASE,
                       both.alpha_v / ALPHA_V_BASE, rtol=0, atol=1e-14)


# ---------------------------------------------------------------------------
# O'Brien
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_obrien_leaves_w_exactly_zero_at_the_top(amrex, aniso_case):
    """Exactly zero, held to round-off rather than a tolerance: making it
    exact is the whole point of the redistribution."""
    s = fwt.Solver(aniso_case(
        obrien={"enable": True},
        anisotropy={"enable": True, "source": "slope",
                    "slope_scale": 0.5, "decay_height": 500.0}))
    s.setup()               # O'Brien runs inside setup, before any solve
    ob = s.obrien

    assert ob["enabled"] is True
    assert int(ob["n_columns"]) > 100
    assert float(ob["max_residual"]) > 0.1, (
        "with nothing to remove, the exactness below proves nothing")
    assert float(ob["max_w_top"]) == 0.0

    # Straight from the field, not just from the reported number.
    w_top = s.velocity[2][-1]
    solid_top = s.mask[-1] == 1
    assert float(np.abs(w_top[~solid_top]).max()) == 0.0


@pytest.mark.slow
def test_obrien_off_touches_nothing(amrex, aniso_case):
    off = fwt.Solver(aniso_case(obrien={"enable": False}))
    off.setup()
    assert int(off.obrien["n_columns"]) == 0


# ---------------------------------------------------------------------------
# The base weights belong to the operator
# ---------------------------------------------------------------------------

def test_base_weights_are_refused_inside_a_solver_config(amrex, aniso_case):
    """Inside a Solver the base weights are ``poisson.alpha_h/alpha_v``.
    A second copy in the anisotropy section could disagree with the
    operator it feeds, so naming it there raises rather than being
    silently overridden."""
    with pytest.raises(ValueError, match="poisson"):
        fwt.Solver(aniso_case(
            anisotropy={"enable": True, "alpha_v_base": 0.3}))


@pytest.mark.slow
def test_base_weights_come_from_the_poisson_section(amrex, aniso_case):
    s = fwt.Solver(aniso_case(
        poisson={"alpha_h": 2.0, "alpha_v": 0.25, "n_projections": 1},
        anisotropy={"enable": False}))
    s.setup()

    assert s.anisotropy["alpha_h_base"] == 2.0
    assert s.anisotropy["alpha_v_base"] == 0.25
    assert float(s.alpha_h.max()) == 2.0
    assert float(s.alpha_v.max()) == 0.25


def test_standalone_anisotropy_accepts_its_own_bases(amrex):
    """There is no Poisson section to take them from, so here they are
    the Anisotropy's own."""
    g = fwt.Grid(GRID)
    t = fwt.Terrain(g, {"flat_elevation": 0.0})
    stand = fwt.Anisotropy(g, t, {"alpha_h_base": 3.0, "alpha_v_base": 0.75})
    assert stand.alpha_h_base == 3.0
    assert stand.alpha_v_base == 0.75
