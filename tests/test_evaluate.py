"""
The scoring (fastwindterrain.evaluate), phase 22a.

These tests exist because the metric is written before the model, and the
whole point of that ordering is that the metric can be wrong in ways a
loss curve will never reveal. A scoring function that quietly ignores the
vertical component, or scores a field with the right speed and the wrong
direction as perfect, produces a beautiful training run and a worthless
surrogate.

So each test here fixes one property by construction rather than by
example: an error of a known size must measure that size, and a field
that is wrong in a way the metric is supposed to catch must not score
well.
"""

import numpy as np
import pytest

from fastwindterrain import evaluate as E
from fastwindterrain.levels import BAND_BASE_M, BAND_TOP_M


# ---------------------------------------------------------------------------
# The mask, rebuilt from the 2D column index the dataset stores.
# ---------------------------------------------------------------------------

def test_the_fluid_mask_rebuilds_from_the_first_fluid_cell():
    """The dataset stores k_first, not the 3D mask, because k_first is what
    a 2D network can predict. That is only safe because terrain is
    single-valued: everything above the first fluid cell is fluid."""
    k_first = np.array([[0, 2], [3, 1]])
    fluid = E.fluid_from_k_first(k_first, nz=5)
    assert fluid.shape == (5, 2, 2)
    assert fluid[:, 0, 0].sum() == 5, "a column with no terrain is all fluid"
    assert fluid[:, 1, 0].sum() == 2, "k_first = 3 of 5 leaves two cells"
    # Solid below, fluid above, never interleaved.
    for j in range(2):
        for i in range(2):
            col = fluid[:, j, i]
            assert list(col) == sorted(col), "a hole in the column"


# ---------------------------------------------------------------------------
# The error itself. Units are metres per second and must stay that way.
# ---------------------------------------------------------------------------

def test_a_known_error_measures_its_own_size():
    """The one test that would catch a normalisation slipping in. A field
    off by exactly 0.5 m/s must score 0.5, not 0.5 over something."""
    ref = np.zeros((3, 4, 5, 6))
    pred = ref.copy()
    pred[0] = 0.5                       # 0.5 m/s in u, nothing else
    st = E.error_stats(pred, ref)
    assert st["rmse"] == pytest.approx(0.5)
    assert st["mae"] == pytest.approx(0.5)
    assert st["max"] == pytest.approx(0.5)


def test_the_error_is_a_vector_error_not_a_speed_error():
    """A field with the RIGHT speed everywhere and the wrong direction is
    wrong. A speed-only metric scores it perfectly, which is the failure
    this guards: the surrogate would learn magnitude and ignore veering.
    """
    ref = np.stack([np.full((2, 2), 10.0), np.zeros((2, 2)),
                    np.zeros((2, 2))])
    pred = np.stack([np.zeros((2, 2)), np.full((2, 2), 10.0),
                     np.zeros((2, 2))])     # 90 degrees off, same speed
    assert E.speed(*pred) == pytest.approx(E.speed(*ref))
    st = E.error_stats(pred, ref)
    assert st["rmse"] == pytest.approx(10.0 * np.sqrt(2.0))


def test_the_vertical_component_is_not_dropped():
    ref = np.zeros((3, 2, 2, 2))
    pred = ref.copy()
    pred[2] = 1.0
    assert E.error_stats(pred, ref)["rmse"] == pytest.approx(1.0)


def test_only_the_selected_cells_count():
    """Solid cells hold fill values, not wind. Scoring them would measure
    the fill and the number would move when the fill changed."""
    ref = np.zeros((3, 2, 2, 2))
    pred = ref.copy()
    pred[:, 0] = 100.0                  # a whole layer, wildly wrong
    fluid = np.ones((2, 2, 2), dtype=bool)
    fluid[0] = False                    # ... and excluded
    assert E.error_stats(pred, ref, sel=fluid)["rmse"] == pytest.approx(0.0)


def test_error_stats_refuses_a_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        E.error_stats(np.zeros((3, 2, 2)), np.zeros((3, 2, 3)))


def test_error_stats_refuses_an_empty_selection():
    with pytest.raises(ValueError, match="no points"):
        E.error_stats(np.zeros((3, 2, 2)), np.zeros((3, 2, 2)),
                      sel=np.zeros((2, 2), dtype=bool))


# ---------------------------------------------------------------------------
# Per level, and the band the deliverable lives in.
# ---------------------------------------------------------------------------

def test_the_band_is_the_engineering_levels_only():
    lv = [5.0, 10, 20, 40, 80, 160, 400, 900, 2000]
    inb = E.band_of(lv)
    assert list(np.asarray(lv)[inb]) == [5.0, 10, 20, 40, 80, 160]
    assert inb[0], "the base of the band is in the band"
    assert inb[5], "160 m is in the band, inclusively"
    assert not inb[6], "the aloft levels are not"


def test_the_band_error_excludes_the_easy_air_aloft():
    """The point of reporting the band separately. Aloft the flow is close
    to the undisturbed profile and easy; averaging it into the column
    flatters the number."""
    lv = np.array([5.0, 10, 20, 40, 80, 160, 400, 900, 2000])
    ref = np.zeros((3, lv.size, 4, 4))
    pred = ref.copy()
    pred[0, :6] = 1.0                   # all the error is in the band
    out = E.level_errors(pred, ref, lv)
    assert out["band"]["rmse"] == pytest.approx(1.0)
    assert out["column"]["rmse"] == pytest.approx(np.sqrt(6.0 / 9.0))
    assert out["band"]["rmse"] > out["column"]["rmse"]


def test_level_errors_reports_every_level_with_its_height():
    lv = np.array([5.0, 10, 20])
    ref = np.zeros((3, 3, 2, 2))
    pred = ref.copy()
    pred[0, 1] = 2.0
    out = E.level_errors(pred, ref, lv)
    assert [p["z_agl"] for p in out["levels"]] == [5.0, 10.0, 20.0]
    assert out["levels"][0]["rmse"] == pytest.approx(0.0)
    assert out["levels"][1]["rmse"] == pytest.approx(2.0)


def test_level_errors_refuses_a_level_count_that_disagrees():
    with pytest.raises(ValueError, match="levels in the field"):
        E.level_errors(np.zeros((3, 4, 2, 2)), np.zeros((3, 4, 2, 2)),
                       [10.0, 20.0])


def test_the_band_constants_are_the_ones_the_dataset_was_built_with():
    """If these ever diverge, the band metric silently starts describing a
    different band than the levels were placed for."""
    assert BAND_BASE_M == 5.0 and BAND_TOP_M == 160.0
    assert E.band_of([BAND_BASE_M, BAND_TOP_M]).all()


# ---------------------------------------------------------------------------
# Skill, and the grouping that keeps a single number from being reported.
# ---------------------------------------------------------------------------

def test_skill_is_one_when_perfect_and_zero_at_the_baseline():
    assert E.skill(0.0, 1.0) == pytest.approx(1.0)
    assert E.skill(1.0, 1.0) == pytest.approx(0.0)
    assert E.skill(2.0, 1.0) == pytest.approx(-1.0), "worse than doing nothing"
    assert np.isnan(E.skill(1.0, 0.0)), "no skill against a perfect baseline"


def test_the_relief_bins_cover_every_window_exactly_once():
    """A window falling in two bins would be counted twice; one falling in
    none would vanish from the report without a trace."""
    edges = [(lo, hi) for _, lo, hi in E.RELIEF_BINS]
    for r in (0.0, 1.0, 199.9, 200.0, 499.9, 500.0, 899.9, 900.0, 5000.0):
        hits = [1 for lo, hi in edges if lo <= r < hi]
        assert sum(hits) == 1, f"relief {r} lands in {sum(hits)} bins"


def test_grouping_keeps_empty_bins():
    """An empty bin is information -- it says the fold has nothing that
    steep. Dropping it makes the table look better covered than it is."""
    rows = E.group_by_relief([{"relief": 100.0, "rmse": 0.4}])
    assert len(rows) == len(E.RELIEF_BINS)
    assert rows[0]["n"] == 1 and rows[0]["mean"] == pytest.approx(0.4)
    assert rows[1]["n"] == 0 and np.isnan(rows[1]["mean"])


def test_grouping_reports_the_worst_and_not_only_the_mean():
    """Wind fields are judged by where they are worst."""
    rows = E.group_by_relief([{"relief": 50.0, "rmse": 0.1},
                              {"relief": 60.0, "rmse": 0.9}])
    assert rows[0]["mean"] == pytest.approx(0.5)
    assert rows[0]["worst"] == pytest.approx(0.9)


def test_the_table_aligns_every_column_including_empty_ones():
    rows = E.group_by_relief([{"relief": 100.0, "rmse": 0.4}])
    txt = E.table(rows, ["group", "n", "mean"],
                  {"group": 9, "n": 4, "mean": 8})
    lines = txt.splitlines()
    assert len({len(x) for x in lines}) == 1, "ragged table:\n" + txt
