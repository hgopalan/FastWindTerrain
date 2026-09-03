"""
The terrain corpus, and the splits the unseen-terrain claim rests on.

Every test here runs OFFLINE: no network, no geo stack, no downloaded tiles.
That is the point of the two-stage build in cases/build_corpus.py -- the
survey needs SRTM, the split needs only committed JSON, and the split is the
half that decides whether a held-out result means anything.

The tests that matter most are the ones that assert the guard FIRES. A
leakage check that has never been seen to reject anything is a comment.
"""

import json
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = os.path.join(ROOT, "cases")
if CASES not in sys.path:
    sys.path.insert(0, CASES)

import casegen                                              # noqa: E402
import corpus                                               # noqa: E402


def fake_site(slug, lat, lon):
    return casegen.Case(slug=slug, name=slug, state="XX", year=2020,
                        lat=lat, lon=lon)


# ---------------------------------------------------------------------------
# Window geometry
# ---------------------------------------------------------------------------

def test_the_download_is_wide_enough_for_every_window():
    """The vendored reader smooths the outer 20% of a tile into a Gaussian.

    If that band reached a window, the window's terrain would be real in the
    middle and quietly flattened at the edges -- which is where flow
    separates, and nothing downstream would report it.
    """
    have, need = corpus.assert_download_is_wide_enough()
    assert have >= need


def test_a_too_narrow_download_is_refused():
    with pytest.raises(ValueError, match="border smoothing"):
        corpus.assert_download_is_wide_enough(half_width=4400.0)


def test_windows_tile_the_extent_and_the_middle_one_is_the_phase16a_domain():
    offs = list(corpus.window_offsets())
    assert len(offs) == corpus.N_WINDOWS_PER_SIDE ** 2

    # Centred: the 3 x 3's middle window is the 5 km box phase 16A built.
    mid = [o for o in offs if o[0] == 1 and o[1] == 1][0]
    centre = corpus.CORPUS_EXTENT_M / 2.0
    assert mid[2] + corpus.WINDOW_M / 2.0 == pytest.approx(centre)
    assert mid[3] + corpus.WINDOW_M / 2.0 == pytest.approx(centre)

    # Inside the extent, every one of them.
    for _, _, x0, y0 in offs:
        assert x0 >= 0.0 and x0 + corpus.WINDOW_M <= corpus.CORPUS_EXTENT_M
        assert y0 >= 0.0 and y0 + corpus.WINDOW_M <= corpus.CORPUS_EXTENT_M


def test_neighbouring_windows_overlap_by_exactly_the_stride():
    offs = {(i, j): (x, y) for i, j, x, y in corpus.window_offsets()}
    assert offs[(1, 0)][0] - offs[(0, 0)][0] == corpus.WINDOW_STRIDE_M
    overlap = corpus.WINDOW_M - corpus.WINDOW_STRIDE_M
    assert overlap == pytest.approx(0.5 * corpus.WINDOW_M)


def test_window_points_cuts_with_a_halo_and_shifts_to_local_coordinates():
    n = 400
    g = np.linspace(-casegen.HALO_M,
                    corpus.CORPUS_EXTENT_M + casegen.HALO_M, n)
    xx, yy = np.meshgrid(g, g)
    zz = 1000.0 + 0.01 * xx
    tile = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    wp = corpus.window_points(tile, 2500.0, 2500.0)
    assert wp[:, 0].min() >= -casegen.HALO_M - 1e-9
    assert wp[:, 0].max() <= corpus.WINDOW_M + casegen.HALO_M + 1e-9
    assert wp[:, 1].min() >= -casegen.HALO_M - 1e-9
    assert wp[:, 1].max() <= corpus.WINDOW_M + casegen.HALO_M + 1e-9
    # z is untouched: absolute metres above sea level, as everywhere else.
    assert wp[:, 2].min() > 1000.0


def test_a_window_outside_the_tile_is_refused_not_silently_empty():
    tile = np.array([[0.0, 0.0, 100.0], [10.0, 10.0, 110.0]])
    with pytest.raises(ValueError, match="no terrain points"):
        corpus.window_points(tile, 50000.0, 50000.0)


# ---------------------------------------------------------------------------
# Gridding and descriptors
# ---------------------------------------------------------------------------

def test_grid_terrain_recovers_a_known_surface():
    n = 300
    g = np.linspace(0.0, corpus.WINDOW_M, n)
    xx, yy = np.meshgrid(g, g)
    zz = 500.0 + 0.05 * xx
    pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    z = corpus.grid_terrain(pts, n=100)
    assert z.shape == (100, 100)
    assert np.all(np.isfinite(z))
    # A plane in x: every row identical, and rising left to right.
    assert np.allclose(z, z[0][None, :], atol=1e-9)
    assert np.all(np.diff(z[0]) > 0.0)


def test_descriptors_separate_a_ramp_from_a_rough_field():
    g = np.linspace(0.0, corpus.WINDOW_M, 100)
    xx, _ = np.meshgrid(g, g)

    ramp = 0.05 * xx                                    # smooth, one slope
    rng = np.random.default_rng(0)
    rough = ramp + 20.0 * rng.standard_normal(ramp.shape)

    d_ramp = corpus.descriptors(ramp)
    d_rough = corpus.descriptors(rough)

    # TRI is what tells them apart -- roughness, not relief. A linear ramp
    # is exactly zero in the interior and picks up a couple of centimetres
    # at the clamped edges, which is three orders below the rough field.
    assert d_ramp["tri"] < 0.1
    assert d_rough["tri"] > 10.0
    # And relief barely moves, which is the point of carrying both.
    assert d_rough["relief"] < 3.0 * d_ramp["relief"]


def test_anisotropy_is_one_for_a_cone_and_large_for_parallel_ridges():
    g = np.linspace(-1.0, 1.0, 100)
    xx, yy = np.meshgrid(g, g)

    cone = -100.0 * np.hypot(xx, yy)
    ridges = 100.0 * np.sin(6.0 * math.pi * xx)         # varies in x only

    assert corpus.descriptors(cone)["aniso"] == pytest.approx(1.0, abs=0.15)
    assert corpus.descriptors(ridges)["aniso"] > 50.0


def test_descriptors_are_all_finite_and_all_present():
    rng = np.random.default_rng(1)
    z = np.cumsum(rng.standard_normal((100, 100)), axis=0)
    d = corpus.descriptors(z)
    assert set(d) == set(corpus.DESCRIPTOR_KEYS)
    assert all(np.isfinite(v) for v in d.values())


# ---------------------------------------------------------------------------
# Screening
# ---------------------------------------------------------------------------

def fake_survey(reliefs, sea=0.0):
    """A survey whose windows have the given reliefs, and nothing else real."""
    return {
        "relief": max(reliefs) if reliefs else 0.0,
        "sea_fraction": sea,
        "windows": [{"id": f"w{n}", "z_min": 1000.0, "z_max": 1000.0 + r}
                    for n, r in enumerate(reliefs)],
    }


def test_flat_terrain_is_rejected_as_nodata_not_accepted_as_a_plain():
    """The failure that cost a day in phase 16A.

    A window of missing SRTM comes back as -32768, the vendored reader
    clamps it to zero, and the result is a flawless sea-level plain: smooth,
    plausible, and entirely fictional.
    """
    ok, reason, _ = corpus.screen_site(fake_survey([0.0] * 9))
    assert not ok
    assert "nodata" in reason


def test_a_flat_window_is_dropped_on_its_own_relief():
    ok, reason = corpus.screen_window({"z_min": 1000.0, "z_max": 1020.0})
    assert not ok and "under" in reason
    assert corpus.screen_window({"z_min": 1000.0, "z_max": 1600.0})[0]


def test_a_site_that_is_a_plate_with_one_steep_corner_is_rejected():
    """Erskine Fire, exactly.

    Its 10 km tile spans 81 m, which reads as terrain. Eight of its nine
    5 km windows hold 10 to 12 m and are plates. Screening the site on the
    tile would have let it in and trained on eight flat samples.
    """
    ok, reason, kept = corpus.screen_site(fake_survey([12.0] * 8 + [71.0]))
    assert not ok
    assert "plate with a corner" in reason
    assert kept == []


def test_a_site_with_one_flat_window_keeps_the_rest():
    ok, _, kept = corpus.screen_site(fake_survey([12.0] + [800.0] * 8))
    assert ok
    assert len(kept) == 8


def solv(div_after=None, speedup=None, n_directions=8):
    """A solvability entry: finite and physical unless told otherwise."""
    return {
        "div_fe_before": [0.1] * n_directions,
        "div_fe_after": (div_after if div_after is not None
                         else [0.05] * n_directions),
        "speedup_max": (speedup if speedup is not None
                        else [1.25] * n_directions),
        "n_directions": n_directions,
    }


def test_a_healthy_solve_passes():
    good = {"z_min": 1000.0, "z_max": 1800.0}
    assert corpus.screen_window(good, solvability=solv())[0]


def test_a_non_finite_field_is_dropped():
    good = {"z_min": 1000.0, "z_max": 1800.0}
    ok, reason = corpus.screen_window(
        good, solvability=solv(div_after=[0.05] * 7 + [float("inf")]))
    assert not ok and "non-finite" in reason


def test_an_absurd_speed_up_is_dropped():
    """The failure Poisson.cpp:218 records -- a 34.8 m/s corrected wind
    from a 10 m/s inflow -- which no divergence norm would catch."""
    good = {"z_min": 1000.0, "z_max": 1800.0}
    ok, reason = corpus.screen_window(
        good, solvability=solv(speedup=[1.2] * 7 + [3.5]))
    assert not ok and "failed solve" in reason


def test_rising_divergence_is_NOT_a_reason_to_drop():
    """The criterion that was tried and rejected, asserted so it stays out.

    "Did L-infinity divergence fall across the projection" would have
    removed about a third of the corpus over differences of 0.3-0.4 m/s --
    the same order as the ~0.25 m/s CFD practice runs at, and well inside
    the 20-30% acceptable for turbulent atmospheric flow. It also tracked
    nothing real: gentle sites failed it, steep ones passed, and no terrain
    descriptor separated them.
    """
    good = {"z_min": 1000.0, "z_max": 1800.0}
    worse = solv(div_after=[0.5] * 8)          # ends far ABOVE its start
    assert corpus.screen_window(good, solvability=worse)[0]


def test_relief_is_still_judged_before_the_solve():
    """A plate that solves beautifully is still a plate."""
    flat = {"z_min": 1000.0, "z_max": 1012.0}
    ok, reason = corpus.screen_window(flat, solvability=solv())
    assert not ok and "under" in reason


def test_without_solvability_data_the_screen_is_geometry_only():
    """A manifest can be built before the measurement exists."""
    good = {"z_min": 1000.0, "z_max": 1800.0}
    assert corpus.screen_window(good)[0]
    assert corpus.screen_window(good, solvability=None)[0]


def test_a_site_whose_windows_do_not_solve_says_so_rather_than_blaming_relief():
    """A site of plates and a site the solver cannot handle are different
    problems and want different responses, so the message must say which."""
    survey = fake_survey([800.0] * 9)
    broken = solv(div_after=[float("inf")] * 8)
    solvability = {w["id"]: broken for w in survey["windows"]}
    ok, reason, kept = corpus.screen_site(survey, solvability=solvability)
    assert not ok
    assert "solve" in reason
    assert "plate" not in reason
    assert kept == []


def test_a_site_with_a_few_unsolvable_windows_keeps_the_rest():
    survey = fake_survey([800.0] * 9)
    solvability = {w["id"]: solv() for w in survey["windows"]}
    solvability[survey["windows"][0]["id"]] = solv(
        div_after=[float("inf")] * 8)
    ok, _, kept = corpus.screen_site(survey, solvability=solvability)
    assert ok and len(kept) == 8


def test_a_mostly_submerged_tile_is_rejected():
    survey = fake_survey([400.0] * 9, sea=0.30)
    ok, reason, _ = corpus.screen_site(survey)
    assert not ok and "sea level" in reason


def test_real_relief_passes():
    ok, reason, kept = corpus.screen_site(fake_survey([1200.0] * 9))
    assert ok and reason == "ok" and len(kept) == 9


# ---------------------------------------------------------------------------
# Geography and clustering
# ---------------------------------------------------------------------------

def test_haversine_against_known_separations():
    # One degree of latitude, anywhere.
    assert corpus.haversine_km(0.0, 0.0, 1.0, 0.0) == pytest.approx(111.19,
                                                                    abs=0.1)
    assert corpus.haversine_km(40.0, -120.0, 40.0, -120.0) == 0.0


def test_the_reference_list_really_does_contain_near_duplicate_fires():
    """The reason cluster_sites exists at all.

    Tubbs and Kincade are 10.3 km apart in the same Mayacamas range. Thomas
    and Woolsey, BOTH in the phase 16A catalogue, are 28.3 km apart. Split
    on fire name and the same ridgeline lands on both sides.
    """
    by_slug = {c.slug: c for c in corpus.candidate_sites()}
    d = corpus.haversine_km(by_slug["tubbs_fire"].lat,
                            by_slug["tubbs_fire"].lon,
                            by_slug["kincade_fire"].lat,
                            by_slug["kincade_fire"].lon)
    assert d < 15.0
    d = corpus.haversine_km(by_slug["thomas_fire"].lat,
                            by_slug["thomas_fire"].lon,
                            by_slug["woolsey_fire"].lat,
                            by_slug["woolsey_fire"].lon)
    assert d < corpus.CLUSTER_RADIUS_KM


def test_clustering_is_transitive():
    """A-B near, B-C near, A-C far: all three must land in one cluster.

    This is why it is single linkage and not a radius search around each
    site. They share ground through B whether or not A can see C.
    """
    sites = [fake_site("a", 40.0, -120.0),
             fake_site("b", 40.35, -120.0),        # ~39 km from a
             fake_site("c", 40.70, -120.0)]        # ~39 km from b, 78 from a
    groups = corpus.cluster_sites(sites, radius_km=45.0)
    assert groups == [["a", "b", "c"]]


def test_clustering_separates_what_is_genuinely_far_apart():
    sites = [fake_site("a", 40.0, -120.0), fake_site("b", 45.0, -110.0)]
    assert corpus.cluster_sites(sites, radius_km=50.0) == [["a"], ["b"]]


def test_clustering_does_not_depend_on_the_order_sites_arrive_in():
    sites = [fake_site(s, lat, -120.0) for s, lat in
             (("a", 40.0), ("b", 40.2), ("c", 44.0), ("d", 44.1))]
    forward = corpus.cluster_sites(sites, radius_km=50.0)
    backward = corpus.cluster_sites(list(reversed(sites)), radius_km=50.0)
    assert forward == backward


def test_the_reference_fires_cluster_into_a_workable_number_of_groups():
    sites = corpus.candidate_sites()
    groups = corpus.cluster_sites(sites)
    assert sum(len(g) for g in groups) == len(sites)
    assert len({s for g in groups for s in g}) == len(sites)
    # Enough groups to split three ways with something held out.
    assert len(groups) >= 12


# ---------------------------------------------------------------------------
# The split
# ---------------------------------------------------------------------------

def test_the_split_covers_every_site_exactly_once():
    sites = corpus.candidate_sites()
    folds = corpus.split_clusters(corpus.cluster_sites(sites))
    assigned = corpus.fold_of(folds)
    assert set(assigned) == {s.slug for s in sites}


def test_a_cluster_is_never_broken_across_folds():
    """The first leakage rule: overlapping windows all go the same way."""
    sites = corpus.candidate_sites()
    clusters = corpus.cluster_sites(sites)
    assigned = corpus.fold_of(corpus.split_clusters(clusters))
    for cluster in clusters:
        assert len({assigned[s] for s in cluster}) == 1, cluster


def test_the_split_is_deterministic_and_the_seed_actually_moves_it():
    clusters = corpus.cluster_sites(corpus.candidate_sites())
    a = corpus.split_clusters(clusters, seed=20)
    assert a == corpus.split_clusters(clusters, seed=20)
    assert any(a != corpus.split_clusters(clusters, seed=s)
               for s in range(21, 32))


def test_the_split_lands_near_the_requested_fractions():
    sites = corpus.candidate_sites()
    folds = corpus.split_clusters(corpus.cluster_sites(sites))
    total = len(sites)
    for fold, want in corpus.DEFAULT_FRACTIONS.items():
        have = sum(len(c) for c in folds[fold]) / total
        # Clusters are indivisible and one holds three sites, so a tenth is
        # about the best that can be asked of 29 fires.
        assert abs(have - want) < 0.10, (fold, have, want)


def test_an_impossible_split_is_refused_rather_than_returning_an_empty_fold():
    with pytest.raises(ValueError, match="empty"):
        corpus.split_clusters([["a"], ["b"]])


def test_fractions_that_do_not_sum_to_one_are_refused():
    with pytest.raises(ValueError, match="sum to 1"):
        corpus.split_clusters([["a"], ["b"], ["c"]],
                              fractions={"train": 0.5, "val": 0.2,
                                         "test": 0.2})


# ---------------------------------------------------------------------------
# The guard -- and it has to be seen to fire
# ---------------------------------------------------------------------------

def test_no_leakage_passes_on_the_real_split_and_reports_the_closest_pair():
    sites = corpus.candidate_sites()
    folds = corpus.split_clusters(corpus.cluster_sites(sites))
    d, a, b = corpus.assert_no_leakage(sites, folds)
    assert d >= corpus.CLUSTER_RADIUS_KM
    assert corpus.fold_of(folds)[a] != corpus.fold_of(folds)[b]


def test_two_fires_on_the_same_mountain_in_different_folds_are_caught():
    """Hand-built leakage: Tubbs in train, Kincade 10 km away in test."""
    sites = [fake_site("tubbs", 38.65, -122.4833),
             fake_site("kincade", 38.6667, -122.6),
             fake_site("far", 44.0, -110.0)]
    folds = {"train": [["tubbs"]], "val": [["far"]], "test": [["kincade"]]}
    with pytest.raises(ValueError, match="same\n?\\s*ground|same ground"):
        corpus.assert_no_leakage(sites, folds)


def test_windows_that_would_physically_overlap_are_caught_by_geometry():
    """The check that is NOT implied by the clustering radius.

    A window centre sits up to a corner offset from its site, so two sites
    just outside the radius can still have windows that overlap. Drop the
    radius below that reach and the geometry check has to be the one that
    objects.
    """
    sites = [fake_site("a", 40.0, -120.0),
             fake_site("b", 40.05, -120.0),        # ~5.6 km apart
             fake_site("far", 44.0, -110.0)]
    folds = {"train": [["a"]], "val": [["far"]], "test": [["b"]]}
    with pytest.raises(ValueError, match="OVERLAP"):
        corpus.assert_no_leakage(sites, folds, radius_km=1.0)


def test_a_site_left_out_of_the_split_is_caught():
    sites = [fake_site("a", 40.0, -120.0), fake_site("b", 44.0, -110.0),
             fake_site("c", 34.0, -118.0)]
    folds = {"train": [["a"]], "val": [["b"]], "test": []}
    with pytest.raises(ValueError, match="in no fold"):
        corpus.assert_no_leakage(sites, folds)


def test_a_site_in_two_folds_at_once_is_caught():
    folds = {"train": [["a"]], "val": [["a"]], "test": [["b"]]}
    with pytest.raises(ValueError, match="appears in both"):
        corpus.fold_of(folds)


def test_everything_in_one_fold_is_not_a_held_out_split():
    sites = [fake_site("a", 40.0, -120.0), fake_site("b", 44.0, -110.0)]
    folds = {"train": [["a"], ["b"]], "val": [], "test": []}
    with pytest.raises(ValueError, match="nothing is held out"):
        corpus.assert_no_leakage(sites, folds)


# ---------------------------------------------------------------------------
# The committed manifest, if one has been built
# ---------------------------------------------------------------------------

needs_manifest = pytest.mark.skipif(
    not os.path.isfile(corpus.MANIFEST),
    reason="no corpus manifest; run cases/build_corpus.py --survey --split")


@needs_manifest
def test_the_committed_manifest_is_free_of_leakage():
    """Re-run the guard against what was actually committed.

    Not a duplicate of the test above: that one checks the function, this
    one checks the artefact. A manifest built with a different radius, an
    older site list or a hand edit would pass the first and fail this.
    """
    manifest = corpus.load_manifest()
    by_slug = {c.slug: c for c in corpus.candidate_sites()}
    sites = [by_slug[s] for s in manifest["sites"]]
    folds = {f: [list(c) for c in cs] for f, cs in manifest["folds"].items()}
    d, _, _ = corpus.assert_no_leakage(
        sites, folds, radius_km=manifest["cluster_radius_km"])
    assert d == pytest.approx(manifest["closest_cross_fold"][0], abs=0.01)


@needs_manifest
def test_every_window_belongs_to_its_site_fold():
    manifest = corpus.load_manifest()
    for w in manifest["windows"]:
        assert w["fold"] == manifest["fold_of"][w["site"]], w["id"]


@needs_manifest
def test_window_ids_are_unique():
    manifest = corpus.load_manifest()
    ids = [w["id"] for w in manifest["windows"]]
    assert len(ids) == len(set(ids))


@needs_manifest
def test_every_window_carries_a_grid_that_straddles_its_own_terrain():
    """The phase 16A hazard, once per window instead of once per case.

    The solver never checks that terrain fits in the domain, and a corpus
    has two hundred chances to get it wrong instead of eight.
    """
    manifest = corpus.load_manifest()
    for w in manifest["windows"]:
        assert w["prob_lo_z"] <= w["z_min"], w["id"]
        assert w["prob_hi_z"] >= w["z_max"], w["id"]
        assert w["prob_hi_z"] - w["z_max"] >= casegen.ATMOSPHERE_M - 1e-6, \
            w["id"]


@needs_manifest
def test_every_window_grid_reproduces_the_solvers_own_accumulation():
    """prob_hi must match what Grid::BuildVerticalStretching will sum to.

    The solver compares its accumulated column against prob_hi - prob_lo at
    a relative tolerance of 1e-8 (Source/Grid.cpp:19) and warns or throws
    otherwise. Recomputing here rather than trusting the recorded number
    means a change to N_CELL or DZ0 shows up as a test failure and not as
    two hundred warnings during a dataset run.
    """
    manifest = corpus.load_manifest()
    for w in manifest["windows"]:
        h = casegen.column_height(casegen.DZ0, w["stretching_ratio"],
                                  casegen.N_CELL[2])
        assert abs((w["prob_lo_z"] + h) - w["prob_hi_z"]) <= \
            1e-8 * abs(w["prob_hi_z"] - w["prob_lo_z"]), w["id"]


@needs_manifest
def test_the_manifest_holds_out_enough_terrain_to_claim_anything():
    manifest = corpus.load_manifest()
    for fold in corpus.FOLDS:
        n = len(corpus.windows_in(manifest, fold))
        assert n > 0, fold
    # Two held-out fires are two anecdotes, not a generalisation test.
    n_test_sites = sum(1 for f in manifest["fold_of"].values() if f == "test")
    assert n_test_sites >= 4


@needs_manifest
def test_rejected_sites_are_recorded_with_a_reason_and_are_not_in_any_fold():
    manifest = corpus.load_manifest()
    for slug, reason in manifest["rejected"].items():
        assert reason and reason != "ok"
        assert slug not in manifest["fold_of"]
        assert slug not in manifest["sites"]


@needs_manifest
def test_the_manifest_is_the_one_build_corpus_would_write_today():
    """Regenerate the split from the committed surveys and compare.

    Guards against a manifest that was edited by hand, or built from a
    corpus that has since changed. Everything it needs is committed, so
    this runs with no tiles and no network.
    """
    import build_corpus

    committed = corpus.load_manifest()
    rebuilt = build_corpus.build_manifest(
        fractions=committed["fractions"], seed=committed["seed"],
        radius_km=committed["cluster_radius_km"])
    assert rebuilt["folds"] == committed["folds"]
    assert rebuilt["sites"] == committed["sites"]
    assert [w["id"] for w in rebuilt["windows"]] == \
        [w["id"] for w in committed["windows"]]


@needs_manifest
def test_json_round_trips_without_losing_the_grid_precision():
    """The stretching ratio is solved to ~1e-15 and then written as text.

    If json shortened it, every window's column would miss its top by
    enough to trip the solver's 1e-8 check.
    """
    manifest = corpus.load_manifest()
    again = json.loads(json.dumps(manifest))
    for a, b in zip(manifest["windows"], again["windows"]):
        assert a["stretching_ratio"] == b["stretching_ratio"]


# ---------------------------------------------------------------------------
# The solver is odd in the inflow direction, which halves the dataset.
# ---------------------------------------------------------------------------

def test_independent_directions_are_half_the_rose():
    assert len(corpus.INDEPENDENT_DIRECTIONS) * 2 == len(
        corpus.WIND_DIRECTIONS)
    # Every direction in the full rose is either generated or the reverse
    # of one that is.
    for d in corpus.WIND_DIRECTIONS:
        assert (d in corpus.INDEPENDENT_DIRECTIONS
                or (d - 180.0) % 360.0 in corpus.INDEPENDENT_DIRECTIONS), d


def test_reverse_of_is_negation():
    a = np.arange(24, dtype=np.float64).reshape(3, 2, 2, 2)
    assert np.array_equal(corpus.reverse_of(a), -a)
    assert np.array_equal(corpus.reverse_of(corpus.reverse_of(a)), a)


@pytest.mark.slow
def test_the_solver_really_is_odd_in_the_inflow(amrex):
    """The measurement the halving rests on, run rather than trusted.

    If this ever fails, INDEPENDENT_DIRECTIONS is wrong and every dataset
    built on it is missing half its conditions. Worth a solve.

    Skipped without a corpus manifest and tile, like the other cases that
    need real terrain.
    """
    import fastwindterrain as fwt

    if not os.path.isfile(corpus.MANIFEST):
        pytest.skip("no corpus manifest")
    manifest = corpus.load_manifest()
    wid = manifest["windows"][0]["id"]
    if not os.path.isfile(corpus.tile_path(manifest["windows"][0]["site"])):
        pytest.skip("no tile for the first corpus window")

    fields = {}
    for d in (0.0, 180.0):
        s = fwt.Solver(corpus.window_config(manifest, wid,
                                            wind_direction=d,
                                            poisson={"n_projections": 2}))
        s.setup()
        s.solve()
        fields[d] = np.stack([np.array(f) for f in s.velocity])
        fluid = s.mask == 0

    scale = np.abs(fields[0.0])[:, fluid].max()
    worst = np.abs(fields[0.0] + fields[180.0])[:, fluid].max()
    assert worst / scale < 1e-12, (
        f"reversing the wind is not exactly a negation: {worst/scale:.2e} "
        f"relative. INDEPENDENT_DIRECTIONS assumes it is.")


# ---------------------------------------------------------------------------
# The dataset format (cases/build_dataset.py). The reader is what the
# training harness will use, so its invariants are asserted here.
# ---------------------------------------------------------------------------

def _fake_dataset(tmp_path):
    """A two-sample dataset on disk, one solved and one derived."""
    import build_dataset as bd

    rng = np.random.default_rng(4)
    arrays = {
        "u_lev": rng.standard_normal((9, 4, 4)).astype("float32"),
        "v_lev": rng.standard_normal((9, 4, 4)).astype("float32"),
        "w_lev": rng.standard_normal((9, 4, 4)).astype("float32"),
        "terrain": rng.standard_normal((4, 4)).astype("float32"),
        "k_first": np.zeros((4, 4), dtype="int16"),
        "z_cc": np.linspace(0.0, 100.0, 6),
        "levels": np.asarray([5.0, 10, 20, 40, 80, 160, 400, 900, 2000]),
    }
    sid = "w:00@000"
    np.savez_compressed(tmp_path / "shard_00000.npz",
                        **{f"{sid}|{k}": v for k, v in arrays.items()})
    man = {
        "shards": 1,
        "samples": [
            {"id": sid, "derived": False, "fold": "train", "has_3d": False},
            {"id": "w:00@180", "derived": True, "derived_from": sid,
             "fold": "train", "has_3d": False},
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(man))
    return bd


def test_the_reader_materialises_the_reverse_directions(tmp_path):
    """Four directions on disk, eight to a consumer."""
    bd = _fake_dataset(tmp_path)
    got = list(bd.load_dataset(str(tmp_path)))
    assert len(got) == 2
    assert sum(1 for i, _ in got if i["derived"]) == 1


def test_the_reader_negates_velocity_and_not_geometry(tmp_path):
    """The failure this guards against would flip the ground upside down
    and still look plausible in a loss curve."""
    bd = _fake_dataset(tmp_path)
    d = {i["id"]: a for i, a in bd.load_dataset(str(tmp_path))}
    a, b = d["w:00@000"], d["w:00@180"]
    for k in ("u_lev", "v_lev", "w_lev"):
        assert np.array_equal(b[k], -a[k]), k
    for k in ("terrain", "k_first", "z_cc", "levels"):
        assert np.array_equal(b[k], a[k]), k


def test_the_reader_filters_by_fold(tmp_path):
    bd = _fake_dataset(tmp_path)
    assert len(list(bd.load_dataset(str(tmp_path), fold="train"))) == 2
    assert len(list(bd.load_dataset(str(tmp_path), fold="test"))) == 0


def test_the_reader_stitches_several_workers_together(tmp_path):
    """The layout a real run produces: one shard file and one manifest per
    worker, dealt round-robin, all in one directory.

    Worth its own test because the single-worker layout is what a smoke
    run exercises, and the naming differs -- shard_NN_MMMMM.npz and
    manifest_NN.json against shard_MMMMM.npz and manifest.json. A reader
    that only understood the smoke layout would return an empty dataset
    from a real run, which looks like a training bug rather than an I/O
    one.
    """
    import build_dataset as bd

    rng = np.random.default_rng(7)
    for part in range(3):
        sid = f"w:{part}0@000"
        arrays = {
            "u_lev": rng.standard_normal((9, 4, 4)).astype("float32"),
            "terrain": rng.standard_normal((4, 4)).astype("float32"),
        }
        np.savez_compressed(tmp_path / f"shard_{part:02d}_00000.npz",
                            **{f"{sid}|{k}": v for k, v in arrays.items()})
        man = {"shards": 1, "samples": [
            {"id": sid, "derived": False, "fold": "train", "has_3d": False},
            {"id": f"w:{part}0@180", "derived": True, "derived_from": sid,
             "fold": "train", "has_3d": False},
        ]}
        (tmp_path / f"manifest_{part:02d}.json").write_text(json.dumps(man))

    got = list(bd.load_dataset(str(tmp_path)))
    assert len(got) == 6, "three workers x two directions"
    assert len({i["id"] for i, _ in got}) == 6
    # And the derived halves still negate velocity only.
    d = {i["id"]: a for i, a in got}
    assert np.array_equal(d["w:10@180"]["u_lev"], -d["w:10@000"]["u_lev"])
    assert np.array_equal(d["w:10@180"]["terrain"], d["w:10@000"]["terrain"])


def test_the_reader_refuses_a_directory_with_no_manifest(tmp_path):
    import build_dataset as bd

    with pytest.raises(FileNotFoundError, match="no manifest"):
        list(bd.load_dataset(str(tmp_path)))
