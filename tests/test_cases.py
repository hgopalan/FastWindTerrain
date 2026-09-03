"""
The real-terrain case catalogue (cases/).

Everything here is offline. The download needs elevation, rasterio, pyproj
and scipy and reaches out to a USGS mirror; none of that belongs in a test
suite, and none of it is what could quietly go wrong. What could quietly go
wrong is the arithmetic:

* a stretching ratio that disagrees with Grid.cpp's accumulation, tripping
  an overshoot warning or an undershoot exception on a case that took ten
  minutes to download
* a domain floor on the wrong side of the terrain, which the solver does not
  check and will not tell you about
* a download box too small for the vendored reader's border smoothing, which
  would flatten the domain's own edges

All three are tested here, against the real solver where possible rather
than against a second copy of the same formula.
"""

import json
import math
import sys
import warnings

import numpy as np
import pytest

import fastwindterrain as fwt
from conftest import REPO

sys.path.insert(0, str(REPO / "cases"))
casegen = pytest.importorskip(
    "casegen", reason="cases/ is not present in this checkout")


# ---------------------------------------------------------------------------
# The catalogue
# ---------------------------------------------------------------------------

def test_catalogue_has_eight_cases():
    cases = casegen.catalogue()
    assert len(cases) == 8
    assert [c.slug for c in cases] == list(casegen.CATALOGUE)


def test_every_catalogue_slug_is_in_the_reference_file():
    """CATALOGUE is a hand-written list; the CSV is vendored. A rename
    upstream would otherwise surface as a KeyError mid-download."""
    reference = casegen.read_reference()
    for name in casegen.CATALOGUE:
        assert name in reference, f"{name} is not in wildfires_reference.csv"


def test_slugs_are_unique_across_the_whole_reference_file():
    """read_reference raises on a collision; this is the assertion that the
    slug rule is actually injective over the real data, not just the eight."""
    reference = casegen.read_reference()
    assert len(reference) == 29, (
        f"the reference CSV has {len(reference)} rows; it had 29 when the "
        f"catalogue was built. If it changed upstream, re-check CATALOGUE.")


def test_coordinates_come_through_unmodified():
    dixie = casegen.load("dixie_fire")
    assert (dixie.lat, dixie.lon) == (40.8521, -121.2334)
    assert dixie.state == "California"
    assert dixie.year == 2021


@pytest.mark.parametrize("name", casegen.CATALOGUE)
def test_slug_round_trips_from_the_fire_name(name):
    case = casegen.load(name)
    assert casegen.slug(case.name) == name


# ---------------------------------------------------------------------------
# The download box
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", casegen.CATALOGUE)
def test_download_box_is_wider_than_the_smoothed_border(name):
    """The vendored tiff_to_xyz_utm blends a Gaussian-smoothed field into
    the outer 20% of each side of whatever tile it is given. If the tile is
    not much larger than the domain, that smoothing reaches into the domain
    and flattens its edges -- which is exactly where terrain-driven flow
    separates, and it would look like perfectly good terrain.

    So: the untouched interior must contain the domain.
    """
    case = casegen.load(name)
    assert case.untouched_interior_m() > casegen.DOMAIN_M


@pytest.mark.parametrize("name", casegen.CATALOGUE)
def test_download_box_covers_the_domain_plus_halo(name):
    """The lat/lon box is approximate -- the exact window is cut in UTM --
    so the only thing that matters is that it is big enough."""
    case = casegen.load(name)
    box = case.bbox_deg()

    needed = 0.5 * casegen.DOMAIN_M + casegen.HALO_M
    half_lat_m = 0.5 * (box["lat_max"] - box["lat_min"]) * 111320.0
    half_lon_m = (0.5 * (box["lon_max"] - box["lon_min"]) * 111320.0
                  * math.cos(math.radians(case.lat)))

    assert half_lat_m > needed
    assert half_lon_m > needed
    # Longitude degrees shrink with latitude; the conversion must undo that,
    # or high-latitude cases would silently get a narrow box.
    assert half_lon_m == pytest.approx(half_lat_m, rel=1e-6)


def test_the_box_is_centred_on_the_fire():
    case = casegen.load("bootleg_fire")
    box = case.bbox_deg()
    assert 0.5 * (box["lat_min"] + box["lat_max"]) == pytest.approx(case.lat)
    assert 0.5 * (box["lon_min"] + box["lon_max"]) == pytest.approx(case.lon)


# ---------------------------------------------------------------------------
# The vertical grid, against the real solver
# ---------------------------------------------------------------------------

RELIEFS = [
    (0.0, 50.0),          # near-flat plateau
    (100.0, 400.0),       # rolling
    (1200.0, 2200.0),     # a Sierra canyon
    (1800.0, 3300.0),     # the high-elevation extreme
]


@pytest.mark.parametrize("z_min,z_max", RELIEFS,
                         ids=lambda v: f"{v:g}")
def test_column_height_matches_what_the_solver_computes(amrex, z_min, z_max):
    """The test that matters most in this file.

    casegen solves for a stretching ratio and then sets prob_hi[2] from its
    own accumulation. Grid::BuildVerticalStretching redoes that sum in C++
    and compares at a relative tolerance of 1e-8, warning on an overshoot
    and THROWING on an undershoot. So: build the grid the generator
    describes and require the solver to accept it silently and land on
    exactly the requested top.

    This is why column_height() uses the running product rather than the
    closed form -- the two differ in the last bits.
    """
    grid = casegen.grid_from_relief(z_min, z_max)

    with warnings.catch_warnings():
        warnings.simplefilter("error")      # an overshoot would raise here
        g = fwt.Grid(grid)

    assert g.prob_hi[2] == grid["prob_hi"][2], (
        "the solver adjusted the domain top, so the generated column does "
        "not match the one it built")
    assert float(g.z_face[0]) == z_min
    assert float(g.z_face[-1]) == pytest.approx(grid["prob_hi"][2], rel=1e-12)


@pytest.mark.parametrize("z_min,z_max", RELIEFS, ids=lambda v: f"{v:g}")
def test_the_air_column_is_measured_from_the_highest_ground(z_min, z_max):
    """ATMOSPHERE_M is the shallowest air column anywhere in the domain, not
    the deepest. Measuring it from the floor instead would leave a ridge
    with almost nothing above it."""
    grid = casegen.grid_from_relief(z_min, z_max)
    assert grid["prob_hi"][2] - z_max == pytest.approx(casegen.ATMOSPHERE_M,
                                                       rel=1e-9)


def test_the_floor_sits_on_the_ground_not_at_sea_level():
    """A 2000 m site must not get a domain starting at zero -- that is
    2000 m of grid spent below the terrain."""
    grid = casegen.grid_from_relief(1800.0, 3300.0)
    assert grid["prob_lo"][2] == 1800.0


def test_every_case_has_the_same_tensor_shape():
    """The vertical extent varies per case; the cell counts must not, or the
    catalogue cannot feed one network."""
    shapes = {casegen.grid_from_relief(a, b)["n_cell"] for a, b in RELIEFS}
    assert shapes == {(100, 100, 60)}


def test_horizontal_resolution_is_50_m():
    grid = casegen.grid_from_relief(0.0, 100.0)
    dx = (grid["prob_hi"][0] - grid["prob_lo"][0]) / grid["n_cell"][0]
    assert dx == 50.0


def test_solve_ratio_hits_the_target():
    for target in (500.0, 1050.0, 2000.0, 3000.0):
        r = casegen.solve_ratio(casegen.DZ0, 60, target)
        assert casegen.column_height(casegen.DZ0, r, 60) == pytest.approx(
            target, rel=1e-12)


def test_solve_ratio_refuses_an_unreachable_target():
    """60 cells of 4 m is already 240 m at ratio 1.0, so anything shorter
    than that cannot be built by stretching -- and saying so beats returning
    a ratio below 1 that compresses the grid downward."""
    with pytest.raises(ValueError, match="overshoots"):
        casegen.solve_ratio(casegen.DZ0, 60, 100.0)


# ---------------------------------------------------------------------------
# The guard the solver does not have
# ---------------------------------------------------------------------------

def test_terrain_below_the_floor_is_refused():
    grid = casegen.grid_from_relief(1000.0, 1400.0)
    z = np.array([1000.0, 1200.0, 999.0])          # one point under the floor
    with pytest.raises(ValueError, match="FLUID"):
        casegen.assert_fits(z, grid["prob_lo"][2], grid["prob_hi"][2])


def test_terrain_above_the_top_is_refused():
    grid = casegen.grid_from_relief(1000.0, 1400.0)
    z = np.array([1000.0, 1200.0, 9000.0])
    with pytest.raises(ValueError, match="SOLID"):
        casegen.assert_fits(z, grid["prob_lo"][2], grid["prob_hi"][2])


def test_terrain_that_fits_passes_silently():
    grid = casegen.grid_from_relief(1000.0, 1400.0)
    casegen.assert_fits(np.array([1000.0, 1400.0]),
                        grid["prob_lo"][2], grid["prob_hi"][2])


# ---------------------------------------------------------------------------
# A generated config, through the real solver
# ---------------------------------------------------------------------------

def synthetic_terrain(z_base, relief, n=60):
    """A ridge at absolute elevation, on the catalogue's domain.

    Stands in for a download: the point is that config() produces something
    the solver accepts and that the mask comes out sane, not that the
    terrain is any particular hill.
    """
    span = casegen.DOMAIN_M + 2.0 * casegen.HALO_M
    g = np.linspace(-casegen.HALO_M, casegen.DOMAIN_M + casegen.HALO_M, n)
    x, y = np.meshgrid(g, g, indexing="ij")
    ridge = 0.5 * (1.0 + np.sin(2.0 * np.pi * x / span))
    z = z_base + relief * ridge
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()])


@pytest.fixture
def prepared_case(tmp_path):
    """A case whose survey.json is a fixture rather than a download."""
    case = casegen.load("bootleg_fire")
    case.folder_override = str(tmp_path)
    survey = {"slug": case.slug, "z_min": 1400.0, "z_max": 1750.0,
              "n_points": 1234}
    (tmp_path / casegen.SURVEY_FILE).write_text(json.dumps(survey))
    return case, tmp_path, survey


def test_grid_regenerates_from_a_committed_survey(prepared_case, monkeypatch):
    """Terrain files are large and not committed; survey.json is a few
    hundred bytes and is. The grid has to be reproducible from it alone, or
    the catalogue is only as reproducible as a USGS mirror."""
    case, folder, survey = prepared_case
    monkeypatch.setattr(type(case), "folder",
                        property(lambda self: str(folder)))

    grid = case.grid()
    assert grid["prob_lo"][2] == survey["z_min"]
    assert grid["prob_hi"][2] - survey["z_max"] == pytest.approx(
        casegen.ATMOSPHERE_M, rel=1e-9)


@pytest.mark.slow
def test_a_generated_config_builds_and_straddles_the_ground(amrex, monkeypatch,
                                                            tmp_path):
    """End to end without a network: synthetic terrain at a realistic
    absolute elevation, through config() into a real Solver.

    The assertion is not "it ran" but that the domain contains a ground
    surface -- 0 < n_solid < n_total. Both bounds are the silent failures:
    all-fluid means the terrain is under the mesh, all-solid means it is
    over it.
    """
    case = casegen.load("creek_fire")
    monkeypatch.setattr(type(case), "folder",
                        property(lambda self: str(tmp_path)))
    (tmp_path / casegen.SURVEY_FILE).write_text(
        json.dumps({"z_min": 1800.0, "z_max": 2600.0}))

    points = synthetic_terrain(1800.0, 800.0)
    cfg = case.config(wind_speed=8.0, wind_direction=225.0, points=points)

    s = fwt.Solver(cfg)
    s.setup()

    n_solid, n_total = casegen.assert_straddles(s, what="creek_fire")
    assert 0 < n_solid < n_total
    # A ridge spanning most of the relief should occupy a real fraction of
    # the column, not three cells in a corner.
    assert 0.02 < n_solid / n_total < 0.9


def test_wind_direction_is_meteorological(amrex, monkeypatch, tmp_path):
    """225 degrees is a SOUTHWESTERLY -- wind FROM the southwest, blowing
    toward the northeast, so both components are positive. Getting this
    backwards would reverse every case in the catalogue and nothing would
    complain."""
    case = casegen.load("bootleg_fire")
    monkeypatch.setattr(type(case), "folder",
                        property(lambda self: str(tmp_path)))
    (tmp_path / casegen.SURVEY_FILE).write_text(
        json.dumps({"z_min": 1400.0, "z_max": 1700.0}))
    points = synthetic_terrain(1400.0, 300.0, n=20)

    sw = case.config(wind_speed=10.0, wind_direction=225.0, points=points)
    assert sw["inflow"]["u_ref"] > 0 and sw["inflow"]["v_ref"] > 0

    north = case.config(wind_speed=10.0, wind_direction=0.0, points=points)
    assert north["inflow"]["u_ref"] == pytest.approx(0.0, abs=1e-12)
    assert north["inflow"]["v_ref"] == pytest.approx(-10.0)

    west = case.config(wind_speed=10.0, wind_direction=270.0, points=points)
    assert west["inflow"]["u_ref"] == pytest.approx(10.0)
    assert west["inflow"]["v_ref"] == pytest.approx(0.0, abs=1e-12)


def test_config_says_how_to_get_the_terrain_when_it_is_missing(monkeypatch,
                                                               tmp_path):
    """Terrain files are gitignored, so a fresh clone hits this. The message
    has to name the command rather than just the missing path."""
    case = casegen.load("rim_fire")
    monkeypatch.setattr(type(case), "folder",
                        property(lambda self: str(tmp_path)))
    (tmp_path / casegen.SURVEY_FILE).write_text(
        json.dumps({"z_min": 1000.0, "z_max": 1500.0}))

    with pytest.raises(FileNotFoundError, match="prepare.py"):
        case.config()


# ---------------------------------------------------------------------------
# Terrain files
# ---------------------------------------------------------------------------

def test_terrain_round_trips_through_the_file(tmp_path):
    points = synthetic_terrain(1200.0, 300.0, n=12)
    path = tmp_path / "terrain.csv"
    casegen.write_terrain(str(path), points, header_lines=["a note"])
    back = casegen.read_terrain(str(path))

    assert back.shape == points.shape
    # Written at 0.1 mm; SRTM is integer metres before interpolation, so the
    # file is far finer than the data in it.
    assert np.abs(back - points).max() < 1e-3


def test_the_terrain_file_states_its_own_coordinate_convention(tmp_path):
    """x and y are local, z is absolute. Mixing those up silently is easy,
    so the file says which is which."""
    path = tmp_path / "terrain.csv"
    casegen.write_terrain(str(path), synthetic_terrain(1000.0, 100.0, n=5))
    head = path.read_text().splitlines()[:5]
    assert any("LOCAL" in line and "ABSOLUTE" in line for line in head)


def test_the_solver_reads_our_terrain_file(amrex, tmp_path):
    """Terrain::ReadPointFile accepts this format -- checked rather than
    assumed, since we write it ourselves rather than using the vendored
    writer."""
    path = tmp_path / "terrain.csv"
    casegen.write_terrain(str(path), synthetic_terrain(0.0, 100.0, n=30))

    grid = casegen.grid_from_relief(0.0, 100.0)
    g = fwt.Grid(grid)
    from_file = fwt.Terrain(g, {"file": str(path)})
    from_array = fwt.Terrain(g, {"points": casegen.read_terrain(str(path))})

    assert from_file.n_points == from_array.n_points > 0
    assert np.array_equal(from_file.z_terrain, from_array.z_terrain)


def test_the_fit_guard_tolerates_the_terrain_file_write_precision():
    """The failure that killed a worker 28 solves into a dataset run.

    write_terrain rounds to casegen._DECIMALS places, so a point read back
    can sit half a quantum below the floor the grid was derived from. On
    coastal_fire:20 that was 3.3e-05 m and the guard aborted the run --
    working exactly as designed, on something that is not a geometry error.

    Which windows it hits is luck: a minimum that rounds DOWN trips, one
    that rounds up does not.
    """
    floor, top = 100.0, 1100.0
    quantum = casegen.FIT_TOL_M

    # Just inside the write precision: tolerated.
    casegen.assert_fits(np.array([floor - 0.4 * quantum, 500.0]), floor, top)
    casegen.assert_fits(np.array([500.0, top + 0.4 * quantum]), floor, top)

    # Well outside it: still caught, which is the point of keeping the
    # tolerance at the quantum rather than making it generous.
    with pytest.raises(ValueError, match="below the domain floor"):
        casegen.assert_fits(np.array([floor - 1.0, 500.0]), floor, top)
    with pytest.raises(ValueError, match="above the domain top"):
        casegen.assert_fits(np.array([500.0, top + 1.0]), floor, top)


def test_the_fit_guard_still_catches_a_real_miss():
    """The tolerance must not blunt what the guard exists for: terrain
    below the mesh or a column entirely solid, which are hundreds of
    metres out, not microns."""
    with pytest.raises(ValueError, match="entirely\n?\\s*FLUID|FLUID"):
        casegen.assert_fits(np.array([-500.0, 10.0]), 0.0, 1000.0)
    with pytest.raises(ValueError, match="SOLID"):
        casegen.assert_fits(np.array([10.0, 5000.0]), 0.0, 1000.0)
