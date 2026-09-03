#!/usr/bin/env python3
"""
casegen.py -- build FastWindTerrain cases from real terrain.

Every case in this repository until now was synthetic: a Gaussian hill or a
linear slope from tools/make_terrain.py, on a 1 km box. This module builds
the other kind -- 5 x 5 km domains over real ground, from SRTM elevation
data, at wildfire locations taken from wildfires_reference.csv.

All the logic lives here. Each case folder holds a twenty-line prepare.py
that imports this and passes its own coordinates, so there are eight case
folders and one implementation.

THE GRID FOLLOWS THE TERRAIN. Cell counts are the same for every case --
(100, 100, 60), so every case yields the same tensor shape -- but the
vertical extent is derived from the relief the download actually finds:

    prob_lo[2] = z_min of the tile          (the floor sits on the ground)
    prob_hi[2] = z_min + H                  (H = relief + 1000 m of air)
    stretching_ratio                        (solved so the column hits H)

WHY THE FLOOR MATTERS MORE THAN IT LOOKS. Nothing in the solver checks that
terrain fits inside the domain: Grid::Params::Validate never sees the
terrain, and Terrain::BuildMask just evaluates z_cc <= z_terrain. So terrain
below prob_lo[2] leaves every cell fluid with the surface under the mesh,
and terrain above prob_hi[2] marks every column solid. Both are SILENT --
no warning, no error, a plausible-looking plotfile either way. SRTM
elevations are absolute metres above sea level and these sites run from a
few hundred metres to over two thousand, so this is the single most likely
way the catalogue could produce confident nonsense. AssertFits() below is
the guard the solver does not have, and it runs before anything is written.

COORDINATES ARE SHIFTED IN x AND y, NOT IN z. Horizontal coordinates become
local metres on [0, 5000]: inverse-distance weighting on seven-digit UTM
northings loses precision in the distances it squares. Elevations stay
absolute, so a height in the output is a real height above sea level and can
be compared with met data.

Usage (normally through a case folder's prepare.py):

    import casegen
    case = casegen.load("bootleg_fire")
    case.prepare()                      # download, survey, write terrain.xyz
    cfg = case.config(wind_speed=8.0, wind_direction=225.0)

Requires, for the download path only: elevation, rasterio, pyproj, scipy
(pip install ".[cases]"). Everything else here is numpy and the standard
library, so the grid arithmetic is testable with no geo stack installed.
"""

import csv
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

REFERENCE_CSV = os.path.join(HERE, "wildfires_reference.csv")

# ---------------------------------------------------------------------------
# The catalogue
# ---------------------------------------------------------------------------

# Eight of the reference file's twenty-nine, chosen for recurrent fire AND
# for spread across wind regime and relief -- eight variations of one canyon
# would teach a surrogate that canyon. Latitudes run 34 to 43 degrees and
# cover the Diablo, Santa Ana/sundowner, Sierra canyon, Cascade canyon,
# coast range and interior plateau regimes.
#
# Adding one back is a line here; the other twenty-one stay in the CSV.
CATALOGUE = (
    "dixie_fire",            # Feather River canyon -- extreme relief
    "creek_fire",            # Sierra NF -- the high-elevation case
    "rim_fire",              # Stanislaus NF / Tuolumne canyon
    "august_complex_fire",   # Mendocino NF inner Coast Ranges
    "tubbs_fire",            # Sonoma/Napa -- Diablo wind corridor
    "thomas_fire",           # Ventura / Santa Ynez -- sundowner
    "woolsey_fire",          # Santa Monica Mountains -- Santa Ana
    "bootleg_fire",          # Interior Oregon plateau -- low-relief contrast
)

# ---------------------------------------------------------------------------
# Domain and grid constants
# ---------------------------------------------------------------------------

DOMAIN_M = 5000.0          # 5 x 5 km, per the phase 16A brief
N_CELL = (100, 100, 60)    # 50 m horizontally; one tensor shape for every case
DZ0 = 4.0                  # surface-adjacent cell height [m]
MAX_GRID_SIZE = 50         # -> 2 x 2 boxes in x and y, so the gather is tested
ATMOSPHERE_M = 1000.0      # air above the HIGHEST ground, not above the floor

# The IDW halo: terrain points are kept this far outside the domain so the
# interpolation at the boundary averages real neighbours rather than
# extrapolating from one side.
HALO_M = 250.0

# How much larger the DOWNLOAD is than the domain, per side.
#
# This is not a round number and the reason matters. The vendored
# tiff_to_xyz_utm calls _smooth_terrain_border(z, fraction=0.2), which blends
# a Gaussian-smoothed field into the outer 20% of each side of whatever tile
# it is given -- weight 1 at the edge, ramping to 0 one fifth of the way in.
# That is right for its own purpose and would be wrong here: with a small
# margin the smoothing reaches into the domain and quietly flattens its
# edges, which is exactly where terrain-driven flow separates.
#
# So the tile is sized to keep its untouched interior larger than the
# domain:  W = 5000 + 2*4400 = 13800 m is wrong;  W = 2*4400 = 8800 m gives a
# 1760 m smoothed band per side and 5280 m of untouched interior, which
# contains the 5000 m domain with 140 m to spare on each side.
#
# The download is ~3x the domain area and costs almost nothing: the elevation
# package fetches whole 1-degree SRTM tiles and clips, so the marginal bytes
# are near zero.
DOWNLOAD_HALF_WIDTH_M = 4400.0

_SMOOTHED_FRACTION = 0.2   # what the vendored reader smooths; see above

# Degrees per metre of latitude. Good to ~0.1% over this latitude range,
# which is far finer than needed: the box is cut exactly in UTM afterwards.
_M_PER_DEG_LAT = 111320.0


# ---------------------------------------------------------------------------
# The reference list
# ---------------------------------------------------------------------------

def slug(name):
    """'August Complex Fire' -> 'august_complex_fire'."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def read_reference(path=REFERENCE_CSV):
    """Every row of the reference CSV, keyed by slug.

    Vendored verbatim from wildfire_levelset, so this reads it as it is
    rather than as we would have written it.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        s = slug(r["name"])
        if s in out:
            raise ValueError(f"duplicate case slug {s!r} in {path}")
        out[s] = {
            "slug": s,
            "name": r["name"],
            "state": r["state"],
            "year": int(r["year"]),
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "area_ha": float(r["area_ha"]),
            "duration_days": float(r["duration_days"]),
        }
    return out


def catalogue(path=REFERENCE_CSV):
    """The eight CATALOGUE cases, in catalogue order."""
    ref = read_reference(path)
    missing = [s for s in CATALOGUE if s not in ref]
    if missing:
        raise ValueError(
            f"the reference file has no row for {missing}; either the CSV "
            f"changed upstream or CATALOGUE names a fire that is not in it")
    return [Case(**ref[s]) for s in CATALOGUE]


def load(name, path=REFERENCE_CSV):
    """One case by slug."""
    for c in catalogue(path):
        if c.slug == name:
            return c
    raise KeyError(f"{name!r} is not in the catalogue: {list(CATALOGUE)}")


# ---------------------------------------------------------------------------
# The vertical grid
# ---------------------------------------------------------------------------

def column_height(dz0, ratio, nz):
    """Sum of dz0 * ratio**k for k = 0 .. nz-1.

    Deliberately the running product-and-sum, NOT the closed form
    dz0*(r**nz - 1)/(r - 1). Grid::BuildVerticalStretching accumulates it
    this way (Source/Grid.cpp:88) and compares the result against
    prob_hi[2] - prob_lo[2] at a relative tolerance of 1e-8
    (Source/Grid.cpp:19). The closed form differs from the accumulation in
    the last bits, which is a needless way to trip an overshoot warning or,
    worse, the undershoot exception.
    """
    h = 0.0
    rk = 1.0
    for _ in range(int(nz)):
        h += dz0 * rk
        rk *= ratio
    return h


def solve_ratio(dz0, nz, target, lo=1.0, hi=1.5, iterations=100):
    """The stretching ratio whose column is exactly `target` metres tall.

    Bisection on column_height, which is monotone in the ratio. Reaches the
    target to about 1e-15 relative, so the caller can then set prob_hi[2]
    from column_height() and the solver's own check passes by construction.
    """
    if target <= 0.0:
        raise ValueError(f"target column height must be positive, got {target}")
    uniform = column_height(dz0, lo, nz)
    if uniform > target:
        raise ValueError(
            f"a column of {nz} cells starting at dz0 = {dz0} m is already "
            f"{uniform:.1f} m tall at ratio {lo}, which overshoots the "
            f"requested {target:.1f} m. Reduce dz0 or n_cell[2].")
    if column_height(dz0, hi, nz) < target:
        raise ValueError(
            f"even a ratio of {hi} does not reach {target:.1f} m with "
            f"nz = {nz}, dz0 = {dz0}")
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if column_height(dz0, mid, nz) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def grid_from_relief(z_min, z_max, n_cell=N_CELL, dz0=DZ0,
                     atmosphere=ATMOSPHERE_M, domain=DOMAIN_M,
                     max_grid_size=MAX_GRID_SIZE):
    """The grid section of a config, derived from the measured relief.

    The floor sits at the tile minimum and the top is `atmosphere` metres
    above the HIGHEST ground, so the shallowest air column anywhere in the
    domain is `atmosphere` -- not the deepest.
    """
    if z_max < z_min:
        raise ValueError(f"z_max {z_max} is below z_min {z_min}")
    target = (z_max - z_min) + atmosphere
    ratio = solve_ratio(dz0, n_cell[2], target)
    height = column_height(dz0, ratio, n_cell[2])
    return {
        "n_cell": tuple(n_cell),
        "prob_lo": (0.0, 0.0, z_min),
        "prob_hi": (domain, domain, z_min + height),
        "dz0": dz0,
        "stretching_ratio": ratio,
        "max_grid_size": max_grid_size,
    }


# ---------------------------------------------------------------------------
# The guard the solver does not have
# ---------------------------------------------------------------------------

def assert_fits(z, prob_lo_z, prob_hi_z, what="terrain", tol=None):
    """Refuse terrain that does not straddle the domain.

    The solver will not tell you. Terrain below prob_lo[2] leaves every cell
    in that column fluid -- the surface is under the mesh and the immersed
    boundary does nothing. Terrain above prob_hi[2] marks the whole column
    solid. Neither raises, neither warns, and both produce output that looks
    entirely reasonable.
    """
    import numpy as np

    z = np.asarray(z)
    tol = FIT_TOL_M if tol is None else tol
    lo, hi = float(np.min(z)), float(np.max(z))
    if lo < prob_lo_z - tol:
        raise ValueError(
            f"{what} reaches {lo:.2f} m, below the domain floor "
            f"{prob_lo_z:.2f} m. Those columns would come out entirely "
            f"FLUID with the ground beneath the mesh, silently -- the "
            f"immersed boundary would do nothing there.")
    if hi > prob_hi_z + tol:
        raise ValueError(
            f"{what} reaches {hi:.2f} m, above the domain top "
            f"{prob_hi_z:.2f} m. Those columns would come out entirely "
            f"SOLID, silently.")


def assert_straddles(solver, what="the case"):
    """After a setup(): the domain really does contain a ground surface.

    n_solid == 0 means the terrain is under the mesh; n_solid == n_total
    means it is over it. Both are the silent failures assert_fits guards
    against, checked here against what the solver actually built rather than
    against what we handed it.
    """
    mask = solver.mask
    n_solid = int((mask == 1).sum())
    n_total = int(mask.size)
    if n_solid == 0:
        raise ValueError(
            f"{what}: no solid cells at all -- the terrain is below the "
            f"domain floor and the immersed boundary is doing nothing.")
    if n_solid == n_total:
        raise ValueError(
            f"{what}: every cell is solid -- the terrain is above the "
            f"domain top.")
    return n_solid, n_total


# ---------------------------------------------------------------------------
# Terrain files
#
# Written here rather than by the vendored write_terrain_xyz, because the
# coordinates have to be shifted first. The format follows this repository's
# own convention (tools/make_terrain.py): a commented header, then x,y,z.
# Terrain::ReadPointFile strips '#' comments and accepts commas or
# whitespace, so either style reads; matching make_terrain.py keeps one
# terrain format in the project rather than two.
# ---------------------------------------------------------------------------

TERRAIN_FILE = "terrain.csv"
SURVEY_FILE = "survey.json"
CASE_FILE = "case.json"

_DECIMALS = 4      # 0.1 mm; SRTM is integer metres before interpolation

#: How far terrain may sit outside the domain before it counts as a real
#: fit failure rather than a rounding artefact.
#:
#: write_terrain rounds to _DECIMALS places, so a point read back from a
#: terrain file can differ from the value the grid was derived from by half
#: a quantum. On coastal_fire:20 that was enough to put the terrain 3.3e-05
#: m below its own floor and abort a dataset run 28 solves in -- the guard
#: working exactly as intended, on an inconsistency that is not a geometry
#: error at all.
#:
#: It is luck which windows this hits: a minimum that rounds DOWN trips,
#: one that rounds up does not. coastal_fire's z_min is 0.3333... and
#: rounded down; most others rounded up.
#:
#: The tolerance is the quantum itself, not a fudge factor. What the guard
#: is for -- terrain below the mesh, or a column entirely solid -- is
#: hundreds of metres out, so nothing it catches is lost.
FIT_TOL_M = 10.0 ** (-_DECIMALS)



def write_terrain(path, points, header_lines=()):
    """Write an (n, 3) point cloud as a FastWindTerrain terrain file."""
    import numpy as np

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"expected an (n, 3) array, got {points.shape}")

    fmt = "%.{0}f,%.{0}f,%.{0}f\n".format(_DECIMALS)
    with open(path, "w") as f:
        f.write("# FastWindTerrain terrain -- real SRTM elevation\n")
        f.write("# generated by cases/casegen.py; do not edit by hand\n")
        f.write("# x,y are LOCAL metres on [0, %g]; z is ABSOLUTE metres "
                "above sea level\n" % DOMAIN_M)
        for line in header_lines:
            f.write("# %s\n" % line)
        f.write("x,y,z\n")
        for x, y, z in points:
            f.write(fmt % (x, y, z))
    return path


def read_terrain(path):
    """An (n, 3) array back.

    Parsed the way Terrain::ReadPointFile does it (Source/Terrain.cpp:42):
    strip anything after '#', treat commas as separators, and skip any line
    that does not yield three numbers. That is what lets the 'x,y,z' header
    through harmlessly, and it means Python and C++ read the same file the
    same way rather than one of them relying on a header line count that
    the other does not know about.
    """
    import numpy as np

    rows = []
    with open(path) as f:
        for line in f:
            fields = line.split("#", 1)[0].replace(",", " ").split()
            if len(fields) != 3:
                continue
            try:
                rows.append([float(v) for v in fields])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"no terrain points found in {path}")
    return np.array(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# Fetching a tile
# ---------------------------------------------------------------------------

def _fetch_tile(srtm, box, tif_path):
    """Download the SRTM window, and clip it without needing a GDAL CLI.

    The vendored download_srtm calls elevation.clip, which shells out to
    gdal_translate through a Makefile. That is fine when the system GDAL
    binaries are healthy and is a common thing for them not to be -- on the
    machine this was written on, gdal_translate aborts because Homebrew's
    libheif references an x265 dylib a later upgrade removed. Nothing about
    that is our terrain's fault, and it is not something a user of this
    catalogue should have to debug.

    So: try the vendored path first, and on failure fall back to seeding the
    tiles with the same library and doing the window read with rasterio,
    which ships its own GDAL and never touches the command line. Either way
    the result is a GeoTIFF for the same bounds, handed on to the vendored
    tiff_to_xyz_utm so the elevation processing is identical.
    """
    try:
        srtm.download_srtm(box["lat_min"], box["lat_max"],
                           box["lon_min"], box["lon_max"], tif_path)
        return "elevation.clip"
    except Exception as exc:
        sys.stderr.write(
            "note: elevation.clip failed (%s: %s); falling back to a "
            "rasterio window read, which needs no GDAL command line\n"
            % (type(exc).__name__, str(exc).splitlines()[0][:120]))

    import elevation
    import elevation.datasource
    import rasterio
    from rasterio.merge import merge

    bounds = (box["lon_min"], box["lat_min"], box["lon_max"], box["lat_max"])

    # Which one-degree tiles the window needs, asked of the same library
    # that names them, so the two agree about the naming convention.
    required = list(elevation.datasource.srtm1_tiles_names(*bounds))
    paths = [_ensure_hgt(name) for name in required]

    datasets = [rasterio.open(p) for p in paths]
    try:
        data, transform = merge(datasets, bounds=bounds)
        nodata = datasets[0].nodata
        crs = datasets[0].crs
    finally:
        for d in datasets:
            d.close()
    data = data[0]

    if data.size == 0:
        raise RuntimeError(
            "the requested window is empty -- the bounds %r do not overlap "
            "the tiles %s" % (bounds, required))

    # THE CHECK THAT MATTERS. Nodata in SRTM is -32768, and the vendored
    # tiff_to_xyz_utm does z = where(z < 0, 0, z) -- so a window with no
    # real data behind it becomes a flawless sea-level plain: terrain that
    # is perfectly smooth, perfectly flat, and perfectly wrong. Loud here
    # beats plausible later.
    if nodata is not None:
        bad = float((data == nodata).mean())
        if bad > 0.01:
            raise RuntimeError(
                "%.1f%% of the window %r is nodata across tiles %s. The "
                "data is missing or corrupt; delete the cached tiles and "
                "re-run rather than trusting this."
                % (100.0 * bad, bounds, required))

    profile = {
        "driver": "GTiff", "dtype": data.dtype, "count": 1,
        "height": data.shape[0], "width": data.shape[1],
        "transform": transform, "crs": crs, "nodata": nodata,
    }
    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(data, 1)
    return "rasterio"


# Where the raw one-degree tiles are kept. Shared with the elevation
# package's own cache root so a machine does not end up with two copies.
_HGT_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi/{slat}/{tile}.gz"


def _hgt_cache_dir():
    import elevation
    return os.path.join(elevation.CACHE_DIR, "SRTM1", "hgt")


def _ensure_hgt(tile_name):
    """One SRTM1 tile as a local .hgt, downloading it if need be.

    tile_name arrives as 'N37/N37W120.tif' -- the elevation package's
    GeoTIFF naming. We want the raw .hgt behind it, because turning .hgt
    into .tif is exactly the step that needs gdal_translate. rasterio reads
    .hgt directly through its own GDAL, so skipping the conversion skips
    the whole class of broken-toolchain failure.
    """
    import gzip
    import shutil
    import urllib.request

    slat = os.path.dirname(tile_name)
    stem = os.path.splitext(os.path.basename(tile_name))[0]
    cache = os.path.join(_hgt_cache_dir(), slat)
    if not os.path.isdir(cache):
        os.makedirs(cache)

    hgt = os.path.join(cache, stem + ".hgt")
    if os.path.isfile(hgt) and os.path.getsize(hgt) > 0:
        return hgt

    url = _HGT_URL.format(slat=slat, tile=stem + ".hgt")
    sys.stderr.write("downloading %s ...\n" % url)
    tmp = hgt + ".part"
    try:
        with urllib.request.urlopen(url, timeout=300) as response, \
                gzip.GzipFile(fileobj=response) as gz, \
                open(tmp, "wb") as out:
            shutil.copyfileobj(gz, out)
        if os.path.getsize(tmp) == 0:
            raise RuntimeError("the download produced an empty file")
        os.rename(tmp, hgt)
    except Exception:
        # Never leave a partial tile behind: a zero-byte or truncated file
        # that looks cached is how this failed in the first place.
        if os.path.isfile(tmp):
            os.remove(tmp)
        raise
    return hgt


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

class Case(object):
    """One wildfire location, and the 5 x 5 km domain centred on it."""

    def __init__(self, slug, name, state, year, lat, lon,
                 area_ha=None, duration_days=None):
        self.slug = slug
        self.name = name
        self.state = state
        self.year = year
        self.lat = lat
        self.lon = lon
        self.area_ha = area_ha
        self.duration_days = duration_days

    def __repr__(self):
        return ("Case(%s, %s %d, %.4f, %.4f)"
                % (self.slug, self.state, self.year, self.lat, self.lon))

    # -- paths -------------------------------------------------------------

    @property
    def folder(self):
        return os.path.join(HERE, self.slug)

    @property
    def terrain_path(self):
        return os.path.join(self.folder, TERRAIN_FILE)

    @property
    def survey_path(self):
        return os.path.join(self.folder, SURVEY_FILE)

    # -- geometry ----------------------------------------------------------

    def bbox_deg(self, half_width=DOWNLOAD_HALF_WIDTH_M):
        """The lat/lon box to download.

        Wider than the domain by design -- see DOWNLOAD_HALF_WIDTH_M, which
        is sized so the vendored reader's border smoothing stays outside the
        5 km domain entirely. The box is generous and approximate because it
        is cut to size exactly in UTM afterwards; only its being large
        enough matters.
        """
        dlat = half_width / _M_PER_DEG_LAT
        dlon = dlat / math.cos(math.radians(self.lat))
        return {
            "lat_min": self.lat - dlat, "lat_max": self.lat + dlat,
            "lon_min": self.lon - dlon, "lon_max": self.lon + dlon,
        }

    def untouched_interior_m(self, half_width=DOWNLOAD_HALF_WIDTH_M):
        """Width of the tile the border smoothing does NOT reach.

        Must exceed DOMAIN_M or the domain's own edges are smoothed. Checked
        by the tests, so a future change to the download width cannot
        quietly start flattening the terrain.
        """
        return 2.0 * half_width * (1.0 - 2.0 * _SMOOTHED_FRACTION)

    # -- terrain -----------------------------------------------------------

    def download_points(self, subsample=1, tif_path=None,
                        half_width=DOWNLOAD_HALF_WIDTH_M, extent=None):
        """Download SRTM, project, clip, and shift into local coordinates.

        Returns an (n, 3) array: x and y local metres, z absolute metres
        above sea level. Needs the geo stack; nothing else here does.

        `extent` is the width of the square the local origin is placed on,
        defaulting to one domain. The corpus (cases/corpus.py) passes a
        larger one so several 5 km windows can be cut out of a single tile;
        `half_width` grows with it so the vendored reader's border smoothing
        still lands outside every window. HALO_M of extra points is kept
        beyond the extent on each side either way, so the interpolation at a
        boundary averages real neighbours instead of extrapolating.
        """
        import numpy as np

        sys.path.insert(0, HERE)
        import srtm_terrain_reader as srtm

        extent = DOMAIN_M if extent is None else float(extent)
        box = self.bbox_deg(half_width=half_width)
        utm_x, utm_y, z = None, None, None
        tmp = None
        if tif_path is None:
            import tempfile
            handle = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
            tif_path = handle.name
            handle.close()
            tmp = tif_path
        try:
            _fetch_tile(srtm, box, tif_path)
            # This applies _fix_srtm_zeros and _smooth_terrain_border; the
            # download width is chosen so the smoothed band falls outside
            # the domain.
            utm_x, utm_y, z = srtm.tiff_to_xyz_utm(tif_path,
                                                   subsample=subsample)
        finally:
            if tmp is not None and os.path.isfile(tmp):
                os.remove(tmp)

        # The domain centre, in the same projection the tile was put into.
        cx, cy = srtm._latlon_to_utm(np.array([self.lat]),
                                     np.array([self.lon]))
        cx, cy = float(cx[0]), float(cy[0])

        half = 0.5 * extent + HALO_M
        x, y, z = utm_x.ravel(), utm_y.ravel(), z.ravel()
        keep = ((np.abs(x - cx) <= half) & (np.abs(y - cy) <= half))
        if not keep.any():
            raise RuntimeError(
                "the downloaded tile does not overlap the domain centre; "
                "the projection or the bounding box is wrong")

        # Local origin at the extent's own corner, so the extent is exactly
        # [0, extent] and the halo points fall just outside it.
        x0, y0 = cx - 0.5 * extent, cy - 0.5 * extent
        return np.column_stack([x[keep] - x0, y[keep] - y0, z[keep]])

    # -- survey ------------------------------------------------------------

    def survey_from_points(self, points, subsample=1):
        """The few hundred bytes worth committing about a downloaded tile.

        Enough to rebuild the grid offline: the extremes drive it, and the
        rest is provenance.
        """
        import numpy as np

        z = np.asarray(points)[:, 2]
        return {
            "slug": self.slug,
            "name": self.name,
            "lat": self.lat,
            "lon": self.lon,
            "n_points": int(len(z)),
            "subsample": int(subsample),
            "z_min": float(np.min(z)),
            "z_max": float(np.max(z)),
            "z_mean": float(np.mean(z)),
            "relief": float(np.max(z) - np.min(z)),
            "domain_m": DOMAIN_M,
            "halo_m": HALO_M,
            "download_half_width_m": DOWNLOAD_HALF_WIDTH_M,
            "source": "SRTM1 via cases/srtm_terrain_reader.py",
        }

    def read_survey(self):
        with open(self.survey_path) as f:
            return json.load(f)

    def prepare(self, subsample=1, download=True, tif_path=None):
        """Download the terrain, survey it, and write both files.

        With download=False the terrain file is left alone and only the
        survey's derived grid is re-checked -- which is how the catalogue's
        grids stay reproducible with no network.
        """
        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)

        if download:
            points = self.download_points(subsample=subsample,
                                          tif_path=tif_path)
            if float(points[:, 2].max() - points[:, 2].min()) <= 0.0:
                raise RuntimeError(
                    f"{self.slug}: the downloaded terrain is perfectly flat "
                    f"(every point at {points[0, 2]:.1f} m). Real ground is "
                    f"not, so this is missing or nodata elevation data, not "
                    f"a plain. Nothing has been written.")
            survey = self.survey_from_points(points, subsample=subsample)
            grid = grid_from_relief(survey["z_min"], survey["z_max"])

            # Before anything is written: the guard the solver does not have.
            assert_fits(points[:, 2], grid["prob_lo"][2], grid["prob_hi"][2],
                        what=f"{self.slug} terrain")

            survey["grid"] = _jsonable_grid(grid)
            write_terrain(self.terrain_path, points, header_lines=[
                f"{self.name} ({self.state} {self.year}) "
                f"at {self.lat:.4f}, {self.lon:.4f}",
                f"{survey['n_points']} points, "
                f"z {survey['z_min']:.1f} to {survey['z_max']:.1f} m ASL "
                f"(relief {survey['relief']:.1f} m)",
            ])
            with open(self.survey_path, "w") as f:
                json.dump(survey, f, indent=2, sort_keys=True)
                f.write("\n")
        else:
            survey = self.read_survey()
            grid = grid_from_relief(survey["z_min"], survey["z_max"])
            survey["grid"] = _jsonable_grid(grid)

        return survey

    # -- the solver config -------------------------------------------------

    def grid(self):
        """The grid section, from the committed survey. No network."""
        survey = self.read_survey()
        return grid_from_relief(survey["z_min"], survey["z_max"])

    def config(self, wind_speed=8.0, wind_direction=225.0, points=None,
               n_projections=4, alpha_v=0.5, anisotropy=True, obrien=True,
               **sections):
        """A complete fwt.Solver config for this case.

        `wind_direction` is meteorological: the direction the wind comes
        FROM, in degrees clockwise from north. 225 is a southwesterly. The
        inflow section takes components, so the conversion happens here
        rather than being left as a trap for the caller.

        `points` defaults to reading the case's terrain file; pass an array
        to drive the case from terrain already in memory.
        """
        import numpy as np

        if points is None:
            if not os.path.isfile(self.terrain_path):
                raise FileNotFoundError(
                    f"{self.terrain_path} does not exist. Terrain files are "
                    f"large and are not committed -- run\n"
                    f"    python3 {os.path.join(self.slug, 'prepare.py')}\n"
                    f"to download it.")
            points = read_terrain(self.terrain_path)

        grid = self.grid()
        assert_fits(np.asarray(points)[:, 2],
                    grid["prob_lo"][2], grid["prob_hi"][2],
                    what=f"{self.slug} terrain")

        theta = math.radians(wind_direction)
        cfg = {
            "grid": grid,
            "terrain": {"points": np.asarray(points, dtype=np.float64)},
            "inflow": {
                "mode": "powerlaw",
                "u_ref": -wind_speed * math.sin(theta),
                "v_ref": -wind_speed * math.cos(theta),
            },
            "anisotropy": {"enable": bool(anisotropy)},
            "obrien": {"enable": bool(obrien)},
            "poisson": {"alpha_v": alpha_v, "n_projections": n_projections},
        }
        for name, section in sections.items():
            cfg.setdefault(name, {}).update(section)
        return cfg


def _jsonable_grid(grid):
    return {k: (list(v) if isinstance(v, tuple) else v)
            for k, v in grid.items()}


# ---------------------------------------------------------------------------
# What the generated scripts call
# ---------------------------------------------------------------------------

def print_survey(case, survey, stream=None):
    """Report a prepared case, the way make_terrain.py reports a terrain."""
    out = stream if stream is not None else sys.stderr
    grid = survey.get("grid") or _jsonable_grid(
        grid_from_relief(survey["z_min"], survey["z_max"]))
    print(f"{case.name} ({case.state} {case.year}) "
          f"at {case.lat:.4f}, {case.lon:.4f}", file=out)
    print(f"  terrain    {survey['n_points']} points, "
          f"subsample {survey.get('subsample', 1)}", file=out)
    print(f"  elevation  {survey['z_min']:.1f} to {survey['z_max']:.1f} m ASL"
          f"   (relief {survey['relief']:.1f} m)", file=out)
    print(f"  domain     {DOMAIN_M:.0f} x {DOMAIN_M:.0f} m, "
          f"{grid['n_cell'][0]} x {grid['n_cell'][1]} x {grid['n_cell'][2]} "
          f"cells   ({DOMAIN_M / grid['n_cell'][0]:.0f} m horizontally)",
          file=out)
    print(f"  column     {grid['prob_lo'][2]:.1f} to {grid['prob_hi'][2]:.1f} "
          f"m ASL, dz0 {grid['dz0']:.1f} m, "
          f"ratio {grid['stretching_ratio']:.6f}", file=out)
    print(f"  air above the highest ground: "
          f"{grid['prob_hi'][2] - survey['z_max']:.0f} m", file=out)


def plot_xz_plane(case, solver, path, wind_speed, wind_direction,
                  air_above=400.0):
    """The x-z plane through the middle of the domain, on its own.

    Two panels over the same slice: wind speed, and vertical velocity.

    They answer different questions. Speed shows where the air accelerates
    -- over a crest, through a saddle -- and where it stalls. Vertical
    velocity shows the terrain doing the work: rising on the windward
    slope, sinking in the lee. A mass-consistent solve that produced no w
    over real ground would not be wrong by a little.

    Speed gets one hue, light to dark, because it is a magnitude. w gets a
    diverging map with a neutral midpoint and LIMITS SYMMETRIC ABOUT ZERO,
    so that the colour zero maps to is the colour of no vertical motion --
    an asymmetric range would paint still air as a weak updraught.

    The panel is drawn stretched, because 5 km of terrain in 600 m of air
    is otherwise a thin strip. That exaggerates every slope, so the factor
    is written in the axis label rather than left for the reader to guess.
    """
    import matplotlib
    matplotlib.use("Agg")                 # no display on a build machine
    import matplotlib.pyplot as plt
    import numpy as np

    z_cc = np.asarray(solver.grid.z_cc)
    nz, ny, nx = solver.shape
    j = ny // 2

    u, v, w = solver.velocity
    solid = solver.mask[:, j, :] == 1
    ground = solver.z_terrain[0, j, :]
    dx = DOMAIN_M / nx
    x = (np.arange(nx) + 0.5) * dx

    speed = np.ma.masked_where(
        solid, np.sqrt(u * u + v * v + w * w)[:, j, :])
    w_slice = np.ma.masked_where(solid, w[:, j, :])
    # Crop to the air, not the domain. On a high-relief case the floor sits
    # a kilometre below the lowest ground in this slice, and drawing all of
    # it spends two thirds of the panel on solid rock.
    top = float(ground.max()) + air_above
    base = float(ground.min()) - 0.04 * (top - float(ground.min()))

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.4), sharex=True)

    # Symmetric about zero, so the midpoint colour means no vertical motion.
    w_lim = float(np.abs(w_slice).max())

    panels = (
        (axes[0], speed, "viridis", None, None, "|U|  [m/s]",
         "Wind speed"),
        (axes[1], w_slice, "RdBu_r", -w_lim, w_lim, "w  [m/s]",
         "Vertical velocity  (red rising, blue sinking)"),
    )
    for ax, field, cmap, vmin, vmax, label, title in panels:
        mesh = ax.pcolormesh(x, z_cc, field, shading="nearest", cmap=cmap,
                             vmin=vmin, vmax=vmax)
        ax.fill_between(x, base, ground, color="#3a2e26", zorder=3)
        ax.plot(x, ground, color="#14100c", lw=1.3, zorder=4)
        ax.set_ylim(base, top)
        ax.set_xlim(0, DOMAIN_M)
        ax.set_ylabel("elevation [m ASL]")
        ax.set_title(title, fontsize=11, loc="left")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)      # recessive axes
        fig.colorbar(mesh, ax=ax, label=label, pad=0.015)

    # In-plane vectors on the speed panel only: on the w panel they would
    # sit on top of the quantity they are made of.
    si, sk = max(1, nx // 30), max(1, nz // 20)
    xq, zq = np.meshgrid(x[::si], z_cc[::sk])
    axes[0].quiver(xq, zq,
                   np.ma.masked_where(solid[::sk, ::si],
                                      u[:, j, :][::sk, ::si]),
                   np.ma.masked_where(solid[::sk, ::si],
                                      w[:, j, :][::sk, ::si]),
                   color="white", alpha=0.7, scale=190, width=0.0020,
                   zorder=5)

    # The picture is stretched; say by how much rather than let the reader
    # infer a slope that is not there.
    box = axes[1].get_position()
    exaggeration = ((top - base) / DOMAIN_M) / (box.height / box.width)
    axes[1].set_xlabel(f"x [m]     (vertical exaggeration "
                       f"{1.0 / exaggeration:.0f}x)")

    fig.suptitle(
        f"{case.name} -- {case.state} {case.year}   |   x-z plane at "
        f"y = {(j + 0.5) * dx:.0f} m (mid-domain)\n"
        f"{wind_speed:.0f} m/s from {wind_direction:.0f} deg   |   "
        f"terrain {ground.min():.0f}-{ground.max():.0f} m ASL along this "
        f"slice", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=145)
    plt.close(fig)
    return path


def plot_midslice(case, solver, path, wind_speed, wind_direction,
                  air_above=400.0):
    """A vertical slice through the middle of the domain, and a plan view.

    The slice is the picture worth having for terrain flow: it shows where
    the air speeds up over a ridge, where it separates in the lee, and --
    because the terrain is drawn from the solver's own mask rather than
    from the point cloud -- whether the immersed boundary is where it
    should be. A slice that showed flow underground would say more than any
    assertion.

    Cropped to `air_above` metres over the highest ground in the slice. The
    domain carries a kilometre of atmosphere so the top boundary is nowhere
    near the terrain; drawing all of it spends the picture on uniform flow
    and squashes the layer that is actually doing something.
    """
    import matplotlib
    matplotlib.use("Agg")                 # no display on a build machine
    import matplotlib.pyplot as plt
    import numpy as np

    grid = solver.grid
    z_cc = np.asarray(grid.z_cc)
    nz, ny, nx = solver.shape
    j = ny // 2

    u, v, w = solver.velocity
    mask = solver.mask
    dx = DOMAIN_M / nx
    x = (np.arange(nx) + 0.5) * dx

    solid = mask[:, j, :] == 1
    speed = np.sqrt(u * u + v * v + w * w)
    slice_speed = np.ma.masked_where(solid, speed[:, j, :])
    ground = solver.z_terrain[0, j, :]

    # A plan view at a fixed height ABOVE GROUND, not at a fixed altitude:
    # the ground moves hundreds of metres across the domain, so a constant
    # elevation would be near the surface on a ridge and high aloft over a
    # valley -- and, worse, inside the rock wherever the terrain rose above
    # it. The level is therefore chosen per column, from the full 2-D
    # terrain, and anything that still lands in a solid cell is masked
    # rather than plotted as a convincing patch of calm air.
    target_agl = 100.0
    ground_2d = solver.z_terrain[0]                       # (ny, nx)
    k_agl = np.abs(z_cc[:, None, None]
                   - (ground_2d[None, :, :] + target_agl)).argmin(axis=0)
    plan = np.take_along_axis(speed, k_agl[None], axis=0)[0]
    plan = np.ma.masked_where(
        np.take_along_axis(mask, k_agl[None], axis=0)[0] == 1, plan)

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(8.6, 10.4),
        gridspec_kw={"height_ratios": [1.0, 1.55]})

    top = float(ground.max()) + air_above
    base = float(ground.min()) - 0.04 * (top - float(ground.min()))
    mesh = ax.pcolormesh(x, z_cc, slice_speed, shading="nearest",
                         cmap="viridis")
    ax.fill_between(x, base, ground, color="#3a2e26", zorder=3)
    ax.plot(x, ground, color="#1a1410", lw=1.2, zorder=4)

    # In-plane vectors: (u, w). Subsampled, or the arrows bury the field.
    si, sk = max(1, nx // 28), max(1, nz // 18)
    xq, zq = np.meshgrid(x[::si], z_cc[::sk])
    uq = np.ma.masked_where(solid[::sk, ::si], u[:, j, :][::sk, ::si])
    wq = np.ma.masked_where(solid[::sk, ::si], w[:, j, :][::sk, ::si])
    ax.quiver(xq, zq, uq, wq, color="white", alpha=0.65, scale=170,
              width=0.0022, zorder=5)

    ax.set_ylim(base, top)
    ax.set_xlim(0, DOMAIN_M)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("elevation [m ASL]")
    ax.set_title(f"Vertical slice through the middle of the domain "
                 f"(y = {(j + 0.5) * dx:.0f} m)")
    fig.colorbar(mesh, ax=ax, label="|U| [m/s]", pad=0.02)

    pm = ax2.pcolormesh(x, x, plan, shading="nearest", cmap="viridis")
    levels = ax2.contour(x, x, solver.z_terrain[0], colors="white",
                         linewidths=0.45, alpha=0.55, levels=10)
    ax2.axhline((j + 0.5) * dx, color="#ff5555", lw=1.1, ls="--")
    ax2.set_aspect("equal")
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    ax2.set_title(f"|U| at {target_agl:.0f} m above ground\n"
                  f"terrain contours in white; dashed line is the slice",
                  fontsize=11)
    fig.colorbar(pm, ax=ax2, label="|U| [m/s]", pad=0.02)
    del levels

    fig.suptitle(
        f"{case.name} -- {case.state} {case.year} -- "
        f"{wind_speed:.0f} m/s from {wind_direction:.0f} deg",
        fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def run_case(case, wind_speed=8.0, wind_direction=225.0, n_projections=4,
             plotfile=None, figure=None, plane=None, stream=None):
    """Solve one case and report the wind, with the sanity checks that
    matter for real terrain.

    Returns 0 if every check passed, 1 otherwise -- so a case folder's
    run.py is usable from a shell script or CI without parsing its output.

    The checks are not "did it run". They are the things that go wrong
    quietly on real ground:

      straddles     0 < n_solid < n_total. All-fluid means the terrain is
                    below the mesh; all-solid means it is above it. The
                    solver reports neither.
      finite        no NaN or inf anywhere in the velocity. An IDW over a
                    point cloud with a coincident pair, or a solve that
                    diverged, shows up here and nowhere else.
      divergence    max|div(u)| falls across the projection. On flat ground
                    it can start at zero and legitimately not move; over
                    real terrain it cannot.
      vertical      w is not identically zero. Real terrain must push air
                    up and down; a w of exactly zero means the terrain is
                    not reaching the flow, which is what an all-fluid mask
                    looks like from the other side.
      speed-up      the fastest air is faster than the reference wind.
                    Flow accelerates over ridges; if the maximum equals the
                    inflow, nothing is happening.
    """
    import numpy as np

    out = stream if stream is not None else sys.stdout
    failures = []

    def check(name, ok, detail):
        mark = "ok  " if ok else "FAIL"
        print(f"  [{mark}] {name:11s} {detail}", file=out)
        if not ok:
            failures.append(name)

    import fastwindterrain as fwt

    cfg = case.config(wind_speed=wind_speed, wind_direction=wind_direction,
                      n_projections=n_projections)
    grid = cfg["grid"]

    print(f"{case.name} -- {wind_speed:.1f} m/s from "
          f"{wind_direction:.0f} deg", file=out)
    print(f"  domain {grid['prob_lo'][2]:.0f} to {grid['prob_hi'][2]:.0f} m "
          f"ASL, {cfg['terrain']['points'].shape[0]} terrain points",
          file=out)

    with fwt.session():
        s = fwt.Solver(cfg)
        s.setup()
        div_before = float(s.max_divergence_fe)
        s.solve()
        s.diagnose()

        mask = s.mask
        fluid = mask == 0
        u, v, w = s.velocity
        speed = np.sqrt(u * u + v * v + w * w)

        n_solid = int((mask == 1).sum())
        n_total = int(mask.size)
        div_after = float(s.max_divergence_fe)

        print(file=out)
        check("straddles", 0 < n_solid < n_total,
              f"{n_solid} of {n_total} cells solid "
              f"({100.0 * n_solid / n_total:.1f}%)")
        check("finite", bool(np.all(np.isfinite(s.velocity))),
              f"|U| {speed[fluid].min():.3f} to {speed[fluid].max():.3f} m/s")
        check("divergence", div_after < div_before,
              f"max|div| {div_before:.4g} -> {div_after:.4g} 1/s "
              f"over {s.n_projections_done} passes")
        check("vertical", float(np.abs(w[fluid]).max()) > 0.0,
              f"w {w[fluid].min():+.3f} to {w[fluid].max():+.3f} m/s")
        check("speed-up", float(speed[fluid].max()) > wind_speed,
              f"{speed[fluid].max() / wind_speed:.2f}x the reference wind")

        print(file=out)
        print(f"  MLMG       {s.solve_iterations} iterations, "
              f"residual {s.solve_residual:.3e}", file=out)
        print(f"  horizontal |U_h| max "
              f"{float(np.sqrt(u * u + v * v)[fluid].max()):.3f} m/s", file=out)

        if figure:
            plot_midslice(case, s, figure, wind_speed, wind_direction)
            print(f"  wrote      {figure}", file=out)
        if plane:
            plot_xz_plane(case, s, plane, wind_speed, wind_direction)
            print(f"  wrote      {plane}", file=out)
        if plotfile:
            s.write_plotfile(plotfile)
            print(f"  wrote      {plotfile}", file=out)

    if failures:
        print(f"\n{len(failures)} check(s) failed: {failures}", file=out)
        return 1
    print("\nAll checks passed.", file=out)
    return 0
