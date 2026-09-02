=====================
Real terrain cases
=====================

``cases/`` is a catalogue of eight 5 × 5 km domains over real ground, at
wildfire locations taken from the ``wildfire_levelset`` reference list.
Every other case in this repository is synthetic -- a Gaussian hill or a
linear slope from ``tools/make_terrain.py`` -- and a surrogate trained on
those learns Gaussian hills.

Each case is a folder with its own scripts::

    python3 cases/creek_fire/prepare.py     # download SRTM, derive the grid
    python3 cases/creek_fire/run.py         # solve it
    python3 cases/creek_fire/run.py --plane xz.png

Everything runs through the Python bindings (:doc:`python`); no inputs
file is generated and ParmParse is never involved, so one case in a
process cannot inherit another's settings.

The eight
=========

Chosen for recurrent fire and for spread across wind regime and relief --
eight variations of one canyon would teach a surrogate that canyon.

.. list-table::
   :widths: 22 12 66
   :header-rows: 1

   * - Case
     - Relief
     - Why it is in the set
   * - ``dixie_fire``
     - extreme
     - Feather River canyon, northern Sierra/Cascade transition
   * - ``creek_fire``
     - 1128 m
     - Sierra NF; the high-elevation case, ground at 1334–2462 m ASL
   * - ``rim_fire``
     - large
     - Stanislaus NF above the Tuolumne canyon
   * - ``august_complex_fire``
     - moderate
     - Mendocino NF, inner Coast Ranges
   * - ``tubbs_fire``
     - moderate
     - Sonoma/Napa, the Diablo wind corridor
   * - ``thomas_fire``
     - steep
     - Ventura / Santa Ynez front, Santa Ana and sundowner
   * - ``woolsey_fire``
     - moderate
     - Santa Monica Mountains, a Santa Ana corridor
   * - ``bootleg_fire``
     - 221 m
     - Fremont–Winema, interior Oregon plateau; the low-relief contrast

``python3 cases/build_catalogue.py --list`` prints the set with
coordinates. The reference CSV holds twenty-nine fires; the other
twenty-one stay in it, and adding one back is a line in ``CATALOGUE``.

The grid follows the terrain
============================

Cell counts are the same for every case -- ``(100, 100, 60)``, so 50 m
horizontally and one tensor shape throughout -- but the vertical extent
is derived from the relief the download actually finds:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Quantity
     - Value
   * - ``prob_lo[2]``
     - the tile's minimum elevation
   * - ``prob_hi[2]``
     - that, plus relief and 1000 m of air above the **highest** ground
   * - ``stretching_ratio``
     - solved so the column reaches exactly that top
   * - ``dz0``
     - 4 m at the surface

So Creek Fire gets a domain from 1334 to 3462 m ASL and Bootleg one from
1387 to 2608 m. Elevations stay absolute, which is why a height in the
output is a real height above sea level.

Why the floor is checked, not assumed
-------------------------------------

**Nothing in the solver checks that terrain fits inside the domain.**
``Grid::Params::Validate`` never sees the terrain, and
``Terrain::BuildMask`` simply evaluates ``z_cc <= z_terrain``. So terrain
below ``prob_lo[2]`` leaves every cell fluid with the surface under the
mesh, and terrain above ``prob_hi[2]`` marks every column solid. Neither
warns. Neither raises. Both write a plotfile that looks entirely
reasonable.

SRTM elevations are absolute metres above sea level, and these sites run
from a few hundred metres to over two thousand, so this is the most
likely way the catalogue could produce confident nonsense. ``casegen``
therefore asserts the fit before writing anything, and ``run.py`` asserts
again afterwards against what the solver actually built:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Check
     - What it catches
   * - ``straddles``
     - ``0 < n_solid < n_total``. All-fluid means the terrain is below the
       mesh; all-solid means it is above it
   * - ``finite``
     - a NaN anywhere in the velocity
   * - ``divergence``
     - ``max|div(u)|`` falls across the projection
   * - ``vertical``
     - ``w`` is not identically zero. Real terrain must push air up and
       down; exactly zero is what an all-fluid mask looks like from the
       other side
   * - ``speed-up``
     - the fastest air beats the reference wind, as it must over a ridge

This is not hypothetical. The first Creek Fire download returned a tile of
nodata, which the elevation reader turns into a flawless sea-level plain;
all three of ``straddles``, ``divergence`` and ``vertical`` failed rather
than producing a plausible answer over terrain that did not exist.

Terrain
=======

``prepare.py`` downloads SRTM1 through ``cases/srtm_terrain_reader.py``,
vendored from ``wildfire_levelset``. It projects to UTM, then:

* **horizontal coordinates are shifted** to local metres on ``[0, 5000]``.
  Inverse-distance weighting on seven-digit UTM northings loses precision
  in the distances it squares.
* **elevations are not shifted**, so the output is directly comparable
  with met data.
* the download box is **8.8 km, not 5 km**. The vendored reader blends a
  Gaussian-smoothed field into the outer 20 % of each side of whatever
  tile it is given; a tighter box would let that smoothing reach into the
  domain and flatten its edges, which is exactly where terrain-driven flow
  separates. 8.8 km leaves 5.28 km untouched, which contains the domain.

Files, and which are committed
------------------------------

``terrain.csv`` is about a megabyte per case and is **gitignored**.
``survey.json`` -- the elevation extremes, the point count and the derived
grid -- is a few hundred bytes and is committed, so every case's grid is
reproducible without a download and ``prepare.py --no-download`` rebuilds
it offline. That is also what lets ``tests/test_cases.py`` run with no
network and no geo stack installed.

Requirements
============

The download needs ``elevation``, ``rasterio``, ``pyproj`` and ``scipy``::

    pip install ".[cases]"

The grid arithmetic needs none of them. If the ``elevation`` package's
GDAL command-line step fails -- a common breakage, since it shells out to
``gdal_translate`` through a Makefile -- ``casegen`` falls back to
fetching the raw ``.hgt`` tiles and reading them with ``rasterio``, which
carries its own GDAL. Either way the elevation processing afterwards is
the vendored reader's, unchanged.

Cost
====

A case is one download of a few seconds and one solve. Measured on the
Creek Fire case (100 × 100 × 60, 71 % solid, 40 000 terrain points):

.. list-table::
   :widths: 40 20 40
   :header-rows: 1

   * - Stage
     - Time
     - Note
   * - ``setup`` (terrain IDW included)
     - 0.8 s
     - the interpolation is 2 % of a sample, not the cost
   * - ``solve``, 4 projection passes
     - 52 s
     - **this is the whole cost**, and it is linear in the pass count
   * - ``diagnose`` + ``fields``
     - 0.04 s
     - negligible

So the lever for a large sweep is ``poisson.n_projections``, not terrain
subsampling: 1 pass is 12.8 s, 2 is 25.8 s, 4 is 52.2 s.

Be aware of what that trades. Over this terrain the extra passes barely
move the divergence -- ``max|div|`` goes 0.1153 → 0.1139 → 0.1018 from
one pass to four -- but they move the *field* a great deal, by 71 % of
``max|u|`` between one pass and four. The projection is approximate and,
on steep partly-blocked terrain, has not converged in the field even
where the residual norm suggests little is left to do. **Fix
``n_projections`` across a dataset and treat it as part of the operator
definition**, because a surrogate trained on such a dataset learns "N
passes of this projection", not "the mass-consistent solution".
