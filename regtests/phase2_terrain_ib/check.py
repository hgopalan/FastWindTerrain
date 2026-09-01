#!/usr/bin/env python3
"""
Phase 2 regtest checker -- terrain surface & immersed-boundary mask.

Runs the FastWindTerrain executable against the Phase 2 input files and
validates:

  inputs_flat      -> no terrain file: z_terrain == 0 everywhere and every
                      cell is fluid
  inputs_hill      -> Gaussian hill on a lattice: z_terrain matches an
                      INDEPENDENT Python IDW interpolation of the same
                      point file, the mask is exactly the binary test
                      z_cc <= z_terrain, and the mask boundary sits in the
                      right cell in every column
  inputs_scattered -> the same hill sampled off-lattice, so the k-nearest
                      search faces irregular spacing

The expected terrain height is recomputed here from the point file rather
than read back from the solver, so the checker does not simply confirm
the solver against itself. The mask is binary (1 solid, 0 fluid); there
are no partial volume fractions.

All cases run in a scratch work directory (default
<repo>/build/regtests/phase2_terrain_ib) so no run artifacts land in the
source tree.

Usage:
    python3 check.py /path/to/fastwindterrain.exe [workdir]

Exits 0 if all cases pass, 1 otherwise (suitable for CI).
"""

import sys
import os
import math
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))       # inputs live here
PHASE = os.path.basename(HERE)
REGTEST_ROOT = os.path.dirname(HERE)
ROOT = os.path.dirname(REGTEST_ROOT)

sys.path.insert(0, REGTEST_ROOT)                        # plotfile.py
sys.path.insert(0, os.path.join(ROOT, "tools"))         # make_terrain.py

from plotfile import Plotfile                           # noqa: E402
from make_terrain import elevation                      # noqa: E402

# Every case runs here; nothing is written next to the inputs.
WORKDIR = os.path.join(ROOT, "build", "regtests", PHASE)

TOL = 1.0e-6          # absolute tolerance in meters for float comparisons
IDW_TOL = 1.0e-9      # C++ vs Python IDW must agree to round-off

# Grid geometry shared by all three cases (see the inputs files).
NX, NY, NZ = 40, 40, 66
XLO, XHI = 0.0, 1000.0
YLO, YHI = 0.0, 1000.0
DX = (XHI - XLO) / NX
DY = (YHI - YLO) / NY

# Terrain shape shared by inputs_hill / inputs_scattered.
PEAK, SIGMA, XC, YC = 100.0, 150.0, 500.0, 500.0

# massconsistent_amr's idw_terrain defaults, which Terrain.cpp ports.
IDW_K = 6
IDW_EXPONENT = 2.0
DISTANCE_EPSILON = 1.0e-12


# ---------------------------------------------------------------------------
# Independent reference implementations
# ---------------------------------------------------------------------------

def read_points(path):
    """Read an x,y,z point file the way the solver does: '#' comments
    stripped, commas treated as separators, unparseable lines skipped."""
    pts = []
    with open(path) as f:
        for line in f:
            line = line.split("#")[0].replace(",", " ").strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                pts.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue          # the "x,y,z" header line
    assert pts, f"no points read from {path}"
    return pts


def idw(xq, yq, pts, k=IDW_K, exponent=IDW_EXPONENT):
    """Inverse-distance-weighted height at (xq, yq) over the k nearest
    points. Independent re-derivation of Terrain::InterpolateIDW; the two
    must agree to round-off."""
    d2 = sorted(((x - xq)**2 + (y - yq)**2, i)
                for i, (x, y, _) in enumerate(pts))
    k = min(k, len(pts))

    wsum = 0.0
    zval = 0.0
    for dist2, i in d2[:k]:
        if dist2 < DISTANCE_EPSILON:
            return pts[i][2]                    # exact hit on an input point
        w = dist2 ** (-exponent / 2.0)
        wsum += w
        zval += w * pts[i][2]
    return zval / wsum


def cell_center_x(i):
    return XLO + (i + 0.5) * DX


def cell_center_y(j):
    return YLO + (j + 0.5) * DY


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def parse_report(path):
    """Parse the plain-text grid report into a dict of scalars plus
    z_face / z_cc arrays."""
    data = {"z_face": {}, "z_cc": {}}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            key = parts[0]
            if key in ("z_face", "z_cc"):
                data[key][int(parts[1])] = float(parts[2])
            elif key == "n_cell":
                data[key] = [int(x) for x in parts[1:]]
            elif key in ("prob_lo", "prob_hi"):
                data[key] = [float(x) for x in parts[1:]]
            else:
                try:
                    data[key] = float(parts[1])
                except ValueError:
                    data[key] = parts[1]
    return data


def run_case(exe, inputs_file, extra=()):
    """Run one case in WORKDIR. Any terrain.file in the inputs is a bare
    name; the absolute path is supplied here so the run directory does
    not have to hold a copy of the point file."""
    cmd = [exe, os.path.join(HERE, inputs_file)] + list(extra)
    return subprocess.run(cmd, cwd=WORKDIR, capture_output=True,
                          text=True, timeout=600)


def terrain_arg(name):
    return [f"terrain.file={os.path.join(HERE, name)}"]


def clean(*names):
    for n in names:
        p = os.path.join(WORKDIR, n)
        if os.path.isdir(p):
            shutil.rmtree(p)          # no stale output can fake a pass
        elif os.path.exists(p):
            os.remove(p)


# ---------------------------------------------------------------------------
# Shared assertions
# ---------------------------------------------------------------------------

def check_mask_is_binary_and_consistent(name, pf):
    """The mask must be exactly the binary test z_cc <= z_terrain, with
    no values other than 0 and 1, in every cell of the domain."""
    zcc = pf.field("z_cc")
    zt = pf.field("terrain_z")
    mask = pf.field("mask")

    bad_value = None
    wrong = []
    for k in range(NZ):
        for j in range(NY):
            for i in range(NX):
                m = mask(i, j, k)
                if m not in (0.0, 1.0):
                    bad_value = (i, j, k, m)
                    break
                expect = 1.0 if (zcc(i, j, k) - zt(i, j, k) <= 0.0) else 0.0
                if m != expect:
                    wrong.append((i, j, k, m, expect,
                                  zcc(i, j, k), zt(i, j, k)))

    assert bad_value is None, (
        f"[{name}] mask must be binary; cell {bad_value[:3]} holds "
        f"{bad_value[3]}")
    assert not wrong, (
        f"[{name}] {len(wrong)} cells disagree with z_cc <= z_terrain; "
        f"first: (i,j,k)={wrong[0][:3]} mask={wrong[0][3]} "
        f"expected={wrong[0][4]} z_cc={wrong[0][5]} z_terrain={wrong[0][6]}")


def check_mask_boundary(name, pf, report):
    """Column by column, the highest solid cell must be the last one whose
    center lies at or below the terrain, and the cell above it must be
    fluid -- i.e. the mask boundary sits in the right cell, not one off."""
    zt = pf.field("terrain_z")
    mask = pf.field("mask")
    z_cc = [report["z_cc"][k] for k in range(NZ)]

    for j in range(0, NY, 7):            # a spread of columns, not all 1600
        for i in range(0, NX, 7):
            h = zt(i, j, 0)
            column = [mask(i, j, k) for k in range(NZ)]

            # Solid cells must form a contiguous block from the ground up.
            n_solid = int(sum(column))
            assert column[:n_solid] == [1.0] * n_solid, (
                f"[{name}] column ({i},{j}) has a hole: solid cells are not "
                f"contiguous from the bottom: {column[:n_solid + 2]}")

            if n_solid > 0:
                assert z_cc[n_solid - 1] <= h + TOL, (
                    f"[{name}] column ({i},{j}): top solid cell center "
                    f"{z_cc[n_solid - 1]} m is above the terrain {h} m")
            if n_solid < NZ:
                assert z_cc[n_solid] > h - TOL, (
                    f"[{name}] column ({i},{j}): first fluid cell center "
                    f"{z_cc[n_solid]} m is below the terrain {h} m")


def check_terrain_against_idw(name, pf, pts, stride=7):
    """z_terrain must match an independent Python IDW of the same points,
    and must be constant along k (it is a 2D field replicated in k)."""
    zt = pf.field("terrain_z")

    worst = 0.0
    worst_at = None
    for j in range(0, NY, stride):
        for i in range(0, NX, stride):
            expect = idw(cell_center_x(i), cell_center_y(j), pts)
            got = zt(i, j, 0)
            err = abs(got - expect)
            if err > worst:
                worst, worst_at = err, (i, j, got, expect)

            # replicated along k
            for k in (1, NZ // 2, NZ - 1):
                assert zt(i, j, k) == got, (
                    f"[{name}] z_terrain varies with k at ({i},{j}): "
                    f"k=0 gives {got}, k={k} gives {zt(i, j, k)}")

    assert worst < IDW_TOL, (
        f"[{name}] z_terrain disagrees with the independent IDW by "
        f"{worst:.3e} m at (i,j)={worst_at[:2]}: solver {worst_at[2]} vs "
        f"reference {worst_at[3]}")
    return worst


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def check_flat(exe):
    name = "inputs_flat"
    clean("plt_flat", "grid_report_flat.txt")

    result = run_case(exe, name)
    assert result.returncode == 0, (
        f"[{name}] expected success (exit 0), got {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")

    report = parse_report(os.path.join(WORKDIR, "grid_report_flat.txt"))
    assert report["terrain_file"] == "none", (
        f"[{name}] expected no terrain file, report says "
        f"{report['terrain_file']}")
    assert report["terrain_n_points"] == 0
    assert abs(report["terrain_z_min"]) < TOL and abs(report["terrain_z_max"]) < TOL, (
        f"[{name}] flat ground must give z_terrain == 0, got "
        f"[{report['terrain_z_min']}, {report['terrain_z_max']}]")
    assert report["terrain_n_solid"] == 0, (
        f"[{name}] flat ground at z = 0 must leave every cell fluid, "
        f"got {report['terrain_n_solid']} solid")

    pf = Plotfile(os.path.join(WORKDIR, "plt_flat"))
    assert pf.var_names == ["z_cc", "dz", "terrain_z", "mask"], (
        f"[{name}] unexpected plotfile fields: {pf.var_names}")

    zt = pf.field("terrain_z")
    mask = pf.field("mask")
    assert zt.min() == 0.0 and zt.max() == 0.0, (
        f"[{name}] z_terrain must be identically 0, got "
        f"[{zt.min()}, {zt.max()}]")
    assert mask.min() == 0.0 and mask.max() == 0.0, (
        f"[{name}] every cell must be fluid, got mask range "
        f"[{mask.min()}, {mask.max()}]")

    check_mask_is_binary_and_consistent(name, pf)
    print(f"[PASS] {name}")


def check_hill(exe):
    name = "inputs_hill"
    clean("plt_hill", "grid_report_hill.txt")

    result = run_case(exe, name, terrain_arg("terrain_hill.csv"))
    assert result.returncode == 0, (
        f"[{name}] expected success (exit 0), got {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")

    pts = read_points(os.path.join(HERE, "terrain_hill.csv"))
    report = parse_report(os.path.join(WORKDIR, "grid_report_hill.txt"))
    assert report["terrain_n_points"] == len(pts), (
        f"[{name}] solver read {report['terrain_n_points']} points, "
        f"the file holds {len(pts)}")

    pf = Plotfile(os.path.join(WORKDIR, "plt_hill"))
    worst = check_terrain_against_idw(name, pf, pts)
    check_mask_is_binary_and_consistent(name, pf)
    check_mask_boundary(name, pf, report)

    # The hill must actually be resolved: some cells solid, most fluid.
    assert report["terrain_n_solid"] > 0, (
        f"[{name}] a 100 m hill must block some cells")
    assert report["terrain_n_solid"] < report["terrain_n_total"], (
        f"[{name}] the hill must not block the whole domain")

    # And the interpolated surface must track the analytic Gaussian it was
    # sampled from -- this is the end-to-end check that the point file,
    # the reader and the interpolation all line up. The tolerance is the
    # IDW smoothing error over a 20 m point spacing, not round-off.
    zt = pf.field("terrain_z")
    worst_shape = 0.0
    for j in range(0, NY, 3):
        for i in range(0, NX, 3):
            analytic = elevation("hill", cell_center_x(i), cell_center_y(j),
                                 peak=PEAK, sigma=SIGMA, xc=XC, yc=YC)
            worst_shape = max(worst_shape, abs(zt(i, j, 0) - analytic))
    assert worst_shape < 2.0, (
        f"[{name}] interpolated terrain departs from the analytic Gaussian "
        f"by {worst_shape:.3f} m, more than IDW smoothing should cause")

    print(f"[PASS] {name}  (IDW agreement {worst:.2e} m, "
          f"shape error {worst_shape:.3f} m, "
          f"{int(report['terrain_n_solid'])} solid cells)")


def check_scattered(exe):
    name = "inputs_scattered"
    clean("plt_scattered", "grid_report_scattered.txt")

    result = run_case(exe, name, terrain_arg("terrain_scattered.csv"))
    assert result.returncode == 0, (
        f"[{name}] expected success (exit 0), got {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")

    pts = read_points(os.path.join(HERE, "terrain_scattered.csv"))
    report = parse_report(os.path.join(WORKDIR, "grid_report_scattered.txt"))
    pf = Plotfile(os.path.join(WORKDIR, "plt_scattered"))

    worst = check_terrain_against_idw(name, pf, pts)
    check_mask_is_binary_and_consistent(name, pf)
    check_mask_boundary(name, pf, report)

    print(f"[PASS] {name}  (IDW agreement {worst:.2e} m over "
          f"{len(pts)} scattered points)")


def check_missing_file(exe):
    """A terrain file that does not exist must abort, not silently fall
    back to flat ground."""
    name = "inputs_hill (missing terrain file)"
    result = run_case(exe, "inputs_hill",
                      ["terrain.file=definitely_not_here.csv",
                       "grid.output_format=ascii",
                       "grid.report_file=grid_report_missing.txt"])
    assert result.returncode != 0, (
        f"[{name}] expected a fatal abort for a missing terrain file, "
        f"got exit 0.\nstdout:\n{result.stdout}")
    print(f"[PASS] {name}")


def main():
    global WORKDIR

    if len(sys.argv) not in (2, 3):
        print(f"usage: {sys.argv[0]} /path/to/fastwindterrain.exe [workdir]")
        return 1

    exe = os.path.abspath(sys.argv[1])
    if not os.path.isfile(exe):
        print(f"executable not found: {exe}")
        return 1

    if len(sys.argv) == 3:
        WORKDIR = os.path.abspath(sys.argv[2])
    os.makedirs(WORKDIR, exist_ok=True)
    print(f"work directory: {WORKDIR}")

    checks = [check_flat, check_hill, check_scattered, check_missing_file]
    failed = []
    for check in checks:
        try:
            check(exe)
        except AssertionError as e:
            print(f"[FAIL] {check.__name__}: {e}")
            failed.append(check.__name__)
        except Exception as e:
            print(f"[ERROR] {check.__name__}: {e}")
            failed.append(check.__name__)

    if failed:
        print(f"\n{len(failed)} Phase 2 regtest case(s) failed: {failed}")
        return 1

    print("\nAll Phase 2 regtest cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
