#!/usr/bin/env python3
"""
terrain.extrapolation regtest -- what a column outside the point cloud
gets.

IDW is an interpolation, but asked for a height outside the data it
still answers: a distance-weighted average of points that all lie to one
side. Because the mask is only ``z_cc <= z_terrain``, a wrong elevation
there does not look wrong -- the column just comes out all fluid or all
solid. ``terrain.extrapolation = nearest`` gives those columns the
nearest input point's elevation instead.

The cases:

  inputs_covered  -- the cloud spans the whole domain, so no column is
                     outside it. The two modes must then produce
                     BIT-IDENTICAL terrain, which is the guarantee that
                     turning the option on cannot disturb a case that
                     did not need it.

  inputs_partial  -- the cloud covers [0, 400] x [0, 600] of a
                     1000 x 1000 m domain. 1216 of the 1600 columns are
                     outside it. Under idw (the default) every column,
                     inside or out, must still match a plain IDW, so the
                     historical behaviour is unchanged. Under nearest the
                     inside columns must be unchanged AND the outside
                     ones must equal the nearest input point's
                     elevation.

  a bad value     -- terrain.extrapolation = corner_average is fatal,
                     not a silent fall back to the default.

Both references -- the IDW and the nearest-point lookup -- are recomputed
here from the point file, so the checker does not confirm the solver
against itself. The count of outside columns is derived from the cloud's
extent and the grid geometry rather than read back from the report.

All cases run in a scratch work directory (default
<repo>/build/regtests/terrain_extrapolation) so no run artifacts land in
the source tree.

Usage:
    python3 check.py /path/to/fastwindterrain.exe [workdir]

Exits 0 if all cases pass, 1 otherwise (suitable for CI).
"""

import sys
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
GROUP = os.path.basename(HERE)
REGTEST_ROOT = os.path.dirname(HERE)
ROOT = os.path.dirname(REGTEST_ROOT)

sys.path.insert(0, REGTEST_ROOT)                        # plotfile.py

from plotfile import Plotfile                           # noqa: E402

WORKDIR = os.path.join(ROOT, "build", "regtests", GROUP)

IDW_TOL = 1.0e-9      # C++ vs Python IDW must agree to round-off

# Grid geometry, shared by both inputs files.
NX, NY = 40, 40
XLO, XHI = 0.0, 1000.0
YLO, YHI = 0.0, 1000.0
DX = (XHI - XLO) / NX
DY = (YHI - YLO) / NY

IDW_K = 6
IDW_EXPONENT = 2.0
DISTANCE_EPSILON = 1.0e-12


# ---------------------------------------------------------------------------
# Independent reference implementations
# ---------------------------------------------------------------------------

def read_points(path):
    """Read an x,y,z point file the way the solver does."""
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
    """Inverse-distance-weighted height over the k nearest points.
    Independent re-derivation of Terrain::InterpolateIDW."""
    d2 = sorted(((x - xq)**2 + (y - yq)**2, i)
                for i, (x, y, _) in enumerate(pts))
    k = min(k, len(pts))

    wsum = 0.0
    zval = 0.0
    for dist2, i in d2[:k]:
        if dist2 < DISTANCE_EPSILON:
            return pts[i][2]
        w = dist2 ** (-exponent / 2.0)
        wsum += w
        zval += w * pts[i][2]
    return zval / wsum


def nearest(xq, yq, pts):
    """Elevation of the single nearest point. Independent re-derivation
    of Terrain::NearestElevation; ties go to the lowest index, which is
    what `min` over an enumerate does."""
    best_i, best_d2 = 0, float("inf")
    for i, (x, y, _) in enumerate(pts):
        d2 = (x - xq)**2 + (y - yq)**2
        if d2 < best_d2:
            best_i, best_d2 = i, d2
    return pts[best_i][2]


def extent(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), max(xs), min(ys), max(ys)


def cell_center_x(i):
    return XLO + (i + 0.5) * DX


def cell_center_y(j):
    return YLO + (j + 0.5) * DY


def is_outside(i, j, ext):
    """The solver's test: strictly outside the cloud's axis-aligned
    extent, so a column sitting exactly on it is interpolated."""
    x_min, x_max, y_min, y_max = ext
    xq, yq = cell_center_x(i), cell_center_y(j)
    return xq < x_min or xq > x_max or yq < y_min or yq > y_max


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def parse_report(path):
    """Parse the plain-text report into a dict of scalars. z_face / z_cc
    are indexed arrays and are collected separately."""
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


def clean(*names):
    for n in names:
        p = os.path.join(WORKDIR, n)
        if os.path.isdir(p):
            shutil.rmtree(p)          # no stale output can fake a pass
        elif os.path.exists(p):
            os.remove(p)


def run(exe, inputs_file, extra=()):
    cmd = [exe, os.path.join(HERE, inputs_file)] + list(extra)
    return subprocess.run(cmd, cwd=WORKDIR, capture_output=True,
                          text=True, timeout=600)


def run_mode(exe, name, inputs_file, terrain_csv, mode, tag):
    """Run one (inputs file, extrapolation mode) pair into its own
    report and plotfile, and return (report, plotfile)."""
    report_name = f"grid_report_{tag}.txt"
    plot_name = f"plt_{tag}"
    clean(report_name, plot_name)

    extra = [f"terrain.file={os.path.join(HERE, terrain_csv)}",
             f"terrain.extrapolation={mode}",
             f"grid.report_file={report_name}",
             f"grid.plot_file={plot_name}"]
    result = run(exe, inputs_file, extra)
    assert result.returncode == 0, (
        f"[{name}] expected success (exit 0), got {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")

    return (parse_report(os.path.join(WORKDIR, report_name)),
            Plotfile(os.path.join(WORKDIR, plot_name)),
            result.stdout)


def column_heights(pf):
    """z_terrain per column, read at k = 0 (it is replicated along k)."""
    zt = pf.field("terrain_z")
    return {(i, j): zt(i, j, 0) for j in range(NY) for i in range(NX)}


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def check_covered_is_untouched(exe):
    """Terrain that covers the domain: no column is outside the cloud,
    so the two modes must agree bit for bit."""
    name = "inputs_covered (both modes agree)"

    pts = read_points(os.path.join(HERE, "terrain_covered.csv"))
    ext = extent(pts)
    expect_outside = sum(1 for j in range(NY) for i in range(NX)
                         if is_outside(i, j, ext))
    assert expect_outside == 0, (
        f"[{name}] the control case is meant to be fully covered, but the "
        f"checker's own geometry says {expect_outside} columns are outside "
        f"x [{ext[0]}, {ext[1]}], y [{ext[2]}, {ext[3]}]")

    rep_idw, pf_idw, out_idw = run_mode(
        exe, name, "inputs_covered", "terrain_covered.csv", "idw", "cov_idw")
    rep_near, pf_near, out_near = run_mode(
        exe, name, "inputs_covered", "terrain_covered.csv", "nearest",
        "cov_near")

    for tag, rep in (("idw", rep_idw), ("nearest", rep_near)):
        assert rep["terrain_n_columns_outside"] == 0, (
            f"[{name}] {tag}: covered terrain must leave no column outside "
            f"the cloud, report says "
            f"{rep['terrain_n_columns_outside']}")

    # Bit for bit, not to a tolerance: the fallback must not have run at
    # all, so there is nothing here for round-off to excuse.
    h_idw = column_heights(pf_idw)
    h_near = column_heights(pf_near)
    differ = [(k, h_idw[k], h_near[k]) for k in h_idw if h_idw[k] != h_near[k]]
    assert not differ, (
        f"[{name}] {len(differ)} column(s) changed when the option was "
        f"turned on over terrain that covers the domain; first: "
        f"(i,j)={differ[0][0]} idw={differ[0][1]} nearest={differ[0][2]}")

    assert rep_idw["terrain_n_solid"] == rep_near["terrain_n_solid"], (
        f"[{name}] the mask changed: {rep_idw['terrain_n_solid']} solid "
        f"cells under idw, {rep_near['terrain_n_solid']} under nearest")
    assert rep_idw["terrain_n_solid"] > 0, (
        f"[{name}] a 100 m hill must block some cells; this case would "
        f"pass trivially if the terrain were doing nothing")

    # Neither mode has anything to say about coverage here.
    for tag, out in (("idw", out_idw), ("nearest", out_near)):
        assert "does not cover the domain" not in out, (
            f"[{name}] {tag}: fully covered terrain must not warn about "
            f"coverage:\n{out}")

    # And the surface is the plain IDW in both modes.
    worst = 0.0
    for j in range(0, NY, 7):
        for i in range(0, NX, 7):
            expect = idw(cell_center_x(i), cell_center_y(j), pts)
            worst = max(worst, abs(h_idw[(i, j)] - expect))
    assert worst < IDW_TOL, (
        f"[{name}] z_terrain disagrees with the independent IDW by "
        f"{worst:.3e} m")

    print(f"[PASS] {name}  (1600 columns, none outside, "
          f"identical in both modes, IDW agreement {worst:.2e} m)")


def check_partial_default_is_unchanged(exe):
    """A cloud smaller than the domain, under the default mode: every
    column, inside or out, is still the plain IDW."""
    name = "inputs_partial (idw, the default)"

    pts = read_points(os.path.join(HERE, "terrain_partial.csv"))
    ext = extent(pts)
    expect_outside = sum(1 for j in range(NY) for i in range(NX)
                         if is_outside(i, j, ext))
    assert expect_outside > 0, (
        f"[{name}] the partial case is meant to leave columns outside the "
        f"cloud; the checker's geometry says none are")

    rep, pf, out = run_mode(exe, name, "inputs_partial",
                            "terrain_partial.csv", "idw", "part_idw")

    assert rep["terrain_n_columns_outside"] == expect_outside, (
        f"[{name}] the run counts {rep['terrain_n_columns_outside']} "
        f"columns outside the cloud, the geometry says {expect_outside}")
    assert rep["terrain_extrapolation"] == "idw"

    # The default must warn -- the whole point is that this case used to
    # be silent.
    assert "does not cover the domain" in out, (
        f"[{name}] a cloud smaller than the domain must say so under the "
        f"default mode:\n{out}")

    h = column_heights(pf)
    worst = 0.0
    worst_at = None
    for j in range(NY):
        for i in range(NX):
            expect = idw(cell_center_x(i), cell_center_y(j), pts)
            err = abs(h[(i, j)] - expect)
            if err > worst:
                worst, worst_at = err, (i, j)
    assert worst < IDW_TOL, (
        f"[{name}] the default mode must be the plain IDW everywhere, but "
        f"column {worst_at} is off by {worst:.3e} m")

    print(f"[PASS] {name}  ({expect_outside} of {NX*NY} columns outside the "
          f"cloud, all still IDW to {worst:.2e} m, warning raised)")


def check_partial_nearest(exe):
    """The same cloud under `nearest`: inside columns unchanged, outside
    columns equal to the nearest input point's elevation."""
    name = "inputs_partial (nearest)"

    pts = read_points(os.path.join(HERE, "terrain_partial.csv"))
    ext = extent(pts)

    rep_idw, pf_idw, _ = run_mode(exe, name, "inputs_partial",
                                  "terrain_partial.csv", "idw", "part_idw2")
    rep, pf, out = run_mode(exe, name, "inputs_partial",
                            "terrain_partial.csv", "nearest", "part_near")

    assert rep["terrain_extrapolation"] == "nearest"
    assert rep["terrain_n_columns_outside"] == \
        rep_idw["terrain_n_columns_outside"], (
        f"[{name}] the count of outside columns is a property of the data, "
        f"not of the mode, but it changed with it")
    assert "does not cover the domain" in out, (
        f"[{name}] nearest mode must still report how much of the domain "
        f"the cloud misses:\n{out}")

    h = column_heights(pf)
    h_idw = column_heights(pf_idw)

    wrong_in, wrong_out, changed_in = [], [], []
    n_out = 0
    moved = 0
    biggest_move = 0.0
    for j in range(NY):
        for i in range(NX):
            xq, yq = cell_center_x(i), cell_center_y(j)
            got = h[(i, j)]
            if is_outside(i, j, ext):
                n_out += 1
                expect = nearest(xq, yq, pts)
                if got != expect:
                    wrong_out.append((i, j, got, expect))
                move = abs(got - h_idw[(i, j)])
                biggest_move = max(biggest_move, move)
                if move > 1.0e-9:
                    moved += 1
            else:
                # Inside the cloud nothing may change: the fallback is
                # for extrapolation, not a different interpolation.
                if got != h_idw[(i, j)]:
                    changed_in.append((i, j, h_idw[(i, j)], got))
                expect = idw(xq, yq, pts)
                if abs(got - expect) > IDW_TOL:
                    wrong_in.append((i, j, got, expect))

    assert not wrong_out, (
        f"[{name}] {len(wrong_out)} outside column(s) did not take the "
        f"nearest point's elevation; first: (i,j)={wrong_out[0][:2]} "
        f"got {wrong_out[0][2]}, nearest point is at {wrong_out[0][3]} m")
    assert not changed_in, (
        f"[{name}] {len(changed_in)} column(s) INSIDE the cloud changed; "
        f"first: (i,j)={changed_in[0][:2]} {changed_in[0][2]} -> "
        f"{changed_in[0][3]}")
    assert not wrong_in, (
        f"[{name}] {len(wrong_in)} inside column(s) disagree with the "
        f"independent IDW; first: (i,j)={wrong_in[0][:2]} got "
        f"{wrong_in[0][2]}, expected {wrong_in[0][3]}")

    # A fallback that returned the IDW value would satisfy everything
    # above by accident on a flat enough terrain. It must actually move
    # the surface, and by more than round-off.
    assert moved > 0.5 * n_out, (
        f"[{name}] only {moved} of {n_out} outside columns changed at all; "
        f"the two modes are not being distinguished")
    assert biggest_move > 1.0, (
        f"[{name}] the largest change any outside column saw is "
        f"{biggest_move:.3e} m; this case cannot tell the fallback from a "
        f"no-op")

    # The mask follows the surface, which is the reason any of this
    # matters.
    assert rep["terrain_n_solid"] != rep_idw["terrain_n_solid"], (
        f"[{name}] the surface moved but the mask did not, which means "
        f"this case is not exercising what it claims to")

    print(f"[PASS] {name}  ({n_out} outside columns take the nearest point, "
          f"{moved} of them moved, largest move {biggest_move:.2f} m, "
          f"{int(rep_idw['terrain_n_solid'])} -> "
          f"{int(rep['terrain_n_solid'])} solid cells)")


def check_unknown_mode_is_fatal(exe):
    """An unrecognized value must abort rather than fall back."""
    name = "inputs_covered (unknown extrapolation)"

    result = run(exe, "inputs_covered",
                 [f"terrain.file={os.path.join(HERE, 'terrain_covered.csv')}",
                  "terrain.extrapolation=corner_average",
                  "grid.output_format=report",
                  "grid.report_file=grid_report_bad_mode.txt"])
    assert result.returncode != 0, (
        f"[{name}] expected a fatal abort for an unrecognized value, got "
        f"exit 0.\nstdout:\n{result.stdout}")
    assert "corner_average" in (result.stdout + result.stderr), (
        f"[{name}] the abort must name the value it rejected.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")

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

    checks = [check_covered_is_untouched,
              check_partial_default_is_unchanged,
              check_partial_nearest,
              check_unknown_mode_is_fatal]
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
        print(f"\n{len(failed)} terrain-extrapolation regtest case(s) "
              f"failed: {failed}")
        return 1

    print("\nAll terrain-extrapolation regtest cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
