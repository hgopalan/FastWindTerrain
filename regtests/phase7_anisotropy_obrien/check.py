#!/usr/bin/env python3
"""
Phase 7 regtest checker -- spatially varying anisotropy and the O'Brien
vertical-velocity adjustment.

Validates:

  inputs_slope -> over a steep hill, alpha_v is suppressed on the flanks
                  and sits at its base value over flat ground; the
                  suppression follows exp(-slope_3d / slope_scale)
                  computed independently here; and after the O'Brien
                  adjustment w is EXACTLY zero at the domain top in every
                  column

  (disabled)   -> with anisotropy.enable = 0 both alphas hold their base
                  values, so the operator is exactly what the earlier
                  phases built

  alpha_h_mode -> 'base' leaves alpha_h alone, as massconsistent_amr
                  does; 'slope' lets the slope factor reach it too

The w = 0 check is the strict one: O'Brien's whole purpose is to make it
exact rather than approximate, so it is held to round-off, not to a
tolerance.

Usage:
    python3 check.py /path/to/fastwindterrain.exe [workdir]

Exits 0 if all cases pass, 1 otherwise (suitable for CI).
"""

import sys
import os
import math
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
GROUP = os.path.basename(HERE)
REGTEST_ROOT = os.path.dirname(HERE)
ROOT = os.path.dirname(REGTEST_ROOT)

sys.path.insert(0, REGTEST_ROOT)

from plotfile import Plotfile                           # noqa: E402

WORKDIR = os.path.join(ROOT, "build", "regtests", GROUP)

NX, NY, NZ = 40, 40, 66
XHI, YHI = 1000.0, 1000.0
DX, DY = XHI / NX, YHI / NY

ALPHA_H_BASE, ALPHA_V_BASE = 1.0, 0.5
SLOPE_SCALE, DECAY_HEIGHT = 0.5, 500.0
MIN_FACTOR, MAX_FACTOR = 0.05, 2.0

SOLID = 1.0
TOL = 1.0e-12


def run_case(exe, inputs_file, extra=()):
    cmd = [exe, os.path.join(HERE, inputs_file)] + list(extra)
    return subprocess.run(cmd, cwd=WORKDIR, capture_output=True,
                          text=True, timeout=3600)


def require_success(name, result):
    assert result.returncode == 0, (
        f"[{name}] expected success (exit 0), got {result.returncode}\n"
        f"stdout:\n{result.stdout[-3000:]}\nstderr:\n{result.stderr[-2000:]}")


def parse_report(path):
    data = {"z_face": {}, "z_cc": {}}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if p[0] in ("z_face", "z_cc"):
                data[p[0]][int(p[1])] = float(p[2])
            elif len(p) >= 2:
                try:
                    data[p[0]] = float(p[1])
                except ValueError:
                    data[p[0]] = p[1]
    return data


def clean(*names):
    for n in names:
        p = os.path.join(WORKDIR, n)
        if os.path.isdir(p):
            shutil.rmtree(p)
        elif os.path.exists(p):
            os.remove(p)


def terrain_arg():
    return [f"terrain.file={os.path.join(HERE, 'terrain_slope.csv')}"]


def column_slopes(zt):
    """|grad z_terrain| per column, from the same central differences the
    solver uses, computed here independently."""
    h = [[zt(i, j, 0) for i in range(NX)] for j in range(NY)]
    slope = [[0.0] * NX for _ in range(NY)]
    for j in range(NY):
        for i in range(NX):
            im1, ip1 = max(0, i - 1), min(NX - 1, i + 1)
            jm1, jp1 = max(0, j - 1), min(NY - 1, j + 1)
            dhdx = (h[j][ip1] - h[j][im1]) / ((ip1 - im1) * DX)
            dhdy = (h[jp1][i] - h[jm1][i]) / ((jp1 - jm1) * DY)
            slope[j][i] = math.hypot(dhdx, dhdy)
    return slope


def check_anisotropy(exe):
    """alpha_v must follow the slope factor, cell by cell."""
    name = "inputs_slope (anisotropy)"
    clean("plt_slope", "grid_report_slope.txt")
    result = run_case(exe, "inputs_slope", terrain_arg())
    require_success(name, result)

    rep = parse_report(os.path.join(WORKDIR, "grid_report_slope.txt"))
    pf = Plotfile(os.path.join(WORKDIR, "plt_slope"))

    assert rep["anisotropy_enable"] == 1
    assert rep["anisotropy_source"] == "slope"

    zt, zcc = pf.field("terrain_z"), pf.field("z_cc")
    ah, av = pf.field("alpha_h"), pf.field("alpha_v")
    slope = column_slopes(zt)

    # The case has to be steep enough to be worth testing.
    assert rep["anisotropy_slope_max"] > 0.3, (
        f"[{name}] the hill is too gentle to exercise the slope factor: "
        f"max |grad z_terrain| = {rep['anisotropy_slope_max']}")

    worst = 0.0
    worst_at = None
    n_suppressed = 0
    for k in range(0, NZ, 5):
        for j in range(0, NY, 5):
            for i in range(0, NX, 5):
                z_agl = max(zcc(i, j, k) - zt(i, j, k), 0.0)
                slope_3d = slope[j][i] * math.exp(-z_agl / DECAY_HEIGHT)
                f_slope = math.exp(-slope_3d / SLOPE_SCALE)

                expect_v = min(max(ALPHA_V_BASE * f_slope,
                                   MIN_FACTOR * ALPHA_V_BASE),
                               MAX_FACTOR * ALPHA_V_BASE)
                err = abs(av(i, j, k) - expect_v)
                if err > worst:
                    worst, worst_at = err, (i, j, k, av(i, j, k), expect_v)
                if av(i, j, k) < 0.99 * ALPHA_V_BASE:
                    n_suppressed += 1

                # alpha_h_mode = base: the horizontal weight is untouched,
                # which is what makes the RATIO fall on slopes.
                assert abs(ah(i, j, k) - ALPHA_H_BASE) < TOL, (
                    f"[{name}] alpha_h_mode is 'base', so alpha_h must "
                    f"stay at {ALPHA_H_BASE}; at ({i},{j},{k}) it is "
                    f"{ah(i,j,k)}")

    assert worst < 1.0e-9, (
        f"[{name}] alpha_v disagrees with an independently computed slope "
        f"factor by {worst:.3e} at (i,j,k)={worst_at[:3]}: solver "
        f"{worst_at[3]} vs reference {worst_at[4]}")
    assert n_suppressed > 0, (
        f"[{name}] alpha_v was never suppressed below base -- the slope "
        f"factor is not doing anything")

    print(f"[PASS] {name}  (alpha_v matches the slope factor to "
          f"{worst:.1e}; suppressed at {n_suppressed} sampled cells, down "
          f"to {rep['anisotropy_alpha_v_min']:.4f} from base "
          f"{ALPHA_V_BASE})")


def check_obrien(exe):
    """w must be exactly zero at the domain top after the adjustment."""
    name = "inputs_slope (O'Brien)"
    rep = parse_report(os.path.join(WORKDIR, "grid_report_slope.txt"))

    assert rep["obrien_enable"] == 1
    assert rep["obrien_n_columns"] > 0, (
        f"[{name}] no columns were adjusted")

    # There must have been a real residual to remove.
    assert rep["obrien_max_residual"] > 1.0e-3, (
        f"[{name}] continuity left nothing at the top "
        f"({rep['obrien_max_residual']}), so the adjustment is untested")

    # And it must be gone. This is the point of O'Brien: exactly zero, not
    # approximately.
    assert rep["obrien_max_w_top"] < 1.0e-12, (
        f"[{name}] w at the domain top is {rep['obrien_max_w_top']}, not "
        f"zero. The quadratic redistribution is meant to make this exact")

    print(f"[PASS] {name}  ({int(rep['obrien_n_columns'])} columns; "
          f"residual {rep['obrien_max_residual']:.3f} m/s removed, "
          f"|w| at top now {rep['obrien_max_w_top']:.1e} m/s)")


def check_disabled_is_inert(exe):
    """With anisotropy off, both alphas must hold their base values, so
    the operator is exactly what the earlier phases built."""
    name = "inputs_slope (anisotropy disabled)"
    clean("plt_slope", "grid_report_slope.txt")
    result = run_case(exe, "inputs_slope",
                      terrain_arg() + ["anisotropy.enable=0"])
    require_success(name, result)

    rep = parse_report(os.path.join(WORKDIR, "grid_report_slope.txt"))
    for key, base in (("anisotropy_alpha_v_min", ALPHA_V_BASE),
                      ("anisotropy_alpha_v_max", ALPHA_V_BASE),
                      ("anisotropy_alpha_h_min", ALPHA_H_BASE),
                      ("anisotropy_alpha_h_max", ALPHA_H_BASE)):
        assert abs(rep[key] - base) < TOL, (
            f"[{name}] {key} is {rep[key]}, expected the base value "
            f"{base}: disabling anisotropy must leave the weights alone")

    print(f"[PASS] {name}  (alpha_h = {ALPHA_H_BASE}, "
          f"alpha_v = {ALPHA_V_BASE} everywhere)")


def check_alpha_h_mode(exe):
    """alpha_h_mode = slope lets the slope factor reach the horizontal
    weight as well; 'base' does not."""
    name = "inputs_slope (alpha_h_mode)"
    clean("plt_slope", "grid_report_slope.txt")
    result = run_case(exe, "inputs_slope",
                      terrain_arg() + ["anisotropy.alpha_h_mode=slope"])
    require_success(name, result)

    rep = parse_report(os.path.join(WORKDIR, "grid_report_slope.txt"))
    assert rep["anisotropy_alpha_h_min"] < 0.99 * ALPHA_H_BASE, (
        f"[{name}] with alpha_h_mode = slope, alpha_h should be "
        f"suppressed somewhere, but its minimum is "
        f"{rep['anisotropy_alpha_h_min']}")

    # The suppression must be the same factor alpha_v gets, since both
    # carry f_slope and differ only in their base.
    ratio_h = rep["anisotropy_alpha_h_min"] / ALPHA_H_BASE
    ratio_v = rep["anisotropy_alpha_v_min"] / ALPHA_V_BASE
    assert abs(ratio_h - ratio_v) < 1.0e-9, (
        f"[{name}] the two weights should carry the SAME slope factor: "
        f"alpha_h fell to {ratio_h:.6f} of base, alpha_v to "
        f"{ratio_v:.6f}")

    print(f"[PASS] {name}  (both weights carry the same factor, "
          f"{ratio_h:.4f} of base)")


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

    failed = []
    # check_obrien reads the report check_anisotropy produced.
    for check in (check_anisotropy, check_obrien, check_disabled_is_inert,
                  check_alpha_h_mode):
        try:
            check(exe)
        except AssertionError as e:
            print(f"[FAIL] {check.__name__}: {e}")
            failed.append(check.__name__)
        except Exception as e:
            print(f"[ERROR] {check.__name__}: {e}")
            failed.append(check.__name__)

    if failed:
        print(f"\n{len(failed)} Phase 7 regtest case(s) failed: {failed}")
        return 1

    print("\nAll Phase 7 regtest cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
