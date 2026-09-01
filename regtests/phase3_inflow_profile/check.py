#!/usr/bin/env python3
"""
Phase 3 regtest checker -- terrain-aware inflow profile.

Runs the FastWindTerrain executable against the Phase 3 input files and
validates:

  inputs_powerlaw         -> u0 matches the analytic power law at every
                             sampled height, and points along (u_ref, v_ref)
  inputs_loglaw           -> same for the log law
  inputs_userfile         -> u0 matches an INDEPENDENT Python 3D IDW of
                             the same six-column file, and reproduces
                             table values exactly at table points
  inputs_powerlaw_bump    -> the profile is anchored to AGL: over a hill,
                             the speed at z_terrain + z_agl equals the
                             flat-ground speed at z_agl
  inputs_boundary_terrain -> terrain intersecting the lateral boundaries:
                             the reported boundary flux matches an
                             independent Python integration, and the
                             resulting imbalance is reported, not hidden
  (fatal paths)           -> calm wind and an unknown mode both abort

The expected profiles are recomputed here rather than read back from the
solver, so the checker does not confirm the solver against itself.

These cases check the INITIAL field, so they read u0/v0/w0 from the
plotfile: from Phase 6 on, u/v/w hold the velocity after the projection.

All cases run in a scratch work directory (default
<repo>/build/regtests/phase3_inflow_profile) so no run artifacts land in
the source tree.

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

from plotfile import Plotfile                           # noqa: E402

WORKDIR = os.path.join(ROOT, "build", "regtests", PHASE)

TOL = 1.0e-9          # solver vs. independent reference, in m/s
FLUX_RTOL = 1.0e-6    # reported flux vs. independently integrated flux

# Grid geometry shared by every case (see the inputs files).
NX, NY, NZ = 40, 40, 66
XLO, XHI = 0.0, 1000.0
YLO, YHI = 0.0, 1000.0
DX = (XHI - XLO) / NX
DY = (YHI - YLO) / NY

# Inflow parameters shared by the analytic cases.
U_REF, V_REF, Z_REF = 8.0, 6.0, 10.0
SPEED_REF = math.hypot(U_REF, V_REF)
ALPHA = 0.14
Z0 = 0.1

IDW_K = 6
IDW_EXPONENT = 2.0
DISTANCE_EPSILON = 1.0e-12

SOLID = 1.0


# ---------------------------------------------------------------------------
# Independent reference implementations
# ---------------------------------------------------------------------------

def powerlaw_speed(z_agl, speed_ref=SPEED_REF, z_ref=Z_REF, alpha=ALPHA,
                   z_agl_min=Z0):
    z = max(z_agl, z_agl_min)
    return speed_ref * (z / z_ref) ** alpha


def loglaw_speed(z_agl, speed_ref=SPEED_REF, z_ref=Z_REF, z0=Z0,
                 z_agl_min=Z0):
    z = max(z_agl, z_agl_min)
    return speed_ref * math.log((z + z0) / z0) / math.log((z_ref + z0) / z0)


def read_velocity_points(path):
    """Read a six-column x,y,z,u,v,w file the way the solver does."""
    pts = []
    with open(path) as f:
        for line in f:
            line = line.split("#")[0].replace(",", " ").strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                vals = [float(v) for v in parts[:6]]
            except ValueError:
                continue                       # header line
            if len(vals) == 5:
                vals.append(0.0)               # five-column file: w = 0
            pts.append(tuple(vals))
    assert pts, f"no velocity points read from {path}"
    return pts


def idw3d(xq, yq, zq, pts, k=IDW_K, exponent=IDW_EXPONENT):
    """3D inverse-distance-weighted velocity. Independent re-derivation of
    Inflow::InterpolateIDW3D."""
    d2 = sorted(((x - xq)**2 + (y - yq)**2 + (z - zq)**2, i)
                for i, (x, y, z, _, _, _) in enumerate(pts))
    k = min(k, len(pts))

    wsum = 0.0
    u = v = w = 0.0
    for dist2, i in d2[:k]:
        if dist2 < DISTANCE_EPSILON:
            return pts[i][3], pts[i][4], pts[i][5]      # exact hit
        weight = dist2 ** (-exponent / 2.0)
        wsum += weight
        u += weight * pts[i][3]
        v += weight * pts[i][4]
        w += weight * pts[i][5]
    return u / wsum, v / wsum, w / wsum


def cell_center_x(i):
    return XLO + (i + 0.5) * DX


def cell_center_y(j):
    return YLO + (j + 0.5) * DY


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def parse_report(path):
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
    cmd = [exe, os.path.join(HERE, inputs_file)] + list(extra)
    return subprocess.run(cmd, cwd=WORKDIR, capture_output=True,
                          text=True, timeout=900)


def path_arg(key, name):
    return [f"{key}={os.path.join(HERE, name)}"]


def clean(*names):
    for n in names:
        p = os.path.join(WORKDIR, n)
        if os.path.isdir(p):
            shutil.rmtree(p)
        elif os.path.exists(p):
            os.remove(p)


def require_success(name, result):
    assert result.returncode == 0, (
        f"[{name}] expected success (exit 0), got {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")


# ---------------------------------------------------------------------------
# Shared assertions
# ---------------------------------------------------------------------------

def check_analytic_profile(name, pf, law, stride=9,
                           dir_x=U_REF / SPEED_REF, dir_y=V_REF / SPEED_REF):
    """u0 must equal the analytic law evaluated at z_agl, and must point
    along the reference wind direction. Solid cells must hold zero.

    Note this doubles as an AGL-anchoring check on a terrain case: z_agl
    is recomputed here from the plotfile's own z_cc and terrain_z, so a
    profile anchored to z = 0 instead would fail it."""
    zcc = pf.field("z_cc")
    zt = pf.field("terrain_z")
    mask = pf.field("mask")
    u, v, w = pf.field("u0"), pf.field("v0"), pf.field("w0")

    worst = 0.0
    worst_at = None
    n_checked = 0
    for k in range(0, NZ, 3):
        for j in range(0, NY, stride):
            for i in range(0, NX, stride):
                if mask(i, j, k) == SOLID:
                    assert u(i, j, k) == 0.0 and v(i, j, k) == 0.0 \
                        and w(i, j, k) == 0.0, (
                        f"[{name}] solid cell ({i},{j},{k}) holds velocity "
                        f"({u(i,j,k)}, {v(i,j,k)}, {w(i,j,k)})")
                    continue

                z_agl = zcc(i, j, k) - zt(i, j, k)
                speed = law(z_agl)
                for got, expect, comp in ((u(i, j, k), speed * dir_x, "u"),
                                          (v(i, j, k), speed * dir_y, "v"),
                                          (w(i, j, k), 0.0, "w")):
                    err = abs(got - expect)
                    if err > worst:
                        worst, worst_at = err, (i, j, k, comp, got, expect)
                n_checked += 1

    assert n_checked > 0, f"[{name}] no fluid cells were checked"
    assert worst < TOL, (
        f"[{name}] u0 disagrees with the analytic profile by {worst:.3e} m/s "
        f"at (i,j,k)={worst_at[:3]} component {worst_at[3]}: "
        f"solver {worst_at[4]} vs reference {worst_at[5]}")
    return worst


def check_finite(name, pf):
    """No NaN or Inf anywhere, including the first cell above ground where
    the log law would diverge without the z_agl floor."""
    for comp in ("u0", "v0", "w0"):
        f = pf.field(comp)
        for value in f.values():
            assert math.isfinite(value), (
                f"[{name}] {comp} holds a non-finite value ({value})")


def integrate_boundary_flux(pf, report):
    """Outward flux over open faces, integrated independently from the
    plotfile fields. Mirrors Inflow::ComputeBoundaryFlux but is written
    from the definition, not from that code."""
    mask = pf.field("mask")
    u, v, w = pf.field("u0"), pf.field("v0"), pf.field("w0")
    z_face = [report["z_face"][k] for k in range(NZ + 1)]
    dz = [z_face[k + 1] - z_face[k] for k in range(NZ)]

    flux_in = 0.0
    flux_out = 0.0

    def accumulate(value):
        nonlocal flux_in, flux_out
        if value > 0.0:
            flux_out += value
        else:
            flux_in -= value

    for k in range(NZ):
        for j in range(NY):
            for i in (0, NX - 1):                       # xlo, xhi
                if mask(i, j, k) == SOLID:
                    continue
                sign = -1.0 if i == 0 else 1.0
                accumulate(sign * u(i, j, k) * DY * dz[k])
        for i in range(NX):
            for j in (0, NY - 1):                       # ylo, yhi
                if mask(i, j, k) == SOLID:
                    continue
                sign = -1.0 if j == 0 else 1.0
                accumulate(sign * v(i, j, k) * DX * dz[k])

    k = NZ - 1                                          # domain top
    for j in range(NY):
        for i in range(NX):
            if mask(i, j, k) == SOLID:
                continue
            accumulate(w(i, j, k) * DX * DY)

    return flux_in, flux_out


def check_reported_flux(name, pf, report):
    """The solver's reported flux must match the independent integration."""
    flux_in, flux_out = integrate_boundary_flux(pf, report)
    scale = max(abs(flux_in), abs(flux_out), 1.0)

    for got, expect, label in ((report["inflow_flux_in"], flux_in, "flux_in"),
                               (report["inflow_flux_out"], flux_out, "flux_out")):
        assert abs(got - expect) / scale < FLUX_RTOL, (
            f"[{name}] reported {label} = {got} but independent integration "
            f"gives {expect} (relative difference "
            f"{abs(got - expect) / scale:.3e})")
    return flux_in, flux_out


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def check_powerlaw(exe):
    name = "inputs_powerlaw"
    clean("plt_powerlaw", "grid_report_powerlaw.txt")
    result = run_case(exe, name)
    require_success(name, result)

    # Later phases append their own fields to the same plotfile, so this
    # asserts the velocity fields are present and lead the list after the
    # grid and terrain fields, not that they are the last ones.
    pf = Plotfile(os.path.join(WORKDIR, "plt_powerlaw"))
    assert pf.var_names[:7] == ["z_cc", "dz", "terrain_z", "mask",
                                "u", "v", "w"], (
        f"[{name}] expected the plotfile to lead with the grid, terrain "
        f"and velocity fields, got {pf.var_names}")
    for f0 in ("u0", "v0", "w0"):
        assert f0 in pf.var_names, (
            f"[{name}] the plotfile must also carry the pre-projection "
            f"field {f0}; u/v/w are post-correction from Phase 6 on")

    worst = check_analytic_profile(name, pf, powerlaw_speed)
    check_finite(name, pf)

    # Flat ground with symmetric open faces: nothing blocked, so inflow
    # and outflow must balance exactly.
    report = parse_report(os.path.join(WORKDIR, "grid_report_powerlaw.txt"))
    check_reported_flux(name, pf, report)
    assert report["inflow_flux_imbalance"] < 1.0e-12, (
        f"[{name}] flat ground must balance exactly, got relative imbalance "
        f"{report['inflow_flux_imbalance']}")

    print(f"[PASS] {name}  (profile agreement {worst:.2e} m/s)")


def check_loglaw(exe):
    name = "inputs_loglaw"
    clean("plt_loglaw", "grid_report_loglaw.txt")
    result = run_case(exe, name)
    require_success(name, result)

    pf = Plotfile(os.path.join(WORKDIR, "plt_loglaw"))
    worst = check_analytic_profile(name, pf, loglaw_speed)
    check_finite(name, pf)

    report = parse_report(os.path.join(WORKDIR, "grid_report_loglaw.txt"))
    check_reported_flux(name, pf, report)

    print(f"[PASS] {name}  (profile agreement {worst:.2e} m/s)")


def check_userfile(exe):
    name = "inputs_userfile"
    clean("plt_userfile", "grid_report_userfile.txt")
    result = run_case(exe, name, path_arg("inflow.file", "user_profile.txt"))
    require_success(name, result)

    pts = read_velocity_points(os.path.join(HERE, "user_profile.txt"))
    report = parse_report(os.path.join(WORKDIR, "grid_report_userfile.txt"))
    assert report["inflow_n_points"] == len(pts), (
        f"[{name}] solver read {report['inflow_n_points']} points, the file "
        f"holds {len(pts)}")
    assert report["inflow_n_columns"] == 6, (
        f"[{name}] expected a six-column file, solver reports "
        f"{report['inflow_n_columns']}")

    pf = Plotfile(os.path.join(WORKDIR, "plt_userfile"))
    zcc, zt = pf.field("z_cc"), pf.field("terrain_z")
    u, v, w = pf.field("u0"), pf.field("v0"), pf.field("w0")

    worst = 0.0
    worst_at = None
    for k in range(0, NZ, 5):
        for j in range(0, NY, 11):
            for i in range(0, NX, 11):
                z_agl = max(zcc(i, j, k) - zt(i, j, k), Z0)
                eu, ev, ew = idw3d(cell_center_x(i), cell_center_y(j),
                                   z_agl, pts)
                for got, expect in ((u(i, j, k), eu), (v(i, j, k), ev),
                                    (w(i, j, k), ew)):
                    err = abs(got - expect)
                    if err > worst:
                        worst, worst_at = err, (i, j, k, got, expect)

    assert worst < TOL, (
        f"[{name}] u0 disagrees with the independent 3D IDW by {worst:.3e} "
        f"m/s at (i,j,k)={worst_at[:3]}: solver {worst_at[3]} vs reference "
        f"{worst_at[4]}")

    # The sixth column must actually be used: w is 0.1 everywhere in the
    # file, so a solver that dropped it would show w == 0.
    assert abs(w(NX // 2, NY // 2, NZ // 2) - 0.1) < 1.0e-9, (
        f"[{name}] w = {w(NX//2, NY//2, NZ//2)}, expected the file's 0.1 -- "
        f"the sixth column looks ignored")

    # Landing exactly on a table point must return that point's values.
    xq, yq, zq, eu, ev, ew = pts[0]
    gu, gv, gw = idw3d(xq, yq, zq, pts)
    assert (gu, gv, gw) == (eu, ev, ew), (
        f"[{name}] an exact hit on a table point must return it unchanged")

    check_finite(name, pf)
    print(f"[PASS] {name}  (IDW agreement {worst:.2e} m/s over "
          f"{len(pts)} points)")


def check_agl_anchoring(exe):
    """The heart of Phase 3: over a hill, the profile must ride the
    terrain, so speed at z_terrain + z_agl equals the flat-ground speed
    at z_agl."""
    name = "inputs_powerlaw_bump"
    clean("plt_bump", "grid_report_bump.txt")
    result = run_case(exe, name, path_arg("terrain.file", "terrain_hill.csv"))
    require_success(name, result)

    pf = Plotfile(os.path.join(WORKDIR, "plt_bump"))
    report = parse_report(os.path.join(WORKDIR, "grid_report_bump.txt"))

    # 1. Against the analytic law evaluated at the column's own AGL.
    worst = check_analytic_profile(name, pf, powerlaw_speed)
    check_finite(name, pf)

    # 2. Against the FLAT run, end to end. The stretched levels of the two
    #    runs never coincide once the hill shifts them, so the flat
    #    profile is interpolated to the hill column's AGL heights.
    flat = Plotfile(os.path.join(WORKDIR, "plt_powerlaw"))
    flat_u, flat_zcc = flat.field("u0"), flat.field("z_cc")
    zcc, zt = pf.field("z_cc"), pf.field("terrain_z")
    u, mask = pf.field("u0"), pf.field("mask")

    # The hill peaks near the domain center; pick a column with real relief.
    ic, jc = NX // 2, NY // 2
    h = zt(ic, jc, 0)
    assert h > 50.0, (
        f"[{name}] expected a hill of real height at the center column, "
        f"got z_terrain = {h} m")

    # Flat ground, so the flat run's AGL height IS its z_cc.
    flat_profile = [(flat_zcc(ic, jc, kk), flat_u(ic, jc, kk))
                    for kk in range(NZ)]

    def flat_u_at(z):
        """Flat-run u linearly interpolated to height z AGL."""
        if z <= flat_profile[0][0]:
            return flat_profile[0][1]
        for (z0_, u0_), (z1_, u1_) in zip(flat_profile, flat_profile[1:]):
            if z0_ <= z <= z1_:
                t = (z - z0_) / (z1_ - z0_)
                return u0_ + t * (u1_ - u0_)
        return flat_profile[-1][1]

    dir_x = U_REF / SPEED_REF
    n_compared = 0
    worst_shift = 0.0
    interp_err = 0.0       # error of the interpolation itself
    teeth = 0.0            # how much the profile actually moved
    for k in range(NZ):
        if mask(ic, jc, k) == SOLID:
            continue
        z_agl = zcc(ic, jc, k) - h
        # Only compare where the flat profile actually spans z_agl;
        # below its lowest level there is nothing to interpolate between.
        if not (flat_profile[0][0] <= z_agl <= flat_profile[-1][0]):
            continue

        interpolated = flat_u_at(z_agl)
        worst_shift = max(worst_shift, abs(u(ic, jc, k) - interpolated))
        # Calibrate the tolerance instead of guessing it: the flat profile
        # linearly interpolated to z_agl differs from the analytic law by
        # this much, so the comparison above cannot do better.
        interp_err = max(interp_err,
                         abs(interpolated - powerlaw_speed(z_agl) * dir_x))
        # Same ABSOLUTE height in the flat run: if the profile were
        # anchored to z = 0 rather than to the ground, this would agree
        # instead, so it must not.
        teeth = max(teeth, abs(u(ic, jc, k) - flat_u(ic, jc, k)))
        n_compared += 1

    assert n_compared > 10, (
        f"[{name}] only {n_compared} levels compared -- the anchoring check "
        f"did not really run")
    assert worst_shift <= interp_err + TOL, (
        f"[{name}] the profile is not anchored to AGL: at the hill column "
        f"the speed differs from the flat profile at the same height above "
        f"ground by {worst_shift:.3e} m/s, more than the "
        f"{interp_err:.3e} m/s the interpolation itself accounts for")
    assert teeth > 1.0, (
        f"[{name}] the profile did not shift with the terrain at all "
        f"(max difference from the flat run at the same absolute height is "
        f"only {teeth:.3e} m/s) -- the test has no teeth")

    assert report["terrain_n_solid"] > 0, (
        f"[{name}] the hill must block some cells")

    print(f"[PASS] {name}  (analytic {worst:.2e} m/s, AGL shift "
          f"{worst_shift:.2e} m/s vs {interp_err:.2e} m/s interpolation "
          f"error over {n_compared} levels; profile moved {teeth:.2f} m/s)")


def check_boundary_terrain(exe):
    """Terrain intersecting the lateral boundaries: the reported flux must
    match an independent integration, and the imbalance must be surfaced
    rather than silently corrected."""
    name = "inputs_boundary_terrain"
    clean("plt_boundary", "grid_report_boundary.txt")
    result = run_case(exe, name, path_arg("terrain.file", "terrain_slope.csv"))
    require_success(name, result)

    pf = Plotfile(os.path.join(WORKDIR, "plt_boundary"))
    report = parse_report(os.path.join(WORKDIR, "grid_report_boundary.txt"))
    check_finite(name, pf)

    # The slope must actually block part of a lateral face, or the case is
    # not testing what it claims to.
    mask = pf.field("mask")
    blocked_xhi = sum(1 for k in range(NZ) for j in range(NY)
                      if mask(NX - 1, j, k) == SOLID)
    blocked_xlo = sum(1 for k in range(NZ) for j in range(NY)
                      if mask(0, j, k) == SOLID)
    assert blocked_xhi > 0, (
        f"[{name}] the xhi face should be partly blocked by the slope")
    assert blocked_xhi > blocked_xlo, (
        f"[{name}] the slope rises toward xhi, so xhi ({blocked_xhi} cells) "
        f"must be more blocked than xlo ({blocked_xlo})")

    flux_in, flux_out = check_reported_flux(name, pf, report)

    # With unequal open areas the field cannot balance, and that must show
    # up in the report rather than being quietly scaled away.
    assert report["inflow_flux_imbalance"] > 1.0e-6, (
        f"[{name}] blocked outflow area should leave a measurable imbalance, "
        f"got {report['inflow_flux_imbalance']}")

    # Nothing was rescaled: the interior profile still matches the law.
    # This case blows along +x only, so the direction differs from the
    # other cases' (u_ref, v_ref).
    check_analytic_profile(name, pf,
                           lambda z: loglaw_speed(z, speed_ref=10.0),
                           dir_x=1.0, dir_y=0.0)

    print(f"[PASS] {name}  (flux in/out {flux_in:.4g}/{flux_out:.4g} m^3/s, "
          f"imbalance {report['inflow_flux_imbalance']:.4f}, "
          f"{blocked_xhi} blocked xhi cells)")


def check_fatal_paths(exe):
    """Calm wind and an unknown mode must both abort."""
    calm = run_case(exe, "inputs_powerlaw",
                    ["inflow.u_ref=0.0", "inflow.v_ref=0.0",
                     "grid.output_format=ascii",
                     "grid.report_file=grid_report_calm.txt"])
    assert calm.returncode != 0, (
        "[calm wind] expected a fatal abort when u_ref = v_ref = 0, got "
        f"exit 0.\nstdout:\n{calm.stdout}")

    bad = run_case(exe, "inputs_powerlaw",
                   ["inflow.mode=ekman_spiral",
                    "grid.output_format=ascii",
                    "grid.report_file=grid_report_badmode.txt"])
    assert bad.returncode != 0, (
        "[unknown mode] expected a fatal abort for an unrecognized "
        f"inflow.mode, got exit 0.\nstdout:\n{bad.stdout}")

    print("[PASS] fatal paths (calm wind, unknown mode)")


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

    # check_agl_anchoring compares against the flat power-law run, so it
    # must follow check_powerlaw.
    checks = [check_powerlaw, check_loglaw, check_userfile,
              check_agl_anchoring, check_boundary_terrain, check_fatal_paths]
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
        print(f"\n{len(failed)} Phase 3 regtest case(s) failed: {failed}")
        return 1

    print("\nAll Phase 3 regtest cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
