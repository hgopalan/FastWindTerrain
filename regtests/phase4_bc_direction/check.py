#!/usr/bin/env python3
"""
Phase 4 regtest checker -- directional boundary conditions.

Runs the FastWindTerrain executable against the Phase 4 input files and
validates:

  inputs_sw      -> SW wind: xlo/ylo inflow, xhi/yhi outflow
  inputs_ne      -> NE wind: every lateral face swaps, relative to sw
  inputs_edge    -> axis-aligned wind: one inflow face, one outflow face,
                    and two TANGENTIAL faces treated as open
  inputs_terrain -> an inflow face partly blocked by terrain: the blocked
                    ghost cells are shut off, the rest carry the profile
  inputs_userfile -> userfile mode has no reference wind vector, so the
                    faces must be classified from the flux the field
                    itself carries

Every boundary cell is checked, not a sample: the solver writes one row
per boundary cell to bc.dump_file, and the expected ghost value is
recomputed here from the profile law rather than read back from the
solver.

The boundary conditions checked, per face type:

  inflow      ghost = the profile at the ghost cell center, evaluated in
              AGL against the adjacent interior column -- or exactly zero
              where terrain blocks that column
  outflow     ghost = interior, exactly (zero gradient)
  tangential  ghost = interior, exactly (open, not sealed)
  noflow      ghost u,v = interior u,v (free slip) and ghost w =
              -interior w, so w averages to zero ON the face

All cases run in a scratch work directory (default
<repo>/build/regtests/phase4_bc_direction).

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
PHASE = os.path.basename(HERE)
REGTEST_ROOT = os.path.dirname(HERE)
ROOT = os.path.dirname(REGTEST_ROOT)

sys.path.insert(0, REGTEST_ROOT)                        # plotfile.py

from plotfile import Plotfile                           # noqa: E402

WORKDIR = os.path.join(ROOT, "build", "regtests", PHASE)

TOL = 1.0e-9

NX, NY, NZ = 40, 40, 66
XLO, XHI = 0.0, 1000.0
YLO, YHI = 0.0, 1000.0
DX = (XHI - XLO) / NX
DY = (YHI - YLO) / NY

SOLID = 1.0

# face name -> (direction, side)
FACES = {"xlo": (0, -1), "xhi": (0, +1),
         "ylo": (1, -1), "yhi": (1, +1),
         "zlo": (2, -1), "zhi": (2, +1)}


# ---------------------------------------------------------------------------
# Independent reference implementations
# ---------------------------------------------------------------------------

def powerlaw_speed(z_agl, speed_ref, z_ref, alpha, z_agl_min):
    return speed_ref * (max(z_agl, z_agl_min) / z_ref) ** alpha


def loglaw_speed(z_agl, speed_ref, z_ref, z0, z_agl_min):
    z = max(z_agl, z_agl_min)
    return speed_ref * math.log((z + z0) / z0) / math.log((z_ref + z0) / z0)


class Profile:
    """The 1D law a case was configured with, as the checker's own
    independent implementation."""

    def __init__(self, mode, u_ref, v_ref, z_ref, alpha=0.14, z0=0.1):
        self.mode = mode
        self.u_ref, self.v_ref = u_ref, v_ref
        self.speed_ref = math.hypot(u_ref, v_ref)
        self.dir_x = u_ref / self.speed_ref
        self.dir_y = v_ref / self.speed_ref
        self.z_ref, self.alpha, self.z0 = z_ref, alpha, z0
        self.z_agl_min = z0            # the solver's default

    def uvw(self, z_agl):
        if self.mode == "powerlaw":
            s = powerlaw_speed(z_agl, self.speed_ref, self.z_ref,
                               self.alpha, self.z_agl_min)
        else:
            s = loglaw_speed(z_agl, self.speed_ref, self.z_ref,
                             self.z0, self.z_agl_min)
        return s * self.dir_x, s * self.dir_y, 0.0


def expected_classification(profile):
    """What each face must be classified as, derived from the wind
    direction alone."""
    out = {}
    for name, (d, side) in FACES.items():
        if d == 2:
            out[name] = "noflow"
            continue
        un = side * (profile.dir_x if d == 0 else profile.dir_y)
        if un < -1.0e-8:
            out[name] = "inflow"
        elif un > 1.0e-8:
            out[name] = "outflow"
        else:
            out[name] = "tangential"
    return out


def expected_lambda(face_type):
    return "neumann" if face_type in ("inflow", "noflow") else "dirichlet"


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def parse_report(path):
    """The report holds `key value...` rows; the bc_* rows carry two
    values (type and lambda condition)."""
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
            elif key.startswith("bc_") and len(parts) == 3:
                data[key] = (parts[1], parts[2])
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


def parse_dump(path):
    """One row per boundary cell: face i j k ghost_uvw int_uvw."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue
            p = line.split()
            rows.append({
                "face": p[0],
                "ijk": (int(p[1]), int(p[2]), int(p[3])),
                "ghost": (float(p[4]), float(p[5]), float(p[6])),
                "interior": (float(p[7]), float(p[8]), float(p[9])),
            })
    assert rows, f"no rows read from {path}"
    return rows


def interior_index(face, i, j, k):
    d, side = FACES[face]
    return ((i - side) if d == 0 else i,
            (j - side) if d == 1 else j,
            (k - side) if d == 2 else k)


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

def check_classification(name, report, profile):
    """Every face must be classified from the wind direction, and its
    lambda condition must follow from that classification."""
    expect = expected_classification(profile)

    for face, want in expect.items():
        got_type, got_lambda = report[f"bc_{face}"]
        assert got_type == want, (
            f"[{name}] face {face}: classified {got_type}, expected {want} "
            f"for wind direction ({profile.dir_x:.3f}, {profile.dir_y:.3f})")
        assert got_lambda == expected_lambda(want), (
            f"[{name}] face {face} is {got_type} so lambda must be "
            f"{expected_lambda(want)}, got {got_lambda}")

    n_inflow = sum(1 for v in expect.values() if v == "inflow")
    assert report["bc_n_inflow"] == n_inflow
    assert 1 <= report["bc_n_inflow"] <= 2, (
        f"[{name}] velocity must be prescribed on one or two lateral "
        f"faces, report says {report['bc_n_inflow']}")

    # The invariant that keeps the Poisson operator non-singular.
    assert report["bc_n_lambda_dirichlet"] >= 1, (
        f"[{name}] every face is Neumann in lambda -- the operator would "
        f"be singular")
    return expect


def check_ghost_values(name, rows, pf, profile, report):
    """Check every boundary cell against the condition its face carries."""
    zt = pf.field("terrain_z")
    mask = pf.field("mask")
    z_cc = [report["z_cc"][k] for k in range(NZ)]

    counts = {}
    worst = {"inflow": 0.0, "outflow": 0.0, "tangential": 0.0, "noflow": 0.0}
    n_blocked = 0

    for row in rows:
        face = row["face"]
        i, j, k = row["ijk"]
        ii, jj, kk = interior_index(face, i, j, k)
        gu, gv, gw = row["ghost"]
        iu, iv, iw = row["interior"]

        ftype = report[f"bc_{face}"][0]
        counts[ftype] = counts.get(ftype, 0) + 1

        if ftype == "inflow":
            if mask(ii, jj, kk) == SOLID:
                # Terrain blocks the face here: the ghost must be shut,
                # not carrying the profile into the ground.
                n_blocked += 1
                err = max(abs(gu), abs(gv), abs(gw))
                assert err < TOL, (
                    f"[{name}] {face} ghost at ({i},{j},{k}) is behind "
                    f"terrain but holds ({gu}, {gv}, {gw})")
                continue

            xq = XLO + (i + 0.5) * DX
            yq = YLO + (j + 0.5) * DY
            z_agl = z_cc[kk] - zt(ii, jj, kk)
            eu, ev, ew = profile.uvw(z_agl)
            err = max(abs(gu - eu), abs(gv - ev), abs(gw - ew))
            worst["inflow"] = max(worst["inflow"], err)
            assert err < TOL, (
                f"[{name}] {face} inflow ghost at ({i},{j},{k}): got "
                f"({gu}, {gv}, {gw}), expected ({eu}, {ev}, {ew}) from the "
                f"profile at z_agl = {z_agl}")

        elif ftype in ("outflow", "tangential"):
            err = max(abs(gu - iu), abs(gv - iv), abs(gw - iw))
            worst[ftype] = max(worst[ftype], err)
            assert err == 0.0, (
                f"[{name}] {face} is {ftype} so the ghost must copy the "
                f"interior exactly; at ({i},{j},{k}) ghost ({gu}, {gv}, "
                f"{gw}) vs interior ({iu}, {iv}, {iw})")

        elif ftype == "noflow":
            # w reflected so the face value is zero; u,v free slip.
            err = max(abs(gu - iu), abs(gv - iv), abs(gw + iw))
            worst["noflow"] = max(worst["noflow"], err)
            assert err == 0.0, (
                f"[{name}] {face} is noflow so ghost w must be -interior w "
                f"and u,v must copy; at ({i},{j},{k}) ghost ({gu}, {gv}, "
                f"{gw}) vs interior ({iu}, {iv}, {iw})")
            # The whole point: w averages to zero ON the face.
            assert abs(0.5 * (gw + iw)) < TOL, (
                f"[{name}] {face}: w at the face is {0.5 * (gw + iw)}, "
                f"not zero")

    # Every face type present must have been exercised.
    for ftype, n in counts.items():
        assert n > 0, f"[{name}] no boundary cells of type {ftype}"

    return counts, worst, n_blocked


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def run_and_check(exe, name, tag, profile, extra=()):
    clean(f"plt_{tag}", f"grid_report_{tag}.txt", f"bc_dump_{tag}.txt")
    result = run_case(exe, name, extra)
    require_success(name, result)

    report = parse_report(os.path.join(WORKDIR, f"grid_report_{tag}.txt"))
    rows = parse_dump(os.path.join(WORKDIR, f"bc_dump_{tag}.txt"))
    pf = Plotfile(os.path.join(WORKDIR, f"plt_{tag}"))

    expect = check_classification(name, report, profile)
    counts, worst, n_blocked = check_ghost_values(name, rows, pf, profile,
                                                  report)
    return report, rows, expect, counts, worst, n_blocked


def check_sw(exe):
    name = "inputs_sw"
    profile = Profile("powerlaw", 8.0, 6.0, 10.0)
    report, _, expect, counts, worst, _ = run_and_check(
        exe, name, "sw", profile)

    assert expect["xlo"] == "inflow" and expect["ylo"] == "inflow"
    assert expect["xhi"] == "outflow" and expect["yhi"] == "outflow"

    print(f"[PASS] {name}  (xlo/ylo inflow, xhi/yhi outflow; "
          f"{sum(counts.values())} boundary cells, worst inflow error "
          f"{worst['inflow']:.2e} m/s)")
    return report


def check_ne(exe, sw_report):
    """The opposite wind must flip every lateral face."""
    name = "inputs_ne"
    profile = Profile("powerlaw", -8.0, -6.0, 10.0)
    report, _, expect, counts, worst, _ = run_and_check(
        exe, name, "ne", profile)

    assert expect["xhi"] == "inflow" and expect["yhi"] == "inflow"
    assert expect["xlo"] == "outflow" and expect["ylo"] == "outflow"

    # Every lateral face must have swapped relative to the SW case.
    for face in ("xlo", "xhi", "ylo", "yhi"):
        sw_type = sw_report[f"bc_{face}"][0]
        ne_type = report[f"bc_{face}"][0]
        assert {sw_type, ne_type} == {"inflow", "outflow"}, (
            f"[{name}] face {face} did not flip between the SW and NE "
            f"cases: {sw_type} -> {ne_type}")

    print(f"[PASS] {name}  (all four lateral faces flipped vs SW; "
          f"worst inflow error {worst['inflow']:.2e} m/s)")


def check_edge(exe):
    """Axis-aligned wind: one inflow face and two tangential ones."""
    name = "inputs_edge"
    profile = Profile("loglaw", 10.0, 0.0, 10.0)
    report, _, expect, counts, worst, _ = run_and_check(
        exe, name, "edge", profile)

    assert expect["xlo"] == "inflow"
    assert expect["xhi"] == "outflow"
    assert expect["ylo"] == "tangential" and expect["yhi"] == "tangential", (
        f"[{name}] with v_ref = 0 the y faces carry no normal flow and "
        f"must be tangential, got {expect['ylo']} / {expect['yhi']}")

    assert report["bc_n_inflow"] == 1, (
        f"[{name}] an axis-aligned wind has exactly one inflow face, "
        f"report says {report['bc_n_inflow']}")
    assert report["bc_n_tangential"] == 2

    # Tangential faces are open, so they add lambda-Dirichlet faces:
    # xhi plus ylo and yhi.
    assert report["bc_n_lambda_dirichlet"] == 3, (
        f"[{name}] expected xhi + ylo + yhi to be lambda Dirichlet, got "
        f"{report['bc_n_lambda_dirichlet']} Dirichlet faces")

    assert counts.get("tangential", 0) > 0, (
        f"[{name}] no tangential boundary cells were checked")

    print(f"[PASS] {name}  (1 inflow, 1 outflow, 2 tangential open; "
          f"{report['bc_n_lambda_dirichlet']:.0f} lambda-Dirichlet faces)")


def check_terrain_blocked(exe):
    """An inflow face partly buried in terrain must shut off the buried
    cells and keep the profile on the rest."""
    name = "inputs_terrain"
    profile = Profile("loglaw", -10.0, 0.0, 10.0)
    report, rows, expect, counts, worst, n_blocked = run_and_check(
        exe, name, "terrain", profile,
        path_arg("terrain.file", "terrain_slope.csv"))

    assert expect["xhi"] == "inflow", (
        f"[{name}] the wind blows along -x, so xhi must be the inflow face")

    assert n_blocked > 0, (
        f"[{name}] the slope should bury part of the xhi inflow face, but "
        f"no ghost cell was found behind terrain")

    # And it must not bury all of it, or the case proves nothing about the
    # cells that do carry the profile.
    n_inflow_cells = counts.get("inflow", 0)
    assert n_blocked < n_inflow_cells, (
        f"[{name}] every inflow ghost is behind terrain ({n_blocked} of "
        f"{n_inflow_cells}); the face is entirely sealed")

    print(f"[PASS] {name}  ({n_blocked} of {n_inflow_cells} inflow ghosts "
          f"shut off by terrain, rest carry the profile; worst error "
          f"{worst['inflow']:.2e} m/s)")


def check_userfile_direction(exe):
    """userfile mode has no reference wind vector, so the faces have to be
    classified from the flux the field itself carries. This is the case
    that a classification keyed on (u_ref, v_ref) cannot handle at all."""
    name = "inputs_userfile"
    tag = "userfile"
    clean(f"plt_{tag}", f"grid_report_{tag}.txt", f"bc_dump_{tag}.txt")

    result = run_case(exe, name, path_arg("inflow.file", "user_profile.txt"))
    require_success(name, result)

    report = parse_report(os.path.join(WORKDIR, f"grid_report_{tag}.txt"))

    # The file has u > 0 and v > 0 everywhere.
    assert report["bc_xlo"][0] == "inflow" and report["bc_ylo"][0] == "inflow", (
        f"[{name}] with u > 0 and v > 0 the flow enters through xlo and "
        f"ylo, got {report['bc_xlo'][0]} / {report['bc_ylo'][0]}")
    assert report["bc_xhi"][0] == "outflow" and report["bc_yhi"][0] == "outflow"
    assert report["bc_n_inflow"] == 2
    assert report["bc_n_lambda_dirichlet"] >= 1

    # The classification must come from real flux, not from a zero vector.
    assert report["bc_flux_xlo"] < 0.0 and report["bc_flux_ylo"] < 0.0, (
        f"[{name}] inflow faces must carry negative outward flux, got "
        f"xlo {report['bc_flux_xlo']}, ylo {report['bc_flux_ylo']}")
    assert report["bc_flux_xhi"] > 0.0 and report["bc_flux_yhi"] > 0.0

    # Zero-gradient still has to hold on the outflow faces.
    rows = parse_dump(os.path.join(WORKDIR, f"bc_dump_{tag}.txt"))
    pf = Plotfile(os.path.join(WORKDIR, f"plt_{tag}"))
    zt, mask = pf.field("terrain_z"), pf.field("mask")
    n_checked = 0
    for row in rows:
        if report[f"bc_{row['face']}"][0] != "outflow":
            continue
        assert row["ghost"] == row["interior"], (
            f"[{name}] {row['face']} outflow ghost at {row['ijk']} does not "
            f"copy the interior: {row['ghost']} vs {row['interior']}")
        n_checked += 1
    assert n_checked > 0, f"[{name}] no outflow cells were checked"

    print(f"[PASS] {name}  (classified from field flux with no reference "
          f"vector: xlo/ylo inflow, xhi/yhi outflow; {n_checked} outflow "
          f"cells verified)")


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
    sw_report = None

    try:
        sw_report = check_sw(exe)
    except AssertionError as e:
        print(f"[FAIL] check_sw: {e}")
        failed.append("check_sw")
    except Exception as e:
        print(f"[ERROR] check_sw: {e}")
        failed.append("check_sw")

    rest = [check_edge, check_terrain_blocked, check_userfile_direction]
    if sw_report is not None:
        rest.insert(0, lambda exe_: check_ne(exe_, sw_report))
        rest[0].__name__ = "check_ne"
    else:
        print("[SKIP] check_ne (needs the SW case to compare against)")

    for check in rest:
        try:
            check(exe)
        except AssertionError as e:
            print(f"[FAIL] {check.__name__}: {e}")
            failed.append(check.__name__)
        except Exception as e:
            print(f"[ERROR] {check.__name__}: {e}")
            failed.append(check.__name__)

    if failed:
        print(f"\n{len(failed)} Phase 4 regtest case(s) failed: {failed}")
        return 1

    print("\nAll Phase 4 regtest cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
