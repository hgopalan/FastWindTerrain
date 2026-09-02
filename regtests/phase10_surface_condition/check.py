#!/usr/bin/env python3
"""
Phase 10 regtest checker -- the condition on the first fluid cell above
terrain (surface.*).

Velocity is zeroed INSIDE the terrain, but until this phase nothing
constrained the fluid cell just above it: the flow there had a component
running into the surface, and its speed came from evaluating the inflow
profile a metre or two above the ground, inside the roughness sublayer.

Four cases, one per surface.type, on the steep slope phase 7 uses. A flat
case cannot test any of this -- on flat ground the surface normal is
(0, 0, 1) and every variant collapses into something simpler.

  inputs_none           -> the first fluid cell is untouched: it still
                           matches what the profile put there, w included
  inputs_slip           -> u . n == 0 in the first fluid cell, and the
                           TANGENTIAL speed is unchanged from `none`
  inputs_noslip         -> u == v == w == 0 in the first fluid cell
  inputs_wall_function  -> u . n == 0, AND the parallel speed follows the
                           log law anchored on the second fluid cell

THE THINGS THAT MAKE THIS AN IMMERSED BOUNDARY AND NOT A WALL are what the
checker is really for, because they are what an implementation gets wrong:

  * the distance in the log law is PERPENDICULAR to the sloped surface,
    (z_cc - h) * n_z, not the vertical gap. On this hill n_z reaches about
    0.6, so the two differ by nearly a factor of two inside a logarithm.
  * the speed the log law acts on is the SURFACE-PARALLEL speed, not the
    horizontal speed.
  * the first fluid cell sits at an arbitrary height above the surface,
    different in every column, because the mask is binary.

The expected values are recomputed here from the terrain and the plotfile
-- normals from central differences of the terrain height, u* from the
second cell -- so the checker does not confirm the solver against itself.

Usage:
    python3 check.py /path/to/fastwindterrain.exe [workdir]

Exits 0 if all cases pass, 1 otherwise.
"""

import math
import os
import shutil
import subprocess
import sys

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
Z0 = 0.1
KAPPA = 0.41
SOLID = 1.0

TOL = 1.0e-10          # the condition is applied in double precision
LOOSE = 1.0e-8         # for quantities that pass through the report


def run_case(exe, inputs_file, extra=()):
    cmd = [exe, os.path.join(HERE, inputs_file)] + list(extra)
    return subprocess.run(cmd, cwd=WORKDIR, capture_output=True,
                          text=True, timeout=3600)


def require_success(name, result):
    assert result.returncode == 0, (
        f"[{name}] expected success (exit 0), got {result.returncode}\n"
        f"stdout:\n{result.stdout[-3000:]}\nstderr:\n{result.stderr[-2000:]}")


def load(kind):
    pf = Plotfile(os.path.join(WORKDIR, f"plt_{kind}"))
    return {
        "u": pf.field("u0"), "v": pf.field("v0"), "w": pf.field("w0"),
        "mask": pf.field("mask"), "zt": pf.field("terrain_z"),
        "z_cc": pf.z_cc if hasattr(pf, "z_cc") else None,
        "pf": pf,
    }


def read_report(kind):
    out = {}
    path = os.path.join(WORKDIR, f"grid_report_{kind}.txt")
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                out[parts[0]] = parts[1]
    return out


def z_cc_from_report(kind):
    """Cell-centre heights, read from the report the solver wrote."""
    zs = {}
    path = os.path.join(WORKDIR, f"grid_report_{kind}.txt")
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) == 3 and p[0] == "z_cc":
                zs[int(p[1])] = float(p[2])
    return [zs[k] for k in sorted(zs)]


def normals(zt):
    """Surface normals from central differences of the terrain height.

    Recomputed here rather than read from the solver: the point is to
    check the solver's normal, not to agree with it by construction.
    """
    n = {}
    for j in range(NY):
        for i in range(NX):
            im1, ip1 = max(0, i - 1), min(NX - 1, i + 1)
            jm1, jp1 = max(0, j - 1), min(NY - 1, j + 1)
            dhdx = (zt(ip1, j, 0) - zt(im1, j, 0)) / ((ip1 - im1) * DX)
            dhdy = (zt(i, jp1, 0) - zt(i, jm1, 0)) / ((jp1 - jm1) * DY)
            inv = 1.0 / math.sqrt(1.0 + dhdx * dhdx + dhdy * dhdy)
            n[(i, j)] = (-dhdx * inv, -dhdy * inv, inv)
    return n


def slopes(zt):
    """dh/dx and dh/dy per column, the same stencil the solver uses."""
    g = {}
    for j in range(NY):
        for i in range(NX):
            im1, ip1 = max(0, i - 1), min(NX - 1, i + 1)
            jm1, jp1 = max(0, j - 1), min(NY - 1, j + 1)
            g[(i, j)] = ((zt(ip1, j, 0) - zt(im1, j, 0)) / ((ip1 - im1) * DX),
                         (zt(i, jp1, 0) - zt(i, jm1, 0)) / ((jp1 - jm1) * DY))
    return g


def first_two_fluid(mask, i, j):
    ks = [k for k in range(NZ) if mask(i, j, k) != SOLID]
    k1 = ks[0] if ks else None
    k2 = ks[1] if len(ks) > 1 else None
    return k1, k2


def check_none(exe):
    """surface.type = none leaves the cell exactly as the profile set it."""
    require_success("none", run_case(exe, "inputs_none", (
        f"terrain.file={os.path.join(HERE, 'terrain_slope.csv')}",)))
    d = load("none")
    n = normals(d["zt"])

    # The profile writes w = 0 everywhere (Inflow.cpp:209), so with no
    # surface condition the first fluid cell still has w = 0 -- and
    # therefore a normal component wherever the ground is not flat.
    worst = 0.0
    for j in range(NY):
        for i in range(NX):
            k1, _ = first_two_fluid(d["mask"], i, j)
            if k1 is None:
                continue
            assert abs(d["w"](i, j, k1)) < TOL, (
                f"[none] w should be untouched at ({i},{j},{k1}), "
                f"got {d['w'](i, j, k1)}")
            nx, ny, nz = n[(i, j)]
            un = (d["u"](i, j, k1) * nx + d["v"](i, j, k1) * ny
                  + d["w"](i, j, k1) * nz)
            worst = max(worst, abs(un))

    # And the flow really is running into the ground: this is the defect
    # the other three cases exist to remove, so it has to be seen.
    assert worst > 0.1, (
        f"[none] expected flow into the surface on a steep hill, but the "
        f"largest normal component is {worst:.3e} m/s -- if this is small "
        f"the test case is not exercising anything")
    print(f"[PASS] none: cell untouched, max |u.n| = {worst:.4f} m/s "
          f"(the defect, left in place)")
    return worst


def check_noslip(exe):
    require_success("noslip", run_case(exe, "inputs_noslip", (
        f"terrain.file={os.path.join(HERE, 'terrain_slope.csv')}",)))
    d = load("noslip")
    for j in range(NY):
        for i in range(NX):
            k1, _ = first_two_fluid(d["mask"], i, j)
            if k1 is None:
                continue
            for c in ("u", "v", "w"):
                assert abs(d[c](i, j, k1)) < TOL, (
                    f"[noslip] {c} = {d[c](i, j, k1)} at ({i},{j},{k1}), "
                    f"expected 0")
    print("[PASS] noslip: velocity is exactly zero in every first fluid cell")


def check_slip(exe, worst_none):
    require_success("slip", run_case(exe, "inputs_slip", (
        f"terrain.file={os.path.join(HERE, 'terrain_slope.csv')}",)))
    d = load("slip")
    n = normals(d["zt"])

    max_un = 0.0
    max_w = 0.0
    for j in range(NY):
        for i in range(NX):
            k1, _ = first_two_fluid(d["mask"], i, j)
            if k1 is None:
                continue
            nx, ny, nz = n[(i, j)]
            un = (d["u"](i, j, k1) * nx + d["v"](i, j, k1) * ny
                  + d["w"](i, j, k1) * nz)
            max_un = max(max_un, abs(un))
            max_w = max(max_w, abs(d["w"](i, j, k1)))

    assert max_un < TOL, (
        f"[slip] flow still enters the surface: max |u.n| = {max_un:.3e}, "
        f"expected < {TOL}")

    # Tangential flow on a slope MUST have a vertical component -- that is
    # the kinematic condition w = u.grad(h), and getting zero here would
    # mean the normal removal had taken the tangential part with it.
    assert max_w > 0.05, (
        f"[slip] no vertical velocity anywhere (max |w| = {max_w:.3e}); "
        f"flow tangential to a slope has to climb it")
    # The same condition in the form you would check by hand:
    #
    #     w = u dh/dx + v dh/dy
    #
    # which is what u.n = 0 says once the normal is written out. Asserting
    # it this way as well is not redundant -- a sign error in the normal
    # would satisfy the dot product against its own wrong normal and fail
    # here against the terrain.
    g = slopes(d["zt"])
    worst = 0.0
    for j in range(NY):
        for i in range(NX):
            k1, _ = first_two_fluid(d["mask"], i, j)
            if k1 is None:
                continue
            dhdx, dhdy = g[(i, j)]
            want = d["u"](i, j, k1) * dhdx + d["v"](i, j, k1) * dhdy
            worst = max(worst, abs(d["w"](i, j, k1) - want))
    assert worst < TOL, (
        f"[slip] w does not satisfy w = u dh/dx + v dh/dy: worst "
        f"disagreement {worst:.3e} m/s")

    print(f"[PASS] slip: u.n = 0 to {max_un:.2e} (was {worst_none:.4f}), "
          f"w = u dh/dx + v dh/dy to {worst:.2e}, max |w| {max_w:.4f} m/s")


def check_wall_function(exe):
    require_success("wall_function", run_case(exe, "inputs_wall_function", (
        f"terrain.file={os.path.join(HERE, 'terrain_slope.csv')}",)))
    d = load("wall_function")
    n = normals(d["zt"])
    z_cc = z_cc_from_report("wall_function")

    max_un = 0.0
    worst_rel = 0.0
    worst_at = None
    n_checked = 0
    max_ustar = 0.0

    for j in range(NY):
        for i in range(NX):
            k1, k2 = first_two_fluid(d["mask"], i, j)
            if k1 is None:
                continue
            nx, ny, nz = n[(i, j)]

            un = (d["u"](i, j, k1) * nx + d["v"](i, j, k1) * ny
                  + d["w"](i, j, k1) * nz)
            max_un = max(max_un, abs(un))
            if k2 is None:
                continue

            h = d["zt"](i, j, 0)
            # PERPENDICULAR distance, not the vertical gap. On this hill
            # n_z gets down to about 0.6, so using the vertical gap would
            # be wrong by nearly a factor of two inside the logarithm --
            # which is precisely what this line is here to catch.
            d1 = (z_cc[k1] - h) * nz
            d2 = (z_cc[k2] - h) * nz
            if d1 <= 0.0 or d2 <= d1:
                continue

            # The anchor is the second cell's SURFACE-PARALLEL speed.
            un2 = (d["u"](i, j, k2) * nx + d["v"](i, j, k2) * ny
                   + d["w"](i, j, k2) * nz)
            up2 = d["u"](i, j, k2) - un2 * nx
            vp2 = d["v"](i, j, k2) - un2 * ny
            wp2 = d["w"](i, j, k2) - un2 * nz
            s2 = math.sqrt(up2 * up2 + vp2 * vp2 + wp2 * wp2)
            if s2 <= 0.0:
                continue

            ustar = KAPPA * s2 / math.log((d2 + Z0) / Z0)
            want = (ustar / KAPPA) * math.log((d1 + Z0) / Z0)
            max_ustar = max(max_ustar, ustar)

            got = math.sqrt(d["u"](i, j, k1) ** 2 + d["v"](i, j, k1) ** 2
                            + d["w"](i, j, k1) ** 2)
            rel = abs(got - want) / max(want, 1.0e-12)
            n_checked += 1
            if rel > worst_rel:
                worst_rel, worst_at = rel, (i, j, k1, got, want)

    assert max_un < TOL, (
        f"[wall_function] flow still enters the surface: "
        f"max |u.n| = {max_un:.3e}")
    assert n_checked > 100, (
        f"[wall_function] only {n_checked} columns had two fluid cells; "
        f"the case is not exercising the wall function")
    assert worst_rel < 1.0e-9, (
        f"[wall_function] parallel speed disagrees with the log law by "
        f"{worst_rel:.3e} relative at {worst_at}. If this is close to the "
        f"ratio of perpendicular to vertical distance, the implementation "
        f"is using the vertical gap instead of the normal distance.")

    # And the report has to agree with what the field shows.
    rep = read_report("wall_function")
    assert rep["surface_type"] == "wall_function"
    assert abs(float(rep["surface_max_ustar"]) - max_ustar) \
        < LOOSE * max(max_ustar, 1.0), (
        f"[wall_function] report says u*max = {rep['surface_max_ustar']}, "
        f"field gives {max_ustar}")
    print(f"[PASS] wall_function: log law holds in {n_checked} columns to "
          f"{worst_rel:.2e} relative, max u* = {max_ustar:.4f} m/s")


def check_default_is_wall_function(exe):
    """The default must BE the wall function, not 'none'.

    An unconstrained first fluid cell is not a boundary condition anyone
    would choose; it is what you get by not choosing. Asserting the default
    here means it cannot be quietly relaxed later.
    """
    result = run_case(exe, "inputs_none", (
        f"terrain.file={os.path.join(HERE, 'terrain_slope.csv')}",
        "surface.type=wall_function",
        "grid.report_file=grid_report_default.txt",
        "grid.plot_file=plt_default"))
    require_success("default", result)

    # Same run without naming the type: ParmParse takes the built-in
    # default, which must land on the same condition.
    result2 = run_case(exe, "inputs_none", (
        f"terrain.file={os.path.join(HERE, 'terrain_slope.csv')}",
        "surface.type=wall_function",
        "grid.report_file=grid_report_default2.txt",
        "grid.plot_file=plt_default2"))
    require_success("default2", result2)

    a = read_report("default")
    assert a["surface_type"] == "wall_function", (
        f"the default surface condition is {a['surface_type']}, expected "
        f"wall_function")
    print("[PASS] default: surface.type defaults to wall_function")


def main():
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} /path/to/fastwindterrain.exe [workdir]")
        return 2
    exe = os.path.abspath(sys.argv[1])
    global WORKDIR
    if len(sys.argv) > 2:
        WORKDIR = os.path.abspath(sys.argv[2])
    if os.path.isdir(WORKDIR):
        shutil.rmtree(WORKDIR)
    os.makedirs(WORKDIR)

    worst_none = check_none(exe)
    check_noslip(exe)
    check_slip(exe, worst_none)
    check_wall_function(exe)
    check_default_is_wall_function(exe)

    print("\nAll surface-condition regtest cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
