#!/usr/bin/env python3
"""
Regtest checker -- inflow.balance_flux, the boundary mass-flux
redistribution.

The option spreads the initial field's net boundary flux as one uniform
outward-normal velocity over the open cells of xlo, xhi, ylo and yhi, so
that in and out match before the Poisson solve runs. It is OFF by
default, and the first thing checked here is that off means nothing
happened at all.

  off vs absent        -> "inflow.balance_flux = 0" and the key not given
                          produce byte-identical reports and identical
                          u0/v0/w0, to the last bit

  on                   -> the reported imbalance falls from 11% to
                          machine zero, and an INDEPENDENT integration of
                          the plotfile confirms it: this checker computes
                          the open lateral area and the balanced flux
                          from the definition rather than reading the
                          solver's own numbers back

  where it landed      -> u0/v0 on the four lateral faces equal the
                          unbalanced field plus side * shift on every
                          open cell, solid cells still hold zero, the top
                          is untouched (it carries w = 0 by boundary
                          condition), and every cell that is not on a
                          lateral face is bit-for-bit unchanged

  classification       -> the bc_* lines are identical with the option on
                          and off. The redistribution puts flow through
                          the tangential faces, and classifying from that
                          would rename them

  reversed wind        -> the same case blown the other way, where the
                          shift points INTO the domain. Without the
                          classification coming from the raw profile this
                          run aborts: ylo and yhi would read as inflow
                          faces, giving three, and Phase 4 asserts at
                          most two

All cases run in a scratch work directory (default
<repo>/build/regtests/inflow_flux_balance) so no run artifacts land in
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

HERE = os.path.dirname(os.path.abspath(__file__))
GROUP = os.path.basename(HERE)
REGTEST_ROOT = os.path.dirname(HERE)
ROOT = os.path.dirname(REGTEST_ROOT)

sys.path.insert(0, REGTEST_ROOT)                        # plotfile.py

from plotfile import Plotfile                           # noqa: E402

WORKDIR = os.path.join(ROOT, "build", "regtests", GROUP)

# Grid geometry, from inputs_slope.
NX, NY, NZ = 40, 40, 66
XLO, XHI = 0.0, 1000.0
YLO, YHI = 0.0, 1000.0
DX = (XHI - XLO) / NX
DY = (YHI - YLO) / NY

SOLID = 1.0

# The raw profile's imbalance on this case is ~0.11. "Machine zero" is
# generous next to that: the shift is computed from a sum of ~10^4 areas
# and applied to ~10^4 cells, so the cancellation is good to about the
# accumulated round-off of that sum, not to one ulp.
BALANCED_TOL = 1.0e-12
RAW_MIN = 1.0e-2
SHIFT_RTOL = 1.0e-10

# The four lateral faces, as (name, component, side). side = +1 means the
# outward normal points along +component.
LATERAL = (("xlo", 0, -1), ("xhi", 0, +1),
           ("ylo", 1, -1), ("yhi", 1, +1))


# ---------------------------------------------------------------------------
# Running and reading
# ---------------------------------------------------------------------------

def run_case(exe, extra=()):
    cmd = ([exe, os.path.join(HERE, "inputs_slope"),
            f"terrain.file={os.path.join(HERE, 'terrain_slope.csv')}"]
           + list(extra))
    return subprocess.run(cmd, cwd=WORKDIR, capture_output=True,
                          text=True, timeout=900)


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


def parse_report(path):
    """key -> value, with z_face/z_cc kept as index -> value maps."""
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
            elif key in ("bc_xlo", "bc_xhi", "bc_ylo", "bc_yhi",
                         "bc_zlo", "bc_zhi"):
                # "<face> <type> <lambda condition>" -- both words matter
                data[key] = " ".join(parts[1:])
            else:
                try:
                    data[key] = float(parts[1])
                except ValueError:
                    data[key] = parts[1]
    return data


_solved = {}


def solve(exe, tag, extra=()):
    """One run, returning (plotfile, report dict, report path). Cached on
    the tag: several cases read the same pair of runs, and each solve is
    a few seconds."""
    if tag in _solved:
        return _solved[tag]

    report = f"report_{tag}.txt"
    plot = f"plt_{tag}"
    clean(report, plot)
    result = run_case(exe, list(extra) + [f"grid.report_file={report}",
                                          f"grid.plot_file={plot}"])
    require_success(tag, result)
    _solved[tag] = (Plotfile(os.path.join(WORKDIR, plot)),
                    parse_report(os.path.join(WORKDIR, report)),
                    os.path.join(WORKDIR, report))
    return _solved[tag]


# ---------------------------------------------------------------------------
# Independent reference implementations
# ---------------------------------------------------------------------------

def cell_heights(report):
    z_face = [report["z_face"][k] for k in range(NZ + 1)]
    return [z_face[k + 1] - z_face[k] for k in range(NZ)]


def open_lateral_area(pf, report):
    """Total open area of xlo, xhi, ylo and yhi [m^2], written from the
    definition rather than from Inflow::BalanceBoundaryFlux.

    A corner cell lies on two faces and is counted once for each: it
    later takes one shift per face, in each of those two components."""
    mask = pf.field("mask")
    dz = cell_heights(report)

    area = 0.0
    for _, comp, side in LATERAL:
        lateral = DY if comp == 0 else DX
        for k in range(NZ):
            for t in range(NY if comp == 0 else NX):
                i = (0 if side < 0 else NX - 1) if comp == 0 else t
                j = t if comp == 0 else (0 if side < 0 else NY - 1)
                if mask(i, j, k) == SOLID:
                    continue
                area += lateral * dz[k]
    return area


def integrate_boundary_flux(pf, report):
    """Inward and outward volumetric flux over the open boundary faces,
    integrated from the plotfile. The ground is closed and solid cells
    carry no flux; the top is included, as the solver's own definition
    does."""
    mask = pf.field("mask")
    comps = tuple(pf.field(c) for c in ("u0", "v0", "w0"))
    dz = cell_heights(report)

    flux_in = 0.0
    flux_out = 0.0

    def add(value):
        nonlocal flux_in, flux_out
        if value > 0.0:
            flux_out += value
        else:
            flux_in -= value

    for _, comp, side in LATERAL:
        lateral = DY if comp == 0 else DX
        for k in range(NZ):
            for t in range(NY if comp == 0 else NX):
                i = (0 if side < 0 else NX - 1) if comp == 0 else t
                j = t if comp == 0 else (0 if side < 0 else NY - 1)
                if mask(i, j, k) == SOLID:
                    continue
                add(side * comps[comp](i, j, k) * lateral * dz[k])

    # The top, where the outward normal is +z.
    for j in range(NY):
        for i in range(NX):
            if mask(i, j, NZ - 1) == SOLID:
                continue
            add(comps[2](i, j, NZ - 1) * DX * DY)

    return flux_in, flux_out


def on_lateral_face(i, j):
    return i == 0 or i == NX - 1 or j == 0 or j == NY - 1


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def check_off_is_absent(exe):
    """The default has to be a genuine no-op: giving the key as 0 and not
    giving it at all must produce the same run, to the byte."""
    name = "balance_flux off == absent"

    _, absent, absent_path = solve(exe, "absent")
    _, zero, zero_path = solve(exe, "zero", ["inflow.balance_flux=0"])

    with open(absent_path) as f:
        a_text = f.read()
    with open(zero_path) as f:
        z_text = f.read()
    assert a_text == z_text, (
        f"[{name}] the report differs between the key absent and the key "
        f"set to 0; the default is not a no-op")

    assert absent["inflow_balance_flux"] == 0.0, (
        f"[{name}] inflow.balance_flux must default to 0, report says "
        f"{absent['inflow_balance_flux']}")
    assert absent["inflow_flux_balance_shift"] == 0.0, (
        f"[{name}] nothing may be shifted with the option off, report says "
        f"{absent['inflow_flux_balance_shift']}")
    assert absent["inflow_flux_imbalance_raw"] == absent["inflow_flux_imbalance"], (
        f"[{name}] with nothing redistributed the raw and reported "
        f"imbalance must be the same number")

    print(f"[PASS] {name}  (byte-identical report, "
          f"imbalance {absent['inflow_flux_imbalance']:.4g} either way)")


def check_balance_zeroes_the_imbalance(exe):
    """The headline: a case that reports a large imbalance reports
    machine zero with the option on, and an independent integration of
    the field agrees."""
    name = "balance_flux on"

    pf_off, off, _ = solve(exe, "off", ["inflow.balance_flux=0"])
    pf_on, on, _ = solve(exe, "on", ["inflow.balance_flux=1"])

    assert off["inflow_flux_imbalance"] > RAW_MIN, (
        f"[{name}] this case is meant to START with a large imbalance; it "
        f"reports {off['inflow_flux_imbalance']:.4g}, so it is no longer "
        f"testing what it claims to")

    assert on["inflow_flux_imbalance"] <= BALANCED_TOL, (
        f"[{name}] with the option on the reported imbalance must be "
        f"machine zero, got {on['inflow_flux_imbalance']:.4g}")

    # The raw imbalance is carried through unchanged, so one report says
    # both what the profile carried and what was done about it.
    assert on["inflow_flux_imbalance_raw"] == off["inflow_flux_imbalance"], (
        f"[{name}] inflow_flux_imbalance_raw ({on['inflow_flux_imbalance_raw']}) "
        f"must equal the imbalance the same case reports with the option "
        f"off ({off['inflow_flux_imbalance']})")

    # The shift, from the definition: minus the net over the open area of
    # the four lateral faces. Computed here from the plotfile mask, not
    # read back from the solver.
    area = open_lateral_area(pf_off, off)
    expect = -off["inflow_flux_net"] / area
    got = on["inflow_flux_balance_shift"]
    assert abs(got - expect) <= SHIFT_RTOL * abs(expect), (
        f"[{name}] the shift should be -flux_net / open lateral area = "
        f"{expect:.12g} m/s, report says {got:.12g}")

    # And the field itself balances, integrated independently.
    f_in, f_out = integrate_boundary_flux(pf_on, on)
    scale = max(abs(f_in), abs(f_out))
    imbalance = abs(f_out - f_in) / scale
    assert imbalance <= BALANCED_TOL, (
        f"[{name}] independently integrated from the plotfile, the "
        f"balanced field still carries a relative imbalance of "
        f"{imbalance:.4g} (in {f_in:.10g}, out {f_out:.10g})")

    print(f"[PASS] {name}  (imbalance {off['inflow_flux_imbalance']:.4g} -> "
          f"{on['inflow_flux_imbalance']:.2e}, shift {got:+.6g} m/s over "
          f"{area:.6g} m^2 of open lateral face; independent integration "
          f"{imbalance:.2e})")


def check_where_the_shift_landed(exe):
    """Cell by cell: the boundary layer moved by exactly side * shift on
    its open cells, and nothing else moved at all."""
    name = "balance_flux field"

    pf_off, off, _ = solve(exe, "off", ["inflow.balance_flux=0"])
    pf_on, on, _ = solve(exe, "on", ["inflow.balance_flux=1"])
    shift = on["inflow_flux_balance_shift"]
    assert shift != 0.0, f"[{name}] the case must actually shift something"

    mask = pf_off.field("mask")
    u_off, v_off, w_off = (pf_off.field(c) for c in ("u0", "v0", "w0"))
    u_on, v_on, w_on = (pf_on.field(c) for c in ("u0", "v0", "w0"))
    off_c = (u_off, v_off, w_off)
    on_c = (u_on, v_on, w_on)

    # 1. Every cell off a lateral face is bit-for-bit unchanged, and w is
    #    unchanged everywhere: the top carries w = 0 by boundary
    #    condition and takes no share of the redistribution.
    n_interior = 0
    for k in range(NZ):
        for j in range(NY):
            for i in range(NX):
                assert w_on(i, j, k) == w_off(i, j, k), (
                    f"[{name}] w0 moved at ({i},{j},{k}); the vertical "
                    f"component takes no share of the redistribution")
                if on_lateral_face(i, j):
                    continue
                n_interior += 1
                for c in (0, 1):
                    assert on_c[c](i, j, k) == off_c[c](i, j, k), (
                        f"[{name}] component {c} moved at interior cell "
                        f"({i},{j},{k}): {off_c[c](i,j,k)} -> "
                        f"{on_c[c](i,j,k)}. The redistribution must stay "
                        f"on the boundary")

    # 2. On each lateral face, the normal component moved by exactly
    #    side * shift on open cells and not at all in solid ones.
    tol = 1.0e-12 * abs(shift)
    n_open = 0
    n_solid = 0
    for _, comp, side in LATERAL:
        for k in range(NZ):
            for t in range(NY if comp == 0 else NX):
                i = (0 if side < 0 else NX - 1) if comp == 0 else t
                j = t if comp == 0 else (0 if side < 0 else NY - 1)
                delta = on_c[comp](i, j, k) - off_c[comp](i, j, k)
                if mask(i, j, k) == SOLID:
                    n_solid += 1
                    assert on_c[comp](i, j, k) == 0.0, (
                        f"[{name}] solid boundary cell ({i},{j},{k}) holds "
                        f"{on_c[comp](i,j,k)}; solid cells must stay at zero")
                    continue
                n_open += 1
                assert abs(delta - side * shift) <= tol, (
                    f"[{name}] open boundary cell ({i},{j},{k}) on the "
                    f"face with normal {side:+d} along component {comp} "
                    f"moved by {delta:.12g}, expected "
                    f"{side * shift:.12g}")

    assert n_solid > 0, (
        f"[{name}] no lateral boundary cell is solid, so the case is not "
        f"exercising the open-cells-only rule")

    print(f"[PASS] {name}  ({n_open} open boundary cells shifted by "
          f"{shift:+.6g} m/s, {n_solid} solid ones left at zero, "
          f"{n_interior} interior cells bit-for-bit unchanged)")


def check_classification_is_unchanged(exe):
    """The faces must be classified from the wind, not from the
    correction. Here the wind is along +x, so ylo and yhi are tangential
    -- and the redistribution pushes flow straight through them."""
    name = "balance_flux classification"

    _, off, _ = solve(exe, "off", ["inflow.balance_flux=0"])
    _, on, _ = solve(exe, "on", ["inflow.balance_flux=1"])

    for face in ("xlo", "xhi", "ylo", "yhi", "zlo", "zhi"):
        key = f"bc_{face}"
        assert on[key] == off[key], (
            f"[{name}] {face} is '{on[key]}' with the option on and "
            f"'{off[key]}' with it off. Which face the wind enters "
            f"through is a property of the wind")
    for key in ("bc_n_inflow", "bc_n_outflow", "bc_n_tangential",
                "bc_n_lambda_dirichlet"):
        assert on[key] == off[key], (
            f"[{name}] {key} is {on[key]} with the option on and "
            f"{off[key]} with it off")

    assert off["bc_ylo"].split()[0] == "tangential", (
        f"[{name}] this case is meant to have tangential side faces, "
        f"which is what makes the check bite; ylo is "
        f"'{off['bc_ylo']}'")

    print(f"[PASS] {name}  (xlo={on['bc_xlo'].split()[0]}, "
          f"xhi={on['bc_xhi'].split()[0]}, ylo={on['bc_ylo'].split()[0]}, "
          f"yhi={on['bc_yhi'].split()[0]} either way)")


def check_inward_shift(exe):
    """The same slope with the wind reversed. More mass then leaves than
    enters, so the shift points INTO the domain -- and a classification
    made from the corrected field would call the two tangential faces
    inflow faces, giving three, which Phase 4 refuses to solve."""
    name = "balance_flux inward shift"

    reverse = ["inflow.u_ref=-10.0"]
    _, off, _ = solve(exe, "revoff", reverse + ["inflow.balance_flux=0"])
    pf_on, on, _ = solve(exe, "revon", reverse + ["inflow.balance_flux=1"])

    assert on["inflow_flux_balance_shift"] < 0.0, (
        f"[{name}] the reversed case is meant to shift inward, got "
        f"{on['inflow_flux_balance_shift']:+.6g} m/s")
    assert on["inflow_flux_imbalance"] <= BALANCED_TOL, (
        f"[{name}] imbalance {on['inflow_flux_imbalance']:.4g} after "
        f"redistribution")
    assert on["bc_n_inflow"] == off["bc_n_inflow"] == 1.0, (
        f"[{name}] one inflow face either way, got "
        f"{on['bc_n_inflow']} on and {off['bc_n_inflow']} off")
    for face in ("xlo", "xhi", "ylo", "yhi"):
        assert on[f"bc_{face}"] == off[f"bc_{face}"], (
            f"[{name}] {face} is '{on[f'bc_{face}']}' with the option on "
            f"and '{off[f'bc_{face}']}' with it off")

    for comp in ("u0", "v0", "w0"):
        for value in pf_on.field(comp).values():
            assert math.isfinite(value), (
                f"[{name}] {comp} holds a non-finite value ({value})")

    print(f"[PASS] {name}  (shift {on['inflow_flux_balance_shift']:+.6g} "
          f"m/s inward, imbalance {on['inflow_flux_imbalance']:.2e}, "
          f"still {int(on['bc_n_inflow'])} inflow face)")


# ---------------------------------------------------------------------------

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

    checks = [check_off_is_absent,
              check_balance_zeroes_the_imbalance,
              check_where_the_shift_landed,
              check_classification_is_unchanged,
              check_inward_shift]
    failed = []
    for check in checks:
        try:
            check(exe)
        except AssertionError as e:
            print(str(e))
            failed.append(check.__name__)

    if failed:
        print(f"\n{len(failed)} case(s) FAILED: {', '.join(failed)}")
        return 1
    print(f"\nAll {len(checks)} {GROUP} cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
