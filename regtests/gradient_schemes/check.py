#!/usr/bin/env python3
"""
Gradient-scheme regtest checker.

Validates the directional-derivative schemes selected by
numerics.gradient_scheme:

  inputs_selftest -> a grid-refinement study of every scheme on both a
                     uniform and a stretched grid, checking the observed
                     order of accuracy, plus the boundedness of the WENO
                     reconstruction across a discontinuity
  inputs_scheme   -> the scheme name round-trips into the report, the
                     default is weno3js, and an unrecognized name aborts

Order of accuracy is the check that actually distinguishes a working
scheme from a plausible-looking one: a sign error or a mis-shifted
stencil still produces smooth-looking output, but it will not converge
at the right rate.

Both norms are checked, because they measure different things for a
nonlinear scheme. WENO3-JS is third order where the field is smooth, but
its Jiang-Shu weights lose an order at critical points, where the
derivative and the smoothness indicators vanish together. L-infinity is
set by those points; L2 shows the order over the field as a whole.

Usage:
    python3 check.py /path/to/fastwindterrain.exe [workdir]

Exits 0 if all cases pass, 1 otherwise (suitable for CI).
"""

import sys
import os
import collections
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
GROUP = os.path.basename(HERE)
REGTEST_ROOT = os.path.dirname(HERE)
ROOT = os.path.dirname(REGTEST_ROOT)

WORKDIR = os.path.join(ROOT, "build", "regtests", GROUP)

# Minimum order accepted on the finest grid pair, per scheme. Set below
# what the schemes actually achieve, but above the next order down, so
# the check has teeth without being brittle.
MIN_ORDER = {
    "central2": 1.85,     # measures ~2.00 uniform, ~1.95 stretched
    "upwind2":  1.80,     # measures ~2.00 uniform, ~1.95 stretched
    "weno3js":  2.50,     # measures ~3.0-4.8; the point is "better than 2"
}

SCHEMES = ("central2", "upwind2", "weno3js")


def run_case(exe, inputs_file, extra=()):
    cmd = [exe, os.path.join(HERE, inputs_file)] + list(extra)
    return subprocess.run(cmd, cwd=WORKDIR, capture_output=True,
                          text=True, timeout=900)


def require_success(name, result):
    assert result.returncode == 0, (
        f"[{name}] expected success (exit 0), got {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")


def parse_report(path):
    data = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                data[parts[0]] = parts[1]
    return data


def parse_selftest(path):
    """Convergence rows keyed by (scheme, grid, norm, advect), plus the
    overshoot rows."""
    table = collections.defaultdict(list)
    overshoot = {}
    with open(path) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue
            p = line.split()
            if p[0] == "overshoot":
                overshoot[p[1]] = float(p[2])
                continue
            scheme, grid, norm, advect = p[0], p[1], p[2], float(p[3])
            table[(scheme, grid, norm, advect)].append(
                (int(p[4]), float(p[5]), float(p[6])))
    assert table, f"no convergence rows read from {path}"
    return table, overshoot


def check_convergence(exe):
    name = "inputs_selftest"
    out = os.path.join(WORKDIR, "gradient_selftest.txt")
    if os.path.exists(out):
        os.remove(out)          # no stale study can fake a pass

    result = run_case(exe, name)
    require_success(name, result)
    assert os.path.isfile(out), (
        f"[{name}] numerics.selftest_file produced no output")

    table, overshoot = parse_selftest(out)

    # Every scheme, on both grids, in both norms, for both upwind branches.
    missing = []
    for scheme in SCHEMES:
        for grid in ("uniform", "stretched"):
            for norm in ("linf", "l2"):
                for advect in (1.0, -1.0):
                    if (scheme, grid, norm, advect) not in table:
                        missing.append((scheme, grid, norm, advect))
    assert not missing, f"[{name}] study is missing cases: {missing[:4]}"

    worst = {}
    for (scheme, grid, norm, advect), rows in sorted(table.items()):
        rows.sort()
        assert len(rows) >= 3, (
            f"[{name}] {scheme}/{grid}/{norm} has only {len(rows)} "
            f"resolutions; an order cannot be established")

        # The error must actually fall with resolution.
        for (n0, e0, _), (n1, e1, _) in zip(rows, rows[1:]):
            assert e1 < e0, (
                f"[{name}] {scheme}/{grid}/{norm} (a={advect}): error rose "
                f"from {e0:.3e} at n={n0} to {e1:.3e} at n={n1}")

        # Order on the finest pair, where the asymptotic rate has settled.
        order = rows[-1][2]
        key = (scheme, grid, norm)
        worst[key] = min(worst.get(key, 9.9), order)
        assert order >= MIN_ORDER[scheme], (
            f"[{name}] {scheme} on the {grid} grid converges at order "
            f"{order:.2f} in {norm} (a={advect}), below the {MIN_ORDER[scheme]} "
            f"expected. Orders across the study: "
            f"{[f'{o:.2f}' for _, _, o in rows[1:]]}")

    # The higher-order scheme must actually be more accurate at the same
    # resolution, or its extra cost buys nothing.
    weno_err = table[("weno3js", "uniform", "l2", 1.0)][-1][1]
    up_err = table[("upwind2", "uniform", "l2", 1.0)][-1][1]
    assert weno_err < up_err, (
        f"[{name}] weno3js ({weno_err:.3e}) is not more accurate than "
        f"upwind2 ({up_err:.3e}) at the finest resolution")

    # Boundedness across a discontinuity: the nonlinear weights exist to
    # suppress the overshoot the linear combination produces.
    assert "weno3js" in overshoot and "linear3" in overshoot, (
        f"[{name}] the study did not report the overshoot test")
    assert overshoot["weno3js"] < 1.0e-10, (
        f"[{name}] the WENO reconstruction overshoots a step by "
        f"{overshoot['weno3js']:.3e}; the limiter is not working")
    assert overshoot["linear3"] > 0.01, (
        f"[{name}] the unlimited linear reconstruction should overshoot a "
        f"step, but measures {overshoot['linear3']:.3e} -- the comparison "
        f"has no teeth")

    summary = ", ".join(
        f"{s} {min(v for (sc, g, nm), v in worst.items() if sc == s):.2f}"
        for s in SCHEMES)
    print(f"[PASS] {name}  (worst finest-grid order: {summary}; "
          f"WENO overshoot {overshoot['weno3js']:.1e} vs linear "
          f"{overshoot['linear3']:.3f})")


def check_scheme_selection(exe):
    name = "inputs_scheme"
    report = os.path.join(WORKDIR, "grid_report_scheme.txt")

    # Default first.
    if os.path.exists(report):
        os.remove(report)
    require_success(name, run_case(exe, name))
    data = parse_report(report)
    assert data["numerics_gradient_scheme"] == "weno3js", (
        f"[{name}] the default scheme must be weno3js, report says "
        f"{data['numerics_gradient_scheme']}")

    # Then each name explicitly.
    for scheme in SCHEMES:
        if os.path.exists(report):
            os.remove(report)
        require_success(name, run_case(
            exe, name, [f"numerics.gradient_scheme={scheme}"]))
        data = parse_report(report)
        assert data["numerics_gradient_scheme"] == scheme, (
            f"[{name}] asked for {scheme}, report says "
            f"{data['numerics_gradient_scheme']}")

    # The stencil radius the schemes need must be reported, since fields
    # they are applied to have to carry that many ghost cells.
    assert int(data["numerics_stencil_radius"]) == 2, (
        f"[{name}] expected a stencil radius of 2, report says "
        f"{data['numerics_stencil_radius']}")

    # An unrecognized name must abort rather than falling back silently.
    bad = run_case(exe, name, ["numerics.gradient_scheme=weno5"])
    assert bad.returncode != 0, (
        f"[{name}] an unrecognized gradient_scheme must be fatal, got "
        f"exit 0.\nstdout:\n{bad.stdout}")

    print(f"[PASS] {name}  (default weno3js; all three names round-trip; "
          f"unknown name aborts)")


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
    for check in (check_convergence, check_scheme_selection):
        try:
            check(exe)
        except AssertionError as e:
            print(f"[FAIL] {check.__name__}: {e}")
            failed.append(check.__name__)
        except Exception as e:
            print(f"[ERROR] {check.__name__}: {e}")
            failed.append(check.__name__)

    if failed:
        print(f"\n{len(failed)} gradient-scheme regtest case(s) failed: "
              f"{failed}")
        return 1

    print("\nAll gradient-scheme regtest cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
