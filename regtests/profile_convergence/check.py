#!/usr/bin/env python3
"""
Profile-gradient convergence regtest checker.

Runs a reduced version of convergence/run_convergence.py and asserts the
observed order of accuracy of each derivative scheme, measured END TO END
through the solver rather than on synthetic data.

WHY THIS IS NOT THE gradient_schemes GROUP

numerics.selftest_file already measures the order of all three schemes,
on a std::vector holding sin(2 pi x). It never builds a Grid, a MultiFab
or a ghost cell. Everything between the stencil and the number the solver
uses is untested there: the column metric, the box decomposition, the
index clamping at the domain ends. This group runs the real path.

WHY THE VERTICAL DERIVATIVE

Over flat ground a powerlaw or loglaw profile has div(u) = 0 identically
-- u and v depend only on z and w is zero -- so a divergence-based study
would measure nothing at all. dU/dz is the only nontrivial derivative in
that problem, and it is where the schemes differ.

The sweep is reduced relative to the full study: three resolutions rather
than four, so CI stays quick. convergence/run_convergence.py is the same
code with the full grid set.

Usage:
    python3 check.py /path/to/fastwindterrain.exe [workdir]

Exits 0 if all cases pass, 1 otherwise (suitable for CI).
"""

import sys
import os
import csv
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
GROUP = os.path.basename(HERE)
REGTEST_ROOT = os.path.dirname(HERE)
ROOT = os.path.dirname(REGTEST_ROOT)
DRIVER = os.path.join(ROOT, "convergence", "run_convergence.py")

WORKDIR = os.path.join(ROOT, "build", "regtests", GROUP)

GRIDS = "64,128,256"

# What each scheme must reach in L2 on the finest pair. The driver holds
# the same thresholds; they are restated here so a change to either is
# visible in a diff of the other.
MIN_ORDER_L2 = {"central2": 1.80, "upwind2": 1.80, "weno3js": 2.50}


def run_driver(extra=()):
    cmd = [sys.executable, DRIVER, EXE, "--workdir", WORKDIR,
           "--grids", GRIDS] + list(extra)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=7200)


def read_results():
    rows = []
    with open(os.path.join(WORKDIR, "convergence_results.csv")) as f:
        for r in csv.DictReader(f):
            for k in ("nz", "n_levels"):
                r[k] = int(r[k])
            for k in ("l2", "linf", "order_l2", "order_linf", "spread",
                      "advect", "dz0", "ratio"):
                r[k] = float(r[k])
            rows.append(r)
    return rows


def check_orders(exe):
    """Every scheme, both profiles, on a stretched grid."""
    name = "stretched grid (order of accuracy)"

    result = run_driver(["--check"])
    assert result.returncode == 0, (
        f"[{name}] the convergence sweep failed:\n"
        f"{result.stdout[-4000:]}\n{result.stderr[-1000:]}")

    rows = read_results()
    assert len(rows) == 2 * 3 * 3, (
        f"[{name}] expected 18 runs (2 profiles x 3 schemes x 3 grids), "
        f"got {len(rows)}")

    finest = {}
    for r in rows:
        key = (r["profile"], r["scheme"])
        if key not in finest or r["nz"] > finest[key]["nz"]:
            finest[key] = r

    for (profile, scheme), r in sorted(finest.items()):
        want = MIN_ORDER_L2[scheme]
        assert r["order_l2"] >= want, (
            f"[{name}] {profile}/{scheme}: L2 order {r['order_l2']:.2f} on "
            f"the finest pair, expected at least {want}")

    orders = ", ".join(
        f"{s} {min(r['order_l2'] for (p, sc), r in finest.items() if sc == s):.2f}"
        for s in ("central2", "upwind2", "weno3js"))
    print(f"[PASS] {name}  (18 runs; worst L2 order per scheme: {orders})")


def check_weno_is_actually_better(exe):
    """Order is one claim; being more accurate on the same grid is
    another, and it is the one a user cares about."""
    name = "stretched grid (weno3js beats the second-order schemes)"

    rows = read_results()
    finest_nz = max(r["nz"] for r in rows)

    for profile in ("powerlaw", "loglaw"):
        err = {r["scheme"]: r["l2"] for r in rows
               if r["profile"] == profile and r["nz"] == finest_nz}
        for other in ("central2", "upwind2"):
            assert err["weno3js"] < err[other], (
                f"[{name}] {profile} at nz={finest_nz}: weno3js L2 "
                f"{err['weno3js']:.3e} is not better than {other} "
                f"{err[other]:.3e}")

    ratio = min(
        max(r["l2"] for r in rows
            if r["profile"] == p and r["nz"] == finest_nz
            and r["scheme"] != "weno3js")
        / next(r["l2"] for r in rows
               if r["profile"] == p and r["nz"] == finest_nz
               and r["scheme"] == "weno3js")
        for p in ("powerlaw", "loglaw"))
    print(f"[PASS] {name}  (at nz={finest_nz}, at least {ratio:.0f}x more "
          f"accurate than the second-order schemes)")


def check_horizontal_uniformity(exe):
    """Over flat ground the gradient must not vary horizontally. The runs
    are decomposed into several boxes, so this is what would catch a
    decomposition or ghost bug -- and it is exactly the kind of bug the
    self-test cannot see, because it never builds a box."""
    name = "flat ground (horizontal uniformity)"

    rows = read_results()
    worst = max(r["spread"] for r in rows)
    assert worst == 0.0, (
        f"[{name}] the vertical gradient varies horizontally over flat "
        f"ground; worst spread across {len(rows)} runs is {worst:.3e}")

    print(f"[PASS] {name}  (spread exactly 0 across all {len(rows)} runs)")


def check_both_upwind_branches(exe):
    """An upwind scheme has two branches and a sign error hides in the
    one that is never measured."""
    name = "upwind branches"

    result = run_driver(["--advect", "minus", "--check",
                         "--schemes", "upwind2,weno3js",
                         "--profiles", "powerlaw"])
    assert result.returncode == 0, (
        f"[{name}] the negative-advect sweep failed:\n"
        f"{result.stdout[-4000:]}\n{result.stderr[-1000:]}")

    rows = read_results()
    assert all(r["advect"] == -1.0 for r in rows), (
        f"[{name}] the sweep did not actually run the negative branch")

    finest_nz = max(r["nz"] for r in rows)
    got = {r["scheme"]: r["order_l2"] for r in rows if r["nz"] == finest_nz}
    for scheme, order in sorted(got.items()):
        assert order >= MIN_ORDER_L2[scheme], (
            f"[{name}] {scheme} on the downwind branch: L2 order "
            f"{order:.2f}, expected at least {MIN_ORDER_L2[scheme]}")

    shown = ", ".join(f"{s} {o:.2f}" for s, o in sorted(got.items()))
    print(f"[PASS] {name}  (advect -1 converges too: {shown})")


def main():
    global WORKDIR, EXE

    if len(sys.argv) not in (2, 3):
        print(f"usage: {sys.argv[0]} /path/to/fastwindterrain.exe [workdir]")
        return 1

    EXE = os.path.abspath(sys.argv[1])
    if not os.path.isfile(EXE):
        print(f"executable not found: {EXE}")
        return 1

    if len(sys.argv) == 3:
        WORKDIR = os.path.abspath(sys.argv[2])
    os.makedirs(WORKDIR, exist_ok=True)
    print(f"work directory: {WORKDIR}")

    failed = []
    # The first three read the CSV the first one produces; the last
    # overwrites it, so it runs last.
    for check in (check_orders, check_weno_is_actually_better,
                  check_horizontal_uniformity, check_both_upwind_branches):
        try:
            check(EXE)
        except AssertionError as e:
            print(f"[FAIL] {check.__name__}: {e}")
            failed.append(check.__name__)
        except Exception as e:
            print(f"[ERROR] {check.__name__}: {e}")
            failed.append(check.__name__)

    if failed:
        print(f"\n{len(failed)} profile-convergence regtest case(s) failed: "
              f"{failed}")
        return 1

    print("\nAll profile-convergence regtest cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
