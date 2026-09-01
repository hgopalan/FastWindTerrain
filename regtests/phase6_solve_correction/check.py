#!/usr/bin/env python3
"""
Phase 6 regtest checker -- linear solve and velocity correction.

Validates:

  inputs_flat -> a horizontally uniform profile over flat ground is
                 already solenoidal, so lambda must come out essentially
                 zero and the corrected velocity must equal u0
  inputs_bump -> over a 100 m hill the projection has real work to do;
                 also run across all three derivative schemes, which must
                 leave the corrected field identical:
                 the divergence in the norm the solve CONTROLS must go
                 down, must keep going down as passes are added, and the
                 corrected wind must stay physical

Three things this checker is deliberate about:

**It checks the divergence the solve controls.** AMReX's nodal operator
is built from its own compact divergence; the configured scheme
(central2/upwind2/weno3js) is a different, wider operator that the
projection does not act on. Asserting on the scheme-based number would
be asserting on something nothing drives to zero.

**It checks velocity extrema.** A projection can reduce divergence
handsomely while wrecking the field. An earlier version of this solver
reduced divergence 15x while turning a 10 m/s inflow into a 35 m/s
corrected wind, and no divergence number showed it -- only the extrema
did.

**It checks that more passes help.** AMReX's nodal projection is
approximate: its divergence and gradient are not an exact factorisation
of the operator, so one pass removes only part of the divergence. That
the remainder shrinks monotonically with passes is the property worth
pinning, not any single threshold.

Usage:
    python3 check.py /path/to/fastwindterrain.exe [workdir]

Exits 0 if all cases pass, 1 otherwise (suitable for CI).
"""

import sys
import os
import math
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
GROUP = os.path.basename(HERE)
REGTEST_ROOT = os.path.dirname(HERE)
ROOT = os.path.dirname(REGTEST_ROOT)

WORKDIR = os.path.join(ROOT, "build", "regtests", GROUP)

# inputs_flat / inputs_bump share these.
U_REF, V_REF, Z_REF, ALPHA = 8.0, 6.0, 10.0, 0.14
SPEED_REF = math.hypot(U_REF, V_REF)


def run_case(exe, inputs_file, extra=()):
    cmd = [exe, os.path.join(HERE, inputs_file)] + list(extra)
    return subprocess.run(cmd, cwd=WORKDIR, capture_output=True,
                          text=True, timeout=3600)


def require_success(name, result):
    assert result.returncode == 0, (
        f"[{name}] expected success (exit 0), got {result.returncode}\n"
        f"stdout:\n{result.stdout[-3000:]}\nstderr:\n{result.stderr[-2000:]}")


def parse_report(path):
    data = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) >= 2:
                try:
                    data[p[0]] = float(p[1])
                except ValueError:
                    data[p[0]] = p[1]
    return data


def clean(*names):
    for n in names:
        p = os.path.join(WORKDIR, n)
        if os.path.exists(p):
            os.remove(p)


def powerlaw_max_speed(z_top):
    """The fastest the undisturbed profile ever gets, at the domain top."""
    return SPEED_REF * (z_top / Z_REF) ** ALPHA


def check_flat(exe):
    """Nothing to correct: lambda ~ 0 and u ~ u0."""
    name = "inputs_flat"
    clean("grid_report_flat.txt")
    result = run_case(exe, name)
    require_success(name, result)

    rep = parse_report(os.path.join(WORKDIR, "grid_report_flat.txt"))

    # The solve must converge, whatever it is asked to solve.
    assert rep["poisson_solve_residual"] < 1.0e-8, (
        f"[{name}] MLMG did not converge: residual "
        f"{rep['poisson_solve_residual']}")

    # A solenoidal field leaves nothing for the projection to do.
    div0 = rep["poisson_div_controlled_before"]
    assert div0 < 1.0e-9, (
        f"[{name}] a uniform profile over flat ground should already be "
        f"divergence free, but the controlled norm reports {div0}")

    assert abs(rep["poisson_lambda_absmax"]) < 1.0e-6, (
        f"[{name}] with nothing to correct, lambda should be ~0, got "
        f"{rep['poisson_lambda_absmax']}")

    # ...and the velocity must come out untouched.
    speed_before = rep["poisson_speed_max_before"]
    speed_after = rep["poisson_speed_max_after"]
    assert abs(speed_after - speed_before) < 1.0e-6, (
        f"[{name}] the projection changed a field it had no reason to "
        f"touch: |U|max {speed_before} -> {speed_after}")

    # w must stay identically zero: the profile is horizontal.
    assert abs(rep["poisson_w_min_after"]) < 1.0e-9 \
        and abs(rep["poisson_w_max_after"]) < 1.0e-9, (
        f"[{name}] w should stay zero over flat ground, got "
        f"[{rep['poisson_w_min_after']}, {rep['poisson_w_max_after']}]")

    print(f"[PASS] {name}  (lambda ~ 0, |U|max unchanged at "
          f"{speed_after:.3f} m/s, w identically 0)")


def check_bump(exe):
    """Over a hill the projection must reduce the divergence it controls,
    without wrecking the field."""
    name = "inputs_bump"
    terrain = [f"terrain.file={os.path.join(HERE, 'terrain_hill.csv')}"]

    clean("grid_report_bump.txt")
    result = run_case(exe, name, terrain)
    require_success(name, result)
    rep = parse_report(os.path.join(WORKDIR, "grid_report_bump.txt"))

    assert rep["poisson_solve_residual"] < 1.0e-8, (
        f"[{name}] MLMG did not converge: residual "
        f"{rep['poisson_solve_residual']}")

    # There must be something to correct, or the case proves nothing.
    div0 = rep["poisson_div_controlled_before"]
    div1 = rep["poisson_div_controlled_after"]
    assert div0 > 1.0e-3, (
        f"[{name}] the hill should leave a real divergence to remove, "
        f"got {div0}")
    assert div1 < div0, (
        f"[{name}] the projection did not reduce the divergence it "
        f"controls: {div0} -> {div1}")

    # The corrected wind must stay physical. The undisturbed profile peaks
    # at the domain top; terrain speed-up of a factor of two would be
    # remarkable, so anything past that means the projection is wrong,
    # however good the divergence looks.
    z_top = 961.2758234855
    profile_max = powerlaw_max_speed(z_top)
    speed_after = rep["poisson_speed_max_after"]
    assert speed_after < 2.0 * profile_max, (
        f"[{name}] corrected |U|max = {speed_after:.2f} m/s against a "
        f"profile peak of {profile_max:.2f} m/s. A projection can reduce "
        f"divergence while wrecking the field; this is the check that "
        f"catches it")

    # No component may run away in either direction.
    for comp in ("u", "v", "w"):
        lo = rep[f"poisson_{comp}_min_after"]
        hi = rep[f"poisson_{comp}_max_after"]
        assert all(math.isfinite(x) for x in (lo, hi)), (
            f"[{name}] {comp} has a non-finite extreme: [{lo}, {hi}]")
        assert max(abs(lo), abs(hi)) < 2.0 * profile_max, (
            f"[{name}] {comp} reaches [{lo:.2f}, {hi:.2f}] m/s against a "
            f"profile peak of {profile_max:.2f}")

    print(f"[PASS] {name}  (controlled div {div0:.4f} -> {div1:.4f}, "
          f"|U|max {speed_after:.2f} m/s vs profile peak "
          f"{profile_max:.2f} m/s)")
    return div0


def check_more_passes_help(exe):
    """AMReX's nodal projection is approximate, so one pass removes only
    part of the divergence. What must hold is that adding passes keeps
    removing more -- that is the property, not any single threshold."""
    name = "inputs_bump (projection passes)"
    terrain = [f"terrain.file={os.path.join(HERE, 'terrain_hill.csv')}"]

    results = []
    for n in (1, 2, 4):
        clean("grid_report_bump.txt")
        result = run_case(exe, "inputs_bump",
                          terrain + [f"poisson.n_projections={n}"])
        require_success(f"{name} n={n}", result)
        rep = parse_report(os.path.join(WORKDIR, "grid_report_bump.txt"))
        results.append((n, rep["poisson_div_controlled_after"],
                        rep["poisson_speed_max_after"]))

    for (n0, d0, _), (n1, d1, _) in zip(results, results[1:]):
        assert d1 < d0, (
            f"[{name}] going from {n0} to {n1} passes did not reduce the "
            f"divergence further: {d0:.5f} -> {d1:.5f}. The projection is "
            f"not contracting")

    # More passes must not destabilise the field either.
    speeds = [s for _, _, s in results]
    assert max(speeds) - min(speeds) < 1.0e-3 * max(speeds), (
        f"[{name}] |U|max drifts with the number of passes: {speeds}")

    trail = " -> ".join(f"{d:.5f}" for _, d, _ in results)
    print(f"[PASS] {name}  (1/2/4 passes: {trail}; |U|max steady at "
          f"{speeds[-1]:.2f} m/s)")


def check_scheme_does_not_affect_projection(exe):
    """The derivative scheme must not touch the projection at all.

    The solve takes its divergence and its gradient from AMReX's own
    compact operators, so weno3js / upwind2 / central2 govern only the
    reported diagnostic and, later, advection. Running all three must
    give identical corrected fields; only the scheme-norm diagnostic may
    differ."""
    name = "inputs_bump (scheme invariance)"
    terrain = [f"terrain.file={os.path.join(HERE, 'terrain_hill.csv')}"]

    out = {}
    for scheme in ("weno3js", "upwind2", "central2"):
        clean("grid_report_bump.txt")
        result = run_case(exe, "inputs_bump",
                          terrain + [f"numerics.gradient_scheme={scheme}"])
        require_success(f"{name} {scheme}", result)
        rep = parse_report(os.path.join(WORKDIR, "grid_report_bump.txt"))
        assert rep["numerics_gradient_scheme"] == scheme
        out[scheme] = rep

    ref = out["weno3js"]
    for scheme, rep in out.items():
        for key in ("poisson_div_controlled_after", "poisson_speed_max_after",
                    "poisson_u_max_after", "poisson_v_max_after",
                    "poisson_w_max_after", "poisson_lambda_absmax"):
            assert rep[key] == ref[key], (
                f"[{name}] {scheme} changed {key}: {rep[key]} vs "
                f"{ref[key]} for weno3js. The scheme must not reach the "
                f"projection")

    # The scheme-based diagnostic, by contrast, is expected to differ --
    # it is a different operator, which is the whole point.
    diag = {s: r["poisson_div_after"] for s, r in out.items()}
    assert len(set(diag.values())) > 1, (
        f"[{name}] every scheme reported the same diagnostic divergence "
        f"({diag}); the diagnostic is not actually using the scheme")

    print(f"[PASS] {name}  (corrected field identical across all three "
          f"schemes; diagnostic differs as expected: "
          + ", ".join(f"{s} {d:.4f}" for s, d in diag.items()) + ")")


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
    for check in (check_flat, check_bump, check_more_passes_help,
                  check_scheme_does_not_affect_projection):
        try:
            check(exe)
        except AssertionError as e:
            print(f"[FAIL] {check.__name__}: {e}")
            failed.append(check.__name__)
        except Exception as e:
            print(f"[ERROR] {check.__name__}: {e}")
            failed.append(check.__name__)

    if failed:
        print(f"\n{len(failed)} Phase 6 regtest case(s) failed: {failed}")
        return 1

    print("\nAll Phase 6 regtest cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
