#!/usr/bin/env python3
"""
Phase 13 regtest checker -- the solver driven from Python.

Validates:

  flat ground   -> Phase 6's inputs_flat assertions, reproduced from
                   Python: a uniform profile over flat ground is already
                   solenoidal, so lambda comes out identically zero, the
                   velocity is untouched and w stays zero. This is the
                   case that catches a projection which "corrects" a
                   field that was already fine

  stepwise      -> project_once() run one pass at a time gives exactly
                   the same field as solve() run with n_projections set
                   to the same number. The loop and the single step are
                   the same code, and this holds them to it

  convergence   -> over a hill the divergence in the norm the solve
                   controls falls monotonically, pass after pass, and
                   MLMG reports its residual and iteration count

  ghost refill  -> THE test Phase 11 could not write. A solver handed a
                   velocity through set_velocity must agree, to the last
                   bit, with the solver that produced it -- measured with
                   the scheme divergence, which reads the ghost cells
                   through a five-point stencil. Stale ghosts would show
                   up here and nowhere else

  no leak       -> a dict-configured run ignores ParmParse, so poisson
                   and anisotropy settings left by an inputs file do not
                   reach it

Usage:
    python3 check.py /path/to/fastwindterrain.exe [workdir]

Exits 0 if all cases pass, 1 otherwise (suitable for CI).
"""

import sys
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
GROUP = os.path.basename(HERE)
REGTEST_ROOT = os.path.dirname(HERE)
ROOT = os.path.dirname(REGTEST_ROOT)

WORKDIR = os.path.join(ROOT, "build", "regtests", GROUP)

TERRAIN_CSV = os.path.join(REGTEST_ROOT, "phase8_diagnostics_output",
                           "terrain_hill.csv")


def bindings_available():
    shim = os.path.join(ROOT, "build", "fastwindterrain-py")
    pkg = os.path.join(ROOT, "build", "python", "fastwindterrain",
                       "__init__.py")
    return os.path.isfile(shim) and os.path.isfile(pkg)


def python_exe():
    shim = os.path.join(ROOT, "build", "fastwindterrain-py")
    with open(shim) as f:
        for line in f:
            if line.startswith("exec "):
                return line.split('"')[1]
    raise RuntimeError(f"could not read the interpreter out of {shim}")


def run_py(code):
    env = dict(os.environ)
    env["PYTHONPATH"] = (os.path.join(ROOT, "build", "python")
                         + os.pathsep + env.get("PYTHONPATH", ""))
    return subprocess.run([python_exe(), "-c", code], capture_output=True,
                          text=True, timeout=3600, env=env, cwd=WORKDIR)


def marks(stdout):
    out = {}
    for line in stdout.split("\n"):
        if line.startswith("::"):
            k, _, v = line[2:].partition(" ")
            out[k] = v
    return out


PRELUDE = f"""
import numpy as np
import fastwindterrain as fwt

PTS = np.loadtxt({TERRAIN_CSV!r}, delimiter=",", comments="#", skiprows=5)

FLAT = {{
    "grid": {{"n_cell": (16, 16, 32), "prob_lo": (0.0, 0.0, 0.0),
              "prob_hi": (1000.0, 1000.0, 128.0), "dz0": 4.0,
              "max_grid_size": 8}},
    "terrain": {{"flat_elevation": 0.0}},
    "inflow": {{"mode": "powerlaw", "u_ref": 8.0, "v_ref": 6.0}},
    "poisson": {{"alpha_v": 0.5, "n_projections": 4}},
}}

def hill(n_proj=4):
    return {{
        "grid": {{"n_cell": (24, 24, 40), "prob_lo": (0.0, 0.0, 0.0),
                  "prob_hi": (1000.0, 1000.0, 483.19909696997223),
                  "dz0": 4.0, "stretching_ratio": 1.05,
                  "max_grid_size": 16}},
        "terrain": {{"points": PTS}},
        "inflow": {{"mode": "powerlaw", "u_ref": 8.0, "v_ref": 6.0}},
        "poisson": {{"alpha_v": 0.5, "n_projections": n_proj}},
    }}
"""


# ---------------------------------------------------------------------------
# 1. flat ground: nothing to correct
# ---------------------------------------------------------------------------

def check_flat_is_untouched(exe):
    name = "flat ground (Phase 6 from Python)"

    r = run_py(PRELUDE + """
with fwt.session():
    s = fwt.Solver(FLAT)
    s.setup()
    v0 = s.velocity0.copy()
    print("::DIV0", repr(float(s.max_divergence_fe)))
    s.solve()
    s.diagnose()
    print("::LAMBDA", repr(float(np.abs(s.lambda_).max())))
    print("::UNCHANGED", bool(np.array_equal(s.velocity, v0)))
    print("::WZERO", repr(float(np.abs(s.velocity[2]).max())))
    print("::DIVMAX", repr(float(s.diagnostics["div_max"])))
    print("::SPEED", repr(float(np.sqrt(v0[0]**2 + v0[1]**2).max())))
""")
    assert r.returncode == 0, (
        f"[{name}] failed:\n{r.stdout[-2500:]}\n{r.stderr[-2000:]}")

    m = marks(r.stdout)
    assert float(m["LAMBDA"]) == 0.0, (
        f"[{name}] lambda is {m['LAMBDA']}, expected identically zero: a "
        f"uniform profile over flat ground is already solenoidal, so there "
        f"is nothing for the projection to correct")
    assert m["UNCHANGED"] == "True", (
        f"[{name}] the projection changed a field that was already "
        f"divergence free")
    assert float(m["WZERO"]) == 0.0, (
        f"[{name}] w is {m['WZERO']}, expected identically zero")
    assert float(m["SPEED"]) > 1.0, (
        f"[{name}] the profile is ~zero, so this proved nothing")

    print(f"[PASS] {name}  (lambda identically 0, velocity untouched, "
          f"w exactly 0, profile peak {float(m['SPEED']):.3g} m/s)")


# ---------------------------------------------------------------------------
# 2. stepwise == the loop
# ---------------------------------------------------------------------------

def check_stepwise_matches_loop(exe):
    name = "project_once vs solve"

    r = run_py(PRELUDE + """
with fwt.session():
    a = fwt.Solver(hill(3))
    a.setup()
    a.solve()

    b = fwt.Solver(hill(3))
    b.setup()
    for _ in range(3):
        b.project_once()

    print("::SAME", bool(np.array_equal(a.velocity, b.velocity)))
    print("::LAM", bool(np.array_equal(a.lambda_, b.lambda_)))
    print("::PASSES %d %d" % (a.n_projections_done, b.n_projections_done))
    print("::NONTRIVIAL", repr(float(np.abs(a.lambda_).max())))
""")
    assert r.returncode == 0, (
        f"[{name}] failed:\n{r.stdout[-2500:]}\n{r.stderr[-2000:]}")

    m = marks(r.stdout)
    a_passes, b_passes = (int(v) for v in m["PASSES"].split())
    assert a_passes == b_passes == 3, (
        f"[{name}] pass counts are {a_passes} and {b_passes}, expected 3")
    assert m["SAME"] == "True", (
        f"[{name}] stepping three passes gives a different field from "
        f"solve() with n_projections = 3. They are supposed to be the same "
        f"code.")
    assert m["LAM"] == "True", (
        f"[{name}] lambda differs between the stepwise and looped runs")
    assert float(m["NONTRIVIAL"]) > 0.0, (
        f"[{name}] lambda is identically zero, so the two agreed about "
        f"nothing")

    print(f"[PASS] {name}  (3 stepwise passes give a bit-identical field "
          f"and lambda; max|lambda| {float(m['NONTRIVIAL']):.4g})")


# ---------------------------------------------------------------------------
# 3. the projection converges, and says so
# ---------------------------------------------------------------------------

def check_convergence_reported(exe):
    name = "convergence and MLMG diagnostics"

    r = run_py(PRELUDE + """
with fwt.session():
    s = fwt.Solver(hill(4))
    s.setup()
    divs = [float(s.max_divergence_fe)]
    resids, iters = [], []
    for _ in range(4):
        resids.append(float(s.project_once()))
        iters.append(int(s.solve_iterations))
        divs.append(float(s.max_divergence_fe))
    print("::DIVS", ",".join(repr(d) for d in divs))
    print("::RESIDMAX", repr(max(resids)))
    print("::ITERS", ",".join(str(i) for i in iters))
""")
    assert r.returncode == 0, (
        f"[{name}] failed:\n{r.stdout[-2500:]}\n{r.stderr[-2000:]}")

    m = marks(r.stdout)
    divs = [float(v) for v in m["DIVS"].split(",")]
    iters = [int(v) for v in m["ITERS"].split(",")]

    assert all(b < a for a, b in zip(divs, divs[1:])), (
        f"[{name}] the controlled divergence does not fall monotonically: "
        f"{divs}")
    assert divs[-1] < 0.6 * divs[0], (
        f"[{name}] four passes reduced the divergence only from {divs[0]} "
        f"to {divs[-1]}")
    assert float(m["RESIDMAX"]) < 1.0e-9, (
        f"[{name}] MLMG's worst residual was {m['RESIDMAX']}; the solve is "
        f"not converging")
    assert all(0 < i < 200 for i in iters), (
        f"[{name}] MLMG iteration counts are {iters}; zero would mean the "
        f"count is not being reported, and 200 is max_iter -- neither is a "
        f"converged solve")

    print(f"[PASS] {name}  (div {divs[0]:.4g} -> {divs[-1]:.4g} "
          f"monotonically over 4 passes; MLMG {iters} iterations, worst "
          f"residual {float(m['RESIDMAX']):.2e})")


# ---------------------------------------------------------------------------
# 4. the ghost refill, finally observable
# ---------------------------------------------------------------------------

def check_ghost_refill(exe):
    """Phase 11 wrote that set_velocity refills the ghosts and could not
    test it, because ghost cells are not exposed. A solve can now follow
    a write, so the claim is testable through the SCHEME divergence,
    which reads them with a five-point stencil.

    Solver B is handed the field solver A produced. If B's ghosts were
    left as its own initial profile rather than refilled from the field
    it was given, its divergence would differ."""
    name = "set_velocity refills the ghosts"

    r = run_py(PRELUDE + """
with fwt.session():
    a = fwt.Solver(hill(2))
    a.setup()
    a.solve()
    va = a.velocity.copy()
    div_a = float(a.max_divergence)

    b = fwt.Solver(hill(2))
    b.setup()
    div_b_before = float(b.max_divergence)
    b.set_velocity(va)
    div_b = float(b.max_divergence)

    print("::A", repr(div_a))
    print("::B", repr(div_b))
    print("::BEFORE", repr(div_b_before))
    print("::FIELD", bool(np.array_equal(b.velocity, va)))
""")
    assert r.returncode == 0, (
        f"[{name}] failed:\n{r.stdout[-2500:]}\n{r.stderr[-2000:]}")

    m = marks(r.stdout)
    assert m["FIELD"] == "True", (
        f"[{name}] the valid region did not round-trip")
    assert float(m["BEFORE"]) != float(m["A"]), (
        f"[{name}] the two fields already had the same divergence, so this "
        f"compared nothing")
    assert float(m["B"]) == float(m["A"]), (
        f"[{name}] after set_velocity the scheme divergence is {m['B']}, "
        f"but the solver that produced that field reports {m['A']}. The "
        f"valid regions match, so the ghost cells do not -- set_velocity "
        f"is not refilling them.")

    print(f"[PASS] {name}  (scheme divergence {float(m['A']):.6g} matches "
          f"exactly after the write, from {float(m['BEFORE']):.6g} before; "
          f"stale ghosts would show here)")


# ---------------------------------------------------------------------------
# 5. no ParmParse leak into the solve
# ---------------------------------------------------------------------------

def check_no_parmparse_leak(exe):
    name = "no ParmParse leak"

    inputs = os.path.join(WORKDIR, "inputs_leak")
    with open(inputs, "w") as f:
        f.write("grid.n_cell = 8 8 8\n")
        f.write("grid.prob_lo = 0.0 0.0 0.0\n")
        f.write("grid.prob_hi = 100.0 100.0 32.0\n")
        f.write("grid.dz0 = 4.0\n")
        f.write("inflow.u_ref = 1.0\n")
        # The values a leak would carry into the dict-configured run.
        f.write("poisson.alpha_v = 0.125\n")
        f.write("poisson.n_projections = 7\n")
        f.write("anisotropy.enable = 1\n")
        f.write("obrien.enable = 1\n")

    r = run_py(PRELUDE + f"""
fwt.initialize([{'"' + inputs + '"'}])
s = fwt.Solver(FLAT)          # poisson.alpha_v = 0.5, n_projections = 4
s.setup()
s.solve()
print("::PASSES", s.n_projections_done)
print("::ALPHAV", repr(float(s.alpha_v.max())))
print("::ANISO_FLAT", bool(s.alpha_v.min() == s.alpha_v.max()))
fwt.finalize()
""")
    assert r.returncode == 0, (
        f"[{name}] failed:\n{r.stdout[-2500:]}\n{r.stderr[-2000:]}")

    m = marks(r.stdout)
    assert int(m["PASSES"]) == 4, (
        f"[{name}] the run did {m['PASSES']} projections; the dict says 4 "
        f"and the inputs file says 7, so ParmParse leaked through")
    assert float(m["ALPHAV"]) == 0.5, (
        f"[{name}] alpha_v is {m['ALPHAV']}; the dict says 0.5 and the "
        f"inputs file says 0.125")
    assert m["ANISO_FLAT"] == "True", (
        f"[{name}] alpha_v varies in space, so anisotropy.enable = 1 "
        f"leaked from the inputs file -- the dict leaves it off")

    print(f"[PASS] {name}  (4 passes and alpha_v 0.5 from the dict, not 7 "
          f"and 0.125 from the inputs file; anisotropy stayed off)")


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

    if not bindings_available():
        print("[SKIP] the Python bindings are not built. Configure with "
              "-DFWT_PYTHON=ON to run this group.")
        return 0

    failed = []
    for check in (check_flat_is_untouched, check_stepwise_matches_loop,
                  check_convergence_reported, check_ghost_refill,
                  check_no_parmparse_leak):
        try:
            check(exe)
        except AssertionError as e:
            print(f"[FAIL] {check.__name__}: {e}")
            failed.append(check.__name__)
        except Exception as e:
            print(f"[ERROR] {check.__name__}: {e}")
            failed.append(check.__name__)

    if failed:
        print(f"\n{len(failed)} Phase 13 regtest case(s) failed: {failed}")
        return 1

    print("\nAll Phase 13 regtest cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
