#!/usr/bin/env python3
"""
Phase 10 regtest checker -- Grid built from Python.

Validates:

  no ParmParse leak -> THE point of the phase. ParmParse is
                       process-global and persists for the life of an
                       AMReX initialization, so a case that omits a
                       parameter would inherit whatever an earlier case
                       set. For one command-line run that never comes up.
                       For a loop generating a few hundred training
                       samples it is silent corruption spread across a
                       dataset, with no failure anywhere to notice

  parity            -> a grid built from a dict and the same grid built
                       from an inputs file agree EXACTLY, not to a
                       tolerance

  errors raise      -> a bad input raises ValueError instead of aborting
                       the interpreter, and an unknown key is refused
                       rather than ignored. ParmParse accepts a typo
                       silently and mentions it once at finalize; that is
                       how a misspelled key produces a whole dataset on
                       the wrong grid

  warnings warn     -> an overshoot is a Python UserWarning, so it can be
                       filtered, captured, or promoted to an error --
                       rather than a line of stdout nobody reads

  repeatable        -> many grids can be built and destroyed inside one
                       AMReX initialization, which is what dataset
                       generation requires

The executable's behaviour must be untouched by all of this: the
undershoot abort, its diagnostic and its nonzero exit are asserted by the
phase1_grid group, which still passes.

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

# A stretched grid whose height is quoted exactly, so it neither
# overshoots nor undershoots.
STRETCHED = {
    "n_cell": (8, 8, 40),
    "prob_lo": (0.0, 0.0, 0.0),
    "prob_hi": (1000.0, 1000.0, 483.19909696997223),
    "dz0": 4.0,
    "stretching_ratio": 1.05,
    "max_grid_size": 16,
}


def bindings_available():
    shim = os.path.join(ROOT, "build", "fastwindterrain-py")
    pkg = os.path.join(ROOT, "build", "python", "fastwindterrain",
                       "__init__.py")
    return os.path.isfile(shim) and os.path.isfile(pkg)


def python_exe():
    """The interpreter the module was built for, read out of the shim."""
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
                          text=True, timeout=1800, env=env, cwd=WORKDIR)


def marks(stdout):
    """Lines the test script emitted as `KEY value`."""
    out = {}
    for line in stdout.split("\n"):
        if line.startswith("::"):
            k, _, v = line[2:].partition(" ")
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# 1. the dict path does not read ParmParse
# ---------------------------------------------------------------------------

def check_no_parmparse_leak(exe):
    """Initialize from an inputs file that sets grid.stretching_ratio =
    1.03, then build a Grid from a dict that OMITS it. The default 1.0
    must win.

    The discriminator is chosen so the two answers cannot be confused:
    with 40 cells of 4 m and ratio 1.0 the column is exactly 160 m, which
    matches the requested top and warns about nothing. A leaked 1.03
    would make it 301.6 m -- an overshoot, a warning, and a different
    prob_hi."""
    name = "no ParmParse leak"

    inputs = os.path.join(REGTEST_ROOT, "phase6_solve_correction",
                          "inputs_flat")
    r = run_py(f"""
import warnings
import fastwindterrain as fwt
fwt.initialize([{inputs!r}])
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    g = fwt.Grid({{"n_cell": (8, 8, 40), "prob_lo": (0.0, 0.0, 0.0),
                   "prob_hi": (1000.0, 1000.0, 160.0), "dz0": 4.0}})
    print("::RATIO", g.stretching_ratio)
    print("::TOP", g.prob_hi[2])
    print("::NWARN", len(w))
    print("::MGS", g.max_grid_size)
fwt.finalize()
""")
    assert r.returncode == 0, (
        f"[{name}] failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")

    m = marks(r.stdout)
    assert float(m["RATIO"]) == 1.0, (
        f"[{name}] the dict omitted stretching_ratio, so the Grid must use "
        f"the default 1.0 -- it used {m['RATIO']}, which is the value the "
        f"inputs file left in ParmParse. Every case in a generation loop "
        f"after the first would be silently wrong.")
    assert float(m["TOP"]) == 160.0, (
        f"[{name}] prob_hi[2] is {m['TOP']}, expected 160.0 (40 x 4 m "
        f"uniform). 301.6 would mean the ratio leaked.")
    assert int(m["NWARN"]) == 0, (
        f"[{name}] the grid matched its requested height exactly, so "
        f"nothing should have warned; got {m['NWARN']} warning(s)")
    assert int(m["MGS"]) == 32, (
        f"[{name}] max_grid_size is {m['MGS']}, expected the default 32; "
        f"the inputs file sets 32 too, but via ParmParse -- this checks "
        f"the default, not the leak")

    print(f"[PASS] {name}  (dict-built Grid takes the default ratio 1.0, "
          f"not the 1.03 the inputs file left in ParmParse)")


# ---------------------------------------------------------------------------
# 2. dict and inputs file agree exactly
# ---------------------------------------------------------------------------

def check_parity_with_inputs_file(exe):
    """The same grid, described two ways, must be the same grid to the
    last bit -- Phase 9's guarantee extended to the new entry point."""
    name = "dict vs inputs file"

    # Write an inputs file describing exactly STRETCHED, run it through
    # the executable, and read z_face back out of the report.
    inputs = os.path.join(WORKDIR, "inputs_parity")
    with open(inputs, "w") as f:
        f.write("# generated by the Phase 10 checker\n")
        f.write(f"grid.n_cell  = {' '.join(str(v) for v in STRETCHED['n_cell'])}\n")
        f.write(f"grid.prob_lo = {' '.join(repr(v) for v in STRETCHED['prob_lo'])}\n")
        f.write(f"grid.prob_hi = {' '.join(repr(v) for v in STRETCHED['prob_hi'])}\n")
        f.write(f"grid.dz0 = {STRETCHED['dz0']!r}\n")
        f.write(f"grid.stretching_ratio = {STRETCHED['stretching_ratio']!r}\n")
        f.write(f"grid.max_grid_size = {STRETCHED['max_grid_size']}\n")
        f.write("terrain.flat_elevation = 0.0\n")
        f.write("inflow.u_ref = 8.0\ninflow.v_ref = 6.0\n")
        f.write("poisson.n_projections = 1\n")
        f.write("grid.output_format = report\n")
        f.write("grid.report_file = grid_report_parity.txt\n")

    r = subprocess.run([exe, inputs], cwd=WORKDIR, capture_output=True,
                       text=True, timeout=3600)
    assert r.returncode == 0, (
        f"[{name}] the executable failed:\n{r.stdout[-2000:]}")

    z_face_file = {}
    with open(os.path.join(WORKDIR, "grid_report_parity.txt")) as f:
        for line in f:
            p = line.split()
            if len(p) == 3 and p[0] == "z_face":
                z_face_file[int(p[1])] = float(p[2])

    r2 = run_py(f"""
import fastwindterrain as fwt
with fwt.session():
    g = fwt.Grid({STRETCHED!r})
    for k, z in enumerate(g.z_face):
        print("::Z%d" % k, repr(float(z)))
""")
    assert r2.returncode == 0, (
        f"[{name}] the Python build failed:\n{r2.stdout[-2000:]}\n"
        f"{r2.stderr[-2000:]}")

    m = marks(r2.stdout)
    z_face_py = {int(k[1:]): float(v) for k, v in m.items()
                 if k.startswith("Z")}

    assert len(z_face_py) == len(z_face_file) == STRETCHED["n_cell"][2] + 1, (
        f"[{name}] expected {STRETCHED['n_cell'][2] + 1} faces, got "
        f"{len(z_face_py)} from Python and {len(z_face_file)} from the "
        f"report")

    for k in sorted(z_face_file):
        assert z_face_py[k] == z_face_file[k], (
            f"[{name}] z_face[{k}] differs: {z_face_py[k]!r} from the dict, "
            f"{z_face_file[k]!r} from the inputs file. The two paths must "
            f"agree exactly, not approximately.")

    print(f"[PASS] {name}  ({len(z_face_file)} faces identical to the last "
          f"bit)")


# ---------------------------------------------------------------------------
# 3. bad input raises; the interpreter survives
# ---------------------------------------------------------------------------

def check_errors_raise(exe):
    name = "bad input raises"

    r = run_py("""
import fastwindterrain as fwt

def expect(label, params):
    try:
        fwt.Grid(params)
        print("::%s accepted" % label)
    except ValueError:
        print("::%s raised" % label)

with fwt.session():
    good = {"n_cell": (8, 8, 40), "prob_lo": (0.0, 0.0, 0.0),
            "prob_hi": (1000.0, 1000.0, 160.0), "dz0": 4.0}

    expect("TYPO", dict(good, stretching_ration=1.05))
    expect("NEGDZ", dict(good, dz0=-1.0))
    expect("ZERORATIO", dict(good, stretching_ratio=0.0))
    expect("SHORTTRIPLE", dict(good, n_cell=(8, 8)))
    expect("NOTNUMBER", dict(good, dz0="four"))
    expect("INVERTED", dict(good, prob_hi=(1000.0, 1000.0, -5.0)))
    undershoot = dict(good, prob_hi=(1000.0, 1000.0, 1000.0), dz0=1.0)
    expect("UNDERSHOOT", undershoot)

    missing = {k: v for k, v in good.items() if k != "dz0"}
    expect("MISSING", missing)

    # The interpreter is still usable after all of that, which is the
    # whole reason these raise instead of aborting.
    g = fwt.Grid(good)
    print("::ALIVE", g.nz)
""")
    assert r.returncode == 0, (
        f"[{name}] the interpreter did not survive -- a bad input aborted "
        f"instead of raising (exit {r.returncode})\n{r.stdout[-2000:]}\n"
        f"{r.stderr[-2000:]}")

    m = marks(r.stdout)
    for label in ("TYPO", "NEGDZ", "ZERORATIO", "SHORTTRIPLE", "NOTNUMBER",
                  "INVERTED", "UNDERSHOOT", "MISSING"):
        assert m.get(label) == "raised", (
            f"[{name}] {label} was {m.get(label)}, expected raised")
    assert m.get("ALIVE") == "40", (
        f"[{name}] a valid Grid could not be built afterwards")

    print(f"[PASS] {name}  (8 bad inputs all raise ValueError, including an "
          f"unknown key; the interpreter survives every one)")


# ---------------------------------------------------------------------------
# 4. an overshoot is a Python warning
# ---------------------------------------------------------------------------

def check_overshoot_warns(exe):
    name = "overshoot warns"

    r = run_py("""
import warnings
import fastwindterrain as fwt
with fwt.session():
    over = {"n_cell": (8, 8, 10), "prob_lo": (0.0, 0.0, 0.0),
            "prob_hi": (1000.0, 1000.0, 10.0), "dz0": 4.0}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        g = fwt.Grid(over)
        print("::NWARN", len(w))
        print("::CATEGORY", w[0].category.__name__ if w else "none")
        print("::TEXT", "overshoots" in str(w[0].message) if w else False)
    print("::TOP", g.prob_hi[2])

    # Promotable to an error, which is what a dataset generator would do
    # rather than let a silently adjusted domain into its samples.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fwt.Grid(over)
        print("::PROMOTED no")
    except UserWarning:
        print("::PROMOTED yes")

    # And a grid that fits exactly must not warn at all.
    with warnings.catch_warnings(record=True) as w2:
        warnings.simplefilter("always")
        fwt.Grid({"n_cell": (8, 8, 10), "prob_lo": (0.0, 0.0, 0.0),
                  "prob_hi": (1000.0, 1000.0, 40.0), "dz0": 4.0})
        print("::QUIET", len(w2))
""")
    assert r.returncode == 0, (
        f"[{name}] failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")

    m = marks(r.stdout)
    assert m["NWARN"] == "1", (
        f"[{name}] expected exactly one warning, got {m['NWARN']}")
    assert m["CATEGORY"] == "UserWarning", (
        f"[{name}] expected a UserWarning, got {m['CATEGORY']}")
    assert m["TEXT"] == "True", (
        f"[{name}] the warning text does not mention the overshoot")
    assert float(m["TOP"]) == 40.0, (
        f"[{name}] prob_hi[2] should have been adjusted to 40.0, got "
        f"{m['TOP']}")
    assert m["PROMOTED"] == "yes", (
        f"[{name}] simplefilter('error') must turn the overshoot into an "
        f"exception; a generator needs that to reject a silently adjusted "
        f"domain")
    assert m["QUIET"] == "0", (
        f"[{name}] an exactly-fitting grid warned about something")

    print(f"[PASS] {name}  (UserWarning, promotable to an error, prob_hi "
          f"adjusted to 40.0; an exact fit stays silent)")


# ---------------------------------------------------------------------------
# 5. many grids in one initialization
# ---------------------------------------------------------------------------

def check_repeatable(exe):
    """Dataset generation means hundreds of cases inside ONE AMReX
    initialization. Each must be independent of the last."""
    name = "many grids per session"

    r = run_py("""
import fastwindterrain as fwt
with fwt.session():
    tops = []
    for nz in range(10, 60, 5):
        g = fwt.Grid({"n_cell": (8, 8, nz), "prob_lo": (0.0, 0.0, 0.0),
                      "prob_hi": (1000.0, 1000.0, 4.0 * nz), "dz0": 4.0})
        tops.append((nz, g.nz, g.prob_hi[2], float(g.z_face[-1])))
        del g
    print("::N", len(tops))
    print("::OK", all(nz == got and top == 4.0 * nz and abs(zf - top) < 1e-9
                      for nz, got, top, zf in tops))
""")
    assert r.returncode == 0, (
        f"[{name}] failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")

    m = marks(r.stdout)
    assert m["N"] == "10", f"[{name}] expected 10 grids, got {m['N']}"
    assert m["OK"] == "True", (
        f"[{name}] a grid did not match its own parameters; construction is "
        f"not independent between cases")

    print(f"[PASS] {name}  (10 grids built and destroyed in one "
          f"initialization, each independent)")


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
    for check in (check_no_parmparse_leak, check_parity_with_inputs_file,
                  check_errors_raise, check_overshoot_warns,
                  check_repeatable):
        try:
            check(exe)
        except AssertionError as e:
            print(f"[FAIL] {check.__name__}: {e}")
            failed.append(check.__name__)
        except Exception as e:
            print(f"[ERROR] {check.__name__}: {e}")
            failed.append(check.__name__)

    if failed:
        print(f"\n{len(failed)} Phase 10 regtest case(s) failed: {failed}")
        return 1

    print("\nAll Phase 10 regtest cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
