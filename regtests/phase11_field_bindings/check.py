#!/usr/bin/env python3
"""
Phase 11 regtest checker -- fields as numpy.

Validates:

  shapes        -> every field comes back as (ncomp, nz, ny, nx) with the
                   leading axis dropped for one component, the nodal
                   lambda as (nz+1, ny+1, nx+1), and the mask as int32

  index order   -> the arrays are compared CELL BY CELL against the
                   plotfile the C++ run wrote. This is the check that
                   catches a transposed array, which is the failure that
                   would otherwise ruin a training set without ever
                   raising anything

  round trip    -> write an array in, read it back, require it BIT-EXACT.
                   Not "close": the values never leave double precision,
                   so anything less would mean the gather or the scatter
                   is doing arithmetic it should not

  decomposition -> the same case at max_grid_size 8 and 32 -- 9 boxes
                   against 1 -- must give identical arrays. A MultiFab is
                   N separate FArrayBoxes, so this is what actually
                   validates the gather

  errors        -> a mismatched shape raises rather than being broadcast,
                   and reading a field before setup() raises rather than
                   returning something empty

Not covered here: the ghost refill that follows set_velocity. Ghost cells
are deliberately not exposed, so nothing in this phase can observe it;
its effect becomes testable in Phase 13, when a solve can follow a write.

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

sys.path.insert(0, REGTEST_ROOT)

WORKDIR = os.path.join(ROOT, "build", "regtests", GROUP)

CASE = os.path.join(REGTEST_ROOT, "phase8_diagnostics_output", "inputs_both")
TERRAIN = os.path.join(REGTEST_ROOT, "phase8_diagnostics_output",
                       "terrain_hill.csv")
NX, NY, NZ = 24, 24, 40


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
                          text=True, timeout=1800, env=env, cwd=WORKDIR)


def marks(stdout):
    out = {}
    for line in stdout.split("\n"):
        if line.startswith("::"):
            k, _, v = line[2:].partition(" ")
            out[k] = v
    return out


SETUP = f"""
import numpy as np
import fastwindterrain as fwt
ARGS = [{CASE!r}, "terrain.file={TERRAIN}", "grid.output_format=report",
        "grid.report_file=ignored.txt"]
"""


# ---------------------------------------------------------------------------
# 1. shapes and dtypes
# ---------------------------------------------------------------------------

def check_shapes(exe):
    name = "field shapes"

    r = run_py(SETUP + """
with fwt.session(ARGS):
    s = fwt.Solver()
    s.setup()
    print("::SHAPE", s.shape)
    print("::BOXES", s.grid.n_boxes)
    for n in ("velocity", "velocity0", "sigma", "mask", "z_terrain",
              "alpha_h", "alpha_v", "lambda_"):
        a = getattr(s, n)
        print("::%s %s|%s" % (n.upper(), tuple(a.shape), a.dtype))
""")
    assert r.returncode == 0, (
        f"[{name}] failed:\n{r.stdout[-2500:]}\n{r.stderr[-2000:]}")

    m = marks(r.stdout)
    assert m["SHAPE"] == f"({NZ}, {NY}, {NX})", (
        f"[{name}] shape is {m['SHAPE']}, expected ({NZ}, {NY}, {NX})")

    expected = {
        "VELOCITY":  (f"(3, {NZ}, {NY}, {NX})", "float64"),
        "VELOCITY0": (f"(3, {NZ}, {NY}, {NX})", "float64"),
        "SIGMA":     (f"(3, {NZ}, {NY}, {NX})", "float64"),
        "MASK":      (f"({NZ}, {NY}, {NX})", "int32"),
        "Z_TERRAIN": (f"({NZ}, {NY}, {NX})", "float64"),
        "ALPHA_H":   (f"({NZ}, {NY}, {NX})", "float64"),
        "ALPHA_V":   (f"({NZ}, {NY}, {NX})", "float64"),
        # Nodal: one more point in every direction.
        "LAMBDA_":   (f"({NZ + 1}, {NY + 1}, {NX + 1})", "float64"),
    }
    for key, (shape, dtype) in expected.items():
        got_shape, _, got_dtype = m[key].partition("|")
        assert got_shape == shape, (
            f"[{name}] {key} has shape {got_shape}, expected {shape}")
        assert got_dtype == dtype, (
            f"[{name}] {key} has dtype {got_dtype}, expected {dtype}")

    assert int(m["BOXES"]) > 1, (
        f"[{name}] the case must be decomposed into more than one box for "
        f"this group to test anything; it has {m['BOXES']}")

    print(f"[PASS] {name}  (8 fields, channels-first, nodal lambda "
          f"({NZ + 1}, {NY + 1}, {NX + 1}), int32 mask, over "
          f"{m['BOXES']} boxes)")


# ---------------------------------------------------------------------------
# 2. index order, against the plotfile
# ---------------------------------------------------------------------------

def check_index_order(exe):
    """arr[c, k, j, i] must be the value the plotfile has at cell
    (i, j, k). A transposed array would pass every other check in this
    file and quietly ruin a training set."""
    name = "index order vs the plotfile"

    r = subprocess.run(
        [exe, CASE, f"terrain.file={TERRAIN}",
         "grid.output_format=both", "output.format=plt",
         "grid.plot_file=plt_fields",
         "grid.report_file=grid_report_fields.txt"],
        cwd=WORKDIR, capture_output=True, text=True, timeout=3600)
    assert r.returncode == 0, (
        f"[{name}] the executable failed:\n{r.stdout[-2000:]}")

    # The Python side runs setup() only, so its velocity is the field
    # BEFORE any correction -- which is what the plotfile calls u0.
    r2 = run_py(SETUP + """
with fwt.session(ARGS):
    s = fwt.Solver()
    s.setup()
    np.save("py_velocity.npy", s.velocity)
    np.save("py_mask.npy", s.mask)
    np.save("py_zterr.npy", s.z_terrain)
    np.save("py_alphav.npy", s.alpha_v)
    print("::OK 1")
""")
    assert r2.returncode == 0, (
        f"[{name}] the Python setup failed:\n{r2.stdout[-2500:]}\n"
        f"{r2.stderr[-2000:]}")

    # Read both sides back in the checker's own interpreter, which is the
    # one that has plotfile.py.
    r3 = run_py(f"""
import numpy as np, sys
sys.path.insert(0, {REGTEST_ROOT!r})
from plotfile import Plotfile

pf = Plotfile("plt_fields")
vel = np.load("py_velocity.npy")
mask = np.load("py_mask.npy")
zt = np.load("py_zterr.npy")
av = np.load("py_alphav.npy")

pairs = [("u0", vel[0]), ("v0", vel[1]), ("w0", vel[2]),
         ("mask", mask), ("terrain_z", zt), ("alpha_v", av)]

worst = 0.0
n = 0
for pfname, arr in pairs:
    f = pf.field(pfname)
    for k in range(0, {NZ}, 3):
        for j in range(0, {NY}, 5):
            for i in range(0, {NX}, 5):
                a = float(arr[k, j, i])
                b = float(f(i, j, k))
                if a != b:
                    worst = max(worst, abs(a - b))
                n += 1
print("::N", n)
print("::WORST", repr(worst))
""")
    assert r3.returncode == 0, (
        f"[{name}] the comparison failed:\n{r3.stdout[-2500:]}\n"
        f"{r3.stderr[-2000:]}")

    m = marks(r3.stdout)
    assert float(m["WORST"]) == 0.0, (
        f"[{name}] arr[c, k, j, i] does not equal the plotfile at "
        f"(i, j, k); worst difference {m['WORST']}. Either the axes are "
        f"transposed or the gather is wrong.")
    assert int(m["N"]) > 500, (
        f"[{name}] only {m['N']} cells were compared; the sampling is "
        f"broken")

    print(f"[PASS] {name}  ({m['N']} samples across 6 fields, all exactly "
          f"equal)")


# ---------------------------------------------------------------------------
# 3. bit-exact round trip
# ---------------------------------------------------------------------------

def check_round_trip(exe):
    name = "round trip"

    r = run_py(SETUP + """
with fwt.session(ARGS):
    s = fwt.Solver()
    s.setup()
    v = s.velocity

    # Values chosen so a transpose, an off-by-one or a dropped component
    # all change the result: every cell is distinct and the three
    # components are far apart in magnitude.
    c, nz, ny, nx = v.shape
    kk, jj, ii = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx),
                             indexing="ij")
    new = np.empty_like(v)
    new[0] = kk + 1e-3 * jj + 1e-6 * ii
    new[1] = -2.0 * (kk + 1e-3 * jj + 1e-6 * ii)
    new[2] = 1e4 + kk - 1e-3 * jj

    s.set_velocity(new)
    back = s.velocity
    print("::EXACT", bool(np.array_equal(back, new)))
    print("::MAXDIFF", repr(float(np.max(np.abs(back - new)))))

    # A second write must not accumulate anything.
    s.set_velocity(new)
    print("::EXACT2", bool(np.array_equal(s.velocity, new)))

    # The returned array is a copy: scribbling on it changes nothing.
    got = s.velocity
    got[:] = 12345.0
    print("::COPY", bool(np.array_equal(s.velocity, new)))
""")
    assert r.returncode == 0, (
        f"[{name}] failed:\n{r.stdout[-2500:]}\n{r.stderr[-2000:]}")

    m = marks(r.stdout)
    assert m["EXACT"] == "True", (
        f"[{name}] the round trip is not bit-exact; largest difference "
        f"{m['MAXDIFF']}. The values never leave double precision, so any "
        f"difference means the gather or scatter is doing arithmetic.")
    assert m["EXACT2"] == "True", (
        f"[{name}] writing the same field twice did not give the same "
        f"result")
    assert m["COPY"] == "True", (
        f"[{name}] the returned array is not a copy -- writing into it "
        f"changed the solver's field")

    print(f"[PASS] {name}  (bit-exact over "
          f"{3 * NZ * NY * NX} values, repeatable, and the array handed "
          f"back is a copy)")


# ---------------------------------------------------------------------------
# 4. the gather does not depend on the decomposition
# ---------------------------------------------------------------------------

def check_decomposition_invariance(exe):
    """A MultiFab is N separate FArrayBoxes. Nine of them must gather to
    the same array as one."""
    name = "decomposition invariance"

    r = run_py(SETUP + """
def fields(mgs):
    with fwt.session(ARGS + ["grid.max_grid_size=%d" % mgs]):
        s = fwt.Solver()
        s.setup()
        return (s.grid.n_boxes,
                {n: getattr(s, n).copy()
                 for n in ("velocity", "mask", "z_terrain", "alpha_v",
                           "sigma", "lambda_")})

n_small, small = fields(8)
n_big, big = fields(64)
print("::BOXES %d %d" % (n_small, n_big))
print("::SAME", all(np.array_equal(small[k], big[k]) for k in small))
for k in small:
    if not np.array_equal(small[k], big[k]):
        print("::DIFF", k)
""")
    assert r.returncode == 0, (
        f"[{name}] failed:\n{r.stdout[-2500:]}\n{r.stderr[-2000:]}")

    m = marks(r.stdout)
    n_small, n_big = (int(v) for v in m["BOXES"].split())
    assert n_small > n_big, (
        f"[{name}] expected max_grid_size=8 to give more boxes than 64; "
        f"got {n_small} and {n_big}")
    assert n_big == 1, (
        f"[{name}] max_grid_size=64 should give a single box, got {n_big}")
    assert m["SAME"] == "True", (
        f"[{name}] the arrays depend on the box decomposition "
        f"({m.get('DIFF', 'unknown field')} differs). The gather is wrong.")

    print(f"[PASS] {name}  (6 fields identical across {n_small} boxes and "
          f"{n_big})")


# ---------------------------------------------------------------------------
# 5. errors
# ---------------------------------------------------------------------------

def check_errors(exe):
    name = "field errors"

    r = run_py(SETUP + """
with fwt.session(ARGS):
    s = fwt.Solver()

    # Before setup there is no field to return, and saying so is better
    # than handing back an empty array.
    try:
        s.velocity
        print("::EARLY accepted")
    except RuntimeError:
        print("::EARLY raised")
    try:
        s.shape
        print("::EARLYSHAPE accepted")
    except RuntimeError:
        print("::EARLYSHAPE raised")

    s.setup()
    v = s.velocity

    def expect(label, arr):
        try:
            s.set_velocity(arr)
            print("::%s accepted" % label)
        except (ValueError, TypeError):
            print("::%s raised" % label)

    expect("SHORT", v[:, :, :, :-1])
    expect("TWOCOMP", v[:2])
    expect("FLAT", v.reshape(-1))
    expect("TRANSPOSED", np.ascontiguousarray(v.transpose(0, 3, 2, 1)))

    # Unchanged after all the rejected writes.
    print("::UNTOUCHED", bool(np.array_equal(s.velocity, v)))

    # A float32 array is fine: it is a widening conversion, not a
    # reinterpretation, and refusing it would be pedantry.
    s.set_velocity(v.astype(np.float32))
    print("::F32", bool(np.allclose(s.velocity, v, rtol=0, atol=1e-6)))
""")
    assert r.returncode == 0, (
        f"[{name}] failed:\n{r.stdout[-2500:]}\n{r.stderr[-2000:]}")

    m = marks(r.stdout)
    assert m["EARLY"] == "raised", (
        f"[{name}] reading a field before setup() must raise")
    assert m["EARLYSHAPE"] == "raised", (
        f"[{name}] reading shape before setup() must raise")
    for label in ("SHORT", "TWOCOMP", "FLAT"):
        assert m[label] == "raised", (
            f"[{name}] {label} was {m[label]}, expected raised")
    # A transposed array has the same shape here only if nx == nz; the
    # case is 24x24x40, so it is genuinely a different shape.
    assert m["TRANSPOSED"] == "raised", (
        f"[{name}] a transposed array was accepted")
    assert m["UNTOUCHED"] == "True", (
        f"[{name}] a rejected write still modified the field")
    assert m["F32"] == "True", (
        f"[{name}] a float32 array should be accepted by widening")

    print(f"[PASS] {name}  (4 bad shapes rejected with the field left "
          f"untouched; float32 widens; reads before setup raise)")


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
    for check in (check_shapes, check_index_order, check_round_trip,
                  check_decomposition_invariance, check_errors):
        try:
            check(exe)
        except AssertionError as e:
            print(f"[FAIL] {check.__name__}: {e}")
            failed.append(check.__name__)
        except Exception as e:
            print(f"[ERROR] {check.__name__}: {e}")
            failed.append(check.__name__)

    if failed:
        print(f"\n{len(failed)} Phase 11 regtest case(s) failed: {failed}")
        return 1

    print("\nAll Phase 11 regtest cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
