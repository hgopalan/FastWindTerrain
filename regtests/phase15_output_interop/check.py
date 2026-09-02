#!/usr/bin/env python3
"""
Phase 15 regtest checker -- output in memory, and the file paths.

Validates:

  in-memory == ascii -> THE cross-check. The same case is written by the
                        C++ ascii backend and read back in memory through
                        fields(), and every value of every field must be
                        IDENTICAL. Both come from CollectOutputFields, so
                        anything less would mean a third assembly of "the
                        output fields" had crept in

  in-memory == plt   -> and the same against the plotfile, so all three
                        consumers of that one gather agree

  writers            -> write_plotfile / write_ascii / write_report put
                        files where they are told, without ParmParse
                        being involved

  output config      -> the output section of the config dict really
                        selects, and an unknown key or value raises

  no leak            -> a dict-configured run writes where the dict says,
                        not where an inputs file left grid.plot_file
                        pointing. Getting this wrong would have case 2 of
                        a generation loop overwrite case 1's output

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

NX, NY, NZ = 24, 24, 40
N_FIELDS = 17


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

def case(**kw):
    cfg = {{
        "grid": {{"n_cell": ({NX}, {NY}, {NZ}), "prob_lo": (0.0, 0.0, 0.0),
                  "prob_hi": (1000.0, 1000.0, 483.19909696997223),
                  "dz0": 4.0, "stretching_ratio": 1.05,
                  "max_grid_size": 16}},
        "terrain": {{"points": PTS}},
        "inflow": {{"mode": "powerlaw", "u_ref": 8.0, "v_ref": 6.0}},
        "anisotropy": {{"enable": True}},
        "obrien": {{"enable": True}},
        "poisson": {{"alpha_v": 0.5, "n_projections": 4}},
    }}
    cfg.update(kw)
    return cfg

def solved(**kw):
    s = fwt.Solver(case(**kw))
    s.setup(); s.solve(); s.diagnose()
    return s
"""


def read_ascii(path):
    """The gathered plain-text field file, as {name: {(i,j,k): value}}."""
    cols = None
    rows = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                body = line[1:].strip()
                if body.startswith("columns:"):
                    cols = body[len("columns:"):].split()
                continue
            if not line.strip():
                continue
            p = line.split()
            i, j, k = int(p[0]), int(p[1]), int(p[2])
            for name, v in zip(cols[5:], p[5:]):
                rows.setdefault(name, {})[(i, j, k)] = float(v)
    return rows


# ---------------------------------------------------------------------------
# 1. in-memory vs the ascii file, and vs the plotfile
# ---------------------------------------------------------------------------

def check_in_memory_matches_files(exe):
    name = "in-memory vs the written files"

    r = run_py(PRELUDE + """
with fwt.session():
    s = solved()
    s.write_ascii("fields_mem.txt")
    s.write_plotfile("plt_mem")
    f = s.fields()
    np.savez("py_fields.npz", **f)
    print("::NFIELDS", len(f))
    print("::NAMES", ",".join(sorted(f)))
""")
    assert r.returncode == 0, (
        f"[{name}] the run failed:\n{r.stdout[-2500:]}\n{r.stderr[-2000:]}")

    m = marks(r.stdout)
    assert int(m["NFIELDS"]) == N_FIELDS, (
        f"[{name}] fields() returned {m['NFIELDS']} fields, expected "
        f"{N_FIELDS}")

    # Compare in the checker's interpreter, which has plotfile.py.
    r2 = run_py(f"""
import numpy as np, sys
sys.path.insert(0, {REGTEST_ROOT!r})
from plotfile import Plotfile

mem = dict(np.load("py_fields.npz"))
pf = Plotfile("plt_mem")

# --- against the plotfile ---
worst_plt, where_plt, n_plt = 0.0, None, 0
for nm in sorted(mem):
    fld = pf.field(nm)
    a = mem[nm]
    for k in range(0, {NZ}, 3):
        for j in range(0, {NY}, 4):
            for i in range(0, {NX}, 4):
                d = abs(float(a[k, j, i]) - float(fld(i, j, k)))
                n_plt += 1
                if d > worst_plt:
                    worst_plt, where_plt = d, (nm, i, j, k)
print("::PLT_WORST", repr(worst_plt))
print("::PLT_N", n_plt)
print("::PLT_WHERE", where_plt)
print("::PLT_NAMES", sorted(pf.var_names) == sorted(mem))
""")
    assert r2.returncode == 0, (
        f"[{name}] the plotfile comparison failed:\n{r2.stdout[-2500:]}\n"
        f"{r2.stderr[-2000:]}")
    m2 = marks(r2.stdout)
    assert m2["PLT_NAMES"] == "True", (
        f"[{name}] fields() and the plotfile carry different field names")
    assert float(m2["PLT_WORST"]) == 0.0, (
        f"[{name}] fields() differs from the plotfile at {m2['PLT_WHERE']} "
        f"by {m2['PLT_WORST']}; both come from one gather, so this means a "
        f"third assembly has crept in")

    # --- against the ascii file, in this interpreter ---
    asc = read_ascii(os.path.join(WORKDIR, "fields_mem.txt"))
    assert len(asc) == N_FIELDS, (
        f"[{name}] the ascii file has {len(asc)} fields, expected "
        f"{N_FIELDS}")

    # The arrays live in the module's interpreter, so they come back
    # through the same marker channel as everything else. The sampling
    # stride is written into the snippet rather than %-formatted around
    # it -- the snippet contains its own % signs.
    idx = [(i, j, k) for k in range(0, NZ, 7) for j in range(0, NY, 8)
           for i in range(0, NX, 8)]
    r3 = run_py(f"""
import numpy as np
IDX = {idx!r}
mem = dict(np.load("py_fields.npz"))
for nm in sorted(mem):
    a = mem[nm]
    vals = ",".join(repr(float(a[k, j, i])) for (i, j, k) in IDX)
    print("::V_" + nm + " " + vals)
""")
    assert r3.returncode == 0, (
        f"[{name}] reading the arrays back failed:\n{r3.stdout[-2000:]}\n"
        f"{r3.stderr[-2000:]}")
    m3 = marks(r3.stdout)

    worst_asc, where_asc, n_asc = 0.0, None, 0
    for nm, table in asc.items():
        assert f"V_{nm}" in m3, (
            f"[{name}] the ascii file has a field {nm!r} that fields() "
            f"does not")
        vals = [float(v) for v in m3[f"V_{nm}"].split(",")]
        assert len(vals) == len(idx), (
            f"[{name}] sampling mismatch for {nm}: {len(vals)} values for "
            f"{len(idx)} points")
        for (i, j, k), v in zip(idx, vals):
            d = abs(v - table[(i, j, k)])
            n_asc += 1
            if d > worst_asc:
                worst_asc, where_asc = d, (nm, i, j, k)

    assert worst_asc == 0.0, (
        f"[{name}] fields() differs from the ascii file at {where_asc} by "
        f"{worst_asc}")

    print(f"[PASS] {name}  ({N_FIELDS} fields; {m2['PLT_N']} values match "
          f"the plotfile and {n_asc} match the ascii file, all exactly)")


# ---------------------------------------------------------------------------
# 2. the writers put files where they are told
# ---------------------------------------------------------------------------

def check_explicit_writers(exe):
    name = "explicit writers"

    r = run_py(PRELUDE + """
import os
with fwt.session():
    s = solved()
    s.write_report("named_report.txt")
    s.write_ascii("named_fields.txt")
    s.write_plotfile("named_plt")
    print("::REPORT", os.path.isfile("named_report.txt"))
    print("::ASCII", os.path.isfile("named_fields.txt"))
    print("::PLT", os.path.isdir("named_plt"))
    # The report carries the diagnostics, so it is a real report and not
    # an empty file with the right name.
    text = open("named_report.txt").read()
    print("::HASDIAG", "diag_div_max" in text)
    print("::HASZFACE", "z_face" in text)

    # Before diagnose() there is nothing to write, and saying so beats
    # writing a file with a missing component.
    t = fwt.Solver(case())
    t.setup()
    try:
        t.write_plotfile("too_early")
        print("::EARLY accepted")
    except RuntimeError:
        print("::EARLY raised")
    print("::NOFILE", not os.path.exists("too_early"))
""")
    assert r.returncode == 0, (
        f"[{name}] failed:\n{r.stdout[-2500:]}\n{r.stderr[-2000:]}")

    m = marks(r.stdout)
    for key in ("REPORT", "ASCII", "PLT", "HASDIAG", "HASZFACE"):
        assert m[key] == "True", (
            f"[{name}] {key} was {m[key]}")
    assert m["EARLY"] == "raised", (
        f"[{name}] writing before diagnose() must raise")
    assert m["NOFILE"] == "True", (
        f"[{name}] the rejected write still created a file")

    print(f"[PASS] {name}  (report, ascii and plotfile all written where "
          f"named; writing before diagnose() raises and creates nothing)")


# ---------------------------------------------------------------------------
# 3. the output section of the config
# ---------------------------------------------------------------------------

def check_output_config(exe):
    name = "output config section"

    r = run_py(PRELUDE + """
import os, shutil
for p in ("cfg_report.txt", "cfg_fields.txt"):
    if os.path.exists(p):
        os.remove(p)
shutil.rmtree("cfg_plt", ignore_errors=True)

with fwt.session():
    s = solved(output={"which": "both", "format": "ascii",
                       "report_file": "cfg_report.txt",
                       "ascii_file": "cfg_fields.txt",
                       "plot_file": "cfg_plt"})
    s.write_output()
    print("::REPORT", os.path.isfile("cfg_report.txt"))
    print("::ASCII", os.path.isfile("cfg_fields.txt"))
    # format = ascii, so no plotfile even though which = both.
    print("::NOPLT", not os.path.exists("cfg_plt"))

    def expect(label, fn):
        try:
            fn()
            print("::%s accepted" % label)
        except ValueError:
            print("::%s raised" % label)

    expect("BADWHICH", lambda: fwt.Solver(case(output={"which": "everything"})))
    expect("BADFORMAT", lambda: fwt.Solver(case(output={"format": "vtk"})))
    expect("TYPO", lambda: fwt.Solver(case(output={"report_fil": "x"})))
""")
    assert r.returncode == 0, (
        f"[{name}] failed:\n{r.stdout[-2500:]}\n{r.stderr[-2000:]}")

    m = marks(r.stdout)
    assert m["REPORT"] == "True" and m["ASCII"] == "True", (
        f"[{name}] the configured report/ascii files were not written")
    assert m["NOPLT"] == "True", (
        f"[{name}] format = ascii still wrote a plotfile")
    for label in ("BADWHICH", "BADFORMAT", "TYPO"):
        assert m[label] == "raised", (
            f"[{name}] {label} was {m[label]}, expected raised")

    print(f"[PASS] {name}  (which/format/filenames all honoured; bad values "
          f"and an unknown key raise at configuration time)")


# ---------------------------------------------------------------------------
# 4. no ParmParse leak into where the output goes
# ---------------------------------------------------------------------------

def check_no_parmparse_leak(exe):
    """Case 2 of a generation loop must not overwrite case 1's output
    because an inputs file left grid.plot_file pointing somewhere."""
    name = "no ParmParse leak"

    inputs = os.path.join(WORKDIR, "inputs_leak")
    with open(inputs, "w") as f:
        f.write("grid.n_cell = 8 8 8\n")
        f.write("grid.prob_lo = 0.0 0.0 0.0\n")
        f.write("grid.prob_hi = 100.0 100.0 32.0\n")
        f.write("grid.dz0 = 4.0\n")
        f.write("inflow.u_ref = 1.0\n")
        f.write("grid.output_format = both\n")
        f.write("grid.report_file = leaked_report.txt\n")
        f.write("grid.plot_file = leaked_plt\n")
        f.write("output.format = both\n")
        f.write("output.ascii_file = leaked_fields.txt\n")

    r = run_py(PRELUDE + f"""
import os
for p in ("leaked_report.txt", "leaked_fields.txt", "dict_report.txt"):
    if os.path.exists(p):
        os.remove(p)

fwt.initialize([{'"' + inputs + '"'}])
s = solved(output={{"which": "report", "report_file": "dict_report.txt"}})
s.write_output()
print("::DICT", os.path.isfile("dict_report.txt"))
print("::LEAK_REPORT", os.path.exists("leaked_report.txt"))
print("::LEAK_ASCII", os.path.exists("leaked_fields.txt"))
print("::LEAK_PLT", os.path.exists("leaked_plt"))
fwt.finalize()
""")
    assert r.returncode == 0, (
        f"[{name}] failed:\n{r.stdout[-2500:]}\n{r.stderr[-2000:]}")

    m = marks(r.stdout)
    assert m["DICT"] == "True", (
        f"[{name}] the dict's report file was not written")
    for key in ("LEAK_REPORT", "LEAK_ASCII", "LEAK_PLT"):
        assert m[key] == "False", (
            f"[{name}] {key}: the inputs file's output settings reached a "
            f"dict-configured run. In a generation loop that is case 2 "
            f"overwriting case 1.")

    print(f"[PASS] {name}  (wrote only the dict's report; the inputs "
          f"file's report, ascii and plotfile names were all ignored)")


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
    for check in (check_in_memory_matches_files, check_explicit_writers,
                  check_output_config, check_no_parmparse_leak):
        try:
            check(exe)
        except AssertionError as e:
            print(f"[FAIL] {check.__name__}: {e}")
            failed.append(check.__name__)
        except Exception as e:
            print(f"[ERROR] {check.__name__}: {e}")
            failed.append(check.__name__)

    if failed:
        print(f"\n{len(failed)} Phase 15 regtest case(s) failed: {failed}")
        return 1

    print("\nAll Phase 15 regtest cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
