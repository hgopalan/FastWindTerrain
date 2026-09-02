#!/usr/bin/env python3
"""
Phase 9 regtest checker -- Python bindings, and C++/Python parity.

Validates:

  lifecycle   -> the module imports, reports its version and the AMReX it
                 was built against, and initialize/finalize work. AMReX's
                 lifecycle is process-global, so double-init and
                 finalize-before-init must RAISE, not crash the
                 interpreter -- a segfault in a notebook loses the
                 session's work, and it is the failure mode this API
                 exists to prevent

  session     -> the context manager finalizes even when the block
                 raises. Leaving AMReX initialized would poison the rest
                 of the process

  parity      -> the same cases, run through the executable and through
                 the bindings, produce BYTE-IDENTICAL output. Not
                 "agree to a tolerance": identical. The module links the
                 same compiled library the executable does, so anything
                 less would mean something other than the solver differs

  argv        -> the shim is argv-compatible, including command-line
                 name=value overrides, which is what lets the regtest
                 driver run the whole suite through Python unchanged

The parity cases are drawn from the other regtest groups rather than
invented here, so this checks the paths those groups actually exercise:
a flat solve, a hill with anisotropy and O'Brien, the manufactured
solution, and the ascii field backend.

WHY BYTE COMPARISON IS LEGITIMATE HERE: an AMReX plotfile carries no
timestamp, host name or run id -- verified by this test passing on the
plotfile as well as the report.

If the bindings were not built (FWT_PYTHON=OFF, the default) this group
SKIPS rather than fails, since a C++-only build is a supported
configuration. CI builds one job with them on.

Usage:
    python3 check.py /path/to/fastwindterrain.exe [workdir]

Exits 0 if all cases pass, 1 otherwise (suitable for CI).
"""

import sys
import os
import filecmp
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
GROUP = os.path.basename(HERE)
REGTEST_ROOT = os.path.dirname(HERE)
ROOT = os.path.dirname(REGTEST_ROOT)

WORKDIR = os.path.join(ROOT, "build", "regtests", GROUP)

# Cases borrowed from the other groups: (inputs file, extra arguments).
# Between them they cover a flat solve, a hill with the full anisotropy
# and O'Brien path, the manufactured solution, and the ascii backend.
PARITY_CASES = [
    ("phase6_solve_correction/inputs_flat", []),
    ("phase6_solve_correction/inputs_bump",
     ["terrain.file={root}/regtests/phase6_solve_correction/terrain_hill.csv"]),
    ("phase7_anisotropy_obrien/inputs_slope",
     ["terrain.file={root}/regtests/phase7_anisotropy_obrien/terrain_slope.csv"]),
    ("phase5_poisson_assembly/inputs_mms", []),
    ("phase8_diagnostics_output/inputs_both",
     ["terrain.file={root}/regtests/phase8_diagnostics_output/terrain_hill.csv"]),
]


def binaries():
    """The C++ executable and the Python shim, located from the build
    directory rather than from argv -- this group needs BOTH, and must
    behave the same whichever one the driver was pointed at."""
    build = os.path.join(ROOT, "build")
    exe = os.path.join(build, "fastwindterrain")
    shim = os.path.join(build, "fastwindterrain-py")
    return exe, shim


def bindings_available():
    _, shim = binaries()
    pkg = os.path.join(ROOT, "build", "python", "fastwindterrain",
                       "__init__.py")
    return os.path.isfile(shim) and os.path.isfile(pkg)


def python_exe():
    """The interpreter the module was built for, read out of the shim.
    The `python3` on PATH is frequently a different minor version, and an
    extension module is built for exactly one."""
    _, shim = binaries()
    with open(shim) as f:
        for line in f:
            if line.startswith("exec "):
                return line.split('"')[1]
    raise RuntimeError(f"could not read the interpreter out of {shim}")


def run_py(code, env_extra=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = (os.path.join(ROOT, "build", "python")
                         + os.pathsep + env.get("PYTHONPATH", ""))
    if env_extra:
        env.update(env_extra)
    return subprocess.run([python_exe(), "-c", code], capture_output=True,
                          text=True, timeout=1800, env=env)


# ---------------------------------------------------------------------------
# 1. the module imports and reports itself
# ---------------------------------------------------------------------------

def check_import(exe):
    name = "module import"

    r = run_py(
        "import fastwindterrain as fwt\n"
        "print('VERSION', fwt.__version__)\n"
        "print('AMREX', fwt.amrex_version())\n"
        "print('INIT', fwt.is_initialized())\n")
    assert r.returncode == 0, (
        f"[{name}] importing the module failed:\n{r.stdout}\n{r.stderr}")

    out = dict(ln.split(None, 1) for ln in r.stdout.split("\n")
               if ln.startswith(("VERSION", "AMREX", "INIT")))
    assert out.get("VERSION"), f"[{name}] no __version__ reported"
    assert out.get("AMREX", "").startswith("2"), (
        f"[{name}] implausible AMReX version: {out.get('AMREX')!r}")
    assert out.get("INIT") == "False", (
        f"[{name}] AMReX must NOT be initialized as a side effect of "
        f"import; it is a process-global decision the caller owns")

    print(f"[PASS] {name}  (version {out['VERSION']}, built against AMReX "
          f"{out['AMREX']}, import does not initialize)")


# ---------------------------------------------------------------------------
# 2. the lifecycle raises instead of crashing
# ---------------------------------------------------------------------------

def check_lifecycle(exe):
    name = "AMReX lifecycle"

    r = run_py(
        "import fastwindterrain as fwt\n"
        "fwt.initialize([])\n"
        "assert fwt.is_initialized()\n"
        "try:\n"
        "    fwt.initialize([])\n"
        "    print('DOUBLE_INIT no-raise')\n"
        "except RuntimeError:\n"
        "    print('DOUBLE_INIT raised')\n"
        "fwt.finalize()\n"
        "assert not fwt.is_initialized()\n"
        "try:\n"
        "    fwt.finalize()\n"
        "    print('DOUBLE_FINI no-raise')\n"
        "except RuntimeError:\n"
        "    print('DOUBLE_FINI raised')\n"
        "fwt.initialize([])\n"
        "fwt.finalize()\n"
        "print('REINIT ok')\n")

    assert r.returncode == 0, (
        f"[{name}] the lifecycle test crashed (exit {r.returncode}) -- a "
        f"guard that segfaults instead of raising is the bug this API "
        f"exists to prevent\n{r.stdout}\n{r.stderr}")
    assert "DOUBLE_INIT raised" in r.stdout, (
        f"[{name}] a second initialize() must raise:\n{r.stdout}")
    assert "DOUBLE_FINI raised" in r.stdout, (
        f"[{name}] finalize() without initialize() must raise:\n{r.stdout}")
    assert "REINIT ok" in r.stdout, (
        f"[{name}] initialize/finalize must be repeatable in one process; "
        f"driving many cases depends on it\n{r.stdout}")

    print(f"[PASS] {name}  (double init and stray finalize both raise; "
          f"init/finalize repeats cleanly)")


def check_session_context(exe):
    name = "session context manager"

    r = run_py(
        "import fastwindterrain as fwt\n"
        "try:\n"
        "    with fwt.session([]):\n"
        "        assert fwt.is_initialized()\n"
        "        raise ValueError('boom')\n"
        "except ValueError:\n"
        "    pass\n"
        "print('AFTER', fwt.is_initialized())\n")

    assert r.returncode == 0, (
        f"[{name}] failed:\n{r.stdout}\n{r.stderr}")
    assert "AFTER False" in r.stdout, (
        f"[{name}] session() must finalize even when the block raises; "
        f"leaving AMReX initialized poisons the rest of the process\n"
        f"{r.stdout}")

    print(f"[PASS] {name}  (finalizes on the exception path)")


def check_run_owns_the_lifecycle(exe):
    """run() initializes and finalizes itself, so calling it inside an
    existing initialization has to be refused rather than silently
    inheriting the outer run's ParmParse state."""
    name = "run() lifecycle guard"

    r = run_py(
        "import fastwindterrain as fwt\n"
        "fwt.initialize([])\n"
        "try:\n"
        "    fwt.run(['nonexistent_inputs'])\n"
        "    print('GUARD no-raise')\n"
        "except RuntimeError:\n"
        "    print('GUARD raised')\n"
        "fwt.finalize()\n")

    assert r.returncode == 0, (
        f"[{name}] crashed:\n{r.stdout}\n{r.stderr}")
    assert "GUARD raised" in r.stdout, (
        f"[{name}] run() must refuse to run inside an existing AMReX "
        f"initialization -- it would inherit the outer ParmParse state\n"
        f"{r.stdout}")

    print(f"[PASS] {name}  (refuses to run inside an existing "
          f"initialization)")


# ---------------------------------------------------------------------------
# 3. byte-for-byte parity
# ---------------------------------------------------------------------------

def _tree(path):
    """Every file under path, relative, sorted."""
    out = []
    for root, _, files in os.walk(path):
        for f in files:
            full = os.path.join(root, f)
            out.append(os.path.relpath(full, path))
    return sorted(out)


def _run_case(binary, workdir, inputs, extra):
    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir)
    args = [binary, os.path.join(REGTEST_ROOT, inputs)]
    args += [a.format(root=ROOT) for a in extra]
    r = subprocess.run(args, cwd=workdir, capture_output=True, text=True,
                       timeout=3600)
    return r


def check_parity(exe):
    name = "C++ vs Python parity"

    cpp, shim = binaries()
    assert os.path.isfile(cpp), (
        f"[{name}] the C++ executable is missing: {cpp}")

    compared = 0
    for inputs, extra in PARITY_CASES:
        case = os.path.basename(inputs)
        d_cpp = os.path.join(WORKDIR, "parity", case, "cpp")
        d_py = os.path.join(WORKDIR, "parity", case, "py")

        r_cpp = _run_case(cpp, d_cpp, inputs, extra)
        r_py = _run_case(shim, d_py, inputs, extra)

        assert r_cpp.returncode == 0, (
            f"[{name}] {case}: the executable failed (exit "
            f"{r_cpp.returncode})\n{r_cpp.stdout[-2000:]}")
        assert r_py.returncode == r_cpp.returncode, (
            f"[{name}] {case}: the executable exited {r_cpp.returncode} "
            f"but Python exited {r_py.returncode}\n{r_py.stdout[-2000:]}\n"
            f"{r_py.stderr[-1000:]}")

        # Same files produced.
        t_cpp, t_py = _tree(d_cpp), _tree(d_py)
        assert t_cpp == t_py, (
            f"[{name}] {case}: the two runs produced different files\n"
            f"  only in C++:    {sorted(set(t_cpp) - set(t_py))}\n"
            f"  only in Python: {sorted(set(t_py) - set(t_cpp))}")
        assert t_cpp, f"[{name}] {case}: the run produced no output at all"

        # And identical bytes in each. Reports, plotfiles and the ascii
        # field dump alike -- a plotfile carries no timestamp, so it
        # compares byte for byte like everything else.
        for rel in t_cpp:
            a = os.path.join(d_cpp, rel)
            b = os.path.join(d_py, rel)
            assert filecmp.cmp(a, b, shallow=False), (
                f"[{name}] {case}: {rel} differs between the executable "
                f"and the bindings. They link the same compiled library, "
                f"so this means something other than the solver changed.")
            compared += 1

        # stdout too: the banner, the per-pass divergence, the extrema.
        assert r_cpp.stdout == r_py.stdout, (
            f"[{name}] {case}: stdout differs.\n"
            f"--- C++ ---\n{r_cpp.stdout[-1500:]}\n"
            f"--- Python ---\n{r_py.stdout[-1500:]}")

    shutil.rmtree(os.path.join(WORKDIR, "parity"), ignore_errors=True)
    print(f"[PASS] {name}  ({len(PARITY_CASES)} cases, {compared} output "
          f"files byte-identical, stdout identical)")


def check_argv_overrides(exe):
    """The shim has to be argv-compatible, not merely able to run an
    inputs file: the regtest driver passes name=value overrides on the
    command line, and several checkers depend on them."""
    name = "argv compatibility"

    cpp, shim = binaries()
    inputs = "phase6_solve_correction/inputs_flat"
    extra = ["poisson.alpha_v=0.25", "poisson.n_projections=2",
             "grid.report_file=grid_report_override.txt"]

    d_cpp = os.path.join(WORKDIR, "argv", "cpp")
    d_py = os.path.join(WORKDIR, "argv", "py")
    r_cpp = _run_case(cpp, d_cpp, inputs, extra)
    r_py = _run_case(shim, d_py, inputs, extra)

    assert r_cpp.returncode == 0 and r_py.returncode == 0, (
        f"[{name}] a run failed: C++ {r_cpp.returncode}, Python "
        f"{r_py.returncode}\n{r_py.stdout[-2000:]}\n{r_py.stderr[-1000:]}")

    report = "grid_report_override.txt"
    assert os.path.isfile(os.path.join(d_py, report)), (
        f"[{name}] the command-line report_file override did not take "
        f"effect under Python")

    # The override must have actually changed the answer, or this would
    # pass even if the arguments were being dropped on the floor.
    with open(os.path.join(d_py, report)) as f:
        text = f.read()
    assert "poisson_alpha_v 0.25" in text, (
        f"[{name}] poisson.alpha_v=0.25 did not reach the solver through "
        f"the shim")

    assert filecmp.cmp(os.path.join(d_cpp, report),
                       os.path.join(d_py, report), shallow=False), (
        f"[{name}] the overridden run differs between the two paths")

    shutil.rmtree(os.path.join(WORKDIR, "argv"), ignore_errors=True)
    print(f"[PASS] {name}  (name=value overrides reach the solver and give "
          f"an identical report)")


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
    for check in (check_import, check_lifecycle, check_session_context,
                  check_run_owns_the_lifecycle, check_parity,
                  check_argv_overrides):
        try:
            check(exe)
        except AssertionError as e:
            print(f"[FAIL] {check.__name__}: {e}")
            failed.append(check.__name__)
        except Exception as e:
            print(f"[ERROR] {check.__name__}: {e}")
            failed.append(check.__name__)

    if failed:
        print(f"\n{len(failed)} Phase 9 regtest case(s) failed: {failed}")
        return 1

    print("\nAll Phase 9 regtest cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
