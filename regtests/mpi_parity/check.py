#!/usr/bin/env python3
"""
MPI regtest -- the solver must give the same answer on any number of
ranks, and it must come back.

Two properties are checked, and the second one is why this group exists.

**Parity.** The domain decomposition is a distribution detail, not
physics. A case run on two ranks must reproduce the one-rank report to
round-off: the same divergence norms, the same velocity extrema, the
same lambda, the same boundary fluxes. Not byte-identical -- a global
sum lands in a different order on a different decomposition, and that
difference is real and irreducible -- but identical to a tolerance far
tighter than anything a genuine parallel bug would fit inside.

**Termination.** A collective call reached by only some ranks does not
produce a wrong answer; it produces no answer at all. The ranks that
skipped it walk on, the ranks that entered it wait forever, and the job
sits there until something kills it.

That is not hypothetical. It is the bug this group was written for:
``Poisson::AppendReport`` returned early on non-IO ranks and then called
``amrex::MultiFab::min`` and ``max``, which are collective -- each one
ends in an all-reduce over every rank. Rank 0 entered the reduction, the
other ranks had already returned, and the run hung after a completely
correct solve. Every number was right. None of them were ever written.

No amount of checking numbers catches that, because the failure is the
absence of numbers. So every run here is under a WALL-CLOCK TIMEOUT, and
expiry is reported as its own kind of failure, with the pattern to go
looking for: a rank-conditional early return followed by a collective.
The collectives to watch are the ones that do not look like collectives
-- ``MultiFab::min``/``max``/``norm0``/``norm2``/``sum``, and anything
built on ``ParallelDescriptor::Reduce*`` or ``ParallelAllReduce``. Take
them BEFORE the IO rank is singled out, then write the results.

The cases are borrowed from the other regtest groups rather than
invented here, so this covers paths those groups actually exercise: a
flat solve, a hill with a real projection to do, and the anisotropy and
O'Brien path. Borrowing also means no terrain data is duplicated, and
that a change to one of those cases is carried over here automatically.

If the executable was not built with MPI (``FWT_MPI=OFF``, the default)
or ``mpirun`` is not on PATH, this group SKIPS rather than fails: a
serial build is a supported configuration, and is what most developers
have. Build with ``-DFWT_MPI=ON`` to run it.

Usage:
    python3 check.py /path/to/fastwindterrain.exe [workdir]

Exits 0 if all cases pass, 1 otherwise (suitable for CI).
"""

import sys
import os
import time
import shutil
import signal
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
GROUP = os.path.basename(HERE)
REGTEST_ROOT = os.path.dirname(HERE)
ROOT = os.path.dirname(REGTEST_ROOT)

WORKDIR = os.path.join(ROOT, "build", "regtests", GROUP)

# The rank count to compare against serial. Two is enough to expose a
# rank-conditional collective, and is what a hosted CI runner can be
# relied on to have cores for. More ranks are worth trying by hand:
#     mpirun -n 4 build-mpi/fastwindterrain <inputs>
NRANKS = 2

# (inputs file relative to regtests/, extra arguments). Terrain paths are
# made absolute because every case runs in the scratch work directory.
CASES = [
    ("phase6_solve_correction/inputs_flat", []),
    ("phase6_solve_correction/inputs_bump",
     ["terrain.file={root}/regtests/phase6_solve_correction/terrain_hill.csv"]),
    ("phase7_anisotropy_obrien/inputs_slope",
     ["terrain.file={root}/regtests/phase7_anisotropy_obrien/terrain_slope.csv"]),
]

# A deadlock waits forever, so this only decides how long we wait before
# calling it one. It is derived from the serial run each case just did,
# rather than fixed, so a slow or loaded runner does not turn into a
# false positive: 20x the measured serial time, never under two minutes.
# A healthy parallel run takes about as long as the serial one, so 20x is
# already enormous margin -- and the ceiling is there so that a case that
# really has deadlocked costs five minutes rather than the twenty that
# scaling off a contended serial run can otherwise ask for.
TIMEOUT_FACTOR = 20.0
TIMEOUT_FLOOR = 120.0
TIMEOUT_CEILING = 300.0

# Relative tolerance for the parity comparison. Reduction order is the
# only thing that differs between decompositions, so the differences are
# a few ulp; 1e-9 leaves six orders of magnitude of headroom over what
# is observed and still shuts out any real discrepancy.
RTOL = 1.0e-9

# Entries whose VALUE IS round-off -- the residue of a near-exact
# cancellation, or a solver residual driven to machine zero. Their
# relative difference between two decompositions is O(1) however right
# they both are, so comparing them relatively would fail on a correct
# run. Each is compared on an absolute scale set by the quantity it is
# the residue of, and the ones that carry a physical bound are checked
# against that bound separately, in check_solve_actually_converged.
ABS_TOL = {
    "poisson_solve_residual":   1.0e-9,     # driven to ~1e-14
    "obrien_max_residual":      1.0e-9,     # ditto
    "inflow_flux_net":          1.0e-3,     # against a ~1e7 m^3/s throughflow
    "inflow_flux_imbalance":    1.0e-10,    # that net, normalised
    # The same two before inflow.balance_flux redistributed anything.
    # With the option off they ARE the same two numbers, so they are the
    # same near-cancellation and need the same absolute scale.
    "inflow_flux_net_raw":      1.0e-3,
    "inflow_flux_imbalance_raw": 1.0e-10,
    "diag_flux_net":            1.0e-3,
    "diag_flux_imbalance":      1.0e-10,
    "diag_flux_top":            1.0e-4,     # a closed lid: 0 in exact arithmetic
}

# The box count is a decomposition detail, not a result: AMReX may chop
# the domain differently for a different rank count. That the answer does
# NOT depend on how it chopped is the entire point of this group.
IGNORED_KEYS = {"n_boxes"}

MPI_MARKER = "MPI initialized with"


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def run_with_timeout(cmd, cwd, timeout):
    """Run cmd, returning (returncode, output, timed_out).

    The child gets its own process group so that a deadlocked run -- the
    failure this group exists to catch -- can be killed along with every
    rank it spawned. Killing mpirun alone would leave the ranks spinning
    in an MPI barrier for the rest of the job.
    """
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True,
                         start_new_session=True)
    try:
        out, _ = p.communicate(timeout=timeout)
        return p.returncode, out, False
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        out, _ = p.communicate()
        return None, out or "", True


def fresh_dir(name):
    """An empty scratch directory. Emptied rather than reused, because
    the solver's report writers open the file in APPEND mode: a leftover
    report from an earlier run would be silently doubled, and every key
    would still parse."""
    d = os.path.join(WORKDIR, name)
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)
    return d


def case_dir(case, tag):
    """A scratch directory per (case, rank count), named after the full
    case path so two groups' identically named inputs cannot collide."""
    return fresh_dir(case.replace("/", "_").replace("inputs_", "") + f"_{tag}")


def case_argv(exe, case, extra):
    return ([exe, os.path.join(REGTEST_ROOT, case)]
            + [a.format(root=ROOT) for a in extra]
            # Overridden so the checker does not have to know what each
            # borrowed case happens to call its report.
            + ["grid.report_file=report.txt"])


def run_serial(exe, case, extra):
    """The baseline: one process, no mpirun. Returns (report path,
    stdout, wall seconds)."""
    d = case_dir(case, "serial")
    t0 = time.time()
    rc, out, timed_out = run_with_timeout(case_argv(exe, case, extra), d,
                                          TIMEOUT_FLOOR * 4)
    dt = time.time() - t0
    assert not timed_out, (
        f"the SERIAL run of {case} did not finish within "
        f"{TIMEOUT_FLOOR * 4:.0f} s -- this group cannot say anything about "
        f"MPI until the one-rank case works\noutput:\n{out[-3000:]}")
    assert rc == 0, (
        f"serial run of {case} exited {rc}\noutput:\n{out[-3000:]}")
    return os.path.join(d, "report.txt"), out, dt


def run_parallel(exe, case, extra, nranks, timeout):
    """The same case under mpirun. Returns (report path, stdout)."""
    d = case_dir(case, f"n{nranks}")
    cmd = ["mpirun", "-n", str(nranks)] + case_argv(exe, case, extra)
    rc, out, timed_out = run_with_timeout(cmd, d, timeout)

    assert not timed_out, (
        f"{case} on {nranks} ranks did not finish within {timeout:.0f} s.\n"
        f"The serial run of the same case finished. A hang that appears "
        f"only on more than one rank is almost always a COLLECTIVE CALL "
        f"REACHED BY SOME RANKS AND NOT OTHERS -- look for an early "
        f"return guarded by ParallelDescriptor::IOProcessor() with a "
        f"MultiFab min/max/norm/sum, or any ParallelDescriptor::Reduce*, "
        f"after it. Hoist the collective above the guard.\n"
        f"Last output before the hang:\n{out[-3000:]}")
    assert rc == 0, (
        f"{case} on {nranks} ranks exited {rc}\noutput:\n{out[-3000:]}")

    ranks_seen = None
    for line in out.splitlines():
        if MPI_MARKER in line and "processes" in line:
            ranks_seen = int(line.split(MPI_MARKER)[1].split()[0])
            break
    assert ranks_seen == nranks, (
        f"asked for {nranks} ranks but AMReX reported {ranks_seen}. "
        f"mpirun is not launching what this test thinks it is, so a pass "
        f"here would mean nothing.")

    return os.path.join(d, "report.txt"), out


# ---------------------------------------------------------------------------
# Comparing
# ---------------------------------------------------------------------------

def parse_report(path):
    """key -> value, floats where they parse and strings otherwise."""
    assert os.path.isfile(path), (
        f"no report at {path}. The run exited 0 but wrote nothing, which "
        f"is what a report writer that aborts partway through looks like.")
    data = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            key, raw = parts[0], parts[1].strip()
            try:
                data[key] = float(raw)
            except ValueError:
                data[key] = raw
    return data


def differs(key, a, b):
    """None if the two values agree, else a message saying how they do
    not."""
    if isinstance(a, str) or isinstance(b, str):
        return None if a == b else f"{a!r} != {b!r}"
    d = abs(a - b)
    if d <= ABS_TOL.get(key, 0.0):
        return None
    if d <= RTOL * max(abs(a), abs(b)):
        return None
    scale = max(abs(a), abs(b))
    rel = (d / scale) if scale > 0.0 else float("inf")
    return f"{a!r} vs {b!r}  (abs {d:.3e}, rel {rel:.3e})"


def compare_reports(case, serial_path, parallel_path, nranks):
    s = parse_report(serial_path)
    p = parse_report(parallel_path)

    missing = sorted(set(s) - set(p))
    extra = sorted(set(p) - set(s))
    assert not missing, (
        f"[{case}] the {nranks}-rank report is missing {len(missing)} "
        f"entries the serial one has: {missing[:8]}\n"
        f"A report that stops partway through is the signature of a "
        f"writer that died or was cut short on the IO rank.")
    assert not extra, (
        f"[{case}] the {nranks}-rank report has entries the serial one "
        f"does not: {extra[:8]}")

    bad = []
    for key in sorted(s):
        if key in IGNORED_KEYS:
            continue
        why = differs(key, s[key], p[key])
        if why is not None:
            bad.append(f"  {key}: {why}")
    assert not bad, (
        f"[{case}] {len(bad)} of {len(s)} report entries differ between 1 "
        f"and {nranks} ranks. The decomposition is not part of the "
        f"physics, so these should agree to round-off:\n"
        + "\n".join(bad[:20]))

    return len(s)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def check_solve_actually_converged(case, report):
    """Parity with a broken serial run would be parity with nothing. The
    quantities ABS_TOL compares loosely are the ones that carry a real
    bound, so they are held to that bound here instead."""
    d = parse_report(report)
    r = d.get("poisson_solve_residual")
    assert r is not None and r < 1.0e-8, (
        f"[{case}] poisson_solve_residual = {r}; the projection did not "
        f"converge, so agreement across ranks would not mean anything")
    imb = d.get("diag_flux_imbalance")
    assert imb is not None and abs(imb) < 1.0e-10, (
        f"[{case}] diag_flux_imbalance = {imb}; the corrected field does "
        f"not conserve mass")


def check_case(exe, case, extra):
    serial_report, _, dt = run_serial(exe, case, extra)
    check_solve_actually_converged(case, serial_report)

    timeout = min(TIMEOUT_CEILING, max(TIMEOUT_FLOOR, TIMEOUT_FACTOR * dt))
    par_report, _ = run_parallel(exe, case, extra, NRANKS, timeout)

    n = compare_reports(case, serial_report, par_report, NRANKS)
    print(f"[PASS] {case}  (finished on {NRANKS} ranks in under "
          f"{timeout:.0f} s; all {n} report entries match the serial run "
          f"to within {RTOL:g} relative)")


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def mpi_build(exe):
    """True if the executable was built with MPI.

    Asked by running it, because that is the only thing that actually
    knows: AMReX announces the communicator size at startup, and a
    serial build says nothing. The probe is a real (short) run rather
    than a bare invocation, so it neither aborts nor drops a backtrace
    file in the work directory.
    """
    d = fresh_dir("probe")
    argv = case_argv(exe, "phase6_solve_correction/inputs_flat", [])
    rc, out, timed_out = run_with_timeout(argv, d, TIMEOUT_FLOOR * 4)
    if timed_out or rc != 0:
        print(f"[ERROR] the probe run failed (exit {rc}, timed out "
              f"{timed_out}); this group cannot tell whether the build "
              f"has MPI\noutput:\n{out[-2000:]}")
        return None
    return MPI_MARKER in out


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

    has_mpi = mpi_build(exe)
    if has_mpi is None:
        return 1
    if not has_mpi:
        print("[SKIP] this executable was not built with MPI. Configure "
              "with -DFWT_MPI=ON to run this group.")
        return 0
    if shutil.which("mpirun") is None:
        print("[SKIP] the executable has MPI but mpirun is not on PATH, "
              "so the parallel runs cannot be launched.")
        return 0

    failed = []
    for case, extra in CASES:
        try:
            check_case(exe, case, extra)
        except AssertionError as e:
            print(f"[FAIL] {case}: {e}")
            failed.append(case)
        except Exception as e:
            print(f"[ERROR] {case}: {e}")
            failed.append(case)

    if failed:
        print(f"\n{len(failed)} MPI regtest case(s) failed: {failed}")
        return 1

    print(f"\nAll MPI regtest cases passed ({NRANKS} ranks).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
