#!/usr/bin/env python3
"""
Master input-file regtest checker.

regtests/inputs_master documents every input the solver reads. A
reference like that is worth exactly as much as its accuracy, and the
usual failure mode is silent: someone adds a ParmParse query, the
reference does not mention it, and a year later the file is quietly
wrong in a way nobody notices until a user trusts it.

So this checker does not read the reference and nod. It:

  1. greps every ParmParse prefix and key out of Source/, reconstructs
     the full input names, and asserts every one of them appears in
     inputs_master -- commented out is fine, absent is not

  2. asserts the reverse: every input named in inputs_master is one the
     code actually reads, so a removed or renamed input cannot leave a
     stale entry behind

  3. RUNS inputs_master, so the reference is a working case and not just
     a plausible-looking one, and checks it produced the report and the
     plotfile it asks for

  4. runs it again with fwt.debug = 1, which echoes every input the run
     resolved, and confirms the report is byte-identical -- documenting
     an input must not change what it does

Usage:
    python3 check.py /path/to/fastwindterrain.exe [workdir]

Exits 0 if all cases pass, 1 otherwise (suitable for CI).
"""

import sys
import os
import re
import glob
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
GROUP = os.path.basename(HERE)
REGTEST_ROOT = os.path.dirname(HERE)
ROOT = os.path.dirname(REGTEST_ROOT)
SOURCE = os.path.join(ROOT, "Source")
MASTER = os.path.join(REGTEST_ROOT, "inputs_master")

WORKDIR = os.path.join(ROOT, "build", "regtests", GROUP)

# ParmParse objects are constructed with a prefix and then queried by
# key; the two are matched up per source file, per variable name.
_PP_CTOR = re.compile(r'ParmParse\s+(\w+)\s*\(\s*"([A-Za-z_]+)"\s*\)')
_PP_QUERY = re.compile(r'\b(\w+)\.(?:query|get|queryarr|getarr)\s*\(\s*"'
                       r'([A-Za-z0-9_]+)"')

# Names that appear in inputs_master but are not ParmParse keys of ours.
# Empty on purpose -- kept so that adding one is a deliberate act with a
# reason next to it, rather than a quiet loosening of the check.
ALLOWED_EXTRA = set()


def source_files():
    return sorted(glob.glob(os.path.join(SOURCE, "*.cpp"))
                  + glob.glob(os.path.join(SOURCE, "*.H")))


def inputs_from_source():
    """Every "<prefix>.<key>" the code reads, with the file it came
    from."""
    found = {}
    for path in source_files():
        with open(path) as f:
            text = f.read()
        # A ParmParse variable can be reused or shadowed across scopes in
        # one file; the last construction before a query is the one in
        # force, so the file is walked in order rather than pre-scanned.
        prefix_of = {}
        for line in text.split("\n"):
            m = _PP_CTOR.search(line)
            if m:
                prefix_of[m.group(1)] = m.group(2)
            for var, key in _PP_QUERY.findall(line):
                if var in prefix_of:
                    found.setdefault(f"{prefix_of[var]}.{key}",
                                     os.path.basename(path))
    return found


def inputs_from_master():
    """Every "<prefix>.<key>" named in the master file, whether it is
    active or commented out. A commented entry still documents the
    input, which is the point of the file."""
    named = {}
    with open(MASTER) as f:
        for n, line in enumerate(f, 1):
            body = line.lstrip("#").strip()
            m = re.match(r'^([a-z_]+\.[A-Za-z0-9_]+)\s*=', body)
            if m:
                named.setdefault(m.group(1), n)
    return named


def check_master_covers_the_code(exe):
    name = "inputs_master (coverage)"

    src = inputs_from_source()
    assert len(src) > 30, (
        f"[{name}] only {len(src)} inputs were found in Source/ -- the "
        f"grep is broken, not the reference")

    master = inputs_from_master()
    missing = sorted(k for k in src if k not in master)
    assert not missing, (
        f"[{name}] {len(missing)} input(s) the code reads are not "
        f"documented in regtests/inputs_master:\n" +
        "\n".join(f"    {k}   (read in {src[k]})" for k in missing))

    print(f"[PASS] {name}  ({len(src)} inputs read across "
          f"{len(set(src.values()))} source files, all documented)")


def check_master_has_nothing_stale(exe):
    name = "inputs_master (no stale entries)"

    src = inputs_from_source()
    master = inputs_from_master()
    stale = sorted(k for k in master
                   if k not in src and k not in ALLOWED_EXTRA)
    assert not stale, (
        f"[{name}] {len(stale)} entry(ies) in regtests/inputs_master name "
        f"an input the code no longer reads:\n" +
        "\n".join(f"    {k}   (line {master[k]})" for k in stale))

    print(f"[PASS] {name}  ({len(master)} documented inputs, all still "
          f"read by the code)")


def check_master_runs(exe):
    """The reference has to be a working input file, not a plausible
    one."""
    name = "inputs_master (runs)"

    for stale in ("grid_report_master.txt", "plt_master"):
        p = os.path.join(WORKDIR, stale)
        shutil.rmtree(p, ignore_errors=True)
        if os.path.isfile(p):
            os.remove(p)

    result = subprocess.run([exe, MASTER], cwd=WORKDIR, capture_output=True,
                            text=True, timeout=3600)
    assert result.returncode == 0, (
        f"[{name}] the master input file does not run (exit "
        f"{result.returncode})\nstdout:\n{result.stdout[-3000:]}\n"
        f"stderr:\n{result.stderr[-2000:]}")

    report = os.path.join(WORKDIR, "grid_report_master.txt")
    assert os.path.isfile(report), (
        f"[{name}] grid.output_format = both wrote no report")
    assert os.path.isdir(os.path.join(WORKDIR, "plt_master")), (
        f"[{name}] grid.output_format = both wrote no plotfile")
    assert not os.path.exists(os.path.join(WORKDIR, "fields.txt")), (
        f"[{name}] output.format = plt should not have written the ascii "
        f"file")

    # AMReX lists unused inputs at Finalize. A typo in the reference, or
    # an input that is only read in some other mode, shows up there
    # rather than as a failure -- so it is made one.
    lines = result.stdout.split("\n")
    unused = []
    for n, line in enumerate(lines):
        if "Unused ParmParse Variables" not in line:
            continue
        for follow in lines[n + 1:]:
            if not follow.startswith("  "):
                break
            unused.append(follow.strip())
    assert not unused, (
        f"[{name}] the master file sets {len(unused)} input(s) this run "
        f"never read. An input that only applies in another mode belongs "
        f"in the file commented out, so it is still documented without "
        f"being set:\n" + "\n".join(f"    {u}" for u in unused))

    print(f"[PASS] {name}  (runs clean, wrote the report and the plotfile "
          f"it asks for, no unused inputs)")


def check_documenting_changes_nothing(exe):
    """fwt.debug echoes every resolved input, including the defaults that
    were never given. Turning it on must not change the answer."""
    name = "inputs_master (debug is inert)"

    base = os.path.join(WORKDIR, "grid_report_master.txt")
    with open(base) as f:
        before = f.read()

    result = subprocess.run(
        [exe, MASTER, "fwt.debug=1",
         "grid.report_file=grid_report_master_debug.txt",
         "grid.output_format=report"],
        cwd=WORKDIR, capture_output=True, text=True, timeout=3600)
    assert result.returncode == 0, (
        f"[{name}] the debug run failed (exit {result.returncode})\n"
        f"{result.stdout[-2000:]}")
    assert "[debug]" in result.stdout, (
        f"[{name}] fwt.debug = 1 printed no debug output")

    with open(os.path.join(WORKDIR,
                           "grid_report_master_debug.txt")) as f:
        after = f.read()

    # The report_file line is the only difference the override introduces.
    assert before == after, (
        f"[{name}] fwt.debug = 1 changed the report; documenting a run "
        f"must not change it")

    n_debug = sum(1 for ln in result.stdout.split("\n") if "[debug]" in ln)
    print(f"[PASS] {name}  ({n_debug} debug lines, report byte-identical)")


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
    for check in (check_master_covers_the_code, check_master_has_nothing_stale,
                  check_master_runs, check_documenting_changes_nothing):
        try:
            check(exe)
        except AssertionError as e:
            print(f"[FAIL] {check.__name__}: {e}")
            failed.append(check.__name__)
        except Exception as e:
            print(f"[ERROR] {check.__name__}: {e}")
            failed.append(check.__name__)

    if failed:
        print(f"\n{len(failed)} master-inputs regtest case(s) failed: "
              f"{failed}")
        return 1

    print("\nAll master-inputs regtest cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
