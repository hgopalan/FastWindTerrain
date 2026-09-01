#!/usr/bin/env python3
"""
FastWindTerrain regtest runner.

Loops over regtests/*/check.py and invokes each against the built
executable. Each folder is self-contained (inputs + check.py, and any
auxiliary data like terrain.csv or user_profile.txt); a folder counts
as a test group as soon as it holds a check.py.

Each check.py runs its cases in a scratch work directory
(<repo>/build/regtests/<phase> by default), so running the tests leaves
no artifacts in the source tree.

Usage:
    python3 run_regtests.py /path/to/fastwindterrain.exe [phase_name ...]
                            [--workdir DIR]

With no phase_name arguments, every regtests/*/check.py is run.
--workdir DIR overrides the scratch root; each phase gets DIR/<phase>.
"""

import sys
import os
import glob
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
REGTEST_ROOT = os.path.join(ROOT, "regtests")


def discover_phase_dirs():
    """Every regtest directory holding a check.py. Group names are not
    required to follow any pattern -- the checker file is what makes a
    directory a test group."""
    return sorted(d for d in glob.glob(os.path.join(REGTEST_ROOT, "*"))
                  if os.path.isfile(os.path.join(d, "check.py")))


def main():
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} /path/to/fastwindterrain.exe [phase_name ...]")
        return 1

    exe = os.path.abspath(sys.argv[1])
    if not os.path.isfile(exe):
        print(f"executable not found: {exe}")
        return 1

    argv = sys.argv[2:]
    workdir_root = os.path.join(ROOT, "build", "regtests")
    if "--workdir" in argv:
        i = argv.index("--workdir")
        if i + 1 >= len(argv):
            print("--workdir requires a directory argument")
            return 1
        workdir_root = os.path.abspath(argv[i + 1])
        del argv[i:i + 2]

    requested = set(argv)
    phase_dirs = discover_phase_dirs()
    if requested:
        phase_dirs = [d for d in phase_dirs if os.path.basename(d) in requested]

    if not phase_dirs:
        print("No matching regtest phase directories found.")
        return 1

    failed = []
    for d in phase_dirs:
        check_py = os.path.join(d, "check.py")
        if not os.path.isfile(check_py):
            print(f"[SKIP] {os.path.basename(d)} (no check.py yet)")
            continue
        print(f"--- Running regtests for {os.path.basename(d)} ---")
        workdir = os.path.join(workdir_root, os.path.basename(d))
        os.makedirs(workdir, exist_ok=True)
        result = subprocess.run([sys.executable, check_py, exe, workdir])
        if result.returncode != 0:
            failed.append(os.path.basename(d))

    print("\n=== Regtest summary ===")
    if failed:
        print(f"FAILED: {failed}")
        return 1
    print("All phase regtests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
