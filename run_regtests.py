#!/usr/bin/env python3
"""
FastWindTerrain regtest runner.

Loops over regtests/phase*_*/check.py and invokes each against the
built executable. Add new phase folders here as they land; each
folder is self-contained (inputs + check.py, and any auxiliary data
like terrain.csv or user_profile.txt).

Usage:
    python3 run_regtests.py /path/to/fastwindterrain.exe [phase_name ...]

With no phase_name arguments, all regtests/phase*/check.py are run.
"""

import sys
import os
import glob
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
REGTEST_ROOT = os.path.join(ROOT, "regtests")


def discover_phase_dirs():
    return sorted(glob.glob(os.path.join(REGTEST_ROOT, "phase*")))


def main():
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} /path/to/fastwindterrain.exe [phase_name ...]")
        return 1

    exe = os.path.abspath(sys.argv[1])
    if not os.path.isfile(exe):
        print(f"executable not found: {exe}")
        return 1

    requested = set(sys.argv[2:])
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
        result = subprocess.run([sys.executable, check_py, exe])
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
