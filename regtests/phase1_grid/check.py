#!/usr/bin/env python3
"""
Phase 1 regtest checker -- grid & data layout scaffolding.

Runs the FastWindTerrain executable against each of the four Phase 1
input files and validates:

  inputs_nominal    -> exact match, no warning, report matches analytic
                       geometric-stretching formula
  inputs_uniform    -> stretching_ratio=1.0 reproduces a plain uniform
                       grid exactly (regression safety net)
  inputs_overshoot  -> non-fatal WARNING, run succeeds, prob_hi[2] in
                       the report is overridden to H_computed
  inputs_undershoot -> fatal abort, nonzero exit code, no report file

Usage:
    python3 check.py /path/to/fastwindterrain.exe

Exits 0 if all cases pass, 1 otherwise (suitable for CI).
"""

import sys
import os
import subprocess
import math

TOL = 1.0e-6  # absolute tolerance in meters for float comparisons

HERE = os.path.dirname(os.path.abspath(__file__))


def analytic_H(dz0, r, nz):
    if abs(r - 1.0) < 1e-14:
        return dz0 * nz
    return dz0 * (r**nz - 1.0) / (r - 1.0)


def parse_report(path):
    """Parse the plain-text grid report into a dict of scalars plus
    z_face / z_cc arrays."""
    data = {"z_face": {}, "z_cc": {}}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            key = parts[0]
            if key in ("z_face", "z_cc"):
                idx = int(parts[1])
                val = float(parts[2])
                data[key][idx] = val
            elif key in ("n_cell",):
                data[key] = [int(x) for x in parts[1:]]
            elif key in ("prob_lo", "prob_hi"):
                data[key] = [float(x) for x in parts[1:]]
            else:
                # scalar float fields: dz0, stretching_ratio, dx, dy, n_boxes
                try:
                    data[key] = float(parts[1])
                except ValueError:
                    data[key] = parts[1]
    return data


def run_case(exe, inputs_file):
    cwd = HERE
    result = subprocess.run(
        [exe, os.path.join(HERE, inputs_file)],
        cwd=cwd, capture_output=True, text=True, timeout=120,
    )
    return result


def check_nominal(exe):
    name = "inputs_nominal"
    result = run_case(exe, name)
    assert result.returncode == 0, (
        f"[{name}] expected success (exit 0), got {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    assert "WARNING" not in result.stdout, (
        f"[{name}] expected no overshoot warning, got:\n{result.stdout}")

    report = parse_report(os.path.join(HERE, "grid_report_nominal.txt"))

    dz0, r, nz = 2.0, 1.05, report["n_cell"][2]
    H_expected = analytic_H(dz0, r, nz)

    assert abs(report["prob_hi"][2] - H_expected) < TOL, (
        f"[{name}] prob_hi[2] = {report['prob_hi'][2]} != "
        f"analytic H = {H_expected}")

    # z_face[0] must be prob_lo[2]; z_face[nz] must equal H (from prob_lo=0)
    assert abs(report["z_face"][0] - report["prob_lo"][2]) < TOL
    assert abs(report["z_face"][nz] - report["prob_hi"][2]) < TOL

    # spot-check the geometric stretching law dz(k) = dz0 * r^k
    for k in [0, 1, 10, nz - 1]:
        dz_k_expected = dz0 * (r ** k)
        dz_k_actual = report["z_face"][k+1] - report["z_face"][k]
        assert abs(dz_k_actual - dz_k_expected) < TOL, (
            f"[{name}] dz[{k}] = {dz_k_actual} != expected {dz_k_expected}")

    print(f"[PASS] {name}")


def check_uniform(exe):
    name = "inputs_uniform"
    result = run_case(exe, name)
    assert result.returncode == 0, (
        f"[{name}] expected success (exit 0), got {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    assert "WARNING" not in result.stdout

    report = parse_report(os.path.join(HERE, "grid_report_uniform.txt"))
    dz0, nz = 25.0, report["n_cell"][2]

    # every dz(k) must equal dz0 exactly (stretching_ratio = 1.0)
    for k in range(nz):
        dz_k = report["z_face"][k+1] - report["z_face"][k]
        assert abs(dz_k - dz0) < TOL, (
            f"[{name}] uniform grid dz[{k}] = {dz_k} != dz0 = {dz0}")

    assert abs(report["prob_hi"][2] - dz0 * nz) < TOL

    print(f"[PASS] {name}")


def check_overshoot(exe):
    name = "inputs_overshoot"
    result = run_case(exe, name)
    assert result.returncode == 0, (
        f"[{name}] expected success (exit 0) despite overshoot, "
        f"got {result.returncode}\nstdout:\n{result.stdout}")
    assert "WARNING" in result.stdout, (
        f"[{name}] expected overshoot WARNING in stdout, got:\n{result.stdout}")

    report = parse_report(os.path.join(HERE, "grid_report_overshoot.txt"))
    dz0, r, nz = 2.0, 1.05, report["n_cell"][2]
    H_expected = analytic_H(dz0, r, nz)

    # prob_hi[2] in the report must be the OVERRIDDEN value (H_computed),
    # not the originally-requested 900.0
    assert abs(report["prob_hi"][2] - H_expected) < TOL, (
        f"[{name}] prob_hi[2] = {report['prob_hi'][2]} was not overridden "
        f"to H_computed = {H_expected}")
    assert report["prob_hi"][2] > 900.0 + TOL, (
        f"[{name}] prob_hi[2] should exceed the originally-requested 900 m")

    print(f"[PASS] {name}")


def check_undershoot(exe):
    name = "inputs_undershoot"
    report_path = os.path.join(HERE, "grid_report_undershoot.txt")
    if os.path.exists(report_path):
        os.remove(report_path)  # ensure stale file from a prior run can't fake a pass

    result = run_case(exe, name)
    assert result.returncode != 0, (
        f"[{name}] expected FATAL abort (nonzero exit), got 0.\n"
        f"stdout:\n{result.stdout}")
    assert not os.path.exists(report_path), (
        f"[{name}] expected no report file to be written on fatal abort")

    print(f"[PASS] {name}")


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} /path/to/fastwindterrain.exe")
        return 1

    exe = os.path.abspath(sys.argv[1])
    if not os.path.isfile(exe):
        print(f"executable not found: {exe}")
        return 1

    checks = [check_nominal, check_uniform, check_overshoot, check_undershoot]
    failed = []
    for check in checks:
        try:
            check(exe)
        except AssertionError as e:
            print(f"[FAIL] {check.__name__}: {e}")
            failed.append(check.__name__)
        except Exception as e:
            print(f"[ERROR] {check.__name__}: {e}")
            failed.append(check.__name__)

    if failed:
        print(f"\n{len(failed)} Phase 1 regtest case(s) failed: {failed}")
        return 1

    print("\nAll Phase 1 regtest cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
