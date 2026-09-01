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
  inputs_plt        -> grid.output_format = both writes BOTH the ascii
                       report and a well-formed AMReX plotfile that leads
                       with the z_cc/dz fields (later phases append their
                       own fields to the same plotfile)
  inputs_badformat  -> unrecognized grid.output_format aborts fatally
  inputs_debug      -> fwt.debug = 1 prints the full diagnostics without
                       changing any result, and stays silent by default

All cases run in a scratch work directory (default
<repo>/build/regtests/phase1_grid) so no run artifacts land in the
source tree.

Usage:
    python3 check.py /path/to/fastwindterrain.exe [workdir]

Exits 0 if all cases pass, 1 otherwise (suitable for CI).
"""

import sys
import os
import subprocess
import math
import shutil

TOL = 1.0e-6  # absolute tolerance in meters for float comparisons

HERE = os.path.dirname(os.path.abspath(__file__))       # inputs live here
PHASE = os.path.basename(HERE)
ROOT = os.path.dirname(os.path.dirname(HERE))

# Every case runs here; nothing is written next to the inputs.
WORKDIR = os.path.join(ROOT, "build", "regtests", PHASE)


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
    cwd = WORKDIR
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

    report = parse_report(os.path.join(WORKDIR, "grid_report_nominal.txt"))

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

    report = parse_report(os.path.join(WORKDIR, "grid_report_uniform.txt"))
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

    report = parse_report(os.path.join(WORKDIR, "grid_report_overshoot.txt"))
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
    report_path = os.path.join(WORKDIR, "grid_report_undershoot.txt")
    if os.path.exists(report_path):
        os.remove(report_path)  # ensure stale file from a prior run can't fake a pass

    result = run_case(exe, name)
    assert result.returncode != 0, (
        f"[{name}] expected FATAL abort (nonzero exit), got 0.\n"
        f"stdout:\n{result.stdout}")
    assert not os.path.exists(report_path), (
        f"[{name}] expected no report file to be written on fatal abort")

    print(f"[PASS] {name}")


def read_plotfile_header(plt_dir):
    """Parse the leading scalars of an AMReX plotfile Header (plain ASCII):
    version, ncomp, then one variable name per line, then dim."""
    with open(os.path.join(plt_dir, "Header")) as f:
        lines = [ln.strip() for ln in f]
    version = lines[0]
    ncomp = int(lines[1])
    var_names = lines[2:2 + ncomp]
    return {"version": version, "ncomp": ncomp, "var_names": var_names}


def check_output_format(exe):
    """grid.output_format = both must emit the ascii report AND a
    well-formed native plotfile."""
    name = "inputs_plt"
    plt_dir = os.path.join(WORKDIR, "plt_grid_test")
    if os.path.isdir(plt_dir):
        shutil.rmtree(plt_dir)  # no stale plotfile can fake a pass

    result = run_case(exe, name)
    assert result.returncode == 0, (
        f"[{name}] expected success (exit 0), got {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")

    # ascii half
    report_path = os.path.join(WORKDIR, "grid_report_plt.txt")
    assert os.path.isfile(report_path), (
        f"[{name}] output_format=both did not write the ascii report")
    report = parse_report(report_path)

    # plt half
    assert os.path.isdir(plt_dir), (
        f"[{name}] output_format=both did not write the plotfile "
        f"{plt_dir}\nstdout:\n{result.stdout}")
    assert os.path.isfile(os.path.join(plt_dir, "Header")), (
        f"[{name}] plotfile {plt_dir} has no Header (not well-formed)")

    # The Phase 1 contract is that the grid fields are present and lead
    # the component list. Later phases add their own fields to the same
    # plotfile (Phase 2 appends terrain_z and mask), so this must not
    # assert the total component count.
    header = read_plotfile_header(plt_dir)
    assert header["var_names"][:2] == ["z_cc", "dz"], (
        f"[{name}] expected the plotfile to lead with ['z_cc', 'dz'], "
        f"got {header['var_names']}")
    assert header["ncomp"] == len(header["var_names"]), (
        f"[{name}] plotfile declares {header['ncomp']} components but "
        f"names {len(header['var_names'])}")

    # The plotfile is a second view of the same grid: its nominal domain
    # top must agree with the ascii report's prob_hi[2].
    nz = report["n_cell"][2]
    assert abs(report["z_face"][nz] - report["prob_hi"][2]) < TOL

    print(f"[PASS] {name}")


def check_bad_output_format(exe):
    """An unrecognized grid.output_format must abort, not silently write
    nothing."""
    name = "inputs_badformat"
    report_path = os.path.join(WORKDIR, "grid_report_badformat.txt")
    if os.path.exists(report_path):
        os.remove(report_path)

    result = run_case(exe, name)
    assert result.returncode != 0, (
        f"[{name}] expected fatal abort on an unrecognized output_format, "
        f"got exit 0.\nstdout:\n{result.stdout}")
    assert not os.path.exists(report_path), (
        f"[{name}] no report should be written for an invalid output_format")

    print(f"[PASS] {name}")


def check_debug(exe):
    """fwt.debug = 1 must print the full diagnostics, agree with the
    ascii report, and change nothing about the run. With debug off
    (the default) not a single [debug] line may appear."""
    name = "inputs_debug"
    result = run_case(exe, name)
    assert result.returncode == 0, (
        f"[{name}] expected success (exit 0), got {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")

    out = result.stdout
    assert "[debug]" in out, (
        f"[{name}] fwt.debug = 1 produced no [debug] output:\n{out}")

    # Every section the diagnostics promise to cover.
    for section in ("Run configuration", "Grid inputs", "Vertical stretching",
                    "z table", "AMReX geometry / decomposition", "box list",
                    "cells per rank", "Output settings"):
        assert f"=== {section}" in out, (
            f"[{name}] debug output is missing the '{section}' section")

    # Debug output must not fabricate the strings the other cases key on:
    # this grid is an exact match, so a WARNING here would be spurious.
    assert "WARNING" not in out, (
        f"[{name}] debug output introduced a spurious WARNING:\n{out}")

    # The debug height arithmetic must agree with the report it wrote.
    report = parse_report(os.path.join(WORKDIR, "grid_report_debug.txt"))
    nz = report["n_cell"][2]
    H_expected = analytic_H(2.0, 1.05, nz)
    assert abs(report["prob_hi"][2] - H_expected) < TOL, (
        f"[{name}] debug run changed the result: prob_hi[2] = "
        f"{report['prob_hi'][2]} != analytic H = {H_expected}")
    assert "height check     = exact match" in out, (
        f"[{name}] expected the exact-match branch to be reported\n{out}")

    # The z table must list every cell (nz = 66 is under the elision cap).
    assert out.count("[debug]   ") >= nz, (
        f"[{name}] z table looks truncated for nz = {nz}")

    # ...and the default must stay silent.
    quiet = run_case(exe, "inputs_nominal")
    assert "[debug]" not in quiet.stdout, (
        "[inputs_nominal] debug output appeared without fwt.debug set; "
        f"the default must be silent:\n{quiet.stdout}")

    print(f"[PASS] {name}")


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

    checks = [check_nominal, check_uniform, check_overshoot,
              check_undershoot, check_output_format,
              check_bad_output_format, check_debug]
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
