#!/usr/bin/env python3
"""
Phase 8 regtest checker -- post-solve diagnostics and the two output
backends.

Validates:

  well-formedness -> the ascii file is ONE file with a parseable header,
                     exactly nx*ny*nz rows, the column count its own
                     header declares, every cell of the domain present
                     exactly once, and no NaN or infinity anywhere

  completeness    -> every field the diagnostics and the phase spec call
                     for is actually in the file: u, v, w, lambda,
                     terrain_z, mask, divergence, and the rest

  backends agree  -> the plotfile and the ascii file, read independently,
                     hold the SAME value in every cell of every
                     component. This is the check that keeps the shared
                     collect routine honest: two backends that each
                     gathered their own fields would pass everything else
                     here and still drift.

  diagnostics     -> the numbers in the report are recomputed from the
                     ascii rows and must agree: max|div| and the
                     volume-weighted L2 over fluid cells, the divergence
                     being exactly zero inside the terrain, and every
                     boundary face flux, summed here from the velocity
                     and the true dz

  selection       -> output.format really selects. plt writes no ascii
                     file, ascii writes no plotfile, and an unrecognized
                     value is fatal rather than a silent fallback

The backend comparison is held to a relative 1e-12: the ascii file is
written at 17 significant digits precisely so this can be a real
comparison and not a test of the formatting.

Usage:
    python3 check.py /path/to/fastwindterrain.exe [workdir]

Exits 0 if all cases pass, 1 otherwise (suitable for CI).
"""

import sys
import os
import math
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
GROUP = os.path.basename(HERE)
REGTEST_ROOT = os.path.dirname(HERE)
ROOT = os.path.dirname(REGTEST_ROOT)

sys.path.insert(0, REGTEST_ROOT)

from plotfile import Plotfile                           # noqa: E402

WORKDIR = os.path.join(ROOT, "build", "regtests", GROUP)

NX, NY, NZ = 24, 24, 40
XHI, YHI = 1000.0, 1000.0
DX, DY = XHI / NX, YHI / NY

SOLID = 1.0

# Fields the phase spec names, plus the ones the earlier phases put in
# the plotfile. A backend that quietly drops one fails here.
REQUIRED_FIELDS = ["z_cc", "dz", "terrain_z", "mask",
                   "u", "v", "w",
                   "sigma_x", "sigma_y", "sigma_z",
                   "u0", "v0", "w0",
                   "alpha_h", "alpha_v",
                   "lambda", "divergence"]


def run_case(exe, inputs_file, extra=()):
    cmd = [exe, os.path.join(HERE, inputs_file)] + list(extra)
    return subprocess.run(cmd, cwd=WORKDIR, capture_output=True,
                          text=True, timeout=3600)


def terrain_arg():
    return [f"terrain.file={os.path.join(HERE, 'terrain_hill.csv')}"]


def require_success(name, result):
    assert result.returncode == 0, (
        f"[{name}] expected success (exit 0), got {result.returncode}\n"
        f"stdout:\n{result.stdout[-3000:]}\nstderr:\n{result.stderr[-2000:]}")


def parse_report(path):
    data = {"z_face": {}, "z_cc": {}}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if p[0] in ("z_face", "z_cc"):
                data[p[0]][int(p[1])] = float(p[2])
            elif len(p) >= 2:
                try:
                    data[p[0]] = float(p[1])
                except ValueError:
                    data[p[0]] = p[1]
    return data


class Ascii:
    """The gathered plain-text field output, parsed with nothing but the
    standard library -- which is the whole reason the backend exists."""

    def __init__(self, path):
        self.path = path
        self.meta = {}
        self.columns = None
        self.rows = {}          # (i,j,k) -> [float, ...]  (field values)
        self.coords = {}        # (i,j,k) -> (x, y)
        self.n_data_lines = 0

        with open(path) as f:
            for line in f:
                if line.startswith("#"):
                    body = line[1:].strip()
                    if body.startswith("columns:"):
                        self.columns = body[len("columns:"):].split()
                    else:
                        p = body.split()
                        if len(p) >= 2:
                            self.meta[p[0]] = p[1:]
                    continue
                if not line.strip():
                    continue
                self.n_data_lines += 1
                p = line.split()
                assert self.columns is not None, (
                    f"{path}: a data row appeared before the column header")
                assert len(p) == len(self.columns), (
                    f"{path}: row {self.n_data_lines} has {len(p)} values "
                    f"but the header declares {len(self.columns)} columns")
                i, j, k = int(p[0]), int(p[1]), int(p[2])
                key = (i, j, k)
                assert key not in self.rows, (
                    f"{path}: cell {key} appears more than once")
                self.coords[key] = (float(p[3]), float(p[4]))
                self.rows[key] = [float(v) for v in p[5:]]

        assert self.columns is not None, f"{path}: no '# columns:' header"
        self.names = self.columns[5:]
        self.index = {n: c for c, n in enumerate(self.names)}

    def get(self, name, i, j, k):
        return self.rows[(i, j, k)][self.index[name]]

    def field(self, name):
        c = self.index[name]
        return {key: v[c] for key, v in self.rows.items()}


def cells():
    for k in range(NZ):
        for j in range(NY):
            for i in range(NX):
                yield i, j, k


# ---------------------------------------------------------------------------
# 1. the ascii file is well formed
# ---------------------------------------------------------------------------

def check_ascii_well_formed(exe):
    name = "inputs_both (ascii well-formedness)"
    result = run_case(exe, "inputs_both", terrain_arg())
    require_success(name, result)

    path = os.path.join(WORKDIR, "fields_both.txt")
    assert os.path.isfile(path), (
        f"[{name}] output.format = both did not write the ascii file: {path}")

    # One file, not a directory of per-rank pieces.
    assert not os.path.isdir(path), (
        f"[{name}] the ascii output must be a single file, not a directory")
    strays = [p for p in os.listdir(WORKDIR)
              if p.startswith("fields_both") and p != "fields_both.txt"]
    assert not strays, (
        f"[{name}] the ascii output must be ONE gathered file, but these "
        f"companions appeared: {strays}")

    a = Ascii(path)

    assert a.meta["n_cell"] == [str(NX), str(NY), str(NZ)], (
        f"[{name}] header n_cell = {a.meta.get('n_cell')}, expected "
        f"{[NX, NY, NZ]}")
    ncomp = int(a.meta["ncomp"][0])
    assert ncomp == len(a.names), (
        f"[{name}] header declares ncomp = {ncomp} but names "
        f"{len(a.names)} field columns")
    n_rows = int(a.meta["n_rows"][0])
    assert n_rows == NX * NY * NZ, (
        f"[{name}] header n_rows = {n_rows}, expected {NX * NY * NZ}")
    assert a.n_data_lines == n_rows, (
        f"[{name}] header declares {n_rows} rows but the file has "
        f"{a.n_data_lines}")

    # Every cell of the domain, exactly once. The duplicate case is caught
    # in the parser; this catches the missing one.
    missing = [c for c in cells() if c not in a.rows]
    assert not missing, (
        f"[{name}] {len(missing)} cells are absent from the ascii output, "
        f"e.g. {missing[:5]}")

    # No NaN, no infinity, anywhere. A field that was never filled shows
    # up here rather than as a plausible-looking number downstream.
    for key, vals in a.rows.items():
        for c, v in enumerate(vals):
            assert math.isfinite(v), (
                f"[{name}] {a.names[c]} at cell {key} is not finite: {v}")

    # The coordinate columns must be the cell centres they claim to be.
    for (i, j, k), (x, y) in a.coords.items():
        assert abs(x - (i + 0.5) * DX) < 1.0e-9, (
            f"[{name}] x at cell ({i},{j},{k}) is {x}, expected "
            f"{(i + 0.5) * DX}")
        assert abs(y - (j + 0.5) * DY) < 1.0e-9, (
            f"[{name}] y at cell ({i},{j},{k}) is {y}, expected "
            f"{(j + 0.5) * DY}")

    print(f"[PASS] {name}  ({a.n_data_lines} rows, {ncomp} components, "
          f"{len(a.columns)} columns, every cell present once, all finite)")


# ---------------------------------------------------------------------------
# 2. every field the spec asks for is there
# ---------------------------------------------------------------------------

def check_field_completeness(exe):
    name = "inputs_both (field completeness)"

    a = Ascii(os.path.join(WORKDIR, "fields_both.txt"))
    pf = Plotfile(os.path.join(WORKDIR, "plt_both"))

    missing_a = [f for f in REQUIRED_FIELDS if f not in a.names]
    assert not missing_a, (
        f"[{name}] the ascii output is missing {missing_a}")

    missing_p = [f for f in REQUIRED_FIELDS if f not in pf.var_names]
    assert not missing_p, (
        f"[{name}] the plotfile is missing {missing_p}")

    # The two backends must not merely both contain the required set --
    # they must contain the SAME set, in the same order, since they are
    # handed the same object.
    assert a.names == list(pf.var_names), (
        f"[{name}] the backends disagree about the field list:\n"
        f"  ascii: {a.names}\n"
        f"  plt:   {list(pf.var_names)}")

    print(f"[PASS] {name}  ({len(REQUIRED_FIELDS)} required fields present "
          f"in both backends, identical ordering)")


# ---------------------------------------------------------------------------
# 3. the two backends hold the same numbers
# ---------------------------------------------------------------------------

def check_backends_agree(exe):
    name = "inputs_both (plt vs ascii)"

    a = Ascii(os.path.join(WORKDIR, "fields_both.txt"))
    pf = Plotfile(os.path.join(WORKDIR, "plt_both"))

    worst, worst_where = 0.0, None
    n_compared = 0

    for fname in a.names:
        pfield = pf.field(fname)
        acol = a.field(fname)
        for key in acol:
            pv = pfield(*key)
            av = acol[key]
            scale = max(abs(pv), abs(av), 1.0)
            rel = abs(pv - av) / scale
            n_compared += 1
            if rel > worst:
                worst, worst_where = rel, (fname, key, pv, av)

    assert worst < 1.0e-12, (
        f"[{name}] the backends disagree: {worst_where[0]} at cell "
        f"{worst_where[1]} is {worst_where[2]} in the plotfile and "
        f"{worst_where[3]} in the ascii file (relative {worst:.3e})")

    print(f"[PASS] {name}  ({n_compared} values across {len(a.names)} "
          f"components agree to {worst:.1e})")


# ---------------------------------------------------------------------------
# 4. the reported diagnostics are what the field actually says
# ---------------------------------------------------------------------------

def check_divergence_diagnostics(exe):
    name = "inputs_both (divergence diagnostics)"

    a = Ascii(os.path.join(WORKDIR, "fields_both.txt"))
    rep = parse_report(os.path.join(WORKDIR, "grid_report_both.txt"))

    # Inside the terrain the divergence is not defined, so it must be
    # exactly zero rather than a stale value that reads as a real one.
    bad = [key for key in a.rows
           if a.get("mask", *key) == SOLID and a.get("divergence", *key) != 0.0]
    assert not bad, (
        f"[{name}] divergence is nonzero in {len(bad)} solid cells, "
        f"e.g. {bad[:3]} -> {a.get('divergence', *bad[0])}")

    fluid = [key for key in a.rows if a.get("mask", *key) != SOLID]
    assert fluid, f"[{name}] the case has no fluid cells"
    assert len(fluid) == int(rep["diag_n_fluid_cells"]), (
        f"[{name}] the report counts {int(rep['diag_n_fluid_cells'])} fluid "
        f"cells, the output has {len(fluid)}")

    dvals = [a.get("divergence", *key) for key in fluid]
    dmax = max(abs(v) for v in dvals)
    dmin = min(dvals)

    assert abs(dmax - rep["diag_div_max"]) <= 1.0e-12 * max(dmax, 1.0), (
        f"[{name}] report diag_div_max = {rep['diag_div_max']} but the "
        f"field's max |div| is {dmax}")
    assert abs(dmin - rep["diag_div_min"]) <= 1.0e-12 * max(abs(dmin), 1.0), (
        f"[{name}] report diag_div_min = {rep['diag_div_min']} but the "
        f"field's min div is {dmin}")

    # The L2 is volume weighted, so a stretched grid cannot let the thin
    # near-surface cells dominate. Recomputed here from the dz column.
    num = sum(a.get("divergence", *key) ** 2 * DX * DY * a.get("dz", *key)
              for key in fluid)
    den = sum(DX * DY * a.get("dz", *key) for key in fluid)
    l2 = math.sqrt(num / den)
    assert abs(l2 - rep["diag_div_l2"]) <= 1.0e-9 * max(l2, 1.0), (
        f"[{name}] report diag_div_l2 = {rep['diag_div_l2']}, recomputed "
        f"{l2}")

    print(f"[PASS] {name}  (max|div| {dmax:.4g}, L2 {l2:.4g} over "
          f"{len(fluid)} fluid cells, both matching the report; div "
          f"identically 0 in all "
          f"{len(a.rows) - len(fluid)} solid cells)")


# ---------------------------------------------------------------------------
# 5. the mass-flux balance and the boundary flux check
# ---------------------------------------------------------------------------

def check_mass_flux(exe):
    name = "inputs_both (mass-flux balance)"

    a = Ascii(os.path.join(WORKDIR, "fields_both.txt"))
    rep = parse_report(os.path.join(WORKDIR, "grid_report_both.txt"))

    # Recompute every face flux from the velocity and the true dz. Sign
    # convention: positive is out of the domain.
    faces = {
        "xlo": [(0, j, k) for j in range(NY) for k in range(NZ)],
        "xhi": [(NX - 1, j, k) for j in range(NY) for k in range(NZ)],
        "ylo": [(i, 0, k) for i in range(NX) for k in range(NZ)],
        "yhi": [(i, NY - 1, k) for i in range(NX) for k in range(NZ)],
        "top": [(i, j, NZ - 1) for i in range(NX) for j in range(NY)],
    }
    comp = {"xlo": "u", "xhi": "u", "ylo": "v", "yhi": "v", "top": "w"}
    sign = {"xlo": -1.0, "xhi": 1.0, "ylo": -1.0, "yhi": 1.0, "top": 1.0}

    total_in, total_out = 0.0, 0.0
    for f, keys in faces.items():
        net = 0.0
        for key in keys:
            if a.get("mask", *key) == SOLID:
                continue
            area = (DX * DY if f == "top"
                    else (DY if f.startswith("x") else DX) * a.get("dz", *key))
            flux = sign[f] * a.get(comp[f], *key) * area
            net += flux
            if flux > 0.0:
                total_out += flux
            else:
                total_in -= flux

        reported = rep[f"diag_flux_{f}"]
        scale = max(abs(reported), abs(net), 1.0)
        assert abs(net - reported) <= 1.0e-9 * scale, (
            f"[{name}] face {f}: the report says {reported} m^3/s, "
            f"recomputed {net} m^3/s from the output fields")

    scale = max(abs(rep["diag_flux_in"]), abs(rep["diag_flux_out"]), 1.0)
    assert abs(total_in - rep["diag_flux_in"]) <= 1.0e-9 * scale, (
        f"[{name}] diag_flux_in = {rep['diag_flux_in']}, recomputed "
        f"{total_in}")
    assert abs(total_out - rep["diag_flux_out"]) <= 1.0e-9 * scale, (
        f"[{name}] diag_flux_out = {rep['diag_flux_out']}, recomputed "
        f"{total_out}")

    # Internal consistency of the reported budget.
    net = rep["diag_flux_out"] - rep["diag_flux_in"]
    assert abs(net - rep["diag_flux_net"]) <= 1.0e-9 * scale, (
        f"[{name}] diag_flux_net = {rep['diag_flux_net']} but out - in = "
        f"{net}")
    imb = abs(rep["diag_flux_net"]) / scale
    assert abs(imb - rep["diag_flux_imbalance"]) <= 1.0e-12, (
        f"[{name}] diag_flux_imbalance = {rep['diag_flux_imbalance']}, "
        f"recomputed {imb}")

    # The projection is what closes this budget: the net boundary flux is
    # the volume integral of the divergence, and the solve drives that to
    # zero in the norm it controls. A tolerance of 1e-6 is far looser
    # than the ~1e-14 the case actually reaches, and is here to catch a
    # broken projection rather than to measure a good one.
    assert rep["diag_flux_imbalance"] < 1.0e-6, (
        f"[{name}] relative flux imbalance {rep['diag_flux_imbalance']} is "
        f"too large: the projected field does not close its mass budget")
    assert rep["diag_flux_within_tolerance"] == 1.0, (
        f"[{name}] the run reported the flux imbalance as outside "
        f"diagnostics.flux_tolerance = {rep['diag_flux_tolerance']}")

    print(f"[PASS] {name}  (all 5 face fluxes recomputed from the output "
          f"match the report; in/out {rep['diag_flux_in']:.6g} / "
          f"{rep['diag_flux_out']:.6g} m^3/s, relative imbalance "
          f"{rep['diag_flux_imbalance']:.2e})")


def check_flux_tolerance_warns(exe):
    """An imbalance above the tolerance must be reported, and reported
    only -- never corrected and never fatal."""
    name = "inputs_both (flux tolerance)"

    result = run_case(exe, "inputs_both", terrain_arg() + [
        "diagnostics.flux_tolerance=0.0",
        "grid.output_format=report",
        "grid.report_file=grid_report_tol.txt"])
    require_success(name, result)

    rep = parse_report(os.path.join(WORKDIR, "grid_report_tol.txt"))
    assert rep["diag_flux_within_tolerance"] == 0.0, (
        f"[{name}] with flux_tolerance = 0 the run should report the "
        f"imbalance as outside tolerance")
    assert "WARNING" in result.stdout, (
        f"[{name}] exceeding the tolerance must warn:\n"
        f"{result.stdout[-2000:]}")

    # The field itself must be untouched: the imbalance is a measurement,
    # not something the code is allowed to quietly fix.
    base = parse_report(os.path.join(WORKDIR, "grid_report_both.txt"))
    assert abs(rep["diag_flux_net"] - base["diag_flux_net"]) < 1.0e-30, (
        f"[{name}] changing the tolerance changed the answer: net flux "
        f"{base['diag_flux_net']} -> {rep['diag_flux_net']}")

    print(f"[PASS] {name}  (imbalance flagged and warned about, field "
          f"unchanged: net still {rep['diag_flux_net']:.4g} m^3/s)")


# ---------------------------------------------------------------------------
# 6. output.format really selects a backend
# ---------------------------------------------------------------------------

def check_backend_selection(exe):
    name = "output.format selection"

    sub = os.path.join(WORKDIR, "selection")
    shutil.rmtree(sub, ignore_errors=True)
    os.makedirs(sub)

    def run_in(subdir, extra):
        cmd = [exe, os.path.join(HERE, "inputs_both")] + terrain_arg() + extra
        return subprocess.run(cmd, cwd=subdir, capture_output=True,
                              text=True, timeout=3600)

    # plt only
    d = os.path.join(sub, "plt")
    os.makedirs(d)
    r = run_in(d, ["output.format=plt"])
    require_success(f"{name} [plt]", r)
    assert os.path.isdir(os.path.join(d, "plt_both")), (
        f"[{name}] output.format = plt wrote no plotfile")
    assert not os.path.exists(os.path.join(d, "fields_both.txt")), (
        f"[{name}] output.format = plt wrote an ascii file anyway")

    # ascii only
    d = os.path.join(sub, "ascii")
    os.makedirs(d)
    r = run_in(d, ["output.format=ascii"])
    require_success(f"{name} [ascii]", r)
    assert os.path.isfile(os.path.join(d, "fields_both.txt")), (
        f"[{name}] output.format = ascii wrote no ascii file")
    assert not os.path.exists(os.path.join(d, "plt_both")), (
        f"[{name}] output.format = ascii wrote a plotfile anyway")

    # grid.output_format = report suppresses the field output entirely,
    # whichever backend was asked for.
    d = os.path.join(sub, "report_only")
    os.makedirs(d)
    r = run_in(d, ["grid.output_format=report", "output.format=both"])
    require_success(f"{name} [report only]", r)
    assert os.path.isfile(os.path.join(d, "grid_report_both.txt")), (
        f"[{name}] grid.output_format = report wrote no report")
    assert not os.path.exists(os.path.join(d, "fields_both.txt")) and \
           not os.path.exists(os.path.join(d, "plt_both")), (
        f"[{name}] grid.output_format = report still wrote field output")

    # An unrecognized value is fatal, not a silent fallback to plt.
    d = os.path.join(sub, "bad")
    os.makedirs(d)
    r = run_in(d, ["output.format=vtk"])
    assert r.returncode != 0, (
        f"[{name}] output.format = vtk should abort, but the run "
        f"succeeded")
    assert not os.path.exists(os.path.join(d, "plt_both")) and \
           not os.path.exists(os.path.join(d, "fields_both.txt")), (
        f"[{name}] an invalid output.format still wrote field output")

    # The alias must keep working: every input file written before
    # grid.output_format gained its clearer value names says 'ascii'.
    d = os.path.join(sub, "alias")
    os.makedirs(d)
    r = run_in(d, ["grid.output_format=ascii"])
    require_success(f"{name} [alias]", r)
    assert os.path.isfile(os.path.join(d, "grid_report_both.txt")), (
        f"[{name}] grid.output_format = ascii (the old spelling of "
        f"'report') no longer writes the report")

    shutil.rmtree(sub, ignore_errors=True)
    print(f"[PASS] {name}  (plt, ascii, both, report-only and the legacy "
          f"aliases all select correctly; an unknown format aborts)")


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
    # check_ascii_well_formed runs the case the rest of them read.
    for check in (check_ascii_well_formed, check_field_completeness,
                  check_backends_agree, check_divergence_diagnostics,
                  check_mass_flux, check_flux_tolerance_warns,
                  check_backend_selection):
        try:
            check(exe)
        except AssertionError as e:
            print(f"[FAIL] {check.__name__}: {e}")
            failed.append(check.__name__)
        except Exception as e:
            print(f"[ERROR] {check.__name__}: {e}")
            failed.append(check.__name__)

    if failed:
        print(f"\n{len(failed)} Phase 8 regtest case(s) failed: {failed}")
        return 1

    print("\nAll Phase 8 regtest cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
