#!/usr/bin/env python3
"""
Grid-convergence study of the derivative schemes, run through the solver.

Sweeps  inflow profile x derivative scheme x vertical resolution,
generates an input file for each combination, runs the solver, and
reports the observed order of accuracy.

WHAT IS BEING MEASURED, AND WHY IT IS THE VERTICAL DERIVATIVE

Over flat ground a powerlaw or loglaw profile has div(u) = 0
*identically*: u and v depend only on z and w is zero, so du/dx, dv/dy
and dw/dz all vanish for every scheme on every grid. A divergence-based
study there would measure nothing.

The only nontrivial derivative in that problem is dU/dz -- which is also
where the three schemes actually differ, and where a wrong vertical
metric on a stretched grid shows up. So the study measures the solver's
own d(u)/dz and d(v)/dz, dumped by verify.gradient_dump_file, against
the analytic derivative of the configured profile law.

The analytic law is implemented here, independently of the C++, so the
solver is not grading its own work.

WHAT THIS COVERS THAT numerics.selftest_file DOES NOT

The self-test measures the same schemes on a std::vector holding
sin(2 pi x). It never builds a Grid, a MultiFab or a ghost cell. This
study runs the real path: the column metric, the box decomposition, the
index clamping at the domain ends, the profile as the solver actually
evaluated it.

WHICH LEVELS COUNT

The error is measured over a FIXED PHYSICAL BAND of height, not over
whatever levels happen to exist. This is the part that makes the study
work at all.

Both profile laws have dU/dz ~ z^(alpha-1) or ~ 1/z, so the derivative
is near-singular as z -> 0. On a stretched grid the first cell thins as
nz grows, so an "all interior levels" norm would march its own lower
limit toward that singularity: measured that way the error INCREASES
under refinement, and the first version of this script duly reported
order -0.4. Holding the band fixed measures the same physical region at
every resolution, which is what a grid-convergence study means.

Three exclusions, all stated rather than assumed:

  * levels outside [--z-window lo, hi], the fixed measurement band
  * the two levels at each end of the column, where the stencil index is
    clamped to the domain and the derivative is one-sided by design
  * any level at or below inflow.z_agl_min, plus a stencil radius above
    it, where the profile is floored and its derivative has a kink.
    A stencil straddling that kink is not measuring a smooth function

REFINEMENT

Vertical resolution is refined at FIXED TOTAL STRETCH: the ratio is
recomputed at each nz so the last cell is always the same multiple of
the first. Holding the ratio itself fixed would change the underlying
mapping as the grid refines, and the study would stop being a
convergence study at all.

Usage:
    python3 convergence/run_convergence.py build/fastwindterrain
    python3 convergence/run_convergence.py <exe> --uniform --advect both
    python3 convergence/run_convergence.py <exe> --grids 32,64,128 --check

Writes convergence_results.csv and a summary table to the work directory.
With --check, exits nonzero if any observed order falls below the
threshold for its scheme.
"""

import argparse
import csv
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Matches fwt::kStencilRadius. Both upwind schemes reach two cells.
STENCIL_RADIUS = 2

PROFILES = ("powerlaw", "loglaw")
SCHEMES = ("central2", "upwind2", "weno3js")
# The coarse end is pre-asymptotic: at nz = 32 the stretched grid puts
# only ~26 levels inside the measurement band, and the band's level set
# changes discretely with nz, which makes the first order estimate jitter.
# The sweep starts where the asymptotic rate is actually visible.
DEFAULT_GRIDS = (64, 128, 256, 512)

# Horizontal size. The profile is horizontally uniform, so x/y resolution
# measures nothing here -- but max_grid_size is set below it so the run
# is decomposed into several boxes, which is part of what this study
# exercises and the self-test cannot.
NX = NY = 16
MAX_GRID_SIZE = 8

LX = LY = 1000.0
HEIGHT = 1000.0
TOTAL_STRETCH = 10.0

U_REF, V_REF = 8.0, 6.0
Z_REF = 10.0
POWERLAW_EXPONENT = 0.14
Z0 = 0.05

# The fixed band the error is measured over [m]. Both ends are well
# inside the domain and well above the profile floor, so the same
# physical region is resolved at every resolution.
Z_WINDOW = (20.0, 900.0)

# Minimum observed order accepted on the finest grid pair, in L2. Set
# below what the schemes reach but above the next order down, so the
# check has teeth without being brittle. Calibrated from measurement --
# see docs/convergence.rst for the numbers this actually produces.
MIN_ORDER_L2 = {
    "central2": 1.80,
    "upwind2": 1.80,
    "weno3js": 2.50,
}


# ---------------------------------------------------------------------------
# The profile laws, implemented independently of the solver
# ---------------------------------------------------------------------------

def profile_speed(mode, z_agl, speed_ref, z_ref, alpha, z0, z_agl_min):
    z = max(z_agl, z_agl_min)
    if mode == "powerlaw":
        return speed_ref * (z / z_ref) ** alpha
    return speed_ref * math.log((z + z0) / z0) / math.log((z_ref + z0) / z0)


def profile_speed_gradient(mode, z_agl, speed_ref, z_ref, alpha, z0,
                           z_agl_min):
    """dU/dz. Zero in the floored region, where U is constant."""
    if z_agl <= z_agl_min:
        return 0.0
    z = z_agl
    if mode == "powerlaw":
        return speed_ref * alpha / z_ref * (z / z_ref) ** (alpha - 1.0)
    return speed_ref / ((z + z0) * math.log((z_ref + z0) / z0))


# ---------------------------------------------------------------------------
# Grid construction, mirroring Grid::BuildVerticalStretching
# ---------------------------------------------------------------------------

def vertical_grid(nz, stretched):
    """(dz0, ratio, prob_hi_z). The domain top is the height the stretched
    grid actually reaches, quoted exactly, so the solver's height check
    sees a match rather than an overshoot it has to correct."""
    if stretched:
        r = TOTAL_STRETCH ** (1.0 / (nz - 1))
    else:
        r = 1.0
    total = sum(r ** k for k in range(nz))
    dz0 = HEIGHT / total
    # Re-sum from the quoted dz0 so prob_hi is exactly what the solver
    # will compute, to the last bit.
    h = 0.0
    for k in range(nz):
        h += dz0 * r ** k
    return dz0, r, h


def z_centers(nz, dz0, r):
    zf = [0.0]
    for k in range(nz):
        zf.append(zf[-1] + dz0 * r ** k)
    return [0.5 * (zf[k] + zf[k + 1]) for k in range(nz)]


# ---------------------------------------------------------------------------
# Running one case
# ---------------------------------------------------------------------------

INPUTS_TEMPLATE = """# Generated by convergence/run_convergence.py -- do not edit by hand.
#
# {profile} profile, {scheme} scheme, nz = {nz}, {gridkind} vertical grid.
# Flat ground, so the only nontrivial derivative is dU/dz.

grid.n_cell           = {nx} {ny} {nz}
grid.prob_lo          = 0.0 0.0 0.0
grid.prob_hi          = {lx!r} {ly!r} {hi!r}
grid.dz0              = {dz0!r}
grid.stretching_ratio = {ratio!r}
grid.max_grid_size    = {mgs}

terrain.flat_elevation = 0.0

inflow.mode              = {profile}
inflow.u_ref             = {u_ref!r}
inflow.v_ref             = {v_ref!r}
inflow.z_ref             = {z_ref!r}
inflow.powerlaw_exponent = {alpha!r}
inflow.z0                = {z0!r}

numerics.gradient_scheme = {scheme}

verify.gradient_dump_file = {dump}
verify.gradient_advect    = {advect!r}

# The study reads the profile gradient, which is taken before the
# projection; one pass is enough to keep the run honest and quick.
poisson.n_projections = 1

grid.output_format = report
grid.report_file   = {report}
"""


def write_inputs(path, profile, scheme, nz, stretched, advect, dump, report):
    dz0, r, hi = vertical_grid(nz, stretched)
    with open(path, "w") as f:
        f.write(INPUTS_TEMPLATE.format(
            profile=profile, scheme=scheme, nz=nz,
            gridkind="stretched" if stretched else "uniform",
            nx=NX, ny=NY, mgs=MAX_GRID_SIZE,
            lx=LX, ly=LY, hi=hi, dz0=dz0, ratio=r,
            u_ref=U_REF, v_ref=V_REF, z_ref=Z_REF,
            alpha=POWERLAW_EXPONENT, z0=Z0,
            dump=dump, advect=advect, report=report))
    return dz0, r


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


def parse_dump(path):
    """(meta, rows) where each row is a dict of the named columns."""
    meta, rows, cols = {}, [], None
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                body = line[1:].strip()
                p = body.split()
                if p and p[0] == "k" and "z_cc" in p:
                    cols = p
                elif len(p) >= 2:
                    meta[p[0]] = p[1]
                continue
            if not line.strip():
                continue
            if cols is None:
                raise ValueError(f"{path}: data before the column header")
            vals = line.split()
            if len(vals) != len(cols):
                raise ValueError(
                    f"{path}: row has {len(vals)} values, header names "
                    f"{len(cols)} columns")
            rows.append({c: float(v) for c, v in zip(cols, vals)})
    return meta, rows


def run_one(exe, workdir, profile, scheme, nz, stretched, advect,
            z_window=Z_WINDOW):
    tag = f"{profile}_{scheme}_{'s' if stretched else 'u'}"
    tag += f"_a{'p' if advect > 0 else 'm'}_n{nz}"
    inputs = os.path.join(workdir, f"inputs_{tag}")
    dump = f"grad_{tag}.txt"
    report = f"report_{tag}.txt"

    dz0, r = write_inputs(inputs, profile, scheme, nz, stretched, advect,
                          dump, report)

    result = subprocess.run([exe, inputs], cwd=workdir, capture_output=True,
                            text=True, timeout=3600)
    if result.returncode != 0:
        raise RuntimeError(
            f"{tag}: the solver exited {result.returncode}\n"
            f"stdout:\n{result.stdout[-2000:]}\n"
            f"stderr:\n{result.stderr[-1000:]}")

    rep = parse_report(os.path.join(workdir, report))
    _, rows = parse_dump(os.path.join(workdir, dump))

    speed_ref = rep["inflow_speed_ref"]
    z_ref = rep["inflow_z_ref"]
    alpha = rep["inflow_powerlaw_exponent"]
    z0 = rep["inflow_z0"]
    z_agl_min = rep["inflow_z_agl_min"]
    dir_x = rep["inflow_u_ref"] / speed_ref
    dir_y = rep["inflow_v_ref"] / speed_ref

    # Levels the study can legitimately measure. The exclusions are
    # properties of the discretization, not of the answer: the clamped
    # ends are one-sided by design, and a stencil straddling the profile
    # floor is not differentiating a smooth function.
    n = len(rows)
    k_lo = STENCIL_RADIUS
    k_hi = n - 1 - STENCIL_RADIUS
    for row in rows:
        if row["z_cc"] <= z_agl_min:
            k_lo = max(k_lo, int(row["k"]) + 1 + STENCIL_RADIUS)
    z_min, z_max = z_window

    num_u = num_v = 0.0
    den = 0.0
    linf = 0.0
    spread = 0.0
    n_levels = 0

    for row in rows:
        k = int(row["k"])
        if k < k_lo or k > k_hi or row["n_fluid"] <= 0:
            continue
        z = row["z_cc"]
        # The fixed band. Without this the lower limit of the norm walks
        # toward z = 0, where dU/dz is near-singular, and the error grows
        # under refinement instead of falling.
        if z < z_min or z > z_max:
            continue
        dUdz = profile_speed_gradient(profile, z, speed_ref, z_ref, alpha,
                                      z0, z_agl_min)
        eu = row["dudz_mid"] - dUdz * dir_x
        ev = row["dvdz_mid"] - dUdz * dir_y
        w = row["dz"]
        num_u += eu * eu * w
        num_v += ev * ev * w
        den += w
        linf = max(linf, abs(eu), abs(ev))
        spread = max(spread, row["dudz_spread"], row["dvdz_spread"])
        n_levels += 1

    if n_levels < 4:
        raise RuntimeError(
            f"{tag}: only {n_levels} levels survive the exclusions; the "
            f"grid is too coarse for this study")

    l2 = math.sqrt((num_u + num_v) / den)
    return {
        "profile": profile, "scheme": scheme, "nz": nz,
        "grid": "stretched" if stretched else "uniform",
        "advect": advect, "dz0": dz0, "ratio": r,
        "l2": l2, "linf": linf, "spread": spread, "n_levels": n_levels,
    }


def observed_order(coarse, fine, key):
    """The grid is refined at fixed total stretch, so every spacing scales
    as 1/nz and the resolution ratio is the refinement ratio."""
    if coarse[key] <= 0.0 or fine[key] <= 0.0:
        return float("nan")
    return (math.log(coarse[key] / fine[key])
            / math.log(fine["nz"] / coarse["nz"]))


def main():
    ap = argparse.ArgumentParser(
        description="Grid-convergence study of the derivative schemes.")
    ap.add_argument("exe", help="path to the fastwindterrain executable")
    ap.add_argument("--workdir", default=None,
                    help="scratch directory (default build/convergence)")
    ap.add_argument("--grids", default=None,
                    help="comma-separated vertical resolutions "
                         f"(default {','.join(str(g) for g in DEFAULT_GRIDS)})")
    ap.add_argument("--profiles", default=",".join(PROFILES))
    ap.add_argument("--schemes", default=",".join(SCHEMES))
    ap.add_argument("--uniform", action="store_true",
                    help="uniform vertical grid instead of stretched")
    ap.add_argument("--both-grids", action="store_true",
                    help="run the uniform and stretched grids")
    ap.add_argument("--advect", default="plus", choices=("plus", "minus",
                                                         "both"),
                    help="which upwind branch to measure (default plus)")
    ap.add_argument("--z-window", default=None,
                    help="fixed measurement band 'lo,hi' in metres "
                         f"(default {Z_WINDOW[0]},{Z_WINDOW[1]})")
    ap.add_argument("--check", action="store_true",
                    help="assert the observed orders and exit nonzero on "
                         "failure")
    args = ap.parse_args()

    exe = os.path.abspath(args.exe)
    if not os.path.isfile(exe):
        print(f"executable not found: {exe}")
        return 1

    workdir = os.path.abspath(args.workdir
                              or os.path.join(ROOT, "build", "convergence"))
    os.makedirs(workdir, exist_ok=True)

    grids = ([int(g) for g in args.grids.split(",")] if args.grids
             else list(DEFAULT_GRIDS))
    profiles = args.profiles.split(",")
    schemes = args.schemes.split(",")
    if args.both_grids:
        grid_kinds = [True, False]
    else:
        grid_kinds = [not args.uniform]
    advects = {"plus": [1.0], "minus": [-1.0], "both": [1.0, -1.0]}[args.advect]
    if args.z_window:
        lo, hi = (float(v) for v in args.z_window.split(","))
        z_window = (lo, hi)
    else:
        z_window = Z_WINDOW

    n_runs = (len(profiles) * len(schemes) * len(grids) * len(grid_kinds)
              * len(advects))
    print(f"work directory: {workdir}")
    print(f"{n_runs} runs: {len(profiles)} profiles x {len(schemes)} schemes "
          f"x {len(grids)} grids x {len(grid_kinds)} grid kinds "
          f"x {len(advects)} advect signs")
    print(f"error measured over the fixed band z in "
          f"[{z_window[0]}, {z_window[1]}] m\n")

    results = []
    for stretched in grid_kinds:
        for advect in advects:
            for profile in profiles:
                for scheme in schemes:
                    for nz in sorted(grids):
                        r = run_one(exe, workdir, profile, scheme, nz,
                                    stretched, advect, z_window)
                        results.append(r)
                        print(f"  {r['grid']:9s} advect{advect:+.0f} "
                              f"{profile:9s} {scheme:9s} nz={nz:4d}  "
                              f"L2={r['l2']:.4e}  Linf={r['linf']:.4e}")

    # Observed orders, per series.
    series = {}
    for r in results:
        series.setdefault(
            (r["grid"], r["advect"], r["profile"], r["scheme"]), []).append(r)
    for runs in series.values():
        runs.sort(key=lambda r: r["nz"])
        for i, r in enumerate(runs):
            if i == 0:
                r["order_l2"] = float("nan")
                r["order_linf"] = float("nan")
            else:
                r["order_l2"] = observed_order(runs[i-1], r, "l2")
                r["order_linf"] = observed_order(runs[i-1], r, "linf")

    csv_path = os.path.join(workdir, "convergence_results.csv")
    fields = ["grid", "advect", "profile", "scheme", "nz", "dz0", "ratio",
              "l2", "linf", "order_l2", "order_linf", "spread", "n_levels"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fields})

    print(f"\n{'grid':10s} {'adv':>4s} {'profile':9s} {'scheme':9s} "
          f"{'nz':>5s} {'L2':>12s} {'ord':>6s} {'Linf':>12s} {'ord':>6s}")
    print("-" * 82)
    for key in sorted(series):
        for r in series[key]:
            o2 = "" if math.isnan(r["order_l2"]) else f"{r['order_l2']:6.2f}"
            oi = ("" if math.isnan(r["order_linf"])
                  else f"{r['order_linf']:6.2f}")
            print(f"{r['grid']:10s} {r['advect']:+4.0f} {r['profile']:9s} "
                  f"{r['scheme']:9s} {r['nz']:5d} {r['l2']:12.4e} {o2:>6s} "
                  f"{r['linf']:12.4e} {oi:>6s}")
        print()

    print(f"wrote {csv_path}")

    failures = []
    # Horizontal uniformity is not part of the order study, but a broken
    # run would quietly change what the study measured, so it is asserted
    # everywhere regardless of --check.
    for r in results:
        if r["spread"] > 1.0e-12:
            failures.append(
                f"{r['profile']}/{r['scheme']}/nz={r['nz']}: the gradient "
                f"is not horizontally uniform over flat ground "
                f"(spread {r['spread']:.3e})")

    if args.check:
        for key, runs in sorted(series.items()):
            finest = runs[-1]
            want = MIN_ORDER_L2[finest["scheme"]]
            got = finest["order_l2"]
            if math.isnan(got) or got < want:
                failures.append(
                    f"{finest['grid']}/{finest['profile']}/"
                    f"{finest['scheme']}: L2 order {got:.2f} on the finest "
                    f"pair, expected at least {want}")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  {f}")
        return 1

    print("\nAll series converged at or above the expected order."
          if args.check else "\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
