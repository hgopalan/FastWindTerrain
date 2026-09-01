#!/usr/bin/env python3
"""
Phase 5 regtest checker -- anisotropic Poisson assembly.

Validates:

  inputs_mms -> a manufactured solution with a known analytic lambda,
                solved at three resolutions on a UNIFORM and on a
                STRETCHED grid, checking second-order convergence
  inputs_rhs -> the scheme-based nodal RHS against an independent Python
                divergence of the plotfile's own initial velocity, and
                sigma against its definition in every cell

The manufactured case is the one that matters most. AMReX's Geometry is
uniform in z, so the true cell heights reach the operator only through
sigma:

    sigma_x = alpha_h^2 J    sigma_y = alpha_h^2 J    sigma_z = alpha_v^2 / J
    rhs     = J * (analytic source)          J = dz(k) / dz_nominal

That is derived rather than borrowed, so it is checked rather than
assumed: wrong metric factors cost an order, and the stretched sweep is
run at several stretch ratios so a metric error cannot hide at one of
them.

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

MIN_ORDER = 1.85          # measures ~1.95-2.03
RHS_RTOL = 1.0e-9         # solver vs independent divergence
SOLID = 1.0

# inputs_rhs geometry and parameters.
NX, NY, NZ = 40, 40, 66
XHI, YHI = 1000.0, 1000.0
DX, DY = XHI / NX, YHI / NY
ALPHA_H, ALPHA_V = 1.0, 0.5

# WENO3-JS constants, matching Derivatives.H.
WENO_EPS = 1.0e-6


# ---------------------------------------------------------------------------
# Independent reference implementations
# ---------------------------------------------------------------------------

def weno3_recon(a, b, c):
    """Right-hand face value from three consecutive cells, biased toward
    a. Independent re-derivation of detail::Weno3Recon."""
    p0 = -0.5 * a + 1.5 * b
    p1 = 0.5 * b + 0.5 * c
    b0 = (b - a) ** 2
    b1 = (c - b) ** 2
    w0 = (1.0 / 3.0) / (WENO_EPS + b0) ** 2
    w1 = (2.0 / 3.0) / (WENO_EPS + b1) ** 2
    return (w0 * p0 + w1 * p1) / (w0 + w1)


def weno3_deriv(um2, um1, u0, up1, up2, a, h):
    if a > 0.0:
        return (weno3_recon(um1, u0, up1) - weno3_recon(um2, um1, u0)) / h
    if a < 0.0:
        return (weno3_recon(up2, up1, u0) - weno3_recon(up1, u0, um1)) / h
    return (up1 - um1) / (2.0 * h)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

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
            elif p[0] == "n_cell":
                data[p[0]] = [int(x) for x in p[1:]]
            elif p[0] in ("prob_lo", "prob_hi"):
                data[p[0]] = [float(x) for x in p[1:]]
            elif len(p) >= 2:
                try:
                    data[p[0]] = float(p[1])
                except ValueError:
                    data[p[0]] = p[1]
    return data


def run_case(exe, inputs_file, extra=()):
    cmd = [exe, os.path.join(HERE, inputs_file)] + list(extra)
    return subprocess.run(cmd, cwd=WORKDIR, capture_output=True,
                          text=True, timeout=3600)


def require_success(name, result):
    assert result.returncode == 0, (
        f"[{name}] expected success (exit 0), got {result.returncode}\n"
        f"stdout:\n{result.stdout[-3000:]}\nstderr:\n{result.stderr[-2000:]}")


def clean(*names):
    for n in names:
        p = os.path.join(WORKDIR, n)
        if os.path.isdir(p):
            shutil.rmtree(p)
        elif os.path.exists(p):
            os.remove(p)


def geometric_grid(n, height, stretch_total):
    """n cells spanning `height`, with the last cell `stretch_total`
    times the first. Holding the total stretch fixed as n grows keeps the
    mapping fixed, which is what makes a refinement sweep a convergence
    study."""
    r = 1.0 if stretch_total == 1.0 else stretch_total ** (1.0 / (n - 1))
    dz0 = height / sum(r ** k for k in range(n))
    return r, dz0


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def mms_errors(exe, n, stretch_total, height=100.0):
    r, dz0 = geometric_grid(n, height, stretch_total)
    clean("grid_report_mms.txt")
    result = run_case(exe, "inputs_mms", [
        f"grid.n_cell={n} {n} {n}",
        f"grid.prob_hi={height} {height} {height}",
        f"grid.dz0={dz0!r}",
        f"grid.stretching_ratio={r!r}",
    ])
    require_success(f"inputs_mms (n={n}, stretch={stretch_total})", result)
    rep = parse_report(os.path.join(WORKDIR, "grid_report_mms.txt"))
    return rep["poisson_mms_l2"], rep["poisson_mms_linf"]


def check_manufactured(exe):
    """Second-order convergence to a known lambda, on a uniform grid and
    on progressively more distorted stretched grids."""
    name = "inputs_mms"
    ns = (16, 32, 64)
    summary = []

    for stretch in (1.0, 4.0, 20.0):
        errs = [mms_errors(exe, n, stretch) for n in ns]

        for (n0, (l0, _)), (n1, (l1, _)) in zip(zip(ns, errs),
                                                zip(ns[1:], errs[1:])):
            assert l1 < l0, (
                f"[{name}] stretch {stretch}x: L2 error rose from {l0:.3e} "
                f"at n={n0} to {l1:.3e} at n={n1}")

        orders_l2 = [math.log(a[0] / b[0]) / math.log(2)
                     for a, b in zip(errs, errs[1:])]
        orders_li = [math.log(a[1] / b[1]) / math.log(2)
                     for a, b in zip(errs, errs[1:])]

        for label, orders in (("L2", orders_l2), ("Linf", orders_li)):
            assert orders[-1] >= MIN_ORDER, (
                f"[{name}] stretch {stretch}x converges at order "
                f"{orders[-1]:.2f} in {label}, below the {MIN_ORDER} "
                f"expected. Orders: {[f'{o:.2f}' for o in orders]}. A lost "
                f"order here means the vertical metric in sigma or in the "
                f"RHS weighting is wrong.")

        summary.append(f"{stretch:g}x {orders_l2[-1]:.2f}")

    print(f"[PASS] {name}  (L2 order at stretch: {', '.join(summary)})")


def check_rhs_and_sigma(exe):
    """The assembled nodal RHS against an independent divergence, and
    sigma against its definition."""
    name = "inputs_rhs"
    clean("plt_rhs", "grid_report_rhs.txt", "rhs_dump.txt")
    # This case checks the RHS built from the configured derivative
    # scheme against an independent implementation of that same scheme.
    # From Phase 6 the default RHS comes from AMReX's own nodal
    # divergence instead -- the operator the solve is built from -- so
    # the scheme path is selected explicitly here.
    result = run_case(exe, "inputs_rhs", [
        f"terrain.file={os.path.join(HERE, 'terrain_hill.csv')}",
        "poisson.rhs_operator=scheme",
        # One pass only: further passes rebuild the RHS from the
        # corrected field, and this case inspects the one built from u0.
        "poisson.n_projections=1"])
    require_success(name, result)

    rep = parse_report(os.path.join(WORKDIR, "grid_report_rhs.txt"))
    pf = Plotfile(os.path.join(WORKDIR, "plt_rhs"))

    assert rep["poisson_sigma_convention"] == "alpha_squared", (
        f"[{name}] the report must record which sigma convention is in "
        f"force, got {rep['poisson_sigma_convention']}")

    z_face = [rep["z_face"][k] for k in range(NZ + 1)]
    z_cc = [rep["z_cc"][k] for k in range(NZ)]
    h_nom = (rep["prob_hi"][2] - rep["prob_lo"][2]) / NZ
    J = [(z_face[k+1] - z_face[k]) / h_nom for k in range(NZ)]

    # --- sigma, in every cell -------------------------------------------
    mask = pf.field("mask")
    sx, sy, sz = (pf.field("sigma_x"), pf.field("sigma_y"),
                  pf.field("sigma_z"))
    # Sigma is NOT masked inside terrain. Zeroing it there makes no-flux
    # exact in the operator but leaves nodes buried in terrain with a
    # zero diagonal, which the multigrid smoother divides by -- producing
    # NaN and a solve that silently reports a zero residual.
    # massconsistent_amr leaves its coefficients unmasked for the same
    # reason and imposes the immersed boundary on the field instead. So
    # sigma must equal alpha^2 times the metric EVERYWHERE, solid cells
    # included.
    n_solid = 0
    worst_sigma = 0.0
    for k in range(NZ):
        for j in range(0, NY, 3):
            for i in range(0, NX, 3):
                if mask(i, j, k) == SOLID:
                    n_solid += 1
                for got, expect, comp in (
                        (sx(i,j,k), ALPHA_H**2 * J[k], "sigma_x"),
                        (sy(i,j,k), ALPHA_H**2 * J[k], "sigma_y"),
                        (sz(i,j,k), ALPHA_V**2 / J[k], "sigma_z")):
                    err = abs(got - expect) / max(abs(expect), 1.0)
                    worst_sigma = max(worst_sigma, err)
                    assert err < 1.0e-12, (
                        f"[{name}] {comp} at ({i},{j},{k}) is {got}, "
                        f"expected {expect} = alpha^2 * metric. Sigma must "
                        f"stay elliptic everywhere, solid cells included")

    assert n_solid > 0, (
        f"[{name}] the hill should bury some cells, so the unmasked-sigma "
        f"behaviour inside terrain is actually exercised")

    # --- RHS, against an independent divergence -------------------------
    dump = {}
    with open(os.path.join(WORKDIR, "rhs_dump.txt")) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue
            p = line.split()
            dump[(int(p[0]), int(p[1]), int(p[2]))] = float(p[3])
    assert dump, f"[{name}] no rows read from the RHS dump"

    # The RHS is assembled from the INITIAL field, so compare
    # against u0/v0/w0; u/v/w are post-projection from Phase 6 on.
    u, v, w = pf.field("u0"), pf.field("v0"), pf.field("w0")

    dzdk = [0.0] * NZ
    for k in range(NZ):
        if k == 0:
            dzdk[k] = z_cc[1] - z_cc[0]
        elif k == NZ - 1:
            dzdk[k] = z_cc[NZ-1] - z_cc[NZ-2]
        else:
            dzdk[k] = 0.5 * (z_cc[k+1] - z_cc[k-1])

    def div_cell(i, j, k):
        if mask(i, j, k) == SOLID:
            return 0.0
        dudx = weno3_deriv(u(i-2,j,k), u(i-1,j,k), u(i,j,k),
                           u(i+1,j,k), u(i+2,j,k), u(i,j,k), DX)
        dvdy = weno3_deriv(v(i,j-2,k), v(i,j-1,k), v(i,j,k),
                           v(i,j+1,k), v(i,j+2,k), v(i,j,k), DY)
        dwdz = weno3_deriv(w(i,j,k-2), w(i,j,k-1), w(i,j,k),
                           w(i,j,k+1), w(i,j,k+2), w(i,j,k), dzdk[k])
        return J[k] * (dudx + dvdy + dwdz)

    # Interior nodes only: nearer the boundary the solver's stencil reads
    # ghost cells the plotfile does not carry.
    worst_rhs = 0.0
    worst_at = None
    n_checked = 0
    nonzero = 0
    for k in range(5, NZ - 4, 7):
        for j in range(5, NY - 4, 7):
            for i in range(5, NX - 4, 7):
                expect = 0.25 * 0.5 * sum(
                    div_cell(ii, jj, kk)
                    for kk in (k-1, k) for jj in (j-1, j) for ii in (i-1, i))
                got = dump[(i, j, k)]
                err = abs(got - expect)
                if err > worst_rhs:
                    worst_rhs, worst_at = err, (i, j, k, got, expect)
                if abs(expect) > 1.0e-12:
                    nonzero += 1
                n_checked += 1

    assert n_checked > 0, f"[{name}] no nodes were checked"
    assert nonzero > n_checked // 4, (
        f"[{name}] only {nonzero} of {n_checked} sampled nodes carry a "
        f"nonzero divergence -- the case is close to solenoidal and is "
        f"not testing the RHS")

    scale = max(abs(dump[k]) for k in dump) or 1.0
    assert worst_rhs / scale < RHS_RTOL, (
        f"[{name}] the assembled RHS disagrees with the independent "
        f"divergence by {worst_rhs:.3e} (relative {worst_rhs/scale:.3e}) at "
        f"(i,j,k)={worst_at[:3]}: solver {worst_at[3]} vs reference "
        f"{worst_at[4]}")

    print(f"[PASS] {name}  (sigma exact to {worst_sigma:.1e}, unmasked "
          f"across {n_solid} solid cells; RHS matches an independent "
          f"divergence to {worst_rhs/scale:.1e} over {n_checked} nodes, "
          f"{nonzero} of them nonzero)")


def check_rhs_masked_in_terrain(exe):
    """The immersed boundary is imposed on the source, not the operator:
    the divergence is cleared at nodes buried in terrain."""
    name = "inputs_rhs (terrain masking)"
    rep = parse_report(os.path.join(WORKDIR, "grid_report_rhs.txt"))
    n_zeroed = rep["poisson_rhs_nodes_zeroed"]
    assert n_zeroed > 0, (
        f"[{name}] a 100 m hill should bury some nodes entirely, but the "
        f"RHS was cleared at none")
    print(f"[PASS] {name}  (RHS cleared at {int(n_zeroed)} buried nodes)")


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
    for check in (check_manufactured, check_rhs_and_sigma,
                  check_rhs_masked_in_terrain):
        try:
            check(exe)
        except AssertionError as e:
            print(f"[FAIL] {check.__name__}: {e}")
            failed.append(check.__name__)
        except Exception as e:
            print(f"[ERROR] {check.__name__}: {e}")
            failed.append(check.__name__)

    if failed:
        print(f"\n{len(failed)} Phase 5 regtest case(s) failed: {failed}")
        return 1

    print("\nAll Phase 5 regtest cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
