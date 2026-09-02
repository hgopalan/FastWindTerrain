#!/usr/bin/env python3
"""
warmstart_study.py -- what is a surrogate worth as an initial condition?

If the surrogate's job is not to replace the solver but to hand it a good
starting field, then RMSE against the truth is the wrong figure of merit.
What matters is ITERATIONS SAVED: how many projection passes the solver
still needs when it starts from a reconstruction instead of from its usual
initial condition.

That is measurable with no machine learning at all. The projection is a
stationary linear iteration -- it converges to the same fixed point from
anywhere, at a rate this solver measures at about 0.87 per pass -- so
seeding it from a better field and counting passes to a fixed target is a
direct measurement of the value a surrogate would add.

A perfect surrogate is not the interesting case. This sweeps DEGRADED
reconstructions, adding relative noise to the level values before
stitching, because a real network will have a few per cent error and the
question is how much of the saving survives it.

    python3 cases/warmstart_study.py --case creek_fire

CAVEAT WORTH KEEPING. The target is a fractional-step solver, whose
iteration is not this one -- it is nonlinear and time-stepping, and warm
starting it is a different question. This measures the mass-consistent
projection, which is the machinery available today, and the mechanism it
demonstrates (a geometric iteration started closer to its fixed point) is
general even where the numbers are not.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import casegen                              # noqa: E402
import fastwindterrain as fwt               # noqa: E402
from fastwindterrain import levels as lv    # noqa: E402

WIND_SPEED, WIND_DIRECTION = 8.0, 225.0
REFERENCE_PASSES = 24        # the field the iteration is heading toward
TARGET_PASS = 12             # cold-start pass whose divergence is the target
MAX_PASSES = 24
NOISE = (0.0, 0.02, 0.05, 0.10, 0.20)


def geometry(solver):
    g = solver.grid
    return (np.asarray(g.z_cc), solver.z_terrain, solver.mask,
            np.diff(np.asarray(g.z_face)),
            (g.prob_hi[0] - g.prob_lo[0]) / g.nx,
            (g.prob_hi[1] - g.prob_lo[1]) / g.ny)


def reconstruct(field, geom, levels, noise=0.0, rng=None):
    """Stitch a field back from `levels`, optionally degrading it first.

    Noise is added to the LEVEL VALUES, not to the 3D field: that is where
    a surrogate's error lives, and putting it there means it propagates
    through the stitching exactly as a real prediction error would.
    """
    z_cc, zt, mask, dz, dx, dy = geom
    scale = float(np.sqrt((field ** 2).sum(axis=0))[mask == 0].max())
    out = []
    for c in range(2):
        k = lv.extract_levels(field[c], z_cc, zt, list(levels), mask=mask)
        if noise:
            k = k + rng.normal(0.0, noise * scale, size=k.shape)
        out.append(lv.stitch_levels(k, list(levels), z_cc, zt, mask=mask))
    u, v = out
    seed = np.zeros_like(u)
    w = lv.obrien_w(u, v, seed, dz, dx, dy, mask)
    return np.stack([u, v, w])


def passes_to(solver, target, cap=MAX_PASSES):
    """Projection passes until max|div| drops to `target`."""
    for n in range(1, cap + 1):
        solver.project_once()
        if float(solver.max_divergence_fe) <= target:
            return n, float(solver.max_divergence_fe)
    return None, float(solver.max_divergence_fe)


def run_case(case, levels, stream=sys.stdout):
    cfg = case.config(wind_speed=WIND_SPEED, wind_direction=WIND_DIRECTION,
                      n_projections=1)

    # The cold-start trajectory, and the target taken from it.
    cold = fwt.Solver(cfg)
    cold.setup()
    geom = geometry(cold)
    traj = []
    for _ in range(REFERENCE_PASSES):
        cold.project_once()
        traj.append(float(cold.max_divergence_fe))
    reference = np.stack(cold.velocity)
    target = traj[TARGET_PASS - 1]

    print(f"\n=== {case.name} ===", file=stream)
    print(f"  cold start reaches max|div| {target:.5f} after {TARGET_PASS} "
          f"passes  (24-pass value {traj[-1]:.5f})", file=stream)
    print(f"  {len(levels)} levels; noise is added to the level values",
          file=stream)
    print(f"\n  {'start':<26} {'passes to target':>16} {'saved':>8} "
          f"{'|u-u_ref| / |u|max':>20}", file=stream)
    print(f"  {'cold (solver default)':<26} {TARGET_PASS:>16d} "
          f"{'--':>8} {_dist(np.stack(fresh(cfg).velocity), reference):>20.4f}",
          file=stream)

    rng = np.random.default_rng(0)
    rows = []
    for eps in NOISE:
        s = fwt.Solver(cfg)
        s.setup()
        warm = reconstruct(reference, geom, levels, noise=eps, rng=rng)
        s.set_velocity(warm)
        d0 = _dist(np.stack(s.velocity), reference)
        n, div = passes_to(s, target)
        saved = None if n is None else TARGET_PASS - n
        rows.append(dict(noise=eps, passes=n, saved=saved, dist=d0))
        label = f"warm, {eps*100:.0f}% level noise"
        got = f"{n:>16d}" if n is not None else f"{'>%d' % MAX_PASSES:>16}"
        sv = f"{saved:>8d}" if saved is not None else f"{'--':>8}"
        print(f"  {label:<26} {got} {sv} {d0:>20.4f}", file=stream)

    return traj, rows, target


def fresh(cfg):
    s = fwt.Solver(cfg)
    s.setup()
    return s


def _dist(field, reference):
    return float(np.abs(field - reference).max()
                 / np.abs(reference).max())


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--case", action="append", default=None, metavar="SLUG")
    p.add_argument("--levels", type=int, default=8, choices=(5, 8, 12),
                   help="how many levels the reconstruction uses")
    args = p.parse_args()

    level_sets = {5: lv.ENGINEERING_LEVELS, 8: lv.DEFAULT_LEVELS,
                  12: (10.0, 30.0, 60.0, 80.0, 100.0, 120.0, 160.0,
                       250.0, 400.0, 700.0, 1100.0, 1600.0)}
    levels = level_sets[args.levels]

    wanted = args.case or ["creek_fire"]
    with fwt.session():
        for slug in wanted:
            c = casegen.load(slug)
            if not os.path.isfile(c.terrain_path):
                print(f"[skip] {slug}: no terrain", file=sys.stderr)
                continue
            run_case(c, levels)
    return 0


if __name__ == "__main__":
    sys.exit(main())
