#!/usr/bin/env python3
"""
linearity_study.py -- does wind SPEED need to be a dataset axis at all?

The corpus is many terrain shapes crossed with a handful of wind
directions. Whether speed belongs in that product decides the dataset size
directly: two speeds is twice the compute, and at ~52 s a solve over a few
hundred windows that is days, not minutes.

There is an argument that it should not be. Every operator in the pipeline
looks linear in the inflow magnitude with the terrain held fixed:

    inflow      the power law scales with u_ref
    O'Brien     w comes from a column integral of div_h(u, v)
    anisotropy  alpha depends on terrain SLOPE, not on the flow
    Poisson     nabla . (alpha^2 grad lambda) = div(u0), and the correction
                is u = u0 - alpha^2 grad lambda

so doubling the inflow should double every field exactly. If that holds,
speed is a free normalisation, the surrogate can predict a field per unit
reference speed, and the dataset only ever has to vary direction.

That is an argument, not a measurement, and the whole point of having a
solver is not to have to argue. This script measures it: solve the same
window at several speeds, scale each result back, and report the largest
disagreement. What it CANNOT establish is that the argument stays true --
f_Ri and f_Fr are hooks in Anisotropy that currently return 1, and a
stability or Froude correction would put the speed back in. The check is
therefore worth re-running whenever those become real, and says so.

Usage:

    python3 cases/linearity_study.py
    python3 cases/linearity_study.py --window creek_fire:11 --speeds 2 8 30
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import corpus                                               # noqa: E402

REFERENCE_SPEED = corpus.REFERENCE_SPEED_MS
SPEEDS = (2.0, 4.0, 16.0, 30.0)
DIRECTION = 225.0


def relative_error(a, b, fluid):
    """max |a - b| over the fluid, as a fraction of the reference scale."""
    import numpy as np

    scale = float(np.abs(b[fluid]).max())
    if scale == 0.0:
        return 0.0
    return float(np.abs(a[fluid] - b[fluid]).max() / scale)


def main(argv=None):
    import numpy as np

    p = argparse.ArgumentParser()
    p.add_argument("--window", default=None,
                   help="window id (default: the first in the manifest)")
    p.add_argument("--speeds", type=float, nargs="+", default=list(SPEEDS))
    p.add_argument("--direction", type=float, default=DIRECTION)
    p.add_argument("--n-projections", type=int, default=4)
    args = p.parse_args(argv)

    import fastwindterrain as fwt

    manifest = corpus.load_manifest()
    wid = args.window or manifest["windows"][0]["id"]

    def solve(speed):
        cfg = corpus.window_config(manifest, wid, wind_speed=speed,
                                   wind_direction=args.direction,
                                   poisson={"n_projections":
                                            args.n_projections})
        s = fwt.Solver(cfg)
        s.setup()
        s.solve()
        return [np.array(f) for f in s.velocity], s.mask == 0

    print(f"window {wid}, {args.direction:.0f} deg, "
          f"{args.n_projections} projections")
    print(f"reference speed {REFERENCE_SPEED:.0f} m/s\n")

    with fwt.session():
        ref, fluid = solve(REFERENCE_SPEED)
        print(f"{'speed':>8s}  {'u':>12s}  {'v':>12s}  {'w':>12s}")
        worst = 0.0
        for speed in args.speeds:
            got, _ = solve(speed)
            k = speed / REFERENCE_SPEED
            errs = [relative_error(g, k * r, fluid)
                    for g, r in zip(got, ref)]
            worst = max(worst, max(errs))
            print(f"{speed:8.1f}  " + "  ".join(f"{e:12.3e}" for e in errs))

    print(f"\nlargest relative disagreement with exact scaling: {worst:.3e}")
    if worst < 1e-10:
        print("\nThe solve is linear in the reference speed to round-off, so\n"
              "speed is a free normalisation and does NOT belong in the\n"
              "dataset: vary direction, solve once per direction, and scale.\n"
              "Re-run this if Anisotropy's f_Ri or f_Fr hooks stop returning\n"
              "1 -- a stability or Froude correction puts the speed back in.")
    else:
        print("\nNOT linear to round-off. Speed is a real dataset axis after\n"
              "all; find out what broke the argument before sizing anything.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
