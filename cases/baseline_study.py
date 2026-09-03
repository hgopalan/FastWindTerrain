#!/usr/bin/env python3
"""
baseline_study.py -- what does a surrogate have to beat?

"Compared to what?" is the question that decides whether a neural operator
was worth building, and it is the one easiest to skip. This measures the
cheap analytical fields in fastwindterrain.baseline against the solver's
own, on the same metric and the same windows as everything else, so the
numbers sit beside the reconstruction ceiling rather than in a different
unit.

Three of them, in increasing ambition:

    undisturbed   the inflow profile, terrain-following but terrain-blind.
                  The "do nothing" field. A surrogate that cannot beat this
                  has learned nothing, and it is the denominator any skill
                  score should use.
    continuity    the profile scaled by how much the terrain squeezes the
                  air column -- one line of mass conservation.
    slope         a Jackson-Hunt-shaped speed-up, linear in local slope and
                  decaying with height.

WHAT A RESULT HERE MEANS. A surrogate trained on this solver cannot be more
accurate than the solver; its target IS the solver's field. So the value
proposition is speed, not accuracy, and these baselines say how much of the
terrain effect is available for nothing -- which is how much the surrogate
is really being asked to add.

Usage:

    python3 cases/baseline_study.py
    python3 cases/baseline_study.py --window ditch_fire:20
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import corpus                                               # noqa: E402
from revalidate_levels import DEFAULT_WINDOWS               # noqa: E402


def main(argv=None):
    import numpy as np
    from fastwindterrain import baseline as B
    from fastwindterrain import levels as L
    import fastwindterrain as fwt

    p = argparse.ArgumentParser()
    p.add_argument("--window", action="append", default=None, metavar="ID")
    p.add_argument("--direction", type=float, default=225.0)
    p.add_argument("--decay", type=float, default=200.0,
                   help="slope baseline decay length [m]")
    args = p.parse_args(argv)

    windows = args.window or DEFAULT_WINDOWS
    m = corpus.load_manifest()

    print(f"{args.direction:.0f} deg, {corpus.REFERENCE_SPEED_MS:.0f} m/s at "
          f"{corpus.REFERENCE_HEIGHT_M:.0f} m, "
          f"{corpus.N_PROJECTIONS} passes\n")

    totals = {}
    with fwt.session():
        for wid in windows:
            t0 = time.time()
            cfg = corpus.window_config(m, wid,
                                       wind_direction=args.direction)
            s = fwt.Solver(cfg)
            s.setup(); s.solve()

            u, v, w = s.velocity
            truth = np.stack([u, v, w])
            z_cc = np.asarray(s.grid.z_cc)
            zt = np.asarray(s.z_terrain)[0]
            mask = np.asarray(s.mask)
            fluid = mask == 0
            agl = L.height_above_ground(z_cc, zt)
            dx = dy = corpus.WINDOW_M / u.shape[2]
            scale = float(np.sqrt(u * u + v * v)[fluid].max())
            relief = float(zt.max() - zt.min())

            band = fluid & (agl >= L.ENGINEERING_LEVELS[0]) \
                         & (agl <= L.ENGINEERING_LEVELS[-1])

            common = dict(speed_ref=corpus.REFERENCE_SPEED_MS,
                          direction_deg=args.direction,
                          z_ref=corpus.REFERENCE_HEIGHT_M)
            fields = {
                "undisturbed": B.undisturbed(z_cc, zt, mask, **common),
                "continuity": B.continuity_speedup(z_cc, zt, mask, dx=dx,
                                                   dy=dy, **common),
                f"slope (L={args.decay:.0f})": B.slope_speedup(
                    z_cc, zt, mask, dx=dx, dy=dy,
                    decay_length=args.decay, **common),
            }

            print(f"=== {wid}   relief {relief:.0f} m, "
                  f"{100*(mask==1).mean():.0f}% solid, "
                  f"|U_h|max {scale:.2f} m/s ===")
            print(f"    {'baseline':22s} {'column':>9s} {'m/s':>8s} "
                  f"{'10-160 m':>10s} {'m/s':>8s} {'|U|max':>8s}")
            for name, f in fields.items():
                d = ((f[0] - truth[0]) ** 2 + (f[1] - truth[1]) ** 2)
                e = float(np.sqrt(d[fluid].mean())) / scale
                eb = float(np.sqrt(d[band].mean())) / scale
                fmax = float(np.sqrt(f[0] ** 2 + f[1] ** 2)[fluid].max())
                totals.setdefault(name, []).append(e)
                print(f"    {name:22s} {e:9.4f} {e*scale:8.3f} "
                      f"{eb:10.4f} {eb*scale:8.3f} {fmax:8.2f}")
            print(f"    (solver |U_h|max {scale:.2f} m/s, "
                  f"{time.time() - t0:.0f} s)\n")

    print(f"{'baseline':22s} {'mean column RMSE':>18s}")
    for name, es in totals.items():
        print(f"{name:22s} {float(np.mean(es)):18.4f}")
    print("\nA surrogate has to beat the best of these to be worth its "
          "complexity;\nit cannot beat the solver, whose field is its "
          "training target.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
