#!/usr/bin/env python3
"""
rotation_test.py -- is the solver equivariant under the square's symmetries?

Stage 1 of the rotation study, and the exact half of it. Rotations by 90
degrees and reflections in the axes map a Cartesian grid onto itself, so
there is no interpolation anywhere in this test and the answer should be
exact to round-off -- the same standard the direction oddness met at
8.6e-16. Continuous angles are a separate and much softer question.

WHY IT SHOULD HOLD. The transmissivity is anisotropic only between
horizontal and vertical; in the horizontal plane it is isotropic. So the
Poisson operator has no preferred compass direction, the terrain-following
inflow rotates with the wind, and there is no Coriolis term to break
chirality. Rotating the terrain and the wind together should rotate the
answer.

WHY IT MIGHT NOT. Everything that is axis-aligned in the implementation
rather than in the physics: which faces are inflow, how the domain is cut
into boxes, the order MLMG traverses them. A failure here is a code
finding, not a physics one, and worth having either way.

WHAT IT BUYS IF IT HOLDS. The eight symmetries of the square are exact
augmentation -- eight terrain fields the network has never seen, from one
solve, with no interpolation error. Combined with the direction oddness
already exploited, sixteen. That is a training-efficiency result that
costs no solver time at all.

Two numbers are reported per operation, and the split matters:

    terrain   how well the ROTATED TERRAIN reproduces the rotation of the
              original. This is sampling, not physics: if the tile's point
              grid is not itself symmetric, the two solves see slightly
              different ground and the velocity comparison inherits it.
    velocity  the equivariance error proper, once the terrain agrees.

Usage:

    python3 cases/rotation_test.py
    python3 cases/rotation_test.py --window carr_fire:12 --op rot90
"""

import argparse
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import corpus                                               # noqa: E402

#: The symmetries tested, as (name, angle in degrees, mirror-in-x).
#: The identity is included deliberately: it costs one solve and proves
#: the harness reports zero when nothing was done, which is the control
#: for every other row.
OPS = [
    ("identity", 0, False),
    ("rot90", 90, False),
    ("rot180", 180, False),
    ("rot270", 270, False),
    ("mirror_x", 0, True),
]


def transform_points(points, L, angle_deg, mirror):
    """Move terrain points by a symmetry of the square [0, L]^2.

    The rotation is counter-clockwise in (x, y). Both operations map the
    square exactly onto itself, so the point SET is unchanged and only its
    labelling moves -- which is what makes this test exact.
    """
    import numpy as np

    p = np.asarray(points, dtype=np.float64).copy()
    x, y = p[:, 0].copy(), p[:, 1].copy()
    if mirror:
        x = L - x
    a = int(round(angle_deg)) % 360
    for _ in range(a // 90):
        x, y = L - y, x
    p[:, 0], p[:, 1] = x, y
    return p


def transform_field(f, angle_deg, mirror):
    """The same symmetry applied to a scalar field indexed ``[..., j, i]``.

    ``i`` runs with x and ``j`` with y. A counter-clockwise rotation by 90
    degrees sends the value at (x, y) to (L - y, x), which in indices is
    ``G[j, i] = F[N - 1 - i, j]``.
    """
    import numpy as np

    g = np.asarray(f)
    if mirror:
        g = g[..., ::-1]
    for _ in range(int(round(angle_deg)) % 360 // 90):
        g = np.swapaxes(g[..., ::-1, :], -2, -1)
    return g


def transform_vector(u, v, angle_deg, mirror):
    """Rotate the horizontal components to match ``transform_field``."""
    import numpy as np

    uu = transform_field(u, angle_deg, mirror)
    vv = transform_field(v, angle_deg, mirror)
    if mirror:
        uu = -uu
    t = math.radians(angle_deg)
    c, s = math.cos(t), math.sin(t)
    return np.asarray(uu) * c - np.asarray(vv) * s, \
        np.asarray(uu) * s + np.asarray(vv) * c


def direction_after(direction_deg, angle_deg, mirror):
    """The wind direction that goes with a transformed terrain.

    Meteorological direction is measured clockwise from north while the
    rotation is counter-clockwise, so the angle SUBTRACTS. A mirror in x
    negates it. Getting this backwards is the most likely way to get a
    large error from a correct solver, so it is derived once here rather
    than inline.
    """
    d = -direction_deg if mirror else direction_deg
    return (d - angle_deg) % 360.0


def main(argv=None):
    import numpy as np
    import fastwindterrain as fwt

    p = argparse.ArgumentParser()
    p.add_argument("--window", action="append", default=None, metavar="ID")
    p.add_argument("--direction", type=float, default=45.0)
    p.add_argument("--op", action="append", default=None,
                   choices=[o[0] for o in OPS])
    args = p.parse_args(argv)

    man = corpus.load_manifest()
    windows = args.window or ["carr_fire:12", "carr_fire:00"]
    missing = [w for w in windows
               if not any(e["id"] == w for e in man["windows"])]
    if missing:
        raise SystemExit(f"not in the manifest: {missing}")
    ops = ([o for o in OPS if o[0] in args.op] if args.op else OPS)

    print(f"direction {args.direction:.0f} deg, "
          f"{corpus.N_PROJECTIONS} projection passes")
    print("errors are max|difference| against the transformed reference\n")

    with fwt.session():
        for wid in windows:
            cfg = corpus.window_config(man, wid,
                                       wind_direction=args.direction)
            pts = np.asarray(cfg["terrain"]["points"], dtype=np.float64)
            L = corpus.WINDOW_M

            t0 = time.time()
            s = fwt.Solver(cfg)
            s.setup()
            s.solve()
            u0, v0, w0 = (np.asarray(a) for a in s.velocity)
            zt0 = np.asarray(s.z_terrain)[0]
            scale = float(np.sqrt(u0 * u0 + v0 * v0).max())
            print(f"{wid}   |U|max {scale:.2f} m/s   "
                  f"({time.time()-t0:.0f} s)")
            print(f"  {'operation':10s} {'direction':>9s} {'terrain':>11s} "
                  f"{'velocity':>11s} {'relative':>10s}")

            for name, ang, mir in ops:
                d2 = direction_after(args.direction, ang, mir)
                cfg2 = corpus.window_config(man, wid, wind_direction=d2)
                cfg2["terrain"] = {
                    "points": transform_points(pts, L, ang, mir)}
                s2 = fwt.Solver(cfg2)
                s2.setup()
                s2.solve()
                u2, v2, w2 = (np.asarray(a) for a in s2.velocity)
                zt2 = np.asarray(s2.z_terrain)[0]

                # Reference: the ORIGINAL solution, transformed.
                ur, vr = transform_vector(u0, v0, ang, mir)
                wr = transform_field(w0, ang, mir)
                ztr = transform_field(zt0, ang, mir)

                e_t = float(np.abs(zt2 - ztr).max())
                e_v = float(np.abs(np.stack([u2 - ur, v2 - vr,
                                             w2 - wr])).max())
                print(f"  {name:10s} {d2:9.0f} {e_t:11.3e} {e_v:11.3e} "
                      f"{e_v / scale:10.2e}")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
