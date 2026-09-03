#!/usr/bin/env python3
"""
direction_check.py -- do the level results hold at every wind direction?

Everything in cases/revalidate_levels.py was measured at 225 degrees. The
dataset will carry eight directions, and direction is not a free parameter
over terrain: the corpus already showed the projection's convergence
depending on it, because the boundary flux imbalance is set by the terrain
slope ALONG the wind. So the placement recipe and the choice to interpolate
w rather than derive it both have to be checked across the rose before a
dataset is generated on them.

Three questions, one per column of the output:

    placement   does the recommended set still beat uniform and
                engineering-anchored inside 10-160 m, at every direction?
    w           does interpolating w still beat deriving it from
                continuity, and by how much?
    ceiling     how much does the reconstruction error itself move with
                direction -- i.e. is a single quoted ceiling meaningful?

One window per invocation so the eight solves for a window can run in
their own process; the windows are independent.

Usage:

    python3 cases/direction_check.py --window ditch_fire:20
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import corpus                                               # noqa: E402


def main(argv=None):
    import numpy as np
    from fastwindterrain import levels as L
    import fastwindterrain as fwt

    p = argparse.ArgumentParser()
    p.add_argument("--window", required=True, metavar="ID")
    p.add_argument("--k", type=int, default=8)
    args = p.parse_args(argv)

    m = corpus.load_manifest()
    rows = []

    with fwt.session():
        for d in corpus.WIND_DIRECTIONS:
            t0 = time.time()
            cfg = corpus.window_config(m, args.window, wind_direction=float(d))
            s = fwt.Solver(cfg)
            s.setup(); s.solve()

            u, v, w = s.velocity
            z_cc = np.asarray(s.grid.z_cc)
            zt = np.asarray(s.z_terrain)[0]
            mask = np.asarray(s.mask)
            fluid = mask == 0
            agl = L.height_above_ground(z_cc, zt)
            dx = dy = corpus.WINDOW_M / u.shape[2]
            scale = float(np.sqrt(u * u + v * v)[fluid].max())
            top = float(agl[fluid].max())
            band = fluid & (agl >= L.ENGINEERING_LEVELS[0]) \
                         & (agl <= L.ENGINEERING_LEVELS[-1])

            def recon(f, lv):
                vals = L.extract_levels(f, z_cc, zt, lv, mask=mask,
                                        frame="agl", dx=dx, dy=dy)
                return L.stitch_levels(vals, lv, z_cc, zt, mask=mask,
                                       frame="agl", dx=dx, dy=dy)

            sets = {
                "recommended": L.recommended_levels(top, n_band=5,
                                                    n_aloft=args.k - 5),
                "uniform": tuple(np.linspace(10.0, top, args.k)),
                "engineering": tuple(
                    list(L.ENGINEERING_LEVELS)
                    + list(np.geomspace(L.ENGINEERING_LEVELS[-1] * 1.6, top,
                                        args.k - len(L.ENGINEERING_LEVELS)))),
            }
            band_err = {}
            for name, lv in sets.items():
                ub, vb = recon(u, lv), recon(v, lv)
                sq = (ub - u) ** 2 + (vb - v) ** 2
                band_err[name] = float(np.sqrt(sq[band].mean())) / scale
                if name == "recommended":
                    col = float(np.sqrt(sq[fluid].mean())) / scale
                    ub_r, vb_r = ub, vb

            wb = recon(w, sets["recommended"])
            w_ob = L.obrien_w(ub_r, vb_r, np.zeros_like(w),
                              np.gradient(z_cc), dx, dy, mask)
            e_i = float(np.sqrt(((wb - w) ** 2)[fluid].mean())) / scale
            e_o = float(np.sqrt(((w_ob - w) ** 2)[fluid].mean())) / scale

            best = min(band_err, key=band_err.get)
            rows.append((d, scale, col, band_err, best, e_i, e_o,
                         time.time() - t0))

    print(f"\n=== {args.window} ===")
    print(f"{'dir':>5s} {'|U_h|max':>9s} {'ceiling':>9s} {'m/s':>7s}  "
          f"{'band: rec':>10s} {'unif':>9s} {'eng':>9s}  {'best':>12s}  "
          f"{'w i':>8s} {'w O.B':>8s} {'ratio':>6s}")
    for d, scale, col, be, best, e_i, e_o, dt in rows:
        print(f"{d:5.0f} {scale:9.2f} {col:9.5f} {col*scale:7.3f}  "
              f"{be['recommended']:10.5f} {be['uniform']:9.5f} "
              f"{be['engineering']:9.5f}  {best:>12s}  "
              f"{e_i:8.5f} {e_o:8.5f} {e_o/max(e_i,1e-12):5.1f}x")

    cols = [r[2] for r in rows]
    ratios = [r[6] / max(r[5], 1e-12) for r in rows]
    n_rec = sum(1 for r in rows if r[4] == "recommended")
    print(f"\n  ceiling across directions: {min(cols):.5f} to {max(cols):.5f}"
          f"  ({max(cols)/max(min(cols),1e-12):.2f}x spread)")
    print(f"  recommended wins the band at {n_rec} of {len(rows)} directions")
    print(f"  interpolated w beats O'Brien by {min(ratios):.1f}x to "
          f"{max(ratios):.1f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
