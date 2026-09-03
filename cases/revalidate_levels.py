#!/usr/bin/env python3
"""
revalidate_levels.py -- do the phase 19 level results still hold?

Phase 19 chose the level set the surrogate will predict, and measured what
reconstructing 3D from it costs. Every one of those numbers was taken with
NOTHING CONSTRAINING THE FIRST FLUID CELL above terrain, so w there was the
zero the profile left. The surface condition now puts the kinematic value
there by construction, which changes w everywhere the projection carries it.

Nothing may be generated from those results until they are re-measured
against the operator that is actually going to run. Specifically:

  1. does the recommended placement still beat uniform and
     engineering-anchored at fixed level count?
  2. what is the reconstruction ceiling now, and is the near-surface band
     still the worst one?
  3. does interpolating w still beat deriving it from continuity, and by
     how much?

Run on CORPUS WINDOWS rather than the eight catalogue cases, and with the
top level scaled to each window's own column -- phase 19's fixed 1600 m top
was tuned on Creek at 1128 m of relief and leaves a tall column
unconstrained above it.

Usage:

    python3 cases/revalidate_levels.py
    python3 cases/revalidate_levels.py --window ditch_fire:20 --k 8
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

# A spread across the corpus's relief range rather than a hand-picked few:
# the placement recipe has to hold on gentle and steep ground alike, and
# phase 19 only ever saw three cases.
DEFAULT_WINDOWS = [
    "bootleg_fire:22",       #   99 m relief -- the gentle end
    "marshall_fire:01",      #  138 m
    "carr_fire:12",          #  312 m
    "delta_fire:20",         #  520 m
    "black_summer_fire:10",  #  765 m
    "apple_fire:10",         #  933 m
    "slinkard_fire:22",      # 1070 m
    "ditch_fire:20",         # 1850 m -- the steep end, past anything
                             #           phase 19 measured
]

BANDS = [(0, 50), (50, 200), (200, 500), (500, 1000), (1000, 99999)]


def main(argv=None):
    import numpy as np
    from fastwindterrain import levels as L
    import fastwindterrain as fwt

    p = argparse.ArgumentParser()
    p.add_argument("--window", action="append", default=None, metavar="ID")
    p.add_argument("--k", type=int, default=8,
                   help="levels per set for the placement comparison")
    p.add_argument("--direction", type=float, default=225.0)
    p.add_argument("--surface", default="wall_function",
                   help="surface.type to measure against")
    p.add_argument("--low", type=float, default=None, metavar="Z",
                   help="also test adding a level at Z metres AGL")
    args = p.parse_args(argv)

    windows = args.window or DEFAULT_WINDOWS
    m = corpus.load_manifest()

    def rmse(a, b, sel, scale):
        return float(np.sqrt(((a - b)[sel] ** 2).mean())) / scale

    print(f"surface.type = {args.surface}, {args.direction:.0f} deg, "
          f"{corpus.N_PROJECTIONS} passes, k = {args.k}\n")

    placement_wins = {}
    with fwt.session():
        for wid in windows:
            t0 = time.time()
            cfg = corpus.window_config(m, wid,
                                       wind_direction=args.direction)
            cfg["surface"] = {"type": args.surface}
            s = fwt.Solver(cfg)
            s.setup()
            s.solve()

            u, v, w = s.velocity
            z_cc = np.asarray(s.grid.z_cc)
            zt = np.asarray(s.z_terrain)
            mask = np.asarray(s.mask)
            fluid = mask == 0
            agl = L.height_above_ground(z_cc, zt)
            top = float(agl[fluid].max())
            scale = float(np.sqrt(u * u + v * v)[fluid].max())
            relief = float(np.asarray(zt)[0].max() - np.asarray(zt)[0].min())

            dxc = dyc = corpus.WINDOW_M / u.shape[2]

            def recon(field, lv, perp=True):
                """Reconstruct. perp=False reproduces the pre-fix path,
                which filled below the lowest level using the VERTICAL gap
                instead of the perpendicular distance the solver's surface
                condition uses."""
                kw = dict(dx=dxc, dy=dyc) if perp else {}
                vals = L.extract_levels(field, z_cc, zt, lv, mask=mask,
                                        frame="agl", **kw)
                return L.stitch_levels(vals, lv, z_cc, zt, mask=mask,
                                       frame="agl", **kw)

            # -- 1. placement at fixed k, with the top scaled to THIS column
            sets = {
                "recommended": L.recommended_levels(top, n_band=5,
                                                    n_aloft=args.k - 5),
                "uniform": tuple(np.linspace(10.0, top, args.k)),
                "log": tuple(np.geomspace(10.0, top, args.k)),
                "engineering-anchored": tuple(
                    list(L.ENGINEERING_LEVELS)
                    + list(np.geomspace(L.ENGINEERING_LEVELS[-1] * 1.6, top,
                                        args.k - len(L.ENGINEERING_LEVELS)))),
            }

            # The engineering band, 10-160 m AGL: where the answer is
            # actually wanted, and what phase 19's headline placement claim
            # was about. A column RMSE cannot confirm or refute it -- the
            # band is a few percent of the cells.
            band = fluid & (agl >= L.ENGINEERING_LEVELS[0]) \
                         & (agl <= L.ENGINEERING_LEVELS[-1])

            print(f"=== {wid}   relief {relief:.0f} m, column top "
                  f"{top:.0f} m AGL, {100*(mask==1).mean():.0f}% solid, "
                  f"|U_h|max {scale:.2f} m/s ===")
            print(f"    {'placement':22s} {'column':>9s} {'col m/s':>9s} "
                  f"{'10-160 m':>10s} {'band m/s':>9s}")
            best, best_e = None, 1e9
            best_band, best_band_e = None, 1e9
            for name, lv in sets.items():
                ub, vb = recon(u, lv), recon(v, lv)
                sq = (ub - u) ** 2 + (vb - v) ** 2
                e = float(np.sqrt(sq[fluid].mean())) / scale
                eb = float(np.sqrt(sq[band].mean())) / scale
                if e < best_e:
                    best, best_e = name, e
                if eb < best_band_e:
                    best_band, best_band_e = name, eb
                print(f"    {name:22s} {e:9.5f} {e*scale:9.3f} "
                      f"{eb:10.5f} {eb*scale:9.3f}")
            placement_wins[wid] = (best, best_band)
            print(f"    -> best on column: {best};  best in band: {best_band}")

            # A LEVEL BELOW THE LOWEST ONE.
            #
            # Everything under levels[0] is filled by extrapolating a log
            # law down from it, so the reconstruction has no information at
            # all in the region where the error is largest. Two variants,
            # because they answer different questions:
            #
            #   added   k+1 levels -- does a low level help AT ALL?
            #   swapped k levels, the top aloft one given up for it -- is
            #           it worth the budget it costs?
            if args.low is not None:
                base = list(sets["recommended"])
                added = tuple([args.low] + base)
                swapped = tuple([args.low] + base[:-1])
                near = fluid & (agl < 50.0)
                print(f"    {'level set':22s} {'0-50 m':>9s} {'m/s':>8s} "
                      f"{'column':>9s} {'m/s':>8s}  k")
                for nm, ls in (("recommended", tuple(base)),
                               (f"+{args.low:g} m (k+1)", added),
                               (f"+{args.low:g} m (same k)", swapped)):
                    ua, va = recon(u, ls), recon(v, ls)
                    sq2 = (ua - u) ** 2 + (va - v) ** 2
                    en = float(np.sqrt(sq2[near].mean())) / scale
                    ec = float(np.sqrt(sq2[fluid].mean())) / scale
                    print(f"    {nm:22s} {en:9.5f} {en*scale:8.3f} "
                          f"{ec:9.5f} {ec*scale:8.3f}  {len(ls)}")

            # -- 2. bands, and -- 3. w two ways, on the recommended set
            lv = sets["recommended"]
            ub, vb, wb = recon(u, lv), recon(v, lv), recon(w, lv)
            dz = np.gradient(z_cc)
            w_ob = L.obrien_w(ub, vb, np.zeros_like(w), dz, dxc, dyc, mask)

            # The near-surface band with and without the perpendicular
            # distance, since that is the fix under test.
            near = fluid & (agl < 50.0)
            uo, vo = recon(u, lv, perp=False), recon(v, lv, perp=False)
            e_perp = float(np.sqrt((((ub - u) ** 2
                                     + (vb - v) ** 2)[near]).mean())) / scale
            e_vert = float(np.sqrt((((uo - u) ** 2
                                     + (vo - v) ** 2)[near]).mean())) / scale
            print(f"    0-50 m fill: perpendicular {e_perp:.5f} "
                  f"({e_perp*scale:.3f} m/s)   vertical {e_vert:.5f} "
                  f"({e_vert*scale:.3f} m/s)   "
                  f"{e_vert/max(e_perp,1e-12):.2f}x")

            e_i = rmse(wb, w, fluid, scale)
            e_o = rmse(w_ob, w, fluid, scale)
            print(f"    w: interpolated {e_i:.5f} ({e_i*scale:.3f} m/s)   "
                  f"O'Brien {e_o:.5f} ({e_o*scale:.3f} m/s)   "
                  f"ratio {e_o / max(e_i, 1e-12):.1f}x")
            worst_band, worst_e = None, -1.0
            for lo, hi in BANDS:
                sel = fluid & (agl >= lo) & (agl < hi)
                if sel.sum() < 50:
                    continue
                e = float(np.sqrt(((ub - u) ** 2
                                   + (vb - v) ** 2)[sel].mean())) / scale
                if e > worst_e:
                    worst_band, worst_e = f"{lo}-{hi}", e
                print(f"      {lo:5d}-{'inf' if hi > 9999 else hi:<6} m "
                      f"horiz {e:8.5f}  ({e*scale:6.3f} m/s)")
            print(f"    -> worst band: {worst_band} at {worst_e:.5f} "
                  f"({worst_e*scale:.3f} m/s)   "
                  f"({time.time() - t0:.0f} s)\n")

    print(f"{'window':24s} {'column winner':22s} band winner")
    for wid, (a, b) in placement_wins.items():
        print(f"  {wid:24s} {a:22s} {b}")
    n_col = sum(1 for a, _ in placement_wins.values() if a == "recommended")
    n_bnd = sum(1 for _, b in placement_wins.values() if b == "recommended")
    n = len(placement_wins)
    print(f"\nrecommended wins the column on {n_col} of {n}, "
          f"the 10-160 m band on {n_bnd} of {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
