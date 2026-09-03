#!/usr/bin/env python3
"""
slope_error.py -- does the reconstruction error follow the terrain slope?

The maps in cases/error_maps.py show error concentrating on steep ground.
That is an eyeball result. This measures it, per level, over the whole
fold, against three descriptors of the ground under each column:

    |grad h|    slope magnitude, terrain-only. The obvious candidate.
    grad h . u  the ALONG-WIND slope: positive on windward faces, negative
                in the lee. This is the physically interesting one --
                windward speed-up and lee separation are different flows,
                and a magnitude-only descriptor averages them together and
                can report no correlation when there is a strong one of
                each sign.
    lap h       curvature. Crests are convex, valley floors concave, and
                a descriptor that separates them says whether the error is
                a slope effect or a shape effect.

PEARSON r MEASURES THE LINEAR PART ONLY. A U-shaped dependence on the
along-wind slope -- error rising on both windward and lee faces -- has
r near zero and is exactly the pattern worth finding, so the binned means
are the primary output here and r is a summary. Do not read r alone.

Usage:

    python3 cases/slope_error.py
    python3 cases/slope_error.py --data data/demo --fold demo
    python3 cases/slope_error.py --what baseline --limit 40
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import corpus                                               # noqa: E402
from error_maps import slab_edges                           # noqa: E402

#: Bin edges for the along-wind slope, dimensionless (rise over run).
#: Symmetric about zero so windward and lee are directly comparable, which
#: is the whole point of using a signed descriptor.
SLOPE_BINS = [-2.0, -0.6, -0.35, -0.2, -0.1, -0.03,
              0.03, 0.1, 0.2, 0.35, 0.6, 2.0]


def terrain_descriptors(zt, dx, dy, direction_deg):
    """Slope magnitude, along-wind slope and curvature, per column.

    The wind vector matches the solver's convention: a direction of 45
    degrees means wind FROM the northeast, so the flow points southwest.
    Getting that backwards would swap windward and lee and invert the
    result, so it is taken from the same expression the inflow uses
    (corpus.window_config).
    """
    import numpy as np

    zt = np.asarray(zt, dtype=np.float64)
    # Central differences, edges one-sided -- np.gradient does both.
    dzdy, dzdx = np.gradient(zt, dy, dx)
    slope = np.sqrt(dzdx ** 2 + dzdy ** 2)

    theta = np.radians(float(direction_deg))
    ux, uy = -np.sin(theta), -np.cos(theta)     # the direction flow GOES
    along = dzdx * ux + dzdy * uy               # >0 climbing, <0 descending

    d2y, _ = np.gradient(dzdy, dy, dx)
    _, d2x = np.gradient(dzdx, dy, dx)
    return slope, along, d2x + d2y


class Pearson:
    """Streaming correlation, so the whole fold never has to be held."""

    def __init__(self):
        self.n = self.sx = self.sy = self.sxx = self.syy = self.sxy = 0.0

    def add(self, x, y):
        import numpy as np
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        self.n += x.size
        self.sx += x.sum()
        self.sy += y.sum()
        self.sxx += (x * x).sum()
        self.syy += (y * y).sum()
        self.sxy += (x * y).sum()

    @property
    def r(self):
        import numpy as np
        n = self.n
        if n < 2:
            return float("nan")
        cov = self.sxy / n - (self.sx / n) * (self.sy / n)
        vx = self.sxx / n - (self.sx / n) ** 2
        vy = self.syy / n - (self.sy / n) ** 2
        if vx <= 0 or vy <= 0:
            return float("nan")
        return float(cov / np.sqrt(vx * vy))


def main(argv=None):
    import numpy as np

    from fastwindterrain import baseline as B
    from fastwindterrain import evaluate as E
    from fastwindterrain import levels as L
    import build_dataset as bd

    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(ROOT, "data", "corpus"))
    p.add_argument("--fold", default="test")
    p.add_argument("--what", default="floor", choices=["floor", "baseline"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=None, metavar="DIR",
                   help="write the figure here (default: no figure)")
    args = p.parse_args(argv)

    corr = None
    binned = None      # (nlev, nbin) sums and counts of error vs along-slope
    lv_ref, n_samples = None, 0

    for info, a in bd.load_dataset(args.data, fold=args.fold, with_3d=True):
        if info["derived"]:
            continue
        if args.limit and n_samples >= args.limit:
            break
        n_samples += 1

        u, v, w = a["u"], a["v"], a["w"]
        z_cc, zt, lv = a["z_cc"], a["terrain"], a["levels"]
        fluid = E.fluid_from_k_first(a["k_first"], u.shape[0])
        solid = (~fluid).astype(np.int32)
        ref = np.stack([u, v, w]).astype(np.float64)
        dx = dy = corpus.WINDOW_M / u.shape[2]

        if args.what == "floor":
            other = np.stack([
                L.stitch_levels(a[k], lv, z_cc, zt, mask=solid, frame="agl",
                                dx=dx, dy=dy)
                for k in ("u_lev", "v_lev", "w_lev")])
        else:
            other = B.undisturbed(z_cc, zt, solid, corpus.REFERENCE_SPEED_MS,
                                  info["direction"],
                                  z_ref=corpus.REFERENCE_HEIGHT_M)

        err = np.sqrt(((other - ref) ** 2).sum(axis=0))
        agl = L.height_above_ground(z_cc, zt)
        slope, along, curv = terrain_descriptors(zt, dx, dy,
                                                 info["direction"])
        lo, hi = slab_edges(lv)

        if corr is None:
            lv_ref = np.asarray(lv, dtype=float)
            corr = {k: [Pearson() for _ in lv_ref]
                    for k in ("slope", "along", "curv")}
            binned = (np.zeros((len(lv_ref), len(SLOPE_BINS) - 1)),
                      np.zeros((len(lv_ref), len(SLOPE_BINS) - 1)))

        for k in range(len(lv_ref)):
            sel = fluid & (agl >= lo[k]) & (agl < hi[k])
            n = sel.sum(axis=0)
            acc = np.where(sel, err ** 2, 0.0).sum(axis=0)
            colerr = np.sqrt(np.divide(acc, n, out=np.zeros_like(acc),
                                       where=n > 0))
            have = n > 0
            if not have.any():
                continue
            e = colerr[have]
            corr["slope"][k].add(slope[have], e)
            corr["along"][k].add(along[have], e)
            corr["curv"][k].add(curv[have], e)

            idx = np.digitize(along[have], SLOPE_BINS) - 1
            ok = (idx >= 0) & (idx < len(SLOPE_BINS) - 1)
            np.add.at(binned[0][k], idx[ok], e[ok])
            np.add.at(binned[1][k], idx[ok], 1.0)

    if corr is None:
        print("no samples matched.", file=sys.stderr)
        return 1

    what = "reconstruction floor" if args.what == "floor" else "baseline"
    print(f"\n{what}, fold '{args.fold}', {n_samples} samples")
    print("Pearson r between per-column error and terrain descriptor\n")
    head = (f"{'level':>8}  {'|grad h|':>9}  {'along-wind':>11}  "
            f"{'curvature':>10}")
    print(head)
    print("-" * len(head))
    for k in range(len(lv_ref)):
        print(f"{lv_ref[k]:>6.0f} m  {corr['slope'][k].r:>9.3f}  "
              f"{corr['along'][k].r:>11.3f}  {corr['curv'][k].r:>10.3f}")

    tot, cnt = binned
    mean = np.divide(tot, cnt, out=np.full_like(tot, np.nan), where=cnt > 0)
    centres = [0.5 * (SLOPE_BINS[i] + SLOPE_BINS[i + 1])
               for i in range(len(SLOPE_BINS) - 1)]

    print("\nmean column error [m/s] by along-wind slope "
          "(negative = lee, positive = windward)\n")
    head = f"{'level':>8}  " + "  ".join(f"{c:>+6.2f}" for c in centres)
    print(head)
    print("-" * len(head))
    for k in range(len(lv_ref)):
        print(f"{lv_ref[k]:>6.0f} m  "
              + "  ".join("   --  " if not np.isfinite(x) else f"{x:>6.3f} "
                          for x in mean[k]).rstrip())

    # The asymmetry, stated as one number per level: is the lee worse than
    # the windward face at the same steepness?
    print("\nlee vs windward, |along-wind slope| > 0.2\n")
    print(f"{'level':>8}  {'lee':>8}  {'windward':>9}  {'lee/wind':>9}")
    print("-" * 40)
    left = [i for i, c in enumerate(centres) if c <= -0.2]
    right = [i for i, c in enumerate(centres) if c >= 0.2]
    for k in range(len(lv_ref)):
        lw = float(tot[k, left].sum() / max(cnt[k, left].sum(), 1))
        wd = float(tot[k, right].sum() / max(cnt[k, right].sum(), 1))
        print(f"{lv_ref[k]:>6.0f} m  {lw:>8.3f}  {wd:>9.3f}  "
              f"{lw / wd if wd > 0 else float('nan'):>9.2f}")

    if args.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(args.out, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9.0, 5.6), constrained_layout=True)
        shades = plt.cm.viridis(np.linspace(0.05, 0.9, len(lv_ref)))
        for k in range(len(lv_ref)):
            ax.plot(centres, mean[k], marker="o", ms=4, lw=1.8,
                    color=shades[k], label=f"{lv_ref[k]:.0f} m")
        ax.axvline(0.0, color="0.5", lw=1.0, ls="--")
        ax.text(-0.02, ax.get_ylim()[1], "lee  ", ha="right", va="top",
                fontsize=9, color="0.4")
        ax.text(0.02, ax.get_ylim()[1], "  windward", ha="left", va="top",
                fontsize=9, color="0.4")
        ax.set_xlabel("along-wind terrain slope  (grad h . flow direction)")
        ax.set_ylabel("mean column error over the level's slab  [m/s]")
        ax.set_title(f"{what} against along-wind slope -- {args.fold} fold, "
                     f"{n_samples} samples", fontsize=11)
        ax.legend(fontsize=8, ncol=3, title="level (AGL)")
        ax.grid(color="0.92", linewidth=0.6)
        ax.set_axisbelow(True)
        path = os.path.join(args.out,
                            f"slope_error_{args.what}_{args.fold}.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        print(f"\n{path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
