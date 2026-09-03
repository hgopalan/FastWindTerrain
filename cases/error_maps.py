#!/usr/bin/env python3
"""
error_maps.py -- where in the DOMAIN the reconstruction error lives.

cases/eval_harness.py says how big the error is and how it varies with
height. It cannot say whether the error sits on ridge crests, in lee
recirculation, or spread evenly, and that is the difference between a
level set that needs another level and one that needs nothing.

WHAT A "LEVEL'S ERROR" MEANS HERE. The stitch reproduces the stored
values exactly AT the levels, so an error map at 80 m is zero by
construction and says nothing. Each level is therefore given the slab it
is responsible for -- from the geometric midpoint below it to the
geometric midpoint above -- and the map is the RMS error over that slab,
per column. Geometric, not arithmetic, because the levels are octaves:
the midpoint between 40 and 80 m belongs at 57 m, not 60.

The lowest level's slab reaches down to the ground, so it carries the
log-law fill below 5 m. That is the band the harness found worst, and on
these maps it is visibly a terrain effect rather than a uniform one.

--histogram answers the other question a single RMSE cannot: not how big
the error is on average, but what FRACTION of the domain is over half a
metre per second, per level. An RMSE of 0.19 can be a uniform 0.19 or a
quiet field with 2 % of it at 1.5 m/s, and those are different problems
with different fixes.

Usage:

    python3 cases/error_maps.py --window carr_fire:12
    python3 cases/error_maps.py --fold demo --data data/demo --limit 2
    python3 cases/error_maps.py --what baseline
    python3 cases/error_maps.py --histogram          # the whole fold
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import corpus                                               # noqa: E402


def slab_edges(levels):
    """The slab each level is responsible for: geometric midpoints.

    The lowest slab reaches the ground so the log-law fill is attributed
    to the level above it, which is the level that would have to change to
    fix it. The highest reaches the domain top.
    """
    import numpy as np

    lv = np.asarray(levels, dtype=float)
    mid = np.sqrt(lv[:-1] * lv[1:])
    lo = np.concatenate([[0.0], mid])
    hi = np.concatenate([mid, [np.inf]])
    return lo, hi


#: Absolute-error bins for --histogram, in m/s. The first edge is 0.5
#: rather than the 0.25 m/s CFD tolerance because 0.25 splits the bulk of
#: the distribution and hides the tail, which is what this is for. The
#: tolerance is marked on the plot instead.
ERROR_BINS = [0.0, 0.5, 1.0, 1.5, 2.0, float("inf")]


def bin_label(lo, hi):
    import numpy as np
    return f"{lo:.1f}-{hi:.1f}" if np.isfinite(hi) else f"{lo:.1f}+"


def histogram(args, out):
    """Fraction of fluid cells in each absolute-error bin, per level.

    Aggregated over the whole fold rather than one window, because the
    tail is what matters here and one window does not have enough of it.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from fastwindterrain import baseline as B
    from fastwindterrain import evaluate as E
    from fastwindterrain import levels as L
    import build_dataset as bd

    counts, lv_ref, n_samples = None, None, 0
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
        lo, hi = slab_edges(lv)
        if counts is None:
            counts = np.zeros((len(lv), len(ERROR_BINS) - 1))
            lv_ref = np.asarray(lv, dtype=float)

        for k in range(len(lv)):
            sel = fluid & (agl >= lo[k]) & (agl < hi[k])
            if not sel.any():
                continue
            c, _ = np.histogram(err[sel], bins=ERROR_BINS)
            counts[k] += c

    if counts is None:
        print("no samples matched.", file=sys.stderr)
        return 1

    frac = 100.0 * counts / counts.sum(axis=1, keepdims=True)
    labels = [bin_label(ERROR_BINS[i], ERROR_BINS[i + 1])
              for i in range(len(ERROR_BINS) - 1)]

    what = ("reconstruction floor" if args.what == "floor"
            else "baseline error")
    print(f"\n{what}, fold '{args.fold}', {n_samples} samples")
    print("percentage of fluid cells by absolute vector error [m/s]\n")
    head = f"{'level':>10}  " + "  ".join(f"{x:>8}" for x in labels)
    print(head)
    print("-" * len(head))
    for k in range(len(lv_ref)):
        print(f"{lv_ref[k]:>8.0f} m  "
              + "  ".join(f"{x:>8.2f}" for x in frac[k]))
    print(f"\n{'all':>10}  "
          + "  ".join(f"{x:>8.2f}" for x in
                      100.0 * counts.sum(axis=0) / counts.sum()))

    # Stacked bars, one per level. The bins are ordered, so they get one
    # hue light to dark -- a categorical palette here would imply the bins
    # are unrelated categories rather than a magnitude scale.
    # Two panels. The left is the whole distribution and is dominated by
    # the first bin; the right drops that bin and rescales, because the
    # tail is the part with any information in it and at 0-100 % it is a
    # few invisible pixels.
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(12.0, 5.4), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.25]})
    shades = plt.cm.YlOrRd(np.linspace(0.12, 0.92, len(labels)))
    y = np.arange(len(lv_ref))

    left = np.zeros(len(lv_ref))
    for i, lab in enumerate(labels):
        axL.barh(y, frac[:, i], left=left, color=shades[i],
                 edgecolor="white", linewidth=0.6, label=f"{lab} m/s")
        left += frac[:, i]
    axL.set_xlim(0, 100)
    axL.set_xlabel("% of fluid cells in the level's slab")
    axL.set_title("the whole distribution", fontsize=10)
    for i, v in enumerate(frac[:, 0]):
        axL.text(2.0, i, f"{v:.1f} % under 0.5 m/s", va="center",
                 fontsize=8, color="0.15")

    tail = frac[:, 1:]
    left = np.zeros(len(lv_ref))
    for i in range(tail.shape[1]):
        axR.barh(y, tail[:, i], left=left, color=shades[i + 1],
                 edgecolor="white", linewidth=0.6,
                 label=f"{labels[i + 1]} m/s")
        left += tail[:, i]
    axR.set_xlim(0, max(1.0, 1.08 * float(tail.sum(axis=1).max())))
    axR.set_xlabel("% of fluid cells over 0.5 m/s")
    axR.set_title("the tail alone, rescaled", fontsize=10)
    for i, v in enumerate(tail.sum(axis=1)):
        axR.text(v + 0.12, i, f"{v:.2f} %", va="center", fontsize=8,
                 color="0.15")
    axR.legend(loc="lower right", fontsize=8, framealpha=0.95,
               title="absolute error")

    for ax in (axL, axR):
        ax.set_yticks(y)
        ax.set_yticklabels([f"{h:.0f} m" for h in lv_ref], fontsize=9)
        ax.invert_yaxis()
        ax.grid(axis="x", color="0.9", linewidth=0.6)
        ax.set_axisbelow(True)
    axL.set_ylabel("level (AGL)")
    fig.suptitle(f"{what} by level -- {args.fold} fold, {n_samples} "
                 f"samples, vector error", fontsize=12)

    path = os.path.join(out, f"errorhist_{args.what}_{args.fold}.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"\n{path}")
    return 0


def main(argv=None):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from fastwindterrain import baseline as B
    from fastwindterrain import evaluate as E
    from fastwindterrain import levels as L
    import build_dataset as bd

    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(ROOT, "data", "corpus"))
    p.add_argument("--fold", default="test")
    p.add_argument("--window", action="append", default=None, metavar="ID",
                   help="window id, without the @direction")
    p.add_argument("--direction", type=float, default=None,
                   help="only this direction (default: the first found)")
    p.add_argument("--what", default="floor", choices=["floor", "baseline"],
                   help="floor: what nine levels cannot carry. baseline: "
                        "what the terrain does to the undisturbed profile, "
                        "i.e. what a surrogate has to learn")
    p.add_argument("--limit", type=int, default=3)
    p.add_argument("--histogram", action="store_true",
                   help="instead of maps, the distribution of absolute "
                        "error per level over the whole fold")
    p.add_argument("--out", default=None, metavar="DIR")
    args = p.parse_args(argv)

    out = args.out or os.environ.get("TMPDIR", "/tmp")
    os.makedirs(out, exist_ok=True)

    if args.histogram:
        # The map default of 3 is far too few for a tail; take the fold.
        if args.limit == 3:
            args.limit = None
        return histogram(args, out)

    man = corpus.load_manifest()
    relief = {w["id"]: float(w["relief"]) for w in man["windows"]}

    made = []
    for info, a in bd.load_dataset(args.data, fold=args.fold, with_3d=True):
        if info["derived"]:
            continue
        wid, _, dtxt = info["id"].partition("@")
        if args.window and wid not in args.window:
            continue
        if args.direction is not None and float(dtxt) != args.direction:
            continue
        if len(made) >= args.limit:
            break

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
            what = "reconstruction floor"
        else:
            other = B.undisturbed(z_cc, zt, solid,
                                  corpus.REFERENCE_SPEED_MS,
                                  info["direction"],
                                  z_ref=corpus.REFERENCE_HEIGHT_M)
            what = "baseline error (the terrain effect)"

        err = np.sqrt(((other - ref) ** 2).sum(axis=0))    # vector, per cell
        agl = L.height_above_ground(z_cc, zt)
        lo, hi = slab_edges(lv)

        # One map per level: RMS over that level's slab, per column.
        maps, peaks = [], []
        for k in range(len(lv)):
            sel = fluid & (agl >= lo[k]) & (agl < hi[k])
            n = sel.sum(axis=0)
            acc = np.where(sel, err ** 2, 0.0).sum(axis=0)
            m = np.sqrt(np.divide(acc, n, out=np.zeros_like(acc),
                                  where=n > 0))
            m[n == 0] = np.nan
            maps.append(m)
            peaks.append(float(np.nanmax(m)) if np.isfinite(m).any() else 0.0)

        # One shared colour scale, so the panels can be compared. Per-panel
        # scaling would make every level look equally bad, which is the
        # opposite of the finding.
        vmax = float(np.nanpercentile(np.stack(maps), 99.5))
        km = np.arange(u.shape[2]) * dx / 1000.0

        # A column with no cell in the slab is NOT zero error, and the two
        # must not both render white. Near the surface the slabs are only
        # a cell or two deep, so this is most of the bottom-left panel.
        cmap = matplotlib.colormaps["magma_r"].with_extremes(bad="0.82")

        fig, axes = plt.subplots(3, 3, figsize=(11.5, 11.0),
                                 constrained_layout=True)
        for k, ax in enumerate(axes.ravel()):
            if k >= len(lv):
                ax.axis("off")
                continue
            im = ax.pcolormesh(km, km, np.ma.masked_invalid(maps[k]),
                               cmap=cmap, vmin=0.0, vmax=vmax,
                               shading="auto")
            empty = 100.0 * np.isnan(maps[k]).mean()
            # Terrain, so the error can be read against the ground that
            # produced it. Thin and grey: it is context, not data.
            ax.contour(km, km, zt, levels=8, colors="0.35",
                       linewidths=0.4, alpha=0.8)
            top = "top" if not np.isfinite(hi[k]) else f"{hi[k]:.0f} m"
            gap = f",  {empty:.0f}% no cells" if empty >= 0.5 else ""
            ax.set_title(f"{lv[k]:.0f} m AGL   ({lo[k]:.0f}-{top})\n"
                         f"peak {peaks[k]:.2f} m/s{gap}", fontsize=9)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)
            if k // 3 == 2:
                ax.set_xlabel("km", fontsize=8)
            if k % 3 == 0:
                ax.set_ylabel("km", fontsize=8)

        fig.colorbar(im, ax=axes, shrink=0.55, location="right",
                     label="vector RMS error over the slab  [m/s]")
        fig.suptitle(
            f"{wid}  at  {float(dtxt):.0f} deg      {what}\n"
            f"relief {relief.get(wid, float('nan')):.0f} m      "
            f"shared colour scale, 0 to {vmax:.2f} m/s      "
            f"grey lines: terrain,  flat grey: no cell in that slab",
            fontsize=11)

        path = os.path.join(
            out, f"errormap_{args.what}_{wid.replace(':', '_')}"
                 f"_{float(dtxt):03.0f}.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        made.append(path)
        print(f"{path}   peak {max(peaks):.2f} m/s at "
              f"{lv[int(np.argmax(peaks))]:.0f} m")

    if not made:
        print("nothing matched -- check --window/--fold against the "
              "dataset, and remember only folds with 3D can be mapped.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
