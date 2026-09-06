#!/usr/bin/env python3
"""
low_level_study.py -- would a level below 5 m be worth another channel?

The near-surface band is the only one outside tolerance, and every
architecture tried lands within 4.5 % of every other there, so the
problem is not the model. cases/eval_harness.py --by-height traced it to
EXTRAPOLATION: below the lowest level the field is filled from a log law,
which carries 0.467 m/s against 0.112 for everything interpolated -- 4.2
times worse, on 1.4 % of cells.

The obvious fix is to put a level underneath, so the bottom cells are
interpolated rather than extrapolated. That costs one output channel and,
because levels are extracted at dataset-generation time, a regenerated
corpus.

THIS MEASURES THE CEILING FIRST, AND COSTS NOTHING. The reconstruction
floor -- stitching from PERFECT levels -- is the best any model with that
level set could achieve. It is computable from the 3D fields already
stored on the test and demo folds, with no solving and no training. If
the floor does not improve, no model built on the new level set can, and
the experiment is dead before it is paid for.

The level sets compared, all exact octaves:

    9 levels   5, 10, 20, 40, 80, 160 + 3 aloft   (the current corpus)
   10 levels   2.5 + the above                     (one below)
   10 levels   the above + one aloft               (the control)

The third is the control that matters. Adding ANY tenth level adds
capacity, so a gain at 2.5 m only means something if the same tenth level
spent aloft does not buy the same thing.

Usage:

    python3 cases/low_level_study.py
    python3 cases/low_level_study.py --data data/demo --fold demo
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


def level_sets(top_agl):
    """The candidate sets, built the way the corpus builds its own."""
    from fastwindterrain import levels as L

    base = L.recommended_levels(top_agl)                    # 9, the corpus
    low = L.recommended_levels(top_agl, n_band=7, base=2.5)  # +1 below
    aloft = L.recommended_levels(top_agl, n_aloft=4)         # +1 above
    return {"9 (corpus)": base, "10 (+2.5 m)": low,
            "10 (+1 aloft)": aloft}


def main(argv=None):
    import numpy as np
    from fastwindterrain import evaluate as E
    from fastwindterrain import levels as L
    import build_dataset as bd

    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(ROOT, "data", "corpus"))
    p.add_argument("--fold", default="test")
    p.add_argument("--limit", type=int, default=80)
    args = p.parse_args(argv)

    bands = [(0.0, 10.0), (10.0, 50.0), (50.0, 160.0), (160.0, np.inf)]
    acc, n, t0 = {}, 0, time.time()

    for info, a in bd.load_dataset(args.data, fold=args.fold, with_3d=True):
        if info["derived"]:
            continue
        if n >= args.limit:
            break
        n += 1

        u, v, w = a["u"], a["v"], a["w"]
        z_cc, zt = a["z_cc"], a["terrain"]
        fluid = E.fluid_from_k_first(a["k_first"], u.shape[0])
        solid = (~fluid).astype(np.int32)
        ref = np.stack([u, v, w]).astype(np.float64)
        dx = dy = corpus.WINDOW_M / u.shape[2]
        agl = L.height_above_ground(z_cc, np.asarray(zt))
        top = float(agl[fluid].max())

        for name, lv in level_sets(top).items():
            # Extract and stitch back, exactly as the dataset does.
            got = np.stack([
                L.stitch_levels(
                    L.extract_levels(f, z_cc, zt, lv, mask=solid,
                                     frame="agl", dx=dx, dy=dy),
                    lv, z_cc, zt, mask=solid, frame="agl", dx=dx, dy=dy)
                for f in (u, v, w)])
            d = acc.setdefault(name, {"n_lev": len(lv),
                                      "bands": np.zeros(len(bands)),
                                      "cnt": np.zeros(len(bands)),
                                      "all": 0.0})
            e2 = ((got - ref) ** 2).sum(axis=0)
            for b, (lo, hi) in enumerate(bands):
                sel = fluid & (agl >= lo) & (agl < hi)
                if sel.any():
                    d["bands"][b] += float(e2[sel].sum())
                    d["cnt"][b] += int(sel.sum())
            d["all"] += float(e2[fluid].mean())

    if not acc:
        print("no 3D samples found", file=sys.stderr)
        return 1

    print(f"{n} samples, fold '{args.fold}', {time.time()-t0:.0f} s")
    print("reconstruction FLOOR -- stitched from PERFECT levels, so this "
          "is the\nbest any model on that level set could reach. m/s.\n")
    names = list(acc)
    hdr = f"{'band (AGL)':>12s}  " + "  ".join(f"{k:>14s}" for k in names)
    print(hdr)
    print("-" * len(hdr))
    for b, (lo, hi) in enumerate(bands):
        lab = f"{lo:.0f}-{hi:.0f} m" if np.isfinite(hi) else f"{lo:.0f}+ m"
        print(f"{lab:>12s}  " + "  ".join(
            f"{np.sqrt(acc[k]['bands'][b] / max(acc[k]['cnt'][b], 1)):14.4f}"
            for k in names))
    print("-" * len(hdr))
    print(f"{'column':>12s}  " + "  ".join(
        f"{np.sqrt(acc[k]['all'] / n):14.4f}" for k in names))
    print(f"{'levels':>12s}  " + "  ".join(
        f"{acc[k]['n_lev']:14d}" for k in names))

    base = np.sqrt(acc[names[0]]["all"] / n)
    print("\nagainst the current 9-level set:")
    for k in names[1:]:
        v = np.sqrt(acc[k]["all"] / n)
        b0 = np.sqrt(acc[names[0]]["bands"][0]
                     / max(acc[names[0]]["cnt"][0], 1))
        b1 = np.sqrt(acc[k]["bands"][0] / max(acc[k]["cnt"][0], 1))
        print(f"  {k:16s} column {100*(v-base)/base:+6.1f} %   "
              f"0-10 m {100*(b1-b0)/b0:+6.1f} %")
    return 0


if __name__ == "__main__":
    sys.exit(main())
