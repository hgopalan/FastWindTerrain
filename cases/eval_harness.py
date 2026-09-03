#!/usr/bin/env python3
"""
eval_harness.py -- phase 22a: the scoring, validated before any model.

Runs fastwindterrain.evaluate over the generated dataset and reports what
a surrogate has to beat and how well it could possibly do, on the same
metric, in the same units, over the same samples.

WHY THIS EXISTS SEPARATELY FROM THE TRAINING CODE. A metric and a model
written together cannot be told apart. The first disappointing number is
then ambiguous -- bad network, or bad scoring? -- and there is nothing to
settle it with. So the scoring is written first and checked against two
fields whose error is already known from a different code path:

    baseline        the undisturbed profile: terrain-following but
                    terrain-blind, the field available for free. Measured
                    earlier at 0.25 m/s on gentle windows and 1.46 m/s on
                    complex ones.
    reconstruction  the dataset's own levels stitched back into 3D. This
                    is the FLOOR: a perfect network reproducing the stored
                    levels exactly still lands here, because nine levels
                    do not carry sixty. Measured earlier at 0.06-0.29 m/s.

If those two come back in the neighbourhood they were measured in, the
scoring is sound and every later model number can be trusted. If they do
not, the scoring is wrong and it is worth knowing that now rather than
after a week of training.

The gap between them is the headroom -- all the accuracy a surrogate can
actually contribute. It is not the same on every window, which is the
whole reason nothing here reports one aggregate number.

Usage:

    python3 cases/eval_harness.py
    python3 cases/eval_harness.py --data data/demo --fold demo
    python3 cases/eval_harness.py --limit 12          # a quick look
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

#: What the earlier studies measured, for the comparison this harness
#: exists to make. Ranges, not points: they came from eight windows at one
#: direction, and this runs hundreds at four, so the numbers should
#: BRACKET these rather than reproduce them.
PRIOR = {
    "baseline": (0.25, 1.46),
    "reconstruction": (0.06, 0.29),
}


def main(argv=None):
    import numpy as np
    from fastwindterrain import baseline as B
    from fastwindterrain import evaluate as E
    from fastwindterrain import levels as L

    import build_dataset as bd

    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(ROOT, "data", "corpus"))
    p.add_argument("--fold", default="test",
                   help="which fold to score (default test -- it is the "
                        "one that stores the 3D field)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--csv", default=None, metavar="PATH",
                   help="also write the per-sample rows")
    args = p.parse_args(argv)

    man = corpus.load_manifest()
    relief = {w["id"]: float(w["relief"]) for w in man["windows"]}

    rows, t0 = [], time.time()
    for info, a in bd.load_dataset(args.data, fold=args.fold, with_3d=True):
        # The derived half is the exact negation of its partner, and both
        # the baseline and the stitch are odd in the wind direction, so it
        # scores identically. Running it would double the time and add no
        # information.
        if info["derived"]:
            continue
        if args.limit and len(rows) >= args.limit:
            break

        u, v, w = a["u"], a["v"], a["w"]
        z_cc, zt, lv = a["z_cc"], a["terrain"], a["levels"]
        nz = u.shape[0]
        fluid = E.fluid_from_k_first(a["k_first"], nz)
        solid = (~fluid).astype(np.int32)
        ref = np.stack([u, v, w]).astype(np.float64)
        dx = dy = corpus.WINDOW_M / u.shape[2]

        # -- the baseline: what the terrain effect is worth --------------
        base = B.undisturbed(z_cc, zt, solid, corpus.REFERENCE_SPEED_MS,
                             info["direction"],
                             z_ref=corpus.REFERENCE_HEIGHT_M)
        e_base = E.error_stats(base, ref, sel=fluid)

        # -- the floor: the stored levels stitched back ------------------
        stitched = np.stack([
            L.stitch_levels(a[k], lv, z_cc, zt, mask=solid, frame="agl",
                            dx=dx, dy=dy)
            for k in ("u_lev", "v_lev", "w_lev")])
        e_recon = E.error_stats(stitched, ref, sel=fluid)

        wid = info["id"].split("@")[0]
        rows.append({
            "id": info["id"],
            "relief": relief.get(wid, float("nan")),
            "baseline": e_base["rmse"],
            "baseline_p95": e_base["p95"],
            "reconstruction": e_recon["rmse"],
            "reconstruction_p95": e_recon["p95"],
            "headroom": e_base["rmse"] - e_recon["rmse"],
            "skill_ceiling": E.skill(e_recon["rmse"], e_base["rmse"]),
        })
        if len(rows) % 20 == 0:
            print(f"  {len(rows)} samples scored "
                  f"({time.time()-t0:.0f} s)", file=sys.stderr)

    if not rows:
        print(f"no samples with a 3D field in fold '{args.fold}' under "
              f"{args.data}.\nThe split folds store 3D on test only; "
              f"data/demo stores it on every sample.", file=sys.stderr)
        return 1

    print(f"{len(rows)} samples, fold '{args.fold}', {args.data}")
    print(f"vector RMSE against the solver, in m/s "
          f"(tolerance in this field is ~0.25 m/s)\n")

    W = {"group": 9, "relief": 10, "n": 4, "mean": 8, "worst": 8}
    for key, title in (
            ("baseline", "BASELINE -- the undisturbed profile, "
                         "terrain-blind. What a surrogate must beat."),
            ("reconstruction", "FLOOR -- the stored levels stitched back. "
                               "No surrogate can do better."),
            ("headroom", "HEADROOM -- baseline minus floor. All the "
                         "accuracy a surrogate can contribute.")):
        print(title)
        print(E.table(E.group_by_relief(rows, key=key),
                      ["group", "relief", "n", "mean", "worst"], W))
        print()

    # -- the check this harness exists for --------------------------------
    print("against the earlier studies (different windows and direction, "
          "so these should BRACKET rather than match):")
    ok = True
    for key, (plo, phi) in PRIOR.items():
        vals = [r[key] for r in rows]
        lo, hi = min(vals), max(vals)
        brackets = lo <= phi and hi >= plo
        ok = ok and brackets
        print(f"  [{'ok  ' if brackets else 'CHECK'}] {key:14s} "
              f"measured {lo:.2f}-{hi:.2f} m/s, "
              f"earlier {plo:.2f}-{phi:.2f}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        print(f"\nper-sample rows: {args.csv}")

    print(f"\n{time.time()-t0:.0f} s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
