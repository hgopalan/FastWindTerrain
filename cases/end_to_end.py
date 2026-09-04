#!/usr/bin/env python3
"""
end_to_end.py -- the whole chain: terrain in, 3D wind field out.

Phase 23, and the paper's second contribution. Everything measured so far
has been one link of it. cases/eval_harness.py measured what stitching
costs from PERFECT levels -- the floor. cases/train_surrogate.py measured
what the network costs AT the levels. Neither says what happens when the
two are composed, and that is the question the surrogate exists to answer:

    terrain + direction  ->  9 levels  ->  60-layer 3D field

THE QUESTION THIS SETTLES. If model error and reconstruction error simply
add in quadrature, the pipeline is sound and only the model needs work.
If stitching AMPLIFIES the model error, the pipeline has a design problem
worth finding before anyone spends a week tuning. Amplification is
plausible: the floor was measured from level values the solver produced,
which are smooth and mutually consistent, while a predicted level field
is neither. Interpolating between two independently-wrong levels can be
worse than either.

So the report separates three things, all vector RMSE in m/s over fluid
cells, against the same solver field:

    floor       stitch(true levels)        what nine levels cannot carry
    levels      model error AT the levels  the network alone
    end to end  stitch(predicted levels)   what a user would actually get
    quadrature  sqrt(floor^2 + levels^2)   what end-to-end SHOULD be if
                                           the two errors are independent
                                           and stitching is neutral

Reported by relief, never as one number, and against the undisturbed
baseline so "is this worth anything" has an answer.

Usage:

    python3 cases/end_to_end.py --run data/runs/unet
    python3 cases/end_to_end.py --run data/runs/unet --data data/demo \\
        --fold demo
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
from predict_maps import load_run                           # noqa: E402


def main(argv=None):
    import numpy as np
    import torch

    from fastwindterrain import baseline as B
    from fastwindterrain import evaluate as E
    from fastwindterrain import levels as L
    from fastwindterrain import training as T
    import build_dataset as bd

    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, metavar="DIR")
    p.add_argument("--data", default=os.path.join(ROOT, "data", "corpus"))
    p.add_argument("--fold", default="test")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--csv", default=None, metavar="PATH")
    args = p.parse_args(argv)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cpu")
    model, ck = load_run(args.run, device)
    scales = np.asarray(ck["scales"])
    u_ref = float(ck.get("u_ref", corpus.REFERENCE_SPEED_MS))

    man = corpus.load_manifest()
    relief_of = {w["id"]: float(w["relief"]) for w in man["windows"]}

    rows, t0 = [], time.time()
    for info, a in bd.load_dataset(args.data, fold=args.fold, with_3d=True):
        # The derived half is the exact negation and every operator in the
        # chain is odd in the direction, so it scores identically.
        if info["derived"]:
            continue
        if args.limit and len(rows) >= args.limit:
            break

        u, v, w = a["u"], a["v"], a["w"]
        z_cc, zt, lv = a["z_cc"], a["terrain"], a["levels"]
        fluid = E.fluid_from_k_first(a["k_first"], u.shape[0])
        solid = (~fluid).astype(np.int32)
        ref = np.stack([u, v, w]).astype(np.float64)
        dx = dy = corpus.WINDOW_M / u.shape[2]
        nlev = lv.size

        ds = T.LevelDataset([(info, a)], u_ref=u_ref,
                            window_m=corpus.WINDOW_M, scales=scales)
        x, y = ds[0]
        with torch.no_grad():
            pred = model(x[None].to(device)).cpu().numpy()[0]
        P = T.to_ms(pred, u_ref, scales).reshape(3, nlev, *pred.shape[1:])
        Y = T.to_ms(y.numpy(), u_ref, scales).reshape(3, nlev,
                                                      *pred.shape[1:])

        def stitch(levels3):
            return np.stack([
                L.stitch_levels(levels3[c], lv, z_cc, zt, mask=solid,
                                frame="agl", dx=dx, dy=dy)
                for c in range(3)])

        e_lev = E.error_stats(P, Y)["rmse"]              # at the levels
        e_floor = E.error_stats(stitch(Y), ref, sel=fluid)["rmse"]
        e_e2e = E.error_stats(stitch(P), ref, sel=fluid)["rmse"]
        base = B.undisturbed(z_cc, zt, solid, corpus.REFERENCE_SPEED_MS,
                             info["direction"],
                             z_ref=corpus.REFERENCE_HEIGHT_M)
        e_base = E.error_stats(base, ref, sel=fluid)["rmse"]

        quad = float(np.hypot(e_floor, e_lev))
        rows.append({
            "id": info["id"],
            "relief": relief_of.get(info["id"].split("@")[0], float("nan")),
            "baseline": e_base,
            "floor": e_floor,
            "levels": e_lev,
            "end_to_end": e_e2e,
            "quadrature": quad,
            # >1 means stitching amplified the model error; ~1 means the
            # two errors simply combined.
            "amplification": e_e2e / quad if quad > 0 else float("nan"),
            "skill": E.skill(e_e2e, e_base),
        })
        if len(rows) % 25 == 0:
            print(f"  {len(rows)} scored ({time.time()-t0:.0f} s)",
                  file=sys.stderr)

    if not rows:
        print(f"no 3D samples in fold {args.fold!r} under {args.data}",
              file=sys.stderr)
        return 1

    print(f"{len(rows)} samples, fold '{args.fold}', model {ck['arch']} "
          f"from {args.run}")
    print("vector RMSE against the solver's 3D field, in m/s\n")

    W = {"group": 9, "relief": 10, "n": 4, "mean": 8, "worst": 8}
    for key, title in (
            ("baseline", "BASELINE  the undisturbed profile"),
            ("floor", "FLOOR  stitched from TRUE levels -- the limit"),
            ("levels", "LEVELS  the model alone, at the levels"),
            ("end_to_end", "END TO END  stitched from PREDICTED levels"),
            ("amplification", "AMPLIFICATION  end-to-end / quadrature; "
                              "1.0 means stitching added nothing")):
        print(title)
        print(E.table(E.group_by_relief(rows, key=key),
                      ["group", "relief", "n", "mean", "worst"], W))
        print()

    import numpy as np
    amp = np.array([r["amplification"] for r in rows])
    e2e = np.array([r["end_to_end"] for r in rows])
    sk = np.array([r["skill"] for r in rows])
    print(f"overall: end to end {e2e.mean():.3f} m/s, "
          f"skill against the baseline {sk.mean():+.3f}")
    print(f"amplification {amp.mean():.3f} "
          f"(median {np.median(amp):.3f}, worst {amp.max():.3f})")
    print("  1.0 means the model error and the reconstruction error "
          "simply combined;\n  above 1.0 means stitching made the model's "
          "error worse than its size implies.")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        print(f"\nper-sample rows: {args.csv}")
    print(f"\n{time.time()-t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
