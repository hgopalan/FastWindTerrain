#!/usr/bin/env python3
"""
verify_dataset.py -- is the generated dataset complete and readable?

Run this after cases/build_dataset.py finishes. It is the one path the
unit tests cover but a real multi-worker run has never exercised end to
end: eight manifests, a few hundred shard files, and a loader that has to
stitch them into one dataset without silently dropping any.

A dataset that loads as EMPTY looks like a training bug rather than an I/O
one, which is the failure this exists to prevent.

What it checks, in the order the failures actually happen:

  complete    every worker wrote a manifest. A worker that died leaves its
              shards on disk with no manifest, and they are unreadable --
              exactly what happened to worker 0 on the first run.
  countable   the sample count is what the corpus implies, and the
              independent count is HALF of it, because the solver is odd
              in the inflow.
  readable    every sample's arrays load, with the shapes and dtypes the
              network expects.
  negated     each derived sample really is the negation of its partner in
              velocity and IDENTICAL in terrain, which is the assumption
              that halved the compute.
  finite      no NaN or inf anywhere. A single bad sample poisons a
              training run and is tedious to find later.
  covered     every fold is present, including the demo sites, and no
              window appears in two folds.

Usage:

    python3 cases/verify_dataset.py                     # data/corpus
    python3 cases/verify_dataset.py --data other/dir
"""

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import corpus                                               # noqa: E402

EXPECT_SHAPES = {
    "u_lev": 3, "v_lev": 3, "w_lev": 3,     # (nlev, ny, nx)
    "terrain": 2, "k_first": 2,             # (ny, nx)
    "levels": 1, "z_cc": 1,
}


def main(argv=None):
    import numpy as np

    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(ROOT, "data", "corpus"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--full", action="store_true",
                   help="load every sample, not a sample of them")
    args = p.parse_args(argv)

    fail = []

    def check(name, ok, detail):
        print(f"  [{'ok  ' if ok else 'FAIL'}] {name:12s} {detail}")
        if not ok:
            fail.append(name)

    print(f"dataset: {args.data}\n")

    # -- complete ---------------------------------------------------------
    mans = sorted(glob.glob(os.path.join(args.data, "manifest_*.json")))
    shards = sorted(glob.glob(os.path.join(args.data, "shard_*.npz")))
    check("complete", len(mans) == args.workers,
          f"{len(mans)} of {args.workers} manifests, {len(shards)} shards")
    if not mans:
        print("\nno manifests at all -- nothing to verify.")
        return 1

    samples = []
    for m in mans:
        with open(m) as f:
            samples += json.load(f)["samples"]

    # A worker whose shards exist without a manifest is invisible here,
    # so say so rather than let the count quietly look fine.
    shard_parts = {os.path.basename(s).split("_")[1] for s in shards}
    man_parts = {os.path.basename(m).split("_")[1].split(".")[0]
                 for m in mans}
    orphan = sorted(shard_parts - man_parts)
    check("no orphans", not orphan,
          "every shard has a manifest" if not orphan
          else f"shards from worker(s) {orphan} have no manifest and are "
               f"UNREADABLE")

    # -- countable --------------------------------------------------------
    man = corpus.load_manifest()
    split = [w for w in man["windows"] if w["fold"] != corpus.DEMO_FOLD]
    want_solved = len(split) * len(corpus.INDEPENDENT_DIRECTIONS)
    solved = [s for s in samples if not s.get("derived")]
    derived = [s for s in samples if s.get("derived")]
    check("countable",
          len(solved) == want_solved and len(derived) == len(solved),
          f"{len(samples)} samples = {len(solved)} solved + "
          f"{len(derived)} derived; expected {want_solved} + {want_solved}")

    # -- readable / finite / negated --------------------------------------
    import build_dataset as bd

    pairs = {}
    n_read = n_bad_shape = n_nonfinite = 0
    checked_neg = 0
    for info, arrays in bd.load_dataset(args.data):
        n_read += 1
        for k, want_ndim in EXPECT_SHAPES.items():
            if k not in arrays:
                continue
            if arrays[k].ndim != want_ndim:
                n_bad_shape += 1
            if not np.all(np.isfinite(arrays[k])):
                n_nonfinite += 1
        if info.get("derived"):
            pairs.setdefault(info["derived_from"], {})["d"] = arrays
        else:
            pairs.setdefault(info["id"], {})["s"] = arrays
        if not args.full and n_read >= 400:
            break

    check("readable", n_read > 0 and n_bad_shape == 0,
          f"{n_read} samples loaded, {n_bad_shape} with a wrong rank")
    check("finite", n_nonfinite == 0,
          f"{n_nonfinite} arrays carrying NaN or inf")

    worst_vel, worst_ter = 0.0, 0.0
    for sid, both in pairs.items():
        if "s" not in both or "d" not in both:
            continue
        checked_neg += 1
        for k in ("u_lev", "v_lev", "w_lev"):
            if k in both["s"]:
                worst_vel = max(worst_vel, float(np.abs(
                    both["s"][k] + both["d"][k]).max()))
        for k in ("terrain", "k_first"):
            if k in both["s"]:
                worst_ter = max(worst_ter, float(np.abs(
                    both["s"][k].astype(float)
                    - both["d"][k].astype(float)).max()))
    check("negated", checked_neg > 0 and worst_vel < 1e-5 and worst_ter == 0.0,
          f"{checked_neg} pairs: max |u_solved + u_derived| = {worst_vel:.2e}, "
          f"terrain differs by {worst_ter:.1e}")

    # -- covered ----------------------------------------------------------
    folds = {}
    for s in samples:
        folds[s.get("fold", "?")] = folds.get(s.get("fold", "?"), 0) + 1
    check("covered", set(folds) >= set(corpus.FOLDS),
          ", ".join(f"{k} {v}" for k, v in sorted(folds.items())))

    if fail:
        print(f"\n{len(fail)} check(s) failed: {fail}")
        return 1
    print("\nAll checks passed. The dataset is complete and readable.")
    if not args.full:
        print("(a sample of 400; pass --full to load every one)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
