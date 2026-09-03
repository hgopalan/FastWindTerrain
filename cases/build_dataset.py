#!/usr/bin/env python3
"""
build_dataset.py -- phase 21: the surrogate's training data.

Solves the corpus and writes what a 2D surrogate trains on: horizontal
wind on nine levels above ground, per window and wind direction, with the
terrain that produced it and enough geometry to rebuild the 3D field.

WHY NOT fastwindterrain.dataset. That module fixes one grid across every
sample and refuses configs whose grid sections differ, which is the right
guard for a synthetic sweep. The corpus deliberately gives each window its
own vertical extent -- the floor follows its own relief -- so the grids
differ by construction. Cell COUNTS are identical, (60, 100, 100); only
z_cc moves. That is fine for a 2D surrogate, which never sees z_cc: levels
are at fixed heights ABOVE GROUND and have the same shape and meaning on
every window. z_cc is stored per sample because rebuilding 3D needs it.

FOUR DIRECTIONS, NOT EIGHT. The operator is exactly odd in the inflow
(corpus.INDEPENDENT_DIRECTIONS), so the reverse of every solve is its
negation, to 8.6e-16. Solving eight would be paying twice for one answer.
The other four are written into the manifest as derived, and
`load_samples` materialises them, so a consumer sees eight without the
dataset containing eight.

WHAT IS AND IS NOT STORED IN 3D. Levels are small -- nine of them, two
components, 100 x 100, is 0.7 MB a sample in float32. The full 3D field is
sixty layers and three components, 7 MB a sample, and 1008 solves of that
is seven gigabytes. It is needed only to check the reconstruction, which
is a validation question and not a training one, so --store-3d defaults to
the test fold alone.

THE DEMO FOLD IS NOT PART OF "EVERYTHING". `--fold` defaults to the three
split folds; demo windows are generated only when asked for by name, into
their own directory. Two reasons, and the second is the important one:
they carry the full 3D field, which the split folds mostly do not; and a
directory that holds no demo samples cannot leak them into training by an
absent-minded glob. Keeping them apart is cheaper than remembering.

Usage:

    python3 cases/build_dataset.py --out data/corpus            # the split
    python3 cases/build_dataset.py --out data/small --fold test --limit 8
    python3 cases/build_dataset.py --out data/x --store-3d all
    python3 cases/build_dataset.py --out data/demo --fold demo --store-3d all
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import corpus                                               # noqa: E402

#: float32 throughout. The solve is double, but the reconstruction ceiling
#: is 0.4-1.5 % and the physical tolerance is around 0.25 m/s, so float32's
#: 1e-7 is seven orders below anything that matters -- and it halves a
#: dataset that is measured in gigabytes.
DTYPE = "float32"


def sample_id(window_id, direction):
    return f"{window_id}@{direction:03.0f}"


def build_one(manifest, wid, direction, levels_fn, store_3d,
              n_projections=None):
    """Solve one window at one direction and return its sample dict."""
    import numpy as np
    from fastwindterrain import levels as L
    import fastwindterrain as fwt

    cfg = corpus.window_config(manifest, wid, wind_direction=float(direction))
    if n_projections is not None:
        cfg["poisson"]["n_projections"] = int(n_projections)
    s = fwt.Solver(cfg)
    s.setup()
    s.solve()
    s.diagnose()

    u, v, w = s.velocity
    z_cc = np.asarray(s.grid.z_cc)
    zt = np.asarray(s.z_terrain)[0]
    mask = np.asarray(s.mask)
    dx = dy = corpus.WINDOW_M / u.shape[2]
    lv = levels_fn(z_cc, zt, mask)

    def at_levels(f):
        return L.extract_levels(f, z_cc, zt, lv, mask=mask, frame="agl",
                               dx=dx, dy=dy)

    out = {
        # -- what the network predicts -----------------------------------
        "u_lev": at_levels(u).astype(DTYPE),
        "v_lev": at_levels(v).astype(DTYPE),
        # w is stored at the levels too. Phase 19 and its revalidation both
        # found interpolating w beats deriving it from continuity by 12-18x,
        # so the network predicting it directly is the cheaper path -- and
        # storing it costs a third of the level data.
        "w_lev": at_levels(w).astype(DTYPE),

        # -- what the network sees ---------------------------------------
        "terrain": zt.astype(DTYPE),
        # The first fluid cell per column: the mask's information in two
        # dimensions, which is what a 2D network can use. The full 3D mask
        # is recoverable from it and z_cc.
        "k_first": L.first_fluid_k(mask).astype("int16"),

        # -- what rebuilding 3D needs ------------------------------------
        "z_cc": z_cc.astype("float64"),
        "levels": np.asarray(lv, dtype="float64"),
    }
    if store_3d:
        out["u"] = np.asarray(u, dtype=DTYPE)
        out["v"] = np.asarray(v, dtype=DTYPE)
        out["w"] = np.asarray(w, dtype=DTYPE)

    dg = s.diagnostics
    info = {
        "window": wid,
        "direction": float(direction),
        "derived": False,
        "solid_fraction": float((mask == 1).mean()),
        "speed_max": float(np.sqrt(u * u + v * v + w * w)[mask == 0].max()),
        "div_l2": float(dg["div_l2"]),
        "div_max_fe": float(s.max_divergence_fe),
        "flux_imbalance": float(dg["flux_imbalance"]),
        "mlmg_iterations": int(s.solve_iterations),
        "surface": {k: (float(x) if isinstance(x, float) else x)
                    for k, x in s.surface.items()},
        "has_3d": bool(store_3d),
    }
    return out, info


def select_windows(manifest, folds=None, limit=None, part=0, of=1):
    """The windows one worker solves.

    ``folds`` defaults to the three split folds, which is what keeps the
    demo sites out of an unqualified run: they are in the corpus manifest
    and would otherwise be swept up by it.
    """
    folds = list(corpus.FOLDS) if not folds else list(folds)
    windows = [w for w in manifest["windows"] if w["fold"] in folds]
    if limit:
        windows = windows[:limit]
    if of > 1:
        windows = windows[part::of]
    return windows


def load_dataset(path, fold=None, with_3d=None):
    """Read a dataset back, materialising the derived reverse directions.

    Yields ``(info, arrays)``. The dataset on disk holds only the four
    independent directions; the reverse of each is produced here by
    negating the velocity fields, which is exact -- see
    ``corpus.INDEPENDENT_DIRECTIONS``. A consumer therefore sees eight
    directions without the dataset paying for eight.

    ``terrain``, ``k_first``, ``z_cc`` and ``levels`` are geometry and are
    NOT negated. Getting that wrong would flip the ground upside down and
    still look plausible in a loss curve, so the velocity keys are named
    explicitly rather than inferred.
    """
    import numpy as np

    VELOCITY = ("u_lev", "v_lev", "w_lev", "u", "v", "w")

    import glob

    mans = sorted(glob.glob(os.path.join(path, "manifest*.json")))
    if not mans:
        raise FileNotFoundError(f"no manifest under {path}")
    samples = []
    for mp in mans:
        with open(mp) as f:
            samples += json.load(f)["samples"]

    # Every shard in the directory, whichever worker wrote it.
    by_id = {}
    for sp in sorted(glob.glob(os.path.join(path, "shard_*.npz"))):
        z = np.load(sp)
        for key in z.files:
            sid, _, field = key.partition("|")
            by_id.setdefault(sid, {})[field] = z[key]

    for info in samples:
        if fold is not None and info.get("fold") != fold:
            continue
        if with_3d is not None and bool(info.get("has_3d")) != with_3d:
            continue
        src = info.get("derived_from") or info["id"]
        arrays = by_id.get(src)
        if arrays is None:
            continue
        if info["derived"]:
            arrays = {k: (-v if k in VELOCITY else v)
                      for k, v in arrays.items()}
        yield info, arrays


def main(argv=None):
    import numpy as np
    from fastwindterrain import levels as L
    import fastwindterrain as fwt

    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, metavar="DIR")
    p.add_argument("--fold", action="append", default=None,
                   choices=list(corpus.FOLDS) + [corpus.DEMO_FOLD],
                   help="restrict to these folds. Default is the three split "
                        "folds; 'demo' is never included unless named, and "
                        "belongs in its own --out directory")
    p.add_argument("--limit", type=int, default=None,
                   help="stop after this many windows, for a smoke run")
    p.add_argument("--part", type=int, default=0, metavar="I",
                   help="this worker's index, for sharding the run across "
                        "processes")
    p.add_argument("--of", type=int, default=1, metavar="N",
                   help="how many workers. Windows are dealt round-robin, "
                        "so every worker gets a mix of cheap and expensive "
                        "ones -- solve time runs 3x between the gentlest "
                        "window and the steepest, and a contiguous split "
                        "would leave one worker running long after the "
                        "others finished.")
    p.add_argument("--store-3d", default="test",
                   choices=["none", "test", "all"],
                   help="which folds keep the full 3D field (default test)")
    p.add_argument("--n-projections", type=int, default=None,
                   help="override the corpus pass count; for smoke runs "
                        "only -- a dataset must use the corpus default so "
                        "every sample comes from one operator")
    p.add_argument("--shard", type=int, default=32,
                   help="samples per shard file (default 32)")
    p.add_argument("--overwrite", action="store_true",
                   help="replace this worker's existing shards and manifest. "
                        "Without it a run that would overwrite them stops "
                        "before solving anything")
    args = p.parse_args(argv)

    manifest = corpus.load_manifest()
    windows = select_windows(manifest, args.fold, args.limit,
                             args.part, args.of)

    def levels_fn(z_cc, zt, mask):
        agl = L.height_above_ground(z_cc, zt)
        return L.recommended_levels(float(agl[mask == 0].max()))

    os.makedirs(args.out, exist_ok=True)

    man_name = ("manifest.json" if args.of == 1
                else f"manifest_{args.part:02d}.json")

    # Refuse to clobber, and refuse BEFORE solving rather than at the first
    # flush. Worker indices restart at 0 for every run, so a second run into
    # a finished directory silently overwrites shard_00_* and its manifest --
    # hours of solves, gone, with the remaining workers' shards left orphaned
    # and no error anywhere. Fail fast and name the directory.
    import glob
    clash = ([os.path.join(args.out, man_name)]
             if os.path.exists(os.path.join(args.out, man_name)) else [])
    clash += sorted(glob.glob(
        os.path.join(args.out, f"shard_{args.part:02d}_*.npz")))
    if clash and not args.overwrite:
        print(f"error: worker {args.part} would overwrite "
              f"{len(clash)} existing file(s) in {args.out}, starting with "
              f"{os.path.basename(clash[0])}.\n"
              f"  Write to a different --out, or pass --overwrite if that "
              f"data really is meant to be replaced.", file=sys.stderr)
        return 1

    n_dir = len(corpus.INDEPENDENT_DIRECTIONS)
    total = len(windows) * n_dir
    print(f"{len(windows)} windows x {n_dir} solved directions = {total} "
          f"solves\n  ({len(windows) * len(corpus.WIND_DIRECTIONS)} samples "
          f"after deriving the reverses, of which "
          f"{total} are independent)\n")

    infos, shard, shard_n, t0 = [], {}, 0, time.time()

    def flush():
        nonlocal shard, shard_n
        if not shard:
            return
        path = os.path.join(
            args.out, f"shard_{args.part:02d}_{shard_n:05d}.npz")
        np.savez_compressed(path, **shard)
        print(f"  wrote {os.path.basename(path)} "
              f"({os.path.getsize(path)/1e6:.1f} MB)")
        shard, shard_n = {}, shard_n + 1

    with fwt.session():
        for n, wentry in enumerate(windows, 1):
            wid = wentry["id"]
            keep3d = (args.store_3d == "all"
                      or (args.store_3d == "test"
                          and wentry["fold"] == "test"))
            for d in corpus.INDEPENDENT_DIRECTIONS:
                ts = time.time()
                arrays, info = build_one(manifest, wid, d, levels_fn,
                                         keep3d, args.n_projections)
                info["fold"] = wentry["fold"]
                sid = sample_id(wid, d)
                info["id"] = sid
                for k, a in arrays.items():
                    shard[f"{sid}|{k}"] = a
                infos.append(info)

                # The reverse direction, free: the operator is odd in the
                # inflow to 8.6e-16 (corpus.INDEPENDENT_DIRECTIONS).
                rd = (d + 180.0) % 360.0
                rinfo = dict(info)
                rinfo.update(id=sample_id(wid, rd), direction=float(rd),
                             derived=True, derived_from=sid)
                infos.append(rinfo)

                print(f"[{n:3d}/{len(windows)}] {sid}  "
                      f"{time.time()-ts:5.1f} s")
                if len(shard) // max(1, len(arrays)) >= args.shard:
                    flush()
        flush()

    man = {
        "note": "Generated by cases/build_dataset.py; do not edit by hand.",
        "corpus_manifest": os.path.relpath(corpus.MANIFEST, ROOT),
        "n_solved": sum(1 for i in infos if not i["derived"]),
        "n_samples": len(infos),
        "independent_directions": list(corpus.INDEPENDENT_DIRECTIONS),
        "all_directions": list(corpus.WIND_DIRECTIONS),
        "reference_speed_ms": corpus.REFERENCE_SPEED_MS,
        "reference_height_m": corpus.REFERENCE_HEIGHT_M,
        "n_projections": (args.n_projections
                          if args.n_projections is not None
                          else corpus.N_PROJECTIONS),
        "dtype": DTYPE,
        "store_3d": args.store_3d,
        "shards": shard_n,
        "samples": infos,
        # The two facts a reader of this dataset must not have to rediscover.
        "wind_speed_is_a_free_scaling": True,
        "reverse_direction_is_a_negation": True,
    }
    with open(os.path.join(args.out, man_name), "w") as f:
        json.dump(man, f, indent=1, sort_keys=True)
        f.write("\n")

    dt = time.time() - t0
    print(f"\n{man['n_solved']} solves in {dt/60:.1f} min "
          f"({dt/max(1,man['n_solved']):.0f} s each)")
    print(f"{man['n_samples']} samples, of which {man['n_solved']} are "
          f"independent -- the rest are exact negations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
