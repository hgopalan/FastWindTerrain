#!/usr/bin/env python3
"""
train_surrogate.py -- phase 22b: train a 2D surrogate on the corpus.

Predicts the nine terrain-following levels from terrain and wind
direction. The architecture is a flag, because the first question anyone
asks of an FNO paper is whether a U-Net does as well, and that question
should cost one string rather than a second codebase.

WHAT IT REPORTS, AND WHY NOT THE LOSS. The training loss is a mean square
on normalised fields and is not comparable to anything -- not to the
0.25 m/s tolerance, not to the analytical baselines, not between runs
with different normalisation. Every number this prints is instead a
vector RMSE in METRES PER SECOND through fastwindterrain.evaluate, on the
same metric as cases/eval_harness.py, grouped by relief. The loss is
shown only to confirm it is falling.

WHAT COUNTS AS SUCCESS is set before the run, by phase 22a:

    beat the baseline   0.542 m/s on gentle terrain, 2.552 on extreme --
                        the undisturbed profile, available for nothing.
                        A model that cannot beat this has learned nothing.
    approach the floor  0.070 to 0.259 m/s by relief -- the stored levels
                        stitched back. No model predicting these levels
                        can do better, so the floor is the target, not
                        zero.

The level error reported here is measured AT THE LEVELS, so it is not the
floor plus the model error -- it is the model error alone. The two combine
only once the field is stitched back to 3D.

Usage:

    python3 cases/train_surrogate.py --arch ufno --epochs 60
    python3 cases/train_surrogate.py --arch unet --epochs 60   # baseline
    python3 cases/train_surrogate.py --arch fno --limit 64 --epochs 3
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

#: What phase 22a measured, so a run says whether it succeeded rather
#: than leaving the reader to look it up. Mean vector RMSE in m/s by
#: relief bin, over the test fold; see docs/surrogate.rst.
BASELINE_MS = {"gentle": 0.542, "moderate": 0.755,
               "complex": 2.069, "extreme": 2.552}
FLOOR_MS = {"gentle": 0.070, "moderate": 0.075,
            "complex": 0.197, "extreme": 0.259}


def pick_device(requested=None):
    import torch
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_fold(path, fold, limit=None):
    """The SOLVED samples of one fold, in memory.

    Only the solved half: the reverses are exact negations and
    LevelDataset derives them on access, which halves the footprint. The
    training fold is about 0.7 GB that way and reads in a few seconds, so
    nothing is paged from disk during training.
    """
    import build_dataset as bd

    out = []
    for info, arrays in bd.load_dataset(path, fold=fold):
        if info["derived"]:
            continue
        out.append((info, arrays))
        if limit and len(out) >= limit:
            break
    if not out:
        raise SystemExit(f"no samples in fold {fold!r} under {path}")
    return out


def evaluate_ms(model, ds, device, batch, relief_of, u_ref, scales=None):
    """Vector RMSE in m/s per sample, and the relief it belongs to."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from fastwindterrain import evaluate as E
    from fastwindterrain import training as T

    model.eval()
    rows, i = [], 0
    loader = DataLoader(ds, batch_size=batch, shuffle=False)
    with torch.no_grad():
        for x, y in loader:
            p = model(x.to(device)).cpu().numpy()
            t = y.numpy()
            for b in range(p.shape[0]):
                # Back to m/s before anything is judged, and reshaped to
                # (3, nlev, ny, nx) so the error is a vector magnitude
                # rather than a per-component one.
                nlev = p.shape[1] // 3
                pp = T.to_ms(p[b], u_ref, scales).reshape(
                    3, nlev, *p.shape[2:])
                tt = T.to_ms(t[b], u_ref, scales).reshape(
                    3, nlev, *p.shape[2:])
                # The aloft levels scale with each window's own column, so
                # the heights are per sample, not a constant.
                st = E.level_errors(pp, tt, ds.levels(i))
                info = ds.info(i)
                rows.append({
                    "id": info["id"],
                    "relief": relief_of.get(info["id"].split("@")[0],
                                            float("nan")),
                    "rmse": st["column"]["rmse"],
                    "band": st.get("band", st["column"])["rmse"],
                })
                i += 1
    model.train()
    return rows


def main(argv=None):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from fastwindterrain import evaluate as E
    from fastwindterrain import models as M
    from fastwindterrain import training as T

    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(ROOT, "data", "corpus"))
    p.add_argument("--arch", default="ufno", choices=list(M.ARCHITECTURES))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4,
                   help="1e-3 diverged at epoch 11 on ufno and never "
                        "recovered; see docs/surrogate.rst")
    p.add_argument("--clip", type=float, default=1.0,
                   help="gradient-norm clip, 0 to disable. The first run "
                        "died on a single loss spike, which is exactly "
                        "what this prevents")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--spectral-lr", type=float, default=None,
                   help="separate learning rate for the complex spectral "
                        "weights. They are the only complex parameters in "
                        "the model and Adam's per-parameter scaling behaves "
                        "differently on them, so a representation limit and "
                        "an optimisation one look identical without this "
                        "knob")
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--modes", type=int, default=16)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--limit", type=int, default=None,
                   help="windows per fold, for a smoke run")
    p.add_argument("--frac", type=float, default=None, metavar="F",
                   help="train on this fraction of the training windows, "
                        "chosen at random with --seed. For the learning "
                        "curve. Validation is never subsampled.")
    p.add_argument("--steps", type=int, default=None, metavar="N",
                   help="total gradient steps, converted to epochs. Use "
                        "this rather than --epochs whenever dataset SIZE "
                        "is the variable: at fixed epochs a smaller set "
                        "gets fewer updates, and the comparison then "
                        "measures training amount as much as data amount.")
    p.add_argument("--spectral", action="store_true",
                   help="six global spectral descriptors as extra input "
                        "planes. Motivated by measurement: Chetco Bar's "
                        "gentle cells are 3.7x worse than Flatirons' at "
                        "identical LOCAL slope, so the region's ruggedness "
                        "matters and a bounded receptive field cannot see "
                        "it. All six are D4-invariant.")
    p.add_argument("--augment-d4", action="store_true",
                   help="the eight symmetries of the square, exact and "
                        "verified against the solver at 1e-13. Training "
                        "only; validation is never augmented.")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, metavar="DIR",
                   help="checkpoint and history (default data/runs/<arch>)")
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    out = args.out or os.path.join(ROOT, "data", "runs", args.arch)
    os.makedirs(out, exist_ok=True)

    man = corpus.load_manifest()
    relief_of = {w["id"]: float(w["relief"]) for w in man["windows"]}

    t0 = time.time()
    train_raw = load_fold(args.data, "train", args.limit)
    if args.frac is not None:
        # Whole WINDOWS, not samples: the four directions of one window
        # share its terrain, so splitting them would leave a window
        # partly in and partly out and overstate how much ground the
        # model saw.
        wins = sorted({i["id"].split("@")[0] for i, _ in train_raw})
        rng = np.random.default_rng(args.seed)
        keep = set(rng.permutation(wins)[:max(1, int(round(
            args.frac * len(wins))))])
        train_raw = [(i, a) for i, a in train_raw
                     if i["id"].split("@")[0] in keep]
        print(f"--frac {args.frac}: {len(keep)} of {len(wins)} windows, "
              f"{len(train_raw)} solved samples")
    val_raw = load_fold(args.data, "val", args.limit)
    u_ref = corpus.REFERENCE_SPEED_MS
    # Fit the channel scales on TRAIN ONLY. w is 6x smaller than u and v
    # and the 5 m level 2.7x smaller than the top one, so an unweighted
    # loss is dominated by the aloft horizontal channels -- the easiest
    # part of the column and the part nobody asked for.
    scales = T.channel_rms(train_raw, u_ref)
    ds_tr = T.LevelDataset(train_raw, u_ref=u_ref, window_m=corpus.WINDOW_M,
                           derive_reverses=True, scales=scales,
                           augment_d4=args.augment_d4,
                           spectral=args.spectral)
    ds_va = T.LevelDataset(val_raw, u_ref=u_ref, window_m=corpus.WINDOW_M,
                           derive_reverses=True, scales=scales,
                           spectral=args.spectral)
    print(f"loaded {len(ds_tr)} train and {len(ds_va)} val samples "
          f"in {time.time()-t0:.1f} s "
          f"({len(train_raw)} + {len(val_raw)} solved, the rest derived)")

    x0, y0 = ds_tr[0]
    model = M.build(args.arch, x0.shape[0], y0.shape[0],
                    **({"width": args.width} if args.arch == "unet" else
                       {"width": args.width, "modes": args.modes,
                        "blocks": args.blocks})).to(device)
    print(f"{args.arch} on {device}: "
          f"{M.count_parameters(model):,} parameters\n")

    if args.spectral_lr is None:
        groups = model.parameters()
    else:
        spec = [q for n, q in model.named_parameters() if ".spectral." in n]
        rest = [q for n, q in model.named_parameters()
                if ".spectral." not in n]
        print(f"  spectral params {sum(q.numel() for q in spec):,} at lr "
              f"{args.spectral_lr}, the other "
              f"{sum(q.numel() for q in rest):,} at {args.lr}")
        groups = [{"params": spec, "lr": args.spectral_lr},
                  {"params": rest, "lr": args.lr}]
    if args.steps:
        per_epoch = max(1, -(-len(ds_tr) // args.batch))
        args.epochs = max(1, int(round(args.steps / per_epoch)))
        print(f"--steps {args.steps}: {per_epoch} steps/epoch -> "
              f"{args.epochs} epochs")

    opt = torch.optim.AdamW(groups, lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loader = DataLoader(ds_tr, batch_size=args.batch, shuffle=True)

    history, best = [], float("inf")
    for epoch in range(1, args.epochs + 1):
        te, tot, n = time.time(), 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x), y)
            loss.backward()
            if args.clip:
                # Not torch's: it cannot take a norm of the complex
                # spectral weights. See models.clip_grad_norm.
                M.clip_grad_norm(model.parameters(), args.clip)
            opt.step()
            tot += loss.detach().item() * x.shape[0]
            n += x.shape[0]
        sched.step()

        rows = evaluate_ms(model, ds_va, device, args.batch, relief_of,
                           u_ref, scales)
        val = float(np.mean([r["rmse"] for r in rows]))
        band = float(np.mean([r["band"] for r in rows]))
        history.append({"epoch": epoch, "loss": tot / n, "val_ms": val,
                        "val_band_ms": band,
                        "lr": sched.get_last_lr()[0],
                        "seconds": time.time() - te})
        flag = ""
        if val < best:
            best = val
            torch.save({"arch": args.arch, "state": model.state_dict(),
                        "args": vars(args), "epoch": epoch,
                        "val_ms": val, "u_ref": u_ref,
                        "scales": scales},
                       os.path.join(out, "best.pt"))
            flag = "  <- best"
        print(f"epoch {epoch:3d}/{args.epochs}  loss {tot/n:.5f}  "
              f"val {val:.4f} m/s  band {band:.4f}  "
              f"{time.time()-te:5.1f} s{flag}")

    # -- the report, on the phase 22a metric ------------------------------
    rows = evaluate_ms(model, ds_va, device, args.batch, relief_of, u_ref,
                       scales)
    print(f"\nvalidation, vector RMSE at the levels [m/s]")
    table = E.group_by_relief(rows, key="rmse")
    for r in table:
        g = r["group"]
        r["baseline"] = BASELINE_MS.get(g, float("nan"))
        r["floor"] = FLOOR_MS.get(g, float("nan"))
        r["vs base"] = (E.skill(r["mean"], r["baseline"])
                        if np.isfinite(r["mean"]) else float("nan"))
    print(E.table(table, ["group", "relief", "n", "mean", "worst",
                          "baseline", "floor", "vs base"],
                  {"group": 9, "relief": 10, "n": 4, "mean": 8,
                   "worst": 8, "baseline": 9, "floor": 7, "vs base": 8}))
    print("  baseline/floor are the phase 22a test-fold numbers, for "
          "scale.\n  'vs base' is skill against the undisturbed profile: "
          "1 is perfect,\n  0 is no better than doing nothing.")

    with open(os.path.join(out, "history.json"), "w") as f:
        json.dump({"args": vars(args), "device": str(device),
                   "parameters": M.count_parameters(model),
                   "history": history, "final": table,
                   "best_val_ms": best}, f, indent=1, default=str)
        f.write("\n")
    print(f"\nbest {best:.4f} m/s -- {out}/best.pt, {out}/history.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
