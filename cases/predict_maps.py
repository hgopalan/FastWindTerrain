#!/usr/bin/env python3
"""
predict_maps.py -- what the surrogate actually predicts, as pictures.

Every number so far has been an RMSE. An RMSE says how wrong a field is
but not how it is wrong, and the difference matters: a prediction that is
smooth and slightly low everywhere is a different problem from one that
misses the ridge-top acceleration entirely, and they read identically at
1.05 m/s.

These are also the figures to show someone who is not going to read a
table. One row per level, three panels across: the solver, the surrogate,
and the difference. Solver and surrogate share a colour scale -- always,
because two fields on separate scales look far more alike than they are.

Usage:

    python3 cases/predict_maps.py --run data/runs/unet
    python3 cases/predict_maps.py --run data/runs/unet --window ditch_fire:10
    python3 cases/predict_maps.py --run data/runs/unet --data data/demo \\
        --fold demo --window chetco_bar:12
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import corpus                                               # noqa: E402

#: Levels to draw, in metres above ground. A met mast height, hub height,
#: and the top of the engineering band -- three rows is what fits on a
#: slide, and these are the three anyone asks about.
SHOW_LEVELS = (10.0, 80.0, 160.0)


def input_channels(ck):
    """How many input planes a checkpoint's model expects.

    Derived from the run's own flags rather than hardcoded: a run with
    --spectral carries six extra planes, and a loader that assumes four
    fails at load_state_dict with a shape mismatch -- loudly, but only
    after the run is finished.
    """
    from fastwindterrain import training as T

    n = len(T.INPUT_CHANNELS)
    if ck.get("args", {}).get("spectral"):
        n += len(T.SPECTRAL_CHANNELS)
    return n


def load_run(run_dir, device):
    """Rebuild the model from a checkpoint, with the run's own settings."""
    import torch
    from fastwindterrain import models as M

    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu",
                    weights_only=False)
    a = ck["args"]
    kw = ({"width": a["width"], "modes": a["modes"],
           "blocks": a["blocks"]} if ck["arch"] in ("fno", "ufno")
          else {"width": a["width"]})
    model = M.build(ck["arch"], input_channels(ck), 27, **kw)
    model.load_state_dict(ck["state"])
    return model.to(device).eval(), ck


def main(argv=None):
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from fastwindterrain import evaluate as E
    from fastwindterrain import training as T
    import build_dataset as bd

    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, metavar="DIR",
                   help="a training run directory holding best.pt")
    p.add_argument("--data", default=os.path.join(ROOT, "data", "corpus"))
    p.add_argument("--fold", default="test")
    p.add_argument("--window", action="append", default=None, metavar="ID")
    p.add_argument("--direction", type=float, default=45.0,
                   help="must be one the dataset solved: 0, 45, 90 or 135")
    p.add_argument("--limit", type=int, default=2)
    p.add_argument("--out", default=None, metavar="DIR")
    args = p.parse_args(argv)

    out = args.out or os.path.join(ROOT, "data", "figures")
    os.makedirs(out, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cpu")
    model, ck = load_run(args.run, device)
    scales = np.asarray(ck["scales"])
    u_ref = float(ck.get("u_ref", corpus.REFERENCE_SPEED_MS))
    arch = ck["arch"]

    man = corpus.load_manifest()
    relief_of = {w["id"]: float(w["relief"]) for w in man["windows"]}

    made = []
    for info, a in bd.load_dataset(args.data, fold=args.fold):
        if info["derived"] or len(made) >= args.limit:
            continue
        wid, _, dtxt = info["id"].partition("@")
        if args.window and wid not in args.window:
            continue
        if float(dtxt) != args.direction:
            continue

        ds = T.LevelDataset([(info, a)], u_ref=u_ref,
                            window_m=corpus.WINDOW_M, scales=scales,
                            spectral=bool(ck["args"].get("spectral")))
        x, y = ds[0]
        with torch.no_grad():
            pred = model(x[None].to(device)).cpu().numpy()[0]

        lv = np.asarray(a["levels"], dtype=float)
        nlev = lv.size
        P = T.to_ms(pred, u_ref, scales).reshape(3, nlev, *pred.shape[1:])
        Y = T.to_ms(y.numpy(), u_ref, scales).reshape(3, nlev,
                                                      *pred.shape[1:])
        zt = np.asarray(a["terrain"], dtype=float)
        km = np.arange(zt.shape[0]) * corpus.WINDOW_M / zt.shape[0] / 1000.0

        rows = [int(np.argmin(np.abs(lv - h))) for h in SHOW_LEVELS]
        fig, axes = plt.subplots(len(rows), 3, figsize=(13.0, 4.1 * len(rows)),
                                 constrained_layout=True)
        for r, k in enumerate(rows):
            sp_t = E.speed(Y[0, k], Y[1, k])
            sp_p = E.speed(P[0, k], P[1, k])
            err = np.sqrt((P[0, k] - Y[0, k]) ** 2 + (P[1, k] - Y[1, k]) ** 2
                          + (P[2, k] - Y[2, k]) ** 2)
            # Solver and surrogate on ONE scale. Two fields on separate
            # scales look far more alike than they are, and that is the
            # single easiest way to make a figure lie.
            vmin = float(min(sp_t.min(), sp_p.min()))
            vmax = float(max(sp_t.max(), sp_p.max()))

            for c, (field, title, cmap, lo, hi) in enumerate((
                    (sp_t, "solver", "viridis", vmin, vmax),
                    (sp_p, f"surrogate ({arch})", "viridis", vmin, vmax),
                    (err, "difference", "magma_r", 0.0,
                     float(np.percentile(err, 99.5))))):
                ax = axes[r, c]
                im = ax.pcolormesh(km, km, field, cmap=cmap, vmin=lo,
                                   vmax=hi, shading="auto")
                ax.contour(km, km, zt, levels=8, colors="0.9"
                           if c < 2 else "0.35", linewidths=0.4, alpha=0.8)
                ax.set_aspect("equal")
                ax.tick_params(labelsize=7)
                if c == 0:
                    ax.set_ylabel(f"{lv[k]:.0f} m AGL\nkm", fontsize=9)
                if r == len(rows) - 1:
                    ax.set_xlabel("km", fontsize=8)
                if r == 0:
                    ax.set_title(title, fontsize=11)
                fig.colorbar(im, ax=ax, shrink=0.8).ax.tick_params(
                    labelsize=7)
            axes[r, 2].set_title(
                f"RMS {np.sqrt((err ** 2).mean()):.2f} m/s   "
                f"peak {err.max():.2f}", fontsize=9)

        fig.suptitle(
            f"{wid}   wind from {float(dtxt):.0f} deg at "
            f"{u_ref:.0f} m/s      relief {relief_of.get(wid, 0):.0f} m\n"
            f"horizontal wind speed [m/s]; solver and surrogate share a "
            f"colour scale per row; grey lines are terrain", fontsize=12)
        path = os.path.join(
            out, f"predict_{arch}_{wid.replace(':', '_')}"
                 f"_{float(dtxt):03.0f}.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        made.append(path)
        print(f"{path}")

    if not made:
        print("nothing matched -- --direction must be one the dataset "
              "solved (0, 45, 90 or 135); the rest are derived.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
