#!/usr/bin/env python3
"""
unseen_report.py -- the unseen sites, side by side.

Five places the model never trained on, each a different terrain form:
a Klamath-Siskiyou canyon, a Front Range slope, a plateau cut by a deep
confined channel, a Portuguese double ridge, and the foot of the Rockies.
Scored one at a time they are five numbers; scored side by side they say
whether the model's difficulty follows the TERRAIN or follows the fact
that it has not seen the place, which is the question the whole held-out
design exists to answer.

Two figures.

*maps* puts one level across every site on a common error scale. The
solver and surrogate rows keep a per-site scale, because a Gorge jet and
a Perdigao ridge do not share a speed range and forcing them to would
make four of the five unreadable. The error row is common, and that is
the row to read across.

*histogram* is the distribution behind those maps. A mean of 0.4 m/s can
be a uniform 0.4 or a quiet field with a bad two per cent, and for a fire
model downstream those are different problems. The cumulative form
answers the question a user actually asks: what fraction of the domain is
within my tolerance?

Usage:

    python3 cases/unseen_report.py --run data/runs/unet_conv
    python3 cases/unseen_report.py --run data/runs/unet_conv --level 10
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import corpus                                               # noqa: E402
from predict_maps import load_run, dataset_kwargs           # noqa: E402

#: Which datasets hold unseen terrain, and under which fold. Flatirons is
#: kept separate from the demo fold on purpose -- it clears a weaker bar
#: (see corpus.MEASUREMENT_SITES) -- but for a picture of how the model
#: behaves on ground it never saw, it belongs beside the others.
SOURCES = (("data/demo", "demo"), ("data/flatirons", "measurement"))

#: Error bins for the histogram, m/s. 0.25 is the CFD tolerance and is
#: marked; the rest double.
BINS = (0.25, 0.5, 1.0, 2.0)


def collect(args, run=None):
    """One representative sample per site: predicted and true levels.

    ``run`` overrides ``args.run``, so the same sites can be scored with a
    second model for a before-and-after comparison. The input flags come
    from each checkpoint's own arguments rather than from a fixed list --
    the two models here differ by an input field, and building the second
    one's inputs to the first one's recipe would score it on data it never
    saw.
    """
    import numpy as np
    import torch
    from fastwindterrain import training as T
    import build_dataset as bd

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cpu")
    model, ck = load_run(run or args.run, device)
    scales = np.asarray(ck["scales"])
    u_ref = float(ck.get("u_ref", corpus.REFERENCE_SPEED_MS))

    out = {}
    for path, fold in SOURCES:
        full = os.path.join(ROOT, path)
        if not os.path.isdir(full):
            print(f"skipping {path}: not generated", file=sys.stderr)
            continue
        for info, a in bd.load_dataset(full, fold=fold):
            if info["derived"] or float(info["direction"]) != args.direction:
                continue
            site = info["id"].split(":")[0]
            ds = T.LevelDataset([(info, a)], u_ref=u_ref,
                                window_m=corpus.WINDOW_M, scales=scales,
                                **dataset_kwargs(ck))
            x, y = ds[0]
            with torch.no_grad():
                pred = model(x[None].to(device)).cpu().numpy()[0]
            lv = np.asarray(a["levels"], dtype=float)
            n = lv.size
            P = T.to_ms(pred, u_ref, scales).reshape(3, n, *pred.shape[1:])
            Y = T.to_ms(y.numpy(), u_ref, scales).reshape(3, n,
                                                          *pred.shape[1:])
            err = np.sqrt(((P - Y) ** 2).sum(axis=0))
            rec = out.setdefault(site, {"err": [], "levels": lv})
            rec["err"].append(err)
            # Keep the first window of each site for the map.
            if "P" not in rec:
                rec.update(P=P, Y=Y, terrain=np.asarray(a["terrain"]),
                           id=info["id"])
    return out, ck


def main(argv=None):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from fastwindterrain import evaluate as E

    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, metavar="DIR")
    p.add_argument("--level", type=float, default=80.0,
                   help="height AGL for the maps, metres")
    p.add_argument("--direction", type=float, default=45.0)
    p.add_argument("--out", default=os.path.join(ROOT, "data", "figures"))
    # 130 dpi is enough to read on screen and is what the docs use. A
    # journal wants the labels sharp at print size, so the figures are
    # regenerated at a higher value for that purpose rather than being
    # upscaled afterwards.
    p.add_argument("--dpi", type=int, default=130)
    # Threshold for the right panel. 0.5 m/s is twice the CFD tolerance
    # and shows the tail on every level; 1.0 isolates the levels where
    # the error is large enough to matter to a model downstream.
    p.add_argument("--compare", default=None, metavar="DIR",
                   help="a second run, drawn alongside the first. The maps "
                        "then show the reference and the error of each "
                        "model on one common scale, and the histogram "
                        "draws the second model hatched over the first.")
    p.add_argument("--tail", type=float, default=0.5,
                   metavar="MS", help="tail threshold for the right "
                                      "panel, m/s")
    args = p.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    sites, ck = collect(args)
    if not sites:
        print("no unseen sites found", file=sys.stderr)
        return 1
    other, ck2 = (collect(args, args.compare) if args.compare
                  else (None, None))
    order = sorted(sites, key=lambda s: np.mean(
        [e.mean() for e in sites[s]["err"]]))
    arch = ck["arch"]

    # ---------------- maps -------------------------------------------
    km = None
    kidx = {s: int(np.argmin(np.abs(sites[s]["levels"] - args.level)))
            for s in order}
    emax = float(np.percentile(np.concatenate(
        [sites[s]["err"][0][kidx[s]].ravel() for s in order]), 99.0))

    if other is not None:
        emax = max(emax, float(np.percentile(np.concatenate(
            [other[s]["err"][0][kidx[s]].ravel() for s in order]), 99.0)))
    nrow = 4 if other is not None else 3
    fig, axes = plt.subplots(nrow, len(order),
                             figsize=(3.5 * len(order), 3.4 * nrow),
                             constrained_layout=True)
    for c, s in enumerate(order):
        d = sites[s]
        k = kidx[s]
        zt = d["terrain"]
        if km is None or len(km) != zt.shape[0]:
            km = np.arange(zt.shape[0]) * corpus.WINDOW_M / zt.shape[0] / 1e3
        sp_t = E.speed(d["Y"][0, k], d["Y"][1, k])
        sp_p = E.speed(d["P"][0, k], d["P"][1, k])
        err = d["err"][0][k]
        lo, hi = float(min(sp_t.min(), sp_p.min())), \
            float(max(sp_t.max(), sp_p.max()))

        panels = [(sp_t, "viridis", lo, hi), (sp_p, "viridis", lo, hi),
                  (err, "magma_r", 0.0, emax)]
        if other is not None:
            panels.append((other[s]["err"][0][kidx[s]], "magma_r",
                           0.0, emax))
        for r, (f, cmap, vlo, vhi) in enumerate(panels):
            ax = axes[r, c]
            im = ax.pcolormesh(km, km, f, cmap=cmap, vmin=vlo, vmax=vhi,
                               shading="auto")
            ax.contour(km, km, zt, levels=7,
                       colors="0.9" if r < 2 else "0.35",
                       linewidths=0.4, alpha=0.8)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=6)
            if r < 2:
                fig.colorbar(im, ax=ax, shrink=0.75).ax.tick_params(
                    labelsize=6)
            if r == 0:
                rel = float(zt.max() - zt.min())
                ax.set_title(f"{s}\nrelief {rel:.0f} m", fontsize=10)
            if r >= 2:
                ax.set_xlabel(f"RMS {np.sqrt((f**2).mean()):.2f} m/s",
                              fontsize=9)
        if c == 0:
            labs = ["solver", f"surrogate ({arch})", "difference"]
            if other is not None:
                labs[2] = "difference, without"
                labs.append("difference, with the anchor field")
            for r, lab in enumerate(labs):
                axes[r, 0].set_ylabel(lab + "\nkm", fontsize=10)
    fig.colorbar(im, ax=axes[2:, :], shrink=0.7, location="right",
                 label="vector error [m/s], COMMON scale")
    fig.suptitle(
        f"Unseen terrain at {args.level:.0f} m AGL, wind from "
        f"{args.direction:.0f} deg at {corpus.REFERENCE_SPEED_MS:.0f} m/s\n"
        f"solver and surrogate share a scale WITHIN each site; the "
        f"difference row{'s are' if other is not None else ' is'} common "
        f"across all of them", fontsize=12)
    tag = "_cmp" if other is not None else ""
    m1 = os.path.join(args.out, f"unseen_maps_{arch}{tag}_"
                                f"{args.level:.0f}m.png")
    fig.savefig(m1, dpi=args.dpi)
    plt.close(fig)
    print(m1)

    # ---------------- histogram ---------------------------------------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.4),
                                   constrained_layout=True)
    shades = plt.cm.viridis(np.linspace(0.05, 0.85, len(order)))
    for i, s in enumerate(order):
        e = np.concatenate([x.ravel() for x in sites[s]["err"]])
        xs = np.linspace(0.0, 2.5, 400)
        cdf = [(e <= t).mean() * 100.0 for t in xs]
        axL.plot(xs, cdf, lw=2.0, color=shades[i],
                 label=f"{s}  (mean {e.mean():.2f})")
        if other is not None:
            e2 = np.concatenate([x.ravel() for x in other[s]["err"]])
            axL.plot(xs, [(e2 <= t).mean() * 100.0 for t in xs], lw=1.6,
                     ls="--", color=shades[i])
    for b in BINS[:2]:
        axL.axvline(b, color="0.6", lw=1.0, ls="--")
    axL.text(0.25, 4, " 0.25 m/s\n CFD tolerance", fontsize=8, color="0.35")
    axL.set_xlim(0, 2.5)
    axL.set_ylim(0, 100)
    axL.set_xlabel("absolute vector error [m/s]")
    axL.set_ylabel("% of level cells at or below")
    axL.set_title("cumulative error distribution"
                  + (" (dashed: with the anchor field)"
                     if other is not None else ""), fontsize=11)
    axL.legend(fontsize=8, loc="lower right")
    axL.grid(color="0.92", lw=0.6)
    axL.set_axisbelow(True)

    width = 0.8 / len(order)
    lv = sites[order[0]]["levels"]
    xpos = np.arange(len(lv))
    for i, s in enumerate(order):
        e = np.stack([x for x in sites[s]["err"]])       # (win, nlev, y, x)
        frac = [(e[:, k] > args.tail).mean() * 100.0
                for k in range(e.shape[1])]
        axR.bar(xpos + i * width, frac, width, color=shades[i], label=s)
        if other is not None:
            e2 = np.stack([x for x in other[s]["err"]])
            f2 = [(e2[:, k] > args.tail).mean() * 100.0
                  for k in range(e2.shape[1])]
            axR.bar(xpos + i * width, f2, width * 0.55, color=shades[i],
                    hatch="////", edgecolor="white", linewidth=0.4)
    axR.set_xticks(xpos + 0.4 - width / 2)
    axR.set_xticklabels([f"{h:.0f}" for h in lv], fontsize=8)
    axR.set_xlabel("level [m AGL]")
    axR.set_ylabel(f"% of cells over {args.tail:g} m/s")
    axR.set_title("where the tail lives, per level"
                  + (" (hatched: with the anchor field)"
                     if other is not None else ""), fontsize=11)
    axR.legend(fontsize=8)
    axR.grid(axis="y", color="0.92", lw=0.6)
    axR.set_axisbelow(True)

    fig.suptitle(f"Unseen-terrain error, {arch}, "
                 f"{sum(len(sites[s]['err']) for s in order)} windows "
                 f"across {len(order)} sites", fontsize=12)
    m2 = os.path.join(args.out, f"unseen_hist_{arch}{tag}.png")
    fig.savefig(m2, dpi=args.dpi)
    plt.close(fig)
    print(m2)

    print(f"\n{'site':16s} {'mean':>7s} {'p95':>7s} "
          + "  ".join(f"<{b}".rjust(7) for b in BINS))
    print("-" * 62)
    for s in order:
        e = np.concatenate([x.ravel() for x in sites[s]["err"]])
        print(f"{s:16s} {e.mean():7.3f} {np.percentile(e, 95):7.3f} "
              + "  ".join(f"{(e <= b).mean()*100:6.1f}%" for b in BINS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
