#!/usr/bin/env python3
"""
coherence_study.py -- how much of the wind does terrain explain, per scale?

The second half of the operator characterisation. cases/slope_error.py
asked what the error correlates with in SPACE; this asks what the field
itself correlates with in WAVENUMBER, which is the question linearised
flow theory has answered analytically since Jackson and Hunt (1975) and
which operational models like WAsP's flow model still rest on: transform
the terrain, multiply by a transfer function per wavenumber, transform
back.

Two quantities, accumulated over the corpus and binned radially in |k|:

    T(k, z)      admittance, <U H*> / <|H|^2>. The linear response of the
                 wind to terrain, per scale and height.
    gamma^2(k,z) coherence, |<U H*>|^2 / (<|U|^2><|H|^2>). The FRACTION of
                 wind variance at wavenumber k explained by terrain at the
                 SAME wavenumber. This is the one that matters.

WHAT THE COHERENCE DECIDES. Near 1: the operator is linear and diagonal
in wavenumber, a transfer function would reproduce the field for nothing,
and the network's job is only the residual. Well below 1: either the
response is nonlinear in terrain amplitude, or terrain at one scale drives
wind at another. The second case is the mechanism behind the architecture
result -- an FNO's spectral layer multiplies each mode independently, so
it is diagonal in wavenumber by construction and STRUCTURALLY cannot
represent cross-scale coupling within a layer. Low coherence would turn
"U-FNO underperforms here" from an empirical finding into an explained
one.

WHY NOT A 3D TRANSFORM. The vertical coordinate is terrain-following and
geometrically stretched, so transforming along it folds the coordinate
distortion into the spectrum. Height is a parameter here, not a transform
axis: one 2D transform per AGL level.

THE CONTROL. w is known analytically near the surface -- the kinematic
condition w = u.grad(h) becomes w(k) = i (u.k) h(k), so w must be coherent
with terrain with a 90 degree phase and an amplitude rising linearly in k.
If the machinery cannot recover that, nothing else it reports is worth
reading. It is checked and printed first.

Usage:

    python3 cases/coherence_study.py
    python3 cases/coherence_study.py --limit 100 --out data/figures
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

#: Radial wavenumber bins as WAVELENGTHS in metres, coarse to fine. The
#: domain is 5 km and the grid 50 m, so 5000 m is the longest resolvable
#: and 100 m the Nyquist wavelength.
LAMBDA_EDGES = (5000.0, 2500.0, 1250.0, 800.0, 500.0, 320.0, 200.0, 100.0)


def prepared(h):
    """Detrend and window a terrain tile before transforming.

    Without this the transform of a tilted, non-periodic tile is dominated
    by the step at its edges and every spectrum describes the window
    rather than the ground.
    """
    import numpy as np

    ny, nx = h.shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
    A = np.stack([np.ones(h.size), xx.ravel(), yy.ravel()], axis=1)
    coef, *_ = np.linalg.lstsq(A, h.ravel(), rcond=None)
    d = h - (A @ coef).reshape(h.shape)
    w = np.hanning(ny)[:, None] * np.hanning(nx)[None, :]
    return d * w, w


def main(argv=None):
    import numpy as np
    import build_dataset as bd

    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(ROOT, "data", "corpus"))
    p.add_argument("--fold", default="train")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--direction", type=float, default=0.0,
                   help="one direction at a time. Mixing them would put "
                        "the same grid wavenumber at different angles to "
                        "the wind in different samples, and the ensemble "
                        "average would be over two different things.")
    p.add_argument("--out", default=None, metavar="DIR",
                   help="write a figure here as well as the tables")
    args = p.parse_args(argv)

    edges = np.asarray(LAMBDA_EDGES, dtype=float)
    nb = edges.size - 1
    acc = None
    lv_ref = None
    n = 0
    t0 = time.time()

    for info, a in bd.load_dataset(args.data, fold=args.fold):
        if info["derived"] or float(info["direction"]) != args.direction:
            continue
        if n >= args.limit:
            break
        n += 1

        zt = np.asarray(a["terrain"], dtype=np.float64)
        ny, nx = zt.shape
        dx = dy = corpus.WINDOW_M / nx
        hw, win = prepared(zt)
        H = np.fft.fft2(hw)

        ky = np.fft.fftfreq(ny, d=dy)[:, None]
        kx = np.fft.fftfreq(nx, d=dx)[None, :]
        k = np.sqrt(kx ** 2 + ky ** 2)
        lam = np.divide(1.0, k, out=np.full_like(k, np.inf), where=k > 0)
        # Bin index per mode; -1 for the ones outside the resolved band.
        idx = np.full(k.shape, -1, dtype=int)
        for b in range(nb):
            idx[(lam <= edges[b]) & (lam > edges[b + 1])] = b

        idx_ref = idx
        theta = np.radians(float(info["direction"]))
        ex, ey = -np.sin(theta), -np.cos(theta)     # the flow direction

        lv = np.asarray(a["levels"], dtype=np.float64)
        if acc is None:
            lv_ref = lv
            nl = lv.size
            # PER MODE, not per bin. Forming the coherence after radially
            # averaging the cross-spectrum cancels any response whose
            # phase varies within the bin -- which is exactly what the
            # kinematic w does, since its phase flips with the sign of
            # u.k. Ensemble-average each mode over samples first, bin the
            # coherence afterwards.
            acc = {key: np.zeros((nl, ny, nx), dtype=np.complex128)
                   for key in ("uH", "vH", "wH")}
            acc.update({key: np.zeros((nl, ny, nx)) for key in
                        ("uu", "vv", "ww")})
            acc["hh"] = np.zeros((nl, ny, nx))

        for kk in range(lv.size):
            u = np.asarray(a["u_lev"][kk], dtype=np.float64)
            v = np.asarray(a["v_lev"][kk], dtype=np.float64)
            w = np.asarray(a["w_lev"][kk], dtype=np.float64)
            # Along- and cross-wind, so the response is in the frame the
            # physics is written in rather than the grid's.
            par = u * ex + v * ey
            per = -u * ey + v * ex
            # Perturbation: remove the level mean, which IS the
            # undisturbed part at an AGL level.
            F = {"u": np.fft.fft2((par - par.mean()) * win),
                 "v": np.fft.fft2((per - per.mean()) * win),
                 "w": np.fft.fft2((w - w.mean()) * win)}
            acc["hh"][kk] += (H * H.conj()).real
            for name in ("u", "v", "w"):
                acc[name + "H"][kk] += F[name] * H.conj()
                acc[name + name][kk] += (F[name] * F[name].conj()).real

    if acc is None:
        print(f"no solved samples at direction {args.direction:.0f}",
              file=sys.stderr)
        return 1

    centres = np.sqrt(edges[:-1] * edges[1:])        # geometric midpoints
    kmid = 1.0 / centres

    def _bin(field):
        """Average a per-mode quantity into the radial wavelength bins."""
        out = np.zeros((lv_ref.size, nb))
        for b in range(nb):
            m = idx_ref == b
            if m.any():
                out[:, b] = field[:, m].mean(axis=1)
        return out

    def coherence(name):
        num = np.abs(acc[name + "H"]) ** 2
        den = acc[name + name] * acc["hh"]
        per_mode = np.divide(num, den, out=np.zeros_like(num),
                             where=den > 1e-300)
        return _bin(per_mode)

    def admittance(name):
        per_mode = np.abs(acc[name + "H"]) / np.maximum(acc["hh"], 1e-300)
        return _bin(per_mode)

    print(f"{n} samples, fold '{args.fold}', direction "
          f"{args.direction:.0f} deg, {time.time()-t0:.0f} s")
    print(f"wavelength bins [m]: "
          + "  ".join(f"{c:.0f}" for c in centres) + "\n")

    # -- the control, first -------------------------------------------
    gw = coherence("w")
    Tw = admittance("w")
    print("CONTROL -- w against terrain. The kinematic condition makes")
    print("w(k) = i (u.k) h(k) near the surface, so coherence should be")
    print("high and the admittance should rise LINEARLY in k.\n")
    k0 = 0
    print(f"{'level':>8s}  " + "  ".join(f"{c:>7.0f}" for c in centres))
    print("-" * (10 + 9 * nb))
    for kk in (0, 1, 3):
        if kk < lv_ref.size:
            print(f"{lv_ref[kk]:>6.0f} m  "
                  + "  ".join(f"{gw[kk, b]:7.3f}" for b in range(nb)))
    slope = np.polyfit(np.log(kmid), np.log(np.maximum(Tw[k0], 1e-30)), 1)[0]
    print(f"\n  admittance |T_w| slope in log-log at {lv_ref[k0]:.0f} m: "
          f"{slope:+.2f}   (kinematic theory says +1)")

    # -- the result ----------------------------------------------------
    for name, label in (("u", "ALONG-WIND"), ("v", "CROSS-WIND")):
        g = coherence(name)
        print(f"\n{label} coherence gamma^2 -- fraction of wind variance "
              f"at k explained by terrain at k")
        print(f"{'level':>8s}  " + "  ".join(f"{c:>7.0f}" for c in centres))
        print("-" * (10 + 9 * nb))
        for kk in range(lv_ref.size):
            print(f"{lv_ref[kk]:>6.0f} m  "
                  + "  ".join(f"{g[kk, b]:7.3f}" for b in range(nb)))

    # -- the exp(-kz) test ---------------------------------------------
    Tu = admittance("u")
    print("\nDECAY WITH HEIGHT. Potential flow gives exp(-k z), so terrain")
    print("of wavelength L should stop mattering above about L/2pi.")
    print(f"{'wavelength':>11s} {'L/2pi':>7s} {'measured e-fold':>16s}")
    print("-" * 38)
    for b in range(nb):
        col = Tu[:, b]
        good = col > 0
        if good.sum() < 3:
            continue
        # Fit ln|T| against z over the band levels only; aloft levels
        # scale with the column and would dominate the fit.
        sel = good & (lv_ref <= 200.0)
        if sel.sum() < 3:
            continue
        s = np.polyfit(lv_ref[sel], np.log(col[sel]), 1)[0]
        efold = -1.0 / s if s < 0 else float("inf")
        print(f"{centres[b]:11.0f} {centres[b]/(2*np.pi):7.0f} "
              f"{efold:16.0f}")

    if args.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(args.out, exist_ok=True)
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 5.2),
                                       constrained_layout=True)
        shades = plt.cm.viridis(np.linspace(0.05, 0.9, lv_ref.size))
        gu = coherence("u")
        for kk in range(lv_ref.size):
            axA.semilogx(centres, gu[kk], marker="o", ms=4, lw=1.8,
                         color=shades[kk], label=f"{lv_ref[kk]:.0f} m")
        axA.set_xlabel("terrain wavelength [m]")
        axA.set_ylabel(r"coherence $\gamma^2$")
        axA.set_title("along-wind coherence with terrain", fontsize=11)
        axA.set_ylim(0, 1)
        axA.invert_xaxis()
        axA.grid(color="0.92", lw=0.6)
        axA.set_axisbelow(True)
        axA.legend(fontsize=7, ncol=2, title="level AGL")

        for b in range(nb):
            axB.semilogy(lv_ref, np.maximum(Tu[:, b], 1e-12), marker="o",
                         ms=4, lw=1.6,
                         color=plt.cm.plasma(b / max(nb - 1, 1)),
                         label=f"{centres[b]:.0f} m")
        axB.set_xlabel("height above ground [m]")
        axB.set_ylabel(r"admittance $|T_u|$")
        axB.set_title("how terrain influence decays with height",
                      fontsize=11)
        axB.set_xscale("log")
        axB.grid(color="0.92", lw=0.6)
        axB.set_axisbelow(True)
        axB.legend(fontsize=7, ncol=2, title="wavelength")
        fig.suptitle(f"Terrain-wind spectral response, {n} samples, "
                     f"'{args.fold}' fold", fontsize=12)
        path = os.path.join(args.out, f"coherence_{args.fold}.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        print(f"\n{path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
