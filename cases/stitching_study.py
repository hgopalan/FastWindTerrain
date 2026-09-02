#!/usr/bin/env python3
"""
stitching_study.py -- how much does 2D-to-3D reconstruction cost?

The surrogate predicts wind on a few horizontal levels and rebuilds the
3D field from them, because training in 3D is expensive. This measures
what that rebuild costs on its own, with NO machine learning anywhere:
take a solved field, throw away everything except u and v on K levels,
stitch it back, and compare against the field that was deleted.

The number that comes out is a CEILING. If reconstructing from perfect
levels loses 20% of the field, no network trained to predict those levels
can do better than 20%. Running this before any training is the cheapest
way to find out whether the architecture can work.

What it varies:

  K        how many levels, from the engineering heights alone up to a
           dense stack
  frame    agl (a fixed height above ground) or cartesian (a fixed
           elevation, so the slice cuts through terrain)
  method   linear in z, or linear in log(z/z0) -- the surface layer is
           logarithmic, so the second should win near the ground
  w        interpolated like u and v, or rebuilt from continuity by the
           O'Brien adjustment
  w0       the vertical velocity at the surface: zero, or the kinematic
           condition u.grad(h), which is what terrain-following flow does

Usage:
    python3 cases/stitching_study.py                       # every case
    python3 cases/stitching_study.py --case creek_fire
    python3 cases/stitching_study.py --figure study.png

Run prepare.py for a case first -- terrain files are not committed.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import casegen  # noqa: E402

import fastwindterrain as fwt              # noqa: E402
from fastwindterrain import levels as lv   # noqa: E402

# The operator the dataset is frozen at -- see docs/surrogate.rst. Four
# passes is not a converged projection; it is a stated, consistent one.
N_PROJECTIONS = 4
WIND_SPEED = 8.0
WIND_DIRECTION = 225.0

#: Level sets, coarse to dense. The first is the engineering heights on
#: their own, which is the natural thing to try and, on this grid, covers
#: only the bottom third of the column.
LEVEL_SETS = {
    "eng-5": lv.ENGINEERING_LEVELS,
    "eng+1": lv.ENGINEERING_LEVELS + (600.0,),
    "eng+3": lv.DEFAULT_LEVELS,
    "dense-12": (10.0, 30.0, 60.0, 80.0, 100.0, 120.0, 160.0,
                 250.0, 400.0, 700.0, 1100.0, 1600.0),
}


def reconstruct(truth, geom, level_set, frame="agl", method="loglinear",
                w_source="obrien", w0="kinematic"):
    """Rebuild a 3D field from K levels of the true u and v."""
    u_t, v_t, w_t = truth
    z_cc, zt, mask, dz, dx, dy = geom
    levels = list(level_set)

    ku = lv.extract_levels(u_t, z_cc, zt, levels, mask=mask, frame=frame)
    kv = lv.extract_levels(v_t, z_cc, zt, levels, mask=mask, frame=frame)
    u = lv.stitch_levels(ku, levels, z_cc, zt, mask=mask, frame=frame,
                         method=method)
    v = lv.stitch_levels(kv, levels, z_cc, zt, mask=mask, frame=frame,
                         method=method)

    if w_source == "interp":
        kw = lv.extract_levels(w_t, z_cc, zt, levels, mask=mask, frame=frame)
        w = lv.stitch_levels(kw, levels, z_cc, zt, mask=mask, frame=frame,
                             method="linear")     # w is not logarithmic
    else:
        seed = (lv.surface_kinematic_w(u, v, zt, dx, dy, mask)
                if w0 == "kinematic" else np.zeros_like(u))
        w = lv.obrien_w(u, v, seed, dz, dx, dy, mask)
    return np.stack([u, v, w])


def errors(recon, truth, mask, engineering_k=None):
    """Relative errors over fluid cells, scaled by the true |U| maximum."""
    fluid = mask == 0
    scale = float(np.sqrt((truth ** 2).sum(axis=0))[fluid].max())
    out = {}
    for n, name in enumerate("uvw"):
        d = np.abs(recon[n] - truth[n])[fluid]
        out[f"rmse_{name}"] = float(np.sqrt((d ** 2).mean())) / scale
        out[f"max_{name}"] = float(d.max()) / scale
    sp_r = np.sqrt((recon ** 2).sum(axis=0))[fluid]
    sp_t = np.sqrt((truth ** 2).sum(axis=0))[fluid]
    out["rmse_speed"] = float(np.sqrt(((sp_r - sp_t) ** 2).mean())) / scale
    if engineering_k is not None:
        d = np.abs(recon[:2] - truth[:2])
        sel = engineering_k & fluid[None]
        out["rmse_uv_eng"] = float(np.sqrt((d[np.broadcast_to(
            sel, d.shape)] ** 2).mean())) / scale
    return out


def divergence_of(solver, field):
    """max|div| of a reconstructed field, measured by the solver itself."""
    solver.set_velocity(field)
    return float(solver.max_divergence_fe)


def run_case(case, stream=sys.stdout):
    cfg = case.config(wind_speed=WIND_SPEED, wind_direction=WIND_DIRECTION,
                      n_projections=N_PROJECTIONS)
    s = fwt.Solver(cfg)
    s.setup()
    s.solve()
    s.diagnose()

    g = s.grid
    z_cc = np.asarray(g.z_cc)
    dz = np.diff(np.asarray(g.z_face))
    dx = (g.prob_hi[0] - g.prob_lo[0]) / g.nx
    dy = (g.prob_hi[1] - g.prob_lo[1]) / g.ny
    mask = s.mask
    zt = s.z_terrain
    geom = (z_cc, zt, mask, dz, dx, dy)

    truth = np.stack(s.velocity)
    div_truth = float(s.max_divergence_fe)

    # Cells within the engineering band, where the answer is actually used.
    agl = lv.height_above_ground(z_cc, zt)
    eng_k = (agl >= lv.ENGINEERING_LEVELS[0]) & (agl <= lv.ENGINEERING_LEVELS[-1])

    print(f"\n=== {case.name} ===", file=stream)
    print(f"  grid {s.shape}, {int((mask == 1).sum())} solid, "
          f"terrain {float(np.asarray(zt)[0].min()):.0f}-"
          f"{float(np.asarray(zt)[0].max()):.0f} m ASL", file=stream)
    print(f"  truth max|div| {div_truth:.5f} 1/s", file=stream)

    rows = []

    def report(tag, **kw):
        r = reconstruct(truth, geom, **kw)
        e = errors(r, truth, mask, eng_k)
        e["div"] = divergence_of(s, r)
        e["tag"] = tag
        rows.append(e)
        print(f"  {tag:34s} rmse|U| {e['rmse_speed']:.4f}  "
              f"uv@eng {e['rmse_uv_eng']:.4f}  "
              f"rmse_w {e['rmse_w']:.4f}  max|div| {e['div']:.4f}",
              file=stream)
        s.set_velocity(truth)          # leave the solver holding the truth
        return e

    print("\n  -- how many levels (agl, loglinear, O'Brien + kinematic w0) --",
          file=stream)
    for name, ls in LEVEL_SETS.items():
        report(f"{name} ({len(ls)} levels)", level_set=ls)

    print("\n  -- ablations at eng+3 (8 levels) --", file=stream)
    base = lv.DEFAULT_LEVELS
    report("frame = cartesian", level_set=base, frame="cartesian")
    report("method = linear", level_set=base, method="linear")
    report("w interpolated, not O'Brien", level_set=base, w_source="interp")
    report("w0 = 0, not kinematic", level_set=base, w0="zero")

    return rows, div_truth


# ---------------------------------------------------------------------------
# Where to put the levels, at a fixed count
# ---------------------------------------------------------------------------

#: The nominal top of the band worth resolving, in metres above ground.
#: The catalogue's domains reach 1000 m above their highest terrain, so
#: this spans essentially the whole column without depending on a
#: particular case's relief.
PLACEMENT_TOP = 1600.0
PLACEMENT_BASE = 10.0          # the lowest resolvable level on a 4 m grid


def placements(k):
    """Level sets of the same size, distributed by different rules.

    The count is what a paper usually reports and the placement is what
    actually has to be reproduced, so at fixed k these are the choice a
    reader has to make. A rule that transfers is worth more than a list
    that does not.
    """
    lo, hi = PLACEMENT_BASE, PLACEMENT_TOP
    out = {
        "uniform": tuple(np.linspace(lo, hi, k)),
        "log": tuple(np.geomspace(lo, hi, k)),
        "quadratic": tuple(lo + (hi - lo) * (np.linspace(0, 1, k) ** 2)),
    }
    # Anchored on the heights people actually ask for, with whatever is
    # left log-spaced above them. This is the practical option: it keeps
    # the engineering levels exact instead of landing near them.
    eng = list(lv.ENGINEERING_LEVELS)
    if k >= len(eng):
        extra = k - len(eng)
        fill = (list(np.geomspace(eng[-1] * 1.6, hi, extra))
                if extra else [])
        out["engineering-anchored"] = tuple(eng + fill)
    return out


def splits(k):
    """Level sets that divide a fixed budget between the band and aloft.

    The engineering band is where the answer is wanted and the column
    above is what has to be spanned for a 3D reconstruction, so a fixed
    number of levels has to be shared between them. This is that trade-off
    made explicit: n_band log-spaced across 10-160 m, the rest log-spaced
    from there to the top.

    Stated as a split rather than as a list, because the split is the
    part that transfers to a different grid.
    """
    out = {}
    for n_band in range(2, k):
        n_aloft = k - n_band
        band = list(np.geomspace(PLACEMENT_BASE, 160.0, n_band))
        aloft = list(np.geomspace(160.0, PLACEMENT_TOP, n_aloft + 1))[1:]
        out[f"{n_band} band + {n_aloft} aloft"] = tuple(band + aloft)
    out["pure log 10-1600"] = tuple(np.geomspace(PLACEMENT_BASE,
                                                 PLACEMENT_TOP, k))
    return out


def split_case(case, stream=sys.stdout):
    """How to divide k levels between 10-160 m and the column above."""
    cfg = case.config(wind_speed=WIND_SPEED, wind_direction=WIND_DIRECTION,
                      n_projections=N_PROJECTIONS)
    s = fwt.Solver(cfg)
    s.setup(); s.solve(); s.diagnose()

    g = s.grid
    z_cc = np.asarray(g.z_cc)
    geom = (z_cc, s.z_terrain, s.mask, np.diff(np.asarray(g.z_face)),
            (g.prob_hi[0] - g.prob_lo[0]) / g.nx,
            (g.prob_hi[1] - g.prob_lo[1]) / g.ny)
    truth = np.stack(s.velocity)
    agl = lv.height_above_ground(z_cc, s.z_terrain)
    eng_k = ((agl >= lv.ENGINEERING_LEVELS[0])
             & (agl <= lv.ENGINEERING_LEVELS[-1]))

    print(f"\n=== {case.name} -- how to split the level budget ===",
          file=stream)
    for k in (8, 12):
        rows = []
        for name, levels in splits(k).items():
            e = errors(reconstruct(truth, geom, level_set=levels),
                       truth, s.mask, eng_k)
            rows.append((name, e["rmse_speed"], e["rmse_uv_eng"], levels))
        best_col = min(rows, key=lambda r: r[1])[0]
        best_band = min(rows, key=lambda r: r[2])[0]
        print(f"\n  k = {k}", file=stream)
        for name, col, band, levels in rows:
            marks = ("<- best column" if name == best_col else "") + \
                    (" <- best band" if name == best_band else "")
            print(f"    {name:22s} rmse|U| {col:.4f}   uv@eng {band:.4f} "
                  f"  {marks}", file=stream)
        print(f"    best column: {best_col};  best band: {best_band}",
              file=stream)
    return None


def placement_case(case, stream=sys.stdout):
    """Same k, different placements, on one solved field."""
    cfg = case.config(wind_speed=WIND_SPEED, wind_direction=WIND_DIRECTION,
                      n_projections=N_PROJECTIONS)
    s = fwt.Solver(cfg)
    s.setup(); s.solve(); s.diagnose()

    g = s.grid
    z_cc = np.asarray(g.z_cc)
    geom = (z_cc, s.z_terrain, s.mask, np.diff(np.asarray(g.z_face)),
            (g.prob_hi[0] - g.prob_lo[0]) / g.nx,
            (g.prob_hi[1] - g.prob_lo[1]) / g.ny)
    truth = np.stack(s.velocity)
    agl = lv.height_above_ground(z_cc, s.z_terrain)
    eng_k = ((agl >= lv.ENGINEERING_LEVELS[0])
             & (agl <= lv.ENGINEERING_LEVELS[-1]))

    print(f"\n=== {case.name} -- placement at fixed level count ===",
          file=stream)
    results = {}
    for k in (5, 8, 12):
        print(f"\n  k = {k}", file=stream)
        for name, levels in placements(k).items():
            r = reconstruct(truth, geom, level_set=levels)
            e = errors(r, truth, s.mask, eng_k)
            results[(k, name)] = e
            shown = ", ".join(f"{x:.0f}" for x in levels)
            print(f"    {name:22s} rmse|U| {e['rmse_speed']:.4f}   "
                  f"uv@eng {e['rmse_uv_eng']:.4f}   [{shown}]", file=stream)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", action="append", default=None, metavar="SLUG",
                   help="case to study (repeatable); default: every prepared one")
    p.add_argument("--figure", default=None, metavar="PATH",
                   help="write an error-vs-levels figure here")
    p.add_argument("--placement", action="store_true",
                   help="sweep level PLACEMENT at fixed count instead")
    p.add_argument("--split", action="store_true",
                   help="sweep how a fixed level budget divides between the "
                        "10-160 m band and the column above")
    args = p.parse_args()

    wanted = args.case or [c.slug for c in casegen.catalogue()]
    cases = []
    for slug in wanted:
        c = casegen.load(slug)
        if os.path.isfile(c.terrain_path):
            cases.append(c)
        else:
            print(f"[skip] {slug}: no terrain, run "
                  f"cases/{slug}/prepare.py", file=sys.stderr)
    if not cases:
        print("no prepared cases", file=sys.stderr)
        return 1

    results = {}
    with fwt.session():
        for c in cases:
            if args.split:
                split_case(c)
            elif args.placement:
                placement_case(c)
            else:
                results[c.name] = run_case(c)
    if args.placement or args.split:
        return 0

    if args.figure:
        plot(results, args.figure)
        print(f"\nwrote {args.figure}")
    return 0


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(LEVEL_SETS)
    counts = [len(LEVEL_SETS[n]) for n in names]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))

    for case_name, (rows, div_truth) in results.items():
        sweep = rows[:len(names)]
        a1.plot(counts, [r["rmse_speed"] for r in sweep], marker="o",
                label=case_name)
        a2.plot(counts, [r["rmse_uv_eng"] for r in sweep], marker="o",
                label=case_name)

    for ax, title in ((a1, "whole column"), (a2, "engineering band, 10-160 m")):
        ax.set_xlabel("levels predicted")
        ax.set_ylabel("reconstruction RMSE / |U|max")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    a1.legend(frameon=False, fontsize=9)
    fig.suptitle("2D-to-3D reconstruction error from perfect levels "
                 "-- the ceiling on any surrogate", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=145)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
