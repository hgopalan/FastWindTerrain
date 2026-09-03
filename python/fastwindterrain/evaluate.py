"""
evaluate -- how a predicted wind field is scored.

Phase 22a. Written and validated BEFORE any model exists, because a metric
and a model that arrive together cannot be told apart: the first
disappointing number is then ambiguous between "the network is bad" and
"the scoring is wrong", and there is no way to settle it. Everything here
is checked against fields whose error is already known from a different
code path -- the analytical baselines and the extract/stitch round trip --
so that when a network's number arrives, only the network is new.

FOUR DECISIONS, EACH ONE MEASURED.

*Metres per second, not a normalised residual.* The tolerance in this
field is physical: roughly 0.25 m/s in CFD practice and 20-30 % on real
turbulent flow. A relative L2 of 0.03 cannot be compared against either
without knowing the scale, so it hides the only judgement that matters.
Normalised forms are reported alongside, never alone.

*Vector error, with speed error beside it.* ``|u_pred - u_ref|`` is the
honest number: a field with the right speed and the wrong direction is
wrong, and a speed-only metric scores it perfectly. Speed error is
reported too because it is what a wind engineer reads, but the vector
error is the one that can fail.

*Grouped by terrain, never aggregated to one number.* The reconstruction
ceiling varies about fourfold between the gentlest window and the
steepest, and the baseline varies sixfold. A single mean sits in the
middle and describes no window in the corpus.

*Divergence is a diagnostic, not a loss.* The targets carry 0.006-0.087
1/s by construction -- the solver stops at a finite number of projection
passes -- so a PDE residual term would train the network against its own
data. ``divergence`` exists here to report, and there is deliberately no
function that turns it into an objective.
"""

import numpy as np

from .levels import BAND_BASE_M, BAND_TOP_M

__all__ = [
    "RELIEF_BINS",
    "fluid_from_k_first",
    "error_stats",
    "speed",
    "level_errors",
    "band_of",
    "skill",
    "group_by_relief",
    "table",
]

#: Relief bins, in metres, for the grouped report. Chosen from the corpus's
#: own distribution -- quartiles of the 270 windows sit near 280, 520 and
#: 680 m -- and rounded, so each bin holds a useful number of windows
#: instead of one bin holding nearly all of them.
#:
#: The names matter more than the edges. "gentle" and "complex" appear in
#: the paper's claims, and they should mean one fixed thing across every
#: table rather than being redefined per figure.
RELIEF_BINS = (
    ("gentle", 0.0, 200.0),
    ("moderate", 200.0, 500.0),
    ("complex", 500.0, 900.0),
    ("extreme", 900.0, np.inf),
)


def fluid_from_k_first(k_first, nz):
    """``(nz, ny, nx)`` boolean: True where the cell is fluid.

    The dataset stores ``k_first``, the first fluid cell per column, rather
    than the full 3D mask -- it is what a 2D network can predict, and the
    mask is recoverable from it because terrain is single-valued: every
    cell above the first fluid one is fluid too.
    """
    k_first = np.asarray(k_first)
    k = np.arange(int(nz)).reshape(-1, 1, 1)
    return k >= k_first[None, :, :]


def speed(u, v, w=None):
    """Wind speed. Horizontal by default -- ``w`` is one to two orders
    smaller and including it changes the number by less than the
    tolerance, but it is accepted for the cases that want the full
    magnitude."""
    s = np.asarray(u, dtype=np.float64) ** 2 + np.asarray(v, np.float64) ** 2
    if w is not None:
        s = s + np.asarray(w, dtype=np.float64) ** 2
    return np.sqrt(s)


def error_stats(pred, ref, sel=None, components=True):
    """Error of ``pred`` against ``ref``, in the units they are given in.

    ``pred`` and ``ref`` are ``(ncomp, ...)`` stacks of velocity components
    or plain arrays of one scalar field. ``sel`` selects the cells that
    count -- pass the fluid mask, or leave it out when every point is
    valid, as it is at AGL levels.

    Returns rmse, mae, p95 and max. The p95 is there because a wind field
    is judged by where it is worst, not on average, and the max alone is
    one cell and often an artefact of a single steep column.
    """
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if pred.shape != ref.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {ref.shape}")

    d = pred - ref
    if components:
        # Vector error: the magnitude of the difference, per point, so a
        # field that is right in speed and wrong in direction still fails.
        d = np.sqrt((d ** 2).sum(axis=0))
    else:
        d = np.abs(d)

    if sel is not None:
        d = d[np.asarray(sel, dtype=bool)]
    d = d.ravel()
    if d.size == 0:
        raise ValueError("no points selected")

    return {
        "rmse": float(np.sqrt((d ** 2).mean())),
        "mae": float(d.mean()),
        "p95": float(np.percentile(d, 95.0)),
        "max": float(d.max()),
        "n": int(d.size),
    }


def band_of(levels, base=BAND_BASE_M, top=BAND_TOP_M):
    """Which levels lie in the engineering band, as a boolean index.

    The band is where the deliverable lives -- a met mast at 10 m, hub
    height at 80-160 m -- and it is also the hardest part of the column,
    because it is the part the terrain controls. Reporting only the whole
    column averages the easy air aloft into the number and flatters it.
    """
    lv = np.asarray(levels, dtype=np.float64)
    return (lv >= base - 1e-9) & (lv <= top + 1e-9)


def level_errors(pred, ref, levels, components=True):
    """Error at each level, plus the column and band summaries.

    ``pred`` and ``ref`` are ``(ncomp, nlev, ny, nx)`` (or ``(nlev, ny,
    nx)`` for a scalar). Every point at an AGL level is fluid by
    construction, so no mask is needed here -- that is one of the reasons
    the surrogate predicts levels rather than the Cartesian field.
    """
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    lv = np.asarray(levels, dtype=np.float64)
    axis = 1 if components else 0
    if pred.shape[axis] != lv.size:
        raise ValueError(
            f"{pred.shape[axis]} levels in the field, {lv.size} heights")

    per = []
    for i in range(lv.size):
        sl = (slice(None), i) if components else (i,)
        st = error_stats(pred[sl], ref[sl], components=components)
        st["z_agl"] = float(lv[i])
        per.append(st)

    inb = band_of(lv)
    out = {
        "levels": per,
        "column": error_stats(pred, ref, components=components),
    }
    if inb.any():
        sl = (slice(None), inb) if components else (inb,)
        out["band"] = error_stats(pred[sl], ref[sl], components=components)
    return out


def skill(err, err_ref):
    """``1 - err/err_ref``: what the prediction adds over a reference.

    1 is perfect, 0 is "no better than the reference", negative is worse
    than doing nothing. Reported against the undisturbed profile, which is
    the honest denominator: it is the field available for free.

    A skill score is NOT a substitute for the m/s number. A skill of 0.9
    against a baseline that was already inside tolerance means the
    surrogate improved something nobody could measure.
    """
    err_ref = float(err_ref)
    if err_ref <= 0.0:
        return float("nan")
    return 1.0 - float(err) / err_ref


def group_by_relief(records, bins=RELIEF_BINS, key="rmse"):
    """Summarise per-sample records into the relief bins.

    ``records`` is an iterable of dicts carrying at least ``relief`` and
    ``key``. Returns one row per bin, empty bins included -- an empty bin
    is information, and dropping it makes a table look better covered than
    it is.
    """
    rows = []
    for name, lo, hi in bins:
        vals = [float(r[key]) for r in records
                if lo <= float(r["relief"]) < hi]
        rows.append({
            "group": name,
            "relief": f"{lo:.0f}-{hi:.0f}" if np.isfinite(hi)
                      else f"{lo:.0f}+",
            "n": len(vals),
            "mean": float(np.mean(vals)) if vals else float("nan"),
            "worst": float(np.max(vals)) if vals else float("nan"),
        })
    return rows


def table(rows, columns, widths=None):
    """A fixed-width table. Plain text on purpose: these numbers get read
    in a terminal beside a running solve and pasted into commit messages."""
    widths = widths or {}
    head = "  ".join(f"{c:>{widths.get(c, 10)}}" for c in columns)
    out = [head, "-" * len(head)]
    for r in rows:
        cells = []
        for c in columns:
            v = r.get(c, "")
            w = widths.get(c, 10)
            if isinstance(v, float):
                cells.append(f"{'--':>{w}}" if not np.isfinite(v)
                             else f"{v:>{w}.3f}")
            else:
                cells.append(f"{v:>{w}}")
        out.append("  ".join(cells))
    return "\n".join(out)
