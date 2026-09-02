#!/usr/bin/env python3
"""
Phase 14 regtest checker -- anisotropy and O'Brien from Python.

Phase 7's assertions, reproduced through the bindings and recomputed in
numpy rather than read out of a plotfile:

  slope factor -> alpha_v must equal
                      clamp(alpha_v_base * exp(-slope_3d / slope_scale))
                      slope_3d = |grad z_terrain| * exp(-z_agl / decay)
                  cell by cell, with |grad z_terrain| computed here from
                  the same central differences the solver uses. Suppressed
                  on the flanks, at base over flat ground

  disabled     -> with enable off both weights hold their base values
                  everywhere, so the feature cannot change results when
                  it is switched off

  alpha_h_mode -> 'base' leaves the horizontal weight alone, as
                  massconsistent_amr does; 'slope' lets the same factor
                  reach it

  O'Brien      -> after the adjustment w is EXACTLY zero at the domain
                  top, held to round-off rather than a tolerance, since
                  making it exact is the whole point of the scheme. The
                  residual it removed must be nonzero, or the case proved
                  nothing

  base weights -> alpha_h_base/alpha_v_base are refused inside a Solver
                  configuration: they are poisson.alpha_h/alpha_v there,
                  and a second copy could disagree with the operator it
                  feeds

Usage:
    python3 check.py /path/to/fastwindterrain.exe [workdir]

Exits 0 if all cases pass, 1 otherwise (suitable for CI).
"""

import sys
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
GROUP = os.path.basename(HERE)
REGTEST_ROOT = os.path.dirname(HERE)
ROOT = os.path.dirname(REGTEST_ROOT)

WORKDIR = os.path.join(ROOT, "build", "regtests", GROUP)

# The steep hill Phase 7 uses: 150 m over a 120 m width, so the slope
# factor has something to bite on.
TERRAIN_CSV = os.path.join(REGTEST_ROOT, "phase7_anisotropy_obrien",
                           "terrain_slope.csv")

NX, NY, NZ = 40, 40, 66
ALPHA_H_BASE, ALPHA_V_BASE = 1.0, 0.5
SLOPE_SCALE, DECAY_HEIGHT = 0.5, 500.0
MIN_FACTOR, MAX_FACTOR = 0.05, 2.0


def bindings_available():
    shim = os.path.join(ROOT, "build", "fastwindterrain-py")
    pkg = os.path.join(ROOT, "build", "python", "fastwindterrain",
                       "__init__.py")
    return os.path.isfile(shim) and os.path.isfile(pkg)


def python_exe():
    shim = os.path.join(ROOT, "build", "fastwindterrain-py")
    with open(shim) as f:
        for line in f:
            if line.startswith("exec "):
                return line.split('"')[1]
    raise RuntimeError(f"could not read the interpreter out of {shim}")


def run_py(code):
    env = dict(os.environ)
    env["PYTHONPATH"] = (os.path.join(ROOT, "build", "python")
                         + os.pathsep + env.get("PYTHONPATH", ""))
    return subprocess.run([python_exe(), "-c", code], capture_output=True,
                          text=True, timeout=3600, env=env, cwd=WORKDIR)


def marks(stdout):
    out = {}
    for line in stdout.split("\n"):
        if line.startswith("::"):
            k, _, v = line[2:].partition(" ")
            out[k] = v
    return out


PRELUDE = f"""
import numpy as np
import fastwindterrain as fwt

PTS = np.loadtxt({TERRAIN_CSV!r}, delimiter=",", comments="#", skiprows=5)

GRID = {{"n_cell": ({NX}, {NY}, {NZ}), "prob_lo": (0.0, 0.0, 0.0),
        "prob_hi": (1000.0, 1000.0, 961.2758234855), "dz0": 2.0,
        "stretching_ratio": 1.05, "max_grid_size": 32}}

ANISO = {{"enable": True, "source": "slope", "alpha_h_mode": "base",
         "slope_scale": {SLOPE_SCALE}, "decay_height": {DECAY_HEIGHT},
         "alpha_h_base": {ALPHA_H_BASE}, "alpha_v_base": {ALPHA_V_BASE}}}

def case(**kw):
    cfg = {{
        "grid": GRID,
        "terrain": {{"points": PTS}},
        "inflow": {{"mode": "powerlaw", "u_ref": 8.0, "v_ref": 6.0}},
        "poisson": {{"alpha_h": {ALPHA_H_BASE}, "alpha_v": {ALPHA_V_BASE},
                    "n_projections": 1}},
    }}
    cfg.update(kw)
    return cfg

def column_slopes(zt):
    \"\"\"|grad z_terrain| per column, from the same central differences the
    solver uses -- computed here independently rather than read back.\"\"\"
    h = zt[0]                       # constant along k
    ny, nx = h.shape
    dx = 1000.0 / {NX}
    dy = 1000.0 / {NY}
    i = np.arange(nx)
    j = np.arange(ny)
    ip1, im1 = np.minimum(i + 1, nx - 1), np.maximum(i - 1, 0)
    jp1, jm1 = np.minimum(j + 1, ny - 1), np.maximum(j - 1, 0)
    dhdx = (h[:, ip1] - h[:, im1]) / ((ip1 - im1)[None, :] * dx)
    dhdy = (h[jp1, :] - h[jm1, :]) / ((jp1 - jm1)[:, None] * dy)
    return np.hypot(dhdx, dhdy)
"""


# ---------------------------------------------------------------------------
# 1. the slope factor, cell by cell
# ---------------------------------------------------------------------------

def check_slope_factor(exe):
    name = "alpha_v follows the slope factor"

    r = run_py(PRELUDE + f"""
with fwt.session():
    g = fwt.Grid(GRID)
    t = fwt.Terrain(g, {{"points": PTS}})
    a = fwt.Anisotropy(g, t, ANISO)

    zt = t.z_terrain
    slope = column_slopes(zt)
    print("::SLOPEMAX", repr(float(slope.max())))
    print("::SLOPEREPORT", repr(float(a.slope_max)))

    z_cc = np.asarray(g.z_cc)[:, None, None]
    z_agl = np.maximum(z_cc - zt, 0.0)
    slope_3d = slope[None, :, :] * np.exp(-z_agl / {DECAY_HEIGHT})
    f_slope = np.exp(-slope_3d / {SLOPE_SCALE})
    expect_v = np.clip({ALPHA_V_BASE} * f_slope,
                       {MIN_FACTOR} * {ALPHA_V_BASE},
                       {MAX_FACTOR} * {ALPHA_V_BASE})

    av = a.alpha_v
    print("::WORST", repr(float(np.abs(av - expect_v).max())))
    print("::NSUPPRESSED", int((av < 0.99 * {ALPHA_V_BASE}).sum()))
    print("::AVMIN", repr(float(av.min())))
    # alpha_h_mode = base: the horizontal weight is untouched.
    print("::AHFLAT", bool(np.all(a.alpha_h == {ALPHA_H_BASE})))
    # The suppression weakens with height above ground: slope_3d carries
    # exp(-z_agl / decay_height), so the per-level minimum can only rise
    # with k. It does NOT reach base by the domain top -- 961 m over a
    # 500 m decay height is under two e-foldings -- so the check is that
    # it decays, not that it has finished.
    per_level = av.min(axis=(1, 2))
    print("::MONOTONE", bool(np.all(np.diff(per_level) >= 0.0)))
    print("::SURFACE", repr(float(per_level[0])))
    print("::TOP", repr(float(per_level[-1])))
""")
    assert r.returncode == 0, (
        f"[{name}] failed:\n{r.stdout[-2500:]}\n{r.stderr[-2000:]}")

    m = marks(r.stdout)
    assert float(m["SLOPEMAX"]) > 0.3, (
        f"[{name}] the hill is too gentle to exercise the slope factor: "
        f"max |grad z_terrain| = {m['SLOPEMAX']}")
    assert abs(float(m["SLOPEREPORT"]) - float(m["SLOPEMAX"])) < 1e-12, (
        f"[{name}] the solver reports slope_max {m['SLOPEREPORT']} but an "
        f"independent central difference gives {m['SLOPEMAX']}")
    assert float(m["WORST"]) < 1.0e-12, (
        f"[{name}] alpha_v differs from the independently recomputed slope "
        f"factor by {m['WORST']}")
    assert int(m["NSUPPRESSED"]) > 100, (
        f"[{name}] only {m['NSUPPRESSED']} cells are suppressed; the case "
        f"is not exercising the factor")
    assert float(m["AVMIN"]) < 0.5 * ALPHA_V_BASE, (
        f"[{name}] alpha_v only fell to {m['AVMIN']} from a base of "
        f"{ALPHA_V_BASE}")
    assert m["AHFLAT"] == "True", (
        f"[{name}] alpha_h_mode = base must leave the horizontal weight at "
        f"its base value everywhere")
    assert m["MONOTONE"] == "True", (
        f"[{name}] the per-level minimum of alpha_v must rise with height: "
        f"slope_3d carries exp(-z_agl / decay_height), so the suppression "
        f"can only weaken going up")
    assert float(m["TOP"]) > 2.0 * float(m["SURFACE"]), (
        f"[{name}] alpha_v is {m['SURFACE']} at the surface and "
        f"{m['TOP']} at the top; the decay is barely visible")

    print(f"[PASS] {name}  (matches an independent recomputation to "
          f"{float(m['WORST']):.1e} over {NX * NY * NZ} cells; suppressed "
          f"to {float(m['AVMIN']):.4f} from base {ALPHA_V_BASE} on a "
          f"{float(m['SLOPEMAX']):.3f} slope, decaying monotonically to "
          f"{float(m['TOP']):.4f} aloft)")


# ---------------------------------------------------------------------------
# 2. disabled is inert; alpha_h_mode
# ---------------------------------------------------------------------------

def check_disabled_and_mode(exe):
    name = "disabled is inert, alpha_h_mode works"

    r = run_py(PRELUDE + f"""
with fwt.session():
    g = fwt.Grid(GRID)
    t = fwt.Terrain(g, {{"points": PTS}})

    off = fwt.Anisotropy(g, t, dict(ANISO, enable=False))
    print("::OFF_AH", bool(np.all(off.alpha_h == {ALPHA_H_BASE})))
    print("::OFF_AV", bool(np.all(off.alpha_v == {ALPHA_V_BASE})))

    none = fwt.Anisotropy(g, t, dict(ANISO, source="none"))
    print("::NONE_AV", bool(np.all(none.alpha_v == {ALPHA_V_BASE})))

    both = fwt.Anisotropy(g, t, dict(ANISO, alpha_h_mode="slope"))
    ratio_h = both.alpha_h_min / {ALPHA_H_BASE}
    ratio_v = both.alpha_v_min / {ALPHA_V_BASE}
    print("::RATIOS %r %r" % (ratio_h, ratio_v))
    # Cell by cell, not just at the minimum.
    print("::SAMEFACTOR", bool(np.allclose(both.alpha_h / {ALPHA_H_BASE},
                                           both.alpha_v / {ALPHA_V_BASE},
                                           rtol=0, atol=1e-14)))
""")
    assert r.returncode == 0, (
        f"[{name}] failed:\n{r.stdout[-2500:]}\n{r.stderr[-2000:]}")

    m = marks(r.stdout)
    assert m["OFF_AH"] == "True" and m["OFF_AV"] == "True", (
        f"[{name}] with enable off both weights must hold their base "
        f"values everywhere")
    assert m["NONE_AV"] == "True", (
        f"[{name}] source = none must also leave the weights at base")
    ratio_h, ratio_v = (float(v) for v in m["RATIOS"].split())
    assert abs(ratio_h - ratio_v) < 1.0e-9, (
        f"[{name}] alpha_h_mode = slope must apply the SAME factor to both "
        f"weights: {ratio_h} vs {ratio_v}")
    assert ratio_h < 0.5, (
        f"[{name}] the factor barely moved ({ratio_h}), so this compared "
        f"nothing")
    assert m["SAMEFACTOR"] == "True", (
        f"[{name}] the two weights carry the same factor at their minima "
        f"but not cell by cell")

    print(f"[PASS] {name}  (off and source=none both inert; "
          f"alpha_h_mode=slope applies the same factor, {ratio_h:.4f} of "
          f"base, in every cell)")


# ---------------------------------------------------------------------------
# 3. O'Brien: w exactly zero at the top
# ---------------------------------------------------------------------------

def check_obrien(exe):
    name = "O'Brien leaves w exactly zero at the top"

    r = run_py(PRELUDE + """
with fwt.session():
    s = fwt.Solver(case(obrien={"enable": True},
                        anisotropy={"enable": True, "source": "slope",
                                    "slope_scale": 0.5,
                                    "decay_height": 500.0}))
    s.setup()               # O'Brien runs inside setup, before any solve
    ob = s.obrien
    print("::ENABLED", ob["enabled"])
    print("::NCOL", ob["n_columns"])
    print("::WTOP", repr(float(ob["max_w_top"])))
    print("::RESID", repr(float(ob["max_residual"])))

    # Straight from the field, not just from the reported number.
    w_top = s.velocity[2][-1]
    solid_top = s.mask[-1] == 1
    print("::FIELD_WTOP", repr(float(np.abs(w_top[~solid_top]).max())))

    off = fwt.Solver(case(obrien={"enable": False}))
    off.setup()
    print("::OFF_NCOL", off.obrien["n_columns"])
    print("::OFF_WTOP", repr(float(np.abs(off.velocity[2][-1]).max())))
""")
    assert r.returncode == 0, (
        f"[{name}] failed:\n{r.stdout[-2500:]}\n{r.stderr[-2000:]}")

    m = marks(r.stdout)
    assert m["ENABLED"] == "True", f"[{name}] the adjustment did not run"
    assert int(m["NCOL"]) > 100, (
        f"[{name}] only {m['NCOL']} columns were adjusted")
    assert float(m["RESID"]) > 0.1, (
        f"[{name}] the residual removed was {m['RESID']}; with nothing to "
        f"remove the exactness below proves nothing")
    assert float(m["WTOP"]) == 0.0, (
        f"[{name}] max|w| at the domain top is {m['WTOP']}, not exactly "
        f"zero. Making it exact rather than small is the whole point of "
        f"the redistribution.")
    assert float(m["FIELD_WTOP"]) == 0.0, (
        f"[{name}] the reported max_w_top is zero but the field itself has "
        f"{m['FIELD_WTOP']} at the top")
    assert int(m["OFF_NCOL"]) == 0, (
        f"[{name}] with the adjustment off, no column should be touched")

    print(f"[PASS] {name}  ({m['NCOL']} columns; a "
          f"{float(m['RESID']):.4g} m/s residual removed, leaving |w| at "
          f"the top exactly 0 in the reported value and in the field)")


# ---------------------------------------------------------------------------
# 4. the base weights belong to the operator
# ---------------------------------------------------------------------------

def check_base_weights_belong_to_poisson(exe):
    name = "base weights are the Poisson section's"

    r = run_py(PRELUDE + """
def expect(label, fn):
    try:
        fn()
        print("::%s accepted" % label)
    except ValueError as e:
        print("::%s raised" % label)
        print("::%s_MSG %s" % (label, str(e)[:80]))

with fwt.session():
    # Inside a Solver configuration the base weights are the Poisson
    # section's, so naming them here is refused rather than silently
    # overridden.
    expect("IN_SOLVER", lambda: fwt.Solver(
        case(anisotropy={"enable": True, "alpha_v_base": 0.3})))

    # And they really do come from the Poisson section.
    s = fwt.Solver(case(poisson={"alpha_h": 2.0, "alpha_v": 0.25,
                                 "n_projections": 1},
                        anisotropy={"enable": False}))
    s.setup()
    a = s.anisotropy
    print("::BASE %r %r" % (a["alpha_h_base"], a["alpha_v_base"]))
    print("::FIELD %r %r" % (float(s.alpha_h.max()), float(s.alpha_v.max())))

    # Standalone, there is no Poisson section, so they are accepted.
    g = fwt.Grid(GRID)
    t = fwt.Terrain(g, {"flat_elevation": 0.0})
    stand = fwt.Anisotropy(g, t, {"alpha_h_base": 3.0, "alpha_v_base": 0.75})
    print("::STANDALONE %r %r" % (stand.alpha_h_base, stand.alpha_v_base))
""")
    assert r.returncode == 0, (
        f"[{name}] failed:\n{r.stdout[-2500:]}\n{r.stderr[-2000:]}")

    m = marks(r.stdout)
    assert m["IN_SOLVER"] == "raised", (
        f"[{name}] alpha_v_base in a Solver's anisotropy section must "
        f"raise, since the operator takes its weights from the Poisson "
        f"section and the two could disagree")
    assert "poisson" in m.get("IN_SOLVER_MSG", ""), (
        f"[{name}] the message should say where the setting belongs; got "
        f"{m.get('IN_SOLVER_MSG')!r}")

    bh, bv = (float(v) for v in m["BASE"].split())
    assert bh == 2.0 and bv == 0.25, (
        f"[{name}] the anisotropy bases are {bh}/{bv}, expected the "
        f"Poisson section's 2.0/0.25")
    fh, fv = (float(v) for v in m["FIELD"].split())
    assert fh == 2.0 and fv == 0.25, (
        f"[{name}] the weight fields are {fh}/{fv}, expected 2.0/0.25")

    sh, sv = (float(v) for v in m["STANDALONE"].split())
    assert sh == 3.0 and sv == 0.75, (
        f"[{name}] a standalone Anisotropy must accept its own bases; got "
        f"{sh}/{sv}")

    print(f"[PASS] {name}  (refused in a Solver config and taken from the "
          f"Poisson section instead; accepted standalone, where there is "
          f"no Poisson section to take them from)")


def main():
    global WORKDIR

    if len(sys.argv) not in (2, 3):
        print(f"usage: {sys.argv[0]} /path/to/fastwindterrain.exe [workdir]")
        return 1

    exe = os.path.abspath(sys.argv[1])
    if not os.path.isfile(exe):
        print(f"executable not found: {exe}")
        return 1

    if len(sys.argv) == 3:
        WORKDIR = os.path.abspath(sys.argv[2])
    os.makedirs(WORKDIR, exist_ok=True)
    print(f"work directory: {WORKDIR}")

    if not bindings_available():
        print("[SKIP] the Python bindings are not built. Configure with "
              "-DFWT_PYTHON=ON to run this group.")
        return 0

    failed = []
    for check in (check_slope_factor, check_disabled_and_mode, check_obrien,
                  check_base_weights_belong_to_poisson):
        try:
            check(exe)
        except AssertionError as e:
            print(f"[FAIL] {check.__name__}: {e}")
            failed.append(check.__name__)
        except Exception as e:
            print(f"[ERROR] {check.__name__}: {e}")
            failed.append(check.__name__)

    if failed:
        print(f"\n{len(failed)} Phase 14 regtest case(s) failed: {failed}")
        return 1

    print("\nAll Phase 14 regtest cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
