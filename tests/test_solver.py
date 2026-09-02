"""
The solver driven from Python (was regtests/phase13_solver_driver).

``Solver`` exposes both ``run()`` and the stages one at a time, so a
notebook can watch an approximate projection converge rather than be told
that it did. The tests hold the two to each other: stepping N passes must
give the same field, to the last bit, as ``solve()`` with
``n_projections`` set to N -- they are the same code, and this is what
keeps them so.
"""

import numpy as np
import pytest

import fastwindterrain as fwt
from conftest import TERRAIN_CSV

FLAT = {
    "grid": {"n_cell": (16, 16, 32), "prob_lo": (0.0, 0.0, 0.0),
             "prob_hi": (1000.0, 1000.0, 128.0), "dz0": 4.0,
             "max_grid_size": 8},
    "terrain": {"flat_elevation": 0.0},
    "inflow": {"mode": "powerlaw", "u_ref": 8.0, "v_ref": 6.0},
    "poisson": {"alpha_v": 0.5, "n_projections": 4},
}


@pytest.fixture
def hill(terrain_points):
    """The hill case, with the number of projection passes as a knob."""
    def _hill(n_proj=4):
        return {
            "grid": {"n_cell": (24, 24, 40), "prob_lo": (0.0, 0.0, 0.0),
                     "prob_hi": (1000.0, 1000.0, 483.19909696997223),
                     "dz0": 4.0, "stretching_ratio": 1.05,
                     "max_grid_size": 16},
            "terrain": {"points": terrain_points},
            "inflow": {"mode": "powerlaw", "u_ref": 8.0, "v_ref": 6.0},
            "poisson": {"alpha_v": 0.5, "n_projections": n_proj},
        }
    return _hill


# ---------------------------------------------------------------------------
# Flat ground: nothing to correct
# ---------------------------------------------------------------------------

def test_flat_ground_is_left_alone(amrex):
    """Phase 6's inputs_flat assertions, from Python.

    A uniform profile over flat ground is already solenoidal, so there is
    nothing for the projection to correct. This is the case that catches
    a projection which "corrects" a field that was already fine -- the
    kind of bug every other case hides, because there the correction is
    supposed to be nonzero.
    """
    s = fwt.Solver(FLAT)
    s.setup()
    v0 = s.velocity0.copy()
    s.solve()
    s.diagnose()

    assert float(np.abs(s.lambda_).max()) == 0.0, "lambda must be exactly 0"
    assert np.array_equal(s.velocity, v0), (
        "the projection changed a field that was already divergence free")
    assert float(np.abs(s.velocity[2]).max()) == 0.0, "w must be exactly 0"
    assert float(np.sqrt(v0[0] ** 2 + v0[1] ** 2).max()) > 1.0, (
        "the profile is ~zero, so this proved nothing")


# ---------------------------------------------------------------------------
# Stepwise == the loop
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_project_once_matches_solve(amrex, hill):
    a = fwt.Solver(hill(3))
    a.setup()
    a.solve()

    b = fwt.Solver(hill(3))
    b.setup()
    for _ in range(3):
        b.project_once()

    assert a.n_projections_done == b.n_projections_done == 3
    assert np.array_equal(a.velocity, b.velocity), (
        "stepping three passes gives a different field from solve() with "
        "n_projections = 3; they are supposed to be the same code")
    assert np.array_equal(a.lambda_, b.lambda_)
    assert float(np.abs(a.lambda_).max()) > 0.0, (
        "lambda is identically zero, so the two agreed about nothing")


# ---------------------------------------------------------------------------
# The projection converges, and says so
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_divergence_falls_monotonically(amrex, hill):
    s = fwt.Solver(hill(4))
    s.setup()

    divs = [float(s.max_divergence_fe)]
    residuals, iterations = [], []
    for _ in range(4):
        residuals.append(float(s.project_once()))
        iterations.append(int(s.solve_iterations))
        divs.append(float(s.max_divergence_fe))

    assert all(b < a for a, b in zip(divs, divs[1:])), (
        f"the controlled divergence does not fall monotonically: {divs}")
    assert divs[-1] < 0.6 * divs[0], (
        f"four passes reduced the divergence only from {divs[0]} to "
        f"{divs[-1]}")
    assert max(residuals) < 1.0e-9, (
        f"MLMG's worst residual was {max(residuals)}; the solve is not "
        f"converging")
    # Zero would mean the count is not being reported at all, and 200 is
    # max_iter -- neither is a converged solve.
    assert all(0 < i < 200 for i in iterations), iterations


# ---------------------------------------------------------------------------
# The ghost refill, finally observable
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_set_velocity_refills_the_ghosts(amrex, hill):
    """The test test_fields.py could not write.

    Ghost cells are not exposed, so nothing at the field level can see
    whether ``set_velocity`` refilled them. A solve can follow a write
    here, and the SCHEME divergence reads the ghosts through a five-point
    stencil -- so if solver B were left with its own initial profile in
    the halo rather than the field it was handed, its divergence would
    differ from A's even though their valid regions match.
    """
    a = fwt.Solver(hill(2))
    a.setup()
    a.solve()
    va = a.velocity.copy()

    b = fwt.Solver(hill(2))
    b.setup()
    before = float(b.max_divergence)
    b.set_velocity(va)

    assert np.array_equal(b.velocity, va), "the valid region did not survive"
    assert before != float(a.max_divergence), (
        "the two fields already had the same divergence, so this compared "
        "nothing")
    assert float(b.max_divergence) == float(a.max_divergence), (
        "the valid regions match but the divergences do not, so the ghost "
        "cells were not refilled")


# ---------------------------------------------------------------------------
# No ParmParse leak into the solve
# ---------------------------------------------------------------------------

def test_solve_ignores_parmparse(run_py, tmp_path):
    """A dict-configured run must not inherit poisson or anisotropy
    settings an inputs file left behind."""
    inputs = tmp_path / "inputs_leak"
    inputs.write_text(
        "grid.n_cell = 8 8 8\n"
        "grid.prob_lo = 0.0 0.0 0.0\n"
        "grid.prob_hi = 100.0 100.0 32.0\n"
        "grid.dz0 = 4.0\n"
        "inflow.u_ref = 1.0\n"
        # The values a leak would carry into the dict-configured run.
        "poisson.alpha_v = 0.125\n"
        "poisson.n_projections = 7\n"
        "anisotropy.enable = 1\n"
        "obrien.enable = 1\n")

    r = run_py(f"""
import numpy as np
import fastwindterrain as fwt
fwt.initialize([{str(inputs)!r}])
s = fwt.Solver({FLAT!r})      # alpha_v = 0.5, n_projections = 4
s.setup()
s.solve()
print("::PASSES", s.n_projections_done)
print("::ALPHAV", repr(float(s.alpha_v.max())))
print("::ANISO_FLAT", bool(s.alpha_v.min() == s.alpha_v.max()))
fwt.finalize()
""")

    assert int(r["PASSES"]) == 4, (
        "the dict says 4 projections and the inputs file says 7")
    assert float(r["ALPHAV"]) == 0.5, (
        "the dict says alpha_v 0.5 and the inputs file says 0.125")
    assert r["ANISO_FLAT"] == "True", (
        "alpha_v varies in space, so anisotropy.enable = 1 leaked from the "
        "inputs file -- the dict leaves it off")


# ---------------------------------------------------------------------------
# Stage ordering
# ---------------------------------------------------------------------------

def test_stages_refuse_to_run_out_of_order(amrex):
    """Saying so beats returning something that looks like an answer."""
    s = fwt.Solver(FLAT)
    with pytest.raises(RuntimeError):
        s.solve()
    s.setup()
    with pytest.raises(RuntimeError):
        s.diagnose()
    with pytest.raises(RuntimeError):
        s.fields()
    s.solve()
    with pytest.raises(RuntimeError):
        s.divergence
    s.diagnose()
    assert s.is_setup and s.is_solved and s.is_diagnosed
    assert s.divergence.shape == s.shape
