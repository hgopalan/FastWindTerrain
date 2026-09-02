"""
Fields as numpy arrays (was regtests/phase11_field_bindings).

The layout is ``(ncomp, nz, ny, nx)`` with the leading axis dropped for a
single component, and the nodal ``lambda_`` one point larger in every
direction. Channels-first is AMReX's own memory order, so a gather is a
memcpy per component rather than a transpose, and it is what PyTorch's
``conv3d`` wants.

The index-order test is the one that matters most. A transposed array
passes every other check in this file and quietly ruins a training set
without ever raising anything, so ``arr[c, k, j, i]`` is compared cell by
cell against an AMReX plotfile read by the same reader the C++ regtests
use.

Not covered here: the ghost refill that follows ``set_velocity``. Ghost
cells are deliberately not exposed, so nothing at this level can observe
it; its effect is tested in test_solver.py, where a solve can follow a
write.
"""

import numpy as np
import pytest

import fastwindterrain as fwt

NX, NY, NZ = 24, 24, 40

# max_grid_size 8 on a 24x24x40 domain is deliberately several boxes: a
# MultiFab is N separate FArrayBoxes and a gather that only ever saw one
# of them would look correct.
BOXY = {"n_cell": (NX, NY, NZ), "prob_lo": (0.0, 0.0, 0.0),
        "prob_hi": (1000.0, 1000.0, 483.19909696997223),
        "dz0": 4.0, "stretching_ratio": 1.05, "max_grid_size": 8}


@pytest.fixture
def setup_solver(amrex, case):
    """A Solver with setup() run and nothing else, on a multi-box grid."""
    s = fwt.Solver(case(grid=dict(BOXY)))
    s.setup()
    return s


# ---------------------------------------------------------------------------
# Shapes and dtypes
# ---------------------------------------------------------------------------

EXPECTED = {
    "velocity":  ((3, NZ, NY, NX), "float64"),
    "velocity0": ((3, NZ, NY, NX), "float64"),
    "sigma":     ((3, NZ, NY, NX), "float64"),
    "mask":      ((NZ, NY, NX), "int32"),
    "z_terrain": ((NZ, NY, NX), "float64"),
    "alpha_h":   ((NZ, NY, NX), "float64"),
    "alpha_v":   ((NZ, NY, NX), "float64"),
    # Nodal: one more point in every direction.
    "lambda_":   ((NZ + 1, NY + 1, NX + 1), "float64"),
}


@pytest.mark.parametrize("name,expected", EXPECTED.items(), ids=list(EXPECTED))
def test_field_shape_and_dtype(setup_solver, name, expected):
    shape, dtype = expected
    a = getattr(setup_solver, name)
    assert a.shape == shape
    assert a.dtype == np.dtype(dtype)


def test_solver_shape_property(setup_solver):
    assert setup_solver.shape == (NZ, NY, NX)


def test_the_case_really_is_decomposed(setup_solver):
    """Otherwise the gather tests below are testing one box against
    itself."""
    assert setup_solver.grid.n_boxes > 1


# ---------------------------------------------------------------------------
# Index order, against a plotfile
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_index_order_matches_the_plotfile(amrex, case, Plotfile, tmp_path):
    """``arr[c, k, j, i]`` must be the value AMReX wrote at cell
    ``(i, j, k)``.

    The plotfile is written by the same solve the arrays come from, and
    read back by regtests/plotfile.py -- the reader the C++ groups use.
    Between them, a transposed axis or a wrong gather has nowhere to
    hide.
    """
    s = fwt.Solver(case(grid=dict(BOXY)))
    s.setup()
    s.solve()
    s.diagnose()
    s.write_plotfile(str(tmp_path / "plt_fields"))

    pf = Plotfile(str(tmp_path / "plt_fields"))
    vel = s.velocity
    pairs = [("u", vel[0]), ("v", vel[1]), ("w", vel[2]),
             ("mask", s.mask), ("terrain_z", s.z_terrain),
             ("alpha_v", s.alpha_v)]

    compared = 0
    for pfname, arr in pairs:
        f = pf.field(pfname)
        for k in range(0, NZ, 3):
            for j in range(0, NY, 5):
                for i in range(0, NX, 5):
                    assert float(arr[k, j, i]) == float(f(i, j, k)), (
                        f"{pfname}[{k},{j},{i}] does not match the plotfile "
                        f"at cell ({i},{j},{k}): either the axes are "
                        f"transposed or the gather is wrong")
                    compared += 1
    assert compared > 500, "the sampling stride is broken"


# ---------------------------------------------------------------------------
# Bit-exact round trip
# ---------------------------------------------------------------------------

@pytest.fixture
def probe_field(setup_solver):
    """A velocity field where a transpose, an off-by-one or a dropped
    component all change the result: every cell distinct, and the three
    components far apart in magnitude."""
    v = setup_solver.velocity
    _, nz, ny, nx = v.shape
    kk, jj, ii = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx),
                             indexing="ij")
    new = np.empty_like(v)
    new[0] = kk + 1e-3 * jj + 1e-6 * ii
    new[1] = -2.0 * (kk + 1e-3 * jj + 1e-6 * ii)
    new[2] = 1e4 + kk - 1e-3 * jj
    return new


def test_round_trip_is_bit_exact(setup_solver, probe_field):
    """Not "close": the values never leave double precision, so any
    difference means the gather or the scatter is doing arithmetic."""
    setup_solver.set_velocity(probe_field)
    assert np.array_equal(setup_solver.velocity, probe_field)


def test_writing_twice_does_not_accumulate(setup_solver, probe_field):
    setup_solver.set_velocity(probe_field)
    setup_solver.set_velocity(probe_field)
    assert np.array_equal(setup_solver.velocity, probe_field)


def test_the_returned_array_is_a_copy(setup_solver, probe_field):
    """A MultiFab is several boxes, the velocity carries two ghost
    layers, and a view would outlive the Solver that owns it -- so what
    comes back is a copy, and scribbling on it changes nothing."""
    setup_solver.set_velocity(probe_field)
    got = setup_solver.velocity
    got[:] = 12345.0
    assert np.array_equal(setup_solver.velocity, probe_field)


# ---------------------------------------------------------------------------
# The gather does not depend on the decomposition
# ---------------------------------------------------------------------------

GATHERED = ("velocity", "mask", "z_terrain", "alpha_v", "sigma", "lambda_")


def test_gather_is_independent_of_the_decomposition(amrex, case):
    """Several FArrayBoxes must gather to the same array as one.

    Both solvers live in the same AMReX initialization, which is also a
    check that two Solvers can coexist -- the thing a generation loop
    does all day.
    """
    def fields(mgs):
        s = fwt.Solver(case(grid=dict(BOXY, max_grid_size=mgs)))
        s.setup()
        return s.grid.n_boxes, {n: getattr(s, n).copy() for n in GATHERED}

    n_small, small = fields(8)
    n_big, big = fields(64)

    assert n_small > n_big
    assert n_big == 1
    for k in GATHERED:
        assert np.array_equal(small[k], big[k]), (
            f"{k} depends on the box decomposition; the gather is wrong")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def test_reading_a_field_before_setup_raises(amrex, case):
    """Better than handing back an empty array that looks like data."""
    s = fwt.Solver(case(grid=dict(BOXY)))
    with pytest.raises(RuntimeError):
        s.velocity
    with pytest.raises(RuntimeError):
        s.shape


def _bad_shapes(v):
    return {
        "one cell short": v[:, :, :, :-1],
        "two components": v[:2],
        "flattened": v.reshape(-1),
        # A transposed array has the same shape only if nx == nz; this
        # case is 24x24x40, so it is genuinely a different shape.
        "transposed": np.ascontiguousarray(v.transpose(0, 3, 2, 1)),
    }


def test_bad_shapes_are_rejected(setup_solver):
    v = setup_solver.velocity
    for label, arr in _bad_shapes(v).items():
        with pytest.raises((ValueError, TypeError)):
            setup_solver.set_velocity(arr)
        assert np.array_equal(setup_solver.velocity, v), (
            f"the rejected write ({label}) still modified the field")


def test_float32_is_accepted_by_widening(setup_solver):
    """A widening conversion, not a reinterpretation. Refusing it would
    be pedantry, and a surrogate's output is float32."""
    v = setup_solver.velocity
    setup_solver.set_velocity(v.astype(np.float32))
    assert np.allclose(setup_solver.velocity, v, rtol=0, atol=1e-6)
