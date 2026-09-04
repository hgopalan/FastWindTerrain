"""
The training pipeline (fastwindterrain.training), phase 22b.

A data pipeline bug does not announce itself. A terrain channel negated
along with the velocity, a normalisation that divides away the amplitude
the answer depends on, a direction convention mirrored -- each produces a
training run that converges to something plausible and wrong, and none of
them shows up as an error. So the invariants are asserted here, with no
model and no torch.
"""

import numpy as np
import pytest

from fastwindterrain import training as T


def _arrays(nlev=9, n=8, seed=0):
    rng = np.random.default_rng(seed)
    a = {k: rng.standard_normal((nlev, n, n)).astype("float32")
         for k in T.TARGET_FIELDS}
    # A ridge, so terrain has real structure and a real gradient.
    y, x = np.mgrid[0:n, 0:n]
    a["terrain"] = (300.0 + 200.0 * np.sin(x * np.pi / n)).astype("float32")
    a["levels"] = np.geomspace(5.0, 1600.0, nlev)
    return a


def _samples(n_windows=3, nlev=9, n=8):
    out = []
    for w in range(n_windows):
        for d in (0.0, 45.0, 90.0, 135.0):
            info = {"id": f"w:{w:02d}@{d:03.0f}", "direction": d,
                    "derived": False, "fold": "train"}
            out.append((info, _arrays(nlev, n, seed=w)))
    return out


# ---------------------------------------------------------------------------
# Normalisation. The failure here is silent and destroys the physics.
# ---------------------------------------------------------------------------

def test_the_target_normalisation_is_exactly_invertible():
    """Scoring happens in m/s. If this round trip is not exact, every
    number that leaves a training run is wrong by the error in it."""
    a = _arrays()
    y = T.make_target(a, u_ref=10.0)
    back = T.to_ms(y, u_ref=10.0)
    for k, i in zip(T.TARGET_FIELDS, range(3)):
        sl = slice(i * 9, (i + 1) * 9)
        assert np.allclose(back[sl], a[k], rtol=0, atol=1e-5), k


def test_wind_speed_is_a_pure_scaling_of_the_target():
    """The measured property the normalisation rests on: every field
    scales exactly with u_ref, so doubling it must exactly halve the
    normalised target and change nothing else."""
    a = _arrays()
    y1 = T.make_target(a, u_ref=10.0)
    y2 = T.make_target(a, u_ref=20.0)
    assert np.allclose(y2, 0.5 * y1, rtol=1e-6, atol=0)


def test_terrain_is_scaled_by_a_constant_not_by_its_own_relief():
    """The tempting normalisation, and the wrong one. Per-sample scaling
    by relief would make a 50 m hill and a 1500 m ridge arrive identical,
    divides away the amplitude that decides how much the flow deflects,
    and cannot be detected from a loss curve.
    """
    small, big = _arrays(), _arrays()
    small["terrain"] = (small["terrain"] - small["terrain"].mean()) * 0.1
    big["terrain"] = (big["terrain"] - big["terrain"].mean()) * 3.0

    cs, _ = T.terrain_channels(small["terrain"], 50.0, 50.0)
    cb, _ = T.terrain_channels(big["terrain"], 50.0, 50.0)
    assert cb.std() > 20.0 * cs.std(), (
        "a thirty-fold difference in relief must survive normalisation")
    # And the constant is the documented one.
    assert np.allclose(cb, (big["terrain"] - big["terrain"].mean())
                       / T.TERRAIN_SCALE_M, atol=1e-6)


def test_terrain_is_centred_because_the_floor_follows_the_window():
    """Each corpus window's domain floor sits at its own minimum
    elevation, so absolute elevation carries no physics -- a 3000 m
    Colorado window and a 100 m coastal one should not differ by a
    constant offset the network has to learn to ignore."""
    low, high = _arrays(), _arrays()
    high["terrain"] = high["terrain"] + 2900.0
    cl, _ = T.terrain_channels(low["terrain"], 50.0, 50.0)
    ch, _ = T.terrain_channels(high["terrain"], 50.0, 50.0)
    assert np.allclose(cl, ch, atol=1e-5)


def test_the_slope_channel_is_the_measured_predictor():
    """slope_error.py found |grad h| correlating with error at r = 0.5-0.72.
    A constant-gradient ramp has a slope this must reproduce exactly."""
    n = 16
    y, x = np.mgrid[0:n, 0:n]
    ramp = (3.0 * x).astype("float32")          # 3 m rise per cell
    _, slope = T.terrain_channels(ramp, dx=50.0, dy=50.0)
    assert np.allclose(slope, 3.0 / 50.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Direction: the convention, and the oddness the dataset is built on.
# ---------------------------------------------------------------------------

def test_the_direction_convention_matches_the_solver():
    """A direction of 45 degrees is wind FROM the northeast, so the flow
    vector points southwest -- both components negative. Mirroring this
    would flip every sample and still train."""
    sx, cy = T.direction_channels(45.0, (2, 2))
    assert sx[0, 0] < 0 and cy[0, 0] < 0
    assert np.allclose(sx, -np.sin(np.radians(45.0)))
    assert np.allclose(cy, -np.cos(np.radians(45.0)))
    # Due north (0 deg): wind from the north, flow due south.
    sx, cy = T.direction_channels(0.0, (2, 2))
    assert np.allclose(sx, 0.0, atol=1e-12) and cy[0, 0] == pytest.approx(-1)


def test_reversing_the_direction_negates_the_direction_channels():
    """The encoding must make the operator's oddness representable. Two
    continuous channels do; a one-hot over eight classes would not, and
    the network would have to rediscover a symmetry it could be handed.
    """
    for d in (0.0, 45.0, 90.0, 135.0):
        a = np.stack(T.direction_channels(d, (2, 2)))
        b = np.stack(T.direction_channels((d + 180.0) % 360.0, (2, 2)))
        assert np.allclose(b, -a, atol=1e-12), d


# ---------------------------------------------------------------------------
# The dataset, and the reverse-derivation that halves it.
# ---------------------------------------------------------------------------

def test_the_dataset_shapes_are_the_contract():
    ds = T.LevelDataset(_samples(), u_ref=10.0, as_tensor=False)
    x, y = ds[0]
    assert x.shape == (len(T.INPUT_CHANNELS), 8, 8)
    assert y.shape == (3 * 9, 8, 8)
    assert x.dtype == np.float32 and y.dtype == np.float32


def test_deriving_the_reverses_doubles_the_dataset():
    solved = _samples()
    ds = T.LevelDataset(solved, derive_reverses=True, as_tensor=False)
    assert len(ds) == 2 * len(solved)


def test_a_derived_sample_negates_velocity_and_NOT_terrain():
    """The invariant this whole file exists for. Negating the terrain
    channel along with the velocity turns every ridge into a valley, is
    invisible in a loss curve, and would poison every result downstream.
    """
    solved = _samples()
    ds = T.LevelDataset(solved, derive_reverses=True, as_tensor=False)
    n = len(solved)
    for i in range(n):
        x0, y0 = ds[i]
        x1, y1 = ds[i + n]
        assert np.allclose(y1, -y0, atol=0), "velocity must negate exactly"
        assert np.array_equal(x1[0], x0[0]), "terrain must NOT negate"
        assert np.array_equal(x1[1], x0[1]), "slope must NOT negate"
        assert np.allclose(x1[2:], -x0[2:], atol=1e-6), "direction flips"


def test_the_derived_half_is_labelled_and_not_a_duplicate_id():
    solved = _samples()
    ds = T.LevelDataset(solved, derive_reverses=True, as_tensor=False)
    ids = [ds.info(i)["id"] for i in range(len(ds))]
    assert len(set(ids)) == len(ids), "ids must stay unique"
    late = ds.info(len(solved))
    assert late["derived"] and late["derived_from"] == solved[0][0]["id"]
    assert late["direction"] == pytest.approx(180.0)


def test_deriving_reverses_refuses_samples_that_are_already_derived():
    """Passing the full dataset with derive_reverses on would produce
    duplicates rather than the reverses, and the count would still look
    plausible."""
    both = _samples()
    both.append(({"id": "w:00@180", "direction": 180.0, "derived": True,
                  "derived_from": "w:00@000", "fold": "train"}, _arrays()))
    with pytest.raises(ValueError, match="SOLVED samples only"):
        T.LevelDataset(both, derive_reverses=True)


def test_the_dataset_yields_torch_tensors_when_asked():
    torch = pytest.importorskip("torch")
    ds = T.LevelDataset(_samples(), as_tensor=True)
    x, y = ds[0]
    assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
    assert x.dtype == torch.float32 and y.shape == (27, 8, 8)


def test_the_dataset_works_with_a_torch_dataloader():
    """It does not subclass torch's Dataset -- the map-style protocol is
    all DataLoader needs, and staying independent keeps this module
    importable and testable with no torch installed."""
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    ds = T.LevelDataset(_samples(), as_tensor=True)
    x, y = next(iter(DataLoader(ds, batch_size=4, shuffle=False)))
    assert x.shape == (4, len(T.INPUT_CHANNELS), 8, 8)
    assert y.shape == (4, 27, 8, 8)
