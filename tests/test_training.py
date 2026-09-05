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


# ---------------------------------------------------------------------------
# D4 augmentation. Exact, because rotations by 90 degrees and reflections
# map a Cartesian grid onto itself -- verified against the solver itself in
# cases/rotation_test.py at 1e-13.
# ---------------------------------------------------------------------------

def test_the_eight_symmetries_are_a_group_under_the_transform():
    """If applying an operation four times (or twice, for a mirror) is not
    the identity, the transform is not the symmetry it claims to be."""
    rng = np.random.default_rng(1)
    f = rng.standard_normal((3, 7, 7))
    g = f
    for _ in range(4):
        g = T.transform_field(g, 90, False)
    assert np.array_equal(g, f), "rot90 four times must be the identity"
    assert np.array_equal(
        T.transform_field(T.transform_field(f, 0, True), 0, True), f)


def test_the_vector_transform_rotates_components_not_only_the_grid():
    """The failure mode this exists to prevent: a field that looks right
    and points the wrong way. Moving the grid without rotating the
    components is invisible in any scalar plot and in every loss curve.
    """
    u = np.ones((4, 4))
    v = np.zeros((4, 4))
    uu, vv = T.transform_vector(u, v, 90, False)
    # A vector pointing +x, rotated a quarter turn, points +y.
    assert np.allclose(uu, 0.0, atol=1e-12)
    assert np.allclose(vv, 1.0)
    # A mirror in x flips the x component and leaves y alone.
    uu, vv = T.transform_vector(u, v, 0, True)
    assert np.allclose(uu, -1.0) and np.allclose(vv, 0.0)


def test_augmentation_multiplies_the_dataset_by_eight():
    solved = _samples()
    plain = T.LevelDataset(solved, derive_reverses=True, as_tensor=False)
    aug = T.LevelDataset(solved, derive_reverses=True, as_tensor=False,
                         augment_d4=True)
    assert len(aug) == 8 * len(plain)


def test_the_unaugmented_samples_are_unchanged_by_turning_it_on():
    """D4_OPS starts with the identity, so the first block of an augmented
    dataset must be bit-identical to the unaugmented one. If it is not,
    augmentation is changing the original data as well as adding to it.
    """
    solved = _samples()
    plain = T.LevelDataset(solved, derive_reverses=True, as_tensor=False)
    aug = T.LevelDataset(solved, derive_reverses=True, as_tensor=False,
                         augment_d4=True)
    n = len(solved)
    for i in range(n):
        xp, yp = plain[i]
        xa, ya = aug[i]
        assert np.array_equal(xp, xa), i
        assert np.array_equal(yp, ya), i


def test_augmentation_preserves_wind_speed_and_terrain_statistics():
    """A symmetry moves the field; it must not change what is in it. The
    distribution of speeds and of terrain heights is invariant under every
    one of the eight, which is a cheap check that no interpolation or
    clipping crept in."""
    solved = _samples()
    aug = T.LevelDataset(solved, derive_reverses=True, as_tensor=False,
                         augment_d4=True)
    n = len(solved)
    x0, y0 = aug[0]
    for op in range(8):
        x, y = aug[op * n]
        assert np.allclose(np.sort(x[0].ravel()), np.sort(x0[0].ravel()))
        speed0 = np.hypot(y0[:9], y0[9:18])
        speed = np.hypot(y[:9], y[9:18])
        assert np.allclose(np.sort(speed.ravel()),
                           np.sort(speed0.ravel()), atol=1e-6), op


def test_augmentation_and_negation_stay_independent():
    """Both multiply the index and they must compose, not collide: eight
    symmetries times two signs is sixteen distinct samples per solve, and
    the negated block must still be the exact negation of its partner."""
    solved = _samples()
    aug = T.LevelDataset(solved, derive_reverses=True, as_tensor=False,
                         augment_d4=True)
    n, half = len(solved), len(aug) // 2
    for op in range(8):
        i = op * n
        x0, y0 = aug[i]
        x1, y1 = aug[i + half]
        assert np.allclose(y1, -y0, atol=0), f"op {op} velocity"
        assert np.array_equal(x1[0], x0[0]), f"op {op} terrain must not flip"
        assert np.allclose(x1[2:], -x0[2:], atol=1e-6), f"op {op} direction"


def test_the_augmented_index_labels_which_symmetry_it_used():
    solved = _samples()
    aug = T.LevelDataset(solved, derive_reverses=True, as_tensor=False,
                         augment_d4=True)
    n = len(solved)
    assert "d4" not in aug.info(0), "the identity block is not relabelled"
    assert aug.info(n)["d4"] == "rot90"
    assert aug.info(4 * n)["d4"] == "rot0_mirror"


# ---------------------------------------------------------------------------
# Global spectral descriptors. Six numbers a convolutional receptive field
# cannot reach, motivated by a measurement: Chetco Bar's gentle cells are
# 3.7x worse than Flatirons' gentle cells at identical LOCAL slope.
# ---------------------------------------------------------------------------

def _ridged(n=64, wavelength=8, amp=100.0, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n]
    return (500.0 + amp * np.sin(2 * np.pi * x / wavelength)
            + 5.0 * rng.standard_normal((n, n)))


def test_the_descriptors_are_invariant_under_every_symmetry():
    """The design constraint that keeps them compatible with augmentation.
    The anisotropy uses the eigenvalue RATIO of the spectral second-moment
    tensor and never its orientation, precisely so that rotating a window
    cannot change it."""
    h = _ridged()
    d0 = T.spectral_descriptors(h, 50.0, 50.0)
    for ang, mir in T.D4_OPS[1:]:
        d = T.spectral_descriptors(T.transform_field(h, ang, mir),
                                   50.0, 50.0)
        # Round-off only: rot180 is a double flip and is exact, while the
        # transposing operations reorder the FFT's summation.
        assert np.abs(d - d0).max() < 1e-4, (ang, mir)


def test_the_descriptors_separate_terrain_the_local_channels_cannot():
    """Two fields with the SAME slope statistics and different structure.
    If the descriptors cannot tell them apart they are not carrying the
    global information they exist for."""
    fine = _ridged(wavelength=4, amp=50.0)
    coarse = _ridged(wavelength=32, amp=50.0 * 8)   # same peak slope
    a = T.spectral_descriptors(fine, 50.0, 50.0)
    b = T.spectral_descriptors(coarse, 50.0, 50.0)
    i_long = T.SPECTRAL_CHANNELS.index("spec_long")
    i_short = T.SPECTRAL_CHANNELS.index("spec_short")
    assert b[i_long] > a[i_long], "the coarse field must be longer-scaled"
    assert a[i_short] > b[i_short]


def test_anisotropy_is_larger_for_a_ridged_field_than_an_isotropic_one():
    rng = np.random.default_rng(3)
    i_a = T.SPECTRAL_CHANNELS.index("spec_aniso")
    ridged = T.spectral_descriptors(_ridged(), 50.0, 50.0)[i_a]
    noise = T.spectral_descriptors(
        500.0 + 30.0 * rng.standard_normal((64, 64)), 50.0, 50.0)[i_a]
    assert ridged > noise


def test_the_descriptors_ignore_a_planar_trend():
    """Terrain is detrended before the transform. Without it the FFT of a
    tilted, non-periodic tile is dominated by the edge discontinuity and
    the descriptors would describe the window rather than the ground."""
    h = _ridged()
    y, x = np.mgrid[0:h.shape[0], 0:h.shape[1]]
    tilted = h + 2.0 * x + 1.5 * y
    assert np.allclose(T.spectral_descriptors(h, 50.0, 50.0),
                       T.spectral_descriptors(tilted, 50.0, 50.0),
                       atol=1e-4)


def test_the_spectral_channels_are_constant_planes_appended_in_order():
    a = _arrays(n=32)
    a["terrain"] = _ridged(n=32).astype("float32")
    x = T.make_input(a, 45.0, 50.0, 50.0, spectral=True)
    assert x.shape[0] == len(T.INPUT_CHANNELS) + len(T.SPECTRAL_CHANNELS)
    d = T.spectral_descriptors(a["terrain"], 50.0, 50.0)
    for j, v in enumerate(d):
        plane = x[len(T.INPUT_CHANNELS) + j]
        assert np.allclose(plane, v), T.SPECTRAL_CHANNELS[j]
        assert plane.min() == plane.max(), "must be constant"


def test_the_spectral_channels_survive_augmentation_untouched():
    """They are D4-invariant, so the transform must pass them through
    rather than rotating them -- rotating a constant plane is harmless but
    rotating the WRONG channel would not be."""
    solved = _samples(n=32)
    for info, a in solved:
        a["terrain"] = _ridged(n=32).astype("float32")
    ds = T.LevelDataset(solved, derive_reverses=True, as_tensor=False,
                        augment_d4=True, spectral=True)
    n = len(solved)
    base = ds[0][0][len(T.INPUT_CHANNELS):]
    for op in range(1, 8):
        got = ds[op * n][0][len(T.INPUT_CHANNELS):]
        assert np.allclose(got, base, atol=1e-4), op
