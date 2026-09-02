"""
Dataset generation -- N cases in one process on a fixed grid.

The point of the whole binding effort. A U-FNO trains on a stack of
samples that all have one tensor shape, so the grid is held fixed and
everything else is swept; the tests here are mostly about the ways that
can go quietly wrong.

Quietly is the operative word. A ragged dataset does not fail here -- it
fails hours into a training run, or not at all if the loader pads. So the
grid is checked rather than assumed, and a config that moves it is
refused at the point it is written.
"""

import json

import numpy as np
import pytest

import fastwindterrain as fwt
from fastwindterrain import dataset

NX, NY, NZ = 12, 12, 40


@pytest.fixture
def base(terrain_points):
    """A small case: this module runs several solves, so the grid is
    coarse enough that a sweep is seconds rather than minutes."""
    return {
        "grid": {"n_cell": (NX, NY, NZ), "prob_lo": (0.0, 0.0, 0.0),
                 "prob_hi": (1000.0, 1000.0, 483.19909696997223),
                 "dz0": 4.0, "stretching_ratio": 1.05, "max_grid_size": 16},
        "terrain": {"points": terrain_points},
        "inflow": {"mode": "powerlaw", "u_ref": 8.0, "v_ref": 6.0},
        "poisson": {"alpha_v": 0.5, "n_projections": 2},
    }


# ---------------------------------------------------------------------------
# Building configs
# ---------------------------------------------------------------------------

def test_sweep_is_the_cartesian_product(base):
    configs = dataset.sweep(base, {"inflow.u_ref": [4.0, 8.0],
                                   "poisson.alpha_v": [0.3, 0.5, 0.7]})
    assert len(configs) == 6
    assert [(c["inflow"]["u_ref"], c["poisson"]["alpha_v"]) for c in configs] \
        == [(4.0, 0.3), (4.0, 0.5), (4.0, 0.7),
            (8.0, 0.3), (8.0, 0.5), (8.0, 0.7)]


def test_sweep_does_not_mutate_the_base(base):
    dataset.sweep(base, {"inflow.u_ref": [4.0, 8.0]})
    assert base["inflow"]["u_ref"] == 8.0


def test_sweep_shares_the_terrain_array(base):
    """Deep-copying a point cloud once per sample turns a sweep of a few
    hundred cases from megabytes into gigabytes, and nothing here mutates
    it."""
    configs = dataset.sweep(base, {"inflow.u_ref": [4.0, 8.0, 12.0]})
    assert all(c["terrain"]["points"] is base["terrain"]["points"]
               for c in configs)


def test_random_sweep_is_reproducible_from_its_seed(base):
    a = dataset.random_sweep(base, {"inflow.u_ref": (2.0, 15.0)}, 5, seed=3)
    b = dataset.random_sweep(base, {"inflow.u_ref": (2.0, 15.0)}, 5, seed=3)
    c = dataset.random_sweep(base, {"inflow.u_ref": (2.0, 15.0)}, 5, seed=4)

    speeds = [x["inflow"]["u_ref"] for x in a]
    assert speeds == [x["inflow"]["u_ref"] for x in b]
    assert speeds != [x["inflow"]["u_ref"] for x in c]
    assert all(2.0 <= s <= 15.0 for s in speeds)


def test_random_sweep_keeps_integer_ranges_integral(base):
    configs = dataset.random_sweep(
        base, {"poisson.n_projections": (2, 6)}, 20, seed=0)
    values = [c["poisson"]["n_projections"] for c in configs]
    assert all(isinstance(v, int) for v in values)
    assert set(values) <= {2, 3, 4, 5, 6}


def test_sweeping_the_grid_is_refused(base):
    """Every sample must have one tensor shape, so this cannot be an
    axis. Refusing it where it is written beats discovering it in a
    training run."""
    with pytest.raises(ValueError, match="fixed"):
        dataset.sweep(base, {"grid.dz0": [2.0, 4.0]})
    with pytest.raises(ValueError, match="fixed"):
        dataset.random_sweep(base, {"grid.dz0": (2.0, 4.0)}, 3)


@pytest.mark.parametrize("key", ["inflow", "inflow.a.b"])
def test_sweep_rejects_paths_that_are_not_section_dot_key(base, key):
    with pytest.raises(ValueError):
        dataset.sweep(base, {key: [1.0]})


# ---------------------------------------------------------------------------
# Running the cases
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_iter_samples_yields_one_sample_per_config(amrex, base):
    configs = dataset.sweep(base, {"inflow.u_ref": [4.0, 10.0]})
    samples = list(dataset.iter_samples(configs, fields=["u", "v", "w"]))

    assert [s.index for s in samples] == [0, 1]
    for s, cfg in zip(samples, configs):
        assert s.config is cfg
        assert sorted(s.arrays) == ["u", "v", "w"]
        assert s.arrays["u"].shape == (NZ, NY, NX)
        assert s.z_cc.shape == (NZ,)
        assert s.z_face.shape == (NZ + 1,)
        assert s.info["solve_iterations"] > 0

    # Different inflows must give different fields, or the loop is
    # returning the same solve twice.
    assert not np.array_equal(samples[0].arrays["u"], samples[1].arrays["u"])


@pytest.mark.slow
def test_a_changed_grid_is_caught(amrex, base):
    """The check that keeps a dataset from going ragged. It names the key
    that moved, because "shapes differ" is not a debuggable message."""
    configs = [base, {**base, "grid": dict(base["grid"], n_cell=(8, 8, 40))}]
    with pytest.raises(ValueError, match="n_cell"):
        list(dataset.iter_samples(configs, fields=["u"]))


def test_iter_samples_needs_an_initialized_amrex(run_py):
    """Deliberately not implicit: AMReX's lifecycle is process-global,
    and a library that initialized it behind your back would be
    impossible to compose with anything else. ``generate`` is the
    convenience wrapper; the iterator says so instead.

    Out of process, since by the time this module runs the session
    fixture has AMReX up -- in-process it would skip and test nothing.
    """
    r = run_py("""
import fastwindterrain as fwt
from fastwindterrain import dataset
assert not fwt.is_initialized()
try:
    list(dataset.iter_samples([{"grid": {}}]))
    print("::RESULT accepted")
except RuntimeError as e:
    print("::RESULT raised")
    print("::MENTIONS_SESSION", "session" in str(e))
""")
    assert r["RESULT"] == "raised"
    assert r["MENTIONS_SESSION"] == "True", (
        "the message should say how to fix it")


def test_unknown_field_names_are_refused(amrex, base):
    with pytest.raises(ValueError, match="unknown field"):
        list(dataset.iter_samples([base], fields=["velocity"]))


# ---------------------------------------------------------------------------
# Writing and reading back
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def generated(request, tmp_path_factory):
    """One three-sample dataset, written once and inspected by several
    tests -- each sample is a full solve."""
    from conftest import TERRAIN_CSV
    request.getfixturevalue("amrex")
    pts = np.loadtxt(TERRAIN_CSV, delimiter=",", comments="#", skiprows=5)
    base = {
        "grid": {"n_cell": (NX, NY, NZ), "prob_lo": (0.0, 0.0, 0.0),
                 "prob_hi": (1000.0, 1000.0, 483.19909696997223),
                 "dz0": 4.0, "stretching_ratio": 1.05, "max_grid_size": 16},
        "terrain": {"points": pts},
        "inflow": {"mode": "powerlaw", "u_ref": 8.0, "v_ref": 6.0},
        "poisson": {"alpha_v": 0.5, "n_projections": 2},
    }
    configs = dataset.sweep(base, {"inflow.u_ref": [4.0, 8.0, 12.0]})
    out = tmp_path_factory.mktemp("dataset") / "wind.npz"
    manifest = dataset.generate(configs, str(out),
                                fields=["u", "v", "w", "mask"], seed=7)
    return out, manifest, configs


@pytest.mark.slow
def test_generate_stacks_every_sample(generated):
    out, _, configs = generated
    arrays, _ = dataset.load(str(out))
    for name in ("u", "v", "w", "mask"):
        assert arrays[name].shape == (len(configs), NZ, NY, NX)


@pytest.mark.slow
def test_the_vertical_coordinate_travels_with_the_data(generated):
    """With the grid fixed, the index-to-height mapping is one constant
    for every sample -- but only if it is actually stored."""
    out, _, _ = generated
    arrays, _ = dataset.load(str(out))
    assert arrays["grid_z_cc"].shape == (NZ,)
    assert arrays["grid_z_face"].shape == (NZ + 1,)
    assert np.all(np.diff(arrays["grid_z_cc"]) > 0)


@pytest.mark.slow
def test_the_manifest_records_provenance(generated):
    out, manifest, configs = generated
    _, loaded = dataset.load(str(out))

    assert loaded["n_samples"] == len(configs)
    assert loaded["fields"] == ["u", "v", "w", "mask"]
    assert loaded["fastwindterrain_version"] == fwt.__version__
    assert loaded["amrex_version"] == fwt.amrex_version()
    assert loaded["seed"] == 7
    assert manifest["created"] == loaded["created"]
    assert len(loaded["solves"]) == len(configs)


@pytest.mark.slow
def test_each_sample_keeps_the_config_that_produced_it(generated):
    out, _, configs = generated
    _, manifest = dataset.load(str(out))
    assert [c["inflow"]["u_ref"] for c in manifest["configs"]] \
        == [c["inflow"]["u_ref"] for c in configs]


@pytest.mark.slow
def test_the_terrain_is_stamped_not_copied(generated):
    """A point cloud written once per sample would dominate the file, and
    the terrain is recoverable from the sample anyway -- terrain_z and
    mask are output fields. So the manifest keeps a hash."""
    out, _, configs = generated
    _, manifest = dataset.load(str(out))

    stamp = manifest["configs"][0]["terrain"]["points"]["__ndarray__"]
    assert stamp["shape"] == list(configs[0]["terrain"]["points"].shape)
    assert len(stamp["sha256"]) == 64
    # Same terrain in every sample, so the same stamp.
    assert all(c["terrain"]["points"]["__ndarray__"]["sha256"]
               == stamp["sha256"] for c in manifest["configs"])


@pytest.mark.slow
def test_the_manifest_is_json(generated):
    """It is stored as JSON inside the npz, so anything unserializable in
    a config has to be caught on the way in rather than at write time."""
    _, manifest, _ = generated
    json.dumps({k: v for k, v in manifest.items() if k != "files"})


@pytest.mark.slow
def test_samples_differ(generated):
    """Three different inflows; three different fields. Without this the
    tests above would pass on a dataset of three identical samples."""
    out, _, _ = generated
    arrays, _ = dataset.load(str(out))
    u = arrays["u"]
    assert not np.array_equal(u[0], u[1])
    assert not np.array_equal(u[1], u[2])


@pytest.mark.slow
def test_float32_halves_the_data_without_changing_the_solve(amrex, base,
                                                            tmp_path):
    """The solve is always double; the cast happens on the way out."""
    configs = dataset.sweep(base, {"inflow.u_ref": [6.0]})
    dataset.generate(configs, str(tmp_path / "f64.npz"), fields=["u"])
    dataset.generate(configs, str(tmp_path / "f32.npz"), fields=["u"],
                     dtype="float32")

    a, _ = dataset.load(str(tmp_path / "f64.npz"))
    b, _ = dataset.load(str(tmp_path / "f32.npz"))
    assert a["u"].dtype == np.float64
    assert b["u"].dtype == np.float32
    assert np.array_equal(b["u"], a["u"].astype(np.float32))


@pytest.mark.slow
def test_shards_hold_the_same_data_as_one_file(amrex, base, tmp_path):
    """Sharding is a memory strategy, not a different dataset."""
    configs = dataset.sweep(base, {"inflow.u_ref": [4.0, 8.0, 12.0]})
    dataset.generate(configs, str(tmp_path / "one.npz"), fields=["u"])
    dataset.generate(configs, str(tmp_path / "many"), fields=["u"],
                     shard_size=2)

    one, m1 = dataset.load(str(tmp_path / "one.npz"))
    many, m2 = dataset.load(str(tmp_path / "many"))

    assert sorted(p.name for p in (tmp_path / "many").iterdir()) == [
        "manifest.json", "shard_00000.npz", "shard_00001.npz"]
    assert np.array_equal(one["u"], many["u"])
    assert np.array_equal(one["grid_z_cc"], many["grid_z_cc"])
    assert m1["n_samples"] == m2["n_samples"] == 3


@pytest.mark.slow
def test_generate_opens_its_own_session_when_it_has_to(base, tmp_path,
                                                       run_py):
    """A script that only generates should not have to know about AMReX's
    lifecycle. Out of process, since this process's AMReX is already up.
    """
    from conftest import TERRAIN_CSV
    r = run_py(f"""
import numpy as np
import fastwindterrain as fwt
from fastwindterrain import dataset

pts = np.loadtxt({str(TERRAIN_CSV)!r}, delimiter=",", comments="#", skiprows=5)
base = {{
    "grid": {{"n_cell": ({NX}, {NY}, {NZ}), "prob_lo": (0.0, 0.0, 0.0),
              "prob_hi": (1000.0, 1000.0, 483.19909696997223),
              "dz0": 4.0, "stretching_ratio": 1.05, "max_grid_size": 16}},
    "terrain": {{"points": pts}},
    "inflow": {{"mode": "powerlaw", "u_ref": 8.0, "v_ref": 6.0}},
    "poisson": {{"alpha_v": 0.5, "n_projections": 1}},
}}
print("::BEFORE", fwt.is_initialized())
m = dataset.generate([base], {str(tmp_path / "auto.npz")!r}, fields=["u"])
print("::AFTER", fwt.is_initialized())
print("::N", m["n_samples"])
""")
    assert r["BEFORE"] == "False"
    assert r["AFTER"] == "False", "generate() left AMReX initialized"
    assert r["N"] == "1"


def test_empty_config_list_is_refused(amrex, tmp_path):
    with pytest.raises(ValueError, match="no configs"):
        dataset.generate([], str(tmp_path / "nothing.npz"))
