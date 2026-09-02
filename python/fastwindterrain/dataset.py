"""
Dataset generation: many cases in one process, on one fixed grid.

This is the module the bindings were built for. A U-Net Fourier Neural
Operator trains on a stack of samples that all have the SAME tensor
shape, so the grid is held fixed across a run and everything else --
inflow speed and direction, anisotropy, terrain -- is free to vary::

    import numpy as np
    import fastwindterrain as fwt
    from fastwindterrain import dataset

    base = {
        "grid": {"n_cell": (24, 24, 40), "prob_lo": (0.0, 0.0, 0.0),
                 "prob_hi": (1000.0, 1000.0, 483.19909696997223),
                 "dz0": 4.0, "stretching_ratio": 1.05},
        "terrain": {"points": np.loadtxt("terrain.csv", delimiter=",")},
        "poisson": {"alpha_v": 0.5, "n_projections": 4},
    }
    configs = dataset.sweep(base, {"inflow.u_ref": [4.0, 8.0, 12.0],
                                   "inflow.v_ref": [0.0, 6.0]})

    with fwt.session():
        dataset.generate(configs, "wind.npz", fields=["u", "v", "w", "mask"])

Six cases, one process, one AMReX initialization, one file.

WHY ONE PROCESS MATTERS. AMReX's Initialize/Finalize are process-global
and cannot be repeated, so a per-case subprocess would pay the AMReX
startup on every sample. More importantly ParmParse is global and
persists, which is why every case here is described by a dict and never
by an inputs file: an absent key means the default, not "whatever the
previous case set". Dict-configured cases are independent by
construction.

THE GRID IS CHECKED, NOT ASSUMED. ``iter_samples`` raises if a config
changes the grid section, naming the key that moved. A silently ragged
dataset is the failure this guards against -- it does not surface until
a training run dies on a shape mismatch hours later, or worse, does not
surface at all because the loader padded.

STRETCHED z IS HARMLESS HERE. The vertical spacing is non-uniform, but
with the grid fixed the index-to-height mapping is the same constant for
every sample, so a network learning in index space is learning under one
coordinate system. ``grid_z_cc`` and ``grid_z_face`` are stored in the
output so that mapping travels with the data.

PARITY IS PER BUILD. Generate a whole dataset with one build of the
bindings. Mixing a wheel and an in-tree build is mixing two
compilations; see pyproject.toml.
"""

from __future__ import annotations

import contextlib
import copy
import datetime
import hashlib
import itertools
import json
import os
import sys
import typing

import numpy as np

from . import __version__, amrex_version, is_initialized, session
from ._fastwindterrain import Solver

__all__ = [
    "FIELDS",
    "Sample",
    "generate",
    "iter_samples",
    "load",
    "random_sweep",
    "sweep",
]

#: Every field ``Solver.fields()`` returns, in the order the C++ output
#: assembles them. Passing a subset to ``generate`` is usually right: a
#: surrogate that predicts velocity from terrain and inflow needs five or
#: six of these, and storing all seventeen costs three times the disk.
FIELDS = (
    "z_cc", "dz", "terrain_z", "mask",
    "u", "v", "w",
    "sigma_x", "sigma_y", "sigma_z",
    "u0", "v0", "w0",
    "alpha_h", "alpha_v",
    "lambda", "divergence",
)


class Sample(typing.NamedTuple):
    """One solved case, as :func:`iter_samples` yields it.

    A NamedTuple, so ``index, arrays, config, info, z_cc, z_face = sample``
    works and so does ``sample.arrays``.
    """

    #: Position in the config sequence, counting from zero.
    index: int
    #: ``{name: ndarray}``, each ``(nz, ny, nx)``, for the selected fields.
    arrays: dict
    #: The config dict this case was solved from -- the same object.
    config: dict
    #: Per-case numbers: MLMG residual and iterations, max divergence,
    #: projections done. JSON-safe scalars, so it goes into the manifest
    #: as it is.
    info: dict
    #: Cell-centre heights, ``(nz,)``. Identical for every sample, since
    #: the grid is fixed; carried on each one so a consumer of a single
    #: sample is not missing its vertical coordinate.
    z_cc: np.ndarray
    #: Face heights, ``(nz+1,)``.
    z_face: np.ndarray


# ---------------------------------------------------------------------------
# Building configs
# ---------------------------------------------------------------------------

def _set_path(cfg, dotted, value):
    """``_set_path(c, "inflow.u_ref", 8.0)`` -> ``c["inflow"]["u_ref"] = 8.0``."""
    head, _, tail = dotted.partition(".")
    if not tail:
        raise ValueError(
            f"{dotted!r} names a whole section, not a parameter. Use "
            f"'section.key', e.g. 'inflow.u_ref'.")
    if "." in tail:
        raise ValueError(
            f"{dotted!r} is nested deeper than the config is: a solver "
            f"config is exactly two levels, 'section.key'.")
    cfg.setdefault(head, {})[tail] = value


def _copy_config(cfg):
    """Deep-copy a config, sharing numpy arrays rather than duplicating them.

    Terrain point clouds run to hundreds of thousands of rows and are the
    same object in every config of a sweep. ``copy.deepcopy`` would make
    one copy per sample; nothing here mutates an array, so sharing is
    safe and is the difference between a sweep costing megabytes and
    gigabytes.
    """
    if isinstance(cfg, dict):
        return {k: _copy_config(v) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [_copy_config(v) for v in cfg]
    if isinstance(cfg, np.ndarray):
        return cfg
    return copy.deepcopy(cfg)


def sweep(base, axes):
    """The cartesian product of ``axes`` over a copy of ``base``.

    ``axes`` maps ``"section.key"`` to a list of values::

        sweep(base, {"inflow.u_ref": [4.0, 8.0], "poisson.alpha_v": [0.3, 0.7]})

    gives four configs. Order is the cartesian product's: the LAST axis
    varies fastest, so runs of the same terrain or inflow stay adjacent.

    Raises if an axis names the grid section -- see ``iter_samples``.
    """
    keys = list(axes)
    _reject_grid_axes(keys)
    out = []
    for combo in itertools.product(*(list(axes[k]) for k in keys)):
        cfg = _copy_config(base)
        for k, v in zip(keys, combo):
            _set_path(cfg, k, v)
        out.append(cfg)
    return out


def random_sweep(base, axes, n, seed=None):
    """``n`` configs with each axis drawn uniformly from a ``(lo, hi)`` range.

    ``axes`` maps ``"section.key"`` to a two-tuple::

        random_sweep(base, {"inflow.u_ref": (2.0, 15.0),
                            "inflow.wind_dir": (0.0, 360.0)}, n=200, seed=0)

    An integer range -- both endpoints ``int`` -- is drawn as an integer
    on ``[lo, hi]`` inclusive, so ``{"poisson.n_projections": (2, 6)}``
    does what it looks like. Everything else is a float on ``[lo, hi)``.

    Pass ``seed`` for a reproducible dataset. The seed is recorded in the
    manifest ``generate`` writes.
    """
    keys = list(axes)
    _reject_grid_axes(keys)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(int(n)):
        cfg = _copy_config(base)
        for k in keys:
            lo, hi = axes[k]
            if isinstance(lo, (int, np.integer)) and \
               isinstance(hi, (int, np.integer)) and \
               not isinstance(lo, bool) and not isinstance(hi, bool):
                _set_path(cfg, k, int(rng.integers(lo, hi + 1)))
            else:
                _set_path(cfg, k, float(rng.uniform(lo, hi)))
        out.append(cfg)
    return out


def _reject_grid_axes(keys):
    bad = [k for k in keys if k.split(".")[0] == "grid"]
    if bad:
        raise ValueError(
            f"the grid is fixed across a dataset, so it cannot be swept: "
            f"{sorted(bad)}. Every sample must have one tensor shape. To "
            f"cover several grids, generate one dataset per grid.")


# ---------------------------------------------------------------------------
# Running the cases
# ---------------------------------------------------------------------------

def _grid_signature(cfg):
    """The grid section, normalized so two equal grids compare equal."""
    g = cfg.get("grid", {})
    if not g:
        raise ValueError("every config needs a 'grid' section")

    def norm(v):
        if isinstance(v, np.ndarray):
            return tuple(v.ravel().tolist())
        if isinstance(v, (list, tuple)):
            return tuple(norm(x) for x in v)
        if isinstance(v, (np.integer, np.floating)):
            return v.item()
        return v

    return {k: norm(v) for k, v in g.items()}


@contextlib.contextmanager
def _quiet(enabled):
    """Silence the solver's output for the duration of the block.

    The chatter comes from ``amrex::Print`` on the C++ side, so it is
    written to file descriptor 1 directly and
    ``contextlib.redirect_stdout`` -- which only rebinds ``sys.stdout``
    -- does not touch it. Redirecting the descriptor does.

    Twenty lines per case is fine for one case and unreadable for two
    hundred. Errors still raise as exceptions, so nothing is lost; only
    the progress narration is dropped.
    """
    if not enabled:
        yield
        return
    sys.stdout.flush()
    try:
        saved = os.dup(1)
    except OSError:                       # no real fd 1 (some notebooks)
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)


def _solve_one(cfg, quiet):
    s = Solver(cfg)
    with _quiet(quiet):
        s.setup()
        s.solve()
        s.diagnose()
    return s


def iter_samples(configs, fields=None, quiet=True):
    """Solve each config in turn, yielding a :class:`Sample`.

    ``sample.arrays`` is ``{name: ndarray}`` with each array
    ``(nz, ny, nx)``; ``sample.info`` carries the per-case numbers worth
    keeping -- the MLMG residual and iteration count, the final max
    divergence.

    Streaming, one solve at a time: use this when a dataset is larger
    than memory, or when samples go somewhere other than an ``.npz``.
    ``generate`` is this loop plus a writer.

    Requires an initialized AMReX; wrap the loop in ``fwt.session()``.
    Every config's grid section must match the first's.
    """
    if not is_initialized():
        raise RuntimeError(
            "AMReX is not initialized. Wrap the loop:\n"
            "    with fwt.session():\n"
            "        for sample in dataset.iter_samples(configs): ...")

    names = list(FIELDS if fields is None else fields)
    unknown = [n for n in names if n not in FIELDS]
    if unknown:
        raise ValueError(f"unknown field(s) {unknown}; known: {list(FIELDS)}")

    reference = None
    shape = None
    for i, cfg in enumerate(configs):
        sig = _grid_signature(cfg)
        if reference is None:
            reference = sig
        elif sig != reference:
            moved = sorted(k for k in set(sig) | set(reference)
                           if sig.get(k) != reference.get(k))
            raise ValueError(
                f"config {i} changes the grid, which is fixed across a "
                f"dataset: {moved} differ from config 0. Every sample must "
                f"have one tensor shape.")

        s = _solve_one(cfg, quiet)
        got = s.fields()
        arrays = {n: got[n] for n in names}

        # The signature check above catches a changed grid *parameter*;
        # this catches anything else that could move the shape, which is
        # the property the whole dataset depends on.
        this_shape = next(iter(arrays.values())).shape
        if shape is None:
            shape = this_shape
        elif this_shape != shape:
            raise RuntimeError(
                f"config {i} produced {this_shape}, not {shape}, with an "
                f"unchanged grid section. This is a bug -- please report it.")

        info = {
            "solve_residual": float(s.solve_residual),
            "solve_iterations": int(s.solve_iterations),
            "max_divergence": float(s.max_divergence),
            "max_divergence_fe": float(s.max_divergence_fe),
            "n_projections_done": int(s.n_projections_done),
        }
        yield Sample(index=i, arrays=arrays, config=cfg, info=info,
                     z_cc=s.grid.z_cc, z_face=s.grid.z_face)
        del s, got, arrays


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def _jsonable(v):
    """A config as JSON, with arrays reduced to a shape/dtype/hash stamp.

    A terrain point cloud must not be written once per sample -- it is
    the same array in every config and would dominate the file. The hash
    identifies it exactly, and the terrain itself is recoverable from the
    sample: ``terrain_z`` and ``mask`` are output fields.
    """
    if isinstance(v, np.ndarray):
        a = np.ascontiguousarray(v)
        return {"__ndarray__": {
            "shape": list(a.shape),
            "dtype": str(a.dtype),
            "sha256": hashlib.sha256(a.tobytes()).hexdigest(),
        }}
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.integer, np.floating, np.bool_)):
        return v.item()
    return v


def _manifest(configs, infos, names, grid, seed, dtype):
    return {
        "format": "fastwindterrain-dataset",
        "format_version": 1,
        "created": datetime.datetime.now(
            datetime.timezone.utc).replace(microsecond=0).isoformat(),
        "fastwindterrain_version": __version__,
        "amrex_version": amrex_version(),
        "n_samples": len(configs),
        "fields": list(names),
        "dtype": str(dtype),
        "seed": seed,
        "shape": list(grid["shape"]),
        "grid": grid["section"],
        "configs": [_jsonable(c) for c in configs],
        "solves": infos,
        # Stated rather than implied: a dataset assembled from two builds
        # of the bindings is two datasets.
        "note": ("bit-for-bit reproducibility holds within one build of "
                 "the bindings, not across builds"),
    }


# ---------------------------------------------------------------------------
# Generating
# ---------------------------------------------------------------------------

def generate(configs, path, fields=None, dtype="float64", quiet=True,
             shard_size=None, compress=False, progress=False, seed=None):
    """Solve every config and write one dataset.

    ``path`` with no ``shard_size`` is a single ``.npz`` holding, for
    each selected field, one array of shape ``(n_samples, nz, ny, nx)``,
    plus ``grid_z_cc`` / ``grid_z_face`` and a ``manifest`` JSON string.
    With ``shard_size`` set, ``path`` is a DIRECTORY of
    ``shard_00000.npz`` files of at most that many samples each, with the
    manifest written once as ``manifest.json``. Shards keep peak memory
    at one shard rather than the whole dataset; ``load`` reads either
    layout.

    ``fields`` selects which of :data:`FIELDS` to keep -- worth setting,
    since all seventeen is usually three times the data a surrogate
    needs. ``dtype="float32"`` halves the file and is what training will
    cast to anyway; the solve itself is always double.

    ``compress`` trades generation time for disk. Field data is smooth
    and compresses about two-fold, but ``savez_compressed`` is slow on
    large arrays, so it is off by default.

    Returns the manifest dict, which is also written into the output.

    AMReX must be initialized; ``generate`` opens a ``fwt.session()``
    itself if it is not, so a script that only generates need not.
    """
    configs = list(configs)
    if not configs:
        raise ValueError("no configs to generate from")
    if shard_size is not None and int(shard_size) < 1:
        raise ValueError("shard_size must be at least 1")

    ctx = contextlib.nullcontext() if is_initialized() else session()
    with ctx:
        return _generate_inner(configs, path, fields, dtype, quiet,
                               shard_size, compress, progress, seed)


def _generate_inner(configs, path, fields, dtype, quiet, shard_size,
                    compress, progress, seed):
    names = list(FIELDS if fields is None else fields)
    dtype = np.dtype(dtype)
    n = len(configs)
    sharded = shard_size is not None
    shard_size = int(shard_size) if sharded else n
    writer = np.savez_compressed if compress else np.savez

    if sharded:
        os.makedirs(path, exist_ok=True)

    infos = []
    grid_meta = None
    buf = None
    filled = 0
    shard_index = 0
    written = []

    def flush():
        nonlocal buf, filled, shard_index
        if filled == 0:
            return
        payload = {k: v[:filled] for k, v in buf.items()}
        payload["grid_z_cc"] = grid_meta["z_cc"]
        payload["grid_z_face"] = grid_meta["z_face"]
        if sharded:
            out = os.path.join(path, f"shard_{shard_index:05d}.npz")
        else:
            out = path
            payload["manifest"] = np.array(json.dumps(
                _manifest(configs, infos, names, grid_meta, seed, dtype)))
        writer(out, **payload)
        written.append(out)
        shard_index += 1
        buf = None
        filled = 0

    for sample in iter_samples(configs, names, quiet=quiet):
        i = sample.index
        if grid_meta is None:
            # Taken once, from the first solve, and constant thereafter:
            # iter_samples has already refused any config that moves the
            # grid, so there is nothing to re-check per sample.
            grid_meta = {
                "shape": next(iter(sample.arrays.values())).shape,
                "z_cc": sample.z_cc,
                "z_face": sample.z_face,
                "section": _jsonable(sample.config.get("grid", {})),
            }
        if buf is None:
            take = min(shard_size, n - i)
            buf = {k: np.empty((take,) + grid_meta["shape"], dtype=dtype)
                   for k in names}
        for k in names:
            buf[k][filled] = sample.arrays[k]
        filled += 1
        infos.append(sample.info)
        if progress:
            print(f"[fwt.dataset] {i + 1}/{n}  "
                  f"residual {sample.info['solve_residual']:.3e}  "
                  f"max|div| {sample.info['max_divergence']:.3e}",
                  file=sys.stderr, flush=True)
        if filled == shard_size:
            flush()

    flush()

    manifest = _manifest(configs, infos, names, grid_meta, seed, dtype)
    if sharded:
        with open(os.path.join(path, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
    manifest["files"] = written
    return manifest


def load(path):
    """Read back what :func:`generate` wrote, single file or shard directory.

    Returns ``(arrays, manifest)``: ``arrays`` maps each field name to
    ``(n_samples, nz, ny, nx)``, and also carries ``grid_z_cc`` and
    ``grid_z_face``. ``manifest["configs"][i]`` is the config that
    produced sample ``i``.
    """
    if os.path.isdir(path):
        with open(os.path.join(path, "manifest.json")) as f:
            manifest = json.load(f)
        shards = sorted(f for f in os.listdir(path)
                        if f.startswith("shard_") and f.endswith(".npz"))
        if not shards:
            raise FileNotFoundError(f"no shard_*.npz in {path}")
        parts = []
        for s in shards:
            with np.load(os.path.join(path, s)) as z:
                parts.append({k: z[k] for k in z.files})
        arrays = {}
        for k in parts[0]:
            if k.startswith("grid_"):
                arrays[k] = parts[0][k]                # identical per shard
            else:
                arrays[k] = np.concatenate([p[k] for p in parts], axis=0)
        return arrays, manifest

    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files if k != "manifest"}
        manifest = json.loads(str(z["manifest"])) if "manifest" in z.files \
            else None
    return arrays, manifest
