"""
training -- turning the dataset into what a network sees.

Phase 22b, and written before any architecture for the same reason the
scoring was (see :mod:`fastwindterrain.evaluate`): a data pipeline bug
does not announce itself. A terrain channel accidentally negated along
with the velocity, or a normalisation that divides away the amplitude the
answer depends on, produces a training run that converges to something
plausible and wrong. So the pipeline is built with invariants that can be
asserted without a model.

FOUR DECISIONS, EACH ALREADY MEASURED ELSEWHERE.

*Normalise by the reference speed.* Every field scales exactly with
``u_ref`` -- 2.1e-15 relative, measured in ``cases/linearity_study.py`` --
so wind speed is not an input and not an axis of the dataset. The network
learns one 10 m/s map and the answer at any other speed is a
multiplication. Targets are therefore dimensionless, and
:func:`to_ms` puts them back.

*Direction as sin/cos, not eight classes.* The operator is exactly odd in
the inflow, so reversing the wind negates the field. Two continuous
channels make that representable -- negating them negates the target --
where a one-hot over eight directions would force the network to
rediscover from data a symmetry it could have been handed.

*Terrain centred, scaled by a CONSTANT.* Per-sample scaling by relief is
the tempting normalisation and it is wrong: it divides away the very
amplitude that determines how much the terrain deflects the flow, so a
50 m hill and a 1500 m ridge would arrive identical. The mean is removed
per column stack because absolute elevation is not physical here -- the
corpus floor follows each window's own relief -- but the scale is fixed.

*Targets scaled per channel, by RMS, with no mean removed.* Measured over
the training fold: ``w`` is six times smaller than ``u`` and ``v``, and
the 5 m level is 2.7 times smaller than the top one. An unweighted mean
square is therefore dominated by the aloft horizontal channels -- the
easiest part of the column and the part nobody asked for -- while the
engineering band contributes almost nothing to the gradient. Dividing
each channel by its own RMS puts them on equal footing.

No mean is subtracted, and that is not an oversight. The dataset contains
every sample together with its exact negation, so the true per-channel
mean is identically zero; subtracting an estimated one would introduce a
small offset that BREAKS the oddness the dataset is built on. Scaling
alone preserves it exactly.

*Slope magnitude as a channel.* Measured in ``cases/slope_error.py``:
slope magnitude correlates with the reconstruction error at r = 0.50-0.72
above 20 m, while the along-wind slope correlates at zero and curvature at
most 0.18. The network is given the terrain itself as well, so nothing is
withheld -- this is a convenience feature, not a substitute.
"""

import numpy as np

__all__ = [
    "TERRAIN_SCALE_M",
    "INPUT_CHANNELS",
    "TARGET_FIELDS",
    "direction_channels",
    "terrain_channels",
    "make_input",
    "make_target",
    "channel_rms",
    "to_ms",
    "LevelDataset",
]

#: Metres. Terrain is centred per sample and divided by this, so a 500 m
#: relief window arrives at order one and a 1970 m one arrives at four --
#: which is the point. See the module docstring on why this is not the
#: per-sample relief.
TERRAIN_SCALE_M = 500.0

#: In order. ``make_input`` builds exactly this and nothing else, so the
#: list is the contract between the pipeline and any architecture.
INPUT_CHANNELS = ("terrain", "slope", "sin_dir", "cos_dir")

#: The three velocity components, each on every level. Stacked in this
#: order, so the target is ``(3 * nlev, ny, nx)``.
TARGET_FIELDS = ("u_lev", "v_lev", "w_lev")


def direction_channels(direction_deg, shape):
    """Two constant planes encoding the wind direction.

    The convention is the solver's: a direction of 45 degrees means wind
    FROM the northeast, so the flow vector points southwest. The same
    expression appears in ``corpus.window_config``; getting it backwards
    would mirror every sample and still train.
    """
    theta = np.radians(float(direction_deg))
    ux, uy = -np.sin(theta), -np.cos(theta)
    return (np.full(shape, ux, dtype=np.float32),
            np.full(shape, uy, dtype=np.float32))


def terrain_channels(terrain, dx, dy, scale=TERRAIN_SCALE_M):
    """Centred terrain and slope magnitude, as ``(2, ny, nx)`` float32."""
    h = np.asarray(terrain, dtype=np.float64)
    dzdy, dzdx = np.gradient(h, dy, dx)
    slope = np.sqrt(dzdx ** 2 + dzdy ** 2)
    return (((h - h.mean()) / float(scale)).astype(np.float32),
            slope.astype(np.float32))


def make_input(arrays, direction_deg, dx, dy, scale=TERRAIN_SCALE_M):
    """``(4, ny, nx)`` float32, channels in :data:`INPUT_CHANNELS` order."""
    ter, slope = terrain_channels(arrays["terrain"], dx, dy, scale)
    sx, cy = direction_channels(direction_deg, ter.shape)
    return np.stack([ter, slope, sx, cy])


def make_target(arrays, u_ref):
    """``(3 * nlev, ny, nx)`` float32, normalised by the reference speed."""
    return np.concatenate(
        [np.asarray(arrays[k], dtype=np.float32) for k in TARGET_FIELDS]
    ) / np.float32(u_ref)


def channel_rms(samples, u_ref, eps=1e-6):
    """Per-channel RMS of the target, in units of ``u_ref``.

    FIT ON THE TRAINING FOLD ONLY. These numbers are a property of the
    data and carry information about it; taking them over train and
    validation together is a small but real leak, and the sort that is
    never noticed afterwards.

    RMS rather than standard deviation because the mean is exactly zero
    by the dataset's symmetry -- see the module docstring.
    """
    acc, n = None, 0
    for info, arrays in samples:
        y = np.concatenate(
            [np.asarray(arrays[k], dtype=np.float64)
             for k in TARGET_FIELDS]) / float(u_ref)
        s = (y ** 2).sum(axis=tuple(range(1, y.ndim)))
        acc = s if acc is None else acc + s
        n += int(np.prod(y.shape[1:]))
    if acc is None:
        raise ValueError("no samples to fit channel scales on")
    return np.maximum(np.sqrt(acc / n), eps).astype(np.float32)


def to_ms(target, u_ref, scales=None):
    """Undo the normalisation: back to metres per second.

    Scoring happens in m/s and only in m/s -- see
    :mod:`fastwindterrain.evaluate` -- so every number that leaves a
    training run goes through here first.
    """
    y = np.asarray(target)
    if scales is not None:
        s = np.asarray(scales, dtype=np.float32)
        y = y * s.reshape(s.shape + (1,) * (y.ndim - 1))
    return y * float(u_ref)


class LevelDataset:
    """The dataset a 2D surrogate trains on.

    Holds the samples in memory: the whole training fold is 1296 samples
    of about 1.1 MB each, and reading it from the shards takes seven
    seconds, so paging from disk every epoch would cost more than it saves.
    With ``derive_reverses`` it holds only the solved half and negates on
    access, which halves the footprint and is exact.

    Subclassing ``torch.utils.data.Dataset`` is deliberately avoided: the
    class satisfies the map-style protocol (``__len__`` and
    ``__getitem__``), which is all ``DataLoader`` requires, and this way
    the module imports and its invariants can be tested with no torch
    installed.
    """

    def __init__(self, samples, u_ref=10.0, window_m=5000.0,
                 scale=TERRAIN_SCALE_M, derive_reverses=False,
                 as_tensor=True, scales=None):
        self.u_ref = float(u_ref)
        self.scales = (None if scales is None
                       else np.asarray(scales, dtype=np.float32))
        self.window_m = float(window_m)
        self.scale = float(scale)
        self.as_tensor = bool(as_tensor)

        items = list(samples)
        if derive_reverses:
            solved = [(i, a) for i, a in items if not i.get("derived")]
            if len(solved) != len(items):
                raise ValueError(
                    "derive_reverses expects the SOLVED samples only; "
                    f"{len(items) - len(solved)} of {len(items)} are "
                    "already derived, and deriving them again would "
                    "produce duplicates rather than the reverses")
            # Index into `solved` plus a sign. The reverse of a solve is
            # its exact negation (0.00e+00 over 1080 pairs), so this
            # costs one multiply and no memory.
            self._items = solved
            self._index = ([(k, 1.0) for k in range(len(solved))]
                           + [(k, -1.0) for k in range(len(solved))])
        else:
            self._items = items
            self._index = [(k, 1.0) for k in range(len(items))]

    def __len__(self):
        return len(self._index)

    def info(self, i):
        """The manifest entry for sample ``i``, with the derived half
        labelled as such rather than silently sharing its partner's id."""
        k, sign = self._index[i]
        info = dict(self._items[k][0])
        if sign < 0:
            d = (float(info["direction"]) + 180.0) % 360.0
            info.update(id=f"{info['id'].split('@')[0]}@{d:03.0f}",
                        direction=d, derived=True,
                        derived_from=self._items[k][0]["id"])
        return info

    def levels(self, i):
        """The level heights, in metres above ground, for sample ``i``.

        Needed to score: the band metric has to know which channels are
        the engineering levels, and the aloft levels scale with each
        window's own column, so this is not a constant.
        """
        k, _ = self._index[i]
        return np.asarray(self._items[k][1]["levels"], dtype=np.float64)

    def __getitem__(self, i):
        k, sign = self._index[i]
        info, arrays = self._items[k]
        direction = float(info["direction"])
        if sign < 0:
            direction = (direction + 180.0) % 360.0

        nx = arrays["terrain"].shape[-1]
        dx = dy = self.window_m / nx
        x = make_input(arrays, direction, dx, dy, self.scale)
        # Only the velocity is negated. The terrain and slope channels are
        # geometry and are IDENTICAL between a solve and its reverse; the
        # direction channels flip because make_input was handed the
        # reversed direction. Negating the terrain here would turn every
        # ridge into a valley and still produce a falling loss curve.
        y = make_target(arrays, self.u_ref) * np.float32(sign)
        if self.scales is not None:
            y = y / self.scales[:, None, None]

        if not self.as_tensor:
            return x, y
        import torch
        return torch.from_numpy(x), torch.from_numpy(y)
