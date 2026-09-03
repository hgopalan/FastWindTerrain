"""
Cheap analytical wind fields, to answer "compared to what?".

A surrogate is only worth its complexity if something simpler does worse.
These are the simpler things. None of them learns anything, none costs more
than a few array operations, and every one of them is what a reviewer will
reach for when asked why a neural operator was needed.

They also bound the claim from the other side. A surrogate trained on the
mass-consistent solver cannot be more accurate than that solver -- its
target IS the solver's field -- so accuracy is not the value proposition;
speed is. What these baselines establish is how much of the terrain effect
is available for free, and therefore how much the surrogate is actually
being asked to add.

    undisturbed          the inflow profile, terrain-following, with no
                         terrain effect on speed at all. The "do nothing"
                         field, and the denominator of any skill score.
    continuity_speedup   the profile scaled by how much the terrain
                         squeezes the air column. A one-line mass argument,
                         and the zeroth-order version of what the
                         mass-consistent solve does properly.
    slope_speedup        a Jackson-Hunt-shaped fractional speed-up: linear
                         in local slope, decaying exponentially with height
                         over a length set by the hill.

WHAT THESE ARE NOT. They are not tuned to the solver, and no coefficient
here was fitted to it. `slope_speedup` in particular has two free constants
that a real Jackson-Hunt analysis would derive from the hill's own length
scale; the defaults are conventional rather than optimal, so treat it as a
family rather than a single model, and say so if it is quoted.
"""

from __future__ import annotations

import numpy as np

from .levels import first_fluid_k, height_above_ground, surface_nz

__all__ = [
    "continuity_speedup",
    "profile_speed",
    "slope_speedup",
    "undisturbed",
]


def profile_speed(z_agl, speed_ref, z_ref, mode="powerlaw", exponent=0.14,
                  z0=0.1, z_agl_min=None):
    """The 1D inflow law, matching ``Inflow::ProfileSpeed``.

    Transcribed rather than imported so a baseline can be evaluated without
    building a solver -- and floored the same way, because both laws
    misbehave as ``z_agl -> 0`` and the solver floors at ``z_agl_min``
    (default ``z0``) before evaluating either.
    """
    z_agl_min = z0 if z_agl_min is None else z_agl_min
    z = np.maximum(np.asarray(z_agl, dtype=np.float64), z_agl_min)
    if mode == "powerlaw":
        return speed_ref * (z / z_ref) ** exponent
    if mode == "loglaw":
        return speed_ref * np.log((z + z0) / z0) / np.log((z_ref + z0) / z0)
    raise ValueError(f"mode must be 'powerlaw' or 'loglaw', got {mode!r}")


def undisturbed(z_cc, z_terrain, mask, speed_ref, direction_deg,
                z_ref=80.0, mode="powerlaw", exponent=0.14, z0=0.1,
                fill=0.0):
    """``(3, nz, ny, nx)``: the profile, terrain-following, terrain-blind.

    Every column gets the same profile against its own ground, and the wind
    keeps one direction everywhere. This is the field the solver starts
    from before O'Brien and the projection, so comparing against it says
    exactly how much the mass-consistent adjustment is worth -- and any
    surrogate that cannot beat it has learned nothing.
    """
    zt = np.asarray(z_terrain, dtype=np.float64)
    if zt.ndim == 3:
        zt = zt[0]
    agl = height_above_ground(z_cc, zt)
    speed = profile_speed(agl, speed_ref, z_ref, mode, exponent, z0)

    theta = np.radians(direction_deg)
    ux, uy = -np.sin(theta), -np.cos(theta)

    out = np.stack([speed * ux, speed * uy, np.zeros_like(speed)])
    solid = np.asarray(mask) == 1
    out[:, solid] = fill
    return out


def continuity_speedup(z_cc, z_terrain, mask, speed_ref, direction_deg,
                       z_ref=80.0, mode="powerlaw", exponent=0.14, z0=0.1,
                       kinematic_w=True, dx=None, dy=None, fill=0.0):
    """The profile, accelerated where the terrain squeezes the air column.

    One line of physics: if the domain is ``D`` deep and the ground rises
    ``h`` into it, the air above has only ``D - h`` to pass through, so it
    speeds up by ``D / (D - h)``. That is mass conservation applied to a
    whole column at once, which is the zeroth-order version of what the
    mass-consistent solve does cell by cell.

    It costs three array operations and no solve, and it captures the
    first-order terrain effect -- flow accelerating over high ground. It
    captures nothing about WHERE the flow goes: no channelling, no
    separation, no deflection around an obstacle, and its speed-up is
    uniform up the column rather than decaying with height.

    With ``kinematic_w`` and the grid spacing it also sets ``w = u.grad(h)``
    in the first fluid cell, the same kinematic condition the solver's
    surface treatment imposes, so that ``w`` is not trivially zero
    everywhere and the comparison on ``w`` is not a straw man.
    """
    zt = np.asarray(z_terrain, dtype=np.float64)
    if zt.ndim == 3:
        zt = zt[0]
    z_cc = np.asarray(z_cc, dtype=np.float64)

    out = undisturbed(z_cc, zt, mask, speed_ref, direction_deg, z_ref,
                      mode, exponent, z0, fill=0.0)

    # Depth of the domain over the LOWEST ground, and what each column has
    # left. Referenced to the lowest ground rather than to prob_lo so the
    # factor is 1 where the terrain is at its floor, instead of depending
    # on how much empty rock the domain happens to contain below it.
    top = float(z_cc[-1])
    base = float(zt.min())
    depth = max(top - base, 1e-9)
    remaining = np.maximum(top - zt, 1e-9)
    factor = (depth / remaining)[None, None, :, :]

    out = out * factor
    solid = np.asarray(mask) == 1
    out[:, solid] = fill

    if kinematic_w and dx is not None and dy is not None:
        _apply_kinematic_w(out, zt, mask, dx, dy)
        out[:, solid] = fill
    return out


def slope_speedup(z_cc, z_terrain, mask, speed_ref, direction_deg,
                  z_ref=80.0, mode="powerlaw", exponent=0.14, z0=0.1,
                  amplitude=2.0, decay_length=None, dx=None, dy=None,
                  kinematic_w=True, fill=0.0):
    """A Jackson-Hunt-shaped fractional speed-up.

    Linearised theory for flow over a low hill gives a fractional speed-up
    that scales with the hill's aspect ratio and decays with height over
    the hill's own length scale. The engineering form used here is

        |U| = U_profile(z_agl) * (1 + amplitude * s * exp(-z_agl / L))

    with ``s`` the local terrain slope magnitude and ``L`` the decay
    length. ``amplitude = 2`` is the conventional coefficient for the crest
    speed-up of a shallow hill.

    TWO HONEST CAVEATS. Linearised theory assumes a SHALLOW hill, and the
    terrain corpus reaches slopes near 2, which is far outside where it
    holds -- expect this to overpredict badly on the steep windows. And
    ``decay_length`` should come from the hill's own half-length, which a
    per-cell slope does not know; defaulting it to a fixed height makes
    this a family of models rather than one, so quote it with its
    parameters or not at all.
    """
    zt = np.asarray(z_terrain, dtype=np.float64)
    if zt.ndim == 3:
        zt = zt[0]
    if dx is None or dy is None:
        raise ValueError("slope_speedup needs dx and dy to form the slope")

    z_cc = np.asarray(z_cc, dtype=np.float64)
    L = float(decay_length) if decay_length is not None else 200.0

    # Slope magnitude from n_z, so the terrain gradient has one definition
    # across the codebase.
    nz_c = surface_nz(zt, dx, dy)
    slope = np.sqrt(np.maximum(1.0 / (nz_c * nz_c) - 1.0, 0.0))

    out = undisturbed(z_cc, zt, mask, speed_ref, direction_deg, z_ref,
                      mode, exponent, z0, fill=0.0)
    agl = height_above_ground(z_cc, zt)
    gain = 1.0 + amplitude * slope[None] * np.exp(-np.maximum(agl, 0.0) / L)
    out = out * gain[None]

    solid = np.asarray(mask) == 1
    out[:, solid] = fill
    if kinematic_w:
        _apply_kinematic_w(out, zt, mask, dx, dy)
        out[:, solid] = fill
    return out


def _apply_kinematic_w(field, zt, mask, dx, dy):
    """``w = u dh/dx + v dh/dy`` in each column's first fluid cell."""
    ny, nx = zt.shape
    ip1 = np.minimum(np.arange(nx) + 1, nx - 1)
    im1 = np.maximum(np.arange(nx) - 1, 0)
    jp1 = np.minimum(np.arange(ny) + 1, ny - 1)
    jm1 = np.maximum(np.arange(ny) - 1, 0)
    dhdx = (zt[:, ip1] - zt[:, im1]) / ((ip1 - im1)[None, :] * dx)
    dhdy = (zt[jp1, :] - zt[jm1, :]) / ((jp1 - jm1)[:, None] * dy)

    k0 = np.minimum(first_fluid_k(mask), field.shape[1] - 1)
    u0 = np.take_along_axis(field[0], k0[None], axis=0)[0]
    v0 = np.take_along_axis(field[1], k0[None], axis=0)[0]
    np.put_along_axis(field[2], k0[None], (u0 * dhdx + v0 * dhdy)[None],
                      axis=0)
