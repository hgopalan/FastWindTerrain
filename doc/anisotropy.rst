=====================================
Anisotropy and the O'Brien adjustment
=====================================

Two terrain-driven refinements to the adjustment: variational weights
that vary cell by cell with the local slope, and a vertical-velocity
correction that makes ``w = 0`` exact at the domain top.

Cell-local anisotropy
=====================

``alpha_h`` and ``alpha_v`` become fields rather than constants:

.. code-block:: none

    alpha_v = clamp( alpha_v_base * f_slope * f_Ri * f_Fr )
    alpha_h = clamp( alpha_h_base * [f_slope] * f_Ri * f_Fr )

    slope    = |grad z_terrain|            central differences
    slope_3d = slope * exp(-z_agl / decay_height)
    f_slope  = exp(-slope_3d / slope_scale)

following ``massconsistent_amr``'s ``cell_local_anisotropy.H``. The
suppression is strongest at the surface on steep ground and decays with
height above it.

Since ``alpha`` is a transmissivity (see :doc:`poisson`), a smaller
``alpha_v`` means *less* vertical adjustment: flow is pushed around steep
terrain rather than over it.

Inputs
------

.. list-table::
   :widths: 30 14 44
   :header-rows: 1

   * - Input
     - Default
     - Meaning
   * - ``anisotropy.enable``
     - ``0``
     - Off by default, as in ``massconsistent_amr``
   * - ``anisotropy.source``
     - ``slope``
     - ``slope`` or ``none``
   * - ``anisotropy.alpha_h_mode``
     - ``base``
     - Whether ``f_slope`` reaches the horizontal weight
   * - ``anisotropy.slope_scale``
     - ``0.5``
     - Slope at which the factor falls to ``1/e``
   * - ``anisotropy.decay_height``
     - ``500.0``
     - Height scale over which the suppression decays [m]
   * - ``anisotropy.min_factor``
     - ``0.05``
     - Lower clamp, as a factor on the base
   * - ``anisotropy.max_factor``
     - ``2.0``
     - Upper clamp

The base values come from ``poisson.alpha_h`` and ``poisson.alpha_v``,
since that is the operator they feed.

Why ``alpha_h_mode`` defaults to ``base``
-----------------------------------------

``massconsistent_amr`` holds ``alpha_h`` at its base value and varies
only ``alpha_v``. That is the default here too, and it is the right one:
what the model needs is for the **ratio** ``alpha_v / alpha_h`` to fall
on slopes. Scaling both weights by the same factor would suppress the
vertical adjustment relative to nothing at all.

Both are nevertheless full fields, so ``alpha_h_mode = slope`` lets the
slope factor reach the horizontal weight as well, and the stability hooks
below are wired into both.

Stability hooks
---------------

``f_Ri`` and ``f_Fr`` are present in the formulation and both return
exactly ``1``, which is the neutral case. They are wired into both
weights so that adding the Richardson and Froude terms later needs no
structural change -- only the two functions.

Disabled by default
-------------------

With ``anisotropy.enable = 0`` both fields hold their base values
everywhere, so the operator is exactly the one the earlier phases built.
A regtest asserts that, so the feature cannot change results when it is
switched off.

The O'Brien vertical-velocity adjustment
========================================

Integrates the horizontal divergence up each column to recover ``w`` from
continuity, then removes whatever is left at the domain top with a
quadratic-in-height redistribution, so ``w = 0`` there **exactly** rather
than approximately.

Two passes over each column, from its first fluid cell ``k_start``:

.. code-block:: none

    1.  w_top = w(k_start) - sum_{k > k_start} Dh(k) dz(k)
        E = w_top                       the residual continuity leaves

    2.  re-integrate, subtracting frac^2 E as it goes, where
        frac = (k - k_start) / (k_top - k_start)

At the top ``frac = 1``, so the accumulated value and the adjustment are
both ``E`` and ``w`` comes out exactly zero. The quadratic weighting puts
the correction aloft and leaves the near-surface ``w`` nearly untouched,
which is the point of the scheme: that is where the divergence estimate
is most trustworthy.

The vertical sum uses the true ``dz(k)``, not the nominal spacing.

Ordering
--------

The adjustment runs on ``u0`` **before** the projection, which is where
``massconsistent_amr`` applies it. Running it afterwards would rewrite
``w`` and put back divergence the solve had just removed.

Enabled with ``obrien.enable = 1``; off by default.

A note on the grid decomposition
--------------------------------

A column integration is non-local in z, so it cannot be done box by box
if a column is split between boxes. The grid is therefore decomposed in
**x and y only** -- every box spans the full height -- and
:doc:`grid` asserts that invariant when it builds the BoxArray.

The cost is nothing in practice, since atmospheric domains have
``nx, ny >> nz`` and the horizontal split carries the parallelism anyway.
Getting this wrong is not a numerical inaccuracy but undefined behaviour,
and it does not announce itself: before the decomposition was fixed, the
integration produced residuals around ``1e107``.
