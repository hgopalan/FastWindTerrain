====
Grid
====

The mesh is Cartesian with uniform ``x``/``y`` spacing and a
geometrically stretched ``z`` spacing, finer near the surface so the
atmospheric boundary layer is resolved without paying for that
resolution aloft.

.. code-block:: none

    dz(k) = dz0 * r^k,   k = 0 .. nz-1

where ``dz0`` is the surface-adjacent cell thickness and ``r`` is
``grid.stretching_ratio``. ``r = 1`` reproduces a plain uniform grid
exactly.

Inputs
======

=========================  =========  ===================================
Input                      Default    Meaning
=========================  =========  ===================================
``grid.n_cell``            --         Cell counts ``nx ny nz``
``grid.prob_lo``           --         Domain lower corner [m]
``grid.prob_hi``           --         Domain upper corner [m]
``grid.dz0``               --         Surface-adjacent cell thickness [m]
``grid.stretching_ratio``  ``1.0``    Geometric ratio ``r``
``grid.max_grid_size``     ``32``     AMReX box splitting size
=========================  =========  ===================================

Domain height policy
====================

``nz``, ``dz0``, ``r`` and the requested domain height are independent
inputs, so they can disagree. The code sums the geometric series to the
actual height ``H_computed`` and compares it against the requested
``H_requested = prob_hi[2] - prob_lo[2]``:

**Match** (within tolerance)
    Proceeds normally.

**Overshoot** -- ``H_computed > H_requested``
    Non-fatal warning, and ``grid.prob_hi[2]`` is overridden to
    ``H_computed`` so the grid and the domain agree exactly. Nothing is
    lost by growing the domain to fit the grid.

**Undershoot** -- ``H_computed < H_requested``
    Fatal. The grid does not reach the requested domain top, so
    proceeding would silently corrupt the top boundary condition and the
    mass balance. Increase ``n_cell[2]``, ``dz0``, or
    ``stretching_ratio``.

Example
=======

.. code-block:: none

    grid.n_cell           = 40 40 66
    grid.prob_lo          = 0.0 0.0 0.0
    grid.prob_hi          = 1000.0 1000.0 961.2758234855
    grid.dz0              = 2.0
    grid.stretching_ratio = 1.05
    grid.max_grid_size    = 32

Here ``H_computed = dz0 (r^nz - 1)/(r - 1) = 961.2758...`` m, matching
the requested height exactly.

A note on the vertical coordinate
=================================

AMReX's ``Geometry`` assumes uniform spacing in every direction. It is
still constructed -- the ``BoxArray``/``MultiFab``/``MFIter`` machinery
needs it, and ``x``/``y`` genuinely are uniform -- but the true vertical
coordinate is tracked separately as ``z_face`` (``nz+1`` entries) and
``z_cc`` (``nz`` entries).

**Anywhere a physical z spacing is needed, use those arrays and not**
``geom().CellSize(2)``. This applies to the Poisson stencil, vertical
integrals, and the flux areas used in the boundary mass balance.
