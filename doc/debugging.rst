=========
Debugging
=========

``fwt.debug = 1`` turns on verbose diagnostics for the whole run::

    ./build/fastwindterrain inputs fwt.debug=1

It prints everything the run resolved and derived:

* every input that was parsed, with the ones that fell back to a
  **default** marked as such -- an unspecified default is what tends to
  be missed when a run does not do what the input file suggests
* the domain-height arithmetic: requested and computed heights, the
  relative difference against the match tolerance, and which of the
  three branches was taken
* the full ``k, z_face, dz, z_cc`` table
* the index domain, ``dx``/``dy``, nominal versus true z spacing,
  periodicity, the box list with owning rank, and cells per rank
* terrain source, point count and ranges, interpolated height range, and
  the solid-cell count and fraction
* the inflow mode, the profile speed at a few heights AGL, and the
  boundary mass flux
* every file written

Guarantees
==========

The switch is off by default, and with it off the output is unchanged
from before the switch existed.

Debug lines carry a ``[debug]`` prefix and never contain the words
``WARNING`` or ``ERROR``, so a debug run cannot confuse the regtest
checkers that key on those strings. A regtest case asserts exactly that,
plus that the default really is silent.

Tables longer than 200 rows are elided in the middle so a large grid does
not bury the rest of the output.

Example
=======

.. code-block:: none

    [debug] === Vertical stretching ===
    [debug] nz               = 66
    [debug] dz(0)            = 2 m   (surface-adjacent)
    [debug] dz(nz-1)         = 47.67980112 m   (domain top)
    [debug] H_requested      = 961.2758235 m
    [debug] H_computed       = 961.2758235 m   (sum of dz0*r^k)
    [debug] relative diff    = 2.933012052e-14   (match tolerance 1e-08)
    [debug] height check     = exact match -> no adjustment
