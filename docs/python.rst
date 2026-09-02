===============
Python bindings
===============

pybind11 bindings over the same solver the executable runs. Off by
default; CMake only -- the GNUmake build stays C++ only.

Building
========

::

    git submodule update --init --recursive
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFWT_PYTHON=ON
    cmake --build build -j 8

pybind11 is vendored as a submodule alongside AMReX, so a recursive clone
needs no network at configure time. An installed pybind11 is used instead
if ``find_package`` can already see one.

The package is laid out under ``build/python``, so an import needs only
``PYTHONPATH``::

    PYTHONPATH=build/python python3 -c "import fastwindterrain as fwt"

Use the interpreter CMake found. An extension module is built for exactly
one minor version of Python, and the ``python3`` on ``PATH`` is often a
different one; ``build/fastwindterrain-py`` records the right one.

Using it
========

.. code-block:: python

    import fastwindterrain as fwt

    fwt.run(["inputs", "poisson.alpha_v=0.3"])

or equivalently from a shell::

    python -m fastwindterrain inputs poisson.alpha_v=0.3

.. list-table::
   :widths: 34 52
   :header-rows: 1

   * - Call
     - Meaning
   * - ``run(args)``
     - One case, exactly as the executable runs it: initialize AMReX,
       run the pipeline, finalize. ``args`` are the arguments **after**
       the program name
   * - ``initialize(args=[])`` / ``finalize()``
     - AMReX's process-global lifecycle, held explicitly
   * - ``session(args=[])``
     - Context manager around the pair, finalizing even if the block
       raises
   * - ``is_initialized()``
     - Whether this module has initialized AMReX
   * - ``__version__``, ``amrex_version()``
     - The project version, and the AMReX it was built against

Why the lifecycle is explicit
-----------------------------

``amrex::Initialize`` and ``amrex::Finalize`` are **process-global**, so
they cannot be run implicitly on import: importing a module twice, or
importing it alongside another AMReX-based module, would then be a crash
rather than a no-op.

They are held explicitly instead, and every misuse **raises** rather than
aborting: a second ``initialize()``, a ``finalize()`` with nothing to
finalize, and ``run()`` called inside an existing initialization. A guard
that segfaults instead of raising loses the whole session's work in a
notebook, which is the failure this API exists to prevent. A regtest
asserts each of them raises.

``run()`` refuses to run inside an existing initialization for a specific
reason: ``amrex::Initialize`` is what parses the inputs file into
ParmParse, so a second run inside the same initialization would silently
inherit the first run's settings.

Bit-for-bit parity
==================

The Python path and the executable produce **byte-identical** output.
Not "agree to a tolerance" -- identical, including the AMReX plotfile
and the printed log.

This is structural rather than aspirational. ``Source/`` builds as one
library, ``fwt_core``, and both the executable and the extension module
**link** it. They share object files, not merely source. Recompiling the
solver sources into the module -- the usual shortcut -- would leave two
compilations that could differ in flags, and floating-point results can
differ in the last bits when they do.

The pipeline itself lives in ``fwt::Solver`` rather than in ``main()``,
so there is one ordering of Grid -> Terrain -> Inflow -> BC -> Anisotropy
-> O'Brien -> Poisson -> projection -> diagnostics -> output, and both
entry points call it. That makes the executable a test of the class the
bindings expose: **every** regtest exercises this code.

What is checked
---------------

``build/fastwindterrain-py`` is an argv-compatible stand-in for the
executable, so the whole suite runs through Python with no checker
knowing the difference::

    python3 run_regtests.py build/fastwindterrain-py

Every group passes. On top of that, ``regtests/phase9_bindings_parity``
runs five cases -- a flat solve, a hill with anisotropy and O'Brien, the
manufactured solution, and the ascii backend -- through both paths and
compares every output file byte for byte, plus stdout.

What parity does and does not cover
-----------------------------------

It holds **within one build tree**. A wheel compiled with different flags
is a different compilation, and nothing here claims otherwise. The
guarantee is that the two entry points of a given build cannot diverge,
because there is only one compiled solver in it.

Scope
=====

This phase exposes the process lifecycle and a whole run, and no more.
The narrow surface is deliberate: the point is to establish parity and
put it under test before there is a wider API to keep honest. Grid,
field and solver-driver bindings follow, and each one inherits a
guarantee that is already green.
