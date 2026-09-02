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

numpy is a runtime requirement. The build checks for it at configure
time rather than letting the failure surface later as an ImportError
inside a property accessor. On a system Python that refuses installs
(PEP 668), point CMake at a virtual environment::

    python3 -m venv build/venv
    build/venv/bin/pip install numpy
    cmake -S . -B build -DFWT_PYTHON=ON \
        -DPython3_EXECUTABLE=$PWD/build/venv/bin/python

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

Building a Grid from Python
===========================

.. code-block:: python

    with fwt.session():
        g = fwt.Grid({
            "n_cell": (24, 24, 40),
            "prob_lo": (0.0, 0.0, 0.0),
            "prob_hi": (1000.0, 1000.0, 483.19909696997223),
            "dz0": 4.0,
            "stretching_ratio": 1.05,   # optional, default 1.0 (uniform)
            "max_grid_size": 16,        # optional, default 32
        })
        g.z_cc        # numpy, (nz,)
        g.z_face      # numpy, (nz+1,)
        g.prob_hi     # as resolved, after any overshoot adjustment

The dict is the real path
-------------------------

Nothing is written to a temporary inputs file, and nothing is read from
ParmParse.

That matters more than it sounds. **ParmParse is process-global and
persists for the life of an AMReX initialization.** A second case in the
same process inherits every parameter the first one set and the second
did not override. For a command-line run that never comes up -- one
process, one case. For a loop generating a few hundred training samples
it is silent corruption spread across a dataset, with no failure anywhere
to notice it.

So ``Grid::Params`` holds the inputs as data, ParmParse is one *source*
of a Params rather than the mechanism, and the defaults live in one place
both paths use. A regtest initializes from an inputs file setting
``grid.stretching_ratio = 1.03``, then builds a Grid from a dict that
omits it, and requires the default ``1.0``.

An unknown key is refused
-------------------------

.. code-block:: python

    fwt.Grid({..., "stretching_ration": 1.05})
    # ValueError: unknown grid parameter 'stretching_ration'. Valid keys are: ...

ParmParse accepts a misspelling and mentions it once at finalize, as one
line in a list of unused variables. That is exactly how a typo produces
an entire dataset on the wrong grid without a single failure. The dict
path refuses.

Errors raise, warnings warn
---------------------------

``amrex::Abort`` kills the process. That is right for a command-line
tool -- a grid that does not reach the domain top must not quietly
produce a plotfile -- and wrong inside Python, where it takes the
interpreter down and loses whatever else the session was holding.

A bad input now throws ``fwt::InputError``, which becomes a
``ValueError``. The executable catches the same exception in ``main()``
and aborts with the same message, so its diagnostic, its nonzero exit and
the absence of a report file are exactly what they were --
``phase1_grid`` still asserts all three.

A grid that *overshoots* its requested top is not an error: the top moves
to where the grid reaches. That goes through a warning handler whose
default prints exactly what the code always printed, and which the
bindings replace with Python's ``warnings.warn``:

.. code-block:: python

    with warnings.catch_warnings(record=True) as w:
        g = fwt.Grid(over)          # w[0] is a UserWarning

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fwt.Grid(over)              # raises instead

A dataset generator wants that second form: promoting the warning to an
error abandons the case rather than quietly adjusting the domain and
carrying on.

``run()`` is the exception: it restores the stdout handler for its own
duration. It is documented as behaving exactly as the executable does,
and where a warning comes out is part of that -- with the Python handler
in force, an overshoot went to ``warnings.warn`` instead of stdout, and
``phase1_grid`` failed under the shim while passing under the
executable. The Python-native API keeps Python-native warnings.

Scope of the migration
----------------------

Only *user input* moves from abort to exception, and in this phase only
Grid's. An internal invariant -- a box split in z when the decomposition
promised otherwise -- stays an assertion: it is a bug in this code, not
something a caller can provoke, and it should abort loudly wherever it
happens. Later phases convert their own modules.

Fields as numpy
===============

.. code-block:: python

    with fwt.session(["inputs"]):
        s = fwt.Solver()
        s.setup()

        u = s.velocity[0]           # (nz, ny, nx) [m/s]
        mask = s.mask               # (nz, ny, nx) int32
        lam = s.lambda_             # (nz+1, ny+1, nx+1), nodal

        s.set_velocity(new_field)   # writes back, ghosts refilled

.. list-table::
   :widths: 22 22 44
   :header-rows: 1

   * - Field
     - Shape
     - Meaning
   * - ``velocity``
     - ``(3, nz, ny, nx)``
     - After the projection once it has run
   * - ``velocity0``
     - ``(3, nz, ny, nx)``
     - Before any correction
   * - ``sigma``
     - ``(3, nz, ny, nx)``
     - Poisson coefficients, vertical metric included
   * - ``mask``
     - ``(nz, ny, nx)`` int32
     - 1 solid, 0 fluid
   * - ``z_terrain``
     - ``(nz, ny, nx)``
     - Surface height of the column [m]
   * - ``alpha_h``, ``alpha_v``
     - ``(nz, ny, nx)``
     - Cell-local variational weights
   * - ``lambda_``
     - ``(nz+1, ny+1, nx+1)``
     - The nodal potential. ``lambda`` is a Python keyword

Layout: channels-first
----------------------

``arr[c, k, j, i]`` is the value the plotfile holds at cell
``(i, j, k)`` of component ``c``. A regtest compares the two cell by
cell, because a transposed array passes every other check and would
quietly ruin a training set.

Channels-first is not arbitrary. It is **AMReX's own memory order** --
component slowest, ``i`` fastest -- so each component is a contiguous
slab and the gather is a memcpy rather than a transpose. It is also what
PyTorch's ``conv3d`` wants, ``(N, C, D, H, W)``, so a dataset generator
can hand the array straight over.

Copies, not views
-----------------

Every accessor returns a **copy**, and writing into it changes nothing;
use ``set_velocity``.

Three reasons, and they compound. A MultiFab is *N separate*
``FArrayBox`` es -- every case here runs with ``max_grid_size`` below the
domain width, so there is no single contiguous array to point at. The
velocity carries two ghost layers, so even a one-box field is not the
shape a caller expects. And a view would hand Python a pointer that
outlives the ``Solver`` the moment it goes out of scope.

The cost is nothing next to the solve: one field of a 24x24x40 case is
184 KB, against four multigrid solves. Zero-copy starts to matter only
for in-situ GPU coupling, where the right answer is pyAMReX rather than a
hand-rolled view -- a different build shape, and not one this project
needs for offline surrogate training.

A regtest builds the same case at ``max_grid_size`` 8 and 64 -- nine
boxes against one -- and requires the arrays to be identical. That is
what actually validates the gather.

Writing a field back
--------------------

``set_velocity`` takes a ``(3, nz, ny, nx)`` array, writes the valid
region, and **refills the ghost cells** through the boundary conditions.

The refill is why this lives on ``Solver`` rather than being a property
assignment: only the boundary conditions know what the ghosts should
hold, and a field written from Python with stale ghosts would give a
quietly wrong divergence two calls later.

A mismatched shape raises rather than being broadcast or reshaped --
that is how a transposed velocity field gets written without anyone
noticing. A ``float32`` array is accepted, since widening to double is a
conversion rather than a reinterpretation.

Scope
=====

Phase 9 exposed the process lifecycle and a whole run, Phase 10 added
Grid, Phase 11 adds the fields. The narrow surface is deliberate: parity
was established and put under test before there was a wider API to keep
honest, and each new piece inherits a guarantee that is already green.

``Solver`` currently offers ``setup()`` and its fields. The remaining
stages -- ``solve()``, ``diagnose()``, ``write_output()`` -- and
dict-based configuration arrive with the terrain and solver-driver
phases. The ghost refill that follows ``set_velocity`` is not directly
observable yet, since ghost cells are deliberately not exposed; it
becomes testable when a solve can follow a write.
