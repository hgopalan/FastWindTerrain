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

Terrain and profile from Python
===============================

A whole case with no inputs file anywhere:

.. code-block:: python

    import numpy as np, fastwindterrain as fwt

    pts = np.loadtxt("terrain.csv", delimiter=",", skiprows=5)   # (n, 3)

    with fwt.session():                     # no arguments: ParmParse is empty
        g = fwt.Grid({"n_cell": (24, 24, 40), ...})
        t = fwt.Terrain(g, {"points": pts})
        inf = fwt.Inflow(g, t, {"mode": "powerlaw", "u_ref": 8.0,
                                "v_ref": 6.0, "z_ref": 10.0})

        t.mask          # (nz, ny, nx) int32
        inf.velocity    # (3, nz, ny, nx) [m/s]

``userfile`` mode takes the six-column table as two ``(n, 3)`` arrays
instead of a file:

.. code-block:: python

    inf = fwt.Inflow(g, t, {"mode": "userfile",
                            "points": xyz, "velocity": uvw})

One interpolation, two ways to fill it
--------------------------------------

The point cloud is part of ``Terrain::Params`` rather than something
``Build`` fetches. "Read them from a file" and "here they are" fill one
field; they are not two routes through the interpolation.

That is what makes the two **bit-for-bit** identical rather than merely
similar, and a regtest holds it: terrain built from a numpy array and
terrain built from the CSV those numbers came out of give identical
``z_terrain`` and ``mask``. The same for the userfile table and its 3D
inverse-distance interpolation.

The stronger version of that test builds an entire case in Python --
grid, terrain points, profile, with AMReX initialized with **no
arguments at all**, so ParmParse holds nothing -- and requires every
field to equal what the equivalent inputs file produces, exactly.

Ambiguity is refused
--------------------

``points`` and ``file`` together raise, rather than one silently winning:
two sources for one thing is a mistake worth reporting, not a precedence
rule to remember. So do a table given in a non-``userfile`` mode, a
``points`` without its ``velocity``, an ``(n,)`` array where an
``(n, 3)`` was wanted, and an unknown key.

A point cloud that gets silently reshaped is a whole dataset built on the
wrong terrain, which is why none of this is forgiving.

More aborts became exceptions
-----------------------------

Phase 10 converted Grid's. This phase converts Terrain's and Inflow's,
including the file readers: a missing terrain file is a typo in an inputs
deck, not a bug in the solver, so it raises ``ValueError`` in Python
while still aborting the executable through ``main()``'s handler.

Driving the solver
==================

A whole case as one nested dict, with no inputs file anywhere:

.. code-block:: python

    with fwt.session():
        s = fwt.Solver({
            "grid":    {"n_cell": (24, 24, 40), ...},
            "terrain": {"points": pts},
            "inflow":  {"mode": "powerlaw", "u_ref": 8.0, "v_ref": 6.0},
            "poisson": {"alpha_v": 0.5, "n_projections": 4},
        })
        s.run()
        u = s.velocity[0]

Sections are ``grid``, ``terrain``, ``inflow``, ``anisotropy``,
``obrien``, ``poisson`` and ``numerics``. An absent section means **use
the defaults**, never "use whatever ParmParse happens to hold" -- which
is what makes a generation loop safe. An unknown section or key raises.

``Solver::Params`` holds all six module Params in one place, and
``FromParmParse`` builds one from an inputs deck. The two paths meet
immediately afterwards, so the executable exercises the same code the
dict does.

Stepwise
--------

.. list-table::
   :widths: 30 54
   :header-rows: 1

   * - Call
     - Meaning
   * - ``setup()``
     - Build every component and keep ``u0``
   * - ``project_once()``
     - **One** projection pass; returns the MLMG residual
   * - ``solve()``
     - The projection loop, or the manufactured solution
   * - ``diagnose()``
     - The divergence field and the post-solve report
   * - ``write_output()``
     - The report and the field output
   * - ``run()``
     - All four in order

``project_once`` exists so a notebook can watch an approximate
projection converge rather than be told that it did:

.. code-block:: python

    s.setup()
    for _ in range(4):
        s.project_once()
        print(s.max_divergence_fe)     # 0.1245, 0.1131, 0.0895, 0.0704...

A regtest requires four stepwise passes to give a **bit-identical** field
to ``solve()`` with ``n_projections = 4``: they are the same code, and it
holds them to it.

What the solve reports
----------------------

.. list-table::
   :widths: 30 54
   :header-rows: 1

   * - Property
     - Meaning
   * - ``solve_residual``
     - MLMG residual of the last solve
   * - ``solve_iterations``
     - MLMG iterations. A solve that hit ``max_iter`` has not converged,
       whatever its residual says
   * - ``max_divergence_fe``
     - ``max|div(u)|`` in the norm the projection controls -- the number
       that measures whether a pass helped
   * - ``max_divergence``
     - the same with the configured derivative scheme. This one reads the
       velocity's **ghost cells**
   * - ``divergence``
     - the per-cell field, after ``diagnose()``
   * - ``diagnostics``
     - the post-solve dict: ``div_max``, ``div_l2``, the flux balance

The ghost refill, finally testable
----------------------------------

Phase 11 said ``set_velocity`` refills the ghost cells and could not test
it, since ghost cells are deliberately not exposed. A solve can now
follow a write, so it is testable through ``max_divergence`` -- the
scheme divergence, which reads them through a five-point stencil.

The regtest hands solver B the field solver A produced and requires their
scheme divergence to match **exactly**. The valid regions are equal by
construction, so any difference would be the ghosts, and B's would be its
own initial profile rather than the field it was given.

Anisotropy and O'Brien
======================

The weights can be built and looked at on their own, which is what
choosing ``slope_scale`` and ``decay_height`` actually needs:

.. code-block:: python

    a = fwt.Anisotropy(grid, terrain,
                       {"enable": True, "slope_scale": 0.5,
                        "decay_height": 500.0,
                        "alpha_h_base": 1.0, "alpha_v_base": 0.5})
    a.alpha_v          # (nz, ny, nx)
    a.slope_max        # max |grad z_terrain|

Inside a solve, the same settings go in the ``anisotropy`` and ``obrien``
sections, and both report what they did:

.. code-block:: python

    s.anisotropy   # enabled, the four alpha extrema, slope_max
    s.obrien       # enabled, n_columns, max_w_top, max_residual

Where the base weights live
---------------------------

``alpha_h_base``/``alpha_v_base`` are accepted by a **standalone**
``Anisotropy``, which has no Poisson section to take them from. Inside a
``Solver`` configuration they are ``poisson.alpha_h``/``alpha_v``, and
naming them in the ``anisotropy`` section **raises**.

That is deliberate. The base weights and the operator coefficients are
the same numbers -- sigma is ``alpha^2`` and the correction multiplies by
the same alpha -- so a second copy could disagree with the operator it
feeds, and silently letting one override the other would be worse than
refusing.

What the regtest holds
----------------------

Phase 7's assertions, recomputed in numpy rather than read back:

* ``alpha_v`` equals
  ``clamp(alpha_v_base * exp(-slope_3d / slope_scale))`` cell by cell,
  with ``|grad z_terrain|`` computed independently from the same central
  differences -- **agreeing to 5.6e-17 over 105 600 cells**, suppressed
  to 0.1045 from a base of 0.5 on a 0.78 slope
* the suppression **decays monotonically with height above ground**. It
  does not reach base by the domain top, and the check says so: 961 m
  over a 500 m decay height is under two e-foldings, so requiring it to
  have finished would be requiring the wrong thing
* with ``enable`` off, or ``source = "none"``, both weights hold their
  base values everywhere -- the feature cannot change results when it is
  switched off
* ``alpha_h_mode = "slope"`` applies the **same** factor to both weights,
  in every cell rather than merely at their minima
* after the O'Brien adjustment ``w`` is **exactly** zero at the domain
  top, in the reported ``max_w_top`` and in the field itself, having
  removed a 9.7 m/s residual over 1600 columns. Exactness is the point of
  the redistribution, so it is held to round-off rather than a tolerance

Output without a file
=====================

.. code-block:: python

    s.setup(); s.solve(); s.diagnose()

    f = s.fields()            # {name: numpy array}, no file involved
    f["u"], f["divergence"], f["lambda"]

    s.write_plotfile("plt_case")     # the production path, still there
    s.write_ascii("fields.txt")
    s.write_report("report.txt")

``fields()`` returns the **same object** the plotfile and ascii backends
are handed. One gather, three consumers. A dataset generator gets exactly
the array the file would have contained, and a regtest requires all three
to agree value for value -- so a third assembly of "the output fields"
cannot creep in without being caught.

``write_plotfile`` is kept deliberately: results stay viewable in VisIt,
ParaView and yt, which no numpy array replaces.

All three writers require ``diagnose()``, since the divergence component
comes from it. Asking earlier raises and writes nothing, rather than
producing a file with a missing field.

The output section
------------------

``write_output()`` no longer reads ParmParse. Its settings are the
``output`` section of the config:

.. list-table::
   :widths: 24 60
   :header-rows: 1

   * - Key
     - Meaning
   * - ``which``
     - ``report`` | ``fields`` | ``both`` -- which outputs are produced
   * - ``format``
     - ``plt`` | ``ascii`` | ``both`` -- which backend writes the fields
   * - ``report_file``, ``plot_file``, ``ascii_file``
     - where each goes

That was the last module reading the global, and it mattered for the
same reason the others did: a generation loop that let ParmParse decide
would have case 2 write over case 1's plotfile, or write nothing,
depending on what an earlier run had set. A regtest initializes from an
inputs file naming its own report, plotfile and ascii file, then runs a
dict-configured solver and requires none of those three names to appear.

Scope
=====

Phase 9 exposed the process lifecycle and a whole run, Phase 10 added
Grid, Phase 11 the fields, Phase 12 terrain and the profile, Phase 13 the
solver, Phase 14 the anisotropy and O'Brien settings, Phase 15 the
output.

**Nothing in a Python-configured run reads ParmParse any more.** That was
the point of the migration, and it is what makes a loop over a few
hundred cases in one process safe. The narrow surface is deliberate: parity
was established and put under test before there was a wider API to keep
honest, and each new piece inherits a guarantee that is already green.

``Solver`` currently offers ``setup()`` and its fields. The remaining
stages -- ``solve()``, ``diagnose()``, ``write_output()`` -- and
dict-based configuration arrive with the terrain and solver-driver
phases. The ghost refill that follows ``set_velocity`` is not directly
observable yet, since ghost cells are deliberately not exposed; it
becomes testable when a solve can follow a write.
