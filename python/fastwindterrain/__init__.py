"""
FastWindTerrain -- mass-consistent wind solver.

Phase 9 exposes the process lifecycle and a whole run. The narrow surface
is deliberate: this phase exists to establish that the Python path and
the executable produce bit-identical results, which is checked by running
the entire regtest suite through the bindings and comparing output files
byte for byte. A richer API follows once that guarantee is in place.

The module links the same compiled library as the executable, so the two
share object files rather than merely source.

    import fastwindterrain as fwt

    fwt.run(["inputs", "poisson.alpha_v=0.3"])   # a whole case

or, equivalently, from a shell::

    python -m fastwindterrain inputs poisson.alpha_v=0.3

Components can also be built directly from Python, with no inputs file
anywhere::

    with fwt.session():
        g = fwt.Grid({"n_cell": (24, 24, 40),
                      "prob_lo": (0.0, 0.0, 0.0),
                      "prob_hi": (1000.0, 1000.0, 483.19909696997223),
                      "dz0": 4.0, "stretching_ratio": 1.05})
        print(g.z_cc)          # numpy, (nz,)

Many cases in one process is what :mod:`fastwindterrain.dataset` is
for -- a fixed grid, everything else swept, one ``.npz`` out::

    from fastwindterrain import dataset

    configs = dataset.sweep(base, {"inflow.u_ref": [4.0, 8.0, 12.0]})
    with fwt.session():
        dataset.generate(configs, "wind.npz", fields=["u", "v", "w", "mask"])

That path never touches ParmParse, which matters when driving many cases
in one process: ParmParse persists for the life of an AMReX
initialization, so a case that omits a parameter would otherwise inherit
whatever an earlier case set.

AMReX's Initialize/Finalize are process-global, so ``initialize()`` and
``finalize()`` are explicit rather than implicit on import, and calling
either out of order raises rather than crashing the interpreter. Use the
``session()`` context manager when driving AMReX by hand.
"""

from contextlib import contextmanager

from ._fastwindterrain import (      # noqa: F401
    Anisotropy,
    Grid,
    Inflow,
    Solver,
    Terrain,
    __version__,
    amrex_version,
    finalize,
    initialize,
    is_initialized,
    run,
)

__all__ = [
    "Anisotropy",
    "Grid",
    "Inflow",
    "Solver",
    "Terrain",
    "__version__",
    "amrex_version",
    "finalize",
    "initialize",
    "is_initialized",
    "dataset",
    "baseline",
    "evaluate",
    "levels",
    "training",
    "run",
    "session",
]


@contextmanager
def session(args=None):
    """Initialize AMReX for the duration of the block, then finalize.

    AMReX's lifecycle is process-global and finalizing is not optional --
    skipping it on an exception leaves the process unable to initialize
    again. This makes that hard to get wrong::

        with fwt.session(["inputs"]):
            ...
    """
    initialize(list(args) if args is not None else [])
    try:
        yield
    finally:
        finalize()


# Imported last: dataset reads __version__ and session out of this module,
# so the names it needs have to exist before it is loaded.
from . import dataset          # noqa: E402,F401
from . import levels           # noqa: E402,F401
from . import baseline         # noqa: E402,F401
from . import evaluate         # noqa: E402,F401
from . import training         # noqa: E402,F401
