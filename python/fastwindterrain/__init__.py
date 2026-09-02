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

AMReX's Initialize/Finalize are process-global, so ``initialize()`` and
``finalize()`` are explicit rather than implicit on import, and calling
either out of order raises rather than crashing the interpreter. Use the
``session()`` context manager when driving AMReX by hand.
"""

from contextlib import contextmanager

from ._fastwindterrain import (      # noqa: F401
    __version__,
    amrex_version,
    finalize,
    initialize,
    is_initialized,
    run,
)

__all__ = [
    "__version__",
    "amrex_version",
    "finalize",
    "initialize",
    "is_initialized",
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
