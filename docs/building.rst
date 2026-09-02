========
Building
========

AMReX is bundled as a git submodule at ``external/amrex``, pinned to
release ``26.08``, so a fresh clone needs::

    git submodule update --init --recursive

Two build systems are supported and kept configured the same way: 3D,
double precision, ``Src/Base`` only, MPI and OpenMP off by default.

CMake (recommended)
===================

::

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j 8
    ctest --test-dir build --output-on-failure

This produces ``build/fastwindterrain``.

``Source/`` builds as one library, ``fwt_core``, which the executable
links and -- when ``FWT_PYTHON=ON`` -- the Python extension module links
too. That is what makes the two entry points bit-for-bit identical: they
share object files rather than merely source. See :doc:`python`.

``CMAKE_POSITION_INDEPENDENT_CODE`` is on for the whole build, including
AMReX, since a static archive cannot otherwise be linked into a shared
module. It is on even in a C++-only build, so that turning the bindings
on does not change the object code of the solver.

.. list-table::
   :widths: 30 10 45
   :header-rows: 1

   * - Option
     - Default
     - Meaning
   * - ``FWT_MPI``
     - ``OFF``
     - Build with MPI
   * - ``FWT_OMP``
     - ``OFF``
     - Build with OpenMP
   * - ``FWT_USE_INTERNAL_AMREX``
     - ``ON``
     - Use the submodule; ``OFF`` uses ``find_package(AMReX)``
   * - ``FWT_ENABLE_TESTS``
     - ``ON``
     - Register the regtests with CTest
   * - ``FWT_PYTHON``
     - ``OFF``
     - Build the pybind11 bindings; see :doc:`python`

To build against an AMReX you already have installed::

    cmake -S . -B build -DFWT_USE_INTERNAL_AMREX=OFF \
        -DAMReX_ROOT=/path/to/amrex/install

GNUmake (AMReX native)
======================

::

    make -j8

This produces ``main3d.gnu.ex``. ``AMREX_HOME`` defaults to the
submodule; override it to build against a different checkout::

    make AMREX_HOME=/path/to/amrex

MPI
===

MPI is off by default. To build with it::

    cmake -S . -B build-mpi -DFWT_MPI=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build build-mpi -j 8

and then run the suite against that executable::

    python3 run_regtests.py build-mpi/fastwindterrain

**MPI is exercised by exactly one regtest group.** Everything else in
the suite runs on a single rank whichever executable it is pointed at,
so ``mpi_parity`` is the only place a parallel build is tested at all.
It runs three cases serially and again under ``mpirun -n 2``, compares
every report entry between the two, and puts the parallel runs under a
wall-clock timeout. Every other group SKIPS nothing and notices nothing
about the rank count.

Two consequences worth being deliberate about:

* the coverage that group has is the coverage MPI has. A parallel bug in
  a code path none of its three cases touch will not be caught here
* a hang is a first-class failure mode in parallel and cannot be caught
  by checking output, since there is none. That is why the timeout is
  part of the test rather than left to the CI job's own limit

The group SKIPS on a serial build, so running the suite the usual way
costs nothing and says nothing about MPI. CI builds one job with
``FWT_MPI=ON`` and runs it -- unlike the GPU jobs, which are
compile-only because a hosted runner has no GPU, a hosted runner can
actually run two ranks.

Linear solvers
==============

``AMReX_LINEAR_SOLVERS`` is currently off in both build systems, since
only ``Src/Base`` is needed so far. It must be turned on together with
the variational Poisson solve.
