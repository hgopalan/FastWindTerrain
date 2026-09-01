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

Linear solvers
==============

``AMReX_LINEAR_SOLVERS`` is currently off in both build systems, since
only ``Src/Base`` is needed so far. It must be turned on together with
the variational Poisson solve.
