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

Linear solvers
==============

``AMReX_LINEAR_SOLVERS`` is currently off in both build systems, since
only ``Src/Base`` is needed so far. It must be turned on together with
the variational Poisson solve.
