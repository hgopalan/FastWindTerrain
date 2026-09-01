# ---------------------------------------------------------------------------
# FastWindTerrain -- top-level CMake build
#
# Quick start (AMReX comes from the bundled submodule):
#
#   git submodule update --init --recursive
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j 8
#   ctest --test-dir build --output-on-failure
#
# To build against an AMReX you already have installed instead:
#
#   cmake -S . -B build -DFWT_USE_INTERNAL_AMREX=OFF -DAMReX_ROOT=/path/to/amrex/install
#
# The GNUmake build (see GNUmakefile) remains supported and is kept
# configured the same way -- 3D, double precision, Src/Base only.
# ---------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.25)   # AMReX 26.08 requires >= 3.25

project(FastWindTerrain
        VERSION 0.1.0
        DESCRIPTION "Mass-consistent wind solver on a Cartesian AMReX mesh"
        LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Keep the executable in <build>/ rather than <build>/Source/ so the
# regtest driver has a predictable path.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

option(FWT_MPI                "Build with MPI"                        OFF)
option(FWT_OMP                "Build with OpenMP"                     OFF)
option(FWT_USE_INTERNAL_AMREX "Use the AMReX git submodule"           ON)
option(FWT_ENABLE_TESTS       "Register the regtests with CTest"      ON)

# --- AMReX ----------------------------------------------------------------
if(FWT_USE_INTERNAL_AMREX)
    if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/amrex/CMakeLists.txt)
        message(FATAL_ERROR
            "The AMReX submodule is empty. Run:\n"
            "  git submodule update --init --recursive\n"
            "or configure with -DFWT_USE_INTERNAL_AMREX=OFF to use an "
            "external AMReX.")
    endif()

    # Normal (non-cache) variables set before add_subdirectory take
    # precedence over AMReX's own option()/set(CACHE) defaults
    # (CMP0077/CMP0126 NEW, guaranteed by AMReX's cmake_minimum_required).
    set(AMReX_SPACEDIM       3)
    set(AMReX_PRECISION      DOUBLE)
    set(AMReX_MPI            ${FWT_MPI})
    set(AMReX_OMP            ${FWT_OMP})
    set(AMReX_FORTRAN        OFF)
    set(AMReX_INSTALL        OFF)
    # The variational Poisson solve uses MLMG's nodal Laplacian.
    set(AMReX_LINEAR_SOLVERS ON)
    set(AMReX_PARTICLES      OFF)
    set(AMReX_AMRLEVEL       OFF)
    set(AMReX_EB             OFF)
    set(AMReX_FFT            OFF)
    set(AMReX_PLOTFILE_TOOLS OFF)

    add_subdirectory(external/amrex)
else()
    find_package(AMReX 26.08 REQUIRED 3D DOUBLE)
endif()

add_subdirectory(Source)

# --- Tests ----------------------------------------------------------------
# Thin wrapper over run_regtests.py; the Python checkers stay the single
# source of truth so `python3 run_regtests.py <exe>` and ctest agree.
if(FWT_ENABLE_TESTS)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    enable_testing()

    # A regtest directory is one holding a check.py; the names follow no
    # required pattern.
    file(GLOB FWT_PHASE_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/regtests
         ${CMAKE_CURRENT_SOURCE_DIR}/regtests/*)
    foreach(phase IN LISTS FWT_PHASE_DIRS)
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/regtests/${phase}/check.py)
            add_test(NAME ${phase}
                     COMMAND ${Python3_EXECUTABLE}
                             ${CMAKE_CURRENT_SOURCE_DIR}/run_regtests.py
                             $<TARGET_FILE:fastwindterrain>
                             ${phase})
        endif()
    endforeach()
endif()
