# FastWindTerrain -- GNUmakefile
#
# AMReX comes from the bundled git submodule by default:
#   git submodule update --init --recursive
# Override to build against a different checkout:
#   make AMREX_HOME=/path/to/amrex
#
# See CMakeLists.txt for the CMake build, which is kept configured the
# same way (3D, double precision, Src/Base only).

AMREX_HOME ?= $(shell pwd)/external/amrex

DEBUG        = FALSE
DIM          = 3
COMP         = gnu
USE_MPI      = FALSE
USE_OMP      = FALSE
TINY_PROFILE = FALSE

Bpack := ./Make.package
Blocs := .

include $(AMREX_HOME)/Tools/GNUMake/Make.defs
include $(Bpack)
include $(AMREX_HOME)/Src/Base/Make.package
include $(AMREX_HOME)/Tools/GNUMake/Make.rules
