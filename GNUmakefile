# FastWindTerrain -- GNUmakefile
#
# Expects AMREX_HOME to point at a built/checked-out AMReX source tree,
# e.g.:  export AMREX_HOME=/path/to/amrex
# (or add it as a git submodule under external/amrex and set the path below)

AMREX_HOME ?= ../amrex

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
