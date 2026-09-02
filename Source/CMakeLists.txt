# The solver is a library, and both entry points link it.
#
# This is what makes the C++ and Python paths bit-for-bit identical: they
# do not merely run the same source, they run the same OBJECT FILES.
# Compiling the sources a second time into the Python module -- the usual
# shortcut, and what massconsistent_amr does -- would leave two
# compilations that could differ in flags, and floating-point results can
# differ in the last bits when they do. There is nothing to keep in step
# here because there is only one compilation.
add_library(fwt_core STATIC
    Grid.cpp
    Derivatives.cpp
    Terrain.cpp
    Inflow.cpp
    BoundaryConditions.cpp
    Anisotropy.cpp
    Obrien.cpp
    Poisson.cpp
    Output.cpp
    Diagnostics.cpp
    Verify.cpp
    Solver.cpp
)

# PUBLIC: anything linking fwt_core needs the headers and AMReX too.
target_include_directories(fwt_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(fwt_core PUBLIC AMReX::amrex_3d)

# A static archive can only be linked into a shared module if its objects
# are position independent. The top-level build sets
# CMAKE_POSITION_INDEPENDENT_CODE globally so AMReX is built the same
# way; this is belt and braces for the target that actually needs it.
set_target_properties(fwt_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# The standalone driver: AMReX's lifecycle, and nothing else.
add_executable(fastwindterrain main.cpp)
target_link_libraries(fastwindterrain PRIVATE fwt_core)
