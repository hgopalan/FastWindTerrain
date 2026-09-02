# Python bindings.
#
# The module links fwt_core, the same archive the executable links, so
# the two entry points share object files rather than merely source.
# Compiling the solver sources a second time here would be the usual
# shortcut and would quietly forfeit that.

# Development.Module, NOT Development.Embed: the latter requires
# libpython, which Homebrew, conda and pyenv builds often do not ship,
# and an extension module never needs it.
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# numpy is a RUNTIME requirement, not a build one: pybind11 loads its C
# API dynamically, so nothing here needs its headers. Checking at
# configure time anyway turns "ImportError deep inside a property
# accessor" into a message at the point the choice was made.
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.__version__)"
    RESULT_VARIABLE FWT_NUMPY_MISSING
    OUTPUT_VARIABLE FWT_NUMPY_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if(FWT_NUMPY_MISSING)
    message(FATAL_ERROR
        "The Python bindings return numpy arrays, and ${Python3_EXECUTABLE} "
        "cannot import numpy.\n"
        "Install it for that interpreter, or point CMake at one that has "
        "it:\n"
        "  cmake -S . -B build -DFWT_PYTHON=ON "
        "-DPython3_EXECUTABLE=/path/to/python\n"
        "A virtual environment is the usual answer on a system Python that "
        "refuses installs (PEP 668).")
endif()

# pybind11 comes from the submodule, matching how AMReX is vendored, so a
# recursive clone needs no network at configure time. An installed
# pybind11 is used if one is already visible.
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    if(NOT EXISTS ${CMAKE_SOURCE_DIR}/external/pybind11/CMakeLists.txt)
        message(FATAL_ERROR
            "The pybind11 submodule is empty. Run:\n"
            "  git submodule update --init --recursive\n"
            "or install pybind11 so find_package can see it.")
    endif()
    add_subdirectory(${CMAKE_SOURCE_DIR}/external/pybind11
                     ${CMAKE_BINARY_DIR}/external/pybind11)
endif()

pybind11_add_module(_fastwindterrain MODULE bindings.cpp)
target_link_libraries(_fastwindterrain PRIVATE fwt_core)
target_compile_definitions(_fastwindterrain
    PRIVATE FWT_VERSION="${PROJECT_VERSION}")

# Under a CUDA build the bindings have to be compiled as CUDA too, since
# fwt_core's headers pull in AMReX device code.
if(FWT_GPU_BACKEND STREQUAL "CUDA")
    set_source_files_properties(bindings.cpp PROPERTIES LANGUAGE CUDA)
endif()

# Lay the package out under <build>/python so that
# PYTHONPATH=<build>/python is all an import needs.
set(FWT_PY_DIR ${CMAKE_BINARY_DIR}/python/fastwindterrain)
set_target_properties(_fastwindterrain PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${FWT_PY_DIR})

# Globbed with CONFIGURE_DEPENDS rather than listed: a new module added
# to the package and forgotten here would import from a wheel and not
# from the build tree, so the two paths would disagree about what the
# package contains -- exactly the kind of split these bindings exist to
# avoid.
file(GLOB FWT_PY_FILES CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/fastwindterrain/*.py)
foreach(f IN LISTS FWT_PY_FILES)
    get_filename_component(name ${f} NAME)
    configure_file(${f} ${CMAKE_BINARY_DIR}/python/fastwindterrain/${name}
                   COPYONLY)
endforeach()

# An executable shim that is argv-compatible with the solver binary, so
# the existing regtest driver can run the WHOLE suite through Python
# without a single checker knowing the difference:
#
#     python3 run_regtests.py build/fastwindterrain-py
#
# That is the parity guarantee this phase exists to establish, and it is
# worth a four-line script to get it for free.
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fastwindterrain-py.in
               ${CMAKE_BINARY_DIR}/fastwindterrain-py
               FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                                GROUP_READ GROUP_EXECUTE
                                WORLD_READ WORLD_EXECUTE
               @ONLY)

# --- wheel install ---------------------------------------------------------
# Under scikit-build-core (pip install .) the module is installed next to
# the pure-Python package, which pyproject.toml supplies via
# wheel.packages. SKBUILD is defined only by that backend, so an ordinary
# `cmake --install` of this project still installs nothing -- there is no
# C++ install target here and adding one by accident would be worse than
# having none.
if(DEFINED SKBUILD)
    install(TARGETS _fastwindterrain LIBRARY DESTINATION fastwindterrain)
endif()

message(STATUS "Python bindings enabled")
message(STATUS "  python:  ${Python3_EXECUTABLE} (numpy ${FWT_NUMPY_VERSION})")
message(STATUS "  module:  ${FWT_PY_DIR}")
message(STATUS "  shim:    ${CMAKE_BINARY_DIR}/fastwindterrain-py")
message(STATUS "  use as:  PYTHONPATH=${CMAKE_BINARY_DIR}/python python3 -c "
               "'import fastwindterrain'")
