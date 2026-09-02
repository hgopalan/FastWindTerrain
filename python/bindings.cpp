// pybind11 bindings for FastWindTerrain.
//
// Phase 9 exposes only the process lifecycle and a whole run. That is
// deliberately narrow: the point of this phase is to establish that the
// Python path and the executable produce BIT-IDENTICAL results, and the
// way to establish it is to run the existing regtest suite through here
// and compare the output files byte for byte. A richer API arrives once
// that guarantee is in place and can be kept green.
//
// The module links fwt_core -- the same archive the executable links --
// so the two entry points do not merely share source, they share object
// files. See Source/CMakeLists.txt.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <AMReX.H>
#include <AMReX_Vector.H>
#include <AMReX_Version.H>

#include <string>
#include <vector>

#include "Solver.H"

namespace py = pybind11;

namespace {

// argv storage for amrex::Initialize, which takes (int&, char**&) and
// may rewrite them. It must outlive the call, so it is held here rather
// than in a local.
std::vector<std::string> g_argv_store;
std::vector<char*> g_argv;

// AMReX has no public "am I initialized" that is safe to call before
// initialization, so the state is tracked here. This module is the only
// thing in the process that may call Initialize/Finalize.
bool g_initialized = false;

void BuildArgv (const std::vector<std::string>& args)
{
    g_argv_store.clear();
    g_argv_store.reserve(args.size() + 1);
    // argv[0] is the program name. It is fixed rather than taken from
    // sys.argv[0] so that a Python run and an executable run differ in
    // nothing that reaches the solver.
    g_argv_store.emplace_back("fastwindterrain");
    for (const std::string& a : args) { g_argv_store.push_back(a); }

    g_argv.clear();
    g_argv.reserve(g_argv_store.size() + 1);
    for (std::string& s : g_argv_store) { g_argv.push_back(s.data()); }
    g_argv.push_back(nullptr);
}

void Initialize (const std::vector<std::string>& args)
{
    if (g_initialized) {
        throw std::runtime_error(
            "AMReX is already initialized. amrex::Initialize is "
            "process-global and must not be called twice; call "
            "fastwindterrain.finalize() first.");
    }
    BuildArgv(args);
    int argc = int(g_argv.size()) - 1;
    char** argv = g_argv.data();
    amrex::Initialize(argc, argv);
    g_initialized = true;
}

void Finalize ()
{
    if (!g_initialized) {
        throw std::runtime_error(
            "AMReX is not initialized, so there is nothing to finalize.");
    }
    amrex::Finalize();
    g_initialized = false;
}

// A whole run, exactly as the executable performs it.
//
// This owns the AMReX lifecycle for the duration, which is why it
// refuses to run inside an existing one: amrex::Initialize is what
// parses the inputs file into ParmParse, so a second run inside the same
// initialization would inherit the first run's settings. Driving many
// cases in one process is a later phase, and it needs the parameters to
// arrive as data rather than through that global.
void Run (const std::vector<std::string>& args)
{
    if (g_initialized) {
        throw std::runtime_error(
            "run() manages the AMReX lifecycle itself and cannot be "
            "called while AMReX is initialized. Call finalize() first.");
    }

    Initialize(args);
    {
        amrex::Vector<std::string> solver_args;
        solver_args.reserve(args.size());
        for (const std::string& a : args) { solver_args.push_back(a); }

        fwt::Solver solver;
        solver.Run(solver_args);
    }
    Finalize();
}

} // namespace

PYBIND11_MODULE(_fastwindterrain, m)
{
    m.doc() = "FastWindTerrain -- mass-consistent wind solver (bindings)";

    m.attr("__version__") = FWT_VERSION;

    m.def("amrex_version", [] () { return std::string(amrex::Version()); },
          "The AMReX version this module was built against.");

    m.def("is_initialized", [] () { return g_initialized; },
          "Whether amrex::Initialize has been called from this module.");

    m.def("initialize", &Initialize, py::arg("args") = std::vector<std::string>{},
          "Initialize AMReX. args are the command-line arguments AFTER the\n"
          "program name -- typically an inputs file followed by any\n"
          "name=value overrides. Raises RuntimeError if already\n"
          "initialized: amrex::Initialize is process-global.");

    m.def("finalize", &Finalize,
          "Finalize AMReX. Raises RuntimeError if not initialized.");

    m.def("run", &Run, py::arg("args"),
          "Run one case exactly as the executable does: initialize AMReX,\n"
          "run the full pipeline, finalize. args are the arguments AFTER\n"
          "the program name.\n\n"
          "Raises RuntimeError if AMReX is already initialized, since this\n"
          "function owns the lifecycle for the duration of the call.");
}
