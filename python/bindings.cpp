// pybind11 bindings for FastWindTerrain.
//
// Phase 9 exposed the process lifecycle and a whole run, establishing
// that the Python path and the executable produce BIT-IDENTICAL results
// -- checked by running the existing regtest suite through here and
// comparing the output files byte for byte.
//
// Phase 10 adds Grid, built from a Python dict rather than from an
// inputs file. The dict is the real path, not a temporary file written
// to disk: ParmParse is process-global and persists for the life of an
// AMReX initialization, so a second case in one process inherits every
// parameter the first set and the second did not override. That is
// silent corruption spread across a dataset, and it is the reason this
// is done properly rather than conveniently.
//
// The module links fwt_core -- the same archive the executable links --
// so the two entry points do not merely share source, they share object
// files. See Source/CMakeLists.txt.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <AMReX.H>
#include <AMReX_Vector.H>
#include <AMReX_Version.H>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "Error.H"
#include "FieldIO.H"
#include "Grid.H"
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
// Restores the C++ default warning handler for a scope, and puts the
// previous one back afterwards -- including on the exception path.
//
// run() is documented as behaving EXACTLY as the executable does, and
// that has to include where a warning comes out. With the Python handler
// installed, an overshoot went to warnings.warn instead of stdout, and
// the phase1_grid regtest -- which greps stdout for it -- failed under
// the shim while passing under the executable. Parity is the point of
// this module; a warning going somewhere else is a difference.
//
// The Python-native API keeps the Python handler. It is only this
// deliberately executable-shaped entry point that reverts.
class DefaultWarningsForScope
{
public:
    DefaultWarningsForScope ()
        : m_previous(fwt::SetWarningHandler(fwt::WarningHandler{})) {}
    ~DefaultWarningsForScope () { fwt::SetWarningHandler(m_previous); }
    DefaultWarningsForScope (const DefaultWarningsForScope&) = delete;
    DefaultWarningsForScope& operator= (const DefaultWarningsForScope&) = delete;
private:
    fwt::WarningHandler m_previous;
};

void Run (const std::vector<std::string>& args)
{
    if (g_initialized) {
        throw std::runtime_error(
            "run() manages the AMReX lifecycle itself and cannot be "
            "called while AMReX is initialized. Call finalize() first.");
    }

    DefaultWarningsForScope stdout_warnings;

    Initialize(args);
    try {
        amrex::Vector<std::string> solver_args;
        solver_args.reserve(args.size());
        for (const std::string& a : args) { solver_args.push_back(a); }

        fwt::Solver solver;
        solver.Run(solver_args);
    }
    catch (...) {
        // AMReX must come down even if the run threw, or the process is
        // left initialized and nothing else can run.
        Finalize();
        throw;
    }
    Finalize();
}

// -----------------------------------------------------------------------
// Grid, from a Python dict
// -----------------------------------------------------------------------

// Every key Grid understands. An unknown key is an ERROR rather than
// something quietly ignored.
//
// ParmParse ignores a misspelling and only mentions it at finalize, as
// one line in a list of unused variables. In a loop generating several
// hundred cases that is how a typo -- "stretching_ration" -- produces an
// entire dataset on the wrong grid without a single failure. The dict
// path refuses instead.
const char* const kGridKeys[] = {
    "n_cell", "prob_lo", "prob_hi", "dz0", "stretching_ratio",
    "max_grid_size",
};

template <typename T>
amrex::Vector<T> AsTriple (const py::handle& h, const char* key)
{
    if (!py::isinstance<py::sequence>(h) || py::isinstance<py::str>(h)) {
        throw fwt::InputError(std::string("grid.") + key +
                              " must be a sequence of 3 numbers");
    }
    auto seq = py::reinterpret_borrow<py::sequence>(h);
    if (py::len(seq) != 3) {
        throw fwt::InputError(std::string("grid.") + key +
                              " must have exactly 3 entries, got " +
                              std::to_string(py::len(seq)));
    }
    amrex::Vector<T> out;
    out.reserve(3);
    for (int d = 0; d < 3; ++d) {
        try {
            out.push_back(py::cast<T>(seq[d]));
        } catch (const py::cast_error&) {
            throw fwt::InputError(std::string("grid.") + key + "[" +
                                  std::to_string(d) + "] is not a number");
        }
    }
    return out;
}

fwt::Grid::Params ParamsFromDict (const py::dict& d)
{
    // Unknown keys first, so a typo is reported before the missing
    // required key that the typo probably caused.
    for (auto item : d) {
        const std::string key = py::cast<std::string>(py::str(item.first));
        bool known = false;
        for (const char* k : kGridKeys) {
            if (key == k) { known = true; break; }
        }
        if (!known) {
            std::string msg = "unknown grid parameter '" + key +
                              "'. Valid keys are:";
            for (const char* k : kGridKeys) { msg += std::string(" ") + k; }
            throw fwt::InputError(msg);
        }
    }

    for (const char* required : {"n_cell", "prob_lo", "prob_hi", "dz0"}) {
        if (!d.contains(required)) {
            throw fwt::InputError(std::string("grid.") + required +
                                  " is required");
        }
    }

    fwt::Grid::Params p;
    p.n_cell  = AsTriple<int>(d["n_cell"], "n_cell");
    p.prob_lo = AsTriple<amrex::Real>(d["prob_lo"], "prob_lo");
    p.prob_hi = AsTriple<amrex::Real>(d["prob_hi"], "prob_hi");

    try {
        p.dz0 = py::cast<amrex::Real>(d["dz0"]);
    } catch (const py::cast_error&) {
        throw fwt::InputError("grid.dz0 is not a number");
    }

    if (d.contains("stretching_ratio")) {
        try {
            p.stretching_ratio = py::cast<amrex::Real>(d["stretching_ratio"]);
        } catch (const py::cast_error&) {
            throw fwt::InputError("grid.stretching_ratio is not a number");
        }
        p.given_stretching_ratio = true;
    }
    if (d.contains("max_grid_size")) {
        try {
            p.max_grid_size = py::cast<int>(d["max_grid_size"]);
        } catch (const py::cast_error&) {
            throw fwt::InputError("grid.max_grid_size is not an integer");
        }
        p.given_max_grid_size = true;
    }

    p.Validate();
    return p;
}

// A numpy COPY, not a view. These are nz+1 doubles, so the copy costs
// nothing, and a view would hand Python a pointer into a Grid that can
// outlive it. Phase 11 is where zero-copy matters and where lifetime
// gets the thought it deserves.
py::array_t<amrex::Real> AsNumpy (const amrex::Vector<amrex::Real>& v)
{
    py::array_t<amrex::Real> out(py::ssize_t(v.size()));
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

// A Grid needs AMReX up, and the failure if it is not is a crash deep
// inside AMReX rather than anything a caller could read.
void RequireInitialized (const char* what)
{
    if (!g_initialized) {
        throw std::runtime_error(
            std::string(what) + " requires AMReX to be initialized. "
            "Use `with fastwindterrain.session():` or call initialize().");
    }
}

// Bridges the solver's warnings to Python's warnings machinery, so a
// notebook can filter them, promote them to errors, or capture them --
// instead of a line of text on stdout it will never look at.
//
// The GIL is held here: everything reaching this point is a synchronous
// call from Python. Acquiring it explicitly keeps that true if a later
// phase releases it around a solve.
void PythonWarningHandler (const std::string& message)
{
    py::gil_scoped_acquire gil;
    // Trailing newlines are for a terminal; a warning does not want them.
    std::string text = message;
    while (!text.empty() && text.back() == '\n') { text.pop_back(); }

    // A filter set to "error" turns the warning into a raised exception,
    // and PyErr_WarnEx reports that by returning -1 with the exception
    // already set. Ignoring it lets C++ carry on with a pending Python
    // error, which surfaces later as an incomprehensible SystemError --
    // so it is turned into a C++ exception here and unwinds normally.
    //
    // That is the behaviour a dataset generator wants: promoting the
    // overshoot warning to an error should abandon the case, not adjust
    // the domain and continue.
    if (PyErr_WarnEx(PyExc_UserWarning, text.c_str(), 1) < 0) {
        throw py::error_already_set();
    }
}

// -----------------------------------------------------------------------
// Fields, as numpy
// -----------------------------------------------------------------------

// Shape for a field: (ncomp, nz, ny, nx), with the leading axis dropped
// for a single-component field. The counts come from the MultiFab's own
// index space, so a nodal field is (nz+1, ny+1, nx+1) with no special
// case.
//
// Channels-first is deliberate. It is AMReX's own memory order, so the
// gather is a memcpy per component rather than a transpose -- and it is
// what PyTorch's conv3d wants, (N, C, D, H, W), so a dataset generator
// can hand the array straight over.
template <typename MF>
std::vector<py::ssize_t> FieldShape (const MF& mf)
{
    const amrex::IntVect len = mf.boxArray().minimalBox().length();
    std::vector<py::ssize_t> shape;
    if (mf.nComp() > 1) { shape.push_back(mf.nComp()); }
    shape.push_back(len[2]);
    shape.push_back(len[1]);
    shape.push_back(len[0]);
    return shape;
}

// A COPY, not a view. See Source/FieldIO.H for why: a MultiFab is N
// separate boxes, the velocity carries ghosts, and a view would outlive
// the Solver that owns it.
py::array FieldToNumpy (const amrex::MultiFab& mf)
{
    const amrex::Vector<amrex::Real> buf = fwt::GatherField(mf);
    py::array_t<amrex::Real> out(FieldShape(mf));
    std::copy(buf.begin(), buf.end(), out.mutable_data());
    return out;
}

py::array IFieldToNumpy (const amrex::iMultiFab& mf)
{
    const amrex::Vector<int> buf = fwt::GatherField(mf);
    py::array_t<int> out(FieldShape(mf));
    std::copy(buf.begin(), buf.end(), out.mutable_data());
    return out;
}

// numpy -> flat buffer, with the shape checked against the field it is
// destined for. Silently reshaping or broadcasting here would be a good
// way to write a transposed velocity field and never find out.
amrex::Vector<amrex::Real> NumpyToBuffer (const py::array& a,
                                          const amrex::MultiFab& mf,
                                          const char* what)
{
    auto arr = py::array_t<amrex::Real, py::array::c_style |
                                        py::array::forcecast>::ensure(a);
    if (!arr) {
        throw fwt::InputError(std::string(what) +
                              " must be an array of numbers");
    }

    const std::vector<py::ssize_t> want = FieldShape(mf);
    std::string exp;
    for (std::size_t k = 0; k < want.size(); ++k) {
        exp += (k ? ", " : "") + std::to_string(want[k]);
    }

    bool ok = (std::size_t(arr.ndim()) == want.size());
    if (ok) {
        for (std::size_t d = 0; d < want.size(); ++d) {
            if (arr.shape(py::ssize_t(d)) != want[d]) { ok = false; break; }
        }
    }
    if (!ok) {
        std::string got;
        for (py::ssize_t k = 0; k < arr.ndim(); ++k) {
            got += (k ? ", " : "") + std::to_string(arr.shape(k));
        }
        throw fwt::InputError(std::string(what) + " has shape (" + got +
                              "), expected (" + exp + ")");
    }

    amrex::Vector<amrex::Real> buf(arr.size());
    std::copy(arr.data(), arr.data() + arr.size(), buf.begin());
    return buf;
}

void RequireSetup (const fwt::Solver& s, const char* what)
{
    if (!s.is_setup()) {
        throw std::runtime_error(
            std::string(what) + " is not available until setup() has run.");
    }
}

} // namespace

PYBIND11_MODULE(_fastwindterrain, m)
{
    m.doc() = "FastWindTerrain -- mass-consistent wind solver (bindings)";

    m.attr("__version__") = FWT_VERSION;

    // A bad input raises ValueError rather than taking the interpreter
    // down. The executable still aborts: main() catches the same
    // exception and turns it back into an abort.
    py::register_exception_translator([] (std::exception_ptr p) {
        try {
            if (p) { std::rethrow_exception(p); }
        } catch (const fwt::InputError& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
        }
    });

    fwt::SetWarningHandler(PythonWarningHandler);

    m.def("amrex_version", [] () { return std::string(amrex::Version()); },
          "The AMReX version this module was built against.");

    m.def("is_initialized", [] () { return g_initialized; },
          "Whether amrex::Initialize has been called from this module.");

    m.def("initialize", &Initialize,
          py::arg("args") = std::vector<std::string>{},
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

    py::class_<fwt::Grid>(m, "Grid", R"doc(
A Cartesian grid: uniform in x and y, optionally stretched in z.

Built from a dict mirroring the ``grid.*`` inputs::

    g = fwt.Grid({
        "n_cell": (24, 24, 40),
        "prob_lo": (0.0, 0.0, 0.0),
        "prob_hi": (1000.0, 1000.0, 483.19909696997223),
        "dz0": 4.0,
        "stretching_ratio": 1.05,   # optional, default 1.0 (uniform)
        "max_grid_size": 16,        # optional, default 32
    })

The dict is the real configuration path: nothing is written to a
temporary inputs file and nothing is read from ParmParse, so a grid built
here cannot inherit a parameter left behind by an earlier case in the
same process.

An unknown key raises ValueError rather than being ignored -- ParmParse
would accept the typo and let a whole dataset be generated on the wrong
grid. So does an out-of-range value, and a stretched grid that fails to
reach ``prob_hi[2]``.

A grid that OVERSHOOTS ``prob_hi[2]`` is not an error: the top is moved
to where the grid actually reaches, and a ``UserWarning`` says so, so
``warnings.catch_warnings`` can capture it or ``simplefilter("error")``
can promote it.

Requires AMReX to be initialized -- use ``fwt.session()``.
)doc")
        .def(py::init([] (const py::dict& d) {
                 RequireInitialized("Grid");
                 auto g = std::make_unique<fwt::Grid>();
                 g->Build(ParamsFromDict(d));
                 return g;
             }), py::arg("params"))
        .def_property_readonly("nx", &fwt::Grid::nx)
        .def_property_readonly("ny", &fwt::Grid::ny)
        .def_property_readonly("nz", &fwt::Grid::nz)
        .def_property_readonly("n_cell", [] (const fwt::Grid& g) {
                 return py::make_tuple(g.nx(), g.ny(), g.nz());
             })
        .def_property_readonly("dz0", &fwt::Grid::dz0)
        .def_property_readonly("stretching_ratio",
                               &fwt::Grid::stretching_ratio)
        .def_property_readonly("max_grid_size", &fwt::Grid::max_grid_size)
        .def_property_readonly("prob_lo", [] (const fwt::Grid& g) {
                 const auto& v = g.prob_lo();
                 return py::make_tuple(v[0], v[1], v[2]);
             })
        .def_property_readonly("prob_hi", [] (const fwt::Grid& g) {
                 // As RESOLVED: prob_hi[2] reflects any overshoot
                 // adjustment, so it is the domain the grid spans rather
                 // than the one that was asked for.
                 const auto& v = g.prob_hi();
                 return py::make_tuple(v[0], v[1], v[2]);
             })
        .def_property_readonly("z_face", [] (const fwt::Grid& g) {
                 return AsNumpy(g.z_face());
             }, "Cell faces, shape (nz+1,) [m]. A copy.")
        .def_property_readonly("z_cc", [] (const fwt::Grid& g) {
                 return AsNumpy(g.z_cc());
             }, "Cell centres, shape (nz,) [m]. A copy.")
        .def_property_readonly("n_boxes", [] (const fwt::Grid& g) {
                 return g.ba().size();
             })
        .def("__repr__", [] (const fwt::Grid& g) {
                 return "<fastwindterrain.Grid n_cell=(" +
                        std::to_string(g.nx()) + ", " +
                        std::to_string(g.ny()) + ", " +
                        std::to_string(g.nz()) + ") dz0=" +
                        std::to_string(g.dz0()) + " r=" +
                        std::to_string(g.stretching_ratio()) + ">";
             });

    py::class_<fwt::Solver>(m, "Solver", R"doc(
The solver pipeline, as an object -- the same one the executable runs.

Phase 11 exposes ``setup()`` and the fields it builds. The remaining
stages (``solve()``, ``diagnose()``, ``write_output()``) and dict-based
configuration arrive in later phases; for now the case is described by
the inputs file AMReX was initialized with::

    with fwt.session(["inputs"]):
        s = fwt.Solver()
        s.setup()
        u = s.velocity[0]          # numpy, (nz, ny, nx)
        s.set_velocity(new_field)  # writes back, ghosts refilled

FIELD LAYOUT. Every field comes back as ``(ncomp, nz, ny, nx)``, with
the leading axis dropped when there is one component -- so ``velocity``
is ``(3, nz, ny, nx)`` and ``mask`` is ``(nz, ny, nx)``. The nodal
``lambda_`` is ``(nz+1, ny+1, nx+1)``.

Channels-first is deliberate: it is AMReX's own memory order, so the
gather is a memcpy per component rather than a transpose, and it is what
PyTorch's ``conv3d`` wants.

COPIES, NOT VIEWS. A MultiFab is several boxes, the velocity carries two
ghost layers, and a view would outlive the Solver that owns it. Writing
into a returned array changes nothing; use ``set_velocity``.
)doc")
        .def(py::init([] () {
                 RequireInitialized("Solver");
                 return std::make_unique<fwt::Solver>();
             }))
        .def("setup", [] (fwt::Solver& s,
                          const std::vector<std::string>& args) {
                 RequireInitialized("Solver.setup");
                 amrex::Vector<std::string> a;
                 a.reserve(args.size());
                 for (const std::string& x : args) { a.push_back(x); }
                 s.Setup(a);
             }, py::arg("args") = std::vector<std::string>{},
             "Build every component from the inputs AMReX was initialized\n"
             "with. args are echoed by fwt.debug and nothing else.")
        .def_property_readonly("is_setup", &fwt::Solver::is_setup)
        .def_property_readonly("grid", &fwt::Solver::grid,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("shape", [] (const fwt::Solver& s) {
                 RequireSetup(s, "shape");
                 return py::make_tuple(s.grid().nz(), s.grid().ny(),
                                       s.grid().nx());
             }, "(nz, ny, nx) -- the shape of a scalar cell field.")

        .def_property_readonly("velocity", [] (const fwt::Solver& s) {
                 RequireSetup(s, "velocity");
                 return FieldToNumpy(s.velocity());
             }, "(3, nz, ny, nx) [m/s]. After the projection once it has run.")
        .def_property_readonly("velocity0", [] (const fwt::Solver& s) {
                 RequireSetup(s, "velocity0");
                 return FieldToNumpy(s.velocity0());
             }, "(3, nz, ny, nx) [m/s] -- the field before any correction.")
        .def_property_readonly("mask", [] (const fwt::Solver& s) {
                 RequireSetup(s, "mask");
                 return IFieldToNumpy(s.terrain().mask());
             }, "(nz, ny, nx) int32: 1 solid, 0 fluid.")
        .def_property_readonly("z_terrain", [] (const fwt::Solver& s) {
                 RequireSetup(s, "z_terrain");
                 return FieldToNumpy(s.terrain().z_terrain());
             }, "(nz, ny, nx) [m] -- the surface height of each column.")
        .def_property_readonly("alpha_h", [] (const fwt::Solver& s) {
                 RequireSetup(s, "alpha_h");
                 return FieldToNumpy(s.anisotropy().alpha_h());
             }, "(nz, ny, nx) -- the horizontal variational weight.")
        .def_property_readonly("alpha_v", [] (const fwt::Solver& s) {
                 RequireSetup(s, "alpha_v");
                 return FieldToNumpy(s.anisotropy().alpha_v());
             }, "(nz, ny, nx) -- the vertical variational weight.")
        .def_property_readonly("sigma", [] (const fwt::Solver& s) {
                 RequireSetup(s, "sigma");
                 return FieldToNumpy(s.poisson().sigma());
             }, "(3, nz, ny, nx) -- Poisson coefficients, metric included.")
        .def_property_readonly("lambda_", [] (const fwt::Solver& s) {
                 RequireSetup(s, "lambda_");
                 return FieldToNumpy(s.poisson().lambda());
             }, "(nz+1, ny+1, nx+1) -- the nodal potential. `lambda` is a\n"
                "Python keyword, hence the trailing underscore.")

        .def("set_velocity", [] (fwt::Solver& s, const py::array& a) {
                 RequireSetup(s, "set_velocity");
                 s.SetVelocity(NumpyToBuffer(a, s.velocity(), "velocity"));
             }, py::arg("array"),
             "Overwrite the velocity from a (3, nz, ny, nx) array.\n\n"
             "The valid region is written and the ghost cells are refilled\n"
             "through the boundary conditions, so the field is immediately\n"
             "consistent for anything that reads a stencil near a face.\n"
             "A mismatched shape raises rather than being broadcast.")

        .def("__repr__", [] (const fwt::Solver& s) {
                 if (!s.is_setup()) {
                     return std::string("<fastwindterrain.Solver (not set up)>");
                 }
                 return "<fastwindterrain.Solver n_cell=(" +
                        std::to_string(s.grid().nx()) + ", " +
                        std::to_string(s.grid().ny()) + ", " +
                        std::to_string(s.grid().nz()) + ")>";
             });
}
