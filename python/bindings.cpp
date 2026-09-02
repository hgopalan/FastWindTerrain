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
#include "Anisotropy.H"
#include "Inflow.H"
#include "Obrien.H"
#include "Terrain.H"
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

// A dict key set, checked strictly. Same reasoning as Grid: ParmParse
// ignores a misspelling and mentions it once at finalize, which is how a
// typo produces a whole dataset from the wrong terrain.
void RejectUnknownKeys (const py::dict& d,
                        const std::vector<std::string>& known,
                        const char* prefix)
{
    for (auto item : d) {
        const std::string key = py::cast<std::string>(py::str(item.first));
        if (std::find(known.begin(), known.end(), key) != known.end()) {
            continue;
        }
        std::string msg = std::string("unknown ") + prefix + " parameter '" +
                          key + "'. Valid keys are:";
        for (const std::string& k : known) { msg += " " + k; }
        throw fwt::InputError(msg);
    }
}

template <typename T>
T GetScalar (const py::dict& d, const char* key, const char* prefix)
{
    try {
        return py::cast<T>(d[key]);
    } catch (const py::cast_error&) {
        throw fwt::InputError(std::string(prefix) + "." + key +
                              " is not a number");
    }
}

// An (n, ncol) float array split into ncol host columns. Anything else --
// a flat array, a transposed one, the wrong width -- raises rather than
// being reshaped, since a silently transposed point cloud is a whole
// dataset built on the wrong terrain.
std::vector<std::vector<amrex::Real>>
Columns (const py::object& obj, int ncol, const char* what)
{
    auto arr = py::array_t<amrex::Real, py::array::c_style |
                                        py::array::forcecast>::ensure(obj);
    if (!arr) {
        throw fwt::InputError(std::string(what) +
                              " must be an array of numbers");
    }
    if (arr.ndim() != 2 || arr.shape(1) != ncol) {
        std::string got;
        for (py::ssize_t k = 0; k < arr.ndim(); ++k) {
            got += (k ? ", " : "") + std::to_string(arr.shape(k));
        }
        throw fwt::InputError(std::string(what) + " has shape (" + got +
                              "), expected (n, " + std::to_string(ncol) +
                              ")");
    }
    const py::ssize_t n = arr.shape(0);
    if (n == 0) {
        throw fwt::InputError(std::string(what) + " is empty");
    }

    std::vector<std::vector<amrex::Real>> cols(ncol);
    for (int c = 0; c < ncol; ++c) { cols[c].resize(std::size_t(n)); }
    const amrex::Real* p = arr.data();
    for (py::ssize_t i = 0; i < n; ++i) {
        for (int c = 0; c < ncol; ++c) {
            cols[c][std::size_t(i)] = p[i * ncol + c];
        }
    }
    return cols;
}

fwt::Terrain::Params TerrainParamsFromDict (const py::dict& d)
{
    RejectUnknownKeys(d, {"file", "flat_elevation", "idw_n_neighbors",
                          "idw_exponent", "points"}, "terrain");

    fwt::Terrain::Params p;
    if (d.contains("file")) {
        p.file = py::cast<std::string>(py::str(d["file"]));
    }
    if (d.contains("flat_elevation")) {
        p.flat_elevation = GetScalar<amrex::Real>(d, "flat_elevation",
                                                  "terrain");
        p.given_flat = true;
    }
    if (d.contains("idw_n_neighbors")) {
        p.idw_n_neighbors = GetScalar<int>(d, "idw_n_neighbors", "terrain");
        p.given_k = true;
    }
    if (d.contains("idw_exponent")) {
        p.idw_exponent = GetScalar<amrex::Real>(d, "idw_exponent", "terrain");
        p.given_p = true;
    }
    if (d.contains("points")) {
        auto cols = Columns(d["points"], 3, "terrain points");
        p.xp = std::move(cols[0]);
        p.yp = std::move(cols[1]);
        p.zp = std::move(cols[2]);
    }

    p.Validate();
    return p;
}

fwt::Inflow::Params InflowParamsFromDict (const py::dict& d)
{
    RejectUnknownKeys(d, {"mode", "u_ref", "v_ref", "z_ref",
                          "powerlaw_exponent", "z0", "z_agl_min",
                          "idw_n_neighbors", "idw_exponent", "file",
                          "points", "velocity"}, "inflow");

    fwt::Inflow::Params p;
    if (d.contains("mode")) {
        p.mode_name = py::cast<std::string>(py::str(d["mode"]));
    }
    if (d.contains("u_ref")) { p.u_ref = GetScalar<amrex::Real>(d, "u_ref", "inflow"); }
    if (d.contains("v_ref")) { p.v_ref = GetScalar<amrex::Real>(d, "v_ref", "inflow"); }
    if (d.contains("z_ref")) {
        p.z_ref = GetScalar<amrex::Real>(d, "z_ref", "inflow");
        p.given_z_ref = true;
    }
    if (d.contains("powerlaw_exponent")) {
        p.powerlaw_exponent = GetScalar<amrex::Real>(d, "powerlaw_exponent",
                                                     "inflow");
        p.given_exponent = true;
    }
    if (d.contains("z0")) {
        p.z0 = GetScalar<amrex::Real>(d, "z0", "inflow");
        p.given_z0 = true;
    }
    if (d.contains("z_agl_min")) {
        p.z_agl_min = GetScalar<amrex::Real>(d, "z_agl_min", "inflow");
        p.given_z_agl_min = true;
    }
    if (d.contains("idw_n_neighbors")) {
        p.idw_n_neighbors = GetScalar<int>(d, "idw_n_neighbors", "inflow");
        p.given_k = true;
    }
    if (d.contains("idw_exponent")) {
        p.idw_exponent = GetScalar<amrex::Real>(d, "idw_exponent", "inflow");
        p.given_p = true;
    }
    if (d.contains("file")) {
        p.file = py::cast<std::string>(py::str(d["file"]));
    }

    const bool has_points = d.contains("points");
    const bool has_vel    = d.contains("velocity");
    if (has_points != has_vel) {
        throw fwt::InputError(
            "a userfile table needs both 'points' (n, 3) and 'velocity' "
            "(n, 3); only one was given");
    }
    if (has_points) {
        auto pc = Columns(d["points"], 3, "inflow points");
        auto vc = Columns(d["velocity"], 3, "inflow velocity");
        if (pc[0].size() != vc[0].size()) {
            throw fwt::InputError(
                "inflow points has " + std::to_string(pc[0].size()) +
                " rows but velocity has " + std::to_string(vc[0].size()));
        }
        p.xp = std::move(pc[0]);
        p.yp = std::move(pc[1]);
        p.zp = std::move(pc[2]);
        p.up = std::move(vc[0]);
        p.vp = std::move(vc[1]);
        p.wp = std::move(vc[2]);
    }

    p.Validate();
    return p;
}

fwt::Anisotropy::Params AnisotropyParamsFromDict (const py::dict& d)
{
    RejectUnknownKeys(d, {"enable", "source", "alpha_h_mode", "slope_scale",
                          "decay_height", "min_factor", "max_factor"},
                      "anisotropy");
    fwt::Anisotropy::Params p;
    if (d.contains("enable")) {
        p.enable = py::cast<bool>(d["enable"]) ? 1 : 0;
    }
    if (d.contains("source")) {
        p.source = py::cast<std::string>(py::str(d["source"]));
    }
    if (d.contains("alpha_h_mode")) {
        p.alpha_h_mode = py::cast<std::string>(py::str(d["alpha_h_mode"]));
    }
    if (d.contains("slope_scale")) {
        p.slope_scale = GetScalar<amrex::Real>(d, "slope_scale", "anisotropy");
    }
    if (d.contains("decay_height")) {
        p.decay_height = GetScalar<amrex::Real>(d, "decay_height",
                                                "anisotropy");
    }
    if (d.contains("min_factor")) {
        p.min_factor = GetScalar<amrex::Real>(d, "min_factor", "anisotropy");
    }
    if (d.contains("max_factor")) {
        p.max_factor = GetScalar<amrex::Real>(d, "max_factor", "anisotropy");
    }
    return p;   // Validate runs in Build, with the bases filled in
}

fwt::Obrien::Params ObrienParamsFromDict (const py::dict& d)
{
    RejectUnknownKeys(d, {"enable"}, "obrien");
    fwt::Obrien::Params p;
    if (d.contains("enable")) {
        p.enable = py::cast<bool>(d["enable"]) ? 1 : 0;
    }
    return p;
}

fwt::Poisson::Params PoissonParamsFromDict (const py::dict& d)
{
    RejectUnknownKeys(d, {"alpha_h", "alpha_v", "lambda_bc", "rhs_operator",
                          "gradient_operator", "n_projections", "max_iter",
                          "reltol", "abstol", "num_pre_smooth",
                          "num_post_smooth", "verbose", "manufactured",
                          "force_all_dirichlet", "rhs_dump_file"},
                      "poisson");
    fwt::Poisson::Params p;
    if (d.contains("alpha_h")) {
        p.alpha_h = GetScalar<amrex::Real>(d, "alpha_h", "poisson");
        p.given_alpha_h = true;
    }
    if (d.contains("alpha_v")) {
        p.alpha_v = GetScalar<amrex::Real>(d, "alpha_v", "poisson");
        p.given_alpha_v = true;
    }
    if (d.contains("lambda_bc")) {
        p.lambda_bc = py::cast<std::string>(py::str(d["lambda_bc"]));
    }
    if (d.contains("rhs_operator")) {
        p.rhs_operator = py::cast<std::string>(py::str(d["rhs_operator"]));
    }
    if (d.contains("gradient_operator")) {
        p.gradient_operator =
            py::cast<std::string>(py::str(d["gradient_operator"]));
    }
    if (d.contains("n_projections")) {
        p.n_projections = GetScalar<int>(d, "n_projections", "poisson");
    }
    if (d.contains("max_iter")) {
        p.max_iter = GetScalar<int>(d, "max_iter", "poisson");
    }
    if (d.contains("reltol")) {
        p.reltol = GetScalar<amrex::Real>(d, "reltol", "poisson");
    }
    if (d.contains("abstol")) {
        p.abstol = GetScalar<amrex::Real>(d, "abstol", "poisson");
    }
    if (d.contains("num_pre_smooth")) {
        p.num_pre_smooth = GetScalar<int>(d, "num_pre_smooth", "poisson");
    }
    if (d.contains("num_post_smooth")) {
        p.num_post_smooth = GetScalar<int>(d, "num_post_smooth", "poisson");
    }
    if (d.contains("verbose")) {
        p.verbose = GetScalar<int>(d, "verbose", "poisson");
    }
    if (d.contains("manufactured")) {
        p.manufactured = py::cast<bool>(d["manufactured"]) ? 1 : 0;
    }
    if (d.contains("force_all_dirichlet")) {
        p.force_all_dirichlet =
            py::cast<bool>(d["force_all_dirichlet"]) ? 1 : 0;
    }
    if (d.contains("rhs_dump_file")) {
        p.rhs_dump_file = py::cast<std::string>(py::str(d["rhs_dump_file"]));
    }
    p.Validate();
    return p;
}

// The whole case, as one nested dict. An absent section means "use the
// defaults", not "use whatever ParmParse happens to hold".
fwt::Solver::Params SolverParamsFromDict (const py::dict& d)
{
    RejectUnknownKeys(d, {"grid", "terrain", "inflow", "anisotropy",
                          "obrien", "poisson", "numerics"}, "solver");

    auto section = [&] (const char* key) {
        return d.contains(key) ? py::cast<py::dict>(d[key]) : py::dict();
    };

    fwt::Solver::Params p;
    if (!d.contains("grid")) {
        throw fwt::InputError("a solver configuration needs a 'grid' section");
    }
    p.grid       = ParamsFromDict(section("grid"));
    p.terrain    = TerrainParamsFromDict(section("terrain"));
    p.inflow     = InflowParamsFromDict(section("inflow"));
    p.anisotropy = AnisotropyParamsFromDict(section("anisotropy"));
    p.obrien     = ObrienParamsFromDict(section("obrien"));
    p.poisson    = PoissonParamsFromDict(section("poisson"));

    if (d.contains("numerics")) {
        py::dict n = section("numerics");
        RejectUnknownKeys(n, {"gradient_scheme"}, "numerics");
        if (n.contains("gradient_scheme")) {
            p.gradient_scheme =
                py::cast<std::string>(py::str(n["gradient_scheme"]));
        }
    }
    return p;
}

void RequireSolved (const fwt::Solver& s, const char* what)
{
    if (!s.is_solved()) {
        throw std::runtime_error(
            std::string(what) + " is not available until a projection has "
            "run. Call solve() or project_once() first.");
    }
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

    py::class_<fwt::Terrain>(m, "Terrain", R"doc(
Terrain height and the immersed-boundary mask.

Built from a dict, with the scattered points handed in directly rather
than read from a CSV::

    t = fwt.Terrain(grid, {"points": pts})       # pts is (n, 3): x, y, z

    t = fwt.Terrain(grid, {"file": "terrain.csv"})   # or from a file
    t = fwt.Terrain(grid, {"flat_elevation": 0.0})   # or flat ground

``points`` and ``file`` are mutually exclusive: two sources for one thing
is a mistake worth reporting, not a precedence rule to remember.

The points go through exactly the inverse-distance interpolation the file
path uses -- there is no second code path -- so a case built this way is
bit-for-bit the case the CSV would have produced.
)doc")
        .def(py::init([] (const fwt::Grid& g, const py::dict& d) {
                 RequireInitialized("Terrain");
                 auto t = std::make_unique<fwt::Terrain>();
                 t->Build(g, TerrainParamsFromDict(d));
                 return t;
             }), py::arg("grid"), py::arg("params") = py::dict())
        .def_property_readonly("z_terrain", [] (const fwt::Terrain& t) {
                 return FieldToNumpy(t.z_terrain());
             }, "(nz, ny, nx) [m] -- surface height, replicated along k.")
        .def_property_readonly("mask", [] (const fwt::Terrain& t) {
                 return IFieldToNumpy(t.mask());
             }, "(nz, ny, nx) int32: 1 solid, 0 fluid.")
        .def_property_readonly("z_min", &fwt::Terrain::z_min)
        .def_property_readonly("z_max", &fwt::Terrain::z_max)
        .def_property_readonly("n_solid", &fwt::Terrain::n_solid)
        .def_property_readonly("n_total", &fwt::Terrain::n_total)
        .def_property_readonly("n_points", &fwt::Terrain::n_points)
        .def("__repr__", [] (const fwt::Terrain& t) {
                 return "<fastwindterrain.Terrain z in [" +
                        std::to_string(t.z_min()) + ", " +
                        std::to_string(t.z_max()) + "] m, " +
                        std::to_string(t.n_solid()) + " solid of " +
                        std::to_string(t.n_total()) + ">";
             });

    py::class_<fwt::Inflow>(m, "Inflow", R"doc(
The initial wind field, anchored to height above ground.

Built from a dict, with no inputs file involved::

    inf = fwt.Inflow(grid, terrain,
                     {"mode": "powerlaw", "u_ref": 8.0, "v_ref": 6.0,
                      "z_ref": 10.0, "powerlaw_exponent": 0.14})

``userfile`` mode takes the table directly, as two (n, 3) arrays instead
of a six-column file::

    inf = fwt.Inflow(grid, terrain,
                     {"mode": "userfile", "points": xyz, "velocity": uvw})

``points``/``velocity`` and ``file`` are mutually exclusive. The table
goes through exactly the 3D inverse-distance interpolation the file path
uses, so the two agree bit for bit.
)doc")
        .def(py::init([] (const fwt::Grid& g, const fwt::Terrain& t,
                          const py::dict& d) {
                 RequireInitialized("Inflow");
                 auto inf = std::make_unique<fwt::Inflow>();
                 inf->Build(g, t, InflowParamsFromDict(d));
                 return inf;
             }), py::arg("grid"), py::arg("terrain"), py::arg("params"))
        .def_property_readonly("velocity", [] (const fwt::Inflow& i) {
                 return FieldToNumpy(i.velocity());
             }, "(3, nz, ny, nx) [m/s] -- the profile mapped onto the grid.")
        .def_property_readonly("mode", [] (const fwt::Inflow& i) {
                 return i.mode_name();
             })
        .def_property_readonly("speed_ref", &fwt::Inflow::speed_ref)
        .def_property_readonly("z_agl_min", &fwt::Inflow::z_agl_min)
        .def_property_readonly("direction", [] (const fwt::Inflow& i) {
                 return py::make_tuple(i.dir_x(), i.dir_y());
             })
        .def_property_readonly("flux_in", &fwt::Inflow::flux_in)
        .def_property_readonly("flux_out", &fwt::Inflow::flux_out)
        .def_property_readonly("flux_net", &fwt::Inflow::flux_net)
        .def_property_readonly("flux_imbalance", &fwt::Inflow::flux_imbalance)
        .def_property_readonly("n_points", &fwt::Inflow::n_points)
        .def("profile_speed", &fwt::Inflow::ProfileSpeed, py::arg("z_agl"),
             "Speed of the 1D law at a height above ground [m/s].\n"
             "Raises for mode = userfile, which is a 3D field.")
        .def("__repr__", [] (const fwt::Inflow& i) {
                 return "<fastwindterrain.Inflow mode=" + i.mode_name() +
                        " speed_ref=" + std::to_string(i.speed_ref()) + ">";
             });

    py::class_<fwt::Solver>(m, "Solver", R"doc(
The solver pipeline, as an object -- the same one the executable runs.

Configured either from a dict, with no inputs file anywhere::

    with fwt.session():
        s = fwt.Solver({"grid": {...}, "terrain": {"points": pts},
                        "inflow": {"u_ref": 8.0, "v_ref": 6.0},
                        "poisson": {"alpha_v": 0.5, "n_projections": 4}})
        s.setup()
        s.solve()
        s.diagnose()
        u = s.velocity[0]

or from the inputs file AMReX was initialized with::

    with fwt.session(["inputs"]):
        s = fwt.Solver()
        s.run()

An absent section means "use the defaults", never "use whatever
ParmParse happens to hold" -- which is what makes a generation loop
safe. An unknown section or key raises.

STEPWISE. ``project_once()`` runs a single projection pass and returns
the MLMG residual, so a notebook can watch an approximate projection
converge rather than be told that it did::

    s.setup()
    for _ in range(4):
        s.project_once()
        print(s.max_divergence_fe)

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
        .def(py::init([] (const py::object& config) {
                 RequireInitialized("Solver");
                 auto s = std::make_unique<fwt::Solver>();
                 if (!config.is_none()) {
                     s->set_config(SolverParamsFromDict(
                         py::cast<py::dict>(config)));
                 }
                 return s;
             }), py::arg("config") = py::none(),
             "A solver. With a config dict the case is described entirely\n"
             "in Python; without one it comes from the inputs file AMReX\n"
             "was initialized with.")
        .def("setup", [] (fwt::Solver& s,
                          const std::vector<std::string>& args) {
                 RequireInitialized("Solver.setup");
                 amrex::Vector<std::string> a;
                 a.reserve(args.size());
                 for (const std::string& x : args) { a.push_back(x); }
                 if (s.has_config()) { s.Setup(s.config(), a); }
                 else                { s.Setup(a); }
             }, py::arg("args") = std::vector<std::string>{},
             "Build every component. args are echoed by fwt.debug and\n"
             "nothing else.")
        .def("solve", [] (fwt::Solver& s) {
                 RequireSetup(s, "solve()");
                 s.Solve();
             }, "Run the projection loop (or the manufactured solution).")
        .def("project_once", [] (fwt::Solver& s) {
                 RequireSetup(s, "project_once()");
                 return s.ProjectOnce();
             }, "One projection pass: rebuild the RHS, solve, correct,\n"
                "refill the ghosts. Returns the MLMG residual.")
        .def("diagnose", [] (fwt::Solver& s) {
                 RequireSolved(s, "diagnose()");
                 s.Diagnose();
             }, "Compute the divergence field and the post-solve report.")
        .def("write_output", [] (const fwt::Solver& s) {
                 s.WriteOutput();
             }, "Write the report and the field output.\n\n"
                "Still driven by ParmParse, so a Python-configured run\n"
                "writes to the default names. In-memory output is the\n"
                "next phase.")
        .def("run", [] (fwt::Solver& s,
                        const std::vector<std::string>& args) {
                 RequireInitialized("Solver.run");
                 amrex::Vector<std::string> a;
                 a.reserve(args.size());
                 for (const std::string& x : args) { a.push_back(x); }
                 if (s.has_config()) { s.Run(s.config(), a); }
                 else                { s.Run(a); }
             }, py::arg("args") = std::vector<std::string>{},
             "setup, solve, diagnose and write_output, in order.")
        .def_property_readonly("is_setup", &fwt::Solver::is_setup)
        .def_property_readonly("is_solved", &fwt::Solver::is_solved)
        .def_property_readonly("is_diagnosed", &fwt::Solver::is_diagnosed)
        .def_property_readonly("n_projections_done",
                               &fwt::Solver::n_projections_done)
        .def_property_readonly("solve_residual", [] (const fwt::Solver& s) {
                 RequireSolved(s, "solve_residual");
                 return s.poisson().solve_residual();
             }, "MLMG residual of the last solve.")
        .def_property_readonly("solve_iterations", [] (const fwt::Solver& s) {
                 RequireSolved(s, "solve_iterations");
                 return s.poisson().solve_iterations();
             }, "MLMG iterations of the last solve. A solve that hit\n"
                "max_iter has not converged, whatever its residual says.")
        .def_property_readonly("max_divergence_fe", [] (fwt::Solver& s) {
                 RequireSetup(s, "max_divergence_fe");
                 return s.MaxDivergenceFE();
             }, "max|div(u)| in the norm the projection controls. This is\n"
                "the number that measures whether a pass helped.")
        .def_property_readonly("max_divergence", [] (const fwt::Solver& s) {
                 RequireSetup(s, "max_divergence");
                 return s.MaxDivergence();
             }, "max|div(u)| with the configured derivative scheme. This\n"
                "one reads the velocity's ghost cells, so it is what shows\n"
                "whether they hold what they should.")
        .def_property_readonly("divergence", [] (const fwt::Solver& s) {
                 RequireSolved(s, "divergence");
                 if (!s.is_diagnosed()) {
                     throw std::runtime_error(
                         "divergence is not available until diagnose() has "
                         "run.");
                 }
                 return FieldToNumpy(s.divergence());
             }, "(nz, ny, nx) [1/s] -- div(u) per cell, zero in solid cells.")
        .def_property_readonly("diagnostics", [] (const fwt::Solver& s) {
                 if (!s.is_diagnosed()) {
                     throw std::runtime_error(
                         "diagnostics are not available until diagnose() "
                         "has run.");
                 }
                 const auto& dg = s.diagnostics();
                 py::dict out;
                 out["div_max"] = dg.div_max();
                 out["div_l2"] = dg.div_l2();
                 out["flux_in"] = dg.flux().in;
                 out["flux_out"] = dg.flux().out;
                 out["flux_net"] = dg.flux().net;
                 out["flux_imbalance"] = dg.flux().imbalance;
                 out["flux_within_tolerance"] = dg.flux_within_tolerance();
                 return out;
             }, "The post-solve diagnostics, as a dict.")
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
