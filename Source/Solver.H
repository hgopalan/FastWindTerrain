#ifndef FWT_SOLVER_H_
#define FWT_SOLVER_H_

#include "Grid.H"
#include "Terrain.H"
#include "Inflow.H"
#include "BoundaryConditions.H"
#include "Anisotropy.H"
#include "Obrien.H"
#include "Surface.H"
#include "Poisson.H"
#include "Diagnostics.H"
#include "Verify.H"
#include "Output.H"

#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

#include <string>

namespace fwt {

// -----------------------------------------------------------------------
// Solver: the whole run, as an object.
//
// WHY THIS EXISTS. Until now the pipeline lived in main(): the order of
// Grid -> Terrain -> Inflow -> BC -> Anisotropy -> Surface -> O'Brien ->
// Poisson ->
// the projection loop -> diagnostics -> output was a sequence of
// statements inside the executable, reachable only by running it.
//
// The Python bindings need that same sequence. If they re-implemented it
// there would be two orderings to keep in step, and nothing would notice
// when they diverged -- which is the failure this codebase has spent
// eight phases designing against. So the pipeline moves here, main()
// becomes a driver over it, and the bindings call the same object.
//
// That makes the executable a test of the class the bindings expose:
// every existing regtest group exercises this code.
//
// THE STAGES are separated so a later phase can drive them one at a time
// from a notebook. Run() is the four of them in order, and is what both
// entry points call today.
//
//     Setup()       -- read inputs, build every component, keep u0
//     Solve()       -- the projection loop, or the manufactured solution
//     Diagnose()    -- the divergence field and the post-solve report
//     WriteOutput() -- the report and the field output
//
// ARGUMENTS. Setup() takes the command-line arguments AFTER the program
// name, purely so fwt.debug can echo them. AMReX has already parsed them
// into ParmParse by the time this runs; passing them again is what keeps
// the debug output of a Python run byte-identical to the executable's.
//
// LIFETIME. This class owns no global state, but AMReX does: it must be
// constructed and destroyed between amrex::Initialize and
// amrex::Finalize. It does not call either itself, so the caller keeps
// control of the process-global lifecycle -- which is what lets Python
// drive many cases inside one initialized AMReX.
// -----------------------------------------------------------------------
class Solver
{
public:

    // -------------------------------------------------------------------
    // A whole case, as data.
    //
    // Every module's Params in one place, so a caller can describe a run
    // without ParmParse being involved anywhere. That is what dataset
    // generation needs: ParmParse persists for the life of an AMReX
    // initialization, so a loop that configured through it would have
    // each case inherit whatever the last one set.
    //
    // FromParmParse builds one from an inputs deck, which is what the
    // executable does. The two paths meet immediately afterwards.
    // -------------------------------------------------------------------
    struct Params
    {
        Grid::Params       grid;
        Terrain::Params    terrain;
        Inflow::Params     inflow;
        Anisotropy::Params anisotropy;
        Obrien::Params     obrien;
        Surface::Params    surface;
        Poisson::Params    poisson;

        OutputParams       output;

        // numerics.gradient_scheme. Empty leaves the current setting,
        // which is what the ParmParse path has already established.
        std::string gradient_scheme;

        static Params FromParmParse ();
    };

    void Setup (const Params& params,
                const amrex::Vector<std::string>& args = {});
    void Setup (const amrex::Vector<std::string>& args = {});

    // One projection pass: rebuild the RHS from the current velocity,
    // solve for lambda, correct, refill the ghosts. Returns the MLMG
    // residual.
    //
    // Solve() is this in a loop. It is a named thing so a notebook can
    // step through the projection one pass at a time and look at lambda
    // in between -- which is the only way to see an approximate
    // projection converging rather than being told that it did.
    amrex::Real ProjectOnce ();

    void Solve ();
    void Diagnose ();
    void WriteOutput () const;

    // -------------------------------------------------------------------
    // The output fields, in memory.
    //
    // This is the SAME object the plotfile and the ascii backends are
    // handed -- one gather, three consumers now instead of two. A
    // notebook or a dataset generator gets exactly the array the file
    // would have contained, without a file.
    //
    // Requires Diagnose(), since the divergence component comes from it.
    // -------------------------------------------------------------------
    OutputFields CollectOutput () const;

    // Explicit destinations, for a caller that has not gone through
    // ParmParse. WriteOutput() is these three behind the configured
    // format switches.
    void WritePlotfile (const std::string& path) const;
    void WriteAscii (const std::string& path) const;
    void WriteReport (const std::string& path) const;

    void Run (const Params& params,
              const amrex::Vector<std::string>& args = {});
    void Run (const amrex::Vector<std::string>& args = {});

    // Read-only access to the components, for the bindings and for any
    // caller that wants to look at the state between stages.
    const Grid&        grid ()      const { return m_grid; }
    const Terrain&     terrain ()   const { return m_terrain; }
    const Inflow&      inflow ()    const { return m_inflow; }
    const Anisotropy&  anisotropy() const { return m_aniso; }
    const Obrien&      obrien ()    const { return m_obrien; }
    const Surface&     surface ()   const { return m_surface; }
    const Poisson&     poisson ()   const { return m_poisson; }
    const Diagnostics& diagnostics () const { return m_diag; }

    // Overwrites the velocity's valid region from a flat buffer in
    // FieldIO's layout, then refills its ghosts through the boundary
    // conditions so the field is immediately consistent for anything
    // that reads a stencil near a face.
    //
    // Refilling here rather than leaving it to the caller is the whole
    // reason this lives on Solver: only the boundary conditions know
    // what the ghosts should hold, and a field written from Python with
    // stale ghosts would give a quietly wrong divergence two calls
    // later.
    void SetVelocity (const amrex::Vector<amrex::Real>& buffer);

    bool is_setup ()     const { return m_setup_done; }
    bool is_solved ()    const { return m_solved; }
    bool is_diagnosed () const { return m_diagnosed; }

    // The velocity after the projection, and the field that went in.
    const amrex::MultiFab& velocity () const { return m_inflow.velocity(); }
    const amrex::MultiFab& velocity0 () const { return m_vel0; }
    const amrex::MultiFab& divergence () const { return m_divergence; }

    const Params& params () const { return m_params; }

    // A configuration held before Setup runs, so a caller can describe
    // the case at construction and build it later.
    void set_config (const Params& p) { m_config = p; m_has_config = true; }
    bool has_config () const { return m_has_config; }
    const Params& config () const { return m_config; }

    // max|div(u)| in the norm the projection controls, for the current
    // velocity. Non-const because AMReX's compDivergence is.
    amrex::Real MaxDivergenceFE ();

    // max|div(u)| computed with the configured derivative scheme. Unlike
    // the FE norm this reads the velocity's GHOST cells through a
    // five-point stencil, which makes it the probe for whether they hold
    // what they should.
    amrex::Real MaxDivergence () const;
    bool manufactured () const { return m_manufactured != 0; }
    // Projection passes actually run, so a caller stepping through with
    // ProjectOnce can see how many it has done.
    int n_projections_done () const { return m_n_proj_done; }
    Poisson::Error mms_error () const { return m_mms_err; }

private:

    Grid                m_grid;
    Terrain             m_terrain;
    Inflow              m_inflow;
    BoundaryConditions  m_bc;
    Verify              m_verify;
    Anisotropy          m_aniso;
    Obrien              m_obrien;
    Surface             m_surface;
    Poisson             m_poisson;
    Diagnostics         m_diag;

    amrex::MultiFab m_vel0;         // the field before the correction
    amrex::MultiFab m_divergence;   // per-cell div(u) after it

    Params m_params;
    Params m_config;
    bool m_has_config = false;
    int m_manufactured = 0;
    int m_n_proj_done = 0;
    Poisson::Error m_mms_err {0.0, 0.0};

    bool m_setup_done = false;
    bool m_solved = false;
    bool m_diagnosed = false;
};

} // namespace fwt

#endif
