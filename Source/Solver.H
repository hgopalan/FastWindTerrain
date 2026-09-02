#ifndef FWT_SOLVER_H_
#define FWT_SOLVER_H_

#include "Grid.H"
#include "Terrain.H"
#include "Inflow.H"
#include "BoundaryConditions.H"
#include "Anisotropy.H"
#include "Obrien.H"
#include "Poisson.H"
#include "Diagnostics.H"
#include "Verify.H"

#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

#include <string>

namespace fwt {

// -----------------------------------------------------------------------
// Solver: the whole run, as an object.
//
// WHY THIS EXISTS. Until now the pipeline lived in main(): the order of
// Grid -> Terrain -> Inflow -> BC -> Anisotropy -> O'Brien -> Poisson ->
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

    void Setup (const amrex::Vector<std::string>& args = {});
    void Solve ();
    void Diagnose ();
    void WriteOutput () const;

    void Run (const amrex::Vector<std::string>& args = {});

    // Read-only access to the components, for the bindings and for any
    // caller that wants to look at the state between stages.
    const Grid&        grid ()      const { return m_grid; }
    const Terrain&     terrain ()   const { return m_terrain; }
    const Inflow&      inflow ()    const { return m_inflow; }
    const Anisotropy&  anisotropy() const { return m_aniso; }
    const Obrien&      obrien ()    const { return m_obrien; }
    const Poisson&     poisson ()   const { return m_poisson; }
    const Diagnostics& diagnostics () const { return m_diag; }

    // The velocity after the projection, and the field that went in.
    const amrex::MultiFab& velocity () const { return m_inflow.velocity(); }
    const amrex::MultiFab& velocity0 () const { return m_vel0; }
    const amrex::MultiFab& divergence () const { return m_divergence; }

    bool manufactured () const { return m_manufactured != 0; }
    Poisson::Error mms_error () const { return m_mms_err; }

private:

    Grid                m_grid;
    Terrain             m_terrain;
    Inflow              m_inflow;
    BoundaryConditions  m_bc;
    Verify              m_verify;
    Anisotropy          m_aniso;
    Obrien              m_obrien;
    Poisson             m_poisson;
    Diagnostics         m_diag;

    amrex::MultiFab m_vel0;         // the field before the correction
    amrex::MultiFab m_divergence;   // per-cell div(u) after it

    int m_manufactured = 0;
    Poisson::Error m_mms_err {0.0, 0.0};

    bool m_setup_done = false;
    bool m_solved = false;
    bool m_diagnosed = false;
};

} // namespace fwt

#endif
