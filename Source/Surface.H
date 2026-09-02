#ifndef FWT_SURFACE_H
#define FWT_SURFACE_H

#include <AMReX_MultiFab.H>

#include "Grid.H"
#include "Terrain.H"

// ---------------------------------------------------------------------------
// Surface.H -- the boundary condition on the first fluid cell above terrain.
//
// WHAT IS WRONG WITHOUT ONE. Velocity is zeroed inside the terrain
// (Inflow.cpp:382), but nothing constrains the FLUID cell just above it. Two
// things follow, and both contaminate exactly the near-surface layer that
// pedestrian and met-mast heights are read from:
//
//   1. The velocity in that cell has a component pointing INTO the surface.
//      Air flows into rock. The immersed boundary marks the cell below as
//      solid and stops there; it never makes the flow tangential.
//
//   2. Its speed comes from evaluating the inflow profile at its own height
//      above ground, which on a stretched grid is a metre or two. That is
//      inside the roughness sublayer where the profile is steepest and least
//      trustworthy, and it takes no account of the terrain-induced speed-up
//      the cell above it already shows.
//
// THE CONDITIONS, selected by surface.type:
//
//   wall_function  (default) no penetration, and the speed set from the
//                            friction velocity of the cell above
//   slip                     no penetration only; the tangential flow is
//                            left as it is (free slip)
//   noslip                   u = 0 in the first fluid cell
//   none                     the cell is not touched
//
// NO PENETRATION removes the surface-normal component:
//
//     n = (-dh/dx, -dh/dy, 1) / sqrt(1 + |grad h|^2)
//     u <- u - (u . n) n
//
// The normal comes from the terrain column heights by the same central
// difference Anisotropy uses for slope, so the terrain gradient has one
// definition in this code rather than two.
//
// THE WALL FUNCTION takes the friction velocity from the SECOND fluid cell
// and uses it to set the speed in the first:
//
//     u* = kappa * U_2 / ln((z_2 + z0) / z0)
//     |U_1| <- (u* / kappa) * ln((z_1 + z0) / z0)
//
// leaving the direction alone. The second cell is the anchor because it is
// the lowest one that is not itself suspect: the first is the cell being
// corrected, and below that the next known value is z0, not a cell.
//
// WHY THE LOG LAW EVEN WHEN inflow.mode = powerlaw. Friction velocity is a
// log-law quantity -- u* is defined through the log-law slope, dU/dln(z) =
// u*/kappa. A power law has no equivalent invariant: its near-surface slope
// is set by an exponent fitted to the whole profile, which says nothing
// about surface stress. Rather than invent an effective u* from that
// exponent, this uses the log law near the surface whatever the outer
// profile is, which is the physics -- the surface layer IS logarithmic and
// the power law is a convenience fit above it. massconsistent_amr does the
// same by blending a log-law wall function into its outer profile over the
// first cells (src/wall_functions.H).
//
// WHERE IT RUNS. On the initial field, after the inflow is built and before
// O'Brien, so the column integration and the projection both see corrected
// near-surface flow. The projection may move it again -- making the field
// mass-consistent is its job, and re-imposing the condition afterwards puts
// divergence back. surface.apply = both is there for callers who would
// rather have the surface condition exactly satisfied than the field
// exactly divergence-free, and the report records which was chosen.
// ---------------------------------------------------------------------------

namespace fwt {

class Surface
{
public:
    enum class Type  { none, slip, noslip, wall_function };
    enum class Apply { initial, both };

    struct Params {
        //: The default is a wall function, not "nothing". An unconstrained
        //: first fluid cell is not a boundary condition anyone would choose;
        //: it is what you get by not choosing.
        Type  type  = Type::wall_function;
        Apply apply = Apply::initial;

        // Roughness length for the wall function. Negative means "take
        // inflow.z0", the documented default, so one run cannot end up with
        // a wall function at one roughness and a profile at another without
        // saying so.
        amrex::Real z0    = -1.0;
        amrex::Real kappa = 0.41;

        bool given_z0 = false;

        void Validate () const;
    };

    static Type  TypeFromString  (const std::string& s);
    static const char* TypeToString (Type t);
    static Params ParseParams ();

    // inflow_z0 supplies the default roughness. Passed in rather than read
    // back out of Inflow, so this stays usable on a field that did not come
    // from a profile at all.
    void Build (const Grid& grid, const Terrain& terrain, const Params& params,
                amrex::Real inflow_z0);

    // Apply the condition in place. Returns the number of columns touched,
    // so a caller can tell "disabled" from "enabled and found nothing to
    // do", which on flat ground produce the same field and are very
    // different situations.
    long ApplyTo (const Grid& grid, const Terrain& terrain,
                  amrex::MultiFab& vel) const;

    Type type ()          const { return m_params.type; }
    bool active ()        const { return m_params.type != Type::none; }
    bool applies_after () const { return m_params.apply == Apply::both; }
    amrex::Real z0 ()     const { return m_z0; }
    amrex::Real kappa ()  const { return m_params.kappa; }

    // Diagnostics from the last ApplyTo, for the report.
    long n_columns ()                 const { return m_n_columns; }
    amrex::Real max_ustar ()          const { return m_max_ustar; }
    amrex::Real max_normal_removed () const { return m_max_normal; }
    amrex::Real max_speed_change ()   const { return m_max_dspeed; }

    void AppendReport (const std::string& filename) const;

private:
    Params m_params;
    amrex::Real m_z0 = 0.1;

    // Surface normals per column, flattened (j*nx + i).
    std::vector<amrex::Real> m_nx, m_ny, m_nz;

    mutable long m_n_columns = 0;
    mutable amrex::Real m_max_ustar  = 0.0;
    mutable amrex::Real m_max_normal = 0.0;
    mutable amrex::Real m_max_dspeed = 0.0;
};

} // namespace fwt

#endif // FWT_SURFACE_H
