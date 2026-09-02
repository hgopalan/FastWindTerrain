#ifndef FWT_INFLOW_H_
#define FWT_INFLOW_H_

#include "Grid.H"
#include "Terrain.H"
#include "Error.H"
#include "Diagnostics.H"

#include <AMReX_MultiFab.H>

#include <string>
#include <vector>

namespace fwt {

// -----------------------------------------------------------------------
// Inflow: the initial velocity field u0, built by mapping a 1D wind
// profile onto every grid column in height above ground level (AGL).
//
// Inputs (inflow.*):
//   mode              -- powerlaw | loglaw | userfile
//   u_ref, v_ref      -- reference velocity components [m/s] at z_ref
//   z_ref             -- reference height, AGL [m]
//   powerlaw_exponent -- power-law alpha (mode = powerlaw)
//   z0                -- roughness length [m] (mode = loglaw)
//   file              -- six-column x,y,z,u,v,w file (mode = userfile)
//   idw_n_neighbors   -- k nearest points for the userfile interpolation
//   idw_exponent      -- IDW power for the userfile interpolation
//   z_agl_min         -- floor applied to z_agl [m] (default: z0)
//
// TERRAIN AWARENESS. The profile is a function of z_agl = z_cc(k) -
// z_terrain(i,j), evaluated per column, so each column's profile starts
// at its own ground. This holds on the lateral boundaries too: terrain
// may intersect any of xlo/xhi/ylo/yhi, and those faces are then only
// partly open. The userfile z coordinate is likewise read as AGL, so all
// three modes are terrain-following.
//
// z_agl is negative inside the terrain and the log law diverges as
// z_agl -> 0, so z_agl is floored at z_agl_min before either law is
// evaluated. Velocity is zeroed in solid cells regardless.
//
// MASS BALANCE. Because terrain blocks part of a face, the open areas of
// the inflow and outflow faces generally differ, so the raw profile has
// a nonzero net boundary flux. That is NOT an error: the mass-consistent
// solve is itself the mass correction, and it is well posed as long as
// at least one face carries a Dirichlet condition on lambda (Phase 4
// guarantees this by refusing to prescribe velocity on all four lateral
// faces). The imbalance is always computed and REPORTED here.
//
// inflow.balance_flux (DEFAULT OFF) additionally REDISTRIBUTES it. The
// net flux is spread as a single uniform outward-normal velocity
//
//   shift = -flux_net / (total open area of xlo, xhi, ylo and yhi)
//
// over every open cell of the four lateral faces, so that in and out
// match to round-off before the Poisson solve ever runs. The top is left
// alone: it carries w = 0 by boundary condition, and pushing flow
// through it would contradict that. The ground is closed.
//
// Note what this is and is not. It does NOT make the answer more
// accurate -- on the terrain corpus the projection already drives the
// post-solve imbalance to between 2e-4 and 3.5e-3 on its own, and as a
// bulk divergence the initial imbalance is three orders below the local
// divergence the solve is actually working against. Measured on two
// corpus windows it moves the projection's divergence sequence by less
// than a part in a thousand, and leaves the post-solve flux imbalance
// slightly WORSE. What it buys is one thing only: a starting field that
// is exactly conservative -- standard practice in urban CFD, and one
// fewer diagnostic that warns on most real terrain windows and is
// expected to.
//
// A whole-field rescale would not do it: multiplying u0 by s scales
// inflow and outflow alike, so the net scales by s and only reaches zero
// if it was already zero. The correction has to be additive and it has
// to live on the boundary.
//
// TWO CONSEQUENCES, both handled rather than hoped about:
//
//   - The shift changes the normal velocity in the boundary cell layer,
//     which is what BoundaryConditions classifies faces from. A face
//     carrying no through-flow (tangential, for an axis-aligned wind)
//     would pick up the shift and be reclassified -- and if the shift
//     points inward, a run with three inflow faces trips Phase 4's
//     assertion. So the classification is made from flux_prebalance(),
//     the flux the incoming wind carries: which face is an inflow face
//     is a property of the wind, not of a correction applied to it --
//     the same reason the projection never reclassifies either.
//   - On an inflow face the ghost cells are prescribed from the profile,
//     so BoundaryConditions adds the same shift there, keeping the ghost
//     and the interior cell it faces on the same field. On outflow and
//     tangential faces the zero-gradient ghosts inherit it already.
// -----------------------------------------------------------------------
class Inflow
{
public:

    enum class Mode { powerlaw, loglaw, userfile };

    // -------------------------------------------------------------------
    // The inputs, as data -- same shape and same reason as Grid::Params.
    //
    // The userfile table is part of the Params, so "read it from a file"
    // and "here it is" fill one field rather than taking two routes
    // through the interpolation. A table handed in from numpy therefore
    // goes through exactly the 3D IDW the file path uses.
    // -------------------------------------------------------------------
    struct Params
    {
        std::string mode_name = "powerlaw";   // powerlaw|loglaw|userfile
        amrex::Real u_ref = 0.0;
        amrex::Real v_ref = 0.0;
        amrex::Real z_ref = 10.0;
        amrex::Real powerlaw_exponent = 0.14;
        amrex::Real z0 = 0.1;
        // Negative means "follow z0", which is the documented default.
        amrex::Real z_agl_min = -1.0;
        int         idw_n_neighbors = 6;
        amrex::Real idw_exponent = 2.0;
        // Redistribute the net boundary flux over the open area of the
        // four lateral faces. Off by default: with it off the field is
        // exactly what it has always been.
        int         balance_flux = 0;

        std::string file;                     // userfile mode
        // The table given directly: x,y,z and u,v,w, all the same
        // length. When set, `file` must be empty.
        std::vector<amrex::Real> xp, yp, zp, up, vp, wp;

        bool given_z_ref = false;
        bool given_exponent = false;
        bool given_z0 = false;
        bool given_z_agl_min = false;
        bool given_k = false;
        bool given_p = false;

        static Params FromParmParse ();
        void Validate () const;

        bool has_table () const { return !xp.empty(); }
    };

    void Build (const Grid& grid, const Terrain& terrain,
                const Params& params);
    void Build (const Grid& grid, const Terrain& terrain);

    // --- Read-only accessors --------------------------------------------
    // Cell-centered velocity, 3 components (u, v, w), one ghost layer.
    // The non-const overload exists so the boundary conditions can fill
    // the ghost cells in place.
    const amrex::MultiFab& velocity () const { return m_vel; }
    amrex::MultiFab& velocity ()             { return m_vel; }

    Mode mode ()                     const { return m_mode; }
    const std::string& mode_name ()  const { return m_mode_name; }

    // Boundary flux diagnostics [m^3/s], over open (fluid) faces only,
    // of the field this object holds -- so AFTER the redistribution when
    // inflow.balance_flux = 1.
    amrex::Real flux_in ()           const { return m_flux.in; }
    amrex::Real flux_out ()          const { return m_flux.out; }
    amrex::Real flux_net ()          const { return m_flux.net; }
    // |net| / max(|in|, |out|); zero when nothing flows at all.
    amrex::Real flux_imbalance ()    const { return m_flux.imbalance; }

    // The same balance for the RAW profile, before any redistribution.
    // Identical to the above when inflow.balance_flux is off. This is
    // what the boundary conditions classify the faces from, so that the
    // redistribution cannot turn a tangential face into an inflow one.
    const FluxBalance& flux_prebalance () const { return m_flux_raw; }

    bool flux_balanced () const { return m_balance_flux != 0; }
    // Outward-normal velocity [m/s] added to every open cell of the four
    // lateral faces. Zero unless inflow.balance_flux is on.
    amrex::Real flux_balance_shift () const { return m_balance_shift; }

    // The roughness length actually in force. Surface takes its default
    // from here so a run cannot have a wall function at one roughness and
    // a profile at another without saying so.
    amrex::Real z0 ()                const { return m_z0; }

    void AppendReport (const std::string& filename) const;

    // Speed of the 1D profile at a height above ground. Exposed so the
    // regtest checkers and later phases can evaluate the same law.
    // Aborts for mode = userfile, which is a 3D field, not a 1D law.
    amrex::Real ProfileSpeed (amrex::Real z_agl) const;

    // Velocity at an arbitrary point, whatever the mode: the profile
    // laws for powerlaw/loglaw, the interpolated file for userfile.
    // Used to fill inflow ghost cells, which sit outside the domain and
    // so cannot be read from the field itself.
    void VelocityAt (amrex::Real x, amrex::Real y, amrex::Real z_agl,
                     amrex::Real& u, amrex::Real& v, amrex::Real& w) const;

    // Horizontal unit vector along (u_ref, v_ref). This is what the
    // boundary-condition classification tests each face normal against.
    amrex::Real dir_x () const { return m_dir_x; }
    amrex::Real dir_y () const { return m_dir_y; }
    amrex::Real speed_ref () const { return m_speed_ref; }
    amrex::Real z_agl_min () const { return m_z_agl_min; }

    static Mode ParseMode (const std::string& s);

    // Reads a six-column x,y,z,u,v,w velocity file. A five-column file
    // (x,y,z,u,v -- massconsistent_amr's read_velocity_file format) is
    // also accepted, with w taken as zero.
    static void ReadVelocityFile (const std::string& filename,
                                  std::vector<amrex::Real>& xp,
                                  std::vector<amrex::Real>& yp,
                                  std::vector<amrex::Real>& zp,
                                  std::vector<amrex::Real>& up,
                                  std::vector<amrex::Real>& vp,
                                  std::vector<amrex::Real>& wp,
                                  int& n_columns);

    // 3D inverse-distance-weighted velocity at (xq, yq, zq).
    static void InterpolateIDW3D (amrex::Real xq, amrex::Real yq, amrex::Real zq,
                                  const std::vector<amrex::Real>& xp,
                                  const std::vector<amrex::Real>& yp,
                                  const std::vector<amrex::Real>& zp,
                                  const std::vector<amrex::Real>& up,
                                  const std::vector<amrex::Real>& vp,
                                  const std::vector<amrex::Real>& wp,
                                  int k, amrex::Real exponent,
                                  amrex::Real& u, amrex::Real& v, amrex::Real& w);

    const Params& params () const { return m_params; }
    int n_points () const { return static_cast<int>(m_xp.size()); }

private:

    // Copies a validated Params into the individual members and derives
    // the reference direction and the AGL floor from them.
    void ApplyParams (const Params& params);
    void BuildVelocity (const Grid& grid, const Terrain& terrain);
    void ComputeBoundaryFlux (const Grid& grid, const Terrain& terrain);
    // Spreads m_flux.net over the open area of xlo/xhi/ylo/yhi. Sets
    // m_balance_shift; the caller recomputes the flux afterwards.
    void BalanceBoundaryFlux (const Grid& grid, const Terrain& terrain);

    // --- inputs ---
    Params      m_params;
    Mode        m_mode = Mode::powerlaw;
    std::string m_mode_name = "powerlaw";
    amrex::Real m_u_ref = 0.0;
    amrex::Real m_v_ref = 0.0;
    amrex::Real m_z_ref = 10.0;
    amrex::Real m_powerlaw_exponent = 0.14;
    amrex::Real m_z0 = 0.1;
    std::string m_file;
    int         m_idw_k = 6;
    amrex::Real m_idw_exponent = 2.0;
    amrex::Real m_z_agl_min = -1.0;      // < 0 means "default to z0"
    int         m_balance_flux = 0;

    // --- derived ---
    amrex::Real m_speed_ref = 0.0;       // hypot(u_ref, v_ref)
    amrex::Real m_dir_x = 0.0;           // unit vector along (u_ref, v_ref)
    amrex::Real m_dir_y = 0.0;

    std::vector<amrex::Real> m_xp, m_yp, m_zp, m_up, m_vp, m_wp;
    int m_n_columns = 0;

    amrex::MultiFab m_vel;

    // The balance of the field as it stands, and of the raw profile
    // before any redistribution. The two are the same object when
    // balancing is off.
    FluxBalance m_flux;
    FluxBalance m_flux_raw;
    amrex::Real m_balance_shift = 0.0;
};

} // namespace fwt

#endif
