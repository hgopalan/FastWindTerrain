#ifndef FWT_POISSON_H_
#define FWT_POISSON_H_

#include "Grid.H"
#include "Terrain.H"
#include "BoundaryConditions.H"
#include "Anisotropy.H"

#include <AMReX_MultiFab.H>
#include <AMReX_MLNodeLaplacian.H>

#include <memory>
#include <string>

namespace fwt {

// -----------------------------------------------------------------------
// Poisson: the anisotropic variational Poisson problem for the potential
// lambda, assembled on nodes.
//
//     div( sigma grad(lambda) ) = div(u0)
//     sigma = (alpha_h^2, alpha_h^2, alpha_v^2)
//
// with the velocity correction that follows in a later phase being
//
//     u = u0 - alpha_h^2 dlambda/dx     (and likewise v)
//     w = w0 - alpha_v^2 dlambda/dz
//
// so alpha is a TRANSMISSIVITY, not a penalty: a smaller alpha_v means
// less vertical adjustment, which is the sense the slope-suppression
// term needs. The operator coefficient and the correction multiplier are
// the same object, and a regtest checks that they stay that way.
//
// THE STRETCHED GRID. AMReX's Geometry is uniform in z, and the nodal
// operator (mlndlap_adotx_ha) builds its stencil from that uniform
// spacing with a per-cell, per-direction sigma. What AMReX calls
// "mapped" mode is not a coordinate mapping and carries no Jacobian --
// it is exactly a spatially varying diagonal coefficient. That is the
// hook the true spacing rides in on.
//
// Writing zeta for the computational vertical coordinate, whose spacing
// is the nominal Geometry dz, and
//
//     J(k) = dz(k) / dz_nominal
//
// for the (dimensionless) metric, the physical problem becomes, in
// computational space:
//
//     sigma_x' = alpha_h^2 * J        sigma_y' = alpha_h^2 * J
//     sigma_z' = alpha_v^2 / J
//     rhs'     = J * (du/dx + dv/dy) + dw/dzeta
//
// The J weighting is the cell volume in computational space, which is
// what keeps the assembled matrix symmetric, so MLMG still sees an SPD
// operator. This is derived, not borrowed: massconsistent_amr runs a
// uniform dz, and its terrain-following option puts metric terms in the
// divergence while leaving the operator coefficients untouched -- which
// would leave the two disagreeing about the coordinate system.
//
// Since it is derived, it is verified rather than assumed:
// poisson.manufactured solves a problem with a known analytic lambda on
// a deliberately stretched grid and reports the error, and the regtest
// checks the convergence rate. Wrong metric factors show up there as a
// lost order.
//
// THE IMMERSED BOUNDARY is no-flux, imposed as sigma = 0 in solid cells:
// zero conductivity carries no flux, which is the Neumann condition.
// Nodes surrounded entirely by solid cells then have an empty row, and
// are pinned via the overset mask rather than left singular.
// -----------------------------------------------------------------------
class Poisson
{
public:

    void Build (const Grid& grid, const Terrain& terrain,
                const BoundaryConditions& bc, const Anisotropy& aniso);

    // Nodal RHS from the cell-centered velocity, in computational space.
    void ComputeRHS (const Grid& grid, const Terrain& terrain,
                     const amrex::MultiFab& vel);

    // Solves for lambda. Returns the final residual reported by MLMG.
    amrex::Real Solve ();

    // Applies the variational correction in place:
    //
    //     u = u0 - alpha_h^2 dlambda/dx     (and likewise v)
    //     w = w0 - alpha_v^2 dlambda/dz
    //
    // lambda is nodal, so each cell-centered derivative is the average of
    // the four node differences spanning the cell in that direction. The
    // vertical one is divided by the cell's TRUE height, which is where
    // the stretched grid enters -- alpha_v^2 itself is the raw
    // coefficient, not the metric-weighted sigma the operator carries.
    //
    // Velocity is re-zeroed inside the terrain afterwards: the correction
    // is not defined there, and the immersed boundary has to hold after
    // the projection as well as before it.
    void ApplyCorrection (const Grid& grid, const Terrain& terrain,
                          amrex::MultiFab& vel) const;

    // vel -> (J u, J v, w), the computational flux variables the operator
    // and sigma are written in.
    void ScaleToComputational (const Grid& grid, const Terrain& terrain,
                               const amrex::MultiFab& vel,
                               amrex::MultiFab& q) const;

    // Largest |div(u)| over fluid cells, in physical units. Computed with
    // the configured scheme, so it measures the same discrete divergence
    // the RHS was built from.
    amrex::Real MaxDivergence (const Grid& grid, const Terrain& terrain,
                               const amrex::MultiFab& vel) const;

    // The divergence in the norm the projection actually controls, using
    // AMReX's own nodal divergence rather than a hand-rolled stand-in.
    amrex::Real MaxDivergenceFE (const Grid& grid, const Terrain& terrain,
                                 const amrex::MultiFab& vel);

    // Component-wise velocity extrema over fluid cells. A corrected wind
    // far larger than the one that went in is the loudest signal that a
    // setup is wrong, so it is measured and reported either side of the
    // projection.
    struct VelRange {
        amrex::Real lo[3];
        amrex::Real hi[3];
        amrex::Real speed_max;
    };
    static VelRange VelocityRange (const Terrain& terrain,
                                   const amrex::MultiFab& vel);

    void set_vel_before (const VelRange& r) { m_vel_before = r;
                                              m_speed_before = r.speed_max; }
    void set_vel_after (const VelRange& r)  { m_vel_after = r;
                                              m_speed_after = r.speed_max; }

    amrex::Real solve_residual () const { return m_resid; }

    const amrex::MultiFab& lambda () const { return m_lambda; }
    const amrex::MultiFab& rhs ()    const { return m_rhs; }
    // sigma carries the metric; component d is the coefficient for
    // direction d.
    const amrex::MultiFab& sigma ()  const { return m_sigma; }

    amrex::Real alpha_h () const { return m_alpha_h; }
    amrex::Real alpha_v () const { return m_alpha_v; }
    long n_pinned_nodes () const { return m_n_pinned; }

    void AppendReport (const std::string& filename) const;

    // Writes one row per node (i, j, k, rhs) so a regtest can check the
    // assembled RHS against an independently computed divergence.
    // Written only when poisson.rhs_dump_file is set; a single-rank test
    // aid, not a production output path.
    void WriteRHSDump (const std::string& filename) const;

    // Fills the RHS from an analytic source and reports the error of the
    // solve against a known lambda, so the discretization can be
    // measured instead of assumed. Returns {l2, linf} of the error.
    struct Error { amrex::Real l2; amrex::Real linf; };
    Error RunManufactured (const Grid& grid);

private:

    void ReadParameters ();
    // Clears the source at nodes buried in terrain, matching how
    // massconsistent_amr imposes the immersed boundary.
    void ZeroRHSInsideTerrain (const Terrain& terrain);
    void CorrectWithScheme (const Grid& grid, const Terrain& terrain,
                            amrex::MultiFab& vel) const;

    // The cell-local weights the operator was built from. Held so the
    // correction uses the same object the operator does -- if sigma
    // varies per cell and the correction does not, they are no longer
    // the same operator pair.
    const Anisotropy* m_aniso = nullptr;
    void BuildSigma (const Grid& grid, const Terrain& terrain,
                     const Anisotropy& aniso);
    void BuildOperator (const Grid& grid, const Terrain& terrain,
                        const BoundaryConditions& bc);

    amrex::Real m_alpha_h = 1.0;
    amrex::Real m_alpha_v = 1.0;
    int  m_max_iter = 200;
    int  m_verbose = 0;
    amrex::Real m_reltol = 1.0e-11;
    amrex::Real m_abstol = 0.0;
    VelRange m_vel_before {};
    VelRange m_vel_after {};
    amrex::Real m_speed_before = 0.0;
    amrex::Real m_speed_after = 0.0;
    // Multigrid struggles as the cell aspect ratio grows, and needs more
    // smoothing sweeps to compensate: about 2x the aspect ratio (8 at
    // AR 4, 16 at AR 8). Chosen from the grid unless set explicitly.
    int  m_pre_smooth = -1;
    int  m_post_smooth = -1;
    amrex::Real m_aspect = 1.0;
    // Which divergence forms the RHS: the FE-consistent nodal
    // divergence, or the configured scheme averaged to nodes.
    std::string m_rhs_operator = "fe";
    // Which lambda boundary conditions the operator uses. See
    // BuildOperator for why the directional mapping is not the default.
    std::string m_lambda_bc = "flowthrough";
    // Which gradient the correction uses: AMReX's own nodal gradient,
    // the one the operator was assembled from, or the configured
    // derivative scheme applied to lambda averaged to cell centres.
    std::string m_gradient_operator = "amrex";

    amrex::MultiFab m_sigma;      // AMREX_SPACEDIM components, cell centered
    amrex::MultiFab m_rhs;        // nodal
    amrex::MultiFab m_lambda;     // nodal
    amrex::iMultiFab m_overset;   // nodal: which nodes are pinned
    mutable amrex::MultiFab m_q;  // cell-centered computational flux vars

    long m_n_pinned = 0;
    long m_n_rhs_zeroed = 0;
    amrex::Real m_resid = 0.0;
    // Divergence before and after the correction, for the report.
    mutable amrex::Real m_div_before = 0.0;
    mutable amrex::Real m_div_after = 0.0;
    // The divergence in the norm the solve actually controls, which is
    // the one that measures whether the projection worked.
    mutable amrex::Real m_div_fe_before = 0.0;
    mutable amrex::Real m_div_fe_after = 0.0;
    mutable int m_n_projections = 1;

public:
    void set_div_before (amrex::Real d) const { m_div_before = d; }
    void set_div_after (amrex::Real d) const { m_div_after = d; }
    void set_div_fe (amrex::Real b, amrex::Real a) const
        { m_div_fe_before = b; m_div_fe_after = a; }
    void set_n_projections (int n) const { m_n_projections = n; }
    amrex::Real div_before () const { return m_div_before; }
    amrex::Real div_after () const { return m_div_after; }
private:
    bool m_all_dirichlet = false; // manufactured mode overrides the BCs

    std::unique_ptr<amrex::MLNodeLaplacian> m_op;
    amrex::BoxArray m_ba;
    amrex::DistributionMapping m_dm;
    amrex::Geometry m_geom;
};

} // namespace fwt

#endif
