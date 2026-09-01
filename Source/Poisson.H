#ifndef FWT_POISSON_H_
#define FWT_POISSON_H_

#include "Grid.H"
#include "Terrain.H"
#include "BoundaryConditions.H"

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
                const BoundaryConditions& bc);

    // Nodal RHS from the cell-centered velocity, in computational space.
    void ComputeRHS (const Grid& grid, const Terrain& terrain,
                     const amrex::MultiFab& vel);

    // Solves for lambda. Returns the final residual reported by MLMG.
    amrex::Real Solve ();

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
    void BuildSigma (const Grid& grid, const Terrain& terrain);
    void BuildOperator (const Grid& grid, const Terrain& terrain,
                        const BoundaryConditions& bc);

    amrex::Real m_alpha_h = 1.0;
    amrex::Real m_alpha_v = 1.0;
    int  m_max_iter = 200;
    int  m_verbose = 0;
    amrex::Real m_reltol = 1.0e-11;
    amrex::Real m_abstol = 0.0;

    amrex::MultiFab m_sigma;      // AMREX_SPACEDIM components, cell centered
    amrex::MultiFab m_rhs;        // nodal
    amrex::MultiFab m_lambda;     // nodal
    amrex::iMultiFab m_overset;   // nodal: which nodes are pinned

    long m_n_pinned = 0;
    bool m_all_dirichlet = false; // manufactured mode overrides the BCs

    std::unique_ptr<amrex::MLNodeLaplacian> m_op;
    amrex::BoxArray m_ba;
    amrex::DistributionMapping m_dm;
    amrex::Geometry m_geom;
};

} // namespace fwt

#endif
