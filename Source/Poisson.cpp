#include "Poisson.H"
#include "Debug.H"
#include "Derivatives.H"

#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_MLMG.H>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>

namespace fwt {

namespace {

// Metric J(k) = dz(k) / dz_nominal, dimensionless. The nominal spacing
// is the one AMReX's Geometry believes in.
amrex::Vector<amrex::Real> CellMetric (const Grid& grid)
{
    const int nz = grid.nz();
    const amrex::Real h = grid.geom().CellSize(2);
    const amrex::Vector<amrex::Real>& zf = grid.z_face();

    amrex::Vector<amrex::Real> J(nz);
    for (int k = 0; k < nz; ++k) { J[k] = (zf[k+1] - zf[k]) / h; }
    return J;
}

// The same metric at nodes, which sit between cells k-1 and k.
amrex::Vector<amrex::Real> NodeMetric (const Grid& grid)
{
    const amrex::Vector<amrex::Real> Jc = CellMetric(grid);
    const int nz = grid.nz();

    amrex::Vector<amrex::Real> J(nz + 1);
    J[0] = Jc[0];
    J[nz] = Jc[nz-1];
    for (int k = 1; k < nz; ++k) { J[k] = 0.5 * (Jc[k-1] + Jc[k]); }
    return J;
}

amrex::LinOpBCType ToLinOpBC (BoundaryConditions::LambdaBC b)
{
    return (b == BoundaryConditions::LambdaBC::dirichlet)
         ? amrex::LinOpBCType::Dirichlet
         : amrex::LinOpBCType::Neumann;
}

} // namespace

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

void Poisson::ReadParameters ()
{
    amrex::ParmParse pp("poisson");

    const bool got_ah = pp.query("alpha_h", m_alpha_h);
    const bool got_av = pp.query("alpha_v", m_alpha_v);
    pp.query("max_iter", m_max_iter);
    pp.query("verbose", m_verbose);
    pp.query("reltol", m_reltol);
    pp.query("abstol", m_abstol);
    pp.query("num_pre_smooth", m_pre_smooth);
    pp.query("num_post_smooth", m_post_smooth);
    pp.query("rhs_operator", m_rhs_operator);
    pp.query("lambda_bc", m_lambda_bc);
    pp.query("gradient_operator", m_gradient_operator);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_gradient_operator == "amrex" || m_gradient_operator == "scheme",
        "poisson.gradient_operator must be 'amrex' or 'scheme'");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_lambda_bc == "flowthrough" || m_lambda_bc == "directional",
        "poisson.lambda_bc must be 'flowthrough' or 'directional'");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_rhs_operator == "fe" || m_rhs_operator == "scheme",
        "poisson.rhs_operator must be 'fe' or 'scheme'");

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_alpha_h > 0.0,
        "poisson.alpha_h must be > 0");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_alpha_v > 0.0,
        "poisson.alpha_v must be > 0");

    FWT_DEBUG_SECTION("Poisson inputs (poisson.*)");
    FWT_DEBUG("alpha_h          = " << m_alpha_h
                                     << (got_ah ? "" : "   [default]"));
    FWT_DEBUG("alpha_v          = " << m_alpha_v
                                     << (got_av ? "" : "   [default]"));
    FWT_DEBUG("alpha is a transmissivity: the correction multiplies "
              "grad(lambda) by alpha^2, so a smaller alpha_v means less "
              "vertical adjustment");
    FWT_DEBUG("max_iter         = " << m_max_iter);
    FWT_DEBUG("reltol           = " << m_reltol);
}

// ---------------------------------------------------------------------------
// Sigma, carrying the vertical metric
// ---------------------------------------------------------------------------

void Poisson::BuildSigma (const Grid& grid, const Terrain& terrain,
                          const Anisotropy& aniso)
{
    m_sigma.define(grid.ba(), grid.dm(), AMREX_SPACEDIM, 0);

    const amrex::Vector<amrex::Real> Jc = CellMetric(grid);
    amrex::Gpu::DeviceVector<amrex::Real> d_J(Jc.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, Jc.begin(), Jc.end(),
                          d_J.begin());
    amrex::Gpu::streamSynchronize();
    const amrex::Real* J = d_J.data();

    const int solid = Terrain::kSolid;

    for (amrex::MFIter mfi(m_sigma); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const& s  = m_sigma.array(mfi);
        auto const& mk = terrain.mask().const_array(mfi);
        // Cell-local weights. With anisotropy disabled these hold the
        // base values, so the operator is exactly what it was before.
        auto const& ah = aniso.alpha_h().const_array(mfi);
        auto const& av = aniso.alpha_v().const_array(mfi);
        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // The metric rides in on the coefficients: horizontal terms
            // are weighted by the cell's true height, the vertical term
            // by its inverse.
            //
            // Sigma is NOT masked inside the terrain. Zeroing it there
            // makes no-flux exact in the operator, but leaves any node
            // buried entirely in terrain with a zero diagonal, which the
            // multigrid smoother divides by -- producing NaN and a solve
            // that silently reports a zero residual. massconsistent_amr
            // leaves its coefficients unmasked for the same reason and
            // imposes the immersed boundary on the FIELD instead: the
            // divergence is zeroed in solid cells when the RHS is built,
            // and the velocity is re-zeroed there after the correction.
            // Lambda is then solved inside the terrain too, where it is
            // meaningless but harmless.
            amrex::ignore_unused(mk, solid);
            const amrex::Real ah2 = ah(i,j,k) * ah(i,j,k);
            const amrex::Real av2 = av(i,j,k) * av(i,j,k);
            s(i,j,k,0) = ah2 * J[k];
            s(i,j,k,1) = ah2 * J[k];
            s(i,j,k,2) = av2 / J[k];
        });
    }
}

// ---------------------------------------------------------------------------
// Operator
// ---------------------------------------------------------------------------

void Poisson::BuildOperator (const Grid& grid, const Terrain& terrain,
                             const BoundaryConditions& bc)
{
    amrex::LPInfo info;
    info.setAgglomeration(true);
    info.setConsolidation(true);

    m_op = std::make_unique<amrex::MLNodeLaplacian>(
        amrex::Vector<amrex::Geometry>{m_geom},
        amrex::Vector<amrex::BoxArray>{m_ba},
        amrex::Vector<amrex::DistributionMapping>{m_dm}, info);
    m_op->setMaxOrder(2);

    amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> lo_bc, hi_bc;
    if (m_all_dirichlet) {
        // The manufactured solution vanishes on every face, so the
        // discretization can be measured without the boundary treatment
        // in the way.
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            lo_bc[d] = amrex::LinOpBCType::Dirichlet;
            hi_bc[d] = amrex::LinOpBCType::Dirichlet;
        }
    } else if (m_lambda_bc == "flowthrough") {
        // The classical mass-consistent convention: lambda = 0 on every
        // flow-through boundary, Neumann where nothing flows through
        // (ground and domain top). Fixed, not derived from the wind.
        //
        // massconsistent_amr fixes its lambda conditions the same way
        // (x Dirichlet, y and z Neumann). All four laterals are Dirichlet
        // here because of the nodal operator, for the reason below.
        //
        // Deriving it from the wind instead -- Neumann on whichever faces
        // the flow enters -- interacts badly with the NODAL divergence.
        // mlndlap_divu deliberately does not see the tangential velocity
        // at a face it considers inflow, and with an oblique wind those
        // faces carry a large tangential component, so zeroing it
        // manufactures an enormous artificial divergence: measured at
        // 6.92 against 0.0216 for the same case with a fixed convention,
        // and a corrected wind of 34.8 m/s against 18.9 m/s from a 10 m/s
        // inflow. massconsistent_amr never meets this because its
        // operator is cell-centered.
        //
        // The Phase 4 classification still governs the VELOCITY boundary
        // conditions, which is where the wind direction genuinely
        // belongs; it just no longer sets the lambda conditions.
        lo_bc[0] = amrex::LinOpBCType::Dirichlet;
        hi_bc[0] = amrex::LinOpBCType::Dirichlet;
        lo_bc[1] = amrex::LinOpBCType::Dirichlet;
        hi_bc[1] = amrex::LinOpBCType::Dirichlet;
        lo_bc[2] = amrex::LinOpBCType::Neumann;   // ground
        hi_bc[2] = amrex::LinOpBCType::Neumann;   // domain top, w = 0
    } else {
        // Face order is xlo, xhi, ylo, yhi, zlo, zhi.
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            lo_bc[d] = ToLinOpBC(bc.lambda_bc(2*d));
            hi_bc[d] = ToLinOpBC(bc.lambda_bc(2*d + 1));
        }
    }
    m_op->setDomainBC(lo_bc, hi_bc);

    // No overset mask is needed. With sigma left elliptic everywhere
    // there are no empty rows, so no node has to be pinned.
    m_op->setSigma(0, m_sigma);
}

void Poisson::Build (const Grid& grid, const Terrain& terrain,
                     const BoundaryConditions& bc, const Anisotropy& aniso)
{
    ReadParameters();
    m_aniso = &aniso;

    {
        amrex::ParmParse pp("poisson");
        int mms = 0;
        pp.query("manufactured", mms);
        int forced = 0;
        pp.query("force_all_dirichlet", forced);
        m_all_dirichlet = (mms != 0) || (forced != 0);
    }

    m_ba = grid.ba();
    m_dm = grid.dm();
    m_geom = grid.geom();

    const amrex::BoxArray nba =
        amrex::convert(m_ba, amrex::IntVect::TheNodeVector());
    m_rhs.define(nba, m_dm, 1, 0);
    m_lambda.define(nba, m_dm, 1, 1);
    m_rhs.setVal(0.0);
    m_lambda.setVal(0.0);

    // Multigrid convergence degrades as cells get more anisotropic,
    // and the cure is more smoothing sweeps: roughly twice the aspect
    // ratio (8 sweeps at 4:1, 16 at 8:1). The surface layer here is the
    // worst case, being the thinnest.
    {
        const amrex::Real dx = grid.geom().CellSize(0);
        const amrex::Real dy = grid.geom().CellSize(1);
        amrex::Real worst = 1.0;
        const amrex::Vector<amrex::Real>& zf = grid.z_face();
        for (int k = 0; k < grid.nz(); ++k) {
            const amrex::Real dz = zf[k+1] - zf[k];
            worst = std::max(worst, std::max(dx, dy) / dz);
            worst = std::max(worst, dz / std::min(dx, dy));
        }
        m_aspect = worst;

        const int autos = std::min(32,
            std::max(2, 2 * static_cast<int>(std::ceil(worst))));
        if (m_pre_smooth  < 0) { m_pre_smooth  = autos; }
        if (m_post_smooth < 0) { m_post_smooth = autos; }
    }

    BuildSigma(grid, terrain, aniso);
    BuildOperator(grid, terrain, bc);

    if (Debug::Enabled()) {
        const amrex::Vector<amrex::Real> Jc = CellMetric(grid);
        FWT_DEBUG_SECTION("Poisson operator");
        FWT_DEBUG("sigma            = (alpha_h^2 J, alpha_h^2 J, "
                  "alpha_v^2 / J)");
        FWT_DEBUG("metric J         = dz(k) / dz_nominal, from "
                  << Jc.front() << " to " << Jc.back());
        FWT_DEBUG("dz_nominal       = " << grid.geom().CellSize(2) << " m");
        FWT_DEBUG("pinned nodes     = " << m_n_pinned
                  << "   (surrounded entirely by solid cells)");
        FWT_DEBUG("cell aspect ratio= " << m_aspect
                  << "   -> smoothing sweeps " << m_pre_smooth << "/"
                  << m_post_smooth);
        FWT_DEBUG("rhs_operator     = " << m_rhs_operator);
        FWT_DEBUG("all-Dirichlet    = " << (m_all_dirichlet ? "yes "
                  "(manufactured mode)" : "no (from the boundary "
                  "conditions)"));
    }
}

// ---------------------------------------------------------------------------
// RHS
// ---------------------------------------------------------------------------

void Poisson::ComputeRHS (const Grid& grid, const Terrain& terrain,
                          const amrex::MultiFab& vel)
{
    // Cell-centered divergence first, with the configured scheme, then
    // averaged to nodes and weighted by the metric. In computational
    // space the vertical term needs no metric of its own:
    //
    //   rhs' = J (du/dx + dv/dy) + dw/dzeta
    //
    // because J dw/dz_phys is exactly dw/dzeta.
    const int nz = grid.nz();
    const amrex::Real dx = grid.geom().CellSize(0);
    const amrex::Real dy = grid.geom().CellSize(1);
    const amrex::Real hz = grid.geom().CellSize(2);
    const amrex::Vector<amrex::Real>& z_cc = grid.z_cc();
    const amrex::Vector<amrex::Real> Jc = CellMetric(grid);

    // Vertical metric for the derivative stencil: dz/dk at cell centers.
    amrex::Vector<amrex::Real> dzdk(nz);
    for (int k = 0; k < nz; ++k) {
        if (k == 0)            { dzdk[k] = z_cc[1] - z_cc[0]; }
        else if (k == nz - 1)  { dzdk[k] = z_cc[nz-1] - z_cc[nz-2]; }
        else                   { dzdk[k] = 0.5 * (z_cc[k+1] - z_cc[k-1]); }
    }

    // Cell-centered divergence, computational-space weighted. One ghost
    // layer, because the node averaging below reaches one cell past each
    // box; ghosts outside the domain stay zero and are skipped there.
    amrex::MultiFab divc(m_ba, m_dm, 1, 1);
    divc.setVal(0.0);
    const Scheme scheme = Numerics::scheme();
    const int solid = Terrain::kSolid;

    amrex::Gpu::DeviceVector<amrex::Real> d_dzdk(dzdk.size()), d_J(Jc.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, dzdk.begin(), dzdk.end(),
                          d_dzdk.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, Jc.begin(), Jc.end(),
                          d_J.begin());
    amrex::Gpu::streamSynchronize();
    const amrex::Real* pdzdk = d_dzdk.data();
    const amrex::Real* pJ = d_J.data();

    const amrex::Box& dom = m_geom.Domain();
    const int klo = dom.smallEnd(2), khi = dom.bigEnd(2);

    for (amrex::MFIter mfi(divc); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const& d  = divc.array(mfi);
        auto const& v  = vel.const_array(mfi);
        auto const& mk = terrain.mask().const_array(mfi);
        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            if (mk(i,j,k) == solid) { d(i,j,k) = 0.0; return; }

            const amrex::Real dudx = Derivative(scheme,
                v(i-2,j,k,0), v(i-1,j,k,0), v(i,j,k,0),
                v(i+1,j,k,0), v(i+2,j,k,0), v(i,j,k,0), dx);
            const amrex::Real dvdy = Derivative(scheme,
                v(i,j-2,k,1), v(i,j-1,k,1), v(i,j,k,1),
                v(i,j+1,k,1), v(i,j+2,k,1), v(i,j,k,1), dy);

            // The vertical stencil is clamped at the domain top and
            // bottom, where the ghost layers hold the reflected values
            // rather than a continuation of the field.
            const int km2 = amrex::max(k-2, klo), km1 = amrex::max(k-1, klo);
            const int kp1 = amrex::min(k+1, khi), kp2 = amrex::min(k+2, khi);
            const amrex::Real dwdz = Derivative(scheme,
                v(i,j,km2,2), v(i,j,km1,2), v(i,j,k,2),
                v(i,j,kp1,2), v(i,j,kp2,2), v(i,j,k,2), pdzdk[k]);

            d(i,j,k) = pJ[k] * (dudx + dvdy) + pJ[k] * dwdz;
        });
    }

    amrex::ignore_unused(hz);

    divc.FillBoundary(m_geom.periodicity());

    if (m_rhs_operator == "fe") {
        // Use AMReX's OWN divergence rather than a hand-rolled one. The
        // operator it assembles is D sigma G for a specific trilinear
        // finite-element D and G, and only that exact pair makes the
        // projection exact. A plausible-looking four-point average is a
        // different operator, and leaves nearly all of the divergence
        // behind -- measured, not assumed.
        //
        // D acts on (J u, J v, w): the computational flux vector whose
        // divergence is J times the physical one, matching the weighting
        // sigma already carries. updateVelocity returns the correction in
        // the same variables, and it is unscaled afterwards.
        ScaleToComputational(grid, terrain, vel, m_q);
        m_op->compDivergence({&m_rhs}, {&m_q});
        ZeroRHSInsideTerrain(terrain);   // IB on the source, not the operator

        FWT_DEBUG("RHS assembled (fe, AMReX compDivergence): nodal, min "
                  << m_rhs.min(0) << " max " << m_rhs.max(0));
        return;
    }

    // Average to nodes over whichever of the eight surrounding cells lie
    // inside the domain.
    m_rhs.setVal(0.0);
    for (amrex::MFIter mfi(m_rhs); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const& r = m_rhs.array(mfi);
        auto const& d = divc.const_array(mfi);

        amrex::LoopOnCpu(bx, [&] (int i, int j, int k) noexcept
        {
            amrex::Real sum = 0.0;
            int n = 0;
            for (int kk = k-1; kk <= k; ++kk) {
            for (int jj = j-1; jj <= j; ++jj) {
            for (int ii = i-1; ii <= i; ++ii) {
                if (ii < dom.smallEnd(0) || ii > dom.bigEnd(0) ||
                    jj < dom.smallEnd(1) || jj > dom.bigEnd(1) ||
                    kk < dom.smallEnd(2) || kk > dom.bigEnd(2)) { continue; }
                sum += d(ii,jj,kk);
                ++n;
            }}}
            r(i,j,k) = (n > 0) ? sum / amrex::Real(n) : 0.0;
        });
    }

    ZeroRHSInsideTerrain(terrain);   // IB on the source, not the operator

    FWT_DEBUG("RHS assembled: nodal, min " << m_rhs.min(0)
              << " max " << m_rhs.max(0));
}

// ---------------------------------------------------------------------------
// Solve
// ---------------------------------------------------------------------------

amrex::Real Poisson::Solve ()
{
    amrex::MLMG mlmg(*m_op);
    mlmg.setMaxIter(m_max_iter);
    mlmg.setVerbose(m_verbose);
    mlmg.setPreSmooth(m_pre_smooth);
    mlmg.setPostSmooth(m_post_smooth);

    amrex::Vector<amrex::MultiFab*> sol {&m_lambda};
    amrex::Vector<const amrex::MultiFab*> rhs {&m_rhs};

    const amrex::Real resid = mlmg.solve(sol, rhs, m_reltol, m_abstol);
    m_resid = resid;
    m_lambda.FillBoundary(m_geom.periodicity());
    FWT_DEBUG("MLMG solve: residual " << resid
              << ", lambda in [" << m_lambda.min(0) << ", "
              << m_lambda.max(0) << "]"
              << (m_lambda.contains_nan() ? "  CONTAINS NaN" : ""));
    return resid;
}

// ---------------------------------------------------------------------------
// Velocity correction
// ---------------------------------------------------------------------------

void Poisson::ScaleToComputational (const Grid& grid, const Terrain& terrain,
                                    const amrex::MultiFab& vel,
                                    amrex::MultiFab& q) const
{
    const amrex::Real hz = grid.geom().CellSize(2);
    const amrex::Vector<amrex::Real>& zf = grid.z_face();
    const int nz = grid.nz();

    amrex::Vector<amrex::Real> Jc(nz);
    for (int k = 0; k < nz; ++k) { Jc[k] = (zf[k+1] - zf[k]) / hz; }
    amrex::Gpu::DeviceVector<amrex::Real> d_J(Jc.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, Jc.begin(), Jc.end(),
                          d_J.begin());
    amrex::Gpu::streamSynchronize();
    const amrex::Real* pJ = d_J.data();

    const int solid = Terrain::kSolid;
    if (!q.ok()) { q.define(m_ba, m_dm, 3, 1); }
    q.setVal(0.0);

    for (amrex::MFIter mfi(q); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const& a = q.array(mfi);
        auto const& v = vel.const_array(mfi);
        auto const& mk = terrain.mask().const_array(mfi);
        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const amrex::Real f = (mk(i,j,k) == solid) ? 0.0 : 1.0;
            a(i,j,k,0) = f * pJ[k] * v(i,j,k,0);
            a(i,j,k,1) = f * pJ[k] * v(i,j,k,1);
            a(i,j,k,2) = f * v(i,j,k,2);
        });
    }
    q.FillBoundary(m_geom.periodicity());
}

void Poisson::ApplyCorrection (const Grid& grid, const Terrain& terrain,
                               amrex::MultiFab& vel) const
{
    if (m_gradient_operator == "scheme") {
        CorrectWithScheme(grid, terrain, vel);
        return;
    }

    // The default. The correction is the gradient AMReX's own operator
    // implies, for the same reason the RHS is its divergence: that pair
    // is what the solve is built from. updateVelocity applies
    // q -= sigma grad(lambda).
    ScaleToComputational(grid, terrain, vel, m_q);
    m_op->updateVelocity({&m_q}, {&m_lambda});

    const amrex::Real hz = grid.geom().CellSize(2);
    const amrex::Vector<amrex::Real>& zf = grid.z_face();
    const int nz = grid.nz();
    amrex::Vector<amrex::Real> Jc(nz);
    for (int k = 0; k < nz; ++k) { Jc[k] = (zf[k+1] - zf[k]) / hz; }
    amrex::Gpu::DeviceVector<amrex::Real> d_J(Jc.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, Jc.begin(), Jc.end(),
                          d_J.begin());
    amrex::Gpu::streamSynchronize();
    const amrex::Real* pJ = d_J.data();

    const int solid = Terrain::kSolid;
    for (amrex::MFIter mfi(vel); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const& v = vel.array(mfi);
        auto const& a = m_q.const_array(mfi);
        auto const& mk = terrain.mask().const_array(mfi);
        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            if (mk(i,j,k) == solid) {
                // No flow inside the terrain, after the projection as
                // well as before it.
                v(i,j,k,0) = 0.0;
                v(i,j,k,1) = 0.0;
                v(i,j,k,2) = 0.0;
                return;
            }
            // Back out of the computational variables.
            v(i,j,k,0) = a(i,j,k,0) / pJ[k];
            v(i,j,k,1) = a(i,j,k,1) / pJ[k];
            v(i,j,k,2) = a(i,j,k,2);
        });
    }
}

// The alternative correction: average lambda to cell centres and take
// its gradient with the configured derivative scheme, which is how
// massconsistent_amr does it (its lambda is cell-centered to begin
// with). Offered because it is the familiar formulation, but it is not
// the gradient the nodal operator was assembled from, so the projection
// is looser -- the regtest reports both.
void Poisson::CorrectWithScheme (const Grid& grid, const Terrain& terrain,
                                 amrex::MultiFab& vel) const
{
    const amrex::Real dx = grid.geom().CellSize(0);
    const amrex::Real dy = grid.geom().CellSize(1);
    const amrex::Vector<amrex::Real>& z_cc = grid.z_cc();
    const int nz = grid.nz();

    // lambda at cell centres: the average of the eight surrounding nodes.
    amrex::MultiFab lcc(m_ba, m_dm, 1, 2);
    lcc.setVal(0.0);
    for (amrex::MFIter mfi(lcc); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const& c = lcc.array(mfi);
        auto const& l = m_lambda.const_array(mfi);
        amrex::LoopOnCpu(bx, [&] (int i, int j, int k) noexcept
        {
            amrex::Real sum = 0.0;
            for (int kk = 0; kk <= 1; ++kk) {
            for (int jj = 0; jj <= 1; ++jj) {
            for (int ii = 0; ii <= 1; ++ii) {
                sum += l(i+ii,j+jj,k+kk);
            }}}
            c(i,j,k) = 0.125 * sum;
        });
    }
    lcc.FillBoundary(m_geom.periodicity());

    amrex::Vector<amrex::Real> dzdk(nz);
    for (int k = 0; k < nz; ++k) {
        if (k == 0)           { dzdk[k] = z_cc[1] - z_cc[0]; }
        else if (k == nz - 1) { dzdk[k] = z_cc[nz-1] - z_cc[nz-2]; }
        else                  { dzdk[k] = 0.5 * (z_cc[k+1] - z_cc[k-1]); }
    }
    amrex::Gpu::DeviceVector<amrex::Real> d_dzdk(dzdk.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, dzdk.begin(), dzdk.end(),
                          d_dzdk.begin());
    amrex::Gpu::streamSynchronize();
    const amrex::Real* pdzdk = d_dzdk.data();

    const Scheme scheme = Numerics::scheme();
    const int solid = Terrain::kSolid;
    const amrex::Box& dom = m_geom.Domain();
    const int klo = dom.smallEnd(2), khi = dom.bigEnd(2);
    const int ilo = dom.smallEnd(0), ihi = dom.bigEnd(0);
    const int jlo = dom.smallEnd(1), jhi = dom.bigEnd(1);

    for (amrex::MFIter mfi(vel); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const& v  = vel.array(mfi);
        auto const& c  = lcc.const_array(mfi);
        auto const& mk = terrain.mask().const_array(mfi);
        // The same cell-local weights sigma was built from.
        auto const& ah = m_aniso->alpha_h().const_array(mfi);
        auto const& av = m_aniso->alpha_v().const_array(mfi);
        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            if (mk(i,j,k) == solid) {
                v(i,j,k,0) = 0.0; v(i,j,k,1) = 0.0; v(i,j,k,2) = 0.0;
                return;
            }
            const int km2 = amrex::max(k-2, klo), km1 = amrex::max(k-1, klo);
            const int kp1 = amrex::min(k+1, khi), kp2 = amrex::min(k+2, khi);

            // No advecting velocity for an elliptic field, so the schemes
            // fall back to their central form.
            // Clamp to the domain. lcc's physical-boundary ghosts are
            // only zeroed, never filled, so an unclamped stencil would
            // quietly differentiate against lambda = 0 outside the
            // domain -- wrong, and silent.
            const int im2 = amrex::max(i-2, ilo), im1 = amrex::max(i-1, ilo);
            const int ip1 = amrex::min(i+1, ihi), ip2 = amrex::min(i+2, ihi);
            const int jm2 = amrex::max(j-2, jlo), jm1 = amrex::max(j-1, jlo);
            const int jp1 = amrex::min(j+1, jhi), jp2 = amrex::min(j+2, jhi);

            const amrex::Real gx = Derivative(scheme,
                c(im2,j,k), c(im1,j,k), c(i,j,k), c(ip1,j,k), c(ip2,j,k),
                0.0, dx);
            const amrex::Real gy = Derivative(scheme,
                c(i,jm2,k), c(i,jm1,k), c(i,j,k), c(i,jp1,k), c(i,jp2,k),
                0.0, dy);
            const amrex::Real gz = Derivative(scheme,
                c(i,j,km2), c(i,j,km1), c(i,j,k), c(i,j,kp1), c(i,j,kp2),
                0.0, pdzdk[k]);

            const amrex::Real ah2 = ah(i,j,k) * ah(i,j,k);
            const amrex::Real av2 = av(i,j,k) * av(i,j,k);
            v(i,j,k,0) -= ah2 * gx;
            v(i,j,k,1) -= ah2 * gy;
            v(i,j,k,2) -= av2 * gz;
        });
    }
}

// ---------------------------------------------------------------------------
// Immersed boundary: mask the source, not the operator
// ---------------------------------------------------------------------------

// massconsistent_amr imposes the immersed boundary entirely on the
// field, never on the operator coefficients: the divergence is set to
// zero in solid cells before the solve, and the velocity is re-zeroed
// there afterwards. The nodal analogue of "solid cell" is a node with no
// fluid cell around it at all, so interface nodes keep their source and
// only nodes buried in terrain are cleared.
void Poisson::ZeroRHSInsideTerrain (const Terrain& terrain)
{
    const int solid = Terrain::kSolid;

    // A nodal box reaches one cell past its cell-centered counterpart, so
    // the mask needs a ghost layer; outside the domain counts as solid.
    amrex::iMultiFab maskg(m_ba, m_dm, 1, 1);
    maskg.setVal(solid);
    amrex::iMultiFab::Copy(maskg, terrain.mask(), 0, 0, 1, 0);
    maskg.FillBoundary(m_geom.periodicity());

    const amrex::Box& dom = m_geom.Domain();
    long n_zeroed = 0;

    for (amrex::MFIter mfi(m_rhs); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const& r  = m_rhs.array(mfi);
        auto const& mk = maskg.const_array(mfi);

        amrex::LoopOnCpu(bx, [&] (int i, int j, int k) noexcept
        {
            bool any_fluid = false;
            for (int kk = k-1; kk <= k; ++kk) {
            for (int jj = j-1; jj <= j; ++jj) {
            for (int ii = i-1; ii <= i; ++ii) {
                if (ii < dom.smallEnd(0) || ii > dom.bigEnd(0) ||
                    jj < dom.smallEnd(1) || jj > dom.bigEnd(1) ||
                    kk < dom.smallEnd(2) || kk > dom.bigEnd(2)) { continue; }
                if (mk(ii,jj,kk) != solid) { any_fluid = true; }
            }}}
            if (!any_fluid) { r(i,j,k) = 0.0; ++n_zeroed; }
        });
    }
    amrex::ParallelDescriptor::ReduceLongSum(n_zeroed);
    m_n_rhs_zeroed = n_zeroed;

    FWT_DEBUG("RHS zeroed at " << n_zeroed << " nodes buried in terrain");
}

// ---------------------------------------------------------------------------
// Velocity extrema
// ---------------------------------------------------------------------------

// Component-wise extrema over FLUID cells. A projection that has gone
// wrong shows up here long before anything subtle does: a corrected wind
// far larger than the one that went in means the setup is wrong, whatever
// the residual says.
Poisson::VelRange Poisson::VelocityRange (const Terrain& terrain,
                                          const amrex::MultiFab& vel)
{
    VelRange r;
    for (int n = 0; n < 3; ++n) {
        r.lo[n] =  std::numeric_limits<amrex::Real>::max();
        r.hi[n] = -std::numeric_limits<amrex::Real>::max();
    }
    r.speed_max = 0.0;

    const int solid = Terrain::kSolid;
    for (amrex::MFIter mfi(vel); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const& v  = vel.const_array(mfi);
        auto const& mk = terrain.mask().const_array(mfi);
        amrex::LoopOnCpu(bx, [&] (int i, int j, int k) noexcept
        {
            if (mk(i,j,k) == solid) { return; }
            amrex::Real s2 = 0.0;
            for (int n = 0; n < 3; ++n) {
                const amrex::Real c = v(i,j,k,n);
                r.lo[n] = std::min(r.lo[n], c);
                r.hi[n] = std::max(r.hi[n], c);
                s2 += c * c;
            }
            r.speed_max = std::max(r.speed_max, std::sqrt(s2));
        });
    }
    for (int n = 0; n < 3; ++n) {
        amrex::ParallelDescriptor::ReduceRealMin(r.lo[n]);
        amrex::ParallelDescriptor::ReduceRealMax(r.hi[n]);
    }
    amrex::ParallelDescriptor::ReduceRealMax(r.speed_max);
    return r;
}

// ---------------------------------------------------------------------------
// Divergence diagnostics
// ---------------------------------------------------------------------------

// The divergence in the norm the projection actually controls: AMReX's
// own nodal divergence, the same operator that built the RHS. Anything
// hand-rolled here measures a different operator -- an earlier version
// of this function did, and disagreed with the real one by a factor of
// thirteen, which made a working projection look broken.
amrex::Real Poisson::MaxDivergenceFE (const Grid& grid,
                                      const Terrain& terrain,
                                      const amrex::MultiFab& vel)
{
    ScaleToComputational(grid, terrain, vel, m_q);

    amrex::MultiFab d(amrex::convert(m_ba, amrex::IntVect::TheNodeVector()),
                      m_dm, 1, 0);
    d.setVal(0.0);
    m_op->compDivergence({&d}, {&m_q});

    return std::max(std::abs(d.min(0)), std::abs(d.max(0)));
}

amrex::Real Poisson::MaxDivergence (const Grid& grid, const Terrain& terrain,
                                    const amrex::MultiFab& vel) const
{
    const amrex::Real dx = grid.geom().CellSize(0);
    const amrex::Real dy = grid.geom().CellSize(1);
    const amrex::Vector<amrex::Real>& z_cc = grid.z_cc();
    const int nz = grid.nz();

    amrex::Vector<amrex::Real> dzdk(nz);
    for (int k = 0; k < nz; ++k) {
        if (k == 0)           { dzdk[k] = z_cc[1] - z_cc[0]; }
        else if (k == nz - 1) { dzdk[k] = z_cc[nz-1] - z_cc[nz-2]; }
        else                  { dzdk[k] = 0.5 * (z_cc[k+1] - z_cc[k-1]); }
    }
    amrex::Gpu::DeviceVector<amrex::Real> d_dzdk(dzdk.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, dzdk.begin(), dzdk.end(),
                          d_dzdk.begin());
    amrex::Gpu::streamSynchronize();
    const amrex::Real* pdzdk = d_dzdk.data();

    const Scheme scheme = Numerics::scheme();
    const int solid = Terrain::kSolid;
    const amrex::Box& dom = m_geom.Domain();
    const int klo = dom.smallEnd(2), khi = dom.bigEnd(2);

    amrex::Real worst = 0.0;
    for (amrex::MFIter mfi(vel); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const& v  = vel.const_array(mfi);
        auto const& mk = terrain.mask().const_array(mfi);

        amrex::LoopOnCpu(bx, [&] (int i, int j, int k) noexcept
        {
            if (mk(i,j,k) == solid) { return; }
            const int km2 = amrex::max(k-2, klo), km1 = amrex::max(k-1, klo);
            const int kp1 = amrex::min(k+1, khi), kp2 = amrex::min(k+2, khi);

            const amrex::Real d =
                Derivative(scheme, v(i-2,j,k,0), v(i-1,j,k,0), v(i,j,k,0),
                           v(i+1,j,k,0), v(i+2,j,k,0), v(i,j,k,0), dx)
              + Derivative(scheme, v(i,j-2,k,1), v(i,j-1,k,1), v(i,j,k,1),
                           v(i,j+1,k,1), v(i,j+2,k,1), v(i,j,k,1), dy)
              + Derivative(scheme, v(i,j,km2,2), v(i,j,km1,2), v(i,j,k,2),
                           v(i,j,kp1,2), v(i,j,kp2,2), v(i,j,k,2), pdzdk[k]);
            worst = std::max(worst, std::abs(d));
        });
    }
    amrex::ParallelDescriptor::ReduceRealMax(worst);
    return worst;
}

// ---------------------------------------------------------------------------
// Manufactured solution
// ---------------------------------------------------------------------------

Poisson::Error Poisson::RunManufactured (const Grid& grid)
{
    // lambda = sin(pi x/Lx) sin(pi y/Ly) sin(pi z/Lz) vanishes on every
    // face, so homogeneous Dirichlet is exact and the boundary treatment
    // cannot mask an error in the interior discretization.
    //
    //   div(sigma grad lambda)
    //     = -[ah^2 (pi/Lx)^2 + ah^2 (pi/Ly)^2 + av^2 (pi/Lz)^2] lambda
    //
    // The RHS is then set analytically, in computational space, so this
    // measures the operator and the metric and nothing else.
    const amrex::Real xlo = grid.geom().ProbLo(0);
    const amrex::Real ylo = grid.geom().ProbLo(1);
    const amrex::Real zlo = grid.geom().ProbLo(2);
    const amrex::Real Lx = grid.geom().ProbHi(0) - xlo;
    const amrex::Real Ly = grid.geom().ProbHi(1) - ylo;
    const amrex::Real Lz = grid.geom().ProbHi(2) - zlo;
    const amrex::Real dx = grid.geom().CellSize(0);
    const amrex::Real dy = grid.geom().CellSize(1);

    const amrex::Real kx = M_PI / Lx, ky = M_PI / Ly, kz = M_PI / Lz;
    const amrex::Real ah2 = m_alpha_h * m_alpha_h;
    const amrex::Real av2 = m_alpha_v * m_alpha_v;
    const amrex::Real fac = -(ah2*kx*kx + ah2*ky*ky + av2*kz*kz);

    const amrex::Vector<amrex::Real>& zf = grid.z_face();
    const amrex::Vector<amrex::Real> Jn = NodeMetric(grid);

    auto exact = [=] (amrex::Real x, amrex::Real y, amrex::Real z) {
        return std::sin(kx * (x - xlo)) * std::sin(ky * (y - ylo))
             * std::sin(kz * (z - zlo));
    };

    for (amrex::MFIter mfi(m_rhs); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const& r = m_rhs.array(mfi);
        amrex::LoopOnCpu(bx, [&] (int i, int j, int k) noexcept
        {
            const amrex::Real x = xlo + amrex::Real(i) * dx;
            const amrex::Real y = ylo + amrex::Real(j) * dy;
            const amrex::Real z = zf[k];              // nodes sit on faces
            r(i,j,k) = Jn[k] * fac * exact(x, y, z);
        });
    }

    m_lambda.setVal(0.0);
    Solve();

    amrex::Real sum2 = 0.0, linf = 0.0;
    long n = 0;
    for (amrex::MFIter mfi(m_lambda); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const& l = m_lambda.const_array(mfi);
        amrex::LoopOnCpu(bx, [&] (int i, int j, int k) noexcept
        {
            const amrex::Real x = xlo + amrex::Real(i) * dx;
            const amrex::Real y = ylo + amrex::Real(j) * dy;
            const amrex::Real z = zf[k];
            const amrex::Real e = l(i,j,k) - exact(x, y, z);
            sum2 += e * e;
            linf = std::max(linf, std::abs(e));
            ++n;
        });
    }
    amrex::ParallelDescriptor::ReduceRealSum(sum2);
    amrex::ParallelDescriptor::ReduceRealMax(linf);
    amrex::ParallelDescriptor::ReduceLongSum(n);

    return {std::sqrt(sum2 / amrex::Real(n)), linf};
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

void Poisson::AppendReport (const std::string& filename) const
{
    if (!amrex::ParallelDescriptor::IOProcessor()) { return; }

    std::ofstream os(filename, std::ios::app);
    os << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10);
    os << "# poisson (Phase 5)\n";
    os << "poisson_alpha_h " << m_alpha_h << "\n";
    os << "poisson_alpha_v " << m_alpha_v << "\n";
    os << "poisson_sigma_convention alpha_squared\n";
    os << "poisson_n_pinned_nodes " << m_n_pinned << "\n";
    os << "poisson_rhs_min " << m_rhs.min(0) << "\n";
    os << "poisson_rhs_max " << m_rhs.max(0) << "\n";
    os << "poisson_solve_residual " << m_resid << "\n";
    os << "poisson_aspect_ratio " << m_aspect << "\n";
    os << "poisson_num_pre_smooth " << m_pre_smooth << "\n";
    os << "poisson_num_post_smooth " << m_post_smooth << "\n";
    os << "poisson_rhs_operator " << m_rhs_operator << "\n";
    os << "poisson_lambda_bc " << m_lambda_bc << "\n";
    os << "poisson_gradient_operator " << m_gradient_operator << "\n";
    os << "poisson_rhs_nodes_zeroed " << m_n_rhs_zeroed << "\n";
    os << "poisson_speed_max_before " << m_speed_before << "\n";
    os << "poisson_speed_max_after " << m_speed_after << "\n";
    for (int n = 0; n < 3; ++n) {
        const char* nm = (n == 0) ? "u" : ((n == 1) ? "v" : "w");
        os << "poisson_" << nm << "_min_after " << m_vel_after.lo[n] << "\n";
        os << "poisson_" << nm << "_max_after " << m_vel_after.hi[n] << "\n";
    }
    os << "poisson_div_before " << m_div_before << "\n";
    os << "poisson_div_after " << m_div_after << "\n";
    os << "poisson_div_controlled_before " << m_div_fe_before << "\n";
    os << "poisson_div_controlled_after " << m_div_fe_after << "\n";
    os << "poisson_n_projections " << m_n_projections << "\n";
    os << "poisson_lambda_absmax "
       << std::max(std::abs(m_lambda.min(0)), std::abs(m_lambda.max(0)))
       << "\n";
    os.close();
}

void Poisson::WriteRHSDump (const std::string& filename) const
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        amrex::ParallelDescriptor::NProcs() == 1,
        "poisson.rhs_dump_file is a single-rank regtest aid and cannot be "
        "written from a multi-rank run");

    std::ofstream os(filename);
    os << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10);
    os << "# i j k rhs   (nodal, in computational space: the physical\n";
    os << "# divergence weighted by the metric J = dz(k)/dz_nominal)\n";

    for (amrex::MFIter mfi(m_rhs); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const& r = m_rhs.const_array(mfi);
        amrex::LoopOnCpu(bx, [&] (int i, int j, int k) noexcept
        {
            os << i << " " << j << " " << k << " " << r(i,j,k) << "\n";
        });
    }
    os.close();

    FWT_DEBUG("wrote nodal RHS dump: " << filename);
}

} // namespace fwt
