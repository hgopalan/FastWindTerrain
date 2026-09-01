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

void Poisson::BuildSigma (const Grid& grid, const Terrain& terrain)
{
    m_sigma.define(grid.ba(), grid.dm(), AMREX_SPACEDIM, 0);

    const amrex::Vector<amrex::Real> Jc = CellMetric(grid);
    amrex::Gpu::DeviceVector<amrex::Real> d_J(Jc.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, Jc.begin(), Jc.end(),
                          d_J.begin());
    amrex::Gpu::streamSynchronize();
    const amrex::Real* J = d_J.data();

    const amrex::Real ah2 = m_alpha_h * m_alpha_h;
    const amrex::Real av2 = m_alpha_v * m_alpha_v;
    const int solid = Terrain::kSolid;

    for (amrex::MFIter mfi(m_sigma); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const& s  = m_sigma.array(mfi);
        auto const& mk = terrain.mask().const_array(mfi);
        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            if (mk(i,j,k) == solid) {
                // No-flux inside the terrain: zero conductivity carries
                // no flux, which is exactly the Neumann condition the
                // immersed boundary needs.
                s(i,j,k,0) = 0.0;
                s(i,j,k,1) = 0.0;
                s(i,j,k,2) = 0.0;
                return;
            }
            // The metric rides in on the coefficients: horizontal terms
            // are weighted by the cell's true height, the vertical term
            // by its inverse.
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
    } else {
        // Face order is xlo, xhi, ylo, yhi, zlo, zhi.
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            lo_bc[d] = ToLinOpBC(bc.lambda_bc(2*d));
            hi_bc[d] = ToLinOpBC(bc.lambda_bc(2*d + 1));
        }
    }
    m_op->setDomainBC(lo_bc, hi_bc);

    // Nodes with no fluid cell around them have an empty row. Pin them
    // rather than leaving the operator singular there.
    m_overset.define(amrex::convert(m_ba, amrex::IntVect::TheNodeVector()),
                     m_dm, 1, 0);
    m_overset.setVal(1);          // 1 = solved for, 0 = pinned/known

    const int solid = Terrain::kSolid;

    // A nodal box reaches one cell past its cell-centered counterpart in
    // every direction, so the mask has to be readable there. Ghosts
    // outside the domain stay solid, which is the right answer: there is
    // no fluid out there for a boundary node to see.
    amrex::iMultiFab maskg(m_ba, m_dm, 1, 1);
    maskg.setVal(solid);
    amrex::iMultiFab::Copy(maskg, terrain.mask(), 0, 0, 1, 0);
    maskg.FillBoundary(m_geom.periodicity());

    long n_pinned = 0;
    for (amrex::MFIter mfi(m_overset); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const& os = m_overset.array(mfi);
        auto const& mk = maskg.const_array(mfi);
        const amrex::Box& dom = m_geom.Domain();

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
            if (!any_fluid) {
                os(i,j,k) = 0;
                ++n_pinned;
            }
        });
    }
    amrex::ParallelDescriptor::ReduceLongSum(n_pinned);
    m_n_pinned = n_pinned;

    if (m_n_pinned > 0) {
        m_op->setOversetMask(0, m_overset);
    }

    m_op->setSigma(0, m_sigma);
}

void Poisson::Build (const Grid& grid, const Terrain& terrain,
                     const BoundaryConditions& bc)
{
    ReadParameters();

    {
        amrex::ParmParse pp("poisson");
        int mms = 0;
        pp.query("manufactured", mms);
        m_all_dirichlet = (mms != 0);
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

    BuildSigma(grid, terrain);
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

    amrex::Vector<amrex::MultiFab*> sol {&m_lambda};
    amrex::Vector<const amrex::MultiFab*> rhs {&m_rhs};

    const amrex::Real resid = mlmg.solve(sol, rhs, m_reltol, m_abstol);
    FWT_DEBUG("MLMG solve: residual " << resid);
    return resid;
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
