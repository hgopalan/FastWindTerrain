#include "Verify.H"
#include "Derivatives.H"
#include "Debug.H"

#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>

namespace fwt {

std::string Verify::DumpFile ()
{
    amrex::ParmParse pp("verify");
    std::string f;
    pp.query("gradient_dump_file", f);
    return f;
}

amrex::Real Verify::AdvectSign ()
{
    amrex::ParmParse pp("verify");
    amrex::Real a = 1.0;
    pp.query("gradient_advect", a);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(a != 0.0,
        "verify.gradient_advect must be nonzero: it selects which upwind "
        "branch the study measures");
    return a;
}

bool Verify::MaybeWriteGradientDump (const Grid& grid,
                                     const Terrain& terrain,
                                     const amrex::MultiFab& vel) const
{
    const std::string filename = DumpFile();
    if (filename.empty()) { return false; }

    const int nz = grid.nz();
    const amrex::Vector<amrex::Real>& z_cc = grid.z_cc();
    const amrex::Vector<amrex::Real>& z_face = grid.z_face();
    const amrex::Vector<amrex::Real> dzdk = ColumnMetric(z_cc);

    const amrex::Real a = AdvectSign();
    const Scheme scheme = Numerics::scheme();
    const int solid = Terrain::kSolid;

    const amrex::Box& dom = grid.geom().Domain();
    const int klo = dom.smallEnd(2), khi = dom.bigEnd(2);

    constexpr amrex::Real kBig = std::numeric_limits<amrex::Real>::max();
    amrex::Vector<amrex::Real> umin(nz,  kBig), umax(nz, -kBig);
    amrex::Vector<amrex::Real> vmin(nz,  kBig), vmax(nz, -kBig);
    amrex::Vector<long> ncell(nz, 0);

    // A host loop, like the other diagnostics: it runs once, it is not in
    // the solve, and under a GPU build it reads through managed memory.
    // Every box spans the full height (see Grid), so a column is never
    // split and the vertical stencil stays inside one box.
    for (amrex::MFIter mfi(vel); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const& v  = vel.const_array(mfi);
        auto const& mk = terrain.mask().const_array(mfi);

        amrex::LoopOnCpu(bx, [&] (int i, int j, int k) noexcept
        {
            if (mk(i,j,k) == solid) { return; }

            // The same clamping the divergence uses, so this measures the
            // real path rather than an idealized one. It makes the two
            // cells at each end one-sided, which is why the convergence
            // driver excludes them.
            const int km2 = amrex::max(k-2, klo), km1 = amrex::max(k-1, klo);
            const int kp1 = amrex::min(k+1, khi), kp2 = amrex::min(k+2, khi);

            const amrex::Real du =
                Derivative(scheme, v(i,j,km2,0), v(i,j,km1,0), v(i,j,k,0),
                           v(i,j,kp1,0), v(i,j,kp2,0), a, dzdk[k]);
            const amrex::Real dv =
                Derivative(scheme, v(i,j,km2,1), v(i,j,km1,1), v(i,j,k,1),
                           v(i,j,kp1,1), v(i,j,kp2,1), a, dzdk[k]);

            umin[k] = std::min(umin[k], du);
            umax[k] = std::max(umax[k], du);
            vmin[k] = std::min(vmin[k], dv);
            vmax[k] = std::max(vmax[k], dv);
            ++ncell[k];
        });
    }

    amrex::ParallelDescriptor::ReduceRealMin(umin.dataPtr(), nz);
    amrex::ParallelDescriptor::ReduceRealMax(umax.dataPtr(), nz);
    amrex::ParallelDescriptor::ReduceRealMin(vmin.dataPtr(), nz);
    amrex::ParallelDescriptor::ReduceRealMax(vmax.dataPtr(), nz);
    amrex::ParallelDescriptor::ReduceLongSum(ncell.dataPtr(), nz);

    if (!amrex::ParallelDescriptor::IOProcessor()) { return true; }

    std::ofstream os(filename);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(os.good(),
        "could not open verify.gradient_dump_file for writing");

    os << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10);
    os << "# profile-gradient dump: d(u)/dz and d(v)/dz as the solver "
          "computes them\n";
    os << "# scheme " << Numerics::name() << "\n";
    os << "# advect " << a << "\n";
    os << "# nz " << nz << "\n";
    os << "# mid is the midpoint of the level's range and spread its "
          "width; over\n";
    os << "# flat ground the field is horizontally uniform, so a correct "
          "run has\n";
    os << "# spread = 0 at every level\n";
    os << "# k z_cc dz dzdk dudz_mid dudz_spread dvdz_mid dvdz_spread "
          "n_fluid\n";

    for (int k = 0; k < nz; ++k) {
        // A level buried entirely in terrain has no sample; it is written
        // as such rather than as a plausible-looking zero.
        const bool has = (ncell[k] > 0);
        const amrex::Real um = has ? 0.5 * (umin[k] + umax[k]) : 0.0;
        const amrex::Real us = has ? (umax[k] - umin[k]) : 0.0;
        const amrex::Real vm = has ? 0.5 * (vmin[k] + vmax[k]) : 0.0;
        const amrex::Real vs = has ? (vmax[k] - vmin[k]) : 0.0;

        os << k << " " << z_cc[k] << " " << (z_face[k+1] - z_face[k])
           << " " << dzdk[k]
           << " " << um << " " << us
           << " " << vm << " " << vs
           << " " << ncell[k] << "\n";
    }
    os.close();

    amrex::Print() << "Wrote profile-gradient dump to " << filename << "\n";
    FWT_DEBUG("profile-gradient dump: scheme " << Numerics::name()
              << ", advect " << a << ", " << nz << " levels");
    return true;
}

} // namespace fwt
