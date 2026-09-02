#include "Diagnostics.H"
#include "Debug.H"

#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>
#include <AMReX_GpuContainers.H>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>

namespace fwt {

const char* FluxBalance::face_name (int f)
{
    switch (f) {
        case 0: return "xlo";
        case 1: return "xhi";
        case 2: return "ylo";
        case 3: return "yhi";
        default: return "top";
    }
}

// ---------------------------------------------------------------------------
// Boundary mass flux
// ---------------------------------------------------------------------------

FluxBalance ComputeFluxBalance (const Grid& grid, const Terrain& terrain,
                                const amrex::MultiFab& vel)
{
    const amrex::Box& domain = grid.geom().Domain();
    const amrex::Real dx = grid.geom().CellSize(0);
    const amrex::Real dy = grid.geom().CellSize(1);

    // z_face lives on the host; one copy so the reduction kernel is
    // valid in a GPU build too.
    const amrex::Vector<amrex::Real>& z_face = grid.z_face();
    amrex::Gpu::DeviceVector<amrex::Real> d_zf(z_face.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, z_face.begin(),
                          z_face.end(), d_zf.begin());
    amrex::Gpu::streamSynchronize();
    const amrex::Real* zf = d_zf.data();

    // The ground is closed, so it is not in this list: no flux crosses
    // it by construction, and including it would only add zeros.
    struct Face { int dir; int side; amrex::Real fac; bool use_dz; };
    const Face faces[FluxBalance::kNFaces] = {
        {0, -1, dy,      true },   // xlo
        {0, +1, dy,      true },   // xhi
        {1, -1, dx,      true },   // ylo
        {1, +1, dx,      true },   // yhi
        {2, +1, dx * dy, false},   // top
    };

    const int solid = Terrain::kSolid;
    FluxBalance fb;

    for (int f = 0; f < FluxBalance::kNFaces; ++f) {
        amrex::Box layer(domain);
        if (faces[f].side < 0) {
            layer.setBig(faces[f].dir, domain.smallEnd(faces[f].dir));
        } else {
            layer.setSmall(faces[f].dir, domain.bigEnd(faces[f].dir));
        }

        amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_op;
        amrex::ReduceData<amrex::Real, amrex::Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

        const int dir = faces[f].dir;
        const amrex::Real sign = amrex::Real(faces[f].side);
        const amrex::Real fac = faces[f].fac;
        const bool use_dz = faces[f].use_dz;

        for (amrex::MFIter mfi(vel); mfi.isValid(); ++mfi) {
            const amrex::Box sect = mfi.tilebox() & layer;
            if (!sect.ok()) { continue; }

            auto const& v  = vel.const_array(mfi);
            auto const& mk = terrain.mask().const_array(mfi);

            reduce_op.eval(sect, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
            {
                if (mk(i,j,k) == solid) { return {0.0, 0.0}; }
                const amrex::Real area =
                    fac * (use_dz ? (zf[k+1] - zf[k]) : amrex::Real(1.0));
                const amrex::Real flux = sign * v(i,j,k,dir) * area;
                return {amrex::max(flux, amrex::Real(0.0)),
                        amrex::min(flux, amrex::Real(0.0))};
            });
        }

        ReduceTuple hv = reduce_data.value(reduce_op);
        amrex::Real f_out = amrex::get<0>(hv);
        amrex::Real f_in  = -amrex::get<1>(hv);   // stored negative
        amrex::ParallelDescriptor::ReduceRealSum(f_out);
        amrex::ParallelDescriptor::ReduceRealSum(f_in);

        fb.face_net[f] = f_out - f_in;
        fb.out += f_out;
        fb.in  += f_in;
    }

    fb.net = fb.out - fb.in;
    const amrex::Real scale = std::max(std::abs(fb.in), std::abs(fb.out));
    fb.imbalance = (scale > 0.0) ? std::abs(fb.net) / scale : 0.0;
    return fb;
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

void Diagnostics::ReadParameters ()
{
    amrex::ParmParse pp("diagnostics");
    pp.query("flux_tolerance", m_flux_tol);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_flux_tol >= 0.0,
        "diagnostics.flux_tolerance must be >= 0");
}

void Diagnostics::Compute (const Grid& grid, const Terrain& terrain,
                           const amrex::MultiFab& vel,
                           const amrex::MultiFab& div)
{
    ReadParameters();

    m_flux = ComputeFluxBalance(grid, terrain, vel);
    m_flux_ok = (m_flux.imbalance <= m_flux_tol);

    // Divergence extrema and an L2 norm over fluid cells only. The L2 is
    // volume weighted, so on a stretched grid the thin near-surface
    // cells do not dominate a norm they occupy little of.
    const amrex::Vector<amrex::Real>& z_face = grid.z_face();
    const amrex::Real dx = grid.geom().CellSize(0);
    const amrex::Real dy = grid.geom().CellSize(1);

    amrex::Real dmax = -std::numeric_limits<amrex::Real>::max();
    amrex::Real dmin =  std::numeric_limits<amrex::Real>::max();
    amrex::Real sum2 = 0.0, vol = 0.0;
    long n_fluid = 0;

    const int solid = Terrain::kSolid;
    for (amrex::MFIter mfi(div); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const& d  = div.const_array(mfi);
        auto const& mk = terrain.mask().const_array(mfi);
        amrex::LoopOnCpu(bx, [&] (int i, int j, int k) noexcept
        {
            if (mk(i,j,k) == solid) { return; }
            const amrex::Real val = d(i,j,k);
            dmax = std::max(dmax, val);
            dmin = std::min(dmin, val);
            const amrex::Real dv = dx * dy * (z_face[k+1] - z_face[k]);
            sum2 += val * val * dv;
            vol  += dv;
            ++n_fluid;
        });
    }
    // A rank that owns no fluid cell contributes its sentinels, which
    // the min/max reductions discard; it must not contribute a zero.
    amrex::ParallelDescriptor::ReduceRealMax(dmax);
    amrex::ParallelDescriptor::ReduceRealMin(dmin);
    amrex::ParallelDescriptor::ReduceRealSum(sum2);
    amrex::ParallelDescriptor::ReduceRealSum(vol);
    amrex::ParallelDescriptor::ReduceLongSum(n_fluid);

    if (n_fluid == 0) { dmax = 0.0; dmin = 0.0; }

    m_div_min  = dmin;
    m_div_max  = std::max(std::abs(dmin), std::abs(dmax));
    m_div_l2   = (vol > 0.0) ? std::sqrt(sum2 / vol) : 0.0;
    m_n_fluid  = n_fluid;

    FWT_DEBUG_SECTION("Diagnostics");
    FWT_DEBUG("fluid cells      = " << m_n_fluid);
    FWT_DEBUG("div u  min       = " << dmin);
    FWT_DEBUG("div u  max       = " << dmax);
    FWT_DEBUG("max |div u|      = " << m_div_max << " 1/s");
    FWT_DEBUG("L2  |div u|      = " << m_div_l2 << " 1/s");
    for (int f = 0; f < FluxBalance::kNFaces; ++f) {
        FWT_DEBUG("flux " << FluxBalance::face_name(f)
                  << " (out +)   = " << m_flux.face_net[f] << " m^3/s");
    }
    FWT_DEBUG("flux_tolerance   = " << m_flux_tol);
}

void Diagnostics::Print () const
{
    amrex::Print() << "Diagnostics: max|div(u)| = " << m_div_max
                   << " 1/s, L2|div(u)| = " << m_div_l2
                   << " 1/s over " << m_n_fluid << " fluid cells\n";
    amrex::Print() << "  mass flux in/out = " << m_flux.in << " / "
                   << m_flux.out << " m^3/s, net = " << m_flux.net
                   << ", relative imbalance = " << m_flux.imbalance << "\n";
    amrex::Print() << "  per face (out +):";
    for (int f = 0; f < FluxBalance::kNFaces; ++f) {
        amrex::Print() << "  " << FluxBalance::face_name(f) << " "
                       << m_flux.face_net[f];
    }
    amrex::Print() << "\n";

    if (!m_flux_ok) {
        amrex::Print() << "  WARNING: boundary flux imbalance "
                       << m_flux.imbalance
                       << " exceeds diagnostics.flux_tolerance = "
                       << m_flux_tol << ". Reported, not corrected.\n";
    }
}

void Diagnostics::AppendReport (const std::string& filename) const
{
    if (!amrex::ParallelDescriptor::IOProcessor()) { return; }

    std::ofstream os(filename, std::ios::app);
    os << std::setprecision(17);
    os << "diag_n_fluid_cells " << m_n_fluid << "\n";
    os << "diag_div_max " << m_div_max << "\n";
    os << "diag_div_min " << m_div_min << "\n";
    os << "diag_div_l2 " << m_div_l2 << "\n";
    os << "diag_flux_in " << m_flux.in << "\n";
    os << "diag_flux_out " << m_flux.out << "\n";
    os << "diag_flux_net " << m_flux.net << "\n";
    os << "diag_flux_imbalance " << m_flux.imbalance << "\n";
    os << "diag_flux_tolerance " << m_flux_tol << "\n";
    os << "diag_flux_within_tolerance " << (m_flux_ok ? 1 : 0) << "\n";
    for (int f = 0; f < FluxBalance::kNFaces; ++f) {
        os << "diag_flux_" << FluxBalance::face_name(f) << " "
           << m_flux.face_net[f] << "\n";
    }
}

} // namespace fwt
