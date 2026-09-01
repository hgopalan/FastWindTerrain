#include "Anisotropy.H"
#include "Debug.H"

#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_GpuContainers.H>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>

namespace fwt {

void Anisotropy::ReadParameters ()
{
    amrex::ParmParse pp("anisotropy");

    pp.query("enable", m_enable);
    pp.query("source", m_source);
    pp.query("alpha_h_mode", m_alpha_h_mode);
    pp.query("slope_scale", m_slope_scale);
    pp.query("decay_height", m_decay_height);
    pp.query("min_factor", m_min_factor);
    pp.query("max_factor", m_max_factor);

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_source == "slope" || m_source == "none",
        "anisotropy.source must be 'slope' or 'none' (the Richardson and "
        "Froude terms are hooks and are not implemented yet)");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_alpha_h_mode == "base" || m_alpha_h_mode == "slope",
        "anisotropy.alpha_h_mode must be 'base' or 'slope'");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_slope_scale > 0.0,
        "anisotropy.slope_scale must be > 0");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_decay_height > 0.0,
        "anisotropy.decay_height must be > 0");

    // The base values live with the operator they feed.
    amrex::ParmParse ppp("poisson");
    ppp.query("alpha_h", m_alpha_h_base);
    ppp.query("alpha_v", m_alpha_v_base);

    FWT_DEBUG_SECTION("Anisotropy inputs (anisotropy.*)");
    FWT_DEBUG("enable           = " << m_enable
              << (m_enable ? "" : "   [alphas hold their base values]"));
    FWT_DEBUG("source           = " << m_source);
    FWT_DEBUG("alpha_h_mode     = " << m_alpha_h_mode);
    FWT_DEBUG("slope_scale      = " << m_slope_scale);
    FWT_DEBUG("decay_height     = " << m_decay_height << " m");
    FWT_DEBUG("alpha_h_base     = " << m_alpha_h_base);
    FWT_DEBUG("alpha_v_base     = " << m_alpha_v_base);
}

void Anisotropy::Build (const Grid& grid, const Terrain& terrain)
{
    ReadParameters();

    m_alpha_h.define(grid.ba(), grid.dm(), 1, 0);
    m_alpha_v.define(grid.ba(), grid.dm(), 1, 0);

    if (m_enable == 0 || m_source == "none") {
        m_alpha_h.setVal(m_alpha_h_base);
        m_alpha_v.setVal(m_alpha_v_base);
        m_ah_min = m_ah_max = m_alpha_h_base;
        m_av_min = m_av_max = m_alpha_v_base;
        m_slope_max = 0.0;
        FWT_DEBUG("anisotropy disabled: alpha_h = " << m_alpha_h_base
                  << ", alpha_v = " << m_alpha_v_base << " everywhere");
        return;
    }

    const int nx = grid.nx();
    const int ny = grid.ny();
    const amrex::Real dx = grid.geom().CellSize(0);
    const amrex::Real dy = grid.geom().CellSize(1);
    const std::vector<amrex::Real>& h = terrain.column_heights();
    const amrex::Vector<amrex::Real>& z_cc = grid.z_cc();

    // Terrain slope magnitude per column, from central differences with
    // one-sided edges -- the same stencil massconsistent_amr uses.
    std::vector<amrex::Real> slope(std::size_t(nx) * ny, 0.0);
    m_slope_max = 0.0;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int im1 = std::max(0, i - 1);
            const int ip1 = std::min(nx - 1, i + 1);
            const int jm1 = std::max(0, j - 1);
            const int jp1 = std::min(ny - 1, j + 1);

            const amrex::Real dhdx =
                (h[std::size_t(j)*nx + ip1] - h[std::size_t(j)*nx + im1])
                / (amrex::Real(ip1 - im1) * dx);
            const amrex::Real dhdy =
                (h[std::size_t(jp1)*nx + i] - h[std::size_t(jm1)*nx + i])
                / (amrex::Real(jp1 - jm1) * dy);

            const amrex::Real s = std::sqrt(dhdx*dhdx + dhdy*dhdy);
            slope[std::size_t(j)*nx + i] = s;
            m_slope_max = std::max(m_slope_max, s);
        }
    }

    amrex::Gpu::DeviceVector<amrex::Real> d_slope(slope.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, slope.begin(),
                          slope.end(), d_slope.begin());
    amrex::Gpu::DeviceVector<amrex::Real> d_h(h.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h.begin(), h.end(),
                          d_h.begin());
    amrex::Gpu::DeviceVector<amrex::Real> d_zcc(z_cc.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, z_cc.begin(),
                          z_cc.end(), d_zcc.begin());
    amrex::Gpu::streamSynchronize();
    const amrex::Real* ps = d_slope.data();
    const amrex::Real* ph = d_h.data();
    const amrex::Real* pz = d_zcc.data();

    const amrex::Real ah_base = m_alpha_h_base;
    const amrex::Real av_base = m_alpha_v_base;
    const amrex::Real scale = m_slope_scale;
    const amrex::Real decay = m_decay_height;
    const amrex::Real lo_f = m_min_factor;
    const amrex::Real hi_f = m_max_factor;
    const bool slope_in_h = (m_alpha_h_mode == "slope");

    for (amrex::MFIter mfi(m_alpha_v); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const& av = m_alpha_v.array(mfi);
        auto const& ah = m_alpha_h.array(mfi);
        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // Height above ground, floored at zero: inside the terrain
            // the decay term would otherwise amplify rather than decay.
            const amrex::Real z_agl =
                amrex::max(pz[k] - ph[std::size_t(j)*nx + i],
                           amrex::Real(0.0));

            const amrex::Real s = ps[std::size_t(j)*nx + i];
            const amrex::Real slope_3d = s * std::exp(-z_agl / decay);
            const amrex::Real f_slope = std::exp(-slope_3d / scale);

            // Hooks for the stability and Froude terms. Neutral case.
            const amrex::Real f_ri = 1.0;
            const amrex::Real f_fr = 1.0;

            const amrex::Real v = av_base * f_slope * f_ri * f_fr;
            av(i,j,k) = amrex::max(lo_f * av_base,
                                   amrex::min(v, hi_f * av_base));

            const amrex::Real hh = ah_base * (slope_in_h ? f_slope : 1.0)
                                 * f_ri * f_fr;
            ah(i,j,k) = amrex::max(lo_f * ah_base,
                                   amrex::min(hh, hi_f * ah_base));
        });
    }

    m_av_min = m_alpha_v.min(0);
    m_av_max = m_alpha_v.max(0);
    m_ah_min = m_alpha_h.min(0);
    m_ah_max = m_alpha_h.max(0);

    amrex::Print() << "Anisotropy: alpha_v in [" << m_av_min << ", "
                   << m_av_max << "], alpha_h in [" << m_ah_min << ", "
                   << m_ah_max << "], max terrain slope " << m_slope_max
                   << "\n";

    if (Debug::Enabled()) {
        FWT_DEBUG_SECTION("Anisotropy");
        FWT_DEBUG("max terrain slope= " << m_slope_max
                  << "   (|grad z_terrain|)");
        FWT_DEBUG("f_slope at that slope, at the surface = "
                  << std::exp(-m_slope_max / m_slope_scale));
        FWT_DEBUG("alpha_v range    = [" << m_av_min << ", " << m_av_max
                                          << "]   (base " << av_base << ")");
        FWT_DEBUG("alpha_h range    = [" << m_ah_min << ", " << m_ah_max
                                          << "]   (base " << ah_base << ")");
        FWT_DEBUG("clamped to [" << lo_f << ", " << hi_f << "] x base");
    }
}

void Anisotropy::AppendReport (const std::string& filename) const
{
    if (!amrex::ParallelDescriptor::IOProcessor()) { return; }

    std::ofstream os(filename, std::ios::app);
    os << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10);
    os << "# anisotropy (Phase 7)\n";
    os << "anisotropy_enable " << m_enable << "\n";
    os << "anisotropy_source " << m_source << "\n";
    os << "anisotropy_alpha_h_mode " << m_alpha_h_mode << "\n";
    os << "anisotropy_slope_scale " << m_slope_scale << "\n";
    os << "anisotropy_decay_height " << m_decay_height << "\n";
    os << "anisotropy_slope_max " << m_slope_max << "\n";
    os << "anisotropy_alpha_v_min " << m_av_min << "\n";
    os << "anisotropy_alpha_v_max " << m_av_max << "\n";
    os << "anisotropy_alpha_h_min " << m_ah_min << "\n";
    os << "anisotropy_alpha_h_max " << m_ah_max << "\n";
    os.close();
}

} // namespace fwt
