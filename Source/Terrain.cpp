#include "Terrain.H"
#include "Debug.H"

#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

namespace fwt {

namespace {
    // Squared-distance below which a query counts as landing exactly on an
    // input point. Same value massconsistent_amr uses (DISTANCE_EPSILON).
    constexpr amrex::Real kDistanceEpsilon = 1.0e-12;
}

// ---------------------------------------------------------------------------
// File reading
// ---------------------------------------------------------------------------

void Terrain::ReadPointFile (const std::string& filename,
                             std::vector<amrex::Real>& xp,
                             std::vector<amrex::Real>& yp,
                             std::vector<amrex::Real>& zp)
{
    std::ifstream f(filename);
    if (!f.is_open()) {
        amrex::Abort("Terrain: cannot open terrain file: " + filename);
    }

    std::string line;
    while (std::getline(f, line)) {
        // Strip comments, then treat commas as separators. A line that
        // does not yield three numbers is skipped, which is what lets a
        // leading "x,y,z" header through harmlessly.
        const auto pos = line.find('#');
        if (pos != std::string::npos) { line = line.substr(0, pos); }
        std::replace(line.begin(), line.end(), ',', ' ');

        std::istringstream ss(line);
        amrex::Real x, y, z;
        if (ss >> x >> y >> z) {
            xp.push_back(x);
            yp.push_back(y);
            zp.push_back(z);
        }
    }

    if (xp.empty()) {
        amrex::Abort("Terrain: no data read from terrain file: " + filename);
    }
}

// ---------------------------------------------------------------------------
// IDW interpolation (port of massconsistent_amr's idw_terrain)
// ---------------------------------------------------------------------------

amrex::Real Terrain::InterpolateIDW (amrex::Real xq, amrex::Real yq,
                                     const std::vector<amrex::Real>& xp,
                                     const std::vector<amrex::Real>& yp,
                                     const std::vector<amrex::Real>& zp,
                                     int k, amrex::Real exponent)
{
    const int n = static_cast<int>(xp.size());
    k = std::min(k, n);

    std::vector<std::pair<amrex::Real,int>> d2(n);
    for (int i = 0; i < n; ++i) {
        const amrex::Real dx = xp[i] - xq;
        const amrex::Real dy = yp[i] - yq;
        d2[i] = {dx*dx + dy*dy, i};
    }
    std::partial_sort(d2.begin(), d2.begin() + k, d2.end());

    amrex::Real wsum = 0.0;
    amrex::Real zval = 0.0;
    for (int i = 0; i < k; ++i) {
        if (d2[i].first < kDistanceEpsilon) {
            return zp[d2[i].second];      // exact hit on an input point
        }
        const amrex::Real w = std::pow(d2[i].first, -exponent / amrex::Real(2.0));
        wsum += w;
        zval += w * zp[d2[i].second];
    }
    return zval / wsum;
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

void Terrain::ReadParameters ()
{
    amrex::ParmParse pp("terrain");

    const bool got_file = pp.query("file", m_file);
    const bool got_flat = pp.query("flat_elevation", m_flat_elevation);
    const bool got_k    = pp.query("idw_n_neighbors", m_idw_k);
    const bool got_p    = pp.query("idw_exponent", m_idw_exponent);

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_idw_k > 0,
        "terrain.idw_n_neighbors must be > 0");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_idw_exponent > 0.0,
        "terrain.idw_exponent must be > 0");

    FWT_DEBUG_SECTION("Terrain inputs (terrain.*)");
    FWT_DEBUG("file             = "
              << (got_file ? m_file : std::string("<none>  [flat ground]")));
    FWT_DEBUG("flat_elevation   = " << m_flat_elevation << " m"
              << (got_flat ? "" : "   [default]")
              << (got_file ? "   [unused: file given]" : ""));
    FWT_DEBUG("idw_n_neighbors  = " << m_idw_k << (got_k ? "" : "   [default]"));
    FWT_DEBUG("idw_exponent     = " << m_idw_exponent
                                     << (got_p ? "" : "   [default]"));
}

void Terrain::BuildTerrainHeight (const Grid& grid)
{
    const int nx = grid.nx();
    const int ny = grid.ny();
    const amrex::Real dx = grid.geom().CellSize(0);
    const amrex::Real dy = grid.geom().CellSize(1);
    const amrex::Real xlo = grid.geom().ProbLo(0);
    const amrex::Real ylo = grid.geom().ProbLo(1);

    // Column heights, computed on the host: the IDW search is a partial
    // sort over the scattered point list, which is host-side work. Every
    // rank does the same nx*ny computation rather than computing a
    // decomposed piece and communicating -- the column array is tiny
    // next to the 3D fields, and this keeps the result independent of
    // the decomposition.
    std::vector<amrex::Real>& h = m_h;
    h.assign(std::size_t(nx) * std::size_t(ny), 0.0);

    if (m_xp.empty()) {
        std::fill(h.begin(), h.end(), m_flat_elevation);
    } else {
        for (int j = 0; j < ny; ++j) {
            const amrex::Real yq = ylo + (amrex::Real(j) + 0.5) * dy;
            for (int i = 0; i < nx; ++i) {
                const amrex::Real xq = xlo + (amrex::Real(i) + 0.5) * dx;
                h[std::size_t(j)*nx + i] =
                    InterpolateIDW(xq, yq, m_xp, m_yp, m_zp,
                                   m_idw_k, m_idw_exponent);
            }
        }
    }

    m_z_min = *std::min_element(h.begin(), h.end());
    m_z_max = *std::max_element(h.begin(), h.end());

    // Scatter the column array into the MultiFab, replicated along k.
    m_z_terrain.define(grid.ba(), grid.dm(), 1, 0);

    amrex::Gpu::DeviceVector<amrex::Real> d_h(h.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h.begin(), h.end(),
                          d_h.begin());
    amrex::Gpu::streamSynchronize();
    const amrex::Real* ph = d_h.data();

    for (amrex::MFIter mfi(m_z_terrain); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const& a = m_z_terrain.array(mfi);
        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            amrex::ignore_unused(k);
            a(i,j,k) = ph[std::size_t(j)*nx + i];
        });
    }
}

void Terrain::BuildMask (const Grid& grid)
{
    m_mask.define(grid.ba(), grid.dm(), 1, 0);

    // z_cc is the true (stretched) cell-center height, so the mask must
    // come from it rather than from geom().CellSize(2).
    amrex::Gpu::DeviceVector<amrex::Real> d_zcc(grid.z_cc().size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, grid.z_cc().begin(),
                          grid.z_cc().end(), d_zcc.begin());
    amrex::Gpu::streamSynchronize();
    const amrex::Real* zcc = d_zcc.data();

    const int solid = kSolid;
    const int fluid = kFluid;

    for (amrex::MFIter mfi(m_mask); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const& m  = m_mask.array(mfi);
        auto const& zt = m_z_terrain.const_array(mfi);
        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // Binary mask, no volume fractions. A cell center sitting
            // exactly on the surface counts as solid, matching
            // massconsistent_amr's is_solid = (z_cc - z_terrain <= 0).
            m(i,j,k) = (zcc[k] - zt(i,j,k) <= amrex::Real(0.0)) ? solid : fluid;
        });
    }

    m_n_solid = static_cast<long>(m_mask.sum(0));   // kSolid == 1
    m_n_total = static_cast<long>(grid.nx()) * grid.ny() * grid.nz();
}

void Terrain::Build (const Grid& grid)
{
    ReadParameters();

    if (!m_file.empty()) {
        // Every rank reads the file. It is a small ASCII point list and
        // reading it everywhere avoids a broadcast whose only purpose
        // would be to save that read.
        ReadPointFile(m_file, m_xp, m_yp, m_zp);
        amrex::Print() << "Terrain: read " << m_xp.size()
                       << " points from " << m_file << "\n";
    }

    BuildTerrainHeight(grid);
    BuildMask(grid);

    if (Debug::Enabled()) {
        FWT_DEBUG_SECTION("Terrain");
        FWT_DEBUG("source           = "
                  << (m_xp.empty() ? "flat (no file)" : m_file));
        FWT_DEBUG("n_points         = " << m_xp.size());
        if (!m_xp.empty()) {
            const auto xmm = std::minmax_element(m_xp.begin(), m_xp.end());
            const auto ymm = std::minmax_element(m_yp.begin(), m_yp.end());
            const auto zmm = std::minmax_element(m_zp.begin(), m_zp.end());
            FWT_DEBUG("point x range    = [" << *xmm.first << ", "
                                              << *xmm.second << "] m");
            FWT_DEBUG("point y range    = [" << *ymm.first << ", "
                                              << *ymm.second << "] m");
            FWT_DEBUG("point z range    = [" << *zmm.first << ", "
                                              << *zmm.second << "] m");
        }
        FWT_DEBUG("z_terrain range  = [" << m_z_min << ", " << m_z_max
                                          << "] m   (interpolated to cell columns)");
        FWT_DEBUG("solid cells      = " << m_n_solid << " of " << m_n_total
                  << "   (" << (100.0 * amrex::Real(m_n_solid)
                                / amrex::Real(m_n_total)) << " %)");
        FWT_DEBUG("mask convention  = " << kSolid << " solid, " << kFluid
                                         << " fluid (binary, no volume fractions)");

        // The lowest fluid cell above the tallest column: the first place
        // a mask-boundary error shows up.
        FWT_DEBUG("first z_cc        = " << grid.z_cc().front() << " m");
        FWT_DEBUG("last  z_cc        = " << grid.z_cc().back() << " m");
    }
}

void Terrain::AppendReport (const std::string& filename) const
{
    if (!amrex::ParallelDescriptor::IOProcessor()) { return; }

    std::ofstream os(filename, std::ios::app);
    os << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10);
    os << "# terrain summary (Phase 2)\n";
    os << "terrain_file " << (m_file.empty() ? "none" : m_file) << "\n";
    os << "terrain_n_points " << m_xp.size() << "\n";
    os << "terrain_idw_n_neighbors " << m_idw_k << "\n";
    os << "terrain_idw_exponent " << m_idw_exponent << "\n";
    os << "terrain_z_min " << m_z_min << "\n";
    os << "terrain_z_max " << m_z_max << "\n";
    os << "terrain_n_solid " << m_n_solid << "\n";
    os << "terrain_n_total " << m_n_total << "\n";
    os.close();

    FWT_DEBUG("appended terrain summary to " << filename);
}

} // namespace fwt
