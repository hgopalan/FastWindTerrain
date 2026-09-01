#include "Inflow.H"
#include "Debug.H"

#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

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
    // input point. Same value the terrain interpolation uses.
    constexpr amrex::Real kDistanceEpsilon = 1.0e-12;
}

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

Inflow::Mode Inflow::ParseMode (const std::string& s)
{
    if (s == "powerlaw") { return Mode::powerlaw; }
    if (s == "loglaw")   { return Mode::loglaw;   }
    if (s == "userfile") { return Mode::userfile; }
    amrex::Abort("inflow.mode = '" + s + "' is not recognized "
                 "(expected powerlaw, loglaw, or userfile)");
    return Mode::powerlaw;   // unreachable; silences the compiler
}

void Inflow::ReadParameters ()
{
    amrex::ParmParse pp("inflow");

    pp.query("mode", m_mode_name);
    m_mode = ParseMode(m_mode_name);

    const bool got_u  = pp.query("u_ref", m_u_ref);
    const bool got_v  = pp.query("v_ref", m_v_ref);
    const bool got_zr = pp.query("z_ref", m_z_ref);
    const bool got_a  = pp.query("powerlaw_exponent", m_powerlaw_exponent);
    const bool got_z0 = pp.query("z0", m_z0);
    const bool got_k  = pp.query("idw_n_neighbors", m_idw_k);
    const bool got_p  = pp.query("idw_exponent", m_idw_exponent);
    const bool got_zm = pp.query("z_agl_min", m_z_agl_min);
    pp.query("file", m_file);

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_z_ref > 0.0, "inflow.z_ref must be > 0");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_z0 > 0.0, "inflow.z0 must be > 0");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_idw_k > 0,
        "inflow.idw_n_neighbors must be > 0");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_idw_exponent > 0.0,
        "inflow.idw_exponent must be > 0");

    // The floor defaults to z0: at z_agl = z0 the log law gives exactly
    // zero speed, which is the physically right place to stop.
    if (m_z_agl_min < 0.0) { m_z_agl_min = m_z0; }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_z_agl_min > 0.0,
        "inflow.z_agl_min must be > 0");

    if (m_mode == Mode::userfile) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!m_file.empty(),
            "inflow.mode = userfile requires inflow.file");
    } else {
        // A calm reference wind leaves no inflow face for Phase 4 to
        // classify, so it is rejected here rather than producing a
        // silently empty flow field.
        const amrex::Real speed = std::hypot(m_u_ref, m_v_ref);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(speed > 0.0,
            "inflow.u_ref and inflow.v_ref are both zero: there is no wind "
            "and no inflow face. Set at least one of them.");
    }

    m_speed_ref = std::hypot(m_u_ref, m_v_ref);
    if (m_speed_ref > 0.0) {
        m_dir_x = m_u_ref / m_speed_ref;
        m_dir_y = m_v_ref / m_speed_ref;
    }

    FWT_DEBUG_SECTION("Inflow inputs (inflow.*)");
    FWT_DEBUG("mode             = " << m_mode_name);
    FWT_DEBUG("u_ref, v_ref     = " << m_u_ref << ", " << m_v_ref << " m/s"
              << (got_u || got_v ? "" : "   [default]"));
    FWT_DEBUG("speed_ref        = " << m_speed_ref << " m/s");
    FWT_DEBUG("direction        = (" << m_dir_x << ", " << m_dir_y << ")");
    FWT_DEBUG("z_ref            = " << m_z_ref << " m AGL"
                                     << (got_zr ? "" : "   [default]"));
    if (m_mode == Mode::powerlaw) {
        FWT_DEBUG("powerlaw_exponent= " << m_powerlaw_exponent
                                         << (got_a ? "" : "   [default]"));
    }
    if (m_mode == Mode::loglaw) {
        FWT_DEBUG("z0               = " << m_z0 << " m"
                                         << (got_z0 ? "" : "   [default]"));
    }
    if (m_mode == Mode::userfile) {
        FWT_DEBUG("file             = " << m_file);
        FWT_DEBUG("idw_n_neighbors  = " << m_idw_k << (got_k ? "" : "   [default]"));
        FWT_DEBUG("idw_exponent     = " << m_idw_exponent
                                         << (got_p ? "" : "   [default]"));
    }
    FWT_DEBUG("z_agl_min        = " << m_z_agl_min << " m"
              << (got_zm ? "" : "   [default: z0]"));
}

// ---------------------------------------------------------------------------
// The 1D laws
// ---------------------------------------------------------------------------

amrex::Real Inflow::ProfileSpeed (amrex::Real z_agl) const
{
    // Floor first: z_agl is negative inside terrain, and the log law
    // diverges as z_agl -> 0.
    const amrex::Real z = std::max(z_agl, m_z_agl_min);

    switch (m_mode) {
    case Mode::powerlaw:
        return m_speed_ref * std::pow(z / m_z_ref, m_powerlaw_exponent);
    case Mode::loglaw:
        // massconsistent_amr's form: (z + z0) in the numerator, so the
        // profile reaches exactly zero at z = 0 rather than -infinity.
        return m_speed_ref * std::log((z + m_z0) / m_z0)
                           / std::log((m_z_ref + m_z0) / m_z0);
    case Mode::userfile:
        amrex::Abort("Inflow::ProfileSpeed: mode = userfile is a 3D field, "
                     "not a 1D law; interpolate the file instead");
        return 0.0;
    }
    return 0.0;
}

void Inflow::VelocityAt (amrex::Real x, amrex::Real y, amrex::Real z_agl,
                        amrex::Real& u, amrex::Real& v, amrex::Real& w) const
{
    if (m_mode == Mode::userfile) {
        InterpolateIDW3D(x, y, std::max(z_agl, m_z_agl_min),
                         m_xp, m_yp, m_zp, m_up, m_vp, m_wp,
                         m_idw_k, m_idw_exponent, u, v, w);
    } else {
        const amrex::Real speed = ProfileSpeed(z_agl);
        u = speed * m_dir_x;
        v = speed * m_dir_y;
        w = 0.0;
    }
}

// ---------------------------------------------------------------------------
// The user velocity file
// ---------------------------------------------------------------------------

void Inflow::ReadVelocityFile (const std::string& filename,
                               std::vector<amrex::Real>& xp,
                               std::vector<amrex::Real>& yp,
                               std::vector<amrex::Real>& zp,
                               std::vector<amrex::Real>& up,
                               std::vector<amrex::Real>& vp,
                               std::vector<amrex::Real>& wp,
                               int& n_columns)
{
    std::ifstream f(filename);
    if (!f.is_open()) {
        amrex::Abort("Inflow: cannot open velocity file: " + filename);
    }

    n_columns = 0;
    std::string line;
    while (std::getline(f, line)) {
        const auto pos = line.find('#');
        if (pos != std::string::npos) { line = line.substr(0, pos); }
        std::replace(line.begin(), line.end(), ',', ' ');

        std::istringstream ss(line);
        amrex::Real x, y, z, u, v, w;
        if (!(ss >> x >> y >> z >> u >> v)) {
            continue;      // header or blank line
        }
        // Six columns is the FastWindTerrain format. Five columns is
        // massconsistent_amr's read_velocity_file, which carries no w;
        // accept it with w = 0 so those files still load.
        const bool has_w = static_cast<bool>(ss >> w);
        if (!has_w) { w = 0.0; }

        const int ncol = has_w ? 6 : 5;
        if (n_columns == 0) {
            n_columns = ncol;
        } else if (ncol != n_columns) {
            amrex::Abort("Inflow: " + filename + " mixes " +
                         std::to_string(n_columns) + "- and " +
                         std::to_string(ncol) + "-column rows");
        }

        xp.push_back(x); yp.push_back(y); zp.push_back(z);
        up.push_back(u); vp.push_back(v); wp.push_back(w);
    }

    if (xp.empty()) {
        amrex::Abort("Inflow: no data read from velocity file: " + filename);
    }
}

void Inflow::InterpolateIDW3D (amrex::Real xq, amrex::Real yq, amrex::Real zq,
                               const std::vector<amrex::Real>& xp,
                               const std::vector<amrex::Real>& yp,
                               const std::vector<amrex::Real>& zp,
                               const std::vector<amrex::Real>& up,
                               const std::vector<amrex::Real>& vp,
                               const std::vector<amrex::Real>& wp,
                               int k, amrex::Real exponent,
                               amrex::Real& u, amrex::Real& v, amrex::Real& w)
{
    const int n = static_cast<int>(xp.size());
    k = std::min(k, n);

    std::vector<std::pair<amrex::Real,int>> d2(n);
    for (int i = 0; i < n; ++i) {
        const amrex::Real dx = xp[i] - xq;
        const amrex::Real dy = yp[i] - yq;
        const amrex::Real dz = zp[i] - zq;
        d2[i] = {dx*dx + dy*dy + dz*dz, i};
    }
    std::partial_sort(d2.begin(), d2.begin() + k, d2.end());

    amrex::Real wsum = 0.0;
    u = v = w = 0.0;
    for (int i = 0; i < k; ++i) {
        const int p = d2[i].second;
        if (d2[i].first < kDistanceEpsilon) {
            u = up[p]; v = vp[p]; w = wp[p];    // exact hit
            return;
        }
        const amrex::Real weight =
            std::pow(d2[i].first, -exponent / amrex::Real(2.0));
        wsum += weight;
        u += weight * up[p];
        v += weight * vp[p];
        w += weight * wp[p];
    }
    u /= wsum; v /= wsum; w /= wsum;
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

void Inflow::BuildVelocity (const Grid& grid, const Terrain& terrain)
{
    // One ghost layer: the physical boundary conditions live there.
    m_vel.define(grid.ba(), grid.dm(), 3, 1);

    const int nx = grid.nx();
    const int ny = grid.ny();
    const int nz = grid.nz();
    const amrex::Real dx = grid.geom().CellSize(0);
    const amrex::Real dy = grid.geom().CellSize(1);
    const amrex::Real xlo = grid.geom().ProbLo(0);
    const amrex::Real ylo = grid.geom().ProbLo(1);

    // The profile is evaluated on the host -- both the 1D laws and the
    // scattered-point search are host work -- into a full (nx,ny,nz,3)
    // buffer, then copied to the device once. The buffer is the same size
    // as the field itself, which at these grid sizes is cheaper than
    // making the interpolation device-callable.
    const std::size_t ncell = std::size_t(nx) * ny * nz;
    std::vector<amrex::Real> buf(3 * ncell);

    // Terrain keeps the per-column heights as host data, computed the
    // same way on every rank, so no gather out of the MultiFab is needed.
    const std::vector<amrex::Real>& h = terrain.column_heights();

    const amrex::Vector<amrex::Real>& z_cc = grid.z_cc();

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            const amrex::Real yq = ylo + (amrex::Real(j) + 0.5) * dy;
            for (int i = 0; i < nx; ++i) {
                const amrex::Real xq = xlo + (amrex::Real(i) + 0.5) * dx;
                const amrex::Real z_agl = z_cc[k] - h[std::size_t(j)*nx + i];

                // One evaluation path for the interior and for the
                // inflow ghost cells the boundary conditions fill.
                amrex::Real u, v, w;
                VelocityAt(xq, yq, z_agl, u, v, w);

                const std::size_t c = (std::size_t(k)*ny + j)*nx + i;
                buf[0*ncell + c] = u;
                buf[1*ncell + c] = v;
                buf[2*ncell + c] = w;
            }
        }
    }

    amrex::Gpu::DeviceVector<amrex::Real> d_buf(buf.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, buf.begin(), buf.end(),
                          d_buf.begin());
    amrex::Gpu::streamSynchronize();
    const amrex::Real* pbuf = d_buf.data();

    const int solid = Terrain::kSolid;
    const std::size_t ncell_c = ncell;

    for (amrex::MFIter mfi(m_vel); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const& vel = m_vel.array(mfi);
        auto const& mk  = terrain.mask().const_array(mfi);
        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const std::size_t c = (std::size_t(k)*ny + j)*nx + i;
            // No flow inside the terrain: the IB and the profile have to
            // agree from the start, not only after the projection.
            const bool is_solid = (mk(i,j,k) == solid);
            vel(i,j,k,0) = is_solid ? amrex::Real(0.0) : pbuf[0*ncell_c + c];
            vel(i,j,k,1) = is_solid ? amrex::Real(0.0) : pbuf[1*ncell_c + c];
            vel(i,j,k,2) = is_solid ? amrex::Real(0.0) : pbuf[2*ncell_c + c];
        });
    }
}

void Inflow::ComputeBoundaryFlux (const Grid& grid, const Terrain& terrain)
{
    // Outward flux through every open (fluid) boundary face, using the
    // stretched dz(k). The lateral faces plus the domain top can carry
    // flux; the bottom is ground.
    const amrex::Box& domain = grid.geom().Domain();
    const amrex::Real dx = grid.geom().CellSize(0);
    const amrex::Real dy = grid.geom().CellSize(1);

    const amrex::Vector<amrex::Real>& z_face = grid.z_face();
    amrex::Gpu::DeviceVector<amrex::Real> d_zf(z_face.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, z_face.begin(),
                          z_face.end(), d_zf.begin());
    amrex::Gpu::streamSynchronize();
    const amrex::Real* zf = d_zf.data();

    struct Face { int dir; int side; amrex::Real fac; bool use_dz; };
    const Face faces[5] = {
        {0, -1, dy,      true },   // xlo
        {0, +1, dy,      true },   // xhi
        {1, -1, dx,      true },   // ylo
        {1, +1, dx,      true },   // yhi
        {2, +1, dx * dy, false},   // top
    };

    const int solid = Terrain::kSolid;
    amrex::Real flux_out = 0.0;
    amrex::Real flux_in  = 0.0;

    for (const Face& f : faces) {
        amrex::Box layer(domain);
        if (f.side < 0) {
            layer.setBig(f.dir, domain.smallEnd(f.dir));
        } else {
            layer.setSmall(f.dir, domain.bigEnd(f.dir));
        }

        amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_op;
        amrex::ReduceData<amrex::Real, amrex::Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

        const int dir = f.dir;
        const amrex::Real sign = amrex::Real(f.side);
        const amrex::Real fac = f.fac;
        const bool use_dz = f.use_dz;

        for (amrex::MFIter mfi(m_vel); mfi.isValid(); ++mfi) {
            const amrex::Box sect = mfi.tilebox() & layer;
            if (!sect.ok()) { continue; }

            auto const& vel = m_vel.const_array(mfi);
            auto const& mk  = terrain.mask().const_array(mfi);

            reduce_op.eval(sect, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
            {
                if (mk(i,j,k) == solid) { return {0.0, 0.0}; }
                const amrex::Real area =
                    fac * (use_dz ? (zf[k+1] - zf[k]) : amrex::Real(1.0));
                const amrex::Real flux = sign * vel(i,j,k,dir) * area;
                return {amrex::max(flux, amrex::Real(0.0)),
                        amrex::min(flux, amrex::Real(0.0))};
            });
        }

        ReduceTuple hv = reduce_data.value(reduce_op);
        flux_out += amrex::get<0>(hv);
        flux_in  -= amrex::get<1>(hv);      // stored negative; report as +
    }

    amrex::ParallelDescriptor::ReduceRealSum(flux_out);
    amrex::ParallelDescriptor::ReduceRealSum(flux_in);

    m_flux_out = flux_out;
    m_flux_in  = flux_in;
    m_flux_net = flux_out - flux_in;

    const amrex::Real scale = std::max(std::abs(flux_in), std::abs(flux_out));
    m_flux_imbalance = (scale > 0.0) ? std::abs(m_flux_net) / scale : 0.0;
}

void Inflow::Build (const Grid& grid, const Terrain& terrain)
{
    ReadParameters();

    if (m_mode == Mode::userfile) {
        ReadVelocityFile(m_file, m_xp, m_yp, m_zp, m_up, m_vp, m_wp,
                         m_n_columns);
        amrex::Print() << "Inflow: read " << m_xp.size() << " velocity points ("
                       << m_n_columns << " columns) from " << m_file << "\n";
    }

    BuildVelocity(grid, terrain);
    ComputeBoundaryFlux(grid, terrain);

    if (Debug::Enabled()) {
        FWT_DEBUG_SECTION("Inflow profile");
        FWT_DEBUG("mode             = " << m_mode_name);
        if (m_mode != Mode::userfile) {
            // A few heights AGL, so the shape of the law is visible.
            for (amrex::Real z : {1.0, 10.0, 50.0, 100.0, 500.0}) {
                FWT_DEBUG("  speed(" << z << " m AGL) = "
                                      << ProfileSpeed(z) << " m/s");
            }
        } else {
            FWT_DEBUG("n_points         = " << m_xp.size());
            FWT_DEBUG("n_columns        = " << m_n_columns
                      << (m_n_columns == 5 ? "   [w taken as 0]" : ""));
        }

        FWT_DEBUG_SECTION("Boundary mass flux (open faces only)");
        FWT_DEBUG("flux_in          = " << m_flux_in << " m^3/s");
        FWT_DEBUG("flux_out         = " << m_flux_out << " m^3/s");
        FWT_DEBUG("flux_net         = " << m_flux_net << " m^3/s");
        FWT_DEBUG("relative imbalance = " << m_flux_imbalance);
        FWT_DEBUG("note: an imbalance here is expected when terrain blocks "
                  "part of a face. It is a diagnostic, not an error -- the "
                  "mass-consistent solve is the correction.");
    }
}

void Inflow::AppendReport (const std::string& filename) const
{
    if (!amrex::ParallelDescriptor::IOProcessor()) { return; }

    std::ofstream os(filename, std::ios::app);
    os << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10);
    os << "# inflow summary (Phase 3)\n";
    os << "inflow_mode " << m_mode_name << "\n";
    os << "inflow_u_ref " << m_u_ref << "\n";
    os << "inflow_v_ref " << m_v_ref << "\n";
    os << "inflow_speed_ref " << m_speed_ref << "\n";
    os << "inflow_z_ref " << m_z_ref << "\n";
    os << "inflow_powerlaw_exponent " << m_powerlaw_exponent << "\n";
    os << "inflow_z0 " << m_z0 << "\n";
    os << "inflow_z_agl_min " << m_z_agl_min << "\n";
    os << "inflow_file " << (m_file.empty() ? "none" : m_file) << "\n";
    os << "inflow_n_points " << m_xp.size() << "\n";
    os << "inflow_n_columns " << m_n_columns << "\n";
    os << "inflow_flux_in " << m_flux_in << "\n";
    os << "inflow_flux_out " << m_flux_out << "\n";
    os << "inflow_flux_net " << m_flux_net << "\n";
    os << "inflow_flux_imbalance " << m_flux_imbalance << "\n";
    os.close();

    FWT_DEBUG("appended inflow summary to " << filename);
}

} // namespace fwt
