#include "Surface.H"
#include "Debug.H"
#include "Error.H"

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>

namespace fwt {

// ---------------------------------------------------------------------------
// THIS IS NOT A BODY-FITTED WALL, and the implementation has to respect that.
//
// The terrain is an immersed boundary on a Cartesian mesh with a BINARY mask
// (Terrain.cpp:206) -- no cut cells, no volume fractions. The surface is a
// staircase through the grid, and three consequences follow that a
// body-fitted wall function never has to think about:
//
// 1. DISTANCE IS PERPENDICULAR, NOT VERTICAL. z_cc[k] - h is the vertical
//    gap to the surface. The log law wants the distance normal to it, which
//    is (z_cc[k] - h) * n_z. On a slope of 1.0 -- 45 degrees, and the
//    corpus reaches 1.89 -- those differ by 30%, and the difference goes
//    straight into ln(z/z0).
//
// 2. THE VELOCITY IS THE SURFACE-PARALLEL ONE, not the horizontal one. The
//    log law describes flow along the surface. On a steep slope the
//    horizontal speed includes a component running into or out of the
//    ground, and scaling that is scaling the wrong quantity.
//
// 3. THE FIRST FLUID CELL SITS AT AN ARBITRARY HEIGHT. With a binary mask
//    its centre is anywhere from just above the surface to a full cell
//    above it, and that height jumps between neighbouring columns. So the
//    correction cannot assume a fixed fraction of a cell, and every height
//    used here is computed per column from the terrain the solver actually
//    built.
//
// The decomposition is the standard one, and the same shape massconsistent_amr
// uses in src/wall_functions.H (apply_terrain_wall_function): split into
// normal and parallel, act on the parallel part, put the normal part back at
// zero. Two differences, both deliberate:
//
//   * u* comes from the SECOND FLUID CELL of this column, not from a single
//     domain-wide reference. A global u* cannot know that a ridge crest is
//     accelerated, which is the whole point over complex terrain.
//   * the distance is perpendicular, not vertical.
//
// THIS CREATES A VERTICAL VELOCITY, and that is the condition working rather
// than a side effect. Flow tangential to a slope has to climb or descend it,
// so removing the normal component leaves w_par satisfying
//
//     u_par . n = 0   <=>   w_par = u_par dh/dx + v_par dh/dy
//
// which is the kinematic surface condition. The Python side computes the
// same quantity in levels.surface_kinematic_w.
//
// WHICH IS WHY THIS RUNS BEFORE O'BRIEN. The profile sets w = 0 in every
// cell (Inflow.cpp:209), so O'Brien's column integration has been starting
// from zero at the surface; phase 19 measured that seeding it with the
// kinematic value instead helps on gentle terrain and hurts on very steep
// terrain, where the flow is being pushed around the obstacle rather than
// over it. With the surface condition applied first the seed IS the
// kinematic value, by construction and not as an extra option.
// ---------------------------------------------------------------------------

void Surface::Params::Validate () const
{
    if (given_z0 && z0 <= 0.0) {
        throw InputError("surface.z0 must be > 0");
    }
    if (kappa <= 0.0) {
        throw InputError("surface.kappa must be > 0");
    }
}

Surface::Type Surface::TypeFromString (const std::string& s)
{
    if (s == "none")          { return Type::none; }
    if (s == "slip")          { return Type::slip; }
    if (s == "noslip")        { return Type::noslip; }
    if (s == "wall_function") { return Type::wall_function; }
    throw InputError(
        "surface.type must be one of 'wall_function', 'slip', 'noslip' or "
        "'none', got '" + s + "'");
}

const char* Surface::TypeToString (Type t)
{
    switch (t) {
    case Type::none:          return "none";
    case Type::slip:          return "slip";
    case Type::noslip:        return "noslip";
    case Type::wall_function: return "wall_function";
    }
    return "wall_function";
}

Surface::Params Surface::ParseParams ()
{
    Params p;
    // A literal prefix, not a parameter: the master-inputs regtest greps
    // ParmParse constructions out of Source/ to prove inputs_master lists
    // every input the code reads, and it can only see a literal.
    amrex::ParmParse pp("surface");

    std::string type = TypeToString(p.type);
    pp.query("type", type);
    p.type = TypeFromString(type);

    std::string apply = "initial";
    pp.query("apply", apply);
    if (apply == "initial")   { p.apply = Apply::initial; }
    else if (apply == "both") { p.apply = Apply::both; }
    else {
        throw InputError("surface.apply must be 'initial' or 'both', got '"
                         + apply + "'");
    }

    p.given_z0 = pp.query("z0", p.z0);
    pp.query("kappa", p.kappa);

    p.Validate();
    return p;
}

void Surface::Build (const Grid& grid, const Terrain& terrain,
                     const Params& params, amrex::Real inflow_z0)
{
    m_params = params;
    m_params.Validate();

    // One roughness in a run. Taking inflow's unless told otherwise means a
    // case cannot quietly end up with a wall function at one z0 and a
    // profile at another.
    m_z0 = m_params.given_z0 ? m_params.z0 : inflow_z0;
    if (m_params.type == Type::wall_function && m_z0 <= 0.0) {
        throw InputError(
            "surface.type = wall_function needs a roughness length > 0; "
            "inflow.z0 is " + std::to_string(inflow_z0) + " and surface.z0 "
            "was not given");
    }

    FWT_DEBUG_SECTION("Surface condition (surface.*)");
    FWT_DEBUG("type             = " << TypeToString(m_params.type));
    if (m_params.type == Type::none) {
        FWT_DEBUG("note: the first fluid cell above terrain is left exactly "
                  "as the profile set it, including any flow into the "
                  "surface.");
        return;
    }
    FWT_DEBUG("apply            = "
              << (m_params.apply == Apply::both ? "both" : "initial"));
    if (m_params.type == Type::wall_function) {
        FWT_DEBUG("z0               = " << m_z0 << " m"
                  << (m_params.given_z0 ? "" : "   [from inflow.z0]"));
        FWT_DEBUG("kappa            = " << m_params.kappa);
    }

    const int nx = grid.nx();
    const int ny = grid.ny();
    const amrex::Real dx = grid.geom().CellSize(0);
    const amrex::Real dy = grid.geom().CellSize(1);
    const std::vector<amrex::Real>& h = terrain.column_heights();

    // Surface normals from the terrain column heights, with the stencil
    // Anisotropy uses for slope (Anisotropy.cpp:125) -- central differences,
    // one-sided at the edges -- so the terrain gradient has one definition
    // in this code and not two that could drift apart.
    m_nx.assign(std::size_t(nx) * ny, 0.0);
    m_ny.assign(std::size_t(nx) * ny, 0.0);
    m_nz.assign(std::size_t(nx) * ny, 1.0);

    amrex::Real min_nz = 1.0;
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

            // n = (-dh/dx, -dh/dy, 1) normalised: the outward normal of the
            // surface z = h(x, y). n_z is cos of the slope angle, which is
            // also the factor turning a vertical gap into a perpendicular
            // distance.
            const amrex::Real inv =
                1.0 / std::sqrt(1.0 + dhdx*dhdx + dhdy*dhdy);
            const std::size_t c = std::size_t(j)*nx + i;
            m_nx[c] = -dhdx * inv;
            m_ny[c] = -dhdy * inv;
            m_nz[c] =  inv;
            min_nz = std::min(min_nz, inv);
        }
    }
    FWT_DEBUG("steepest column  = " << std::sqrt(1.0/(min_nz*min_nz) - 1.0)
              << " slope, n_z " << min_nz
              << "   (perpendicular distance is n_z x the vertical gap)");
}

long Surface::ApplyTo (const Grid& grid, const Terrain& terrain,
                       amrex::MultiFab& vel) const
{
    m_n_columns  = 0;
    m_max_ustar  = 0.0;
    m_max_normal = 0.0;
    m_max_dspeed = 0.0;
    if (m_params.type == Type::none) { return 0; }

    const int nx = grid.nx();
    const amrex::Vector<amrex::Real>& z_cc = grid.z_cc();
    const std::vector<amrex::Real>& h = terrain.column_heights();
    const int solid = Terrain::kSolid;
    const amrex::Real kappa = m_params.kappa;
    const amrex::Real z0 = m_z0;
    const Type type = m_params.type;

    // A host loop over columns, like the O'Brien pass and the ghost fill.
    // It touches two cells per column and is nowhere near the solve; see
    // docs/index.rst on which paths are host loops and why.
    for (amrex::MFIter mfi(vel); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const& a  = vel.array(mfi);
        auto const& mk = terrain.mask().const_array(mfi);

        const auto lo = amrex::lbound(bx);
        const auto hi = amrex::ubound(bx);
        for (int j = lo.y; j <= hi.y; ++j) {
        for (int i = lo.x; i <= hi.x; ++i) {
            // The first two fluid cells in this column. With a binary mask
            // the first can sit anywhere from just above the surface to a
            // whole cell above it, so nothing here assumes a fixed height.
            int k1 = -1, k2 = -1;
            for (int k = lo.z; k <= hi.z; ++k) {
                if (mk(i,j,k) == solid) { continue; }
                if (k1 < 0) { k1 = k; } else { k2 = k; break; }
            }
            if (k1 < 0) { continue; }        // column solid throughout

            const std::size_t c = std::size_t(j)*nx + i;
            ++m_n_columns;

            if (type == Type::noslip) {
                a(i,j,k1,0) = 0.0;
                a(i,j,k1,1) = 0.0;
                a(i,j,k1,2) = 0.0;
                continue;
            }

            const amrex::Real nxc = m_nx[c], nyc = m_ny[c], nzc = m_nz[c];

            // -- decompose: normal, then what is left is parallel ---------
            const amrex::Real un = a(i,j,k1,0)*nxc + a(i,j,k1,1)*nyc
                                 + a(i,j,k1,2)*nzc;
            amrex::Real up = a(i,j,k1,0) - un*nxc;
            amrex::Real vp = a(i,j,k1,1) - un*nyc;
            amrex::Real wp = a(i,j,k1,2) - un*nzc;
            m_max_normal = std::max(m_max_normal, std::abs(un));

            if (type == Type::wall_function && k2 >= 0) {
                // PERPENDICULAR distances, not vertical gaps: the surface is
                // sloped and the log law is normal to it.
                const amrex::Real d1 = (z_cc[k1] - h[c]) * nzc;
                const amrex::Real d2 = (z_cc[k2] - h[c]) * nzc;

                // The anchor is the SECOND cell's surface-parallel speed --
                // parallel, because the log law describes flow along the
                // surface, and on a steep slope the horizontal speed is not
                // that.
                const amrex::Real un2 = a(i,j,k2,0)*nxc + a(i,j,k2,1)*nyc
                                      + a(i,j,k2,2)*nzc;
                const amrex::Real up2 = a(i,j,k2,0) - un2*nxc;
                const amrex::Real vp2 = a(i,j,k2,1) - un2*nyc;
                const amrex::Real wp2 = a(i,j,k2,2) - un2*nzc;
                const amrex::Real s2 =
                    std::sqrt(up2*up2 + vp2*vp2 + wp2*wp2);

                const amrex::Real l2 = std::log((d2 + z0) / z0);
                const amrex::Real s1 = std::sqrt(up*up + vp*vp + wp*wp);

                if (d1 > 0.0 && d2 > d1 && s2 > 0.0 && l2 > 0.0) {
                    const amrex::Real ustar = kappa * s2 / l2;
                    const amrex::Real s1_want =
                        (ustar / kappa) * std::log((d1 + z0) / z0);

                    if (s1 > 0.0) {
                        const amrex::Real f = s1_want / s1;
                        up *= f; vp *= f; wp *= f;
                    } else {
                        // No direction of its own: take the cell above's.
                        const amrex::Real f = s1_want / s2;
                        up = up2*f; vp = vp2*f; wp = wp2*f;
                    }
                    m_max_ustar  = std::max(m_max_ustar, ustar);
                    m_max_dspeed = std::max(m_max_dspeed,
                                            std::abs(s1_want - s1));
                }
            }

            // Put it back with no normal component: slip, and wall_function
            // once it has rescaled the parallel part.
            a(i,j,k1,0) = up;
            a(i,j,k1,1) = vp;
            a(i,j,k1,2) = wp;
        }}
    }

    amrex::ParallelDescriptor::ReduceLongSum(m_n_columns);
    amrex::ParallelDescriptor::ReduceRealMax(m_max_ustar);
    amrex::ParallelDescriptor::ReduceRealMax(m_max_normal);
    amrex::ParallelDescriptor::ReduceRealMax(m_max_dspeed);

    FWT_DEBUG("Surface: " << TypeToString(type) << " on " << m_n_columns
              << " columns, max u* " << m_max_ustar
              << " m/s, max normal removed " << m_max_normal
              << " m/s, max |dU_par| " << m_max_dspeed << " m/s");
    return m_n_columns;
}

void Surface::AppendReport (const std::string& filename) const
{
    if (!amrex::ParallelDescriptor::IOProcessor()) { return; }

    std::ofstream os(filename, std::ios::app);
    os << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10);
    os << "# surface\n";
    os << "surface_type " << TypeToString(m_params.type) << "\n";
    os << "surface_apply "
       << (m_params.apply == Apply::both ? "both" : "initial") << "\n";
    os << "surface_z0 " << m_z0 << "\n";
    os << "surface_kappa " << m_params.kappa << "\n";
    os << "surface_n_columns " << m_n_columns << "\n";
    os << "surface_max_ustar " << m_max_ustar << "\n";
    os << "surface_max_normal_removed " << m_max_normal << "\n";
    os << "surface_max_speed_change " << m_max_dspeed << "\n";
    os.close();
}

} // namespace fwt
