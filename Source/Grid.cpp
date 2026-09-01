#include "Grid.H"

#include <AMReX.H>
#include <AMReX_Print.H>

#include <fstream>
#include <iomanip>
#include <cmath>
#include <limits>

namespace fwt {

namespace {
    // Relative tolerance used when comparing the computed stretched-grid
    // height to the requested domain height.
    constexpr amrex::Real kHeightMatchRelTol = 1.0e-8;
}

void Grid::ReadParameters ()
{
    amrex::ParmParse pp("grid");

    pp.getarr("n_cell", m_n_cell);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_n_cell.size() == 3,
        "grid.n_cell must have 3 entries (nx ny nz)");

    pp.getarr("prob_lo", m_prob_lo);
    pp.getarr("prob_hi", m_prob_hi);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_prob_lo.size() == 3 && m_prob_hi.size() == 3,
        "grid.prob_lo / grid.prob_hi must each have 3 entries");

    pp.get("dz0", m_dz0);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_dz0 > 0.0, "grid.dz0 must be > 0");

    // Default 1.0 reproduces a uniform z grid exactly (backward compatible).
    pp.query("stretching_ratio", m_stretch_ratio);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_stretch_ratio > 0.0,
        "grid.stretching_ratio must be > 0");

    pp.query("max_grid_size", m_max_grid_size);
}

void Grid::BuildVerticalStretching ()
{
    const int nz = m_n_cell[2];
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(nz > 0, "grid.n_cell[2] must be > 0");

    const amrex::Real r = m_stretch_ratio;

    // dz(k) = dz0 * r^k,  k = 0 .. nz-1  (k=0 is the surface-adjacent cell)
    amrex::Vector<amrex::Real> dz(nz);
    amrex::Real H_computed = 0.0;
    amrex::Real rk = 1.0;
    for (int k = 0; k < nz; ++k) {
        dz[k] = m_dz0 * rk;
        H_computed += dz[k];
        rk *= r;
    }

    const amrex::Real H_requested = m_prob_hi[2] - m_prob_lo[2];
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(H_requested > 0.0,
        "grid.prob_hi[2] must be greater than grid.prob_lo[2]");

    const amrex::Real rel_diff = (H_computed - H_requested) / H_requested;

    if (rel_diff > kHeightMatchRelTol) {
        // Overshoot: non-fatal. Warn and adjust the domain top so the
        // Geometry and the stretched grid agree exactly.
        amrex::Print() << "WARNING [Grid]: stretched grid overshoots the "
            "requested domain height.\n"
            << "  requested H (grid.prob_hi[2]-grid.prob_lo[2]) = "
            << H_requested << " m\n"
            << "  computed H  (sum of dz0*r^k, k=0.." << nz-1 << ")   = "
            << H_computed  << " m\n"
            << "  --> overriding grid.prob_hi[2] to "
            << (m_prob_lo[2] + H_computed)
            << " m so the grid and domain top agree exactly.\n";
        m_prob_hi[2] = m_prob_lo[2] + H_computed;

    } else if (rel_diff < -kHeightMatchRelTol) {
        // Undershoot: fatal. The grid does not reach the requested domain
        // top; proceeding would silently corrupt the top BC / mass balance.
        amrex::Print() << "ERROR [Grid]: stretched grid does not reach the "
            "requested domain height.\n"
            << "  requested H (grid.prob_hi[2]-grid.prob_lo[2]) = "
            << H_requested << " m\n"
            << "  computed H  (sum of dz0*r^k, k=0.." << nz-1 << ")   = "
            << H_computed  << " m\n"
            << "  Increase grid.n_cell[2], grid.dz0, or "
               "grid.stretching_ratio.\n";
        amrex::Abort("Grid::BuildVerticalStretching: undershoot of "
                      "requested domain height (fatal, see message above).");
    }
    // else: match within tolerance, proceed with no adjustment.

    // Build z_face (nz+1 entries) and z_cc (nz entries) from prob_lo_z.
    m_z_face.resize(nz + 1);
    m_z_cc.resize(nz);

    m_z_face[0] = m_prob_lo[2];
    for (int k = 0; k < nz; ++k) {
        m_z_face[k+1] = m_z_face[k] + dz[k];
        m_z_cc[k]     = 0.5 * (m_z_face[k] + m_z_face[k+1]);
    }
}

void Grid::BuildAMReXGeometry ()
{
    amrex::IntVect dom_lo(0, 0, 0);
    amrex::IntVect dom_hi(m_n_cell[0]-1, m_n_cell[1]-1, m_n_cell[2]-1);
    amrex::Box domain(dom_lo, dom_hi);

    m_ba.define(domain);
    m_ba.maxSize(m_max_grid_size);

    m_dm.define(m_ba);

    // NOTE: AMReX's Geometry assumes uniform spacing in every direction.
    // We still construct it (needed for MultiFab/MFIter/BoxArray machinery
    // and for the x,y directions which ARE uniform); the true, possibly
    // non-uniform z coordinate is tracked separately via m_z_face/m_z_cc
    // and must be used instead of geom().CellSize(2) anywhere a physical
    // z spacing is needed (Phase 5 Poisson stencil, Phase 7 vertical
    // integrals, Phase 8 output).
    amrex::RealBox real_box({AMREX_D_DECL(m_prob_lo[0], m_prob_lo[1], m_prob_lo[2])},
                             {AMREX_D_DECL(m_prob_hi[0], m_prob_hi[1], m_prob_hi[2])});

    amrex::Array<int,3> is_periodic {0, 0, 0}; // all physical BCs (Phase 4)

    m_geom.define(domain, real_box, amrex::CoordSys::cartesian, is_periodic);
}

void Grid::Build ()
{
    ReadParameters();
    BuildVerticalStretching();  // may abort (undershoot) or adjust prob_hi[2] (overshoot)
    BuildAMReXGeometry();       // uses the possibly-adjusted prob_hi[2]
}

void Grid::WriteReport (const std::string& filename) const
{
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream os(filename);
        os << "# FastWindTerrain Phase 1 grid report\n";
        os << "n_cell " << m_n_cell[0] << " " << m_n_cell[1] << " " << m_n_cell[2] << "\n";
        os << "prob_lo " << m_prob_lo[0] << " " << m_prob_lo[1] << " " << m_prob_lo[2] << "\n";
        os << "prob_hi " << m_prob_hi[0] << " " << m_prob_hi[1] << " " << m_prob_hi[2] << "\n";
        os << "dz0 " << m_dz0 << "\n";
        os << "stretching_ratio " << m_stretch_ratio << "\n";
        os << "dx " << m_geom.CellSize(0) << "\n";
        os << "dy " << m_geom.CellSize(1) << "\n";
        os << "n_boxes " << m_ba.size() << "\n";
        os << "# z_face array (" << m_z_face.size() << " entries)\n";
        os << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10);
        for (int k = 0; k < (int)m_z_face.size(); ++k) {
            os << "z_face " << k << " " << m_z_face[k] << "\n";
        }
        for (int k = 0; k < (int)m_z_cc.size(); ++k) {
            os << "z_cc " << k << " " << m_z_cc[k] << "\n";
        }
        os.close();
    }
}

} // namespace fwt
