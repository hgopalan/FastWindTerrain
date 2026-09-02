#include "Grid.H"
#include "Debug.H"

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
    const bool got_ratio = pp.query("stretching_ratio", m_stretch_ratio);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_stretch_ratio > 0.0,
        "grid.stretching_ratio must be > 0");

    const bool got_mgs = pp.query("max_grid_size", m_max_grid_size);

    // Echo the resolved inputs, marking the ones that fell back to a
    // default -- an unspecified default is exactly what tends to be
    // missed when a run does not do what the input file suggests.
    FWT_DEBUG_SECTION("Grid inputs (grid.*)");
    FWT_DEBUG("n_cell           = " << m_n_cell[0] << " " << m_n_cell[1]
                                     << " " << m_n_cell[2]);
    FWT_DEBUG("prob_lo          = " << m_prob_lo[0] << " " << m_prob_lo[1]
                                     << " " << m_prob_lo[2] << " m");
    FWT_DEBUG("prob_hi          = " << m_prob_hi[0] << " " << m_prob_hi[1]
                                     << " " << m_prob_hi[2] << " m  (as requested)");
    FWT_DEBUG("dz0              = " << m_dz0 << " m");
    FWT_DEBUG("stretching_ratio = " << m_stretch_ratio
                                     << (got_ratio ? "" : "   [default]"));
    FWT_DEBUG("max_grid_size    = " << m_max_grid_size
                                     << (got_mgs ? "" : "   [default]"));
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

    FWT_DEBUG_SECTION("Vertical stretching");
    FWT_DEBUG("nz               = " << nz);
    FWT_DEBUG("dz(0)            = " << dz[0] << " m   (surface-adjacent)");
    FWT_DEBUG("dz(nz-1)         = " << dz[nz-1] << " m   (domain top)");
    FWT_DEBUG("H_requested      = " << H_requested << " m");
    FWT_DEBUG("H_computed       = " << H_computed << " m   (sum of dz0*r^k)");
    FWT_DEBUG("relative diff    = " << rel_diff
                                     << "   (match tolerance "
                                     << kHeightMatchRelTol << ")");
    FWT_DEBUG("height check     = "
              << (rel_diff > kHeightMatchRelTol
                      ? "overshoot -> prob_hi[2] will be overridden"
                      : (rel_diff < -kHeightMatchRelTol
                             ? "undershoot -> fatal"
                             : "exact match -> no adjustment")));

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

    if (Debug::Enabled()) {
        FWT_DEBUG_SECTION("z table (k, z_face[k], dz[k], z_cc[k]) [m]");
        for (int k = 0; k < nz; ++k) {
            if (!Debug::ShowRow(k, nz)) {
                if (k == Debug::kMaxTableRows/2) {
                    FWT_DEBUG("  ... " << nz - Debug::kMaxTableRows
                                        << " rows elided ...");
                }
                continue;
            }
            FWT_DEBUG("  " << k << "  " << m_z_face[k] << "  " << dz[k]
                            << "  " << m_z_cc[k]);
        }
        FWT_DEBUG("  " << nz << "  " << m_z_face[nz]
                        << "   (top face; no cell above)");
    }
}

void Grid::BuildAMReXGeometry ()
{
    amrex::IntVect dom_lo(0, 0, 0);
    amrex::IntVect dom_hi(m_n_cell[0]-1, m_n_cell[1]-1, m_n_cell[2]-1);
    amrex::Box domain(dom_lo, dom_hi);

    m_ba.define(domain);
    // Split in x and y only: every box spans the FULL height.
    //
    // Anything that integrates up a column -- the O'Brien vertical
    // velocity adjustment, and whatever else wants continuity along z --
    // is non-local in that direction and cannot be done box by box if a
    // column is cut between boxes. Reading past the end of a box that
    // way is undefined behaviour and does not announce itself: it
    // produced values around 1e107 before this was fixed.
    //
    // The cost is nothing in practice. Atmospheric domains have
    // nx, ny >> nz, so the horizontal split is what carries the
    // parallelism anyway.
    m_ba.maxSize(amrex::IntVect(m_max_grid_size, m_max_grid_size,
                                domain.length(2)));

    // Hold the invariant explicitly rather than trusting maxSize: a
    // later change to the decomposition would otherwise reintroduce the
    // bug silently.
    for (int ibox = 0; ibox < m_ba.size(); ++ibox) {
        const amrex::Box& b = m_ba[ibox];
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            b.smallEnd(2) == domain.smallEnd(2) &&
            b.bigEnd(2)   == domain.bigEnd(2),
            "the grid was split in z: every box must span the full height, "
            "or a column integration cannot be done box by box");
    }

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

    if (Debug::Enabled()) {
        FWT_DEBUG_SECTION("AMReX geometry / decomposition");
        FWT_DEBUG("index domain     = " << domain);
        FWT_DEBUG("prob_lo          = " << m_prob_lo[0] << " " << m_prob_lo[1]
                                         << " " << m_prob_lo[2] << " m");
        FWT_DEBUG("prob_hi          = " << m_prob_hi[0] << " " << m_prob_hi[1]
                                         << " " << m_prob_hi[2]
                                         << " m  (post height check)");
        FWT_DEBUG("dx, dy           = " << m_geom.CellSize(0) << ", "
                                         << m_geom.CellSize(1) << " m");
        FWT_DEBUG("nominal dz       = " << m_geom.CellSize(2)
                  << " m   (Geometry is uniform in z; use z_face/z_cc "
                     "for the true spacing)");
        FWT_DEBUG("is_periodic      = " << is_periodic[0] << " "
                                         << is_periodic[1] << " "
                                         << is_periodic[2]
                                         << "   (all physical BCs)");
        FWT_DEBUG("max_grid_size    = " << m_max_grid_size);
        FWT_DEBUG("n_boxes          = " << m_ba.size());
        FWT_DEBUG("n_ranks          = "
                  << amrex::ParallelDescriptor::NProcs());

        const int nbox = static_cast<int>(m_ba.size());
        FWT_DEBUG_SECTION("box list (index, box, rank)");
        for (int i = 0; i < nbox; ++i) {
            if (!Debug::ShowRow(i, nbox)) {
                if (i == Debug::kMaxTableRows/2) {
                    FWT_DEBUG("  ... " << nbox - Debug::kMaxTableRows
                                        << " rows elided ...");
                }
                continue;
            }
            FWT_DEBUG("  " << i << "  " << m_ba[i] << "  rank " << m_dm[i]);
        }

        // Cells per rank: the quantity that actually predicts load imbalance.
        const int nranks = amrex::ParallelDescriptor::NProcs();
        amrex::Vector<long> cells_per_rank(nranks, 0L);
        for (int i = 0; i < nbox; ++i) {
            cells_per_rank[m_dm[i]] += m_ba[i].numPts();
        }
        FWT_DEBUG_SECTION("cells per rank");
        for (int r = 0; r < nranks; ++r) {
            FWT_DEBUG("  rank " << r << "  " << cells_per_rank[r] << " cells");
        }
    }
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
        // Full round-trip precision for every field, not just the z arrays:
        // the regtest checkers compare report values against analytic
        // heights at 1e-6 m, which default 6-digit output cannot satisfy.
        os << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10);
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
        for (int k = 0; k < (int)m_z_face.size(); ++k) {
            os << "z_face " << k << " " << m_z_face[k] << "\n";
        }
        for (int k = 0; k < (int)m_z_cc.size(); ++k) {
            os << "z_cc " << k << " " << m_z_cc[k] << "\n";
        }
        os.close();
    }
    FWT_DEBUG("wrote ascii grid report: " << filename);
}

// This switch says WHICH outputs the run produces, not what format the
// field output is written in -- output.format answers that. The two
// were one thing until the field output gained a plain-text backend, so
// the original value names survive as aliases: ascii = report, plt =
// fields. Every input file written before that keeps working, and a new
// one can say what it means.
Grid::OutputFormat Grid::ParseOutputFormat (const std::string& s)
{
    if (s == "report" || s == "ascii") { return OutputFormat::ascii; }
    if (s == "fields" || s == "plt")   { return OutputFormat::plt;   }
    if (s == "both")                   { return OutputFormat::both;  }
    // Not a debug-only line: an unrecognized format is always fatal.
    amrex::Abort("grid.output_format = '" + s +
                 "' is not recognized (expected report, fields, or both; "
                 "ascii and plt are accepted as aliases for the first two)");
    return OutputFormat::ascii;   // unreachable; silences the compiler
}

} // namespace fwt
