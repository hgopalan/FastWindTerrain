#ifndef FWT_GRID_H_
#define FWT_GRID_H_

#include <AMReX_Geometry.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Vector.H>
#include <AMReX_ParmParse.H>

namespace fwt {

// -----------------------------------------------------------------------
// Grid: builds the AMReX BoxArray/DistributionMapping/Geometry for a
// Cartesian mesh with uniform x,y spacing and an optionally-stretched
// z spacing (geometric stretching from a surface-adjacent cell size).
//
// Free inputs (grid.*):
//   n_cell_x, n_cell_y, n_cell_z   -- cell counts
//   prob_lo_x/y/z, prob_hi_x/y/z   -- requested domain extents
//   dz0                             -- first (surface-adjacent) cell
//                                      thickness in z [m]
//   stretching_ratio                -- geometric ratio r for dz(k) =
//                                      dz0 * r^k (default 1.0 = uniform)
//   max_grid_size                   -- AMReX box splitting size
//
// Validation policy (see Grid::Build):
//   H_computed = sum_{k=0}^{nz-1} dz0 * r^k
//   H_requested = prob_hi_z - prob_lo_z
//
//   H_computed > H_requested  -> WARNING, non-fatal. prob_hi_z is
//                                 overridden to prob_lo_z + H_computed
//                                 so the grid and domain agree exactly.
//   H_computed < H_requested  -> FATAL. amrex::Abort(); grid does not
//                                 reach the requested domain top.
//   H_computed == H_requested (to floating point tolerance) -> normal.
// -----------------------------------------------------------------------
class Grid
{
public:

    void Build ();

    // Read-only accessors -------------------------------------------------
    const amrex::Geometry&            geom ()   const { return m_geom; }
    const amrex::BoxArray&            ba ()     const { return m_ba; }
    const amrex::DistributionMapping& dm ()     const { return m_dm; }

    int nx () const { return m_n_cell[0]; }
    int ny () const { return m_n_cell[1]; }
    int nz () const { return m_n_cell[2]; }

    // z_face has nz+1 entries (cell faces, k=0..nz), z_cc has nz entries
    // (cell centers). Both are measured from prob_lo_z. x,y are uniform
    // and can be queried directly from geom().
    const amrex::Vector<amrex::Real>& z_face () const { return m_z_face; }
    const amrex::Vector<amrex::Real>& z_cc   () const { return m_z_cc; }

    amrex::Real dz0 ()               const { return m_dz0; }
    amrex::Real stretching_ratio ()  const { return m_stretch_ratio; }

    // Output (selected by grid.output_format = ascii | plt | both;
    // see Grid::OutputFormat / main.cpp):
    //
    //   ascii -- plain-text summary of the grid (parameters + full
    //            z_face/z_cc arrays). Used directly by the Phase 1
    //            regtest checker, so it stays the default.
    //   plt   -- AMReX native plotfile carrying the true stretched
    //            vertical coordinate and the terrain fields.
    // The plotfile is written by fwt::WritePlotfile (Output.H), which
    // combines the grid fields with the terrain fields into one file.
    void WriteReport (const std::string& filename) const;

    // Parses grid.output_format; aborts on an unrecognized value.
    enum class OutputFormat { ascii, plt, both };
    static OutputFormat ParseOutputFormat (const std::string& s);
    static bool WantsAscii (OutputFormat f) { return f != OutputFormat::plt; }
    static bool WantsPlt   (OutputFormat f) { return f != OutputFormat::ascii; }

private:

    void ReadParameters ();
    void BuildVerticalStretching ();   // fills m_z_face, m_z_cc; may
                                        // adjust m_prob_hi[2]
    void BuildAMReXGeometry ();        // constructs m_ba, m_dm, m_geom

    // --- inputs ---
    amrex::Vector<int>  m_n_cell   {{0,0,0}};
    amrex::Vector<amrex::Real> m_prob_lo {{0.,0.,0.}};
    amrex::Vector<amrex::Real> m_prob_hi {{0.,0.,0.}};
    amrex::Real m_dz0            = 0.0;
    amrex::Real m_stretch_ratio  = 1.0;
    int         m_max_grid_size  = 32;

    // --- derived ---
    amrex::Vector<amrex::Real> m_z_face;
    amrex::Vector<amrex::Real> m_z_cc;

    amrex::Geometry            m_geom;
    amrex::BoxArray            m_ba;
    amrex::DistributionMapping m_dm;
};

} // namespace fwt

#endif
