#ifndef FWT_TERRAIN_H_
#define FWT_TERRAIN_H_

#include "Grid.H"
#include "Error.H"

#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

#include <string>
#include <vector>

namespace fwt {

// -----------------------------------------------------------------------
// Terrain: the terrain surface z_terrain(i,j) and the binary immersed-
// boundary mask derived from it.
//
// Inputs (terrain.*):
//   file            -- x,y,z point file. If absent, the ground is flat at
//                      `flat_elevation` and no file is read.
//   flat_elevation  -- elevation used when no file is given [m] (default 0)
//   idw_n_neighbors -- k nearest points used by the IDW interpolation
//                      (default 6, matching massconsistent_amr)
//   idw_exponent    -- IDW power p; weight = d^-p (default 2)
//
// The file format is the one massconsistent_amr's read_terrain_file
// accepts: one `x y z` point per line, comma OR whitespace separated,
// '#' comments stripped, and any line that does not parse as three
// numbers skipped (so a leading `x,y,z` header is fine).
//
// z_terrain is a genuinely 2D field, but it is stored in a normal
// cell-centered MultiFab replicated along k. That costs nz times more
// memory than a column array and buys two things worth more at this
// scale: it is directly writable to a plotfile as `terrain_z`, and
// kernels can read it with the same (i,j,k) indexing as every other
// field, with no special-cased k.
//
// The mask is BINARY -- 1 = solid, 0 = fluid -- with no partial volume
// fractions, matching massconsistent_amr's
//     is_solid = (z_cc - z_terrain <= 0)
// A cell whose center lies exactly on the terrain surface is solid.
// -----------------------------------------------------------------------
class Terrain
{
public:

    // Mask values. Named rather than bare 0/1 so the sense of the test
    // is unambiguous at every use site.
    static constexpr int kFluid = 0;
    static constexpr int kSolid = 1;

    // -------------------------------------------------------------------
    // The inputs, as data. Same shape as Grid::Params and for the same
    // reason: ParmParse is process-global and sticky, so the Python path
    // has to be able to describe a case without touching it.
    //
    // The scattered points are part of the Params rather than something
    // Build fetches, so "read them from a file" and "here they are" are
    // two ways of filling one field instead of two code paths through
    // the interpolation. A point cloud handed in from numpy therefore
    // goes through exactly the IDW the file path uses -- which is what
    // makes the two bit-for-bit identical rather than merely similar.
    // -------------------------------------------------------------------
    struct Params
    {
        std::string file;                    // empty -> flat, or points
        amrex::Real flat_elevation = 0.0;
        int         idw_n_neighbors = 6;     // massconsistent_amr default
        amrex::Real idw_exponent    = 2.0;

        // Scattered x,y,z points given directly. When these are set,
        // `file` must be empty: two sources for one thing is a mistake
        // worth reporting rather than a precedence rule to remember.
        std::vector<amrex::Real> xp, yp, zp;

        bool given_flat = false;
        bool given_k    = false;
        bool given_p    = false;

        static Params FromParmParse ();
        void Validate () const;

        bool has_points () const { return !xp.empty(); }
    };

    void Build (const Grid& grid, const Params& params);
    void Build (const Grid& grid);

    // --- Read-only accessors --------------------------------------------
    // terrain_z: cell-centered, 1 component, replicated along k [m].
    const amrex::MultiFab&  z_terrain () const { return m_z_terrain; }
    // mask: cell-centered, 1 component, kSolid / kFluid.
    const amrex::iMultiFab& mask ()      const { return m_mask; }

    // Terrain height per (i,j) column: nx*ny entries, i fastest. Computed
    // identically on every rank, so anything needing the surface height
    // as plain host data (the Phase 3 profile, for one) can read it here
    // instead of gathering it back out of the MultiFab.
    const std::vector<amrex::Real>& column_heights () const { return m_h; }

    bool has_file ()          const { return !m_params.file.empty(); }
    const std::string& file() const { return m_params.file; }
    int  n_points ()          const { return static_cast<int>(m_zp.size()); }
    const Params& params ()   const { return m_params; }

    // Diagnostics filled in by Build(), also written to the grid report.
    amrex::Real z_min ()      const { return m_z_min; }
    amrex::Real z_max ()      const { return m_z_max; }
    long n_solid ()           const { return m_n_solid; }
    long n_total ()           const { return m_n_total; }

    // Appends the terrain summary to an existing ascii grid report.
    void AppendReport (const std::string& filename) const;

    // Inverse-distance-weighted terrain height at (xq, yq), using the k
    // nearest of the scattered input points. Ported from
    // massconsistent_amr's idw_terrain so the two agree point for point.
    // Exposed for testing and reuse; a query landing exactly on an input
    // point returns that point's elevation.
    static amrex::Real InterpolateIDW (amrex::Real xq, amrex::Real yq,
                                       const std::vector<amrex::Real>& xp,
                                       const std::vector<amrex::Real>& yp,
                                       const std::vector<amrex::Real>& zp,
                                       int k, amrex::Real exponent);

    // Reads an x,y,z point file in the format described above. Aborts if
    // the file cannot be opened or contains no parseable point.
    static void ReadPointFile (const std::string& filename,
                               std::vector<amrex::Real>& xp,
                               std::vector<amrex::Real>& yp,
                               std::vector<amrex::Real>& zp);

private:

    void BuildTerrainHeight (const Grid& grid);   // fills m_z_terrain
    void BuildMask (const Grid& grid);            // fills m_mask

    // --- inputs ---
    Params m_params;

    // --- scattered input points (empty for the flat case) ---
    std::vector<amrex::Real> m_xp, m_yp, m_zp;

    // --- derived ---
    std::vector<amrex::Real> m_h;        // per-column height, nx*ny
    amrex::MultiFab  m_z_terrain;
    amrex::iMultiFab m_mask;

    amrex::Real m_z_min = 0.0;
    amrex::Real m_z_max = 0.0;
    long m_n_solid = 0;
    long m_n_total = 0;
};

} // namespace fwt

#endif
