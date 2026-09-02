#ifndef FWT_OUTPUT_H_
#define FWT_OUTPUT_H_

#include "Grid.H"
#include "Terrain.H"
#include "Inflow.H"
#include "Poisson.H"
#include "Anisotropy.H"

#include <AMReX_Vector.H>

#include <string>

namespace fwt {

// -----------------------------------------------------------------------
// Output: the field output, in either of two backends.
//
// THE ONE SOURCE. Both backends are handed the SAME object -- the
// MultiFab and the names that CollectOutputFields built -- rather than
// each assembling its own copy of "the fields we write". Two backends
// that each gather their own fields drift: one gains a component, the
// other does not, and the regtest that reads the plain-text file stops
// testing the file a user actually looks at. Here they cannot, because
// there is only one gather and a regtest compares the two files cell by
// cell.
//
// Selected by output.format:
//
//   plt   -- AMReX native plotfile (default, the production path).
//            Parallel, binary, one directory of per-rank FABs.
//   ascii -- ONE plain-text file, gathered to rank 0, one row per cell.
//            A REGTEST AID: it is serial, it is O(50) bytes per cell,
//            and it has no place in a production case. It exists so a
//            checker can read the fields with nothing but the standard
//            library.
//   both  -- both, which is what the cross-check regtest runs.
//
// Note that grid.output_format is a different switch and answers a
// different question: WHICH outputs the run produces (the text summary
// report, the field output, or both). output.format chooses the format
// of the field output itself.
// -----------------------------------------------------------------------

// Components, in order. The first fifteen are the ones the plotfile has
// carried since Phase 7 and are not reordered -- the earlier checkers
// index them by name, but a user's saved ParaView state does not.
//
//   z_cc      -- true (stretched) cell-center height [m]
//   dz        -- true cell thickness [m]
//   terrain_z -- terrain surface height for the column [m]
//   mask      -- 1 solid, 0 fluid (binary immersed boundary)
//   u, v, w    -- velocity after the projection [m/s]
//   sigma_x/y/z -- Poisson operator coefficients, metric included
//   u0, v0, w0 -- the initial field, before the correction [m/s]
//   alpha_h, alpha_v -- the cell-local variational weights
//   lambda    -- the potential, averaged from its eight surrounding
//                nodes to the cell centre [m^2/s]
//   divergence -- discrete div(u) after the projection [1/s]
//
// z_cc/dz are written because AMReX's Geometry is uniform in z, so the
// plotfile's own vertical coordinate is only nominal and a visualization
// tool needs these fields to see where the cells actually sit.
struct OutputFields
{
    amrex::MultiFab mf;                    // cell centered, no ghosts
    amrex::Vector<std::string> names;

    int ncomp () const { return int(names.size()); }
};

// Gathers every output field into one MultiFab. div is the per-cell
// divergence Poisson computed; passing it in rather than recomputing it
// keeps the field in the file identical to the number in the report.
OutputFields CollectOutputFields (const Grid& grid,
                                  const Terrain& terrain,
                                  const Inflow& inflow,
                                  const Poisson& poisson,
                                  const amrex::MultiFab& vel0,
                                  const Anisotropy& aniso,
                                  const amrex::MultiFab& div);

void WritePlotfile (const std::string& plotfilename,
                    const Grid& grid,
                    const OutputFields& fields);

// One gathered plain-text file, never per-rank or per-box. Rows are
// ordered with i fastest and k slowest; columns are named in the header.
void WriteAscii (const std::string& filename,
                 const Grid& grid,
                 const OutputFields& fields);

// -----------------------------------------------------------------------
// Where the output goes, as data -- the last module to stop reading
// ParmParse directly.
//
// It matters for the same reason the others did: a dataset generator
// that let ParmParse decide would have case 2 write over case 1's
// plotfile, or write nothing, depending on what an earlier run had set.
// -----------------------------------------------------------------------
struct OutputParams
{
    // grid.output_format: which outputs the run produces.
    std::string which = "report";        // report | fields | both
    // output.format: which backend writes the field output.
    std::string format = "plt";          // plt | ascii | both

    std::string report_file = "grid_report.txt";
    std::string plot_file   = "plt_grid";
    std::string ascii_file  = "fields.txt";

    static OutputParams FromParmParse ();
    void Validate () const;
};

// output.format
enum class FieldFormat { plt, ascii, both };
FieldFormat ParseFieldFormat (const std::string& s);
inline bool WantsFieldPlt   (FieldFormat f) { return f != FieldFormat::ascii; }
inline bool WantsFieldAscii (FieldFormat f) { return f != FieldFormat::plt; }

} // namespace fwt

#endif
