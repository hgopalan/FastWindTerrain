#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Version.H>

#include "Grid.H"
#include "Debug.H"

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        amrex::Print() << "FastWindTerrain -- Phase 1: grid & data layout scaffolding\n";

        // fwt.debug = 1 turns on verbose diagnostics everywhere.
        fwt::Debug::Init();
        FWT_DEBUG_SECTION("Run configuration");
        FWT_DEBUG("AMReX version    = " << amrex::Version());
        FWT_DEBUG("MPI ranks        = " << amrex::ParallelDescriptor::NProcs());
        FWT_DEBUG("amrex::Real      = " << sizeof(amrex::Real) * 8 << "-bit");
        for (int i = 1; i < argc; ++i) {
            FWT_DEBUG("argv[" << i << "]         = " << argv[i]);
        }

        fwt::Grid grid;
        grid.Build();   // aborts on undershoot (see Grid::BuildVerticalStretching)

        amrex::Print() << "Grid built: n_cell = ("
                        << grid.nx() << ", " << grid.ny() << ", " << grid.nz()
                        << "), n_boxes = " << grid.ba().size() << "\n";

        // Output is user-selectable: ascii (plain-text grid report),
        // plt (AMReX native plotfile), or both. ascii is the default
        // so the Phase 1 regtest checkers keep working unchanged.
        std::string report_file   = "grid_report.txt";
        std::string plot_file     = "plt_grid";
        std::string output_format = "ascii";

        amrex::ParmParse pp("grid");
        pp.query("report_file", report_file);
        pp.query("plot_file", plot_file);
        pp.query("output_format", output_format);

        const auto fmt = fwt::Grid::ParseOutputFormat(output_format);

        FWT_DEBUG_SECTION("Output settings");
        FWT_DEBUG("output_format    = " << output_format);
        FWT_DEBUG("report_file      = " << report_file
                  << (fwt::Grid::WantsAscii(fmt) ? "" : "   [not written]"));
        FWT_DEBUG("plot_file        = " << plot_file
                  << (fwt::Grid::WantsPlt(fmt) ? "" : "   [not written]"));

        if (fwt::Grid::WantsAscii(fmt)) {
            grid.WriteReport(report_file);
            amrex::Print() << "Wrote grid report to " << report_file << "\n";
        }
        if (fwt::Grid::WantsPlt(fmt)) {
            grid.WritePlotfile(plot_file);
            amrex::Print() << "Wrote plotfile to " << plot_file << "\n";
        }
    }
    amrex::Finalize();
    return 0;
}
