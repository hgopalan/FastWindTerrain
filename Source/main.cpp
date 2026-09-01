#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>

#include "Grid.H"

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        amrex::Print() << "FastWindTerrain -- Phase 1: grid & data layout scaffolding\n";

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
