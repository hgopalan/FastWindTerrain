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

        std::string report_file = "grid_report.txt";
        amrex::ParmParse pp("grid");
        pp.query("report_file", report_file);
        grid.WriteReport(report_file);

        amrex::Print() << "Wrote grid report to " << report_file << "\n";
    }
    amrex::Finalize();
    return 0;
}
