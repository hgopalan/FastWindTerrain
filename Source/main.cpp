#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Version.H>

#include <fstream>
#include <iomanip>

#include "Grid.H"
#include "Terrain.H"
#include "Inflow.H"
#include "BoundaryConditions.H"
#include "Poisson.H"
#include "Output.H"
#include "Debug.H"
#include "Derivatives.H"

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        amrex::Print() << "FastWindTerrain -- mass-consistent wind solver\n";

        // fwt.debug = 1 turns on verbose diagnostics everywhere.
        fwt::Debug::Init();
        FWT_DEBUG_SECTION("Run configuration");
        FWT_DEBUG("AMReX version    = " << amrex::Version());
        FWT_DEBUG("MPI ranks        = " << amrex::ParallelDescriptor::NProcs());
        FWT_DEBUG("amrex::Real      = " << sizeof(amrex::Real) * 8 << "-bit");
        for (int i = 1; i < argc; ++i) {
            FWT_DEBUG("argv[" << i << "]         = " << argv[i]);
        }

        // Directional-derivative scheme (numerics.gradient_scheme).
        fwt::Numerics::Init();

        std::string selftest_file;
        {
            amrex::ParmParse pp("numerics");
            pp.query("selftest_file", selftest_file);
        }
        if (!selftest_file.empty()) {
            fwt::RunGradientSelfTest(selftest_file);
        }

        fwt::Grid grid;
        grid.Build();   // aborts on undershoot (see Grid::BuildVerticalStretching)

        fwt::Terrain terrain;
        terrain.Build(grid);

        fwt::Inflow inflow;
        inflow.Build(grid, terrain);

        fwt::BoundaryConditions bc;
        bc.Build(grid, terrain, inflow, inflow.velocity());

        fwt::Poisson poisson;
        poisson.Build(grid, terrain, bc);

        int manufactured = 0;
        {
            amrex::ParmParse pp("poisson");
            pp.query("manufactured", manufactured);
        }

        fwt::Poisson::Error mms_err {0.0, 0.0};
        if (manufactured) {
            mms_err = poisson.RunManufactured(grid);
            amrex::Print() << "Manufactured solution: L2 error = "
                           << mms_err.l2 << ", Linf error = "
                           << mms_err.linf << "\n";
        } else {
            poisson.ComputeRHS(grid, terrain, inflow.velocity());
        }

        {
            amrex::ParmParse pp("poisson");
            std::string rhs_dump;
            if (pp.query("rhs_dump_file", rhs_dump) && !rhs_dump.empty()) {
                poisson.WriteRHSDump(rhs_dump);
            }
        }

        amrex::Print() << "Grid built: n_cell = ("
                        << grid.nx() << ", " << grid.ny() << ", " << grid.nz()
                        << "), n_boxes = " << grid.ba().size() << "\n";

        amrex::Print() << "Terrain: z in [" << terrain.z_min() << ", "
                       << terrain.z_max() << "] m, solid cells = "
                       << terrain.n_solid() << " of " << terrain.n_total()
                       << "\n";

        amrex::Print() << "Inflow: mode = " << inflow.mode_name()
                       << ", boundary flux in/out = " << inflow.flux_in()
                       << " / " << inflow.flux_out()
                       << " m^3/s, relative imbalance = "
                       << inflow.flux_imbalance() << "\n";

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
            terrain.AppendReport(report_file);
            inflow.AppendReport(report_file);
            bc.AppendReport(report_file);
            fwt::AppendNumericsReport(report_file);
            poisson.AppendReport(report_file);
            if (manufactured) {
                std::ofstream os(report_file, std::ios::app);
                os << std::setprecision(17);
                os << "poisson_mms_l2 " << mms_err.l2 << "\n";
                os << "poisson_mms_linf " << mms_err.linf << "\n";
            }
            amrex::Print() << "Wrote grid report to " << report_file << "\n";
        }
        if (fwt::Grid::WantsPlt(fmt)) {
            fwt::WritePlotfile(plot_file, grid, terrain, inflow, poisson);
            amrex::Print() << "Wrote plotfile to " << plot_file << "\n";
        }
    }
    amrex::Finalize();
    return 0;
}
