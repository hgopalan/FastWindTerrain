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
#include "Anisotropy.H"
#include "Obrien.H"
#include "Poisson.H"
#include "Output.H"
#include "Diagnostics.H"
#include "Verify.H"
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

        // Verification dump, if asked for. It runs HERE, on the raw
        // profile, before O'Brien and the projection rewrite the field:
        // the quantity under study is the gradient of the inflow profile,
        // not of whatever the adjustment left behind.
        fwt::Verify verify;
        verify.MaybeWriteGradientDump(grid, terrain, inflow.velocity());

        // Cell-local variational weights, from the terrain slope.
        fwt::Anisotropy aniso;
        aniso.Build(grid, terrain);

        // O'Brien runs on u0, BEFORE the projection: it rewrites w from
        // continuity, and doing that afterwards would put back the
        // divergence the solve had just removed. massconsistent_amr
        // applies it at the same point.
        fwt::Obrien obrien;
        if (obrien.Apply(grid, terrain, inflow.velocity()) > 0) {
            bc.RefillGhosts(grid, terrain, inflow, inflow.velocity());
        }

        fwt::Poisson poisson;
        poisson.Build(grid, terrain, bc, aniso);

        // Keep the initial field: the projection corrects in place, and
        // both are worth having in the output -- the checkers compare
        // against u0, and a user wants to see what the adjustment did.
        amrex::MultiFab vel0(grid.ba(), grid.dm(), 3,
                             inflow.velocity().nGrow());
        amrex::MultiFab::Copy(vel0, inflow.velocity(), 0, 0, 3,
                              inflow.velocity().nGrow());

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

            const amrex::Real div0 =
                poisson.MaxDivergence(grid, terrain, inflow.velocity());
            poisson.set_div_before(div0);
            poisson.set_vel_before(
                fwt::Poisson::VelocityRange(terrain, inflow.velocity()));

            const amrex::Real fe0 =
                poisson.MaxDivergenceFE(grid, terrain, inflow.velocity());

            // AMReX's nodal projection is approximate: its divergence and
            // gradient are not an exact factorisation of the operator, so
            // one pass removes only part of the divergence. Repeating the
            // projection drives the remainder down geometrically.
            int n_proj = 4;
            {
                amrex::ParmParse pp("poisson");
                pp.query("n_projections", n_proj);
            }
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(n_proj >= 1,
                "poisson.n_projections must be >= 1");

            for (int ip = 0; ip < n_proj; ++ip) {
                if (ip > 0) {
                    poisson.ComputeRHS(grid, terrain, inflow.velocity());
                }
                poisson.Solve();
                poisson.ApplyCorrection(grid, terrain, inflow.velocity());

                // The correction changed the interior, so the ghosts are
                // refreshed before anything reads a stencil near a face.
                // The classification is NOT redone: which face is an
                // inflow face follows from the incoming wind, not from
                // the corrected field.
                bc.RefillGhosts(grid, terrain, inflow, inflow.velocity());

                if (n_proj > 1) {
                    amrex::Print() << "  projection pass " << (ip + 1)
                        << ": max|div| (controlled norm) = "
                        << poisson.MaxDivergenceFE(grid, terrain,
                                                   inflow.velocity())
                        << "\n";
                }
            }

            const amrex::Real div1 =
                poisson.MaxDivergence(grid, terrain, inflow.velocity());
            poisson.set_div_after(div1);

            amrex::Print() << "Projection: max|div(u)| " << div0 << " -> "
                           << div1 << "  (factor "
                           << (div1 > 0.0 ? div0 / div1 : 0.0) << ")\n";
            const amrex::Real fe1 =
                poisson.MaxDivergenceFE(grid, terrain, inflow.velocity());
            poisson.set_div_fe(fe0, fe1);
            poisson.set_n_projections(n_proj);
            amrex::Print() << "  in the norm the solve controls: "
                           << fe0 << " -> " << fe1 << "\n";

            // A corrected wind far larger than the one that went in means
            // the setup is wrong, whatever the residual says.
            const auto vr =
                fwt::Poisson::VelocityRange(terrain, inflow.velocity());
            poisson.set_vel_after(vr);
            amrex::Print() << "  velocity u [" << vr.lo[0] << ", " << vr.hi[0]
                           << "]  v [" << vr.lo[1] << ", " << vr.hi[1]
                           << "]  w [" << vr.lo[2] << ", " << vr.hi[2]
                           << "]  |U|max " << vr.speed_max << " m/s\n";
        }

        // Post-solve diagnostics. The divergence field is computed once
        // and then used everywhere: the scalar in the report, the norms,
        // and the component in the output file are all reductions or
        // copies of this one array, so they cannot disagree.
        amrex::MultiFab divergence(grid.ba(), grid.dm(), 1, 0);
        poisson.ComputeDivergenceField(grid, terrain, inflow.velocity(),
                                       divergence);

        fwt::Diagnostics diag;
        diag.Compute(grid, terrain, inflow.velocity(), divergence);
        diag.Print();

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

        // Two independent switches:
        //   grid.output_format -- WHICH outputs the run produces: the
        //                         plain-text report, the field output,
        //                         or both. report is the default, so the
        //                         Phase 1 checkers keep working.
        //   output.format      -- which backend writes the field output:
        //                         plt (default, production) or ascii
        //                         (one gathered plain-text file, a
        //                         regtest aid).
        std::string report_file   = "grid_report.txt";
        std::string plot_file     = "plt_grid";
        std::string output_format = "report";
        std::string ascii_file    = "fields.txt";
        std::string field_format  = "plt";

        amrex::ParmParse pp("grid");
        pp.query("report_file", report_file);
        pp.query("plot_file", plot_file);
        pp.query("output_format", output_format);
        {
            amrex::ParmParse ppo("output");
            ppo.query("format", field_format);
            ppo.query("ascii_file", ascii_file);
        }

        const auto fmt  = fwt::Grid::ParseOutputFormat(output_format);
        const auto ffmt = fwt::ParseFieldFormat(field_format);

        FWT_DEBUG_SECTION("Output settings");
        FWT_DEBUG("grid.output_format = " << output_format);
        FWT_DEBUG("output.format      = " << field_format
                  << (fwt::Grid::WantsPlt(fmt) ? "" : "   [no field output]"));
        FWT_DEBUG("report_file      = " << report_file
                  << (fwt::Grid::WantsAscii(fmt) ? "" : "   [not written]"));
        FWT_DEBUG("plot_file        = " << plot_file
                  << ((fwt::Grid::WantsPlt(fmt) && fwt::WantsFieldPlt(ffmt))
                      ? "" : "   [not written]"));
        FWT_DEBUG("ascii_file       = " << ascii_file
                  << ((fwt::Grid::WantsPlt(fmt) && fwt::WantsFieldAscii(ffmt))
                      ? "" : "   [not written]"));

        if (fwt::Grid::WantsAscii(fmt)) {
            grid.WriteReport(report_file);
            terrain.AppendReport(report_file);
            inflow.AppendReport(report_file);
            bc.AppendReport(report_file);
            fwt::AppendNumericsReport(report_file);
            aniso.AppendReport(report_file);
            obrien.AppendReport(report_file);
            poisson.AppendReport(report_file);
            diag.AppendReport(report_file);
            if (manufactured) {
                std::ofstream os(report_file, std::ios::app);
                os << std::setprecision(17);
                os << "poisson_mms_l2 " << mms_err.l2 << "\n";
                os << "poisson_mms_linf " << mms_err.linf << "\n";
            }
            amrex::Print() << "Wrote grid report to " << report_file << "\n";
        }
        if (fwt::Grid::WantsPlt(fmt)) {
            // One gather, both backends. Neither assembles its own idea
            // of what the fields are, so they cannot drift apart.
            const fwt::OutputFields fields =
                fwt::CollectOutputFields(grid, terrain, inflow, poisson,
                                         vel0, aniso, divergence);
            if (fwt::WantsFieldPlt(ffmt)) {
                fwt::WritePlotfile(plot_file, grid, fields);
                amrex::Print() << "Wrote plotfile to " << plot_file << "\n";
            }
            if (fwt::WantsFieldAscii(ffmt)) {
                fwt::WriteAscii(ascii_file, grid, fields);
                amrex::Print() << "Wrote ascii field output to "
                               << ascii_file << "\n";
            }
        }
    }
    amrex::Finalize();
    return 0;
}
