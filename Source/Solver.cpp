#include "Solver.H"
#include "Output.H"
#include "FieldIO.H"
#include "Debug.H"
#include "Derivatives.H"

#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Version.H>

#include <fstream>
#include <iomanip>

namespace fwt {

Solver::Params Solver::Params::FromParmParse ()
{
    Params p;
    {
        amrex::ParmParse pp("numerics");
        pp.query("gradient_scheme", p.gradient_scheme);
    }
    p.grid       = Grid::Params::FromParmParse();
    p.terrain    = Terrain::Params::FromParmParse();
    p.inflow     = Inflow::Params::FromParmParse();
    p.anisotropy = Anisotropy::Params::FromParmParse();
    p.obrien     = Obrien::Params::FromParmParse();
    p.poisson    = Poisson::Params::FromParmParse();
    p.output     = OutputParams::FromParmParse();
    return p;
}

void Solver::Setup (const amrex::Vector<std::string>& args)
{
    Setup(Params::FromParmParse(), args);
}

void Solver::Setup (const Params& params,
                    const amrex::Vector<std::string>& args)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!m_setup_done,
        "Solver::Setup called twice on the same solver");

    m_params = params;

    amrex::Print() << "FastWindTerrain -- mass-consistent wind solver\n";

    // fwt.debug = 1 turns on verbose diagnostics everywhere.
    fwt::Debug::Init();
    FWT_DEBUG_SECTION("Run configuration");
    FWT_DEBUG("AMReX version    = " << amrex::Version());
    FWT_DEBUG("MPI ranks        = " << amrex::ParallelDescriptor::NProcs());
    FWT_DEBUG("amrex::Real      = " << sizeof(amrex::Real) * 8 << "-bit");
    for (int i = 0; i < int(args.size()); ++i) {
        FWT_DEBUG("argv[" << (i + 1) << "]         = " << args[i]);
    }

    // Directional-derivative scheme. An empty name leaves whatever is
    // already in force, which for the ParmParse path is what
    // FromParmParse just read.
    if (m_params.gradient_scheme.empty()) {
        fwt::Numerics::Init();
    } else {
        fwt::Numerics::Set(m_params.gradient_scheme);
    }

    std::string selftest_file;
    {
        amrex::ParmParse pp("numerics");
        pp.query("selftest_file", selftest_file);
    }
    if (!selftest_file.empty()) {
        fwt::RunGradientSelfTest(selftest_file);
    }

    m_grid.Build(m_params.grid);   // throws on undershoot
    m_terrain.Build(m_grid, m_params.terrain);
    m_inflow.Build(m_grid, m_terrain, m_params.inflow);
    m_bc.Build(m_grid, m_terrain, m_inflow, m_inflow.velocity());

    // Verification dump, if asked for. It runs HERE, on the raw profile,
    // before O'Brien and the projection rewrite the field: the quantity
    // under study is the gradient of the inflow profile, not of whatever
    // the adjustment left behind.
    m_verify.MaybeWriteGradientDump(m_grid, m_terrain, m_inflow.velocity());

    // Cell-local variational weights, from the terrain slope.
    // The base weights live with the operator they feed, so the
    // anisotropy takes them from the Poisson parameters rather than
    // keeping a second copy that could disagree.
    Anisotropy::Params aniso_params = m_params.anisotropy;
    aniso_params.alpha_h_base = m_params.poisson.alpha_h;
    aniso_params.alpha_v_base = m_params.poisson.alpha_v;
    m_aniso.Build(m_grid, m_terrain, aniso_params);

    // O'Brien runs on u0, BEFORE the projection: it rewrites w from
    // continuity, and doing that afterwards would put back the divergence
    // the solve had just removed. massconsistent_amr applies it at the
    // same point.
    if (m_obrien.Apply(m_grid, m_terrain, m_inflow.velocity(),
                       m_params.obrien) > 0) {
        m_bc.RefillGhosts(m_grid, m_terrain, m_inflow, m_inflow.velocity());
    }

    m_poisson.Build(m_grid, m_terrain, m_bc, m_aniso, m_params.poisson);

    // Keep the initial field: the projection corrects in place, and both
    // are worth having in the output -- the checkers compare against u0,
    // and a user wants to see what the adjustment did.
    m_vel0.define(m_grid.ba(), m_grid.dm(), 3, m_inflow.velocity().nGrow());
    amrex::MultiFab::Copy(m_vel0, m_inflow.velocity(), 0, 0, 3,
                          m_inflow.velocity().nGrow());

    m_manufactured = m_params.poisson.manufactured;

    m_setup_done = true;
}

void Solver::Solve ()
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_setup_done,
        "Solver::Solve called before Setup");

    if (m_manufactured) {
        m_mms_err = m_poisson.RunManufactured(m_grid);
        amrex::Print() << "Manufactured solution: L2 error = "
                       << m_mms_err.l2 << ", Linf error = "
                       << m_mms_err.linf << "\n";
        m_solved = true;
        return;
    }

    m_poisson.ComputeRHS(m_grid, m_terrain, m_inflow.velocity());

    const amrex::Real div0 =
        m_poisson.MaxDivergence(m_grid, m_terrain, m_inflow.velocity());
    m_poisson.set_div_before(div0);
    m_poisson.set_vel_before(
        fwt::Poisson::VelocityRange(m_terrain, m_inflow.velocity()));

    const amrex::Real fe0 =
        m_poisson.MaxDivergenceFE(m_grid, m_terrain, m_inflow.velocity());

    // AMReX's nodal projection is approximate: its divergence and
    // gradient are not an exact factorisation of the operator, so one
    // pass removes only part of the divergence. Repeating the projection
    // drives the remainder down geometrically.
    const int n_proj = m_params.poisson.n_projections;

    for (int ip = 0; ip < n_proj; ++ip) {
        // The first pass reuses the RHS Solve() computed above; the rest
        // rebuild it from the corrected field, which ProjectOnce does.
        if (ip == 0) {
            m_poisson.Solve();
            m_poisson.ApplyCorrection(m_grid, m_terrain,
                                      m_inflow.velocity());
            m_bc.RefillGhosts(m_grid, m_terrain, m_inflow,
                              m_inflow.velocity());
            ++m_n_proj_done;
        } else {
            ProjectOnce();
        }

        if (n_proj > 1) {
            amrex::Print() << "  projection pass " << (ip + 1)
                << ": max|div| (controlled norm) = "
                << m_poisson.MaxDivergenceFE(m_grid, m_terrain,
                                             m_inflow.velocity())
                << "\n";
        }
    }

    const amrex::Real div1 =
        m_poisson.MaxDivergence(m_grid, m_terrain, m_inflow.velocity());
    m_poisson.set_div_after(div1);

    amrex::Print() << "Projection: max|div(u)| " << div0 << " -> "
                   << div1 << "  (factor "
                   << (div1 > 0.0 ? div0 / div1 : 0.0) << ")\n";
    const amrex::Real fe1 =
        m_poisson.MaxDivergenceFE(m_grid, m_terrain, m_inflow.velocity());
    m_poisson.set_div_fe(fe0, fe1);
    m_poisson.set_n_projections(n_proj);
    amrex::Print() << "  in the norm the solve controls: "
                   << fe0 << " -> " << fe1 << "\n";

    // A corrected wind far larger than the one that went in means the
    // setup is wrong, whatever the residual says.
    const auto vr =
        fwt::Poisson::VelocityRange(m_terrain, m_inflow.velocity());
    m_poisson.set_vel_after(vr);
    amrex::Print() << "  velocity u [" << vr.lo[0] << ", " << vr.hi[0]
                   << "]  v [" << vr.lo[1] << ", " << vr.hi[1]
                   << "]  w [" << vr.lo[2] << ", " << vr.hi[2]
                   << "]  |U|max " << vr.speed_max << " m/s\n";

    m_solved = true;
}

amrex::Real Solver::MaxDivergenceFE ()
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_setup_done,
        "Solver::MaxDivergenceFE called before Setup");
    return m_poisson.MaxDivergenceFE(m_grid, m_terrain, m_inflow.velocity());
}

amrex::Real Solver::MaxDivergence () const
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_setup_done,
        "Solver::MaxDivergence called before Setup");
    return m_poisson.MaxDivergence(m_grid, m_terrain, m_inflow.velocity());
}

amrex::Real Solver::ProjectOnce ()
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_setup_done,
        "Solver::ProjectOnce called before Setup");

    m_poisson.ComputeRHS(m_grid, m_terrain, m_inflow.velocity());
    const amrex::Real resid = m_poisson.Solve();
    m_poisson.ApplyCorrection(m_grid, m_terrain, m_inflow.velocity());

    // The correction changed the interior, so the ghosts are refreshed
    // before anything reads a stencil near a face. The classification is
    // NOT redone: which face is an inflow face follows from the incoming
    // wind, not from the corrected field.
    m_bc.RefillGhosts(m_grid, m_terrain, m_inflow, m_inflow.velocity());

    ++m_n_proj_done;
    m_solved = true;
    return resid;
}

void Solver::Diagnose ()
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_solved,
        "Solver::Diagnose called before Solve");

    // Post-solve diagnostics. The divergence field is computed once and
    // then used everywhere: the scalar in the report, the norms, and the
    // component in the output file are all reductions or copies of this
    // one array, so they cannot disagree.
    m_divergence.define(m_grid.ba(), m_grid.dm(), 1, 0);
    m_poisson.ComputeDivergenceField(m_grid, m_terrain, m_inflow.velocity(),
                                     m_divergence);

    m_diag.Compute(m_grid, m_terrain, m_inflow.velocity(), m_divergence);
    m_diag.Print();

    if (!m_params.poisson.rhs_dump_file.empty()) {
        m_poisson.WriteRHSDump(m_params.poisson.rhs_dump_file);
    }

    amrex::Print() << "Grid built: n_cell = ("
                    << m_grid.nx() << ", " << m_grid.ny() << ", "
                    << m_grid.nz()
                    << "), n_boxes = " << m_grid.ba().size() << "\n";

    amrex::Print() << "Terrain: z in [" << m_terrain.z_min() << ", "
                   << m_terrain.z_max() << "] m, solid cells = "
                   << m_terrain.n_solid() << " of " << m_terrain.n_total()
                   << "\n";

    amrex::Print() << "Inflow: mode = " << m_inflow.mode_name()
                   << ", boundary flux in/out = " << m_inflow.flux_in()
                   << " / " << m_inflow.flux_out()
                   << " m^3/s, relative imbalance = "
                   << m_inflow.flux_imbalance() << "\n";

    m_diagnosed = true;
}

OutputFields Solver::CollectOutput () const
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_diagnosed,
        "Solver::CollectOutput called before Diagnose");
    return CollectOutputFields(m_grid, m_terrain, m_inflow, m_poisson,
                               m_vel0, m_aniso, m_divergence);
}

void Solver::WritePlotfile (const std::string& path) const
{
    fwt::WritePlotfile(path, m_grid, CollectOutput());
    amrex::Print() << "Wrote plotfile to " << path << "\n";
}

void Solver::WriteAscii (const std::string& path) const
{
    fwt::WriteAscii(path, m_grid, CollectOutput());
    amrex::Print() << "Wrote ascii field output to " << path << "\n";
}

void Solver::WriteReport (const std::string& path) const
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_diagnosed,
        "Solver::WriteReport called before Diagnose");

    m_grid.WriteReport(path);
    m_terrain.AppendReport(path);
    m_inflow.AppendReport(path);
    m_bc.AppendReport(path);
    fwt::AppendNumericsReport(path);
    m_aniso.AppendReport(path);
    m_obrien.AppendReport(path);
    m_poisson.AppendReport(path);
    m_diag.AppendReport(path);
    if (m_manufactured) {
        std::ofstream os(path, std::ios::app);
        os << std::setprecision(17);
        os << "poisson_mms_l2 " << m_mms_err.l2 << "\n";
        os << "poisson_mms_linf " << m_mms_err.linf << "\n";
    }
    amrex::Print() << "Wrote grid report to " << path << "\n";
}

void Solver::WriteOutput () const
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_diagnosed,
        "Solver::WriteOutput called before Diagnose");

    // Two independent switches, both now held as data:
    //   which  -- WHICH outputs the run produces: the plain-text report,
    //             the field output, or both
    //   format -- which backend writes the field output: plt (default,
    //             production) or ascii (one gathered plain-text file, a
    //             regtest aid)
    const OutputParams& o = m_params.output;
    const auto fmt  = fwt::Grid::ParseOutputFormat(o.which);
    const auto ffmt = fwt::ParseFieldFormat(o.format);

    FWT_DEBUG_SECTION("Output settings");
    FWT_DEBUG("grid.output_format = " << o.which);
    FWT_DEBUG("output.format      = " << o.format
              << (fwt::Grid::WantsPlt(fmt) ? "" : "   [no field output]"));
    FWT_DEBUG("report_file      = " << o.report_file
              << (fwt::Grid::WantsAscii(fmt) ? "" : "   [not written]"));
    FWT_DEBUG("plot_file        = " << o.plot_file
              << ((fwt::Grid::WantsPlt(fmt) && fwt::WantsFieldPlt(ffmt))
                  ? "" : "   [not written]"));
    FWT_DEBUG("ascii_file       = " << o.ascii_file
              << ((fwt::Grid::WantsPlt(fmt) && fwt::WantsFieldAscii(ffmt))
                  ? "" : "   [not written]"));

    if (fwt::Grid::WantsAscii(fmt)) {
        WriteReport(o.report_file);
    }
    if (fwt::Grid::WantsPlt(fmt)) {
        // One gather, both backends -- and the same one CollectOutput
        // hands to Python. Neither assembles its own idea of what the
        // fields are, so they cannot drift apart.
        const OutputFields fields = CollectOutput();
        if (fwt::WantsFieldPlt(ffmt)) {
            fwt::WritePlotfile(o.plot_file, m_grid, fields);
            amrex::Print() << "Wrote plotfile to " << o.plot_file << "\n";
        }
        if (fwt::WantsFieldAscii(ffmt)) {
            fwt::WriteAscii(o.ascii_file, m_grid, fields);
            amrex::Print() << "Wrote ascii field output to "
                           << o.ascii_file << "\n";
        }
    }
}

void Solver::SetVelocity (const amrex::Vector<amrex::Real>& buffer)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_setup_done,
        "Solver::SetVelocity called before Setup");

    ScatterField(buffer, m_inflow.velocity());
    m_bc.RefillGhosts(m_grid, m_terrain, m_inflow, m_inflow.velocity());
}

void Solver::Run (const amrex::Vector<std::string>& args)
{
    Run(Params::FromParmParse(), args);
}

void Solver::Run (const Params& params,
                  const amrex::Vector<std::string>& args)
{
    Setup(params, args);
    Solve();
    Diagnose();
    WriteOutput();
}

} // namespace fwt
