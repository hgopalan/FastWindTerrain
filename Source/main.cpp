// The standalone driver.
//
// Everything this program does lives in fwt::Solver, so the executable
// and the Python bindings run the same code rather than two orderings
// that have to be kept in step. main() owns only what is genuinely the
// process's business: AMReX's global lifecycle, and turning a bad input
// into the abort a command-line tool should give.

#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_Vector.H>

#include <string>

#include "Error.H"
#include "Solver.H"

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        // The arguments after the program name, purely so fwt.debug can
        // echo them. AMReX has already parsed them into ParmParse.
        amrex::Vector<std::string> args;
        args.reserve(argc > 1 ? argc - 1 : 0);
        for (int i = 1; i < argc; ++i) { args.emplace_back(argv[i]); }

        try {
            fwt::Solver solver;
            solver.Run(args);
        }
        catch (const fwt::InputError& e) {
            // A bad input aborts here, which is what a command-line tool
            // should do: print the diagnostic, exit nonzero, write no
            // output. The solver throws instead of aborting so that the
            // Python bindings can raise, and this is where that becomes
            // an abort again -- the printed text and the nonzero exit
            // are exactly what they were before the change.
            amrex::Print() << e.what();
            amrex::Abort("FastWindTerrain: invalid input "
                         "(see message above).");
        }
    }
    amrex::Finalize();
    return 0;
}
