// The standalone driver.
//
// Everything this program does lives in fwt::Solver, so the executable
// and the Python bindings run the same code rather than two orderings
// that have to be kept in step. main() owns only what is genuinely the
// process's business: AMReX's global lifecycle.
//
// That makes every regtest an exercise of the class the bindings expose.

#include <AMReX.H>
#include <AMReX_Vector.H>

#include <string>

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

        fwt::Solver solver;
        solver.Run(args);
    }
    amrex::Finalize();
    return 0;
}
