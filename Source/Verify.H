#ifndef FWT_VERIFY_H_
#define FWT_VERIFY_H_

#include "Grid.H"
#include "Terrain.H"

#include <AMReX_MultiFab.H>

#include <string>

namespace fwt {

// -----------------------------------------------------------------------
// Verify: a test aid that exposes a derivative the solver actually
// computes, so its ORDER can be measured end to end.
//
// WHY THIS EXISTS. numerics.selftest_file already measures the order of
// every scheme -- but on a std::vector holding sin(2 pi x). It never
// builds a Grid, a MultiFab or a ghost cell, so it verifies the stencil
// and nothing between the stencil and the number the solver uses: the
// column metric, the box decomposition, the index clamping at the domain
// ends. Any of those can cost an order without the self-test noticing.
//
// WHY THE VERTICAL DERIVATIVE. On flat ground a powerlaw or loglaw
// profile has div(u) = 0 IDENTICALLY -- u and v depend only on z and
// w is zero, so all three terms vanish for every scheme on every grid.
// There is nothing to converge on. The only nontrivial derivative in
// that problem is dU/dz, which is also where the schemes differ and
// where a wrong vertical metric shows up.
//
// WHAT IT WRITES. One row per k:
//
//     k  z_cc  dz  dzdk  dudz_mid  dudz_spread  dvdz_mid  dvdz_spread
//
// The field is horizontally uniform over flat ground, so `mid` is the
// midpoint of the level's range and `spread` is its width: on a correct
// run the spread is zero, and a bug that breaks horizontal uniformity
// cannot hide behind a single sampled column. Taking min and max also
// makes the dump rank-independent, so it is not restricted to a single
// rank the way the other dumps are.
//
// The analytic comparison is deliberately NOT done here. The profile law
// is already implemented independently in Python by the checkers, and
// keeping the comparison there keeps this from grading its own work.
// -----------------------------------------------------------------------
class Verify
{
public:

    // Writes the dump if verify.gradient_dump_file is set. Returns true
    // if it wrote anything.
    bool MaybeWriteGradientDump (const Grid& grid, const Terrain& terrain,
                                 const amrex::MultiFab& vel) const;

private:

    // The advecting velocity handed to the upwind branch selection.
    // Fixed rather than taken from w, which is identically zero on flat
    // ground: a study of an upwind scheme has to say which branch it is
    // measuring. Swept over both signs by the convergence driver.
    static amrex::Real AdvectSign ();

    static std::string DumpFile ();
};

} // namespace fwt

#endif
