#ifndef FWT_DEBUG_H_
#define FWT_DEBUG_H_

#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>

namespace fwt {

// -----------------------------------------------------------------------
// Verbose run diagnostics, switched on from the input file:
//
//     fwt.debug = 1
//
// When enabled, the code prints every input it parsed (including the
// defaults that were NOT given in the input file), the arithmetic behind
// the domain-height check, the full z tables, the box decomposition and
// its rank distribution, and every file it writes.
//
// Default is off, and with it off the code prints exactly what it printed
// before this switch existed. Debug lines carry a "[debug]" prefix and
// deliberately never contain the words WARNING or ERROR, so the regtest
// checkers that key on those strings are unaffected by a debug run.
//
// Long tables are elided in the middle past kMaxTableRows entries so that
// a large grid does not bury the rest of the output.
// -----------------------------------------------------------------------
class Debug
{
public:
    static constexpr int kMaxTableRows = 200;

    static void Init ()
    {
        amrex::ParmParse pp("fwt");
        int flag = 0;
        pp.query("debug", flag);
        s_enabled = (flag != 0);
    }

    static bool Enabled () { return s_enabled; }
    static void Set (bool v) { s_enabled = v; }

    // True if row `i` of an `n`-row table should be printed.
    static bool ShowRow (int i, int n)
    {
        if (n <= kMaxTableRows) { return true; }
        return (i < kMaxTableRows/2) || (i >= n - kMaxTableRows/2);
    }

private:
    inline static bool s_enabled = false;
};

} // namespace fwt

// Usage: FWT_DEBUG("dz0 = " << dz0 << " m");
#define FWT_DEBUG(MSG)                                                   \
    do {                                                                 \
        if (::fwt::Debug::Enabled()) {                                   \
            amrex::Print() << "[debug] " << MSG << "\n";                 \
        }                                                                \
    } while (false)

// Section rule, e.g. FWT_DEBUG_SECTION("Grid inputs (grid.*)");
#define FWT_DEBUG_SECTION(TITLE)                                         \
    do {                                                                 \
        if (::fwt::Debug::Enabled()) {                                   \
            amrex::Print() << "[debug] === " << TITLE << " ===\n";       \
        }                                                                \
    } while (false)

#endif
