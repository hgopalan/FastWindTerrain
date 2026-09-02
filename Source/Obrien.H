#ifndef FWT_OBRIEN_H_
#define FWT_OBRIEN_H_

#include "Grid.H"
#include "Terrain.H"

#include <AMReX_MultiFab.H>

#include <string>

namespace fwt {

// -----------------------------------------------------------------------
// O'Brien vertical-velocity adjustment.
//
// Integrates the horizontal divergence up each column to get w from
// continuity, then removes the residual at the domain top with a
// quadratic-in-height redistribution, so w = 0 there EXACTLY rather than
// approximately. Ported from massconsistent_amr's
// enable_obrien_w_adjustment, with the stretched dz(k) in the sum in
// place of its uniform dz.
//
// Two passes over each column, starting at its first fluid cell:
//
//   1.  w_top = w(k_start) - sum_{k>k_start} Dh(k) dz(k)
//       which is the residual E left at the top by continuity alone.
//
//   2.  re-integrate, subtracting frac^2 E as it goes, where
//       frac = (k - k_start) / (k_top - k_start).
//
// At the top frac = 1, so the accumulated value and the adjustment are
// both E and w comes out exactly zero. The quadratic weighting puts the
// correction aloft, leaving the near-surface w nearly untouched -- which
// is the point of O'Brien's scheme, since that is where the divergence
// estimate is most trustworthy.
//
// ORDERING. This runs on u0 BEFORE the projection, which is where
// massconsistent_amr applies it. Running it afterwards would rewrite w
// and put back divergence the solve had just removed.
//
// Inputs (obrien.*):
//   enable  -- off by default, as in massconsistent_amr
// -----------------------------------------------------------------------
class Obrien
{
public:

    struct Params
    {
        int enable = 0;
        static Params FromParmParse ();
    };

    // Modifies the velocity in place. Returns the number of columns
    // adjusted.
    long Apply (const Grid& grid, const Terrain& terrain,
                amrex::MultiFab& vel, const Params& params);
    long Apply (const Grid& grid, const Terrain& terrain,
                amrex::MultiFab& vel);

    bool enabled () const { return m_enable != 0; }
    // Largest |w| left at the domain top after the adjustment; it should
    // be zero to round-off, and the regtest holds it to that.
    amrex::Real max_w_top () const { return m_max_w_top; }
    amrex::Real max_residual () const { return m_max_residual; }
    // Columns the adjustment touched. Zero when it is switched off, and
    // also zero if every column turned out to be solid -- which is worth
    // being able to tell apart from "it ran and did nothing".
    long n_columns () const { return m_n_columns; }

    void AppendReport (const std::string& filename) const;

private:

    int m_enable = 0;
    long m_n_columns = 0;
    amrex::Real m_max_w_top = 0.0;
    amrex::Real m_max_residual = 0.0;
};

} // namespace fwt

#endif
