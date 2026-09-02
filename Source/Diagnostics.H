#ifndef FWT_DIAGNOSTICS_H_
#define FWT_DIAGNOSTICS_H_

#include "Grid.H"
#include "Terrain.H"

#include <AMReX_MultiFab.H>

#include <string>

namespace fwt {

// -----------------------------------------------------------------------
// Diagnostics: what the run is asked to prove about itself once the
// projection is done.
//
//   1. the discrete divergence as a FIELD, one value per cell, so a
//      failure can be located rather than only detected. The scalar
//      max|div| the projection already reports says whether something is
//      wrong; this says where.
//
//   2. the mass-flux balance: volumetric flux through each open face of
//      the domain, in and out, and the net.
//
//   3. the boundary flux check: the net as a fraction of the throughput,
//      against diagnostics.flux_tolerance. Exceeding it WARNS -- it is
//      never corrected and never fatal. A mass-consistent adjustment
//      with Dirichlet lambda on the laterals does not enforce a closed
//      budget, and quietly forcing one would hide the very thing this
//      number is here to show.
//
// Everything here is measured, not enforced.
// -----------------------------------------------------------------------

// Volumetric flux through the open boundaries of the domain: the four
// laterals and the top, with the stretched dz(k) as the lateral cell
// height. The ground is closed and solid cells carry no flux.
//
// This is the one implementation. Inflow reports the same quantity for
// the initial field by calling it, so the before/after pair cannot come
// from two subtly different definitions.
struct FluxBalance
{
    amrex::Real in  = 0.0;         // total inward flux  [m^3/s], positive
    amrex::Real out = 0.0;         // total outward flux [m^3/s], positive
    amrex::Real net = 0.0;         // out - in
    amrex::Real imbalance = 0.0;   // |net| / max(in, out)

    // Per face, in the order xlo, xhi, ylo, yhi, top. Signed: positive
    // is out of the domain.
    static constexpr int kNFaces = 5;
    amrex::Real face_net[kNFaces] = {0.0, 0.0, 0.0, 0.0, 0.0};
    static const char* face_name (int f);
};

FluxBalance ComputeFluxBalance (const Grid& grid, const Terrain& terrain,
                                const amrex::MultiFab& vel);

class Diagnostics
{
public:

    // div must already hold the per-cell divergence (Poisson owns that
    // discretization; computing a second one here is exactly the drift
    // this phase is trying to remove). Solid cells are expected to be
    // zero there and are skipped in the extrema.
    void Compute (const Grid& grid, const Terrain& terrain,
                  const amrex::MultiFab& vel, const amrex::MultiFab& div);

    void Print () const;
    void AppendReport (const std::string& filename) const;

    const FluxBalance& flux () const { return m_flux; }
    amrex::Real div_max () const { return m_div_max; }
    amrex::Real div_l2 ()  const { return m_div_l2; }
    bool flux_within_tolerance () const { return m_flux_ok; }

private:

    void ReadParameters ();

    FluxBalance m_flux;
    amrex::Real m_div_max = 0.0;    // max |div u| over fluid cells
    amrex::Real m_div_l2  = 0.0;    // volume-weighted L2 of div u
    amrex::Real m_div_min = 0.0;
    long        m_n_fluid = 0;
    amrex::Real m_flux_tol = 1.0e-3;
    bool        m_flux_ok = true;
};

} // namespace fwt

#endif
