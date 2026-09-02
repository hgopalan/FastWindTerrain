#include "Obrien.H"
#include "Debug.H"
#include "Derivatives.H"

#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>

namespace fwt {

Obrien::Params Obrien::Params::FromParmParse ()
{
    Params p;
    amrex::ParmParse pp("obrien");
    pp.query("enable", p.enable);
    return p;
}

long Obrien::Apply (const Grid& grid, const Terrain& terrain,
                    amrex::MultiFab& vel)
{
    return Apply(grid, terrain, vel, Params::FromParmParse());
}

long Obrien::Apply (const Grid& grid, const Terrain& terrain,
                    amrex::MultiFab& vel, const Params& params)
{
    m_enable = params.enable;

    FWT_DEBUG_SECTION("O'Brien adjustment (obrien.*)");
    FWT_DEBUG("enable           = " << m_enable
              << (m_enable ? "" : "   [w left as the profile set it]"));

    if (m_enable == 0) { return 0; }

    const amrex::Real dx = grid.geom().CellSize(0);
    const amrex::Real dy = grid.geom().CellSize(1);
    const amrex::Vector<amrex::Real>& zf = grid.z_face();
    const amrex::Vector<amrex::Real>& z_cc = grid.z_cc();
    const int nz = grid.nz();

    // True cell heights for the vertical sum, and the spacing the
    // derivative stencil sees.
    amrex::Vector<amrex::Real> dz(nz), dzdk(nz);
    for (int k = 0; k < nz; ++k) {
        dz[k] = zf[k+1] - zf[k];
        if (k == 0)           { dzdk[k] = z_cc[1] - z_cc[0]; }
        else if (k == nz - 1) { dzdk[k] = z_cc[nz-1] - z_cc[nz-2]; }
        else                  { dzdk[k] = 0.5 * (z_cc[k+1] - z_cc[k-1]); }
    }

    const Scheme scheme = Numerics::scheme();
    const int solid = Terrain::kSolid;
    const amrex::Box& dom = grid.geom().Domain();
    const int klo = dom.smallEnd(2), khi = dom.bigEnd(2);
    const int ilo = dom.smallEnd(0), ihi = dom.bigEnd(0);
    const int jlo = dom.smallEnd(1), jhi = dom.bigEnd(1);

    // The grid is split in x and y only, so every box already spans the
    // full height and a column integration can be done in place. Grid
    // asserts that invariant when it builds the BoxArray.
    long n_columns = 0;
    amrex::Real max_w_top = 0.0;
    amrex::Real max_residual = 0.0;

    // Column-wise, so this runs on the host: each column is a sequential
    // integration and the work is a vanishing fraction of a solve.
    for (amrex::MFIter mfi(vel); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const& v  = vel.array(mfi);
        auto const& mk = terrain.mask().const_array(mfi);

        // Horizontal divergence at one cell, with the configured scheme.
        // No advecting velocity for this purpose, so the schemes fall
        // back to their central form. Indices are clamped to the domain
        // so the stencil never reaches past a physical boundary, which is
        // how massconsistent_amr handles its edges too.
        auto Dh = [&] (int i, int j, int k) -> amrex::Real
        {
            const int im2 = amrex::max(i-2, ilo), im1 = amrex::max(i-1, ilo);
            const int ip1 = amrex::min(i+1, ihi), ip2 = amrex::min(i+2, ihi);
            const int jm2 = amrex::max(j-2, jlo), jm1 = amrex::max(j-1, jlo);
            const int jp1 = amrex::min(j+1, jhi), jp2 = amrex::min(j+2, jhi);

            const amrex::Real dudx = Derivative(scheme,
                v(im2,j,k,0), v(im1,j,k,0), v(i,j,k,0),
                v(ip1,j,k,0), v(ip2,j,k,0), 0.0, dx);
            const amrex::Real dvdy = Derivative(scheme,
                v(i,jm2,k,1), v(i,jm1,k,1), v(i,j,k,1),
                v(i,jp1,k,1), v(i,jp2,k,1), 0.0, dy);
            return dudx + dvdy;
        };

        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
        for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {

            // The first fluid cell in this column.
            int k_start = klo;
            while (k_start <= khi && mk(i,j,k_start) == solid) { ++k_start; }
            if (k_start >= khi) { continue; }   // buried, or no room above

            // Pass 1: integrate continuity to the top and keep what is
            // left over there.
            amrex::Real w = v(i,j,k_start,2);
            for (int k = k_start + 1; k <= khi; ++k) {
                w -= Dh(i,j,k) * dz[k];
            }
            const amrex::Real E = w;
            max_residual = std::max(max_residual, std::abs(E));

            // Pass 2: integrate again, removing the residual with a
            // quadratic weight so it vanishes exactly at the top and
            // barely touches the near-surface values.
            const amrex::Real span = amrex::Real(khi - k_start);
            amrex::Real w_current = v(i,j,k_start,2);
            for (int k = k_start + 1; k <= khi; ++k) {
                w_current -= Dh(i,j,k) * dz[k];
                const amrex::Real frac = amrex::Real(k - k_start) / span;
                v(i,j,k,2) = w_current - frac * frac * E;
            }

            max_w_top = std::max(max_w_top, std::abs(v(i,j,khi,2)));
            ++n_columns;
        }}
    }

    amrex::ParallelDescriptor::ReduceLongSum(n_columns);
    amrex::ParallelDescriptor::ReduceRealMax(max_w_top);
    amrex::ParallelDescriptor::ReduceRealMax(max_residual);

    m_n_columns = n_columns;
    m_max_w_top = max_w_top;
    m_max_residual = max_residual;

    amrex::Print() << "O'Brien: adjusted " << n_columns
                   << " columns, residual max " << max_residual
                   << " m/s, |w| at the top now " << max_w_top
                   << " m/s\n";

    FWT_DEBUG_SECTION("O'Brien adjustment");
    FWT_DEBUG("columns adjusted = " << n_columns);
    FWT_DEBUG("max residual E   = " << max_residual << " m/s"
              << "   (what continuity alone left at the top)");
    FWT_DEBUG("max |w| at top   = " << max_w_top << " m/s"
              << "   (should be zero to round-off)");

    return n_columns;
}

void Obrien::AppendReport (const std::string& filename) const
{
    if (!amrex::ParallelDescriptor::IOProcessor()) { return; }

    std::ofstream os(filename, std::ios::app);
    os << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10);
    os << "# obrien (Phase 7)\n";
    os << "obrien_enable " << m_enable << "\n";
    os << "obrien_n_columns " << m_n_columns << "\n";
    os << "obrien_max_residual " << m_max_residual << "\n";
    os << "obrien_max_w_top " << m_max_w_top << "\n";
    os.close();
}

} // namespace fwt
