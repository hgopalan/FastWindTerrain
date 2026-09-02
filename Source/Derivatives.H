#ifndef FWT_DERIVATIVES_H_
#define FWT_DERIVATIVES_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_Vector.H>

#include <string>

namespace fwt {

// -----------------------------------------------------------------------
// Derivative schemes for directional (advective) derivatives.
//
//   numerics.gradient_scheme = weno3js | upwind2 | central2
//
// weno3js is the default. All three take the same five-point stencil
// centered on the point, plus the advecting velocity whose sign selects
// the upwind side.
//
// WHAT THESE ARE FOR. WENO3-JS and second-order upwind are upwind-biased
// reconstructions built for hyperbolic transport: the advection term of
// a future fractional-step solver, and scalar transport. They are the
// right tool there, and central differencing is not.
//
// THE PROJECTION IS A SEPARATE QUESTION. A discrete projection is exact
// only when its divergence D and its gradient G satisfy G = -D^T and
// D G is the very operator the linear solve inverted. Central D and
// central G are such a pair, which is why a central projection drives
// div(u) to solver tolerance.
//
// Using the SAME scheme on both sides is not enough to recover that.
// WENO is nonlinear -- its weights depend on the solution -- so
// D_weno G_weno is not a fixed linear operator, and in particular is
// not the Laplacian MLMG inverted. The corrected field then carries
// div(u) = O(h^p) rather than machine zero: an approximate projection,
// which is a legitimate published approach but changes what a
// divergence-free check can assert.
//
// So the scheme selected here governs directional derivatives. Which
// operators the projection itself is built from is decided where the
// projection is assembled, not here.
//
// STRETCHED DIRECTIONS. The schemes are written on a uniformly spaced
// index coordinate, and `h` is the local metric d(coordinate)/d(index).
// For x and y that is simply dx or dy. For the stretched z it is
// dz/dk, so passing (z_cc[k+1] - z_cc[k-1])/2 gives the mapped-coordinate
// form. Formal order is retained for a smoothly stretched grid, which
// the geometric stretching used here is.
//
// STENCIL WIDTH. Both upwind schemes reach two cells to the upwind side,
// so any field they are applied to needs a two-cell halo.
// -----------------------------------------------------------------------

enum class Scheme { central2, upwind2, weno3js };

Scheme ParseScheme (const std::string& s);
const char* SchemeName (Scheme s);

// Reads numerics.gradient_scheme once and caches it. Default weno3js.
class Numerics
{
public:
    static void Init ();
    static Scheme scheme ()          { return s_scheme; }
    static const std::string& name () { return s_name; }
private:
    inline static Scheme s_scheme = Scheme::weno3js;
    inline static std::string s_name = "weno3js";
};

namespace detail {

// WENO3-JS smoothness-weighted reconstruction at the right-hand face of
// the middle cell, from three consecutive values (a, b, c) biased toward
// a. Mirror the argument order for the other bias.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real Weno3Recon (amrex::Real a, amrex::Real b, amrex::Real c)
{
    constexpr amrex::Real eps = amrex::Real(1.0e-6);

    // Candidate polynomials and Jiang-Shu smoothness indicators.
    const amrex::Real p0 = amrex::Real(-0.5) * a + amrex::Real(1.5) * b;
    const amrex::Real p1 = amrex::Real(0.5) * b + amrex::Real(0.5) * c;
    const amrex::Real b0 = (b - a) * (b - a);
    const amrex::Real b1 = (c - b) * (c - b);

    // Linear weights 1/3 and 2/3, scaled by the smoothness indicators so
    // a stencil spanning a discontinuity is suppressed.
    const amrex::Real w0 = (amrex::Real(1.0) / amrex::Real(3.0))
                         / ((eps + b0) * (eps + b0));
    const amrex::Real w1 = (amrex::Real(2.0) / amrex::Real(3.0))
                         / ((eps + b1) * (eps + b1));

    return (w0 * p0 + w1 * p1) / (w0 + w1);
}

} // namespace detail

// Second-order central difference. No upwind bias, so the advecting
// velocity is not consulted.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real Central2 (amrex::Real um1, amrex::Real up1, amrex::Real h)
{
    return (up1 - um1) / (amrex::Real(2.0) * h);
}

// Second-order one-sided difference, leaning into the flow.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real Upwind2 (amrex::Real um2, amrex::Real um1, amrex::Real u0,
                     amrex::Real up1, amrex::Real up2,
                     amrex::Real a, amrex::Real h)
{
    if (a > amrex::Real(0.0)) {
        return (amrex::Real(3.0) * u0 - amrex::Real(4.0) * um1 + um2)
             / (amrex::Real(2.0) * h);
    }
    if (a < amrex::Real(0.0)) {
        return (amrex::Real(-3.0) * u0 + amrex::Real(4.0) * up1 - up2)
             / (amrex::Real(2.0) * h);
    }
    return Central2(um1, up1, h);      // no flow, no upwind side
}

// WENO3-JS: third order where the field is smooth, dropping to second
// order and staying non-oscillatory across a discontinuity. The
// derivative is the difference of the two reconstructed face values.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real Weno3JS (amrex::Real um2, amrex::Real um1, amrex::Real u0,
                     amrex::Real up1, amrex::Real up2,
                     amrex::Real a, amrex::Real h)
{
    if (a > amrex::Real(0.0)) {
        const amrex::Real fp = detail::Weno3Recon(um1, u0, up1);   // i+1/2
        const amrex::Real fm = detail::Weno3Recon(um2, um1, u0);   // i-1/2
        return (fp - fm) / h;
    }
    if (a < amrex::Real(0.0)) {
        // Mirrored stencils for the opposite bias.
        const amrex::Real fp = detail::Weno3Recon(up2, up1, u0);   // i+1/2
        const amrex::Real fm = detail::Weno3Recon(up1, u0, um1);   // i-1/2
        return (fp - fm) / h;
    }
    return Central2(um1, up1, h);
}

// Dispatch on the configured scheme. `h` is the local metric
// d(coordinate)/d(index); `a` is the advecting velocity.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real Derivative (Scheme s,
                        amrex::Real um2, amrex::Real um1, amrex::Real u0,
                        amrex::Real up1, amrex::Real up2,
                        amrex::Real a, amrex::Real h)
{
    switch (s) {
    case Scheme::central2: return Central2(um1, up1, h);
    case Scheme::upwind2:  return Upwind2(um2, um1, u0, up1, up2, a, h);
    case Scheme::weno3js:  return Weno3JS(um2, um1, u0, up1, up2, a, h);
    }
    return Central2(um1, up1, h);
}

// How many cells each scheme reaches to either side. Fields the schemes
// are applied to need at least this many ghost cells.
constexpr int kStencilRadius = 2;

// dz/dk at cell centres: the local metric the schemes take as `h` in the
// stretched direction. One-sided at the two ends, centred inside.
//
// Every vertical derivative in the code goes through this, so the
// divergence diagnostic and the profile-gradient verification cannot end
// up measuring two slightly different vertical coordinates.
amrex::Vector<amrex::Real>
ColumnMetric (const amrex::Vector<amrex::Real>& z_cc);

// Appends the resolved numerics settings to the ascii report.
void AppendNumericsReport (const std::string& filename);

// Runs a grid-refinement study of every scheme and writes the observed
// orders of accuracy. A test aid, written only when
// numerics.selftest_file is set.
void RunGradientSelfTest (const std::string& filename);

} // namespace fwt

#endif
