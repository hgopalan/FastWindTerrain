#include "Derivatives.H"
#include "Debug.H"

#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <vector>

namespace fwt {

Scheme ParseScheme (const std::string& s)
{
    if (s == "weno3js")  { return Scheme::weno3js;  }
    if (s == "upwind2")  { return Scheme::upwind2;  }
    if (s == "central2") { return Scheme::central2; }
    amrex::Abort("numerics.gradient_scheme = '" + s + "' is not recognized "
                 "(expected weno3js, upwind2, or central2)");
    return Scheme::weno3js;   // unreachable; silences the compiler
}

const char* SchemeName (Scheme s)
{
    switch (s) {
    case Scheme::central2: return "central2";
    case Scheme::upwind2:  return "upwind2";
    case Scheme::weno3js:  return "weno3js";
    }
    return "?";
}

void Numerics::Init ()
{
    amrex::ParmParse pp("numerics");
    const bool given = pp.query("gradient_scheme", s_name);
    s_scheme = ParseScheme(s_name);

    FWT_DEBUG_SECTION("Numerics");
    FWT_DEBUG("gradient_scheme  = " << s_name
              << (given ? "" : "   [default]"));
    FWT_DEBUG("stencil radius   = " << kStencilRadius
              << " cells   (fields need this many ghost cells)");
}

void AppendNumericsReport (const std::string& filename)
{
    if (!amrex::ParallelDescriptor::IOProcessor()) { return; }

    std::ofstream os(filename, std::ios::app);
    os << "# numerics\n";
    os << "numerics_gradient_scheme " << Numerics::name() << "\n";
    os << "numerics_stencil_radius " << kStencilRadius << "\n";
    os.close();
}

// ---------------------------------------------------------------------------
// Self test
// ---------------------------------------------------------------------------

namespace {

// A periodic test function and its exact derivative, so the study is
// measuring the scheme rather than a boundary treatment.
amrex::Real TestF (amrex::Real x)  { return std::sin(2.0 * M_PI * x); }
amrex::Real TestDF (amrex::Real x) { return 2.0 * M_PI * std::cos(2.0 * M_PI * x); }

struct Norms { amrex::Real linf; amrex::Real l2; };

// Both norms are reported, because for a nonlinear scheme they measure
// different things. WENO3-JS is third order where the field is smooth,
// but its Jiang-Shu weights lose an order at critical points, where the
// derivative and the smoothness indicators vanish together. L-infinity
// is set by exactly those points; L2 shows the order the scheme achieves
// over the field as a whole.
Norms Accumulate (const std::vector<amrex::Real>& err)
{
    amrex::Real linf = 0.0;
    amrex::Real sum2 = 0.0;
    for (amrex::Real e : err) {
        linf = std::max(linf, std::abs(e));
        sum2 += e * e;
    }
    return {linf, std::sqrt(sum2 / amrex::Real(err.size()))};
}

// Uniform periodic grid of n cells.
Norms UniformError (Scheme s, int n, amrex::Real a)
{
    const amrex::Real h = 1.0 / amrex::Real(n);
    std::vector<amrex::Real> f(n);
    for (int i = 0; i < n; ++i) {
        f[i] = TestF((amrex::Real(i) + 0.5) * h);
    }

    auto at = [&](int i) { return f[((i % n) + n) % n]; };   // periodic

    std::vector<amrex::Real> err(n);
    for (int i = 0; i < n; ++i) {
        const amrex::Real d = Derivative(s, at(i-2), at(i-1), at(i),
                                         at(i+1), at(i+2), a, h);
        err[i] = d - TestDF((amrex::Real(i) + 0.5) * h);
    }
    return Accumulate(err);
}

// Geometrically stretched grid, exercising the mapped form: the scheme
// runs on the index coordinate and h is the local metric dz/dk.
//
// The stretching ratio is chosen per resolution so the LAST cell is
// always kStretchTotal times the first. That holds the underlying
// mapping fixed as n grows, which is what makes this a convergence study
// at all -- keeping the ratio itself fixed would distort the grid
// further at every refinement and measure nothing.
Norms StretchedError (Scheme s, int n, amrex::Real a)
{
    constexpr amrex::Real kStretchTotal = 10.0;
    const amrex::Real r = std::pow(kStretchTotal, 1.0 / amrex::Real(n - 1));

    amrex::Real sum = 0.0;
    for (int k = 0; k < n; ++k) { sum += std::pow(r, amrex::Real(k)); }
    const amrex::Real dz0 = 1.0 / sum;

    std::vector<amrex::Real> zf(n + 1), zc(n), f(n);
    zf[0] = 0.0;
    for (int k = 0; k < n; ++k) {
        zf[k+1] = zf[k] + dz0 * std::pow(r, amrex::Real(k));
        zc[k] = 0.5 * (zf[k] + zf[k+1]);
        f[k] = TestF(zc[k]);
    }

    // No periodicity available here, so the stencil-width margin is
    // excluded and only the interior is measured.
    std::vector<amrex::Real> err;
    err.reserve(n);
    for (int k = kStencilRadius; k < n - kStencilRadius; ++k) {
        // Local metric dz/dk at the cell center, itself a second-order
        // central difference of the coordinate.
        const amrex::Real dzdk = 0.5 * (zc[k+1] - zc[k-1]);
        const amrex::Real d = Derivative(s, f[k-2], f[k-1], f[k],
                                         f[k+1], f[k+2], a, dzdk);
        err.push_back(d - TestDF(zc[k]));
    }
    return Accumulate(err);
}

amrex::Real Order (amrex::Real e_coarse, amrex::Real e_fine)
{
    if (e_fine <= 0.0 || e_coarse <= 0.0) { return 0.0; }
    return std::log(e_coarse / e_fine) / std::log(2.0);
}

// Convergence order is only half of what WENO is for. The other half is
// staying bounded across a discontinuity, which the linear third-order
// combination does not.
//
// Reconstructs the right-hand face value of every cell of a step
// profile and returns how far the worst one strays outside the range of
// the two cells it sits between. A bounded reconstruction returns zero;
// the unlimited linear combination overshoots.
amrex::Real StepOvershoot (bool limited)
{
    constexpr int n = 40;
    std::vector<amrex::Real> u(n);
    for (int i = 0; i < n; ++i) { u[i] = (i < n/2) ? 0.0 : 1.0; }

    amrex::Real worst = 0.0;
    for (int i = 1; i < n - 1; ++i) {
        amrex::Real f;
        if (limited) {
            f = detail::Weno3Recon(u[i-1], u[i], u[i+1]);
        } else {
            // The same two candidate polynomials at their fixed linear
            // weights 1/3 and 2/3: third order, and unbounded.
            const amrex::Real p0 = -0.5 * u[i-1] + 1.5 * u[i];
            const amrex::Real p1 = 0.5 * u[i] + 0.5 * u[i+1];
            f = (1.0 / 3.0) * p0 + (2.0 / 3.0) * p1;
        }
        const amrex::Real lo = std::min(u[i], u[i+1]);
        const amrex::Real hi = std::max(u[i], u[i+1]);
        worst = std::max(worst, std::max(lo - f, f - hi));
    }
    return std::max(worst, amrex::Real(0.0));
}

} // namespace

void RunGradientSelfTest (const std::string& filename)
{
    if (!amrex::ParallelDescriptor::IOProcessor()) { return; }

    const int ns[] = {64, 128, 256, 512};
    const Scheme schemes[] = {Scheme::central2, Scheme::upwind2,
                              Scheme::weno3js};
    // Both signs, so each scheme's two upwind branches are measured.
    const amrex::Real advect[] = {1.0, -1.0};

    std::ofstream os(filename);
    os << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10);
    os << "# gradient scheme convergence study\n";
    os << "# d/dx of sin(2 pi x). The uniform grid is periodic; the\n";
    os << "# stretched grid holds its total stretch fixed as it refines,\n";
    os << "# measures the interior only, and uses the mapped form with\n";
    os << "# h = dz/dk\n";
    os << "# scheme grid norm advect n error order\n";

    for (Scheme s : schemes) {
        for (amrex::Real a : advect) {
            for (int gi = 0; gi < 2; ++gi) {
                const bool uniform = (gi == 0);
                const char* grid = uniform ? "uniform" : "stretched";
                Norms prev {0.0, 0.0};
                for (int n : ns) {
                    const Norms e = uniform ? UniformError(s, n, a)
                                            : StretchedError(s, n, a);
                    const amrex::Real o_inf = (prev.linf > 0.0)
                        ? Order(prev.linf, e.linf) : 0.0;
                    const amrex::Real o_l2 = (prev.l2 > 0.0)
                        ? Order(prev.l2, e.l2) : 0.0;
                    os << SchemeName(s) << " " << grid << " linf " << a << " "
                       << n << " " << e.linf << " " << o_inf << "\n";
                    os << SchemeName(s) << " " << grid << " l2 " << a << " "
                       << n << " " << e.l2 << " " << o_l2 << "\n";
                    prev = e;
                }
            }
        }
    }
    // Boundedness across a discontinuity, the property the nonlinear
    // weights exist to provide.
    os << "# overshoot <reconstruction> <worst excursion outside the local "
          "data range>\n";
    os << "overshoot weno3js " << StepOvershoot(true) << "\n";
    os << "overshoot linear3 " << StepOvershoot(false) << "\n";

    os.close();

    amrex::Print() << "Wrote gradient self test to " << filename << "\n";
    FWT_DEBUG("gradient self test written: " << filename);
}

} // namespace fwt
