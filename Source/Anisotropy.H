#ifndef FWT_ANISOTROPY_H_
#define FWT_ANISOTROPY_H_

#include "Grid.H"
#include "Terrain.H"
#include "Error.H"

#include <AMReX_MultiFab.H>

#include <string>

namespace fwt {

// -----------------------------------------------------------------------
// Anisotropy: the cell-local variational weights alpha_h(i,j,k) and
// alpha_v(i,j,k) that the Poisson operator is built from.
//
//     alpha_v = clamp( alpha_v_base * f_slope * f_Ri * f_Fr )
//     alpha_h = clamp( alpha_h_base * [f_slope] * f_Ri * f_Fr )
//
// with, following massconsistent_amr's cell_local_anisotropy.H,
//
//     slope    = |grad z_terrain|, from central differences
//     slope_3d = slope * exp(-z_agl / decay_height)
//     f_slope  = exp(-slope_3d / slope_scale)
//
// so the suppression is strongest at the surface on steep ground and
// decays with height above it. alpha is a TRANSMISSIVITY (see Poisson),
// so a smaller alpha_v means less vertical adjustment: flow is pushed
// around steep terrain rather than over it.
//
// NEUTRAL CASE. f_Ri and f_Fr are hooks for the stability and Froude
// terms and both return exactly 1 here, which is the neutral case. They
// are wired into both alphas so that adding them later needs no
// structural change.
//
// ALPHA_H. massconsistent_amr holds alpha_h at its base value and varies
// only alpha_v. Both are full fields here, and anisotropy.alpha_h_mode
// chooses whether the slope factor reaches the horizontal weight:
//
//   base  (default)  alpha_h = alpha_h_base * f_Ri * f_Fr
//                    -- reproduces massconsistent_amr exactly in the
//                    neutral case, where it is simply the base value
//   slope            alpha_h also carries f_slope
//
// Leaving alpha_h alone is what makes the RATIO alpha_v/alpha_h fall on
// slopes, which is the behaviour the model wants; scaling both by the
// same factor would suppress the vertical adjustment relative to nothing.
//
// Disabled by default, again matching massconsistent_amr: with
// anisotropy.enable = 0 both fields hold their base values and the
// operator is exactly what the earlier phases built.
// -----------------------------------------------------------------------
class Anisotropy
{
public:

    struct Params
    {
        int enable = 0;
        std::string source = "slope";
        std::string alpha_h_mode = "base";
        amrex::Real slope_scale = 0.5;
        amrex::Real decay_height = 500.0;
        amrex::Real min_factor = 0.05;
        amrex::Real max_factor = 2.0;

        // The base values live with the operator they feed, so they come
        // from the Poisson parameters rather than being duplicated here.
        // FromParmParse reads them out of poisson.*; the dict path is
        // handed them by Solver, which has both in hand.
        amrex::Real alpha_h_base = 1.0;
        amrex::Real alpha_v_base = 1.0;

        static Params FromParmParse ();
        void Validate () const;
    };

    void Build (const Grid& grid, const Terrain& terrain,
                const Params& params);
    void Build (const Grid& grid, const Terrain& terrain);

    // Cell-centered, one component each.
    const amrex::MultiFab& alpha_h () const { return m_alpha_h; }
    const amrex::MultiFab& alpha_v () const { return m_alpha_v; }

    bool enabled ()             const { return m_enable != 0; }
    amrex::Real alpha_h_base () const { return m_alpha_h_base; }
    amrex::Real alpha_v_base () const { return m_alpha_v_base; }
    const Params& params () const { return m_params; }

    // Extrema over fluid cells, for the report.
    amrex::Real alpha_v_min () const { return m_av_min; }
    amrex::Real alpha_v_max () const { return m_av_max; }
    amrex::Real alpha_h_min () const { return m_ah_min; }
    amrex::Real alpha_h_max () const { return m_ah_max; }
    amrex::Real slope_max ()   const { return m_slope_max; }

    void AppendReport (const std::string& filename) const;

private:


    Params      m_params;
    int         m_enable = 0;
    std::string m_source = "slope";
    std::string m_alpha_h_mode = "base";
    amrex::Real m_alpha_h_base = 1.0;
    amrex::Real m_alpha_v_base = 1.0;
    amrex::Real m_slope_scale = 0.5;
    amrex::Real m_decay_height = 500.0;
    // The clamp massconsistent_amr applies, as a factor on the base.
    amrex::Real m_min_factor = 0.05;
    amrex::Real m_max_factor = 2.0;

    amrex::MultiFab m_alpha_h;
    amrex::MultiFab m_alpha_v;

    amrex::Real m_av_min = 0.0, m_av_max = 0.0;
    amrex::Real m_ah_min = 0.0, m_ah_max = 0.0;
    amrex::Real m_slope_max = 0.0;
};

} // namespace fwt

#endif
