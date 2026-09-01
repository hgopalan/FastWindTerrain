#ifndef FWT_BOUNDARY_CONDITIONS_H_
#define FWT_BOUNDARY_CONDITIONS_H_

#include "Grid.H"
#include "Terrain.H"
#include "Inflow.H"

#include <AMReX_MultiFab.H>

#include <array>
#include <string>

namespace fwt {

// -----------------------------------------------------------------------
// BoundaryConditions: classifies each domain face against the wind
// direction and fills the velocity ghost cells accordingly.
//
// LATERAL FACES are classified from the net outward volumetric flux the
// initial field carries through each one, integrated over its open
// (fluid) cells:
//
//   flux < 0   inflow      velocity prescribed from the profile
//   flux > 0   outflow     zero-gradient, velocity free to adjust
//   flux = 0   tangential  no normal flow either way
//
// Classifying from the field rather than from (u_ref, v_ref) matters:
// in userfile mode there is no reference vector at all, and a file may
// carry a direction that varies across the domain. For the analytic
// laws, whose direction is uniform, this gives exactly the same answer
// as testing the wind vector against each face normal.
//
// An oblique wind gives two inflow faces and two outflow faces; an
// axis-aligned wind gives one of each plus two tangential faces. Zero
// inflow faces is an error: there would be nowhere for the wind to
// enter.
//
// TANGENTIAL FACES are treated as OPEN (zero-gradient), not sealed.
// Terrain deflects flow sideways, and that is exactly what a
// mass-consistent adjustment over complex terrain has to be free to do;
// sealed side walls would channel the deflected flow back into the
// domain and show up as a spurious speed-up around hills.
//
// TOP is w = 0: no flow through the domain top.
// BOTTOM is ground, and the terrain surface inside the domain is
// handled by the immersed-boundary mask rather than by a face
// condition.
//
// LAMBDA BOUNDARY CONDITIONS follow from the velocity treatment:
//
//   velocity prescribed (inflow)     -> the correction must vanish
//                                       there -> Neumann on lambda
//   velocity free (outflow/tangential) -> Dirichlet on lambda
//   no flow (top, ground)            -> Neumann on lambda
//
// The Poisson operator is singular if every face is Neumann: lambda is
// then determined only up to a constant, and an RHS that does not
// integrate to zero has no solution at all, so the solve would either
// fail or return something that quietly does not conserve mass. The
// class therefore asserts that AT LEAST ONE FACE IS DIRICHLET, checked
// on the assembled array rather than on the wind classification, so
// that every route to an all-Neumann problem is caught.
// -----------------------------------------------------------------------
class BoundaryConditions
{
public:

    enum class FaceType { inflow, outflow, tangential, noflow };
    enum class LambdaBC { dirichlet, neumann };

    // Face ordering: xlo, xhi, ylo, yhi, zlo, zhi.
    static constexpr int nfaces = 6;
    static const char* FaceName (int face);
    static const char* TypeName (FaceType t);
    static const char* LambdaName (LambdaBC b);

    // Classifies the faces, then fills the velocity ghost cells. The
    // velocity MultiFab is modified in place, so it is taken by
    // reference rather than read off the Inflow object.
    void Build (const Grid& grid, const Terrain& terrain,
                const Inflow& inflow, amrex::MultiFab& vel);

    FaceType type (int face)      const { return m_type[face]; }
    LambdaBC lambda_bc (int face) const { return m_lambda[face]; }

    int n_inflow ()     const { return m_n_inflow; }
    int n_outflow ()    const { return m_n_outflow; }
    int n_tangential () const { return m_n_tangential; }
    int n_dirichlet ()  const { return m_n_dirichlet; }

    // Net outward flux each lateral face carries [m^3/s]; what the
    // classification was made from.
    amrex::Real face_flux (int face) const { return m_flux[face]; }

    // Re-applies the EXISTING classification to the ghost cells, without
    // reclassifying. The projection changes the interior, so the ghosts
    // have to be refreshed before anything reads a stencil near a face --
    // but which face is an inflow face is a property of the incoming
    // wind, not of the corrected field, and must not be re-derived from
    // it.
    void RefillGhosts (const Grid& grid, const Terrain& terrain,
                       const Inflow& inflow, amrex::MultiFab& vel)
    { FillGhosts(grid, terrain, inflow, vel); }

    void AppendReport (const std::string& filename) const;

    // Writes one row per boundary cell (face, i, j, k, ghost u/v/w,
    // interior u/v/w) so a regtest can check every boundary cell rather
    // than a sample. Written only when bc.dump_file is set; this is a
    // test aid, not a production output path.
    void WriteDump (const std::string& filename, const Grid& grid,
                    const amrex::MultiFab& vel) const;

private:

    void Classify (const Grid& grid, const Terrain& terrain,
                   const amrex::MultiFab& vel);
    void FillGhosts (const Grid& grid, const Terrain& terrain,
                     const Inflow& inflow, amrex::MultiFab& vel);

    std::array<FaceType, nfaces> m_type {};
    std::array<LambdaBC, nfaces> m_lambda {};

    int m_n_inflow = 0;
    int m_n_outflow = 0;
    int m_n_tangential = 0;
    int m_n_dirichlet = 0;

    std::array<amrex::Real, nfaces> m_flux {};
};

} // namespace fwt

#endif
