#include "BoundaryConditions.H"
#include "Debug.H"

#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>

namespace fwt {

namespace {
    // A face counts as tangential when its net outward flux is this
    // small relative to the largest face flux in the run.
    constexpr amrex::Real kFluxRelTol = 1.0e-8;

    // face index <-> (direction, side)
    constexpr int face_dir (int face)  { return face / 2; }
    constexpr int face_side (int face) { return (face % 2 == 0) ? -1 : +1; }
}

const char* BoundaryConditions::FaceName (int face)
{
    static const char* names[nfaces] = {"xlo", "xhi", "ylo", "yhi",
                                        "zlo", "zhi"};
    return names[face];
}

const char* BoundaryConditions::TypeName (FaceType t)
{
    switch (t) {
    case FaceType::inflow:     return "inflow";
    case FaceType::outflow:    return "outflow";
    case FaceType::tangential: return "tangential";
    case FaceType::noflow:     return "noflow";
    }
    return "?";
}

const char* BoundaryConditions::LambdaName (LambdaBC b)
{
    return (b == LambdaBC::dirichlet) ? "dirichlet" : "neumann";
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

void BoundaryConditions::Classify (const Grid& grid, const Terrain& terrain,
                                   const amrex::MultiFab& vel)
{
    // Net outward flux through each lateral face, over its open cells
    // only and using the stretched dz(k). Classifying from the field
    // rather than from (u_ref, v_ref) is what makes userfile mode work,
    // where there is no reference vector to test against.
    const amrex::Box& domain = grid.geom().Domain();
    const amrex::Real dx = grid.geom().CellSize(0);
    const amrex::Real dy = grid.geom().CellSize(1);
    const amrex::Vector<amrex::Real>& z_face = grid.z_face();

    m_flux.fill(0.0);

    for (int face = 0; face < 4; ++face) {
        const int d = face_dir(face);
        const int side = face_side(face);
        const amrex::Real lateral = (d == 0) ? dy : dx;

        amrex::Real flux = 0.0;

        for (amrex::MFIter mfi(vel); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.validbox();
            const int dom_edge = (side < 0) ? domain.smallEnd(d)
                                            : domain.bigEnd(d);
            const int box_edge = (side < 0) ? bx.smallEnd(d) : bx.bigEnd(d);
            if (box_edge != dom_edge) { continue; }

            amrex::Box fbx(bx);
            fbx.setSmall(d, dom_edge);
            fbx.setBig(d, dom_edge);

            auto const& a  = vel.const_array(mfi);
            auto const& mk = terrain.mask().const_array(mfi);

            const auto lo = amrex::lbound(fbx);
            const auto hi = amrex::ubound(fbx);
            for (int k = lo.z; k <= hi.z; ++k) {
            for (int j = lo.y; j <= hi.y; ++j) {
            for (int i = lo.x; i <= hi.x; ++i) {
                if (mk(i,j,k) == Terrain::kSolid) { continue; }
                const amrex::Real area = lateral * (z_face[k+1] - z_face[k]);
                flux += amrex::Real(side) * a(i,j,k,d) * area;
            }}}
        }

        amrex::ParallelDescriptor::ReduceRealSum(flux);
        m_flux[face] = flux;
    }

    // Tolerance relative to the largest face flux, so "no normal flow"
    // means no flow compared with the flow that is actually there.
    amrex::Real scale = 0.0;
    for (int face = 0; face < 4; ++face) {
        scale = std::max(scale, std::abs(m_flux[face]));
    }
    const amrex::Real tol = kFluxRelTol * scale;

    for (int face = 0; face < 4; ++face) {
        if (m_flux[face] < -tol) {
            m_type[face] = FaceType::inflow;
        } else if (m_flux[face] > tol) {
            m_type[face] = FaceType::outflow;
        } else {
            m_type[face] = FaceType::tangential;
        }
    }

    // Ground and domain top both carry no flow through them.
    m_type[4] = FaceType::noflow;      // zlo
    m_type[5] = FaceType::noflow;      // zhi

    m_n_inflow = m_n_outflow = m_n_tangential = 0;
    for (int face = 0; face < 4; ++face) {
        switch (m_type[face]) {
        case FaceType::inflow:     ++m_n_inflow;     break;
        case FaceType::outflow:    ++m_n_outflow;    break;
        case FaceType::tangential: ++m_n_tangential; break;
        default: break;
        }
    }

    // Velocity is prescribed exactly on the inflow faces. One face for an
    // axis-aligned wind, two for an oblique one; anything else means the
    // classification is not what this code assumes.
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_n_inflow >= 1,
        "no lateral face is an inflow face: the initial field carries no "
        "net flow into the domain through any of xlo/xhi/ylo/yhi");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_n_inflow <= 2,
        "velocity may be prescribed on at most two of xlo/xhi/ylo/yhi; "
        "with more, every face would be Neumann in lambda and the Poisson "
        "problem would be singular");

    // Velocity treatment determines the lambda condition per face.
    for (int face = 0; face < nfaces; ++face) {
        switch (m_type[face]) {
        case FaceType::inflow:                 // prescribed -> no correction
        case FaceType::noflow:                 // ground, top
            m_lambda[face] = LambdaBC::neumann;
            break;
        case FaceType::outflow:                // free to adjust
        case FaceType::tangential:             // open, see the header
            m_lambda[face] = LambdaBC::dirichlet;
            break;
        }
    }

    m_n_dirichlet = 0;
    for (int face = 0; face < nfaces; ++face) {
        if (m_lambda[face] == LambdaBC::dirichlet) { ++m_n_dirichlet; }
    }

    // The check that actually matters, made on the assembled array so
    // that any future route to an all-Neumann problem trips it too.
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_n_dirichlet >= 1,
        "every face is Neumann in lambda: the Poisson operator is singular "
        "and the solve cannot be trusted. At least one face must be open.");
}

// ---------------------------------------------------------------------------
// Ghost cells
// ---------------------------------------------------------------------------

void BoundaryConditions::FillGhosts (const Grid& grid, const Terrain& terrain,
                                     const Inflow& inflow,
                                     amrex::MultiFab& vel)
{
    // Interior ghost cells first; the physical faces are filled below and
    // must not be overwritten afterwards.
    vel.FillBoundary(grid.geom().periodicity());

    const amrex::Box& domain = grid.geom().Domain();
    const amrex::Real dx = grid.geom().CellSize(0);
    const amrex::Real dy = grid.geom().CellSize(1);
    const amrex::Real xlo = grid.geom().ProbLo(0);
    const amrex::Real ylo = grid.geom().ProbLo(1);

    const int nx = grid.nx();
    const amrex::Vector<amrex::Real>& z_cc = grid.z_cc();
    const std::vector<amrex::Real>& h = terrain.column_heights();

    // The ghost fill evaluates the profile and reads the terrain column
    // array, both host data, so it runs on the host. It touches only the
    // boundary layers, which is a vanishing fraction of the field.
    for (amrex::MFIter mfi(vel); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const& a = vel.array(mfi);
        auto const& mk = terrain.mask().const_array(mfi);

        for (int face = 0; face < nfaces; ++face) {
            const int d = face_dir(face);
            const int side = face_side(face);

            // Does this box touch this physical face?
            const int dom_edge = (side < 0) ? domain.smallEnd(d)
                                            : domain.bigEnd(d);
            const int box_edge = (side < 0) ? bx.smallEnd(d) : bx.bigEnd(d);
            if (box_edge != dom_edge) { continue; }

            // The one-cell ghost layer just outside this face.
            amrex::Box gbx(bx);
            if (side < 0) {
                gbx.setSmall(d, dom_edge - 1);
                gbx.setBig(d, dom_edge - 1);
            } else {
                gbx.setSmall(d, dom_edge + 1);
                gbx.setBig(d, dom_edge + 1);
            }

            const auto lo = amrex::lbound(gbx);
            const auto hi = amrex::ubound(gbx);
            for (int k = lo.z; k <= hi.z; ++k) {
            for (int j = lo.y; j <= hi.y; ++j) {
            for (int i = lo.x; i <= hi.x; ++i) {
                // The interior cell this ghost mirrors.
                const int ii = (d == 0) ? i - side : i;
                const int jj = (d == 1) ? j - side : j;
                const int kk = (d == 2) ? k - side : k;

                switch (m_type[face]) {

                case FaceType::inflow: {
                    // Dirichlet: the profile evaluated at the ghost cell
                    // center. Terrain is not defined outside the domain,
                    // so the height comes from the adjacent interior
                    // column, which keeps the ghost on the same ground as
                    // the cell it feeds.
                    const amrex::Real xq = xlo + (amrex::Real(i) + 0.5) * dx;
                    const amrex::Real yq = ylo + (amrex::Real(j) + 0.5) * dy;
                    const amrex::Real zt = h[std::size_t(jj)*nx + ii];
                    const amrex::Real z_agl = z_cc[kk] - zt;

                    if (mk(ii,jj,kk) == Terrain::kSolid) {
                        // Terrain blocks the face here. Prescribing the
                        // profile would drive flow straight into the
                        // ground; the face is simply shut.
                        a(i,j,k,0) = 0.0;
                        a(i,j,k,1) = 0.0;
                        a(i,j,k,2) = 0.0;
                        break;
                    }

                    amrex::Real u, v, w;
                    inflow.VelocityAt(xq, yq, z_agl, u, v, w);
                    a(i,j,k,0) = u;
                    a(i,j,k,1) = v;
                    a(i,j,k,2) = w;
                    break;
                }

                case FaceType::outflow:
                case FaceType::tangential:
                    // Zero gradient: the flow leaves as it arrives.
                    a(i,j,k,0) = a(ii,jj,kk,0);
                    a(i,j,k,1) = a(ii,jj,kk,1);
                    a(i,j,k,2) = a(ii,jj,kk,2);
                    break;

                case FaceType::noflow:
                    // Reflect the normal component so it averages to zero
                    // ON the face, and leave the tangential components
                    // with zero gradient (free slip).
                    a(i,j,k,0) = a(ii,jj,kk,0);
                    a(i,j,k,1) = a(ii,jj,kk,1);
                    a(i,j,k,2) = -a(ii,jj,kk,2);
                    break;
                }
            }}}
        }
    }
}

void BoundaryConditions::Build (const Grid& grid, const Terrain& terrain,
                                const Inflow& inflow, amrex::MultiFab& vel)
{
    Classify(grid, terrain, vel);
    FillGhosts(grid, terrain, inflow, vel);

    amrex::Print() << "Boundary conditions: ";
    for (int face = 0; face < 4; ++face) {
        amrex::Print() << FaceName(face) << "=" << TypeName(m_type[face])
                       << (face < 3 ? ", " : "");
    }
    amrex::Print() << "\n";

    if (Debug::Enabled()) {
        FWT_DEBUG_SECTION("Boundary conditions");
        FWT_DEBUG("classified from the initial field's face fluxes");
        for (int face = 0; face < nfaces; ++face) {
            if (face < 4) {
                FWT_DEBUG("  " << FaceName(face) << "  outward flux = "
                                << m_flux[face] << " m^3/s  -> "
                                << TypeName(m_type[face])
                                << ", lambda " << LambdaName(m_lambda[face]));
            } else {
                FWT_DEBUG("  " << FaceName(face) << "  " << TypeName(m_type[face])
                                << ", lambda " << LambdaName(m_lambda[face]));
            }
        }
        FWT_DEBUG("inflow faces     = " << m_n_inflow
                  << "   (velocity prescribed)");
        FWT_DEBUG("outflow faces    = " << m_n_outflow);
        FWT_DEBUG("tangential faces = " << m_n_tangential
                  << "   (treated as open)");
        FWT_DEBUG("lambda Dirichlet = " << m_n_dirichlet
                  << " of " << nfaces
                  << "   (>= 1 required, else the operator is singular)");
    }

    // The dump is a regtest aid: one row per boundary cell, so a checker
    // can verify every one rather than a sample.
    std::string dump_file;
    amrex::ParmParse pp("bc");
    if (pp.query("dump_file", dump_file) && !dump_file.empty()) {
        WriteDump(dump_file, grid, vel);
    }
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

void BoundaryConditions::AppendReport (const std::string& filename) const
{
    if (!amrex::ParallelDescriptor::IOProcessor()) { return; }

    std::ofstream os(filename, std::ios::app);
    os << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10);
    os << "# boundary conditions (Phase 4)\n";
    for (int face = 0; face < nfaces; ++face) {
        os << "bc_" << FaceName(face) << " " << TypeName(m_type[face])
           << " " << LambdaName(m_lambda[face]) << "\n";
    }
    for (int face = 0; face < 4; ++face) {
        os << "bc_flux_" << FaceName(face) << " " << m_flux[face] << "\n";
    }
    os << "bc_n_inflow " << m_n_inflow << "\n";
    os << "bc_n_outflow " << m_n_outflow << "\n";
    os << "bc_n_tangential " << m_n_tangential << "\n";
    os << "bc_n_lambda_dirichlet " << m_n_dirichlet << "\n";
    os.close();

    FWT_DEBUG("appended boundary-condition summary to " << filename);
}

void BoundaryConditions::WriteDump (const std::string& filename,
                                    const Grid& grid,
                                    const amrex::MultiFab& vel) const
{
    // Gathering ghost layers across ranks would need a communication path
    // that nothing else in the code wants; this is a single-rank test aid.
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        amrex::ParallelDescriptor::NProcs() == 1,
        "bc.dump_file is a single-rank regtest aid and cannot be written "
        "from a multi-rank run");

    std::ofstream os(filename);
    os << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10);
    os << "# face i j k  ghost_u ghost_v ghost_w  int_u int_v int_w\n";
    os << "# one row per boundary cell; ghost is the cell outside the "
          "face, int the interior cell it mirrors\n";

    const amrex::Box& domain = grid.geom().Domain();

    for (amrex::MFIter mfi(vel); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const& a = vel.const_array(mfi);

        for (int face = 0; face < nfaces; ++face) {
            const int d = face_dir(face);
            const int side = face_side(face);
            const int dom_edge = (side < 0) ? domain.smallEnd(d)
                                            : domain.bigEnd(d);
            const int box_edge = (side < 0) ? bx.smallEnd(d) : bx.bigEnd(d);
            if (box_edge != dom_edge) { continue; }

            amrex::Box gbx(bx);
            gbx.setSmall(d, dom_edge + side);
            gbx.setBig(d, dom_edge + side);

            const auto lo = amrex::lbound(gbx);
            const auto hi = amrex::ubound(gbx);
            for (int k = lo.z; k <= hi.z; ++k) {
            for (int j = lo.y; j <= hi.y; ++j) {
            for (int i = lo.x; i <= hi.x; ++i) {
                const int ii = (d == 0) ? i - side : i;
                const int jj = (d == 1) ? j - side : j;
                const int kk = (d == 2) ? k - side : k;
                os << FaceName(face) << " " << i << " " << j << " " << k
                   << " " << a(i,j,k,0) << " " << a(i,j,k,1)
                   << " " << a(i,j,k,2)
                   << " " << a(ii,jj,kk,0) << " " << a(ii,jj,kk,1)
                   << " " << a(ii,jj,kk,2) << "\n";
            }}}
        }
    }
    os.close();

    FWT_DEBUG("wrote boundary dump: " << filename);
}

} // namespace fwt
