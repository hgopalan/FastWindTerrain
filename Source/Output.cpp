#include "Output.H"
#include "Debug.H"

#include <AMReX_MultiFab.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_GpuContainers.H>

namespace fwt {

void WritePlotfile (const std::string& plotfilename,
                    const Grid& grid,
                    const Terrain& terrain,
                    const Inflow& inflow,
                    const Poisson& poisson,
                    const amrex::MultiFab& vel0,
                    const Anisotropy& aniso)
{
    constexpr int ncomp = 15;
    amrex::MultiFab mf(grid.ba(), grid.dm(), ncomp, 0);

    // z_face lives on the host; copy it once so the fill kernel is valid
    // in a GPU build too.
    amrex::Gpu::DeviceVector<amrex::Real> d_z_face(grid.z_face().size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, grid.z_face().begin(),
                          grid.z_face().end(), d_z_face.begin());
    amrex::Gpu::streamSynchronize();
    const amrex::Real* zf = d_z_face.data();

    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const& a  = mf.array(mfi);
        auto const& zt = terrain.z_terrain().const_array(mfi);
        auto const& mk = terrain.mask().const_array(mfi);
        auto const& vl = inflow.velocity().const_array(mfi);
        auto const& sg = poisson.sigma().const_array(mfi);
        auto const& v0 = vel0.const_array(mfi);
        auto const& ah = aniso.alpha_h().const_array(mfi);
        auto const& av = aniso.alpha_v().const_array(mfi);
        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            a(i,j,k,0) = amrex::Real(0.5) * (zf[k] + zf[k+1]);   // z_cc
            a(i,j,k,1) = zf[k+1] - zf[k];                        // dz
            a(i,j,k,2) = zt(i,j,k);                              // terrain_z
            a(i,j,k,3) = amrex::Real(mk(i,j,k));                 // mask
            a(i,j,k,4) = vl(i,j,k,0);                            // u
            a(i,j,k,5) = vl(i,j,k,1);                            // v
            a(i,j,k,6) = vl(i,j,k,2);                            // w
            a(i,j,k,7) = sg(i,j,k,0);                            // sigma_x
            a(i,j,k,8) = sg(i,j,k,1);                            // sigma_y
            a(i,j,k,9) = sg(i,j,k,2);                            // sigma_z
            a(i,j,k,10) = v0(i,j,k,0);                           // u0
            a(i,j,k,11) = v0(i,j,k,1);                           // v0
            a(i,j,k,12) = v0(i,j,k,2);                           // w0
            a(i,j,k,13) = ah(i,j,k);                             // alpha_h
            a(i,j,k,14) = av(i,j,k);                             // alpha_v
        });
    }

    amrex::WriteSingleLevelPlotfile(plotfilename, mf,
                                    {"z_cc", "dz", "terrain_z", "mask",
                                     "u", "v", "w",
                                     "sigma_x", "sigma_y", "sigma_z",
                                     "u0", "v0", "w0",
                                     "alpha_h", "alpha_v"},
                                    grid.geom(), 0.0, 0);

    FWT_DEBUG("wrote plotfile: " << plotfilename << "  (" << ncomp
              << " components: z_cc, dz, terrain_z, mask, u, v, w, "
                 "sigma_x, sigma_y, sigma_z, u0, v0, w0, "
                 "alpha_h, alpha_v; "
              << grid.ba().size() << " boxes)");
}

} // namespace fwt
