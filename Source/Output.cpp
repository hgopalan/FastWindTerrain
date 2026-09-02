#include "Output.H"
#include "Debug.H"

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_ParallelDescriptor.H>

#include <fstream>
#include <iomanip>

namespace fwt {

// A gathered ascii file is one rank writing every cell. Past this many
// cells that is slow enough and large enough to be worth saying out
// loud, since the only way to ask for it is to have chosen it.
static constexpr long kAsciiWarnCells = 2000000;

FieldFormat ParseFieldFormat (const std::string& s)
{
    if (s == "plt")   { return FieldFormat::plt; }
    if (s == "ascii") { return FieldFormat::ascii; }
    if (s == "both")  { return FieldFormat::both; }
    amrex::Abort("ERROR: unrecognized output.format = '" + s +
                 "'. Valid values are plt, ascii, both.");
    return FieldFormat::plt;
}

OutputFields CollectOutputFields (const Grid& grid,
                                  const Terrain& terrain,
                                  const Inflow& inflow,
                                  const Poisson& poisson,
                                  const amrex::MultiFab& vel0,
                                  const Anisotropy& aniso,
                                  const amrex::MultiFab& div)
{
    OutputFields out;
    out.names = {"z_cc", "dz", "terrain_z", "mask",
                 "u", "v", "w",
                 "sigma_x", "sigma_y", "sigma_z",
                 "u0", "v0", "w0",
                 "alpha_h", "alpha_v",
                 "lambda", "divergence"};

    const int ncomp = out.ncomp();
    out.mf.define(grid.ba(), grid.dm(), ncomp, 0);

    // z_face lives on the host; copy it once so the fill kernel is valid
    // in a GPU build too.
    amrex::Gpu::DeviceVector<amrex::Real> d_z_face(grid.z_face().size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, grid.z_face().begin(),
                          grid.z_face().end(), d_z_face.begin());
    amrex::Gpu::streamSynchronize();
    const amrex::Real* zf = d_z_face.data();

    for (amrex::MFIter mfi(out.mf); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const& a  = out.mf.array(mfi);
        auto const& zt = terrain.z_terrain().const_array(mfi);
        auto const& mk = terrain.mask().const_array(mfi);
        auto const& vl = inflow.velocity().const_array(mfi);
        auto const& sg = poisson.sigma().const_array(mfi);
        auto const& v0 = vel0.const_array(mfi);
        auto const& ah = aniso.alpha_h().const_array(mfi);
        auto const& av = aniso.alpha_v().const_array(mfi);
        auto const& lm = poisson.lambda().const_array(mfi);
        auto const& dv = div.const_array(mfi);
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

            // lambda is nodal; the cell value is the average of the
            // eight nodes around it. All eight are inside the nodal
            // valid box of this cell's box, so no ghost is read.
            a(i,j,k,15) = amrex::Real(0.125) *
                (lm(i  ,j  ,k  ) + lm(i+1,j  ,k  ) +
                 lm(i  ,j+1,k  ) + lm(i+1,j+1,k  ) +
                 lm(i  ,j  ,k+1) + lm(i+1,j  ,k+1) +
                 lm(i  ,j+1,k+1) + lm(i+1,j+1,k+1));

            a(i,j,k,16) = dv(i,j,k);                             // divergence
        });
    }

    return out;
}

void WritePlotfile (const std::string& plotfilename,
                    const Grid& grid,
                    const OutputFields& fields)
{
    amrex::WriteSingleLevelPlotfile(plotfilename, fields.mf, fields.names,
                                    grid.geom(), 0.0, 0);

    FWT_DEBUG("wrote plotfile: " << plotfilename << "  ("
              << fields.ncomp() << " components; "
              << grid.ba().size() << " boxes)");
}

void WriteAscii (const std::string& filename,
                 const Grid& grid,
                 const OutputFields& fields)
{
    const amrex::Box& domain = grid.geom().Domain();
    const long ncell = domain.numPts();
    const int ncomp = fields.ncomp();

    if (ncell > kAsciiWarnCells) {
        amrex::Print() << "  WARNING: output.format = ascii on "
                       << ncell << " cells. This is a single gathered "
                          "plain-text file written by one rank; it is a "
                          "regtest aid, not a production output path.\n";
    }

    // Gather onto one box owned by rank 0. The requirement is a SINGLE
    // file: per-rank or per-box files would push the reassembly into
    // every checker, which is where a format quietly stops being one
    // format.
    amrex::BoxArray ba_one(domain);
    amrex::Vector<int> pmap {0};
    amrex::DistributionMapping dm_one(pmap);
    amrex::MultiFab all(ba_one, dm_one, ncomp, 0);
    all.ParallelCopy(fields.mf, 0, 0, ncomp);

    if (!amrex::ParallelDescriptor::IOProcessor()) { return; }

    const amrex::Real dx = grid.geom().CellSize(0);
    const amrex::Real dy = grid.geom().CellSize(1);
    const amrex::Real xlo = grid.geom().ProbLo(0);
    const amrex::Real ylo = grid.geom().ProbLo(1);

    std::ofstream os(filename);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(os.good(),
        "could not open the ascii output file for writing");

    os << "# FastWindTerrain ascii field output\n";
    os << "# one row per cell; i fastest, k slowest\n";
    os << "# n_cell " << grid.nx() << " " << grid.ny() << " "
       << grid.nz() << "\n";
    os << "# n_rows " << ncell << "\n";
    os << "# ncomp " << ncomp << "\n";
    os << "# x, y are cell centers [m]; the cell-center height is the "
          "z_cc column\n";
    os << "# columns: i j k x y";
    for (const std::string& n : fields.names) { os << " " << n; }
    os << "\n";

    // Full double precision: the ascii file has to be comparable with the
    // binary plotfile to round-off, or the regtest that checks the two
    // backends agree would only be measuring the formatting.
    os << std::setprecision(17);

    for (amrex::MFIter mfi(all); mfi.isValid(); ++mfi) {
        auto const& a = all.const_array(mfi);
        const amrex::Box& bx = mfi.validbox();
        for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k) {
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
        for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
            os << i << " " << j << " " << k
               << " " << (xlo + (amrex::Real(i) + amrex::Real(0.5)) * dx)
               << " " << (ylo + (amrex::Real(j) + amrex::Real(0.5)) * dy);
            for (int n = 0; n < ncomp; ++n) { os << " " << a(i,j,k,n); }
            os << "\n";
        }}}
    }

    FWT_DEBUG("wrote ascii field output: " << filename << "  ("
              << ncomp << " components, " << ncell << " rows)");
}

} // namespace fwt
