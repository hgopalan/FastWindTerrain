#include "FieldIO.H"
#include "Error.H"

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_ParallelDescriptor.H>

namespace fwt {

namespace {

// One box covering the whole (possibly nodal) index space of mf, owned
// by the IO rank.
template <typename MF>
MF SingleBoxCopy (const MF& mf)
{
    const amrex::Box domain = mf.boxArray().minimalBox();
    amrex::BoxArray ba_one(domain);
    amrex::Vector<int> pmap {0};
    amrex::DistributionMapping dm_one(pmap);

    MF all(ba_one, dm_one, mf.nComp(), 0);
    all.ParallelCopy(mf, 0, 0, mf.nComp());
    return all;
}

template <typename MF, typename T>
amrex::Vector<T> GatherImpl (const MF& mf)
{
    MF all = SingleBoxCopy(mf);

    amrex::Vector<T> out;
    for (amrex::MFIter mfi(all); mfi.isValid(); ++mfi) {
        const auto& fab = all[mfi];
        const long n = long(fab.box().numPts()) * mf.nComp();
        out.resize(n);
        // The FAB's own layout is component slowest, i fastest -- which
        // is the layout this buffer is defined to have, so the whole
        // thing is one contiguous copy.
        const T* src = fab.dataPtr();
        std::copy(src, src + n, out.begin());
    }
    return out;   // empty on every rank but the one owning the box
}

template <typename MF>
long SizeImpl (const MF& mf)
{
    return long(mf.boxArray().minimalBox().numPts()) * mf.nComp();
}

} // namespace

amrex::Vector<amrex::Real> GatherField (const amrex::MultiFab& mf)
{
    return GatherImpl<amrex::MultiFab, amrex::Real>(mf);
}

amrex::Vector<int> GatherField (const amrex::iMultiFab& mf)
{
    return GatherImpl<amrex::iMultiFab, int>(mf);
}

long FieldSize (const amrex::MultiFab& mf)  { return SizeImpl(mf); }
long FieldSize (const amrex::iMultiFab& mf) { return SizeImpl(mf); }

void ScatterField (const amrex::Vector<amrex::Real>& buffer,
                   amrex::MultiFab& mf)
{
    const long expected = FieldSize(mf);
    if (amrex::ParallelDescriptor::IOProcessor() &&
        long(buffer.size()) != expected) {
        throw InputError(
            "field has " + std::to_string(expected) + " values but " +
            std::to_string(buffer.size()) + " were given");
    }

    const amrex::Box domain = mf.boxArray().minimalBox();
    amrex::BoxArray ba_one(domain);
    amrex::Vector<int> pmap {0};
    amrex::DistributionMapping dm_one(pmap);

    amrex::MultiFab all(ba_one, dm_one, mf.nComp(), 0);
    for (amrex::MFIter mfi(all); mfi.isValid(); ++mfi) {
        auto& fab = all[mfi];
        const long n = long(fab.box().numPts()) * mf.nComp();
        std::copy(buffer.begin(), buffer.begin() + n, fab.dataPtr());
    }

    // Valid region only. Ghost cells keep whatever they held: only the
    // boundary conditions know what belongs there, and silently
    // inventing values would be worse than leaving them stale.
    mf.ParallelCopy(all, 0, 0, mf.nComp());
}

} // namespace fwt
