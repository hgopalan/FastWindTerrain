#ifndef FWT_FIELDIO_H_
#define FWT_FIELDIO_H_

#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_Vector.H>

namespace fwt {

// -----------------------------------------------------------------------
// Moving a whole field between a distributed MultiFab and one flat host
// buffer.
//
// WHY A COPY RATHER THAN A VIEW. A MultiFab is N separate FArrayBoxes,
// one per box, and every case in this repository runs with
// max_grid_size below the domain width -- the Phase 8 case has four
// boxes. There is no single contiguous array to point at. On top of
// that the velocity carries two ghost layers, so even a one-box field
// is not the shape a caller expects, and a view would hand Python a
// pointer that outlives the MultiFab the moment a Solver goes out of
// scope.
//
// So this copies, and the cost is nothing next to the solve: one field
// of a 24x24x40 case is 184 KB, while the projection it came from ran
// four multigrid solves. Zero-copy starts to matter only for in-situ
// GPU coupling, where the right answer is pyAMReX rather than a
// hand-rolled view -- see docs/python.rst.
//
// LAYOUT. The buffer is AMReX's own: component slowest, then k, then j,
// with i fastest. That is exactly a C-order numpy array of shape
// (ncomp, nz, ny, nx), so each component is a contiguous slab and the
// gather is a memcpy per component rather than a transpose.
//
// It is also channels-first, which is what PyTorch's conv3d wants
// (N, C, D, H, W). A dataset generator can hand the array straight over.
//
// SERIAL. The gather assembles the whole domain on the IO rank, so
// under MPI the returned buffer is empty everywhere else. The bindings
// are serial today; when that changes this is the piece to revisit.
// -----------------------------------------------------------------------

// The valid region of mf, gathered onto the IO rank.
// Shape (nComp, nz, ny, nx) flattened; empty on other ranks.
// Ghost cells are NOT included.
amrex::Vector<amrex::Real> GatherField (const amrex::MultiFab& mf);
amrex::Vector<int>         GatherField (const amrex::iMultiFab& mf);

// The inverse: scatter a buffer of exactly that shape back into the
// valid region of mf. Ghost cells are left alone -- refilling them is
// the caller's business, since only the boundary conditions know what
// they should hold.
void ScatterField (const amrex::Vector<amrex::Real>& buffer,
                   amrex::MultiFab& mf);

// How many values GatherField returns for mf: nComp * the number of
// points in its (possibly nodal) covering box.
long FieldSize (const amrex::MultiFab& mf);
long FieldSize (const amrex::iMultiFab& mf);

} // namespace fwt

#endif
