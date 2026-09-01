#ifndef FWT_OUTPUT_H_
#define FWT_OUTPUT_H_

#include "Grid.H"
#include "Terrain.H"
#include "Inflow.H"
#include "Poisson.H"

#include <string>

namespace fwt {

// Writes the AMReX native plotfile. Components, in order:
//
//   z_cc      -- true (stretched) cell-center height [m]
//   dz        -- true cell thickness [m]
//   terrain_z -- terrain surface height for the column [m]
//   mask      -- 1 solid, 0 fluid (binary immersed boundary)
//   u, v, w    -- velocity after the projection [m/s]
//   u0, v0, w0 -- the initial field, before the correction [m/s]
//   sigma_x/y/z -- Poisson operator coefficients, metric included
//
// z_cc/dz are written because AMReX's Geometry is uniform in z, so the
// plotfile's own vertical coordinate is only nominal and a visualization
// tool needs these fields to see where the cells actually sit.
void WritePlotfile (const std::string& plotfilename,
                    const Grid& grid,
                    const Terrain& terrain,
                    const Inflow& inflow,
                    const Poisson& poisson,
                    const amrex::MultiFab& vel0);

} // namespace fwt

#endif
