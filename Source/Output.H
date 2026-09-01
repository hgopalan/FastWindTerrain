#ifndef FWT_OUTPUT_H_
#define FWT_OUTPUT_H_

#include "Grid.H"
#include "Terrain.H"

#include <string>

namespace fwt {

// Writes the AMReX native plotfile. Components, in order:
//
//   z_cc      -- true (stretched) cell-center height [m]
//   dz        -- true cell thickness [m]
//   terrain_z -- terrain surface height for the column [m]
//   mask      -- 1 solid, 0 fluid (binary immersed boundary)
//
// z_cc/dz are written because AMReX's Geometry is uniform in z, so the
// plotfile's own vertical coordinate is only nominal and a visualization
// tool needs these fields to see where the cells actually sit.
void WritePlotfile (const std::string& plotfilename,
                    const Grid& grid,
                    const Terrain& terrain);

} // namespace fwt

#endif
