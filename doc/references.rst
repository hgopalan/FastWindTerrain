==========
References
==========

The methods this solver implements, and where each is used.

Mass-consistent adjustment
==========================

**Sherman, C. A. (1978).** A mass-consistent model for wind fields over
complex terrain. *Journal of Applied Meteorology*, 17(3), 312-319.

  The variational formulation the solver rests on: adjust an initial wind
  field as little as possible, in a weighted least-squares sense, subject
  to being divergence free. The Lagrange multiplier of that constraint is
  ``lambda``, and the weights are ``alpha_h`` and ``alpha_v``. See
  :doc:`poisson`.

**O'Brien, J. J. (1970).** Alternative solutions to the classical
vertical velocity problem. *Journal of Applied Meteorology*, 9(2),
197-203.

  The vertical-velocity adjustment: integrate continuity up a column,
  then redistribute the residual with a height-weighted correction so the
  vertical velocity satisfies its upper boundary condition exactly. See
  :doc:`anisotropy`.

**Mason, P. J., & King, J. C. (1985).** Measurements and predictions of
flow over a succession of large hills. *Boundary-Layer Meteorology*,
32(4), 339-359.

  Terrain-following coordinates with metric terms. Not implemented here;
  noted because ``massconsistent_amr`` offers it as an option and it is a
  natural extension.

Numerics
========

**Jiang, G.-S., & Shu, C.-W. (1996).** Efficient implementation of
weighted ENO schemes. *Journal of Computational Physics*, 126(1),
202-228.

  The smoothness indicators and nonlinear weights of the WENO scheme.
  Their behaviour at critical points is why the convergence study reports
  an L2 norm alongside L-infinity. See :doc:`numerics`.

**Almgren, A. S., Bell, J. B., & Szymczak, W. G. (1996).** A numerical
method for the incompressible Navier-Stokes equations based on an
approximate projection. *SIAM Journal on Scientific Computing*, 17(2),
358-369.

  Approximate projections: the divergence and gradient need not be an
  exact factorisation of the operator being inverted, at the cost of a
  residual divergence. This is what the nodal projection here is, and why
  repeating it helps. See :doc:`poisson`.

Software
========

**AMReX.** https://amrex-codes.github.io/amrex/

  The mesh, the linear solvers, and the plotfile format. Bundled as a
  submodule, pinned to release 26.08. The nodal Poisson solve uses
  ``MLNodeLaplacian`` with ``MLMG``.

**massconsistent_amr.** https://hgopalan.github.io/massconsistent_amr/

  The source project. This solver ports its mass-consistent core, and
  matches it deliberately in several places: the terrain and velocity
  file formats, the inverse-distance interpolation, the immersed-boundary
  treatment, the cell-local anisotropy formulation, the O'Brien
  adjustment, and the convention of fixing the ``lambda`` boundary
  conditions rather than deriving them from the wind.

  Where this solver differs, it is deliberate and documented: ``lambda``
  is nodal here rather than cell-centered, so that a future
  fractional-step solver can build on the same layout, and the vertical
  grid is stretched, which requires the metric to be carried in the
  operator coefficients.
