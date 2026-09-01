===================
Boundary conditions
===================

Each domain face is classified against the flow, and the velocity ghost
cells are filled according to what that face is.

Classification
==============

Lateral faces are classified from the **net outward volumetric flux the
initial field carries through them**, integrated over open (fluid) cells
using the stretched ``dz(k)``:

============  ============  ================================================
Flux          Face type     Velocity treatment
============  ============  ================================================
``< 0``       inflow        Prescribed from the profile
``> 0``       outflow       Zero gradient, free to adjust
``= 0``       tangential    No normal flow either way
============  ============  ================================================

Classifying from the field rather than from ``(u_ref, v_ref)`` matters:
``userfile`` mode has no reference vector at all, and a file may carry a
direction that varies across the domain. For the analytic laws, whose
direction is uniform, it gives exactly the same answer as testing the
wind vector against each face normal.

An oblique wind gives two inflow faces and two outflow faces. An
axis-aligned wind gives one of each, plus two tangential faces. **Zero
inflow faces is a fatal error** -- there would be nowhere for the flow to
enter.

The domain top and the ground are always ``noflow``. Terrain inside the
domain is handled by the immersed-boundary mask rather than by a face
condition.

Tangential faces are open
=========================

A tangential face is treated as **open** (zero gradient), not sealed.

Terrain deflects flow sideways, and being free to do that is the point of
a mass-consistent adjustment over complex terrain. Sealed side walls
would channel the deflected flow back into the domain and show up as a
spurious speed-up around hills. Sealing them would also cost two of the
``lambda``-Dirichlet faces that keep the Poisson operator non-singular,
taking an axis-aligned run from three down to one.

Ghost cell values
=================

Velocity carries one ghost layer, filled per face type:

============  ==================================================================
Face type     Ghost value
============  ==================================================================
inflow        The profile at the ghost cell center, evaluated in AGL against
              the adjacent interior column -- or exactly zero where terrain
              blocks that column
outflow       Copy of the interior cell (zero gradient)
tangential    Copy of the interior cell (open)
noflow        ``u``, ``v`` copied (free slip); ``w`` negated, so ``w``
              averages to zero **on** the face
============  ==================================================================

A blocked inflow cell is worth spelling out: where terrain buries part of
an inflow face, prescribing the profile there would drive flow straight
into the ground, so those ghost cells are shut off instead while the rest
of the face still carries the profile.

Lambda boundary conditions
==========================

The ``lambda`` condition follows from how the velocity is treated:

===========================  ==========================================
Velocity treatment           ``lambda``
===========================  ==========================================
Prescribed (inflow)          Neumann -- the correction must vanish there
Free (outflow, tangential)   Dirichlet
No flow (top, ground)        Neumann
===========================  ==========================================

**At least one face must be Dirichlet.** With every face Neumann the
operator is singular: ``lambda`` is determined only up to a constant, and
an RHS that does not integrate to zero has no solution at all, so the
solve would either fail or return something that quietly does not
conserve mass.

Two guards enforce this. Velocity may be prescribed on **at most two**
lateral faces, and separately the assembled ``lambda`` array is asserted
to hold at least one Dirichlet face. The second check is the one that
matters: making it on the assembled array rather than on the wind
classification means every future route to an all-Neumann problem trips
it, not just this one.

Test aid
========

``bc.dump_file`` writes one row per boundary cell -- face, index, ghost
``u,v,w``, interior ``u,v,w`` -- so a regtest can check every boundary
cell rather than a sample. It is a single-rank test aid, not a production
output path, and is written only when the input is set.
