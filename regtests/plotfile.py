#!/usr/bin/env python3
"""
plotfile.py -- minimal reader for AMReX single-level native plotfiles.

Standard library only. The regtests deliberately carry no third-party
dependency (yt would do this and much more, but pulls in a large stack
for what is, at single level with no ghost cells, a short parse).

What it understands, which is exactly what FastWindTerrain writes:

    <plotfile>/Header            -- ASCII: version, ncomp, variable names,
                                    dim, time, finest_level, prob_lo/hi,
                                    domain box, cell size
    <plotfile>/Level_0/Cell_H    -- ASCII: "FabOnDisk: <file> <offset>"
                                    entries, one per box
    <plotfile>/Level_0/Cell_D_*  -- each FAB is self-describing: an ASCII
                                    header giving real size, byte order,
                                    its box and its component count,
                                    followed by raw binary

Because every FAB carries its own box, the BoxArray section of Cell_H
never has to be parsed -- the FabOnDisk offsets are enough.

Usage:
    from plotfile import Plotfile
    pf = Plotfile("plt_grid")
    mask = pf.field("mask")
    print(mask(3, 4, 5))              # value at cell (3,4,5)
    print(pf.var_names, pf.n_cell)
"""

import array
import os
import re
import sys


# FAB header, e.g.
#   FAB ((8, (64 11 52 0 1 12 0 1023)),(8, (8 7 6 5 4 3 2 1)))((0,0,0) (3,3,3) (0,0,0)) 4
_FAB_RE = re.compile(
    r"FAB\s+\(\((\d+),\s*\(([^)]*)\)\),\s*\((\d+),\s*\(([^)]*)\)\)\)"
    r"\((\([^)]*\))\s+(\([^)]*\))\s+(\([^)]*\))\)\s+(\d+)")

_TUPLE_RE = re.compile(r"-?\d+")


def _ints(s):
    return [int(v) for v in _TUPLE_RE.findall(s)]


class Field:
    """One component of a plotfile, addressable by global cell index."""

    def __init__(self, name, lo, hi):
        self.name = name
        self.lo = tuple(lo)
        self.hi = tuple(hi)
        self.n = tuple(h - l + 1 for l, h in zip(lo, hi))
        self._data = [None] * (self.n[0] * self.n[1] * self.n[2])

    def _index(self, i, j, k):
        i -= self.lo[0]
        j -= self.lo[1]
        k -= self.lo[2]
        if not (0 <= i < self.n[0] and 0 <= j < self.n[1] and 0 <= k < self.n[2]):
            raise IndexError(f"{self.name}: cell out of domain: "
                             f"{(i + self.lo[0], j + self.lo[1], k + self.lo[2])} "
                             f"not in {self.lo}..{self.hi}")
        return (k * self.n[1] + j) * self.n[0] + i

    def __call__(self, i, j, k):
        v = self._data[self._index(i, j, k)]
        if v is None:
            raise ValueError(f"{self.name}: cell ({i},{j},{k}) was never "
                             f"filled -- the plotfile's boxes do not cover "
                             f"the domain")
        return v

    def values(self):
        """Every value, in no particular order. Raises if any cell is unset."""
        if any(v is None for v in self._data):
            missing = sum(1 for v in self._data if v is None)
            raise ValueError(f"{self.name}: {missing} cells were never filled")
        return list(self._data)

    def min(self):
        return min(self.values())

    def max(self):
        return max(self.values())


class Plotfile:
    """A single-level AMReX plotfile."""

    def __init__(self, path):
        self.path = path
        if not os.path.isdir(path):
            raise FileNotFoundError(f"not a plotfile directory: {path}")

        self._read_header()
        self._fields = {}

    # -- Header -----------------------------------------------------------
    def _read_header(self):
        hpath = os.path.join(self.path, "Header")
        with open(hpath) as f:
            lines = [ln.rstrip("\n") for ln in f]

        self.version = lines[0]
        self.ncomp = int(lines[1])
        self.var_names = [lines[2 + i].strip() for i in range(self.ncomp)]

        i = 2 + self.ncomp
        self.dim = int(lines[i]); i += 1
        if self.dim != 3:
            raise ValueError(f"{self.path}: only 3D plotfiles are supported, "
                             f"got dim = {self.dim}")
        self.time = float(lines[i]); i += 1
        self.finest_level = int(lines[i]); i += 1
        self.prob_lo = [float(v) for v in lines[i].split()]; i += 1
        self.prob_hi = [float(v) for v in lines[i].split()]; i += 1
        i += 1                                   # refinement ratios (empty)
        domain = _ints(lines[i]); i += 1         # ((lo) (hi) (type))
        self.domain_lo = domain[0:3]
        self.domain_hi = domain[3:6]
        self.n_cell = [h - l + 1 for l, h in zip(self.domain_lo, self.domain_hi)]
        i += 1                                   # level steps
        self.cell_size = [float(v) for v in lines[i].split()]

        if self.finest_level != 0:
            raise ValueError(f"{self.path}: only single-level plotfiles are "
                             f"supported, finest_level = {self.finest_level}")

    # -- Data -------------------------------------------------------------
    def _fab_locations(self):
        """(file, offset) for every FAB, from Level_0/Cell_H."""
        cell_h = os.path.join(self.path, "Level_0", "Cell_H")
        locs = []
        with open(cell_h) as f:
            for line in f:
                if line.startswith("FabOnDisk:"):
                    _, name, offset = line.split()
                    locs.append((os.path.join(self.path, "Level_0", name),
                                 int(offset)))
        if not locs:
            raise ValueError(f"{cell_h}: no FabOnDisk entries found")
        return locs

    def _read_all(self):
        """Read every FAB and scatter it into one Field per component."""
        fields = [Field(name, self.domain_lo, self.domain_hi)
                  for name in self.var_names]

        for path, offset in self._fab_locations():
            with open(path, "rb") as f:
                f.seek(offset)
                header = b""
                while not header.endswith(b"\n"):
                    ch = f.read(1)
                    if not ch:
                        raise ValueError(f"{path}: FAB header ran off the "
                                         f"end of the file")
                    header += ch

                m = _FAB_RE.match(header.decode("ascii", "replace"))
                if not m:
                    raise ValueError(f"{path}: unrecognized FAB header: "
                                     f"{header[:120]!r}")

                real_size = int(m.group(1))
                order = _ints(m.group(4))
                lo, hi = _ints(m.group(5)), _ints(m.group(6))
                nfabcomp = int(m.group(8))

                if real_size != 8:
                    raise ValueError(f"{path}: only 8-byte reals are "
                                     f"supported, got {real_size}")
                if nfabcomp != self.ncomp:
                    raise ValueError(f"{path}: FAB has {nfabcomp} components "
                                     f"but the plotfile header declares "
                                     f"{self.ncomp}")

                n = [h - l + 1 for l, h in zip(lo, hi)]
                npts = n[0] * n[1] * n[2]

                raw = f.read(npts * nfabcomp * real_size)
                if len(raw) != npts * nfabcomp * real_size:
                    raise ValueError(f"{path}: FAB data is truncated")

                data = array.array("d")
                data.frombytes(raw)
                # Descending order tuple (8 7 ... 1) means little endian.
                file_little = order == sorted(order, reverse=True)
                if file_little != (sys.byteorder == "little"):
                    data.byteswap()

            # Components are stored one whole box after another, each in
            # Fortran order (i fastest).
            for c in range(nfabcomp):
                base = c * npts
                fld = fields[c]
                p = base
                for k in range(lo[2], hi[2] + 1):
                    for j in range(lo[1], hi[1] + 1):
                        for i in range(lo[0], hi[0] + 1):
                            fld._data[fld._index(i, j, k)] = data[p]
                            p += 1

        return {f.name: f for f in fields}

    def field(self, name):
        """The named component. Reads the data on first use."""
        if not self._fields:
            self._fields = self._read_all()
        if name not in self._fields:
            raise KeyError(f"{self.path}: no component '{name}' "
                           f"(have {self.var_names})")
        return self._fields[name]

    def cell_center(self, i, j, k):
        """Nominal cell center. NOTE: AMReX's Geometry is uniform in z, so
        the z returned here is nominal -- use the z_cc field for the true
        stretched height."""
        return tuple(self.prob_lo[d] + (idx + 0.5) * self.cell_size[d]
                     for d, idx in enumerate((i, j, k)))


def main():
    """Dump a summary; handy when a checker fails and you want a look."""
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <plotfile>")
        return 1
    pf = Plotfile(sys.argv[1])
    print(f"{pf.path}: {pf.ncomp} components, n_cell = {pf.n_cell}")
    print(f"  prob_lo = {pf.prob_lo}, prob_hi = {pf.prob_hi}")
    for name in pf.var_names:
        f = pf.field(name)
        print(f"  {name:12s} min {f.min():14.6f}  max {f.max():14.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
