#!/usr/bin/env python3
"""
make_terrain.py -- generate synthetic terrain files for FastWindTerrain.

Writes the same x,y,z point format the solver reads (see the Phase 2
terrain reader, and read_terrain_file in massconsistent_amr): one point
per line, comma separated, '#' comments ignored, header line optional.

The shape functions are plain, importable, side-effect-free callables:

    from make_terrain import elevation
    z = elevation("hill", x, y, peak=100.0, sigma=150.0, ...)

so a regtest checker can compute the expected z_terrain analytically
instead of trusting the same code path the solver used. Standard library
only -- the regtests deliberately carry no third-party dependencies.

Examples:
    # Single Gaussian hill over a 1000 x 1000 m footprint
    python3 tools/make_terrain.py --shape hill --peak 100 --sigma 150 \\
        --xhi 1000 --yhi 1000 --nx 51 --ny 51 -o terrain.csv

    # Flat ground (the Phase 2 control case)
    python3 tools/make_terrain.py --shape flat -o terrain_flat.csv

    # Scattered (non-lattice) points, to exercise the IDW interpolation
    python3 tools/make_terrain.py --shape hill --jitter 0.4 --seed 1
"""

import argparse
import math
import random
import sys


# ---------------------------------------------------------------------------
# Shape functions: z = f(x, y). Each takes the full parameter dict and
# ignores what it does not use, so the CLI can pass one dict to any shape.
# ---------------------------------------------------------------------------

def _flat(x, y, p):
    """Constant elevation. The Phase 2 control case: z_terrain == base."""
    return p["base"]


def _hill(x, y, p):
    """Single Gaussian hill:

        z = base + peak * exp(-r^2 / (2 sigma^2)),  r = dist to (xc, yc)
    """
    r2 = (x - p["xc"])**2 + (y - p["yc"])**2
    return p["base"] + p["peak"] * math.exp(-r2 / (2.0 * p["sigma"]**2))


def _valley(x, y, p):
    """Gaussian valley -- the hill inverted, floored at the base minus
    peak so the terrain never goes below `base - peak`."""
    r2 = (x - p["xc"])**2 + (y - p["yc"])**2
    return p["base"] - p["peak"] * math.exp(-r2 / (2.0 * p["sigma"]**2))


def _ridge(x, y, p):
    """Gaussian ridge running along y: a hill in x, uniform in y.
    Useful as a 2D case where the answer is independent of j."""
    dx2 = (x - p["xc"])**2
    return p["base"] + p["peak"] * math.exp(-dx2 / (2.0 * p["sigma"]**2))


def _slope(x, y, p):
    """Constant-gradient plane:

        z = base + slope_x * (x - xlo) + slope_y * (y - ylo)

    The Phase 7 anisotropy/O'Brien case wants a terrain with a known,
    constant slope everywhere."""
    return (p["base"]
            + p["slope_x"] * (x - p["xlo"])
            + p["slope_y"] * (y - p["ylo"]))


SHAPES = {
    "flat":   _flat,
    "hill":   _hill,
    "valley": _valley,
    "ridge":  _ridge,
    "slope":  _slope,
}


def elevation(shape, x, y, **params):
    """Analytic terrain elevation at (x, y) for the named shape.

    Keyword parameters default to the same values the CLI defaults to,
    so a checker can call elevation("hill", x, y, peak=100, sigma=150,
    xc=500, yc=500) and get exactly what the CLI wrote."""
    if shape not in SHAPES:
        raise ValueError(f"unknown shape '{shape}' "
                         f"(expected one of {sorted(SHAPES)})")
    p = dict(DEFAULT_PARAMS)
    p.update(params)
    return SHAPES[shape](x, y, p)


DEFAULT_PARAMS = {
    "base": 0.0,
    "peak": 100.0,
    "sigma": 150.0,
    "xc": 500.0,
    "yc": 500.0,
    "xlo": 0.0,
    "ylo": 0.0,
    "slope_x": 0.1,
    "slope_y": 0.0,
}


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class TerrainGenerator:
    """Samples a shape function on a regular nx x ny lattice spanning
    [xlo, xhi] x [ylo, yhi], optionally jittering the sample locations so
    the output is genuinely scattered rather than gridded."""

    def __init__(self, shape="hill", nx=51, ny=51,
                 xlo=0.0, xhi=1000.0, ylo=0.0, yhi=1000.0,
                 jitter=0.0, seed=0, **params):
        if shape not in SHAPES:
            raise ValueError(f"unknown shape '{shape}' "
                             f"(expected one of {sorted(SHAPES)})")
        if nx < 2 or ny < 2:
            raise ValueError("nx and ny must both be >= 2")
        if xhi <= xlo or yhi <= ylo:
            raise ValueError("require xhi > xlo and yhi > ylo")
        if not 0.0 <= jitter < 0.5:
            raise ValueError("jitter must be in [0, 0.5) cell widths")

        self.shape = shape
        self.nx, self.ny = nx, ny
        self.xlo, self.xhi = xlo, xhi
        self.ylo, self.yhi = ylo, yhi
        self.jitter = jitter
        self.seed = seed

        self.dx = (xhi - xlo) / (nx - 1)
        self.dy = (yhi - ylo) / (ny - 1)

        # Shape parameters, defaulted then overridden. xc/yc default to
        # the domain center rather than to a fixed coordinate.
        self.params = dict(DEFAULT_PARAMS)
        self.params.update({
            "xc": 0.5 * (xlo + xhi),
            "yc": 0.5 * (ylo + yhi),
            "xlo": xlo,
            "ylo": ylo,
        })
        self.params.update({k: v for k, v in params.items() if v is not None})

        self.points = []

    def generate(self):
        """Return (and cache) the list of (x, y, z) points."""
        rng = random.Random(self.seed)
        f = SHAPES[self.shape]
        self.points = []

        for j in range(self.ny):
            for i in range(self.nx):
                x = self.xlo + i * self.dx
                y = self.ylo + j * self.dy

                if self.jitter > 0.0:
                    # Keep the perturbed point inside the domain: the
                    # corners must still bound the interpolation region.
                    x += rng.uniform(-self.jitter, self.jitter) * self.dx
                    y += rng.uniform(-self.jitter, self.jitter) * self.dy
                    x = min(max(x, self.xlo), self.xhi)
                    y = min(max(y, self.ylo), self.yhi)

                self.points.append((x, y, f(x, y, self.params)))

        return self.points

    def write_csv(self, filename, decimals=10):
        """Write the point list in the solver's terrain format."""
        if not self.points:
            self.generate()

        fmt = f"{{:.{decimals}f}}"
        with open(filename, "w") as f:
            f.write(f"# FastWindTerrain synthetic terrain -- shape = {self.shape}\n")
            f.write(f"# generated by tools/make_terrain.py; "
                    f"do not edit by hand\n")
            f.write(f"# domain: [{self.xlo}, {self.xhi}] x "
                    f"[{self.ylo}, {self.yhi}] m, "
                    f"{self.nx} x {self.ny} points, "
                    f"dx = {self.dx} m, dy = {self.dy} m\n")
            f.write(f"# params: "
                    + ", ".join(f"{k}={self.params[k]}"
                                for k in sorted(self.params))
                    + "\n")
            if self.jitter > 0.0:
                f.write(f"# scattered: jitter = {self.jitter} cells, "
                        f"seed = {self.seed}\n")
            f.write("x,y,z\n")
            for x, y, z in self.points:
                f.write(f"{fmt.format(x)},{fmt.format(y)},{fmt.format(z)}\n")

    def stats(self):
        if not self.points:
            self.generate()
        zs = [z for _, _, z in self.points]
        return {
            "shape": self.shape,
            "n_points": len(self.points),
            "nx": self.nx, "ny": self.ny,
            "dx": self.dx, "dy": self.dy,
            "x_range": (self.xlo, self.xhi),
            "y_range": (self.ylo, self.yhi),
            "z_min": min(zs), "z_max": max(zs),
            "z_mean": sum(zs) / len(zs),
        }

    def print_stats(self, stream=sys.stderr):
        s = self.stats()
        print(f"terrain: shape = {s['shape']}, {s['n_points']} points "
              f"({s['nx']} x {s['ny']})", file=stream)
        print(f"  domain    : x [{s['x_range'][0]}, {s['x_range'][1]}] m, "
              f"y [{s['y_range'][0]}, {s['y_range'][1]}] m", file=stream)
        print(f"  spacing   : dx = {s['dx']:.3f} m, dy = {s['dy']:.3f} m",
              file=stream)
        print(f"  elevation : min {s['z_min']:.3f} m, max {s['z_max']:.3f} m, "
              f"mean {s['z_mean']:.3f} m", file=stream)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate synthetic terrain for FastWindTerrain.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument("-o", "--output", default="terrain.csv",
                    help="output file")
    ap.add_argument("--shape", default="hill", choices=sorted(SHAPES),
                    help="terrain shape")

    ap.add_argument("--nx", type=int, default=51, help="sample points in x")
    ap.add_argument("--ny", type=int, default=51, help="sample points in y")
    ap.add_argument("--xlo", type=float, default=0.0, help="domain x min [m]")
    ap.add_argument("--xhi", type=float, default=1000.0, help="domain x max [m]")
    ap.add_argument("--ylo", type=float, default=0.0, help="domain y min [m]")
    ap.add_argument("--yhi", type=float, default=1000.0, help="domain y max [m]")

    ap.add_argument("--base", type=float, default=0.0,
                    help="base elevation [m]")
    ap.add_argument("--peak", type=float, default=100.0,
                    help="hill/valley/ridge amplitude [m]")
    ap.add_argument("--sigma", type=float, default=150.0,
                    help="hill/valley/ridge width [m]")
    ap.add_argument("--center-x", type=float, default=None, dest="xc",
                    help="feature center x [m] (default: domain center)")
    ap.add_argument("--center-y", type=float, default=None, dest="yc",
                    help="feature center y [m] (default: domain center)")
    ap.add_argument("--slope-x", type=float, default=0.1, dest="slope_x",
                    help="dz/dx for --shape slope")
    ap.add_argument("--slope-y", type=float, default=0.0, dest="slope_y",
                    help="dz/dy for --shape slope")

    ap.add_argument("--jitter", type=float, default=0.0,
                    help="randomly displace each sample by up to this many "
                         "cell widths, in [0, 0.5), to produce scattered "
                         "rather than gridded points")
    ap.add_argument("--seed", type=int, default=0,
                    help="random seed used by --jitter")
    ap.add_argument("--decimals", type=int, default=10,
                    help="digits written after the decimal point. The "
                         "default keeps the write error well under the "
                         "1e-6 m tolerance the regtest checkers use")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="do not print the summary to stderr")

    args = ap.parse_args()

    try:
        gen = TerrainGenerator(
            shape=args.shape, nx=args.nx, ny=args.ny,
            xlo=args.xlo, xhi=args.xhi, ylo=args.ylo, yhi=args.yhi,
            jitter=args.jitter, seed=args.seed,
            base=args.base, peak=args.peak, sigma=args.sigma,
            xc=args.xc, yc=args.yc,
            slope_x=args.slope_x, slope_y=args.slope_y)
        gen.generate()
        gen.write_csv(args.output, decimals=args.decimals)
    except (ValueError, OSError) as e:
        print(f"make_terrain: {e}", file=sys.stderr)
        return 1

    if not args.quiet:
        gen.print_stats()
        print(f"  wrote     : {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
