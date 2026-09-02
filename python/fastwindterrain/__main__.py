"""
Run one case from the command line, argv-compatible with the solver
executable::

    python -m fastwindterrain inputs [name=value ...]

Identical in behaviour to::

    ./build/fastwindterrain inputs [name=value ...]

which is what lets the existing regtest driver run the whole suite
through the bindings without any checker knowing the difference.
"""

import sys

from . import run


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: python -m fastwindterrain <inputs> [name=value ...]",
              file=sys.stderr)
        return 2
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
