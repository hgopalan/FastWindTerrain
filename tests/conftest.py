"""
Shared fixtures for the bindings test suite.

WHY THESE TESTS ARE PYTEST AND THE PHASE 1-8 GROUPS ARE NOT. Those groups
test the executable: they run a binary on an inputs file and read its
output files back. ``run_regtests.py`` is the right driver for that, and
they stay there. These test a Python API, where pytest's fixtures,
parametrization and assertion rewriting are worth having -- and where the
old checkers had to shell out to a subprocess per case and parse ``::KEY
value`` lines back out of stdout to say anything at all.

ONE AMReX PER PROCESS. ``amrex::Initialize``/``Finalize`` are
process-global and cannot be repeated, so the ``amrex`` fixture is
session-scoped: it initializes once for the whole run and finalizes at
the end. Consequences, both of which the suite depends on:

* A test that takes ``amrex`` must NOT open ``fwt.session()`` -- that is
  a second initialize, and it raises.
* Anything needing its own AMReX lifecycle (initializing from an inputs
  file, testing initialize/finalize ordering, watching a warning raised
  during construction under a fresh ParmParse) runs in a subprocess via
  the ``run_py`` fixture.

That every other test shares one initialization is not merely
convenient. Hundreds of independent cases inside one AMReX is exactly
what dataset generation does, so running the suite this way exercises the
property the bindings exist to provide.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
BUILD_PY = REPO / "build" / "python"

# An in-tree build is importable from build/python; an installed wheel is
# importable already. Preferring the installed one means `pip install .`
# followed by pytest tests the wheel, which is what the packaging job
# wants; falling back to build/python means a developer who has only run
# CMake needs no install step.
try:
    import fastwindterrain as fwt
except ImportError:
    if BUILD_PY.is_dir():
        sys.path.insert(0, str(BUILD_PY))
    try:
        import fastwindterrain as fwt
    except ImportError as exc:                              # pragma: no cover
        raise ImportError(
            f"cannot import fastwindterrain ({exc}).\n"
            f"Either install it:\n"
            f"    pip install .\n"
            f"or build it in tree:\n"
            f"    cmake -S . -B build -DFWT_PYTHON=ON && cmake --build build\n"
            f"If build/python exists and this still fails, the module was "
            f"built for a different Python than the {sys.version_info.major}."
            f"{sys.version_info.minor} running pytest -- an extension module "
            f"is tied to one minor version. build/fastwindterrain-py records "
            f"the interpreter CMake used; run pytest with that one."
        ) from exc

from fastwindterrain import dataset                    # noqa: E402

TERRAIN_CSV = REPO / "regtests" / "phase8_diagnostics_output" / "terrain_hill.csv"

#: A stretched column whose height is quoted exactly, so it neither
#: overshoots nor undershoots: 40 cells from 4 m at ratio 1.05.
STRETCHED_TOP = 483.19909696997223


def pytest_addoption(parser):
    parser.addoption(
        "--fwt-exe", action="store", default=None,
        help="Path to the solver executable, for the tests that compare the "
             "Python path against it. Defaults to build/fastwindterrain.")


def pytest_report_header(config):
    return [f"fastwindterrain {fwt.__version__} (AMReX {fwt.amrex_version()})",
            f"module: {fwt.__file__}"]


# ---------------------------------------------------------------------------
# The AMReX lifecycle
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def amrex():
    """AMReX, initialized once for the whole session.

    Takes no inputs file, so ParmParse starts empty and stays that way:
    every test here configures its case with a dict. A test that needs
    ParmParse populated wants ``run_py``.
    """
    fwt.initialize([])
    try:
        yield fwt
    finally:
        fwt.finalize()


# ---------------------------------------------------------------------------
# Out-of-process cases
# ---------------------------------------------------------------------------

class PyResult:
    """A finished subprocess, with the ``::KEY value`` lines parsed out."""

    def __init__(self, proc):
        self.proc = proc
        self.returncode = proc.returncode
        self.stdout = proc.stdout
        self.stderr = proc.stderr
        self.marks = {}
        for line in proc.stdout.splitlines():
            if line.startswith("::"):
                k, _, v = line[2:].partition(" ")
                self.marks[k] = v

    def __getitem__(self, key):
        assert key in self.marks, (
            f"the subprocess printed no ::{key} line.\n"
            f"--- stdout ---\n{self.stdout[-3000:]}\n"
            f"--- stderr ---\n{self.stderr[-3000:]}")
        return self.marks[key]

    def ok(self):
        assert self.returncode == 0, (
            f"the subprocess exited {self.returncode}\n"
            f"--- stdout ---\n{self.stdout[-3000:]}\n"
            f"--- stderr ---\n{self.stderr[-3000:]}")
        return self


@pytest.fixture
def run_py(tmp_path):
    """Run a snippet in a fresh interpreter, and read its ``::KEY`` lines.

    For the cases that cannot share the session's AMReX: initializing
    from an inputs file, initialize/finalize ordering, and anything that
    must abort. Runs in ``tmp_path``, so a case that writes files leaves
    them there and not in the source tree.
    """
    def _run(code, args=(), cwd=None, expect_ok=True, timeout=1800):
        env = dict(os.environ)
        # The parent found the module somehow -- installed, or via the
        # build/python this conftest prepended. Handing the child the
        # parent's sys.path means the two always agree about which build
        # is under test.
        env["PYTHONPATH"] = os.pathsep.join(
            [p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(
            os.pathsep)
        proc = subprocess.run(
            [sys.executable, "-c", code, *[str(a) for a in args]],
            capture_output=True, text=True, timeout=timeout,
            env=env, cwd=str(cwd or tmp_path))
        result = PyResult(proc)
        return result.ok() if expect_ok else result

    return _run


# ---------------------------------------------------------------------------
# The executable, for the parity tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def solver_exe(request):
    """The solver binary, or a skip.

    Only the tests that assert Python and C++ agree need this. A wheel
    installed somewhere else has no executable next to it, and skipping
    those few tests is correct -- the parity guarantee is a property of a
    build tree, not of a wheel (see pyproject.toml).
    """
    given = request.config.getoption("--fwt-exe")
    exe = Path(given) if given else REPO / "build" / "fastwindterrain"
    if not exe.is_file():
        pytest.skip(f"solver executable not found at {exe} "
                    f"(build it, or pass --fwt-exe)")
    return exe


@pytest.fixture
def run_exe(solver_exe, tmp_path):
    """Run the solver executable on an inputs file, in ``tmp_path``."""
    def _run(inputs, args=(), cwd=None, expect_ok=True):
        proc = subprocess.run(
            [str(solver_exe), str(inputs), *[str(a) for a in args]],
            capture_output=True, text=True, timeout=3600,
            cwd=str(cwd or tmp_path))
        if expect_ok:
            assert proc.returncode == 0, (
                f"the executable exited {proc.returncode}\n"
                f"{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
        return proc

    return _run


@pytest.fixture(scope="session")
def Plotfile():
    """The plotfile reader the C++ regtests use.

    Shared rather than reimplemented: a second reader that agreed with
    the bindings and disagreed with AMReX would prove nothing.
    """
    sys.path.insert(0, str(REPO / "regtests"))
    try:
        from plotfile import Plotfile as _Plotfile
    except ImportError:                                 # pragma: no cover
        pytest.skip("regtests/plotfile.py is not available here")
    return _Plotfile


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def terrain_points():
    """The hill point cloud the phase 8 group uses, as an (N, 3) array.

    Shared with the C++ regtests on purpose: a Python-built case and a
    file-driven one are then the same case, which is what makes the
    parity assertions meaningful.
    """
    import numpy as np
    if not TERRAIN_CSV.is_file():                       # pragma: no cover
        pytest.skip(f"terrain data not found at {TERRAIN_CSV}")
    return np.loadtxt(TERRAIN_CSV, delimiter=",", comments="#", skiprows=5)


@pytest.fixture
def grid_params():
    """A stretched grid that fits its requested height exactly."""
    return {
        "n_cell": (8, 8, 40),
        "prob_lo": (0.0, 0.0, 0.0),
        "prob_hi": (1000.0, 1000.0, STRETCHED_TOP),
        "dz0": 4.0,
        "stretching_ratio": 1.05,
        "max_grid_size": 16,
    }


@pytest.fixture
def case(grid_params, terrain_points):
    """A factory for a complete solver config, hill terrain included.

    ``case()`` is the reference case; ``case(poisson={"alpha_v": 0.3})``
    replaces a whole section, and ``case(poisson__alpha_v=0.3)`` sets one
    key inside one, leaving the rest of that section alone.
    """
    def _case(**kw):
        cfg = {
            "grid": dict(grid_params),
            "terrain": {"points": terrain_points},
            "inflow": {"mode": "powerlaw", "u_ref": 8.0, "v_ref": 6.0},
            "anisotropy": {"enable": True},
            "obrien": {"enable": True},
            "poisson": {"alpha_v": 0.5, "n_projections": 4},
        }
        for k, v in kw.items():
            if "__" in k:
                section, _, key = k.partition("__")
                cfg.setdefault(section, {})[key] = v
            else:
                cfg[k] = v
        return cfg

    return _case


@pytest.fixture
def solved(amrex, case):
    """A factory for a fully solved case: setup, solve, diagnose.

    Takes the same arguments as ``case``.
    """
    def _solved(**kw):
        s = fwt.Solver(case(**kw))
        s.setup()
        s.solve()
        s.diagnose()
        return s

    return _solved


# ---------------------------------------------------------------------------
# Small helpers the test modules share
# ---------------------------------------------------------------------------

def read_ascii(path):
    """The gathered plain-text field file, as ``{name: {(i, j, k): value}}``.

    The header line names the columns; the first five are i, j, k and the
    cell centre, and the rest are the fields.
    """
    cols = None
    rows = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                body = line[1:].strip()
                if body.startswith("columns:"):
                    cols = body[len("columns:"):].split()
                continue
            if not line.strip():
                continue
            p = line.split()
            key = (int(p[0]), int(p[1]), int(p[2]))
            for name, v in zip(cols[5:], p[5:]):
                rows.setdefault(name, {})[key] = float(v)
    return rows
