"""
Output in memory, and where the files go (was
regtests/phase15_output_interop).

``fields()`` hands back the SAME object the plotfile and ascii backends
are given -- one gather, three consumers. The cross-check below is what
holds that claim up: a dataset generator gets exactly the array the file
would have contained, so nothing has to be written and read back to know
what a case produced.

Anything less than exact equality here would mean a third assembly of
"the output fields" has crept in, which is the drift this codebase has
spent fifteen phases designing against.
"""

import os

import numpy as np
import pytest

import fastwindterrain as fwt
from conftest import read_ascii

NX, NY, NZ = 24, 24, 40
N_FIELDS = 17


@pytest.fixture
def grid_params():
    """Override: the phase 15 case is 24x24x40, not the default 8x8x40."""
    return {"n_cell": (NX, NY, NZ), "prob_lo": (0.0, 0.0, 0.0),
            "prob_hi": (1000.0, 1000.0, 483.19909696997223),
            "dz0": 4.0, "stretching_ratio": 1.05, "max_grid_size": 16}


@pytest.fixture(scope="module")
def written(request, tmp_path_factory):
    """One solved case, written three ways: in memory, ascii, plotfile.

    Module-scoped because the solve is the expensive part and every
    comparison below reads the same three renderings of it.
    """
    # Fixtures cannot be requested across scopes, so the case is built
    # here rather than taken from the function-scoped `case` factory.
    from conftest import TERRAIN_CSV
    amrex = request.getfixturevalue("amrex")
    del amrex
    pts = np.loadtxt(TERRAIN_CSV, delimiter=",", comments="#", skiprows=5)
    out = tmp_path_factory.mktemp("output")

    s = fwt.Solver({
        "grid": {"n_cell": (NX, NY, NZ), "prob_lo": (0.0, 0.0, 0.0),
                 "prob_hi": (1000.0, 1000.0, 483.19909696997223),
                 "dz0": 4.0, "stretching_ratio": 1.05, "max_grid_size": 16},
        "terrain": {"points": pts},
        "inflow": {"mode": "powerlaw", "u_ref": 8.0, "v_ref": 6.0},
        "anisotropy": {"enable": True},
        "obrien": {"enable": True},
        "poisson": {"alpha_v": 0.5, "n_projections": 4},
    })
    s.setup()
    s.solve()
    s.diagnose()
    s.write_ascii(str(out / "fields_mem.txt"))
    s.write_plotfile(str(out / "plt_mem"))
    return s, s.fields(), out


# ---------------------------------------------------------------------------
# In memory vs the written files
# ---------------------------------------------------------------------------

def test_fields_returns_every_output_field(written):
    _, mem, _ = written
    assert len(mem) == N_FIELDS
    assert all(a.shape == (NZ, NY, NX) for a in mem.values())


@pytest.mark.slow
def test_in_memory_matches_the_plotfile(written, Plotfile):
    _, mem, out = written
    pf = Plotfile(str(out / "plt_mem"))
    assert sorted(pf.var_names) == sorted(mem), (
        "fields() and the plotfile carry different field names")

    for name in sorted(mem):
        fld = pf.field(name)
        a = mem[name]
        for k in range(0, NZ, 3):
            for j in range(0, NY, 4):
                for i in range(0, NX, 4):
                    assert float(a[k, j, i]) == float(fld(i, j, k)), (
                        f"{name} differs from the plotfile at ({i},{j},{k}); "
                        f"both come from one gather, so this means a third "
                        f"assembly has crept in")


@pytest.mark.slow
def test_in_memory_matches_the_ascii_file(written):
    _, mem, out = written
    asc = read_ascii(str(out / "fields_mem.txt"))
    assert len(asc) == N_FIELDS

    for name, table in asc.items():
        assert name in mem, (
            f"the ascii file has a field {name!r} that fields() does not")
        a = mem[name]
        for (i, j, k), want in table.items():
            assert float(a[k, j, i]) == want, (
                f"{name} differs from the ascii file at ({i},{j},{k})")


# ---------------------------------------------------------------------------
# The writers put files where they are told
# ---------------------------------------------------------------------------

def test_explicit_writers_use_the_names_given(written, tmp_path):
    s, _, _ = written
    s.write_report(str(tmp_path / "named_report.txt"))
    s.write_ascii(str(tmp_path / "named_fields.txt"))
    s.write_plotfile(str(tmp_path / "named_plt"))

    assert (tmp_path / "named_report.txt").is_file()
    assert (tmp_path / "named_fields.txt").is_file()
    assert (tmp_path / "named_plt").is_dir()

    # A real report, not an empty file with the right name.
    text = (tmp_path / "named_report.txt").read_text()
    assert "diag_div_max" in text
    assert "z_face" in text


def test_writing_before_diagnose_raises_and_creates_nothing(amrex, case,
                                                            tmp_path):
    """Saying so beats writing a file with a missing component."""
    s = fwt.Solver(case())
    s.setup()
    with pytest.raises(RuntimeError):
        s.write_plotfile(str(tmp_path / "too_early"))
    assert not (tmp_path / "too_early").exists()


# ---------------------------------------------------------------------------
# The output section of the config
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_output_section_selects_what_is_written(solved, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = solved(output={"which": "both", "format": "ascii",
                       "report_file": "cfg_report.txt",
                       "ascii_file": "cfg_fields.txt",
                       "plot_file": "cfg_plt"})
    s.write_output()

    assert os.path.isfile("cfg_report.txt")
    assert os.path.isfile("cfg_fields.txt")
    # format = ascii, so no plotfile even though which = both.
    assert not os.path.exists("cfg_plt")


BAD_OUTPUT = {
    "unknown which": {"which": "everything"},
    "unknown format": {"format": "vtk"},
    "misspelled key": {"report_fil": "x"},
}


@pytest.mark.parametrize("output", BAD_OUTPUT.values(), ids=list(BAD_OUTPUT))
def test_bad_output_config_raises_at_configuration_time(amrex, case, output):
    """At configuration time, not after a solve has already run."""
    with pytest.raises(ValueError):
        fwt.Solver(case(output=output))


# ---------------------------------------------------------------------------
# No ParmParse leak into where the output goes
# ---------------------------------------------------------------------------

def test_output_paths_ignore_parmparse(run_py, tmp_path, terrain_points):
    """Case 2 of a generation loop must not overwrite case 1's output
    because an inputs file left grid.plot_file pointing somewhere."""
    inputs = tmp_path / "inputs_leak"
    inputs.write_text(
        "grid.n_cell = 8 8 8\n"
        "grid.prob_lo = 0.0 0.0 0.0\n"
        "grid.prob_hi = 100.0 100.0 32.0\n"
        "grid.dz0 = 4.0\n"
        "inflow.u_ref = 1.0\n"
        "grid.output_format = both\n"
        "grid.report_file = leaked_report.txt\n"
        "grid.plot_file = leaked_plt\n"
        "output.format = both\n"
        "output.ascii_file = leaked_fields.txt\n")

    from conftest import TERRAIN_CSV
    r = run_py(f"""
import os
import numpy as np
import fastwindterrain as fwt

pts = np.loadtxt({str(TERRAIN_CSV)!r}, delimiter=",", comments="#", skiprows=5)
fwt.initialize([{str(inputs)!r}])
s = fwt.Solver({{
    "grid": {{"n_cell": (12, 12, 40), "prob_lo": (0.0, 0.0, 0.0),
              "prob_hi": (1000.0, 1000.0, 483.19909696997223),
              "dz0": 4.0, "stretching_ratio": 1.05, "max_grid_size": 16}},
    "terrain": {{"points": pts}},
    "inflow": {{"mode": "powerlaw", "u_ref": 8.0, "v_ref": 6.0}},
    "poisson": {{"alpha_v": 0.5, "n_projections": 1}},
    "output": {{"which": "report", "report_file": "dict_report.txt"}},
}})
s.setup(); s.solve(); s.diagnose()
s.write_output()
print("::DICT", os.path.isfile("dict_report.txt"))
print("::LEAK_REPORT", os.path.exists("leaked_report.txt"))
print("::LEAK_ASCII", os.path.exists("leaked_fields.txt"))
print("::LEAK_PLT", os.path.exists("leaked_plt"))
fwt.finalize()
""")

    assert r["DICT"] == "True", "the dict's report file was not written"
    for key in ("LEAK_REPORT", "LEAK_ASCII", "LEAK_PLT"):
        assert r[key] == "False", (
            f"{key}: the inputs file's output settings reached a "
            f"dict-configured run. In a generation loop that is case 2 "
            f"overwriting case 1.")
