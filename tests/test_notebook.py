"""
The example notebook, executed.

A notebook in a repository rots the moment an API changes, and the rot is
invisible: the committed outputs still look right. So this runs it for
real, top to bottom, and fails on the first cell that raises.

It is skipped when nbclient, ipykernel or matplotlib are missing --
running the suite should not require a Jupyter stack. CI installs them
(`pip install ".[notebook,test]"`), so the notebook is executed there on
every pull request.
"""

import os
import shutil
import sys

import pytest

from conftest import REPO

NOTEBOOKS = sorted((REPO / "examples").glob("*.ipynb")) \
    if (REPO / "examples").is_dir() else []

nbformat = pytest.importorskip("nbformat", reason="nbformat is not installed")
nbclient = pytest.importorskip("nbclient", reason="nbclient is not installed")


@pytest.mark.slow
@pytest.mark.skipif(not NOTEBOOKS, reason="no notebooks in examples/")
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_runs(path, monkeypatch):
    pytest.importorskip("matplotlib",
                        reason="the notebook plots; matplotlib is not "
                               "installed")
    if shutil.which("jupyter") is None and "ipykernel" not in sys.modules:
        pytest.importorskip("ipykernel", reason="no kernel to run on")

    # The kernel is a fresh interpreter with its own environment, and it
    # runs in examples/ -- so a relative PYTHONPATH pointing at
    # build/python would not resolve. Hand it the absolute paths this
    # process actually imported from.
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(p for p in sys.path if p))

    nb = nbformat.read(str(path), as_version=4)
    client = nbclient.NotebookClient(
        nb, timeout=1800, kernel_name="python3",
        # Run in the notebook's own directory, the way a reader would.
        resources={"metadata": {"path": str(path.parent)}},
        # Matplotlib must not try to open a window on a headless runner.
        extra_arguments=["--InlineBackend.figure_format=png"],
    )
    client.execute()


@pytest.mark.skipif(not NOTEBOOKS, reason="no notebooks in examples/")
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_committed_without_outputs(path):
    """Outputs in a committed notebook make every diff a binary blob and
    invite the reader to trust a result nobody re-ran. The test above is
    what proves the code still works."""
    nb = nbformat.read(str(path), as_version=4)
    dirty = [i for i, c in enumerate(nb.cells)
             if c.cell_type == "code" and (c.get("outputs")
                                           or c.get("execution_count"))]
    assert dirty == [], (
        f"cells {dirty} carry saved output; clear them before committing "
        f"(jupyter nbconvert --clear-output --inplace {path.name})")
