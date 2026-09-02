# Sphinx configuration for the FastWindTerrain documentation.
#
#     sphinx-build -b html doc doc/_build
#
# The .rst files are also readable as plain text, so this is a
# convenience rather than a requirement.

project = "FastWindTerrain"
author = "FastWindTerrain contributors"
copyright = "FastWindTerrain contributors"

extensions = []
templates_path = []
exclude_patterns = ["_build"]

html_theme = "alabaster"
html_static_path = []

# One document tree rooted at index.rst.
master_doc = "index"
