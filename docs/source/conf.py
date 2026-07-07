"""Sphinx configuration for the statrl documentation site."""

import os
import sys

# Fallback for local builds without `pip install .`: the repo directory itself is the
# `statrl` package, so put its PARENT on the path to make `import statrl` resolve.
sys.path.insert(0, os.path.abspath("../../.."))

# -- Project information ------------------------------------------------------
project = "statrl"
copyright = "2026, StatisticalRL"
author = "StatisticalRL"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",   # scikit-learn-style summary tables + per-object pages
    "sphinx.ext.napoleon",      # NumPy- and Google-style docstrings (no numpydoc dep)
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_design",            # grid cards on the landing page
]

autosummary_generate = True
templates_path = ["_templates"]

# gymnasium isn't installed in the docs build; mock it so the environment/interaction
# modules import and their docstrings render.
autodoc_mock_imports = ["gymnasium"]

# Napoleon handles the predominant NumPy style; a few docstrings use rst field lists,
# which autodoc parses natively.
napoleon_numpy_docstring = True
napoleon_google_docstring = False

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "gymnasium": ("https://gymnasium.farama.org/", None),
}

exclude_patterns = ["_build"]

# -- HTML output (scikit-learn's theme) --------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "statrl"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/StatisticalRL/statrl",
            "icon": "fa-brands fa-github",
        },
    ],
    "navbar_align": "left",
    "show_prev_next": True,
    "navigation_with_keys": False,
}
