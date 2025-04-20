# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "batdetect2"
copyright = "2025, Oisin Mac Aodha, Santiago Martinez Balvanera"
author = "Oisin Mac Aodha, Santiago Martinez Balvanera"
release = "1.1.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.autodoc_pydantic",
    "numpydoc",
    "myst_parser",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = []

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "soundevent": ("https://mbsantiago.github.io/soundevent/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}

# -- Options for autodoc ------------------------------------------------------
autosummary_generate = True
autosummary_imported_members = True

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "special-members": False,
    "inherited-members": False,
    "show-inheritance": True,
    "module-first": True,
}
