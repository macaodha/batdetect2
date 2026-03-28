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
    "sphinx_click",
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
html_theme_options = {
    "home_page_in_toc": True,
    "show_navbar_depth": 2,
    "show_toc_level": 2,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "click": ("https://click.palletsprojects.com/en/stable/", None),
    "librosa": ("https://librosa.org/doc/latest/", None),
    "lightning": ("https://lightning.ai/docs/pytorch/stable/", None),
    "loguru": ("https://loguru.readthedocs.io/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "omegaconf": ("https://omegaconf.readthedocs.io/en/latest/", None),
    "pytorch": ("https://pytorch.org/docs/stable/", None),
    "soundevent": ("https://mbsantiago.github.io/soundevent/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}

# -- Options for autodoc ------------------------------------------------------
autosummary_generate = False
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

numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False
