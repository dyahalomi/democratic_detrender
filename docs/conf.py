# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os


project = 'democratic_detrender'
copyright = '2024, Daniel Yahalomi'
author = 'Daniel Yahalomi'
release = '0.0.1'

# if you eventually refactor into a package format, add the path here
sys.path.insert(0, os.path.abspath(".."))
# sys.path.insert(
#     0, os.path.abspath("../democratic_detrender")
# )  # If your modules are in a parent directory

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "sphinx_automodapi.automodapi",
    "myst_parser",
    "sphinxcontrib.video",
    # "sphinx.ext.pngmath",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "visualizations"]
source_suffix = [".rst", ".md"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_copy_source = True
html_show_sourcelink = True
html_sourcelink_suffix = ""
html_title = "democratic detrender"
# html_favicon = "_static/jupiter.png"
html_static_path = ["_static"]

html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/dyahalomi/democratic_detrender",
    "repository_branch": "main",
    "use_repository_button": True,
    "use_download_button": True,
    "show_prev_next": False,
    "logo": {
        "text": "democratic detrender",
        "image_light": "_static/logo_placeholder.avif",
        "image_dark": "_static/logo_placeholder.avif",
    },
}

html_context = {"default_mode": "light"}

html_sidebars = {
    "**": ["navbar-logo.html", "search-field.html", "sbt-sidebar-nav.html"]
}