# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from democratic_detrender import __version__

release = __version__
project = 'democratic_detrender'
author = 'Daniel Yahalomi'
copyright = f"{datetime.datetime.now().year}, {author}"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
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
    "sphinx.ext.inheritance_diagram",
    "sphinx_automodapi.smart_resolver",
]

mathjax_path = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS_HTML"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "visualizations"]
source_suffix = [".rst", ".md"]
# The master toctree document.
master_doc = "index"
# Treat everything in single ` as a Python reference.
default_role = "py:obj"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_copy_source = True
html_show_sourcelink = True
html_sourcelink_suffix = ""
html_title = "democratic detrender"
# html_favicon = "_static/jupiter.png"
html_static_path = ["_static"]
autoclass_content = "both"

# TODO: remove unused
html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/dyahalomi/democratic_detrender",
    "repository_branch": "main",
    "use_repository_button": True,
    "use_download_button": True,
    "show_prev_next": False,
    "logo": {
        "text": "democratic detrender",
        "image_light": "_static/democratic_detrender_logo_final.png",
        "image_dark": "_static/democratic_detrender_logo_final.png",
    },
}

html_context = {"default_mode": "light"}

html_sidebars = {
    "**": ["navbar-logo.html", "search-field.html", "sbt-sidebar-nav.html"]
}
