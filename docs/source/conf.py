# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Hello XDNA!'
author = 'Tamino Steinert and Alex Breuer'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_copybutton',
]

templates_path = ['_templates']
exclude_patterns = []
numfig = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = 'sphinx_book_theme'
html_theme_options = {
    "repository_url": "https://github.com/scalable-analyses/xdna",
    "use_repository_button": True,
    "use_download_button": False,
    "use_source_button": False,
    "repository_branch": "main",
    "path_to_docs": "docs/source",
}
html_title = "Hello XDNA"
html_static_path = ['_static']

def setup(app):
    app.add_css_file('custom.css')
