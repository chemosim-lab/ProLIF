# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('./source/'))
sys.path.insert(0, os.path.abspath('./notebooks/'))
from recommonmark.transform import AutoStructify
from jupyter_client import kernelspec

# -- Project information -----------------------------------------------------

project = 'ProLIF'
copyright = '2020-2021, Cédric Bouysset'
author = 'Cédric Bouysset'


# -- General configuration ---------------------------------------------------

github_doc_root = 'https://github.com/chemosim-lab/ProLIF/tree/master/docs/'
needs_sphinx = '4.1.2'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.intersphinx',
              'sphinx.ext.mathjax', 'sphinx.ext.viewcode',
              'sphinx.ext.napoleon', 'sphinx.ext.autosectionlabel',
              'sphinx_rtd_theme', 'recommonmark',
              'IPython.sphinxext.ipython_console_highlighting',
              'IPython.sphinxext.ipython_directive',
              'nbsphinx',
]

mathjax_path = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML'
autosectionlabel_prefix_document = True
napoleon_google_docstring = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

source_suffix = ['.rst', '.md']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
pygments_style = 'sphinx'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = [] # ['_static']

intersphinx_mapping = {'https://docs.python.org/3/': None,
                       'https://numpy.org/doc/stable/': None,
                       'https://docs.mdanalysis.org/': None,
                       'https://www.rdkit.org/docs/': None,
                       'https://pandas.pydata.org/docs/': None,
                       }

# app setup hook
def setup(app):
    app.add_config_value('recommonmark_config', {
        #'url_resolver': lambda url: github_doc_root + url,
        'auto_toc_tree_section': 'Contents',
        'enable_math': False,
        'enable_inline_math': False,
        'enable_eval_rst': True,
    }, True)
    app.add_transform(AutoStructify)
