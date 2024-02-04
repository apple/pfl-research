# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pfl'
copyright = '2023, Apple Inc.'  # noqa: A001
author = 'Apple Inc.'
with open('../../VERSION') as f:
    release = f.read().strip()

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc', 'sphinx.ext.todo', 'sphinx.ext.mathjax',
    'sphinx.ext.githubpages', 'sphinx_autodoc_typehints',
    'sphinx_last_updated_by_git'
]

autodoc_member_order = 'bysource'
autodoc_mock_imports = [
    'horovod', 'numpy', 'scipy', 'sklearn', 'tensorflow',
    'tensorflow_probability', 'torch', 'pandas', 'pyarrow',
    'pfl.internal.ops.selector', 'xgboost'
]

templates_path = ['_templates']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_last_updated_fmt = ''
