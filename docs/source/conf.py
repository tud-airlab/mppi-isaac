import mock
import sys
 
MOCK_MODULES = ['isaacgym']
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'mppiisaac'
copyright = '2024, Chadi Salmi, Corrado Pezzato, Elia Trevisan, Max Spahn'
author = 'Chadi Salmi, Corrado Pezzato, Elia Trevisan, Max Spahn'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
