# Arquivo de configuração para o Sphinx documentation builder (Português)
#
# Para a lista completa de valores de configuração, veja a documentação:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import os
sys.path.insert(0, os.path.abspath('../..'))

# -- Informações do projeto -------------------------------------------------
project = 'Local Inference Calculator'
copyright = '2025, Local Inference Calculator'
author = 'Local Inference Calculator'
release = '0.1.0'
language = 'pt_BR'

# -- Configuração geral ------------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

templates_path = ['../_templates']
exclude_patterns = ['../_build', 'Thumbs.db', '.DS_Store']

# -- Opções para saída HTML -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['../_static']
html_context = {
    'language': 'pt_BR',
    'languages': ['en', 'pt_BR'],
    'current_language': 'pt_BR',
    'language_links': True,
}
html_title = 'Local Inference Calculator'
html_short_title = 'LIC'

# Configurações do Napoleon
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Configurações do Autodoc
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Mapeamento do Intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}

# Configurações regionais
locale_dirs = ['../locales']
gettext_compact = False
