# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from xynergy._meta import (
    APP_NAME,
    APP_TAGLINE,
    CONTACT_EMAIL,
    COPYRIGHT_HOLDER,
    DEVELOPED_BY,
    __version__,
)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = APP_NAME
copyright = f"{date.today().year}, {COPYRIGHT_HOLDER}"
author = f"{DEVELOPED_BY} ({CONTACT_EMAIL})"
version = __version__
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc"]
autodoc_mock_imports = ["xgboost"]

templates_path = ["_templates"]
exclude_patterns = []

default_role = "code"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = f"{APP_NAME} {__version__}"
html_theme_options = {
    "logo": {"text": APP_NAME},
}
html_context = {
    "app_tagline": APP_TAGLINE,
    "contact_email": CONTACT_EMAIL,
}
