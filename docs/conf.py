import os
import sys

# Make project package importable by Sphinx
sys.path.insert(0, os.path.abspath(".."))

autodoc_mock_imports = [
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "sklearn",
    "scanpy",
    "anndata",
    "torch",
    "torch_geometric",
    "louvain",
    "harmonypy",
]

project = "SMODER"
author = "Shucun Xiong"
copyright = "2026, Shucun Xiong"
release = "0.1.0"

html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "/")

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"

# Keep docs builds lightweight and deterministic
nbsphinx_execute = "never"

# Optional: cleaner autodoc output
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = True