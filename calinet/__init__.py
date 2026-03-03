"""
calinet package
"""

from importlib.metadata import version
import os

__version__ = version("calinet")

# Absolute path to the installed package directory
PACKAGE_ROOT = os.path.dirname(__file__)

# Absolute path to the repository root (one level up)
REPO_ROOT = os.path.abspath(os.path.join(PACKAGE_ROOT, ".."))

# Re-export selected high-level functionality
# from . import get_manuscript_figures
# from . import convert
# from . import methods

# __all__ = [
#     "convert",
#     "methods",
# ]
