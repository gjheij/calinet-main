# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

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
