# -*- coding: utf-8 -*-

__all__ = ["GaussianProcess", "terms"]

from . import terms
from .celeriac_version import __version__  # noqa
from .gp import GaussianProcess

__uri__ = "https://github.com/dfm/celeriac"
__author__ = "Dan Foreman-Mackey"
__email__ = "foreman.mackey@gmail.com"
__description__ = "celerite in JAX"
__license__ = "MIT"
