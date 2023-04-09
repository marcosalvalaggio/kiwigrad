import os
os.environ['CYTHON_WARNINGS'] = 'none'
os.environ['CYTHON_NO_PYTHON_FALLTHROUGH'] = '1'
os.environ['CYTHON_WARN_UNUSED_FALLTHROUGH'] = '0'
os.environ['CYTHON_FLAGS'] = '--warn=unused-fallthrough'
import pyximport
pyximport.install(language_level=3)
from .engine import Value
from .nn import (Neuron, Layer, MLP)

__version__ = "0.15"

__all__ = [
    "Value",
    "draw_dot",
    "Neuron",
    "Layer",
    "MLP"
]