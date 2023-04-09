from .graph import draw_dot
import os
os.environ['CYTHON_FLAGS'] = '--directive=warn.implicit_fallthrough=False'
import pyximport
pyximport.install(language_level=3)
from .engine import Value
from .nn import (Neuron, Layer, MLP)

__version__ = "0.12"

__all__ = [
    "Value",
    "draw_dot",
    "Neuron",
    "Layer",
    "MLP"
]