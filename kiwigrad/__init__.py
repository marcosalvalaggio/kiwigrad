from .graph import draw_dot
import pyximport
import os
os.environ['CYTHON_WARNINGS'] = 'none'
os.environ['CYTHON_WARN_UNUSED_FALLTHROUGH'] = 'false'
os.environ['CYTHON_FLAGS'] = '-Wno-unused-fallthrough'
import pyximport
pyximport.install()
from .engine import Value
from .nn import (Neuron, Layer, MLP)

__version__ = "0.11"

__all__ = [
    "Value",
    "draw_dot",
    "Neuron",
    "Layer",
    "MLP"
]