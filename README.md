# Kiwigrad

<h1 align="center">
<img src="logo.png" width="300">
</h1><br>

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity) [![PyPI version fury.io](https://badge.fury.io/py/kiwigrad.svg)](https://pypi.python.org/pypi/kiwigrad/)

Kiwigrad? yes, another [micrograd](https://github.com/karpathy/micrograd) version.

## Install 

To install the current release,

```console
pip install kiwigrad
```

## Kiwigrad Library

Kiwigrad library is a modified version of the micrograd package with additional features. The main features added to Kiwigrad are:

* A method for saving and loading the weights of a trained model.
* Support for RNN1 feedforward neural networks.

### Example Implementation

Here's an example implementation of a MLP net using Kiwigrad:

```python 
from kiwigrad import * 

class PotNet(MLP):
    def __init__(self):
        layers = [
            Layer(nin=2, nout=16, bias=True, activation="relu"),
            Layer(nin=16, nout=16, bias=True, activation="relu"),
            Layer(nin=16, nout=1, bias=True, activation="linear")
        ]
        super().__init__(layers=layers)

model = PotNet()
```