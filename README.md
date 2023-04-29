# Kiwigrad

<h1 align="center">
<img src="logo.png" width="200">
</h1><br>

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity) 

- [Kiwigrad](#kiwigrad)
  - [Install](#install)
  - [Kiwigrad Library](#kiwigrad-library)
    - [Examples](#examples)
  - [TODOS](#todos)

Kiwigrad? yes, it is another version of [micrograd](https://github.com/karpathy/micrograd) that was created just for fun and experimentation.

## Install 

To install the current release,

```console
pip install kiwigrad
```

## Kiwigrad Library

Kiwigrad library is a modified version of the [micrograd](https://github.com/karpathy/micrograd) and the [minigrad](https://github.com/goktug97/minigrad) packages with additional features. The main features added to Kiwigrad are:

* Training is faster due to the C implementation of the Value object.
* Methods for saving and loading the weights of a trained model.
* Support for RNN(1) feedforward neural networks.

### Examples

* In the [examples](examples/) folder, you can find examples of models trained using the Kiwigrad library.
* Here is a declaration example of an MLP net using Kiwigrad:
  
```python 
from kiwigrad import MLP, Layer

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

## TODOS

* Include the activation functions tanh in the Value object.