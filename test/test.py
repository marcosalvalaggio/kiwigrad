import kiwigrad
print(kiwigrad.__version__)
from kiwigrad import *
from sklearn.datasets import make_moons
import numpy as np

X, y = make_moons(n_samples=100, noise=0.1)
inputs = [list(map(Value, xrow)) for xrow in X]
print(X.shape, type(X))
print(y.shape, type(y))

class PotNet(MLP):
    def __init__(self):
        layers = [
            Layer(nin=2, nout=16, bias=True, activation="relu"),
            Layer(nin=16, nout=16, bias=True, activation="relu"),
            Layer(nin=16, nout=1, bias=True, activation="linear")
        ]
        super().__init__(layers=layers)

model = PotNet()

print(model(input[0]))