# cython: warning_errors=-Wno-fallthrough
# cython: language_level=3

import random
from .engine import Value
import pickle


class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron(Module):

    def __init__(self, nin, bias: bool = True, nonlin: bool =True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        if bias: 
            self.b = Value(random.uniform(-1,1))
        else: 
            self.b = Value(0.)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):

    def __init__(self, nin, nout, bias: bool = True, **kwargs):
        self.neurons = [Neuron(nin, bias=bias, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):

    def __init__(self, nin, nouts, bias: bool = True):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1, bias = bias) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
    
    def save(self, save_name: str = "weights"):
        cdef list weights = []
        for layer in self.layers:
            for neuron in layer.neurons:
                weights.append([i.data for i in neuron.w])
                weights.append(neuron.b.data)
        file_name = f'{save_name}.pkl'
        with open(file_name, 'wb') as f:
            pickle.dump(weights, f)

    def load(self, save_name: str = "weights"):
        file_name = f'{save_name}.pkl'
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        cdef list load_weights = []
        for param in params:
            if isinstance(param, list):
                load_param = [Value(i) for i in param]
                load_weights.append(load_param)
            else:
                load_param = Value(param)
                load_weights.append(load_param)
        cdef int i = 0
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.w = load_weights[i]
                neuron.b = load_weights[i+1]
                i += 2