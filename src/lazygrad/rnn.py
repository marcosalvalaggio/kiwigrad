import random
from .engine import Value
from .nn import Module


class RNNNeuron(Module):
 
    def __init__(self, nin, nonlin=True):
        self.w_xh = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.w_hh = Value(random.uniform(-1,1)) 
        self.b = Value(random.uniform(-1,1))
        self.h = Value(0)
        self.nonlin = nonlin
 
    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w_xh, x)), self.b) + (self.w_hh*self.h)
        self.h = act.relu() if self.nonlin else act
        return act.relu() if self.nonlin else act
 
    def parameters(self):
        return self.w_xh + [self.w_hh] + [self.b]
 
    def __repr__(self):
       return f"{'ReLU' if self.nonlin else 'Linear'} RNNNeuron({len(self.w_xh)})"


class RNNLayer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [RNNNeuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"RNN Layer of [{', '.join(str(n) for n in self.neurons)}]"


class RNN(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [RNNLayer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"RNN of [{', '.join(str(layer) for layer in self.layers)}]"


if __name__ == "__main__":

    # NEURON 
    x = Value(1.0)
    print("Neuron forward")
    n = RNNNeuron(1)
    print(n)
    out = n([x])
    print(out)
    print(f"ht = {n.h}")
    print(n.parameters())

    # LAYER
    print("\nLayer forward")
    l = RNNLayer(1, 3)
    print(l)
    out = l([x])
    print(out)
    print(l.parameters())

    # RNN
    print("\nRNN")
    model = RNN(1, [3,1])
    out = model([x])
    print(out)
