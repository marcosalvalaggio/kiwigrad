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
    
    def load(self, save_name: str = "weights"):
        file_name = f'{save_name}.pkl'
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        load_weights = []
        for param in params:
            if isinstance(param, list):
                load_param = [Value(i) for i in param]
                load_weights.append(load_param)
            else:
                load_param = Value(param)
                load_weights.append(load_param)
        i = 0
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.w = load_weights[i]
                neuron.b = load_weights[i+1]
                i += 2

    def save(self, save_name: str = "weights"):
        weights = []
        for layer in self.layers:
            for neuron in layer.neurons:
                weights.append([i.data for i in neuron.w])
                weights.append(neuron.b.data)
        file_name = f'{save_name}.pkl'
        with open(file_name, 'wb') as f:
            pickle.dump(weights, f)



if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.metrics import r2_score
    df_x = pd.read_csv("../../test/data/x.csv", sep=";")
    xs = df_x.to_numpy()
    print(xs.shape)
    df_y = pd.read_csv("../../test/data/x.csv/y.csv", sep=";")
    y = df_y.to_numpy().squeeze()
    print(y.shape)
    model = MLP(6, [16, 1], bias=True) # 1-layer neural network
    print(model)
    print("number of parameters", len(model.parameters()))

    # loop
    for k in range(30):
        for i in range(len(xs)):
            output = model(xs[i])
            target = y[i]
            loss = ((output - target) ** 2)
            loss.backward()
            for p in model.parameters():
                p.data += -0.001 * p.grad 
            model.zero_grad()
        if k%5 == 0:
            print(k, loss)
    
    print('\nTEST')
    output = model(xs[0])
    target = y[0]
    print(f'output: {output.data}', f'target: {target}')

    # r2 score
    output = [model(x).data for x in xs]
    r2 = r2_score(y, output)
    print(f'r2 score: {r2}')

