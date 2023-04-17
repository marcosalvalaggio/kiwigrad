from engine import Value
import random 
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

    
class MLP(Module):

    # def __init__(self, nin, nouts, bias: bool = True):
    #     sz = [nin] + nouts
    #     self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1, bias = bias) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
    
    def save(self, save_name: str = "weights"):
        weights = []
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


if __name__ == "__main__":

    from sklearn.datasets import make_moons
    import numpy as np
    X, y = make_moons(n_samples=100, noise=0.1)
    print(X.shape, type(X))
    print(y.shape, type(y)) 
    print(X[:5])
    print(y[:5])
    class PotNet(MLP):
        def __init__(self) -> None:
            self.layers = [
                            Layer(nin=2, nout=16, bias=True),
                            Layer(nin=16, nout=16, bias=True),
                            Layer(nin=16, nout=1, bias=True, nonlin=False)
                        ]
    model = PotNet()
    print("number of parameters", len(model.parameters()))

    def accuracy_val(model, X ,y_true):
        y_pred = np.array([model(X[i]).sigmoid().data for i in range(X.shape[0])]).round()
        correct_results = np.sum(y_pred == y_true)
        acc = correct_results/y_true.shape[0]
        acc = np.round(acc * 100)
        return acc

    epochs = 100 
    for epoch in range(epochs): 
        for i in range(X.shape[0]):
            output = model(X[i]).sigmoid()
            neg_output = 1-output
            target = y[i]
            bce_loss = -(target * output.log() + (1 - target) * neg_output.log())
            bce_loss.backward()
            for p in model.parameters():
                p.data += -0.001 * p.grad 
            model.zero_grad()
        acc = accuracy_val(model=model, X=X, y_true=y)
        if epoch%10 == 0:
            print(f"epoch: {epoch} |", f"loss: {bce_loss.data:.2f} |", f"accuracy: {acc}")