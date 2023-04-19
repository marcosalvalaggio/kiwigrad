import kiwigrad
print(kiwigrad.__version__)
from kiwigrad import *
from sklearn.datasets import make_moons
import numpy as np

X, y = make_moons(n_samples=100, noise=0.1)
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

model.save()