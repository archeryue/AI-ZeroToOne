# neural network module

import random
from tinyflow.autograd import Scalar

class Module:
    def param(self):
        return []

    def zero_grad(self):
        for p in self.param():
            p.grad = 0

class Neuron(Module):
    def __init__(self, D_in, act=True):
        self.w = [Scalar(random.random()) for _ in range(D_in)]
        self.b = Scalar(random.random())
        self.act = act

    def __call__(self, x):
        h = Scalar(0.0)
        for wi, xi in zip(self.w, x):
            h = wi * xi + h
        h = h + self.b
        return h.relu() if self.act else h

    def param(self):
        return self.w + [self.b]

class Layer(Module):
    def __init__(self, D_in, D_out, act=True):
        self.neurons = [Neuron(D_in, act) for _ in range(D_out)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def param(self):
        return [p for n in self.neurons for p in n.param()]
        
class MLP(Module):
    def __init__(self, D_in, D_outs):
        D_layers = [D_in] + D_outs
        # last layer is linear, no activation
        self.layers = [Layer(D_layers[i], D_layers[i+1], i != len(D_outs) - 1) for i in range(len(D_outs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
    def param(self):
        return [p for layer in self.layers for p in layer.param()]

class MSELoss:
    def __call__(self, y_pred, y_true):
        loss = Scalar(0.0)
        for yp, yt in zip(y_pred, y_true):
            delta = yp - yt
            square = delta ** 2
            loss = loss + square
        return loss / Scalar(len(y_pred))