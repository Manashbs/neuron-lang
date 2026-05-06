"""
Neuron Optimizers
=================
SGD, Adam, AdamW, RMSprop optimizers for training.
"""
import numpy as np
from neuron.environment import NeuronModule
from neuron.interpreter import BuiltinFunction
from neuron.stdlib.tensor import NeuronTensor


class SGD:
    def __init__(self, params, lr=0.01, momentum=0.0):
        self.params = params if isinstance(params, list) else self._extract_params(params)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(p.data) for p in self.params]

    def _extract_params(self, model):
        if hasattr(model, 'parameters'):
            return model.parameters() if callable(model.parameters) else model.parameters
        return []

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is not None:
                grad = np.array(p.grad) if not isinstance(p.grad, np.ndarray) else p.grad
                if self.momentum:
                    self.velocities[i] = self.momentum * self.velocities[i] + grad
                    p.data -= self.lr * self.velocities[i]
                else:
                    p.data -= self.lr * grad

    def zero_grad(self):
        for p in self.params:
            p.grad = None


class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params if isinstance(params, list) else self._extract_params(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def _extract_params(self, model):
        if hasattr(model, 'parameters'):
            return model.parameters() if callable(model.parameters) else model.parameters
        return []

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is not None:
                grad = np.array(p.grad) if not isinstance(p.grad, np.ndarray) else p.grad
                self.m[i] = self.beta1 * self.m[i] + (1-self.beta1) * grad
                self.v[i] = self.beta2 * self.v[i] + (1-self.beta2) * grad**2
                m_hat = self.m[i] / (1-self.beta1**self.t)
                v_hat = self.v[i] / (1-self.beta2**self.t)
                p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad = None


class RMSprop:
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8):
        self.params = params if isinstance(params, list) else self._extract_params(params)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.v = [np.zeros_like(p.data) for p in self.params]

    def _extract_params(self, model):
        if hasattr(model, 'parameters'):
            return model.parameters() if callable(model.parameters) else model.parameters
        return []

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is not None:
                grad = np.array(p.grad) if not isinstance(p.grad, np.ndarray) else p.grad
                self.v[i] = self.alpha * self.v[i] + (1-self.alpha) * grad**2
                p.data -= self.lr * grad / (np.sqrt(self.v[i]) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad = None


def create_optim_module():
    attrs = {}
    bf = BuiltinFunction
    attrs['SGD'] = bf('SGD', lambda a, **kw: SGD(a[0], lr=kw.get('lr', a[1] if len(a) > 1 else 0.01), momentum=kw.get('momentum', 0.0)))
    attrs['Adam'] = bf('Adam', lambda a, **kw: Adam(a[0], lr=kw.get('lr', a[1] if len(a) > 1 else 0.001)))
    attrs['RMSprop'] = bf('RMSprop', lambda a, **kw: RMSprop(a[0], lr=kw.get('lr', a[1] if len(a) > 1 else 0.01)))
    return NeuronModule("optim", attrs)
