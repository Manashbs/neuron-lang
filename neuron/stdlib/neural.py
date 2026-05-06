"""
Neuron Neural Network Library
==============================
Layers, activations, loss functions for building AI models.
"""
import numpy as np
from neuron.environment import NeuronModule, NeuronEntity, NeuronInstance
from neuron.interpreter import BuiltinFunction
from neuron.stdlib.tensor import NeuronTensor, _ensure_tensor


class Layer:
    """Base layer class."""
    def forward(self, x): raise NotImplementedError
    def parameters(self): return []
    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()


class Dense(Layer):
    """Fully connected layer."""
    def __init__(self, in_features, out_features, bias=True):
        scale = np.sqrt(2.0 / in_features)
        self.weights = NeuronTensor(np.random.randn(in_features, out_features) * scale, requires_grad=True)
        self.bias_tensor = NeuronTensor(np.zeros(out_features), requires_grad=True) if bias else None

    def forward(self, x):
        x = _ensure_tensor(x)
        out = x.matmul(self.weights)
        if self.bias_tensor:
            out = out + self.bias_tensor
        return out

    def parameters(self):
        p = [self.weights]
        if self.bias_tensor: p.append(self.bias_tensor)
        return p


class ReLU(Layer):
    def forward(self, x):
        x = _ensure_tensor(x)
        return NeuronTensor(np.maximum(0, x.data))


class Sigmoid(Layer):
    def forward(self, x):
        x = _ensure_tensor(x)
        return NeuronTensor(1 / (1 + np.exp(-np.clip(x.data, -500, 500))))


class Tanh(Layer):
    def forward(self, x):
        x = _ensure_tensor(x)
        return NeuronTensor(np.tanh(x.data))


class Softmax(Layer):
    def forward(self, x, axis=-1):
        x = _ensure_tensor(x)
        e = np.exp(x.data - np.max(x.data, axis=axis, keepdims=True))
        return NeuronTensor(e / np.sum(e, axis=axis, keepdims=True))


class LeakyReLU(Layer):
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    def forward(self, x):
        x = _ensure_tensor(x)
        return NeuronTensor(np.where(x.data > 0, x.data, self.alpha * x.data))


class GELU(Layer):
    def forward(self, x):
        x = _ensure_tensor(x)
        return NeuronTensor(0.5 * x.data * (1 + np.tanh(np.sqrt(2/np.pi) * (x.data + 0.044715 * x.data**3))))


class Dropout(Layer):
    def __init__(self, rate=0.5):
        self.rate = rate
        self.training = True
    def forward(self, x):
        x = _ensure_tensor(x)
        if self.training:
            mask = np.random.binomial(1, 1-self.rate, x.data.shape) / (1-self.rate)
            return NeuronTensor(x.data * mask)
        return x


class BatchNorm(Layer):
    def __init__(self, features, eps=1e-5, momentum=0.1):
        self.gamma = NeuronTensor(np.ones(features), requires_grad=True)
        self.beta = NeuronTensor(np.zeros(features), requires_grad=True)
        self.eps = eps
    def forward(self, x):
        x = _ensure_tensor(x)
        mean = np.mean(x.data, axis=0)
        var = np.var(x.data, axis=0)
        x_norm = (x.data - mean) / np.sqrt(var + self.eps)
        return NeuronTensor(self.gamma.data * x_norm + self.beta.data)
    def parameters(self):
        return [self.gamma, self.beta]


class Flatten(Layer):
    def forward(self, x):
        x = _ensure_tensor(x)
        return NeuronTensor(x.data.reshape(x.data.shape[0], -1) if x.data.ndim > 1 else x.data.flatten())


class Sequential(Layer):
    def __init__(self, layers):
        self.layers = layers
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    def parameters(self):
        params = []
        for l in self.layers:
            params.extend(l.parameters())
        return params


class Embedding(Layer):
    def __init__(self, vocab_size, embed_dim):
        self.weights = NeuronTensor(np.random.randn(vocab_size, embed_dim) * 0.01, requires_grad=True)
    def forward(self, indices):
        if isinstance(indices, NeuronTensor):
            indices = indices.data.astype(int)
        return NeuronTensor(self.weights.data[indices])
    def parameters(self):
        return [self.weights]


# Loss Functions
class MSELoss:
    def compute(self, pred, target):
        pred, target = _ensure_tensor(pred), _ensure_tensor(target)
        diff = pred.data - target.data
        loss_val = np.mean(diff ** 2)
        out = NeuronTensor(loss_val, requires_grad=True)
        out._children = [pred]
        def _backward():
            pred.grad = (pred.grad or 0) + 2 * diff / diff.size * out.grad
        out._grad_fn = _backward
        return out


class CrossEntropyLoss:
    def compute(self, pred, target):
        pred, target = _ensure_tensor(pred), _ensure_tensor(target)
        # Apply softmax
        e = np.exp(pred.data - np.max(pred.data, axis=-1, keepdims=True))
        probs = e / np.sum(e, axis=-1, keepdims=True)
        # Compute loss
        if target.data.ndim == 1 and pred.data.ndim == 2:
            n = pred.data.shape[0]
            log_probs = -np.log(probs[np.arange(n), target.data.astype(int)] + 1e-10)
            loss_val = np.mean(log_probs)
        else:
            loss_val = -np.mean(np.sum(target.data * np.log(probs + 1e-10), axis=-1))
        return NeuronTensor(loss_val)


class BinaryCrossEntropyLoss:
    def compute(self, pred, target):
        pred, target = _ensure_tensor(pred), _ensure_tensor(target)
        p = np.clip(pred.data, 1e-10, 1-1e-10)
        loss_val = -np.mean(target.data * np.log(p) + (1-target.data) * np.log(1-p))
        return NeuronTensor(loss_val)


def _wrap_layer(cls):
    """Wrap a layer class as a Neuron callable."""
    def creator(args, **kwargs):
        return cls(*args, **kwargs)
    return BuiltinFunction(cls.__name__, creator)


def _wrap_layer_with_property(layer_instance):
    """Wrap a layer instance so its forward/parameters are accessible."""
    class LayerWrapper:
        def __init__(self, layer):
            self._layer = layer
        def __getattr__(self, name):
            if name == 'forward':
                return BuiltinFunction('forward', lambda a, **kw: self._layer.forward(*a))
            if name == 'parameters':
                return BuiltinFunction('parameters', lambda a, **kw: self._layer.parameters())
            return getattr(self._layer, name)
    return LayerWrapper(layer_instance)


def create_neural_module():
    attrs = {}
    bf = BuiltinFunction

    # Layers
    attrs['Dense'] = bf('Dense', lambda a, **kw: Dense(*a))
    attrs['ReLU'] = bf('ReLU', lambda a, **kw: ReLU())
    attrs['Sigmoid'] = bf('Sigmoid', lambda a, **kw: Sigmoid())
    attrs['Tanh'] = bf('Tanh', lambda a, **kw: Tanh())
    attrs['Softmax'] = bf('Softmax', lambda a, **kw: Softmax())
    attrs['LeakyReLU'] = bf('LeakyReLU', lambda a, **kw: LeakyReLU(a[0] if a else 0.01))
    attrs['GELU'] = bf('GELU', lambda a, **kw: GELU())
    attrs['Dropout'] = bf('Dropout', lambda a, **kw: Dropout(a[0] if a else 0.5))
    attrs['BatchNorm'] = bf('BatchNorm', lambda a, **kw: BatchNorm(*a))
    attrs['Flatten'] = bf('Flatten', lambda a, **kw: Flatten())
    attrs['Sequential'] = bf('Sequential', lambda a, **kw: Sequential(a[0] if a else []))
    attrs['Embedding'] = bf('Embedding', lambda a, **kw: Embedding(*a))

    # Losses
    attrs['MSELoss'] = bf('MSELoss', lambda a, **kw: MSELoss())
    attrs['CrossEntropy'] = bf('CrossEntropy', lambda a, **kw: CrossEntropyLoss())
    attrs['BinaryCrossEntropy'] = bf('BinaryCrossEntropy', lambda a, **kw: BinaryCrossEntropyLoss())

    return NeuronModule("neural", attrs)
