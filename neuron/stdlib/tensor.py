"""
Neuron Tensor Library
=====================
NumPy-backed tensor operations with autograd support.
"""
import numpy as np
from neuron.environment import NeuronModule, NeuronEntity, NeuronInstance, NeuronFunction
from neuron.interpreter import BuiltinFunction


class NeuronTensor:
    """Wrapper around NumPy arrays with autograd."""

    def __init__(self, data, requires_grad=False):
        if isinstance(data, NeuronTensor):
            self.data = data.data.copy()
        elif isinstance(data, np.ndarray):
            self.data = data.astype(np.float64)
        else:
            self.data = np.array(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None
        self._grad_fn = None
        self._children = []

    @property
    def shape(self):
        return list(self.data.shape)

    @property
    def size(self):
        return int(self.data.size)

    @property
    def value(self):
        if self.data.size == 1:
            return float(self.data.flat[0])
        return self.data.tolist()

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        topo = []
        visited = set()
        def build(t):
            if id(t) not in visited:
                visited.add(id(t))
                for child in t._children:
                    build(child)
                topo.append(t)
        build(self)
        for t in reversed(topo):
            if t._grad_fn:
                t._grad_fn()

    def zero_grad(self):
        self.grad = None

    # Arithmetic
    def __add__(self, other):
        other = _ensure_tensor(other)
        out = NeuronTensor(self.data + other.data, self.requires_grad or other.requires_grad)
        if out.requires_grad:
            out._children = [self, other]
            def _backward():
                if self.requires_grad:
                    self.grad = (self.grad or 0) + out.grad
                if other.requires_grad:
                    other.grad = (other.grad or 0) + out.grad
            out._grad_fn = _backward
        return out

    def __sub__(self, other):
        other = _ensure_tensor(other)
        out = NeuronTensor(self.data - other.data, self.requires_grad or other.requires_grad)
        if out.requires_grad:
            out._children = [self, other]
            def _backward():
                if self.requires_grad:
                    self.grad = (self.grad or 0) + out.grad
                if other.requires_grad:
                    other.grad = (other.grad or 0) - out.grad
            out._grad_fn = _backward
        return out

    def __mul__(self, other):
        other = _ensure_tensor(other)
        out = NeuronTensor(self.data * other.data, self.requires_grad or other.requires_grad)
        if out.requires_grad:
            out._children = [self, other]
            def _backward():
                if self.requires_grad:
                    self.grad = (self.grad or 0) + other.data * out.grad
                if other.requires_grad:
                    other.grad = (other.grad or 0) + self.data * out.grad
            out._grad_fn = _backward
        return out

    def __truediv__(self, other):
        other = _ensure_tensor(other)
        return NeuronTensor(self.data / other.data)

    def __neg__(self):
        return NeuronTensor(-self.data, self.requires_grad)

    def __pow__(self, power):
        return NeuronTensor(self.data ** power)

    def matmul(self, other):
        other = _ensure_tensor(other)
        out = NeuronTensor(self.data @ other.data, self.requires_grad or other.requires_grad)
        if out.requires_grad:
            out._children = [self, other]
            def _backward():
                if self.requires_grad:
                    self.grad = (self.grad or 0) + out.grad @ other.data.T
                if other.requires_grad:
                    other.grad = (other.grad or 0) + self.data.T @ out.grad
            out._grad_fn = _backward
        return out

    def transpose(self):
        return NeuronTensor(self.data.T, self.requires_grad)

    def reshape(self, *shape):
        return NeuronTensor(self.data.reshape(*shape), self.requires_grad)

    def flatten(self):
        return NeuronTensor(self.data.flatten(), self.requires_grad)

    def sum(self, axis=None):
        out = NeuronTensor(np.sum(self.data, axis=axis), self.requires_grad)
        if out.requires_grad:
            out._children = [self]
            def _backward():
                self.grad = (self.grad or 0) + np.ones_like(self.data) * out.grad
            out._grad_fn = _backward
        return out

    def mean(self, axis=None):
        out = NeuronTensor(np.mean(self.data, axis=axis), self.requires_grad)
        if out.requires_grad:
            out._children = [self]
            def _backward():
                self.grad = (self.grad or 0) + np.ones_like(self.data) * out.grad / self.data.size
            out._grad_fn = _backward
        return out

    def max(self, axis=None): return NeuronTensor(np.max(self.data, axis=axis))
    def min(self, axis=None): return NeuronTensor(np.min(self.data, axis=axis))
    def argmax(self, axis=None): return int(np.argmax(self.data, axis=axis)) if axis is None else NeuronTensor(np.argmax(self.data, axis=axis))
    def argmin(self, axis=None): return int(np.argmin(self.data, axis=axis)) if axis is None else NeuronTensor(np.argmin(self.data, axis=axis))
    def abs(self): return NeuronTensor(np.abs(self.data))
    def exp(self): return NeuronTensor(np.exp(self.data))
    def log(self): return NeuronTensor(np.log(self.data))
    def sqrt(self): return NeuronTensor(np.sqrt(self.data))
    def clip(self, low, high): return NeuronTensor(np.clip(self.data, low, high))
    def squeeze(self): return NeuronTensor(np.squeeze(self.data))
    def unsqueeze(self, axis): return NeuronTensor(np.expand_dims(self.data, axis))
    def tolist(self): return self.data.tolist()
    def copy(self): return NeuronTensor(self.data.copy(), self.requires_grad)

    def __repr__(self):
        return f"tensor({self.data})"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        result = self.data[idx]
        if isinstance(result, np.ndarray):
            return NeuronTensor(result)
        return float(result)

    def __iter__(self):
        for item in self.data:
            yield NeuronTensor(item) if isinstance(item, np.ndarray) else float(item)


def _ensure_tensor(x):
    if isinstance(x, NeuronTensor): return x
    return NeuronTensor(x)


def create_tensor_module():
    """Create the tensor standard library module."""
    attrs = {}
    bf = BuiltinFunction

    def _tensor_create(args, **kw):
        rg = kw.get('requires_grad', False)
        return NeuronTensor(args[0], requires_grad=rg)

    attrs['create'] = bf('create', _tensor_create)
    attrs['zeros'] = bf('zeros', lambda a, **kw: NeuronTensor(np.zeros(tuple(a))))
    attrs['ones'] = bf('ones', lambda a, **kw: NeuronTensor(np.ones(tuple(a))))
    attrs['random'] = bf('random', lambda a, **kw: NeuronTensor(np.random.randn(*a)))
    attrs['rand'] = bf('rand', lambda a, **kw: NeuronTensor(np.random.rand(*a)))
    attrs['eye'] = bf('eye', lambda a, **kw: NeuronTensor(np.eye(a[0])))
    attrs['arange'] = bf('arange', lambda a, **kw: NeuronTensor(np.arange(*a)))
    attrs['linspace'] = bf('linspace', lambda a, **kw: NeuronTensor(np.linspace(a[0], a[1], int(a[2]) if len(a) > 2 else 50)))
    attrs['concat'] = bf('concat', lambda a, **kw: NeuronTensor(np.concatenate([t.data for t in a[0]], axis=a[1] if len(a) > 1 else 0)))
    attrs['stack'] = bf('stack', lambda a, **kw: NeuronTensor(np.stack([t.data for t in a[0]], axis=a[1] if len(a) > 1 else 0)))

    return NeuronModule("tensor", attrs)
