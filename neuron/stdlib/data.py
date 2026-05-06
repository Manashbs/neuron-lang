"""Neuron Data Processing Library"""
import numpy as np
from neuron.environment import NeuronModule
from neuron.interpreter import BuiltinFunction
from neuron.stdlib.tensor import NeuronTensor

class Dataset:
    def __init__(self, x, y=None):
        self.x = NeuronTensor(x) if not isinstance(x, NeuronTensor) else x
        self.y = NeuronTensor(y) if y is not None and not isinstance(y, NeuronTensor) else y
    def __len__(self): return len(self.x.data)
    def __getitem__(self, idx):
        if self.y is not None:
            return NeuronTensor(self.x.data[idx]), NeuronTensor(self.y.data[idx])
        return NeuronTensor(self.x.data[idx])
    def batches(self, batch_size):
        n = len(self)
        for i in range(0, n, batch_size):
            bx = NeuronTensor(self.x.data[i:i+batch_size])
            by = NeuronTensor(self.y.data[i:i+batch_size]) if self.y is not None else None
            batch = type('Batch', (), {'x': bx, 'y': by})()
            yield batch
    def shuffle(self):
        idx = np.random.permutation(len(self))
        self.x = NeuronTensor(self.x.data[idx])
        if self.y is not None:
            self.y = NeuronTensor(self.y.data[idx])
        return self

def train_test_split(x, y=None, test_ratio=0.2):
    n = len(x) if isinstance(x, list) else len(x.data)
    idx = np.random.permutation(n)
    split = int(n * (1 - test_ratio))
    if isinstance(x, NeuronTensor):
        x_train = NeuronTensor(x.data[idx[:split]])
        x_test = NeuronTensor(x.data[idx[split:]])
    else:
        x_train = [x[i] for i in idx[:split]]
        x_test = [x[i] for i in idx[split:]]
    if y is not None:
        if isinstance(y, NeuronTensor):
            return x_train, x_test, NeuronTensor(y.data[idx[:split]]), NeuronTensor(y.data[idx[split:]])
        y_train = [y[i] for i in idx[:split]]
        y_test = [y[i] for i in idx[split:]]
        return x_train, x_test, y_train, y_test
    return x_train, x_test

def one_hot(labels, num_classes):
    labels = np.array(labels, dtype=int)
    result = np.zeros((len(labels), num_classes))
    result[np.arange(len(labels)), labels] = 1
    return NeuronTensor(result)

def normalize(data):
    data = data.data if isinstance(data, NeuronTensor) else np.array(data)
    mn, mx = data.min(), data.max()
    return NeuronTensor((data - mn) / (mx - mn + 1e-10))

def standardize(data):
    data = data.data if isinstance(data, NeuronTensor) else np.array(data)
    return NeuronTensor((data - data.mean()) / (data.std() + 1e-10))

def create_data_module():
    bf = BuiltinFunction
    attrs = {
        'Dataset': bf('Dataset', lambda a, **kw: Dataset(a[0], a[1] if len(a) > 1 else None)),
        'split': bf('split', lambda a, **kw: train_test_split(a[0], a[1] if len(a) > 1 else None, kw.get('test_ratio', 0.2))),
        'one_hot': bf('one_hot', lambda a, **kw: one_hot(a[0], a[1])),
        'normalize': bf('normalize', lambda a, **kw: normalize(a[0])),
        'standardize': bf('standardize', lambda a, **kw: standardize(a[0])),
    }
    return NeuronModule("data", attrs)
