"""Neuron Data Preprocessing Library"""
import numpy as np
from neuron.environment import NeuronModule
from neuron.interpreter import BuiltinFunction
from neuron.stdlib.tensor import NeuronTensor, _ensure_tensor

class StandardScaler:
    def __init__(self): self.mean_ = None; self.std_ = None
    def fit(self, data):
        d = data.data if isinstance(data, NeuronTensor) else np.array(data)
        self.mean_ = np.mean(d, axis=0); self.std_ = np.std(d, axis=0) + 1e-10
        return self
    def transform(self, data):
        d = data.data if isinstance(data, NeuronTensor) else np.array(data)
        return NeuronTensor((d - self.mean_) / self.std_)
    def fit_transform(self, data): self.fit(data); return self.transform(data)
    def inverse_transform(self, data):
        d = data.data if isinstance(data, NeuronTensor) else np.array(data)
        return NeuronTensor(d * self.std_ + self.mean_)

class MinMaxScaler:
    def __init__(self): self.min_ = None; self.max_ = None
    def fit(self, data):
        d = data.data if isinstance(data, NeuronTensor) else np.array(data)
        self.min_ = np.min(d, axis=0); self.max_ = np.max(d, axis=0)
        return self
    def transform(self, data):
        d = data.data if isinstance(data, NeuronTensor) else np.array(data)
        return NeuronTensor((d - self.min_) / (self.max_ - self.min_ + 1e-10))
    def fit_transform(self, data): self.fit(data); return self.transform(data)

class LabelEncoder:
    def __init__(self): self.classes_ = {}; self.inverse_ = {}
    def fit(self, labels):
        unique = sorted(set(labels))
        self.classes_ = {l: i for i, l in enumerate(unique)}
        self.inverse_ = {i: l for l, i in self.classes_.items()}
        return self
    def transform(self, labels): return [self.classes_[l] for l in labels]
    def fit_transform(self, labels): self.fit(labels); return self.transform(labels)
    def inverse_transform(self, encoded): return [self.inverse_[e] for e in encoded]

class PCA:
    def __init__(self, n_components): self.n = n_components; self.components_ = None; self.mean_ = None
    def fit(self, data):
        d = data.data if isinstance(data, NeuronTensor) else np.array(data, dtype=float)
        self.mean_ = np.mean(d, axis=0)
        centered = d - self.mean_
        cov = np.cov(centered.T)
        vals, vecs = np.linalg.eigh(cov)
        idx = np.argsort(vals)[::-1][:self.n]
        self.components_ = vecs[:, idx]
        return self
    def transform(self, data):
        d = data.data if isinstance(data, NeuronTensor) else np.array(data, dtype=float)
        return NeuronTensor((d - self.mean_) @ self.components_)
    def fit_transform(self, data): self.fit(data); return self.transform(data)

def create_preprocess_module():
    bf = BuiltinFunction
    attrs = {
        'StandardScaler': bf('StandardScaler', lambda a, **kw: StandardScaler()),
        'MinMaxScaler': bf('MinMaxScaler', lambda a, **kw: MinMaxScaler()),
        'LabelEncoder': bf('LabelEncoder', lambda a, **kw: LabelEncoder()),
        'PCA': bf('PCA', lambda a, **kw: PCA(a[0])),
    }
    return NeuronModule("preprocess", attrs)
