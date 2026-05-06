"""Neuron ML Metrics Library"""
import numpy as np
from neuron.environment import NeuronModule
from neuron.interpreter import BuiltinFunction
from neuron.stdlib.tensor import NeuronTensor, _ensure_tensor

def _to_np(x):
    if isinstance(x, NeuronTensor): return x.data
    return np.array(x)

def accuracy(pred, target):
    p, t = _to_np(pred), _to_np(target)
    if p.ndim > 1: p = np.argmax(p, axis=-1)
    return float(np.mean(p == t))

def precision(pred, target, avg='macro'):
    p, t = _to_np(pred), _to_np(target)
    if p.ndim > 1: p = np.argmax(p, axis=-1)
    classes = np.unique(np.concatenate([p, t]))
    precs = []
    for c in classes:
        tp = np.sum((p == c) & (t == c))
        fp = np.sum((p == c) & (t != c))
        precs.append(tp / (tp + fp + 1e-10))
    return float(np.mean(precs))

def recall(pred, target, avg='macro'):
    p, t = _to_np(pred), _to_np(target)
    if p.ndim > 1: p = np.argmax(p, axis=-1)
    classes = np.unique(np.concatenate([p, t]))
    recs = []
    for c in classes:
        tp = np.sum((p == c) & (t == c))
        fn = np.sum((p != c) & (t == c))
        recs.append(tp / (tp + fn + 1e-10))
    return float(np.mean(recs))

def f1_score(pred, target):
    p = precision(pred, target)
    r = recall(pred, target)
    return 2 * p * r / (p + r + 1e-10)

def mse(pred, target):
    return float(np.mean((_to_np(pred) - _to_np(target))**2))

def rmse(pred, target):
    return float(np.sqrt(mse(pred, target)))

def mae(pred, target):
    return float(np.mean(np.abs(_to_np(pred) - _to_np(target))))

def r2_score(pred, target):
    t = _to_np(target)
    ss_res = np.sum((_to_np(pred) - t)**2)
    ss_tot = np.sum((t - np.mean(t))**2)
    return float(1 - ss_res / (ss_tot + 1e-10))

def confusion_matrix(pred, target, num_classes=None):
    p, t = _to_np(pred).astype(int), _to_np(target).astype(int)
    if p.ndim > 1: p = np.argmax(p, axis=-1)
    nc = num_classes or max(p.max(), t.max()) + 1
    cm = np.zeros((nc, nc), dtype=int)
    for pi, ti in zip(p, t): cm[ti, pi] += 1
    return cm.tolist()

def create_metrics_module():
    bf = BuiltinFunction
    attrs = {
        'accuracy': bf('accuracy', lambda a, **kw: accuracy(a[0], a[1])),
        'precision': bf('precision', lambda a, **kw: precision(a[0], a[1])),
        'recall': bf('recall', lambda a, **kw: recall(a[0], a[1])),
        'f1': bf('f1', lambda a, **kw: f1_score(a[0], a[1])),
        'mse': bf('mse', lambda a, **kw: mse(a[0], a[1])),
        'rmse': bf('rmse', lambda a, **kw: rmse(a[0], a[1])),
        'mae': bf('mae', lambda a, **kw: mae(a[0], a[1])),
        'r2': bf('r2', lambda a, **kw: r2_score(a[0], a[1])),
        'confusion_matrix': bf('confusion_matrix', lambda a, **kw: confusion_matrix(a[0], a[1])),
    }
    return NeuronModule("metrics", attrs)
