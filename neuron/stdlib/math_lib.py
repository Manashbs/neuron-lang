"""Neuron Math Library"""
import numpy as np
from neuron.environment import NeuronModule
from neuron.interpreter import BuiltinFunction
from neuron.stdlib.tensor import NeuronTensor

def create_math_module():
    bf = BuiltinFunction
    attrs = {
        'pi': np.pi, 'e': np.e, 'inf': float('inf'), 'nan': float('nan'),
        'sin': bf('sin', lambda a, **kw: np.sin(a[0])),
        'cos': bf('cos', lambda a, **kw: np.cos(a[0])),
        'tan': bf('tan', lambda a, **kw: np.tan(a[0])),
        'asin': bf('asin', lambda a, **kw: np.arcsin(a[0])),
        'acos': bf('acos', lambda a, **kw: np.arccos(a[0])),
        'atan': bf('atan', lambda a, **kw: np.arctan(a[0])),
        'exp': bf('exp', lambda a, **kw: np.exp(a[0])),
        'log': bf('log', lambda a, **kw: np.log(a[0])),
        'log2': bf('log2', lambda a, **kw: np.log2(a[0])),
        'log10': bf('log10', lambda a, **kw: np.log10(a[0])),
        'sqrt': bf('sqrt', lambda a, **kw: np.sqrt(a[0])),
        'ceil': bf('ceil', lambda a, **kw: int(np.ceil(a[0]))),
        'floor': bf('floor', lambda a, **kw: int(np.floor(a[0]))),
        'abs': bf('abs', lambda a, **kw: abs(a[0])),
        'pow': bf('pow', lambda a, **kw: a[0] ** a[1]),
        'factorial': bf('factorial', lambda a, **kw: int(np.math.factorial(int(a[0])))),
        'gcd': bf('gcd', lambda a, **kw: np.gcd(int(a[0]), int(a[1]))),
        'mean': bf('mean', lambda a, **kw: np.mean(a[0])),
        'median': bf('median', lambda a, **kw: np.median(a[0])),
        'std': bf('std', lambda a, **kw: np.std(a[0])),
        'var': bf('var', lambda a, **kw: np.var(a[0])),
        'random': bf('random', lambda a, **kw: np.random.random()),
        'randint': bf('randint', lambda a, **kw: np.random.randint(a[0], a[1])),
    }
    return NeuronModule("math", attrs)
