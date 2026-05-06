"""
Neuron Standard Library
=======================
Module registry for all built-in Neuron libraries.
"""

from neuron.environment import NeuronModule


_MODULE_CACHE = {}


def get_module(name):
    """Load and return a Neuron standard library module,
    or fall back to importing a real Python package."""
    if name in _MODULE_CACHE:
        return _MODULE_CACHE[name]

    # 1. Try Neuron built-in stdlib
    loader = _MODULE_LOADERS.get(name)
    if loader:
        module = loader()
        _MODULE_CACHE[name] = module
        return module

    # 2. Fall back to importing a real Python package
    module = _try_python_import(name)
    if module:
        _MODULE_CACHE[name] = module
        return module

    return None


def _try_python_import(name):
    """Try to import a real Python package. If not installed, auto-install it."""
    try:
        import importlib
        py_module = importlib.import_module(name)
        return PythonModuleWrapper(name, py_module)
    except ImportError:
        # Auto-install the package
        from neuron.package_manager import auto_install_and_import
        py_module = auto_install_and_import(name)
        if py_module:
            return PythonModuleWrapper(name, py_module)
        return None
    except Exception:
        return None


class PythonModuleWrapper:
    """Wraps a real Python module so it works seamlessly in Neuron.
    Supports attribute access, function calls, and sub-module access."""

    def __init__(self, name, py_module):
        self.name = name
        self._module = py_module

    def get(self, attr_name, line=None, column=None):
        if hasattr(self._module, attr_name):
            attr = getattr(self._module, attr_name)
            # Wrap sub-modules
            if hasattr(attr, '__module__') and hasattr(attr, '__path__'):
                return PythonModuleWrapper(f"{self.name}.{attr_name}", attr)
            return _wrap_python_value(attr)
        from neuron.errors import NeuronAttributeError
        raise NeuronAttributeError(
            f"Python module '{self.name}' has no attribute '{attr_name}'",
            line=line, column=column
        )

    @property
    def attributes(self):
        return {k: _wrap_python_value(getattr(self._module, k))
                for k in dir(self._module) if not k.startswith('_')}

    def __repr__(self):
        return f"<python module '{self.name}'>"


def _wrap_python_value(value):
    """Make Python values usable from Neuron. Callables get
    wrapped so they can be called with Neuron's calling convention."""
    if callable(value) and not isinstance(value, type):
        return PythonCallable(getattr(value, '__name__', str(value)), value)
    if hasattr(value, '__dict__') and hasattr(value, '__name__'):
        # It's a module-like object
        return PythonModuleWrapper(value.__name__, value)
    return value


class PythonCallable:
    """Wraps a Python callable so Neuron's interpreter can call it."""

    def __init__(self, name, func):
        self.name = name
        self.func = func

    def __repr__(self):
        return f"<python {self.name}>"

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def _load_tensor():
    from neuron.stdlib.tensor import create_tensor_module
    return create_tensor_module()

def _load_neural():
    from neuron.stdlib.neural import create_neural_module
    return create_neural_module()

def _load_optim():
    from neuron.stdlib.optim import create_optim_module
    return create_optim_module()

def _load_data():
    from neuron.stdlib.data import create_data_module
    return create_data_module()

def _load_math():
    from neuron.stdlib.math_lib import create_math_module
    return create_math_module()

def _load_io():
    from neuron.stdlib.io_lib import create_io_module
    return create_io_module()

def _load_plot():
    from neuron.stdlib.plot import create_plot_module
    return create_plot_module()

def _load_metrics():
    from neuron.stdlib.metrics import create_metrics_module
    return create_metrics_module()

def _load_preprocess():
    from neuron.stdlib.preprocess import create_preprocess_module
    return create_preprocess_module()

def _load_vision():
    from neuron.stdlib.vision import create_vision_module
    return create_vision_module()

def _load_nlp():
    from neuron.stdlib.nlp import create_nlp_module
    return create_nlp_module()


_MODULE_LOADERS = {
    "tensor": _load_tensor,
    "neural": _load_neural,
    "optim": _load_optim,
    "data": _load_data,
    "math": _load_math,
    "io": _load_io,
    "plot": _load_plot,
    "metrics": _load_metrics,
    "preprocess": _load_preprocess,
    "vision": _load_vision,
    "nlp": _load_nlp,
}
