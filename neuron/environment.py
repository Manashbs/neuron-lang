"""
Neuron Environment (Scope Management)
======================================
Manages variable scopes with lexical scoping and parent chains.
"""

from neuron.errors import NeuronNameError


class Environment:
    """A scope/environment for variable bindings."""

    def __init__(self, parent=None, name="<module>"):
        self.parent = parent
        self.name = name
        self.variables = {}
        self.globals = set()  # Variables declared as 'shared'

    def get(self, name, line=None, column=None):
        if name in self.globals and self.parent:
            return self.get_global(name, line, column)
        if name in self.variables:
            return self.variables[name]
        if self.parent:
            return self.parent.get(name, line, column)
        raise NeuronNameError(f"'{name}' is not defined", line=line, column=column)

    def set(self, name, value):
        if name in self.globals and self.parent:
            self.set_global(name, value)
            return
        self.variables[name] = value

    def assign(self, name, value, line=None, column=None):
        """Assign to existing variable, walking up scopes."""
        if name in self.globals and self.parent:
            self.set_global(name, value)
            return
        if name in self.variables:
            self.variables[name] = value
            return
        if self.parent:
            try:
                self.parent.assign(name, value, line, column)
                return
            except NeuronNameError:
                pass
        self.variables[name] = value

    def get_global(self, name, line=None, column=None):
        env = self
        while env.parent:
            env = env.parent
        if name in env.variables:
            return env.variables[name]
        raise NeuronNameError(f"shared variable '{name}' is not defined", line=line, column=column)

    def set_global(self, name, value):
        env = self
        while env.parent:
            env = env.parent
        env.variables[name] = value

    def declare_shared(self, name):
        self.globals.add(name)

    def has(self, name):
        if name in self.variables:
            return True
        if self.parent:
            return self.parent.has(name)
        return False

    def delete(self, name, line=None, column=None):
        if name in self.variables:
            del self.variables[name]
            return
        raise NeuronNameError(f"Cannot drop '{name}': not defined", line=line, column=column)

    def child(self, name="<block>"):
        return Environment(parent=self, name=name)


class NeuronFunction:
    """Represents a user-defined Neuron function."""

    def __init__(self, name, params, defaults, body, closure):
        self.name = name
        self.params = params
        self.defaults = defaults
        self.body = body
        self.closure = closure
        self.is_generator = False

    def __repr__(self):
        return f"<do {self.name}({', '.join(self.params)})>"


class NeuronEntity:
    """Represents a Neuron entity (class) definition."""

    def __init__(self, name, parent=None, methods=None, class_vars=None):
        self.name = name
        self.parent = parent
        self.methods = methods or {}
        self.class_vars = class_vars or {}

    def find_method(self, name):
        if name in self.methods:
            return self.methods[name]
        if self.parent:
            return self.parent.find_method(name)
        return None

    def __repr__(self):
        return f"<entity {self.name}>"


class NeuronBrain(NeuronEntity):
    """Represents a Neuron brain (AI model class)."""

    def __repr__(self):
        return f"<brain {self.name}>"


class NeuronInstance:
    """An instance of a NeuronEntity or NeuronBrain."""

    def __init__(self, entity):
        self.entity = entity
        self.properties = {}

    def get_property(self, name, line=None, column=None):
        if name in self.properties:
            return self.properties[name]
        method = self.entity.find_method(name)
        if method:
            return BoundMethod(self, method)
        if name in self.entity.class_vars:
            return self.entity.class_vars[name]
        from neuron.errors import NeuronAttributeError
        raise NeuronAttributeError(
            f"'{self.entity.name}' has no attribute '{name}'",
            line=line, column=column
        )

    def set_property(self, name, value):
        self.properties[name] = value

    def __repr__(self):
        return f"<{self.entity.name} instance>"


class BoundMethod:
    """A method bound to an instance."""

    def __init__(self, instance, function):
        self.instance = instance
        self.function = function

    def __repr__(self):
        return f"<bound method {self.function.name} of {self.instance}>"


class NeuronModule:
    """Represents an imported module."""

    def __init__(self, name, attributes=None):
        self.name = name
        self.attributes = attributes or {}

    def get(self, name, line=None, column=None):
        if name in self.attributes:
            return self.attributes[name]
        from neuron.errors import NeuronAttributeError
        raise NeuronAttributeError(f"Module '{self.name}' has no attribute '{name}'", line=line, column=column)

    def __repr__(self):
        return f"<module {self.name}>"
