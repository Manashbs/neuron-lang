"""
Neuron Interpreter
==================
Tree-walking interpreter that executes Neuron AST nodes.
"""

from neuron import ast_nodes as ast
from neuron.environment import (
    Environment, NeuronFunction, NeuronEntity, NeuronBrain,
    NeuronInstance, BoundMethod, NeuronModule
)
from neuron.errors import *
from neuron.lexer import Lexer
from neuron.parser import Parser
import sys


class Interpreter:
    """Tree-walking interpreter for Neuron."""

    def __init__(self):
        self.global_env = Environment(name="<global>")
        self._setup_builtins()
        self.call_stack = []

    def _setup_builtins(self):
        env = self.global_env
        env.set("show", BuiltinFunction("show", self._builtin_show))
        env.set("ask", BuiltinFunction("ask", self._builtin_ask))
        env.set("len", BuiltinFunction("len", self._builtin_len))
        env.set("upto", BuiltinFunction("upto", self._builtin_upto))
        env.set("type", BuiltinFunction("type", self._builtin_type))
        env.set("str", BuiltinFunction("str", self._builtin_str))
        env.set("int", BuiltinFunction("int", self._builtin_int))
        env.set("float", BuiltinFunction("float", self._builtin_float))
        env.set("list", BuiltinFunction("list", self._builtin_list))
        env.set("dict", BuiltinFunction("dict", self._builtin_dict))
        env.set("abs", BuiltinFunction("abs", lambda a, **kw: abs(a[0])))
        env.set("min", BuiltinFunction("min", lambda a, **kw: min(*a) if len(a) > 1 else min(a[0])))
        env.set("max", BuiltinFunction("max", lambda a, **kw: max(*a) if len(a) > 1 else max(a[0])))
        env.set("sum", BuiltinFunction("sum", lambda a, **kw: sum(a[0])))
        env.set("round", BuiltinFunction("round", lambda a, **kw: round(a[0], a[1] if len(a) > 1 else 0)))
        env.set("sorted", BuiltinFunction("sorted", lambda a, **kw: sorted(a[0])))
        env.set("reversed", BuiltinFunction("reversed", lambda a, **kw: list(reversed(a[0]))))
        env.set("enumerate", BuiltinFunction("enumerate", lambda a, **kw: list(enumerate(a[0]))))
        env.set("zip", BuiltinFunction("zip", lambda a, **kw: [list(x) for x in zip(*a)]))
        env.set("map", BuiltinFunction("map", self._builtin_map))
        env.set("filter", BuiltinFunction("filter", self._builtin_filter))
        env.set("isinstance", BuiltinFunction("isinstance", self._builtin_isinstance))
        env.set("hasattr", BuiltinFunction("hasattr", self._builtin_hasattr))
        env.set("open", BuiltinFunction("open", self._builtin_open))
        env.set("exit", BuiltinFunction("exit", lambda a, **kw: sys.exit(a[0] if a else 0)))
        env.set("quit", BuiltinFunction("quit", lambda a, **kw: sys.exit(a[0] if a else 0)))

    # ===== Built-in implementations =====
    def _builtin_show(self, args, **kwargs):
        print(*args, end=kwargs.get("end", "\n"))
        return None

    def _builtin_ask(self, args, **kwargs):
        prompt = args[0] if args else ""
        return input(prompt)

    def _builtin_len(self, args, **kwargs):
        return len(args[0])

    def _builtin_upto(self, args, **kwargs):
        if len(args) == 1:
            return list(range(args[0]))
        elif len(args) == 2:
            return list(range(args[0], args[1]))
        elif len(args) == 3:
            return list(range(args[0], args[1], args[2]))
        raise NeuronRuntimeError("upto() takes 1 to 3 arguments")

    def _builtin_type(self, args, **kwargs):
        v = args[0]
        if isinstance(v, bool): return "boolean"
        if isinstance(v, int): return "integer"
        if isinstance(v, float): return "float"
        if isinstance(v, str): return "string"
        if isinstance(v, list): return "list"
        if isinstance(v, dict): return "dict"
        if v is None: return "nothing"
        if isinstance(v, NeuronInstance): return v.entity.name
        if isinstance(v, NeuronFunction): return "function"
        if isinstance(v, BuiltinFunction): return "builtin"
        return type(v).__name__

    def _builtin_str(self, args, **kwargs):
        return self._to_string(args[0])

    def _builtin_int(self, args, **kwargs):
        return int(args[0])

    def _builtin_float(self, args, **kwargs):
        return float(args[0])

    def _builtin_list(self, args, **kwargs):
        if not args: return []
        return list(args[0])

    def _builtin_dict(self, args, **kwargs):
        if not args: return {}
        return dict(args[0])

    def _builtin_map(self, args, **kwargs):
        fn, iterable = args[0], args[1]
        return [self._call_function(fn, [item]) for item in iterable]

    def _builtin_filter(self, args, **kwargs):
        fn, iterable = args[0], args[1]
        return [item for item in iterable if self._call_function(fn, [item])]

    def _builtin_isinstance(self, args, **kwargs):
        obj, entity = args
        if not isinstance(obj, NeuronInstance): return False
        e = obj.entity
        while e:
            if e is entity or e.name == (entity.name if isinstance(entity, NeuronEntity) else entity):
                return True
            e = e.parent
        return False

    def _builtin_hasattr(self, args, **kwargs):
        obj, name = args
        if isinstance(obj, NeuronInstance):
            return name in obj.properties or obj.entity.find_method(name) is not None
        return hasattr(obj, name)

    def _builtin_open(self, args, **kwargs):
        return open(args[0], args[1] if len(args) > 1 else 'r')

    # ===== Core execution =====

    def execute(self, source, filename="<stdin>"):
        lexer = Lexer(source, filename)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source, filename)
        program = parser.parse()
        return self.visit(program, self.global_env)

    def execute_file(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        return self.execute(source, filepath)

    def visit(self, node, env):
        method = f"visit_{type(node).__name__}"
        visitor = getattr(self, method, None)
        if visitor:
            return visitor(node, env)
        raise NeuronRuntimeError(f"Unknown AST node: {type(node).__name__}", line=getattr(node, 'line', None))

    # ===== Visit methods =====

    def visit_Program(self, node, env):
        result = None
        for stmt in node.statements:
            result = self.visit(stmt, env)
        return result

    def visit_ExpressionStatement(self, node, env):
        return self.visit(node.expression, env)

    def visit_NumberLiteral(self, node, env):
        return node.value

    def visit_StringLiteral(self, node, env):
        return node.value

    def visit_FStringLiteral(self, node, env):
        parts = []
        for ptype, pval in node.parts:
            if ptype == 'text':
                parts.append(pval)
            else:
                lexer = Lexer(pval, "<fstring>")
                tokens = lexer.tokenize()
                parser = Parser(tokens, pval, "<fstring>")
                expr_ast = parser.parse()
                val = self.visit(expr_ast, env)
                parts.append(self._to_string(val))
        return "".join(parts)

    def visit_BooleanLiteral(self, node, env):
        return node.value

    def visit_NothingLiteral(self, node, env):
        return None

    def visit_ListLiteral(self, node, env):
        return [self.visit(el, env) for el in node.elements]

    def visit_DictLiteral(self, node, env):
        return {self.visit(k, env): self.visit(v, env) for k, v in node.pairs}

    def visit_TupleLiteral(self, node, env):
        return tuple(self.visit(el, env) for el in node.elements)

    def visit_Identifier(self, node, env):
        return env.get(node.name, line=node.line, column=node.column)

    def visit_MeExpression(self, node, env):
        return env.get("me", line=node.line, column=node.column)

    def visit_Assignment(self, node, env):
        value = self.visit(node.value, env)
        env.set(node.target, value)
        return value

    def visit_AugmentedAssignment(self, node, env):
        current = env.get(node.target, line=node.line)
        rhs = self.visit(node.value, env)
        ops = {'+': lambda a, b: a + b, '-': lambda a, b: a - b,
               '*': lambda a, b: a * b, '/': lambda a, b: a / b}
        result = ops[node.op](current, rhs)
        env.set(node.target, result)
        return result

    def visit_MultiAssignment(self, node, env):
        values = [self.visit(v, env) for v in node.values]
        if len(values) == 1 and hasattr(values[0], '__iter__') and not isinstance(values[0], str):
            values = list(values[0])
        for name, val in zip(node.targets, values):
            env.set(name, val)

    def visit_AttributeAssignment(self, node, env):
        obj = self.visit(node.object, env)
        value = self.visit(node.value, env)
        if isinstance(obj, NeuronInstance):
            obj.set_property(node.attribute, value)
        elif isinstance(obj, NeuronModule):
            obj.attributes[node.attribute] = value
        else:
            raise NeuronRuntimeError(f"Cannot set attribute on {type(obj).__name__}", line=node.line)
        return value

    def visit_IndexAssignment(self, node, env):
        obj = self.visit(node.object, env)
        idx = self.visit(node.index, env)
        value = self.visit(node.value, env)
        obj[idx] = value
        return value

    def visit_BinaryOp(self, node, env):
        left = self.visit(node.left, env)
        right = self.visit(node.right, env)
        ops = {
            '+': lambda: left + right, '-': lambda: left - right,
            '*': lambda: left * right, '**': lambda: left ** right,
            '%': lambda: left % right, '//': lambda: left // right,
        }
        if node.op == '/':
            if right == 0:
                raise NeuronZeroDivisionError("Division by zero", line=node.line)
            return left / right
        if node.op in ops:
            try:
                return ops[node.op]()
            except TypeError:
                raise NeuronTypeError(f"Cannot apply '{node.op}' to {type(left).__name__} and {type(right).__name__}", line=node.line)
        raise NeuronRuntimeError(f"Unknown operator: {node.op}", line=node.line)

    def visit_UnaryOp(self, node, env):
        operand = self.visit(node.operand, env)
        if node.op == '-': return -operand
        if node.op == 'nah': return not operand
        raise NeuronRuntimeError(f"Unknown unary operator: {node.op}", line=node.line)

    def visit_ComparisonOp(self, node, env):
        left = self.visit(node.left, env)
        right = self.visit(node.right, env)
        ops = {'==': lambda: left == right, '!=': lambda: left != right,
               '<': lambda: left < right, '>': lambda: left > right,
               '<=': lambda: left <= right, '>=': lambda: left >= right,
               'within': lambda: left in right}
        return ops[node.op]()

    def visit_LogicalOp(self, node, env):
        left = self.visit(node.left, env)
        if node.op == 'also':
            return left and self.visit(node.right, env)
        if node.op == 'either':
            return left or self.visit(node.right, env)

    def visit_FunctionDef(self, node, env):
        func = NeuronFunction(node.name, node.params, node.defaults, node.body, env)
        env.set(node.name, func)
        return func

    def visit_FunctionCall(self, node, env):
        callee = self.visit(node.callee, env)
        args = [self.visit(a, env) for a in node.args]
        kwargs = {k: self.visit(v, env) for k, v in node.kwargs.items()}
        return self._call_function(callee, args, kwargs, node)

    def _call_function(self, callee, args, kwargs=None, node=None):
        kwargs = kwargs or {}
        line = getattr(node, 'line', None)

        if isinstance(callee, BuiltinFunction):
            return callee.func(args, **kwargs)

        if isinstance(callee, NeuronFunction):
            func_env = callee.closure.child(callee.name)
            params = callee.params
            # Bind parameters
            for i, param in enumerate(params):
                if i < len(args):
                    func_env.set(param, args[i])
                elif param in kwargs:
                    func_env.set(param, kwargs[param])
                elif param in callee.defaults:
                    default_val = self.visit(callee.defaults[param], callee.closure)
                    func_env.set(param, default_val)
                else:
                    raise NeuronRuntimeError(f"Missing argument '{param}' in call to {callee.name}()", line=line)

            self.call_stack.append({"function": callee.name, "line": line, "file": "<stdin>"})
            try:
                for stmt in callee.body:
                    self.visit(stmt, func_env)
            except NeuronReturnSignal as ret:
                self.call_stack.pop()
                return ret.value
            self.call_stack.pop()
            return None

        if isinstance(callee, BoundMethod):
            return self._call_function(callee.function, [callee.instance] + args, kwargs, node)

        if isinstance(callee, (NeuronEntity, NeuronBrain)):
            instance = NeuronInstance(callee)
            create_method = callee.find_method("create")
            if create_method:
                self._call_function(create_method, [instance] + args, kwargs, node)
            return instance

        # Python interop: handle real Python callables and classes
        from neuron.stdlib import PythonCallable, PythonModuleWrapper
        if isinstance(callee, PythonCallable):
            try:
                return callee.func(*args, **kwargs)
            except TypeError:
                # If kwargs don't match, try positional only
                return callee.func(*args)

        # Python class instantiation (e.g., numpy.ndarray, pd.DataFrame)
        if isinstance(callee, type):
            try:
                return callee(*args, **kwargs)
            except TypeError:
                return callee(*args)

        # Any other callable Python object
        if callable(callee):
            try:
                return callee(*args, **kwargs)
            except TypeError:
                return callee(*args)

        raise NeuronRuntimeError(f"'{self._to_string(callee)}' is not callable", line=line)

    def visit_PropertyAccess(self, node, env):
        obj = self.visit(node.object, env)
        prop = node.property

        if isinstance(obj, NeuronInstance):
            return obj.get_property(prop, line=node.line, column=node.column)
        if isinstance(obj, NeuronModule):
            return obj.get(prop, line=node.line)
        # Python interop: handle wrapped Python modules
        from neuron.stdlib import PythonModuleWrapper
        if isinstance(obj, PythonModuleWrapper):
            return obj.get(prop, line=node.line)
        if isinstance(obj, NeuronEntity) or isinstance(obj, NeuronBrain):
            m = obj.find_method(prop)
            if m: return m
            if prop in obj.class_vars: return obj.class_vars[prop]

        # Built-in methods on Python types
        if isinstance(obj, list):
            list_methods = {
                'append': BuiltinFunction('append', lambda a, **kw: obj.append(a[0])),
                'pop': BuiltinFunction('pop', lambda a, **kw: obj.pop(a[0] if a else -1)),
                'insert': BuiltinFunction('insert', lambda a, **kw: obj.insert(a[0], a[1])),
                'remove': BuiltinFunction('remove', lambda a, **kw: obj.remove(a[0])),
                'index': BuiltinFunction('index', lambda a, **kw: obj.index(a[0])),
                'count': BuiltinFunction('count', lambda a, **kw: obj.count(a[0])),
                'sort': BuiltinFunction('sort', lambda a, **kw: obj.sort()),
                'reverse': BuiltinFunction('reverse', lambda a, **kw: obj.reverse()),
                'extend': BuiltinFunction('extend', lambda a, **kw: obj.extend(a[0])),
                'copy': BuiltinFunction('copy', lambda a, **kw: obj.copy()),
                'clear': BuiltinFunction('clear', lambda a, **kw: obj.clear()),
            }
            if prop == 'length': return len(obj)
            if prop in list_methods: return list_methods[prop]

        if isinstance(obj, str):
            str_methods = {
                'upper': BuiltinFunction('upper', lambda a, **kw: obj.upper()),
                'lower': BuiltinFunction('lower', lambda a, **kw: obj.lower()),
                'strip': BuiltinFunction('strip', lambda a, **kw: obj.strip()),
                'split': BuiltinFunction('split', lambda a, **kw: obj.split(a[0] if a else None)),
                'join': BuiltinFunction('join', lambda a, **kw: obj.join(a[0])),
                'replace': BuiltinFunction('replace', lambda a, **kw: obj.replace(a[0], a[1])),
                'startswith': BuiltinFunction('startswith', lambda a, **kw: obj.startswith(a[0])),
                'endswith': BuiltinFunction('endswith', lambda a, **kw: obj.endswith(a[0])),
                'find': BuiltinFunction('find', lambda a, **kw: obj.find(a[0])),
                'format': BuiltinFunction('format', lambda a, **kw: obj.format(*a)),
                'contains': BuiltinFunction('contains', lambda a, **kw: a[0] in obj),
            }
            if prop == 'length': return len(obj)
            if prop in str_methods: return str_methods[prop]

        if isinstance(obj, dict):
            dict_methods = {
                'keys': BuiltinFunction('keys', lambda a, **kw: list(obj.keys())),
                'values': BuiltinFunction('values', lambda a, **kw: list(obj.values())),
                'items': BuiltinFunction('items', lambda a, **kw: [list(x) for x in obj.items()]),
                'get': BuiltinFunction('get', lambda a, **kw: obj.get(a[0], a[1] if len(a) > 1 else None)),
                'has': BuiltinFunction('has', lambda a, **kw: a[0] in obj),
                'update': BuiltinFunction('update', lambda a, **kw: obj.update(a[0])),
                'pop': BuiltinFunction('pop', lambda a, **kw: obj.pop(a[0], a[1] if len(a) > 1 else None)),
            }
            if prop == 'length': return len(obj)
            if prop in dict_methods: return dict_methods[prop]

        # Generic fallback: try Python attribute access
        if hasattr(obj, prop):
            attr = getattr(obj, prop)
            if callable(attr):
                return BuiltinFunction(prop, lambda a, _m=attr, **kw: _m(*a, **kw))
            return attr

        raise NeuronAttributeError(f"'{self._type_name(obj)}' has no attribute '{prop}'", line=node.line)

    def visit_IndexAccess(self, node, env):
        obj = self.visit(node.object, env)
        idx = self.visit(node.index, env)
        try:
            return obj[idx]
        except (IndexError, KeyError) as e:
            raise NeuronIndexError(str(e), line=node.line)

    def visit_SliceAccess(self, node, env):
        obj = self.visit(node.object, env)
        start = self.visit(node.start, env) if node.start else None
        end = self.visit(node.end, env) if node.end else None
        step = self.visit(node.step, env) if node.step else None
        return obj[start:end:step]

    def visit_PipelineOp(self, node, env):
        left_val = self.visit(node.left, env)
        # Right side should be a function or function call
        if isinstance(node.right, ast.FunctionCall):
            callee = self.visit(node.right.callee, env)
            args = [left_val] + [self.visit(a, env) for a in node.right.args]
            kwargs = {k: self.visit(v, env) for k, v in node.right.kwargs.items()}
            return self._call_function(callee, args, kwargs, node)
        else:
            callee = self.visit(node.right, env)
            return self._call_function(callee, [left_val], {}, node)

    def visit_QuickFunction(self, node, env):
        return NeuronFunction("<quick>", node.params, {}, [ast.ReturnStatement(node.body)], env)

    def visit_IfStatement(self, node, env):
        if self.visit(node.condition, env):
            for stmt in node.body:
                self.visit(stmt, env)
            return
        for cond, body in node.elif_clauses:
            if self.visit(cond, env):
                for stmt in body:
                    self.visit(stmt, env)
                return
        if node.else_body:
            for stmt in node.else_body:
                self.visit(stmt, env)

    def visit_WhileLoop(self, node, env):
        while self.visit(node.condition, env):
            try:
                for stmt in node.body:
                    self.visit(stmt, env)
            except NeuronBreakSignal:
                break
            except NeuronContinueSignal:
                continue

    def visit_ForLoop(self, node, env):
        iterable = self.visit(node.iterable, env)
        for item in iterable:
            env.set(node.variable, item)
            try:
                for stmt in node.body:
                    self.visit(stmt, env)
            except NeuronBreakSignal:
                break
            except NeuronContinueSignal:
                continue

    def visit_ReturnStatement(self, node, env):
        value = self.visit(node.value, env) if node.value else None
        raise NeuronReturnSignal(value)

    def visit_BreakStatement(self, node, env):
        raise NeuronBreakSignal()

    def visit_ContinueStatement(self, node, env):
        raise NeuronContinueSignal()

    def visit_PassStatement(self, node, env):
        pass

    def visit_EntityDef(self, node, env):
        parent = None
        if node.parent:
            parent = env.get(node.parent, line=node.line)
        methods = {}
        class_vars = {}
        entity_env = env.child(node.name)
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef):
                func = NeuronFunction(stmt.name, stmt.params, stmt.defaults, stmt.body, entity_env)
                methods[stmt.name] = func
            elif isinstance(stmt, ast.Assignment):
                class_vars[stmt.target] = self.visit(stmt.value, entity_env)
            elif isinstance(stmt, ast.ExpressionStatement):
                self.visit(stmt, entity_env)
        entity = NeuronEntity(node.name, parent, methods, class_vars)
        env.set(node.name, entity)
        return entity

    def visit_BrainDef(self, node, env):
        parent = None
        if node.parent:
            parent = env.get(node.parent, line=node.line)
        methods = {}
        class_vars = {}
        brain_env = env.child(node.name)
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef):
                func = NeuronFunction(stmt.name, stmt.params, stmt.defaults, stmt.body, brain_env)
                methods[stmt.name] = func
            elif isinstance(stmt, ast.Assignment):
                class_vars[stmt.target] = self.visit(stmt.value, brain_env)
            elif isinstance(stmt, ast.ExpressionStatement):
                self.visit(stmt, brain_env)
        brain = NeuronBrain(node.name, parent, methods, class_vars)
        env.set(node.name, brain)
        return brain

    def visit_TryCatch(self, node, env):
        try:
            for stmt in node.body:
                self.visit(stmt, env)
        except (NeuronReturnSignal, NeuronBreakSignal, NeuronContinueSignal):
            raise
        except NeuronError as e:
            for error_type, alias, handler in node.rescue_clauses:
                if error_type is None or error_type in e.error_type():
                    if alias:
                        env.set(alias, e.message)
                    for stmt in handler:
                        self.visit(stmt, env)
                    return
            raise
        except Exception as e:
            for error_type, alias, handler in node.rescue_clauses:
                if alias:
                    env.set(alias, str(e))
                for stmt in handler:
                    self.visit(stmt, env)
                return
            raise

    def visit_GrabStatement(self, node, env):
        module = self._load_module(node.module_name, node)
        name = node.alias or node.module_name
        env.set(name, module)

    def _load_module(self, name, node=None):
        from neuron.stdlib import get_module
        module = get_module(name)
        if module:
            return module
        raise NeuronImportError(f"Module '{name}' not found", line=getattr(node, 'line', None))

    def visit_DropStatement(self, node, env):
        if isinstance(node.target, ast.Identifier):
            env.delete(node.target.name, line=node.line)
        else:
            raise NeuronRuntimeError("Can only drop variables", line=node.line)

    def visit_EnsureStatement(self, node, env):
        if not self.visit(node.condition, env):
            msg = self.visit(node.message, env) if node.message else "Assertion failed"
            raise NeuronAssertionError(msg, line=node.line)

    def visit_EmitStatement(self, node, env):
        value = self.visit(node.value, env) if node.value else None
        raise NeuronEmitSignal(value)

    def visit_SharedStatement(self, node, env):
        for name in node.names:
            env.declare_shared(name)

    # ===== Helpers =====

    def _to_string(self, value):
        if value is None: return "nothing"
        if isinstance(value, bool): return "yes" if value else "no"
        if isinstance(value, list): return "[" + ", ".join(self._to_string(v) for v in value) + "]"
        if isinstance(value, dict):
            pairs = [f"{self._to_string(k)}: {self._to_string(v)}" for k, v in value.items()]
            return "{" + ", ".join(pairs) + "}"
        return str(value)

    def _type_name(self, value):
        if value is None: return "nothing"
        if isinstance(value, bool): return "boolean"
        if isinstance(value, int): return "integer"
        if isinstance(value, float): return "float"
        if isinstance(value, str): return "string"
        if isinstance(value, list): return "list"
        if isinstance(value, dict): return "dict"
        if isinstance(value, NeuronInstance): return value.entity.name
        return type(value).__name__


class BuiltinFunction:
    """Wraps a Python function as a Neuron built-in."""
    def __init__(self, name, func):
        self.name = name
        self.func = func
    def __repr__(self):
        return f"<builtin {self.name}>"
