"""Static semantic passes for Neuron source code."""

from neuron import ast_nodes as ast
from neuron.errors import NeuronCompileError
from neuron.quantum import QuantumCircuitAnalyzer


class StaticAnalyzer:
    """Checks statically declared tensor shapes and infers known tensor shapes."""

    _TENSOR_CONSTRUCTORS = {"random", "rand", "zeros", "ones"}

    def __init__(self, source="", filename="<stdin>"):
        self.source_lines = source.splitlines()
        self.filename = filename
        self.tensor_shapes = {}
        self.quantum_analyzer = QuantumCircuitAnalyzer(source, filename)

    def analyze(self, program):
        self._analyze_statements(program.statements, self.tensor_shapes)
        return program

    def _analyze_statements(self, statements, shapes):
        for statement in statements:
            if isinstance(statement, ast.TypedAssignment):
                declared_shape = tuple(statement.type_node.dimensions)
                inferred_shape = self._infer_shape(statement.value, shapes)
                if inferred_shape is not None and inferred_shape != declared_shape:
                    self._error(
                        statement,
                        "Tensor shape mismatch for '{}': declared {}, inferred {}".format(
                            statement.target, self._format_shape(declared_shape),
                            self._format_shape(inferred_shape)
                        )
                    )
                shapes[statement.target] = declared_shape
            elif isinstance(statement, ast.QuantumCircuitDef):
                self.quantum_analyzer.analyze(statement)
            elif isinstance(statement, (
                    ast.QuantumQubitDeclaration, ast.QuantumTensorDeclaration)):
                self._error(statement, "Qubit declarations must appear inside a quantum circuit")
            elif isinstance(statement, ast.Assignment):
                inferred_shape = self._infer_shape(statement.value, shapes)
                if inferred_shape is None:
                    shapes.pop(statement.target, None)
                else:
                    shapes[statement.target] = inferred_shape
            elif isinstance(statement, ast.FunctionDef):
                self._analyze_statements(statement.body, dict(shapes))
            elif isinstance(statement, (ast.EntityDef, ast.BrainDef)):
                for member in statement.body:
                    if isinstance(member, ast.FunctionDef):
                        self._analyze_statements(member.body, dict(shapes))
            elif isinstance(statement, ast.IfStatement):
                self._analyze_statements(statement.body, dict(shapes))
                for _, body in statement.elif_clauses:
                    self._analyze_statements(body, dict(shapes))
                if statement.else_body:
                    self._analyze_statements(statement.else_body, dict(shapes))
            elif isinstance(statement, (ast.WhileLoop, ast.ForLoop)):
                self._analyze_statements(statement.body, dict(shapes))
            elif isinstance(statement, ast.TryCatch):
                self._analyze_statements(statement.body, dict(shapes))
                for _, _, body in statement.rescue_clauses:
                    self._analyze_statements(body, dict(shapes))

    def _infer_shape(self, expression, shapes):
        if isinstance(expression, ast.Identifier):
            return shapes.get(expression.name)

        if isinstance(expression, ast.FunctionCall):
            callee = expression.callee
            if isinstance(callee, ast.PropertyAccess):
                if (isinstance(callee.object, ast.Identifier)
                        and callee.object.name == "tensor"
                        and callee.property in self._TENSOR_CONSTRUCTORS):
                    dimensions = self._literal_dimensions(expression.args)
                    return tuple(dimensions) if dimensions is not None else None
                if callee.property == "matmul" and expression.args:
                    left_shape = self._infer_shape(callee.object, shapes)
                    right_shape = self._infer_shape(expression.args[0], shapes)
                    if left_shape is not None and right_shape is not None:
                        if len(left_shape) != 2 or len(right_shape) != 2:
                            self._error(
                                expression,
                                "matmul currently requires two rank-2 tensors; got {} and {}".format(
                                    self._format_shape(left_shape),
                                    self._format_shape(right_shape)
                                )
                            )
                        if left_shape[1] != right_shape[0]:
                            self._error(
                                expression,
                                "matmul shape mismatch: {} cannot be multiplied by {}".format(
                                    self._format_shape(left_shape),
                                    self._format_shape(right_shape)
                                )
                            )
                        return (left_shape[0], right_shape[1])
                    return None
                if callee.property == "transpose":
                    shape = self._infer_shape(callee.object, shapes)
                    if shape is not None and len(shape) == 2:
                        return (shape[1], shape[0])

        if isinstance(expression, ast.BinaryOp):
            left = self._infer_shape(expression.left, shapes)
            right = self._infer_shape(expression.right, shapes)
            if left is None or right is None:
                return None
            if expression.op in ("+", "-", "*", "/", "//", "%"):
                return self._broadcast_shape(left, right, expression)

        return None

    def _literal_dimensions(self, arguments):
        dimensions = []
        for argument in arguments:
            if not isinstance(argument, ast.NumberLiteral) or not isinstance(argument.value, int):
                return None
            if argument.value < 0:
                return None
            dimensions.append(argument.value)
        return dimensions or None

    def _broadcast_shape(self, left, right, node):
        result = []
        for left_dim, right_dim in zip(reversed(left), reversed(right)):
            if left_dim != right_dim and left_dim != 1 and right_dim != 1:
                self._error(
                    node,
                    "Tensor shape mismatch: {} and {} cannot be broadcast".format(
                        self._format_shape(left), self._format_shape(right)
                    )
                )
            result.append(max(left_dim, right_dim))
        longer = left if len(left) > len(right) else right
        result.extend(reversed(longer[:abs(len(left) - len(right))]))
        return tuple(reversed(result))

    def _error(self, node, message):
        source_line = None
        if node.line and 0 < node.line <= len(self.source_lines):
            source_line = self.source_lines[node.line - 1]
        raise NeuronCompileError(
            message, line=node.line, column=node.column,
            file=self.filename, source_line=source_line
        )

    @staticmethod
    def _format_shape(shape):
        return "[" + ", ".join(str(dimension) for dimension in shape) + "]"
