"""Static quantum-circuit checks and OpenQASM 3 code generation."""

from neuron import ast_nodes as ast
from neuron.errors import NeuronCompileError


class QuantumCircuitAnalyzer:
    """Checks qubit ownership, gate operands, and measurement lifetimes."""

    _SINGLE_QUBIT_GATES = {"h", "x", "y", "z", "s", "t"}
    _TWO_QUBIT_GATES = {"cx", "cz", "swap"}
    _ROTATION_GATES = {"rx", "ry", "rz"}

    def __init__(self, source="", filename="<stdin>"):
        self.source_lines = source.splitlines()
        self.filename = filename

    def analyze(self, circuit):
        qubits = {}
        registers = {}
        declared_names = set()
        for statement in circuit.body:
            if isinstance(statement, ast.QuantumQubitDeclaration):
                if statement.name in declared_names:
                    self._error(statement, "Quantum register '{}' is declared more than once".format(statement.name))
                declared_names.add(statement.name)
                registers[statement.name] = [statement.name]
                self._declare_qubit(statement.name, qubits)
            elif isinstance(statement, ast.QuantumTensorDeclaration):
                if statement.name in declared_names:
                    self._error(statement, "Quantum register '{}' is declared more than once".format(statement.name))
                if statement.size <= 0:
                    self._error(statement, "qtensor size must be greater than zero")
                declared_names.add(statement.name)
                registers[statement.name] = [
                    "{}[{}]".format(statement.name, index)
                    for index in range(statement.size)
                ]
                for qubit_name in registers[statement.name]:
                    self._declare_qubit(qubit_name, qubits)

        if not qubits:
            self._error(circuit, "Quantum circuit '{}' must declare at least one qubit".format(circuit.name))

        for statement in circuit.body:
            if isinstance(statement, ast.GateOperation):
                self._analyze_operation(statement, qubits, registers)
            elif isinstance(statement, ast.DropStatement):
                self._analyze_drop(statement, qubits, registers)
            elif isinstance(statement, (ast.Assignment, ast.MultiAssignment)):
                self._analyze_assignment(statement, qubits, registers)
            elif not isinstance(statement, (
                    ast.QuantumQubitDeclaration, ast.QuantumTensorDeclaration)):
                self._error(statement, "Unsupported statement in quantum circuit")
        return circuit

    @staticmethod
    def _declare_qubit(name, qubits):
        qubits[name] = {
            "index": len(qubits),
            "measured": False,
            "entangled": {name},
        }

    def _analyze_operation(self, operation, qubits, registers):
        name = operation.name
        args = operation.arguments
        if name in self._SINGLE_QUBIT_GATES:
            if len(args) != 1:
                self._error(operation, "Gate '{}' expects one qubit".format(name))
            operands = args
        elif name in self._TWO_QUBIT_GATES:
            if len(args) != 2:
                self._error(operation, "Gate '{}' expects two qubits".format(name))
            operands = args
        elif name in self._ROTATION_GATES:
            if len(args) != 2 or not self._is_numeric_expression(args[0]):
                self._error(operation, "Gate '{}' expects a numeric angle and one qubit".format(name))
            operands = args[1:]
        elif name == "measure":
            if len(args) != 1:
                self._error(operation, "measure() expects one qubit")
            operands = args
        else:
            self._error(operation, "Unsupported quantum gate '{}'".format(name))

        names = [self._qubit_name(arg, qubits, registers, operation) for arg in operands]
        if len(set(names)) != len(names):
            self._error(operation, "A multi-qubit gate requires distinct qubit operands")
        for qubit_name in names:
            if qubits[qubit_name]["measured"]:
                self._error(operation, "Measured qubit '{}' cannot be used again".format(qubit_name))

        if name in self._TWO_QUBIT_GATES:
            left, right = names
            component = (
                qubits[left]["entangled"] | qubits[right]["entangled"]
            )
            for connected in component:
                qubits[connected]["entangled"] = component.copy()
        elif name == "measure":
            qubits[names[0]]["measured"] = True

    def _analyze_drop(self, statement, qubits, registers):
        if isinstance(statement.target, ast.Identifier) and statement.target.name in registers:
            names = registers[statement.target.name]
        else:
            names = [self._qubit_name(statement.target, qubits, registers, statement)]
        for name in names:
            if not qubits[name]["measured"]:
                self._error(
                    statement,
                    "Quantum invariant violation: cannot drop unmeasured qubit '{}'".format(name)
                )

    def _analyze_assignment(self, statement, qubits, registers):
        value = statement.value if isinstance(statement, ast.Assignment) else None
        targets = [statement.target] if isinstance(statement, ast.Assignment) else statement.targets
        values = [value] if value is not None else statement.values
        if any(self._contains_qubit(item, qubits, registers) for item in values):
            self._error(
                statement,
                "No-cloning invariant violation: qubit values cannot be copied or aliased"
            )
        if any(target in qubits or target in registers for target in targets):
            self._error(
                statement,
                "No-cloning invariant violation: an owned qubit cannot be reassigned"
            )
        self._error(statement, "Classical assignments are not supported inside quantum circuits")

    def _contains_qubit(self, expression, qubits, registers):
        if isinstance(expression, ast.Identifier):
            return expression.name in qubits or expression.name in registers
        if isinstance(expression, ast.IndexAccess):
            return (isinstance(expression.object, ast.Identifier)
                    and expression.object.name in registers)
        if isinstance(expression, ast.FunctionCall):
            return (self._contains_qubit(expression.callee, qubits, registers)
                    or any(self._contains_qubit(arg, qubits, registers) for arg in expression.args)
                    or any(self._contains_qubit(arg, qubits, registers)
                           for arg in expression.kwargs.values()))
        if isinstance(expression, ast.PropertyAccess):
            return self._contains_qubit(expression.object, qubits, registers)
        if isinstance(expression, ast.BinaryOp):
            return (self._contains_qubit(expression.left, qubits, registers)
                    or self._contains_qubit(expression.right, qubits, registers))
        if isinstance(expression, ast.ListLiteral):
            return any(self._contains_qubit(item, qubits, registers) for item in expression.elements)
        if isinstance(expression, ast.TupleLiteral):
            return any(self._contains_qubit(item, qubits, registers) for item in expression.elements)
        if isinstance(expression, ast.DictLiteral):
            return any(
                self._contains_qubit(key, qubits, registers)
                or self._contains_qubit(value, qubits, registers)
                for key, value in expression.pairs
            )
        return False

    def _qubit_name(self, expression, qubits, registers, node):
        if isinstance(expression, ast.Identifier) and expression.name in qubits:
            return expression.name
        if (isinstance(expression, ast.IndexAccess)
                and isinstance(expression.object, ast.Identifier)
                and expression.object.name in registers
                and isinstance(expression.index, ast.NumberLiteral)
                and isinstance(expression.index.value, int)):
            name = "{}[{}]".format(expression.object.name, expression.index.value)
            if name in qubits:
                return name
        self._error(node, "Gate operands must be declared qubit names or constant qtensor indices")

    @staticmethod
    def _is_numeric_expression(expression):
        if isinstance(expression, ast.NumberLiteral):
            return isinstance(expression.value, (int, float))
        return (isinstance(expression, ast.UnaryOp)
                and expression.op == "-"
                and isinstance(expression.operand, ast.NumberLiteral))

    def _error(self, node, message):
        source_line = None
        if node.line and 0 < node.line <= len(self.source_lines):
            source_line = self.source_lines[node.line - 1]
        raise NeuronCompileError(
            message, line=node.line, column=node.column,
            file=self.filename, source_line=source_line
        )


class OpenQASM3Backend:
    """Lower one statically checked Neuron quantum circuit to OpenQASM 3."""

    def __init__(self, source="", filename="<stdin>"):
        self.analyzer = QuantumCircuitAnalyzer(source, filename)

    def compile(self, program):
        circuits = [statement for statement in program.statements
                    if isinstance(statement, ast.QuantumCircuitDef)]
        if len(circuits) != 1 or len(program.statements) != 1:
            raise NeuronCompileError(
                "OpenQASM 3 output requires a source file containing exactly one top-level quantum circuit",
                file=self.analyzer.filename
            )
        circuit = self.analyzer.analyze(circuits[0])
        qubit_names = []
        for statement in circuit.body:
            if isinstance(statement, ast.QuantumQubitDeclaration):
                qubit_names.append(statement.name)
            elif isinstance(statement, ast.QuantumTensorDeclaration):
                qubit_names.extend(
                    "{}[{}]".format(statement.name, index)
                    for index in range(statement.size)
                )
        lines = [
            "OPENQASM 3.0;",
            'include "stdgates.inc";',
            "qubit[{}] q;".format(len(qubit_names)),
            "bit[{}] c;".format(len(qubit_names)),
        ]
        indexes = {name: index for index, name in enumerate(qubit_names)}
        for statement in circuit.body:
            if isinstance(statement, ast.GateOperation):
                lines.append(self._emit_operation(statement, indexes))
        return "\n".join(lines) + "\n"

    def _emit_operation(self, operation, indexes):
        name = operation.name
        if name == "measure":
            index = indexes[self._operand_key(operation.arguments[0])]
            return "c[{}] = measure q[{}];".format(index, index)
        if name in QuantumCircuitAnalyzer._ROTATION_GATES:
            angle = self._format_angle(operation.arguments[0])
            index = indexes[self._operand_key(operation.arguments[1])]
            return "{}({}) q[{}];".format(name, angle, index)
        qubits = ", ".join(
            "q[{}]".format(indexes[self._operand_key(argument)])
            for argument in operation.arguments
        )
        return "{} {};".format(name, qubits)

    @staticmethod
    def _operand_key(expression):
        if isinstance(expression, ast.Identifier):
            return expression.name
        return "{}[{}]".format(expression.object.name, expression.index.value)

    @staticmethod
    def _format_angle(expression):
        if isinstance(expression, ast.UnaryOp):
            return "-{}".format(repr(expression.operand.value))
        return repr(expression.value)
