"""
Neuron AST Node Definitions
============================
All Abstract Syntax Tree node types for the Neuron language.
Each node carries source location info for error reporting.
"""


class ASTNode:
    """Base class for all AST nodes."""

    def __init__(self, line=None, column=None):
        self.line = line
        self.column = column

    def accept(self, visitor):
        method_name = f"visit_{type(self).__name__}"
        visitor_method = getattr(visitor, method_name, None)
        if visitor_method:
            return visitor_method(self)
        return visitor.generic_visit(self)


# ===== PROGRAM =====

class Program(ASTNode):
    """Root node: a list of statements."""

    def __init__(self, statements, line=None, column=None):
        super().__init__(line, column)
        self.statements = statements


# ===== STATEMENTS =====

class ExpressionStatement(ASTNode):
    """A standalone expression used as a statement."""

    def __init__(self, expression, line=None, column=None):
        super().__init__(line, column)
        self.expression = expression


class Assignment(ASTNode):
    """Variable assignment: x = expr"""

    def __init__(self, target, value, line=None, column=None):
        super().__init__(line, column)
        self.target = target
        self.value = value


class TensorType(ASTNode):
    """A statically declared tensor shape, such as tensor[32, 8]."""

    def __init__(self, dimensions, line=None, column=None):
        super().__init__(line, column)
        self.dimensions = dimensions


class TypedAssignment(ASTNode):
    """A variable assignment with a declared type."""

    def __init__(self, target, type_node, value, line=None, column=None):
        super().__init__(line, column)
        self.target = target
        self.type_node = type_node
        self.value = value


class QuantumQubitDeclaration(ASTNode):
    """Declare one owned qubit inside a quantum circuit."""

    def __init__(self, name, line=None, column=None):
        super().__init__(line, column)
        self.name = name


class QuantumTensorDeclaration(ASTNode):
    """Declare a statically sized register of owned qubits."""

    def __init__(self, name, size, line=None, column=None):
        super().__init__(line, column)
        self.name = name
        self.size = size


class GateOperation(ASTNode):
    """A quantum gate or measurement operation."""

    def __init__(self, name, arguments, line=None, column=None):
        super().__init__(line, column)
        self.name = name
        self.arguments = arguments


class QuantumCircuitDef(ASTNode):
    """Quantum circuit block lowered by a quantum compiler backend."""

    def __init__(self, name, body, line=None, column=None):
        super().__init__(line, column)
        self.name = name
        self.body = body


class AugmentedAssignment(ASTNode):
    """Augmented assignment: x += expr, x -= expr, etc."""

    def __init__(self, target, op, value, line=None, column=None):
        super().__init__(line, column)
        self.target = target
        self.op = op  # '+', '-', '*', '/'
        self.value = value


class MultiAssignment(ASTNode):
    """Multiple assignment: a, b = expr1, expr2"""

    def __init__(self, targets, values, line=None, column=None):
        super().__init__(line, column)
        self.targets = targets
        self.values = values


class FunctionDef(ASTNode):
    """Function definition: do name(params): body"""

    def __init__(self, name, params, defaults, body, line=None, column=None):
        super().__init__(line, column)
        self.name = name
        self.params = params        # List of parameter names
        self.defaults = defaults    # Dict of param_name -> default_value_expr
        self.body = body            # List of statements


class EntityDef(ASTNode):
    """Entity (class) definition: entity Name [from Parent]: body"""

    def __init__(self, name, parent, body, line=None, column=None):
        super().__init__(line, column)
        self.name = name
        self.parent = parent   # Parent class name (string) or None
        self.body = body       # List of statements (methods, assignments)


class BrainDef(ASTNode):
    """Brain (AI model) definition: brain Name [from Parent]: body"""

    def __init__(self, name, parent, body, line=None, column=None):
        super().__init__(line, column)
        self.name = name
        self.parent = parent
        self.body = body


class ReturnStatement(ASTNode):
    """Return statement: give expr"""

    def __init__(self, value=None, line=None, column=None):
        super().__init__(line, column)
        self.value = value


class IfStatement(ASTNode):
    """If statement: when condition: body [orwhen: ...] [otherwise: ...]"""

    def __init__(self, condition, body, elif_clauses=None, else_body=None, line=None, column=None):
        super().__init__(line, column)
        self.condition = condition
        self.body = body
        self.elif_clauses = elif_clauses or []  # List of (condition, body) tuples
        self.else_body = else_body              # List of statements or None


class WhileLoop(ASTNode):
    """While loop: repeat condition: body"""

    def __init__(self, condition, body, line=None, column=None):
        super().__init__(line, column)
        self.condition = condition
        self.body = body


class ForLoop(ASTNode):
    """For loop: each var within iterable: body"""

    def __init__(self, variable, iterable, body, line=None, column=None):
        super().__init__(line, column)
        self.variable = variable   # Variable name (string)
        self.iterable = iterable   # Expression
        self.body = body


class BreakStatement(ASTNode):
    """Break: stop"""
    pass


class ContinueStatement(ASTNode):
    """Continue: skip"""
    pass


class PassStatement(ASTNode):
    """Pass/no-op: ..."""
    pass


class TryCatch(ASTNode):
    """Try/catch: safe: body rescue ErrorType as name: handler"""

    def __init__(self, body, rescue_clauses=None, line=None, column=None):
        super().__init__(line, column)
        self.body = body
        self.rescue_clauses = rescue_clauses or []
        # Each clause: (error_type_name, alias_name, handler_body)


class GrabStatement(ASTNode):
    """Import: grab module_name"""

    def __init__(self, module_name, alias=None, items=None, line=None, column=None):
        super().__init__(line, column)
        self.module_name = module_name
        self.alias = alias      # grab module as alias
        self.items = items       # grab item from module


class DropStatement(ASTNode):
    """Delete: drop variable"""

    def __init__(self, target, line=None, column=None):
        super().__init__(line, column)
        self.target = target


class EnsureStatement(ASTNode):
    """Assert: ensure condition, message"""

    def __init__(self, condition, message=None, line=None, column=None):
        super().__init__(line, column)
        self.condition = condition
        self.message = message


class EmitStatement(ASTNode):
    """Yield: emit value"""

    def __init__(self, value=None, line=None, column=None):
        super().__init__(line, column)
        self.value = value


class SharedStatement(ASTNode):
    """Global: shared var_name"""

    def __init__(self, names, line=None, column=None):
        super().__init__(line, column)
        self.names = names  # List of variable names


# ===== EXPRESSIONS =====

class NumberLiteral(ASTNode):
    """Numeric literal: 42, 3.14"""

    def __init__(self, value, line=None, column=None):
        super().__init__(line, column)
        self.value = value


class StringLiteral(ASTNode):
    """String literal: "hello" """

    def __init__(self, value, line=None, column=None):
        super().__init__(line, column)
        self.value = value


class FStringLiteral(ASTNode):
    """Interpolated string: "hello {name}" """

    def __init__(self, parts, line=None, column=None):
        super().__init__(line, column)
        self.parts = parts  # List of (type, value) where type is 'text' or 'expr'


class BooleanLiteral(ASTNode):
    """Boolean literal: yes, no"""

    def __init__(self, value, line=None, column=None):
        super().__init__(line, column)
        self.value = value


class NothingLiteral(ASTNode):
    """Nothing literal: nothing"""

    def __init__(self, line=None, column=None):
        super().__init__(line, column)
        self.value = None


class ListLiteral(ASTNode):
    """List literal: [1, 2, 3]"""

    def __init__(self, elements, line=None, column=None):
        super().__init__(line, column)
        self.elements = elements


class DictLiteral(ASTNode):
    """Dict literal: {key: value, ...}"""

    def __init__(self, pairs, line=None, column=None):
        super().__init__(line, column)
        self.pairs = pairs  # List of (key_expr, value_expr)


class TupleLiteral(ASTNode):
    """Tuple literal: (1, 2, 3)"""

    def __init__(self, elements, line=None, column=None):
        super().__init__(line, column)
        self.elements = elements


class Identifier(ASTNode):
    """Variable reference: x"""

    def __init__(self, name, line=None, column=None):
        super().__init__(line, column)
        self.name = name


class MeExpression(ASTNode):
    """Self reference: me"""

    def __init__(self, line=None, column=None):
        super().__init__(line, column)


class BinaryOp(ASTNode):
    """Binary operation: a + b, a * b, etc."""

    def __init__(self, left, op, right, line=None, column=None):
        super().__init__(line, column)
        self.left = left
        self.op = op      # '+', '-', '*', '/', '//', '%', '**'
        self.right = right


class UnaryOp(ASTNode):
    """Unary operation: -x, nah x"""

    def __init__(self, op, operand, line=None, column=None):
        super().__init__(line, column)
        self.op = op        # '-', 'nah'
        self.operand = operand


class ComparisonOp(ASTNode):
    """Comparison: a == b, a < b, etc."""

    def __init__(self, left, op, right, line=None, column=None):
        super().__init__(line, column)
        self.left = left
        self.op = op     # '==', '!=', '<', '>', '<=', '>='
        self.right = right


class LogicalOp(ASTNode):
    """Logical operation: a also b, a either b"""

    def __init__(self, left, op, right, line=None, column=None):
        super().__init__(line, column)
        self.left = left
        self.op = op      # 'also', 'either'
        self.right = right


class FunctionCall(ASTNode):
    """Function call: name(args)"""

    def __init__(self, callee, args, kwargs=None, line=None, column=None):
        super().__init__(line, column)
        self.callee = callee    # Expression (Identifier, PropertyAccess, etc.)
        self.args = args        # List of expressions
        self.kwargs = kwargs or {}  # Dict of name -> expression


class PropertyAccess(ASTNode):
    """Property/method access: obj.prop"""

    def __init__(self, object, property, line=None, column=None):
        super().__init__(line, column)
        self.object = object
        self.property = property  # String


class IndexAccess(ASTNode):
    """Index access: obj[index]"""

    def __init__(self, object, index, line=None, column=None):
        super().__init__(line, column)
        self.object = object
        self.index = index


class SliceAccess(ASTNode):
    """Slice access: obj[start:end:step]"""

    def __init__(self, object, start, end, step=None, line=None, column=None):
        super().__init__(line, column)
        self.object = object
        self.start = start
        self.end = end
        self.step = step


class PipelineOp(ASTNode):
    """Pipeline: expr |> func"""

    def __init__(self, left, right, line=None, column=None):
        super().__init__(line, column)
        self.left = left
        self.right = right    # Function call or identifier


class QuickFunction(ASTNode):
    """Lambda: quick(params): expr"""

    def __init__(self, params, body, line=None, column=None):
        super().__init__(line, column)
        self.params = params
        self.body = body  # Single expression


class ListComprehension(ASTNode):
    """List comprehension: [expr each var within iterable when condition]"""

    def __init__(self, expression, variable, iterable, condition=None, line=None, column=None):
        super().__init__(line, column)
        self.expression = expression
        self.variable = variable
        self.iterable = iterable
        self.condition = condition


class AttributeAssignment(ASTNode):
    """Attribute assignment: obj.attr = value"""

    def __init__(self, object, attribute, value, line=None, column=None):
        super().__init__(line, column)
        self.object = object
        self.attribute = attribute
        self.value = value


class IndexAssignment(ASTNode):
    """Index assignment: obj[idx] = value"""

    def __init__(self, object, index, value, line=None, column=None):
        super().__init__(line, column)
        self.object = object
        self.index = index
        self.value = value
