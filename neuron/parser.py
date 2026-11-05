"""
Neuron Parser
=============
Recursive-descent parser that converts tokens into an AST.
"""

from neuron.tokens import Token, TokenType, KEYWORDS
from neuron.errors import NeuronSyntaxError
from neuron import ast_nodes as ast


class Parser:
    """Parses Neuron tokens into an Abstract Syntax Tree."""

    def __init__(self, tokens, source="", filename="<stdin>"):
        self.tokens = tokens
        self.pos = 0
        self.source = source
        self.source_lines = source.split('\n') if source else []
        self.filename = filename

    def error(self, message):
        token = self.current()
        source_line = None
        if token and 0 < token.line <= len(self.source_lines):
            source_line = self.source_lines[token.line - 1]
        raise NeuronSyntaxError(message, line=token.line, column=token.column,
                                file=self.filename, source_line=source_line)

    def current(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return self.tokens[-1]  # EOF

    def peek(self, offset=0):
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return self.tokens[-1]

    def advance(self):
        token = self.current()
        if token.type != TokenType.EOF:
            self.pos += 1
        return token

    def expect(self, token_type, message=None):
        if self.current().type == token_type:
            return self.advance()
        msg = message or f"Expected {token_type.name}, got {self.current().type.name} ({self.current().value!r})"
        self.error(msg)

    def match(self, *types):
        if self.current().type in types:
            return self.advance()
        return None

    def skip_newlines(self):
        while self.current().type == TokenType.NEWLINE:
            self.advance()

    def parse(self):
        self.skip_newlines()
        stmts = []
        while self.current().type != TokenType.EOF:
            stmt = self.parse_statement()
            if stmt:
                stmts.append(stmt)
            self.skip_newlines()
        return ast.Program(stmts, line=1)

    # ===== STATEMENTS =====

    def parse_statement(self):
        t = self.current().type
        if t == TokenType.DO:
            return self.parse_function_def()
        elif t == TokenType.ENTITY:
            return self.parse_entity_def()
        elif t == TokenType.BRAIN:
            return self.parse_brain_def()
        elif t == TokenType.QUANTUM:
            return self.parse_quantum_circuit()
        elif t == TokenType.QUBIT:
            token = self.advance()
            name = self.expect(TokenType.IDENTIFIER, "Expected qubit name").value
            self.match(TokenType.NEWLINE)
            return ast.QuantumQubitDeclaration(name, line=token.line, column=token.column)
        elif t == TokenType.QTENSOR:
            token = self.advance()
            self.expect(TokenType.LBRACKET, "Expected '[' after qtensor")
            size = self.expect(TokenType.INTEGER, "qtensor size must be an integer literal").value
            self.expect(TokenType.RBRACKET, "Expected ']' after qtensor size")
            name = self.expect(TokenType.IDENTIFIER, "Expected qtensor register name").value
            self.match(TokenType.NEWLINE)
            return ast.QuantumTensorDeclaration(
                name, size, line=token.line, column=token.column
            )
        elif t == TokenType.WHEN:
            return self.parse_if_statement()
        elif t == TokenType.REPEAT:
            return self.parse_while_loop()
        elif t == TokenType.EACH:
            return self.parse_for_loop()
        elif t == TokenType.GIVE:
            return self.parse_return()
        elif t == TokenType.STOP:
            return self.parse_break()
        elif t == TokenType.SKIP:
            return self.parse_continue()
        elif t == TokenType.ELLIPSIS:
            return self.parse_pass()
        elif t == TokenType.SAFE:
            return self.parse_try_catch()
        elif t == TokenType.GRAB:
            return self.parse_grab()
        elif t == TokenType.DROP:
            return self.parse_drop()
        elif t == TokenType.ENSURE:
            return self.parse_ensure()
        elif t == TokenType.EMIT:
            return self.parse_emit()
        elif t == TokenType.SHARED:
            return self.parse_shared()
        elif (t == TokenType.IDENTIFIER and self.current().value == "tensor"
              and self.peek(1).type == TokenType.LBRACKET):
            return self.parse_tensor_declaration()
        else:
            return self.parse_expression_statement()

    def parse_tensor_declaration(self):
        tensor_token = self.expect(TokenType.IDENTIFIER)
        self.expect(TokenType.LBRACKET)
        dimensions = [self.expect(
            TokenType.INTEGER, "Tensor dimensions must be integer literals"
        ).value]
        while self.match(TokenType.COMMA):
            dimensions.append(self.expect(
                TokenType.INTEGER, "Tensor dimensions must be integer literals"
            ).value)
        self.expect(TokenType.RBRACKET, "Expected ']' after tensor dimensions")
        target = self.expect(
            TokenType.IDENTIFIER, "Expected a variable name after tensor type"
        )
        self.expect(TokenType.ASSIGN, "Expected '=' in tensor declaration")
        value = self.parse_expression()
        self.match(TokenType.NEWLINE)
        tensor_type = ast.TensorType(
            dimensions, line=tensor_token.line, column=tensor_token.column
        )
        return ast.TypedAssignment(
            target.value, tensor_type, value,
            line=tensor_token.line, column=tensor_token.column
        )

    def parse_quantum_circuit(self):
        token = self.expect(TokenType.QUANTUM)
        name = self.expect(TokenType.IDENTIFIER, "Expected circuit name").value
        self.expect(TokenType.COLON, "Expected ':' after quantum circuit name")
        body = self.parse_block()
        operations = []
        for statement in body:
            if isinstance(statement, ast.ExpressionStatement):
                expression = statement.expression
                if (not isinstance(expression, ast.FunctionCall)
                        or not isinstance(expression.callee, ast.Identifier)):
                    self.error("Quantum circuit statements must be gate calls")
                statement = ast.GateOperation(
                    expression.callee.name, expression.args,
                    line=expression.line, column=expression.column
                )
            elif not isinstance(statement, (
                    ast.QuantumQubitDeclaration, ast.QuantumTensorDeclaration, ast.DropStatement,
                    ast.Assignment, ast.MultiAssignment)):
                self.error("Only qubit/register declarations, gate calls, and drop are allowed in a quantum circuit")
            operations.append(statement)
        return ast.QuantumCircuitDef(name, operations, line=token.line, column=token.column)

    def parse_function_def(self):
        token = self.expect(TokenType.DO)
        # Allow 'create' keyword as function name (constructor)
        if self.current().type == TokenType.CREATE:
            name = self.advance().value
        else:
            name = self.expect(TokenType.IDENTIFIER, "Expected function name after 'do'").value
        self.expect(TokenType.LPAREN, "Expected '(' after function name")
        params, defaults = self.parse_param_list()
        self.expect(TokenType.RPAREN, "Expected ')' after parameters")
        self.expect(TokenType.COLON, "Expected ':' after function parameters")
        body = self.parse_block()
        return ast.FunctionDef(name, params, defaults, body, line=token.line, column=token.column)

    def parse_param_list(self):
        params = []
        defaults = {}
        while self.current().type not in (TokenType.RPAREN, TokenType.EOF):
            if self.current().type == TokenType.ME:
                self.advance()
                params.append("me")
            else:
                name = self.expect(TokenType.IDENTIFIER, "Expected parameter name").value
                params.append(name)
                if self.match(TokenType.ASSIGN):
                    defaults[name] = self.parse_expression()
            if not self.match(TokenType.COMMA):
                break
        return params, defaults

    def parse_entity_def(self):
        token = self.expect(TokenType.ENTITY)
        name = self.expect(TokenType.IDENTIFIER, "Expected entity name").value
        parent = None
        if self.match(TokenType.FROM):
            parent = self.expect(TokenType.IDENTIFIER, "Expected parent entity name").value
        self.expect(TokenType.COLON, "Expected ':' after entity declaration")
        body = self.parse_block()
        return ast.EntityDef(name, parent, body, line=token.line, column=token.column)

    def parse_brain_def(self):
        token = self.expect(TokenType.BRAIN)
        name = self.expect(TokenType.IDENTIFIER, "Expected brain name").value
        parent = None
        if self.match(TokenType.FROM):
            parent = self.expect(TokenType.IDENTIFIER, "Expected parent name").value
        self.expect(TokenType.COLON, "Expected ':' after brain declaration")
        body = self.parse_block()
        return ast.BrainDef(name, parent, body, line=token.line, column=token.column)

    def parse_if_statement(self):
        token = self.expect(TokenType.WHEN)
        condition = self.parse_expression()
        self.expect(TokenType.COLON, "Expected ':' after 'when' condition")
        body = self.parse_block()
        elif_clauses = []
        while self.current().type == TokenType.ORWHEN:
            self.advance()
            elif_cond = self.parse_expression()
            self.expect(TokenType.COLON, "Expected ':' after 'orwhen' condition")
            elif_body = self.parse_block()
            elif_clauses.append((elif_cond, elif_body))
        else_body = None
        if self.current().type == TokenType.OTHERWISE:
            self.advance()
            self.expect(TokenType.COLON, "Expected ':' after 'otherwise'")
            else_body = self.parse_block()
        return ast.IfStatement(condition, body, elif_clauses, else_body, line=token.line, column=token.column)

    def parse_while_loop(self):
        token = self.expect(TokenType.REPEAT)
        condition = self.parse_expression()
        self.expect(TokenType.COLON, "Expected ':' after 'repeat' condition")
        body = self.parse_block()
        return ast.WhileLoop(condition, body, line=token.line, column=token.column)

    def parse_for_loop(self):
        token = self.expect(TokenType.EACH)
        var_name = self.expect(TokenType.IDENTIFIER, "Expected variable name after 'each'").value
        self.expect(TokenType.WITHIN, "Expected 'within' after loop variable")
        iterable = self.parse_expression()
        self.expect(TokenType.COLON, "Expected ':' after iterable")
        body = self.parse_block()
        return ast.ForLoop(var_name, iterable, body, line=token.line, column=token.column)

    def parse_return(self):
        token = self.expect(TokenType.GIVE)
        value = None
        if self.current().type not in (TokenType.NEWLINE, TokenType.EOF, TokenType.DEDENT):
            value = self.parse_expression()
        self.match(TokenType.NEWLINE)
        return ast.ReturnStatement(value, line=token.line, column=token.column)

    def parse_break(self):
        token = self.expect(TokenType.STOP)
        self.match(TokenType.NEWLINE)
        return ast.BreakStatement(line=token.line, column=token.column)

    def parse_continue(self):
        token = self.expect(TokenType.SKIP)
        self.match(TokenType.NEWLINE)
        return ast.ContinueStatement(line=token.line, column=token.column)

    def parse_pass(self):
        token = self.expect(TokenType.ELLIPSIS)
        self.match(TokenType.NEWLINE)
        return ast.PassStatement(line=token.line, column=token.column)

    def parse_try_catch(self):
        token = self.expect(TokenType.SAFE)
        self.expect(TokenType.COLON, "Expected ':' after 'safe'")
        body = self.parse_block()
        rescue_clauses = []
        while self.current().type == TokenType.RESCUE:
            self.advance()
            error_type = None
            alias = None
            if self.current().type == TokenType.IDENTIFIER:
                error_type = self.advance().value
            if self.match(TokenType.AS):
                alias = self.expect(TokenType.IDENTIFIER, "Expected variable name after 'as'").value
            self.expect(TokenType.COLON, "Expected ':' after rescue clause")
            handler = self.parse_block()
            rescue_clauses.append((error_type, alias, handler))
        return ast.TryCatch(body, rescue_clauses, line=token.line, column=token.column)

    def parse_grab(self):
        token = self.expect(TokenType.GRAB)
        module = self.expect(TokenType.IDENTIFIER, "Expected module name after 'grab'").value
        alias = None
        items = None
        if self.match(TokenType.AS):
            alias = self.expect(TokenType.IDENTIFIER).value
        self.match(TokenType.NEWLINE)
        return ast.GrabStatement(module, alias, items, line=token.line, column=token.column)

    def parse_drop(self):
        token = self.expect(TokenType.DROP)
        target = self.parse_expression()
        self.match(TokenType.NEWLINE)
        return ast.DropStatement(target, line=token.line, column=token.column)

    def parse_ensure(self):
        token = self.expect(TokenType.ENSURE)
        condition = self.parse_expression()
        message = None
        if self.match(TokenType.COMMA):
            message = self.parse_expression()
        self.match(TokenType.NEWLINE)
        return ast.EnsureStatement(condition, message, line=token.line, column=token.column)

    def parse_emit(self):
        token = self.expect(TokenType.EMIT)
        value = None
        if self.current().type not in (TokenType.NEWLINE, TokenType.EOF):
            value = self.parse_expression()
        self.match(TokenType.NEWLINE)
        return ast.EmitStatement(value, line=token.line, column=token.column)

    def parse_shared(self):
        token = self.expect(TokenType.SHARED)
        names = [self.expect(TokenType.IDENTIFIER).value]
        while self.match(TokenType.COMMA):
            names.append(self.expect(TokenType.IDENTIFIER).value)
        self.match(TokenType.NEWLINE)
        return ast.SharedStatement(names, line=token.line, column=token.column)

    def parse_block(self):
        """Parse an indented block of statements."""
        self.skip_newlines()
        self.expect(TokenType.INDENT, "Expected indented block")
        stmts = []
        while self.current().type not in (TokenType.DEDENT, TokenType.EOF):
            self.skip_newlines()
            if self.current().type in (TokenType.DEDENT, TokenType.EOF):
                break
            stmt = self.parse_statement()
            if stmt:
                stmts.append(stmt)
            self.skip_newlines()
        if self.current().type == TokenType.DEDENT:
            self.advance()
        return stmts

    def parse_expression_statement(self):
        """Parse assignment or expression statement."""
        expr = self.parse_expression()

        # Check for assignment: identifier = expr
        if self.current().type == TokenType.ASSIGN:
            self.advance()
            value = self.parse_expression()
            if isinstance(expr, ast.Identifier):
                node = ast.Assignment(expr.name, value, line=expr.line, column=expr.column)
            elif isinstance(expr, ast.PropertyAccess):
                node = ast.AttributeAssignment(expr.object, expr.property, value, line=expr.line, column=expr.column)
            elif isinstance(expr, ast.IndexAccess):
                node = ast.IndexAssignment(expr.object, expr.index, value, line=expr.line, column=expr.column)
            else:
                self.error("Invalid assignment target")
            self.match(TokenType.NEWLINE)
            return node

        # Augmented assignment: +=, -=, *=, /=
        aug_ops = {
            TokenType.PLUSASSIGN: '+', TokenType.MINUSASSIGN: '-',
            TokenType.STARASSIGN: '*', TokenType.SLASHASSIGN: '/',
        }
        if self.current().type in aug_ops:
            op = aug_ops[self.current().type]
            self.advance()
            value = self.parse_expression()
            if isinstance(expr, ast.Identifier):
                target = expr.name
            else:
                self.error("Invalid augmented assignment target")
            node = ast.AugmentedAssignment(target, op, value, line=expr.line, column=expr.column)
            self.match(TokenType.NEWLINE)
            return node

        # Multi-assignment: a, b = 1, 2
        if self.current().type == TokenType.COMMA and isinstance(expr, ast.Identifier):
            targets = [expr.name]
            while self.match(TokenType.COMMA):
                targets.append(self.expect(TokenType.IDENTIFIER).value)
            self.expect(TokenType.ASSIGN)
            values = [self.parse_expression()]
            while self.match(TokenType.COMMA):
                values.append(self.parse_expression())
            self.match(TokenType.NEWLINE)
            return ast.MultiAssignment(targets, values, line=expr.line, column=expr.column)

        self.match(TokenType.NEWLINE)
        return ast.ExpressionStatement(expr, line=expr.line, column=expr.column)

    # ===== EXPRESSIONS (precedence climbing) =====

    def parse_expression(self):
        return self.parse_pipeline()

    def parse_pipeline(self):
        expr = self.parse_or()
        while self.match(TokenType.PIPE):
            right = self.parse_or()
            expr = ast.PipelineOp(expr, right, line=expr.line, column=expr.column)
        return expr

    def parse_or(self):
        expr = self.parse_and()
        while self.current().type == TokenType.EITHER:
            self.advance()
            right = self.parse_and()
            expr = ast.LogicalOp(expr, 'either', right, line=expr.line, column=expr.column)
        return expr

    def parse_and(self):
        expr = self.parse_not()
        while self.current().type == TokenType.ALSO:
            self.advance()
            right = self.parse_not()
            expr = ast.LogicalOp(expr, 'also', right, line=expr.line, column=expr.column)
        return expr

    def parse_not(self):
        if self.current().type == TokenType.NAH:
            token = self.advance()
            operand = self.parse_not()
            return ast.UnaryOp('nah', operand, line=token.line, column=token.column)
        return self.parse_comparison()

    def parse_comparison(self):
        expr = self.parse_addition()
        comp_ops = {
            TokenType.EQUAL: '==', TokenType.NOTEQUAL: '!=',
            TokenType.LESS: '<', TokenType.GREATER: '>',
            TokenType.LESSEQUAL: '<=', TokenType.GREATEREQUAL: '>=',
        }
        if self.current().type in comp_ops:
            op = comp_ops[self.current().type]
            self.advance()
            right = self.parse_addition()
            expr = ast.ComparisonOp(expr, op, right, line=expr.line, column=expr.column)
        # 'within' operator for membership testing
        if self.current().type == TokenType.WITHIN:
            self.advance()
            right = self.parse_addition()
            expr = ast.ComparisonOp(expr, 'within', right, line=expr.line, column=expr.column)
        return expr

    def parse_addition(self):
        expr = self.parse_multiplication()
        while self.current().type in (TokenType.PLUS, TokenType.MINUS):
            op = '+' if self.current().type == TokenType.PLUS else '-'
            self.advance()
            right = self.parse_multiplication()
            expr = ast.BinaryOp(expr, op, right, line=expr.line, column=expr.column)
        return expr

    def parse_multiplication(self):
        expr = self.parse_power()
        while self.current().type in (TokenType.STAR, TokenType.SLASH, TokenType.DOUBLESLASH, TokenType.PERCENT):
            op_map = {TokenType.STAR: '*', TokenType.SLASH: '/', TokenType.DOUBLESLASH: '//', TokenType.PERCENT: '%'}
            op = op_map[self.current().type]
            self.advance()
            right = self.parse_power()
            expr = ast.BinaryOp(expr, op, right, line=expr.line, column=expr.column)
        return expr

    def parse_power(self):
        expr = self.parse_unary()
        if self.current().type == TokenType.DOUBLESTAR:
            self.advance()
            right = self.parse_unary()
            expr = ast.BinaryOp(expr, '**', right, line=expr.line, column=expr.column)
        return expr

    def parse_unary(self):
        if self.current().type == TokenType.MINUS:
            token = self.advance()
            operand = self.parse_unary()
            return ast.UnaryOp('-', operand, line=token.line, column=token.column)
        return self.parse_postfix()

    def parse_postfix(self):
        expr = self.parse_primary()
        while True:
            if self.current().type == TokenType.LPAREN:
                self.advance()
                args, kwargs = self.parse_arg_list()
                self.expect(TokenType.RPAREN, "Expected ')' after arguments")
                expr = ast.FunctionCall(expr, args, kwargs, line=expr.line, column=expr.column)
            elif self.current().type == TokenType.DOT:
                self.advance()
                prop = self.expect(TokenType.IDENTIFIER, "Expected property name after '.'").value
                expr = ast.PropertyAccess(expr, prop, line=expr.line, column=expr.column)
            elif self.current().type == TokenType.LBRACKET:
                self.advance()
                index = self.parse_expression()
                # Check for slice
                if self.current().type == TokenType.COLON:
                    self.advance()
                    end = None
                    if self.current().type != TokenType.RBRACKET and self.current().type != TokenType.COLON:
                        end = self.parse_expression()
                    step = None
                    if self.match(TokenType.COLON):
                        step = self.parse_expression()
                    self.expect(TokenType.RBRACKET)
                    expr = ast.SliceAccess(expr, index, end, step, line=expr.line, column=expr.column)
                else:
                    self.expect(TokenType.RBRACKET, "Expected ']'")
                    expr = ast.IndexAccess(expr, index, line=expr.line, column=expr.column)
            else:
                break
        return expr

    def parse_arg_list(self):
        args = []
        kwargs = {}
        while self.current().type not in (TokenType.RPAREN, TokenType.EOF):
            # Check for keyword argument: name=value
            if (self.current().type == TokenType.IDENTIFIER and
                    self.peek(1).type == TokenType.ASSIGN):
                name = self.advance().value
                self.advance()  # consume =
                value = self.parse_expression()
                kwargs[name] = value
            else:
                args.append(self.parse_expression())
            if not self.match(TokenType.COMMA):
                break
        return args, kwargs

    def parse_primary(self):
        token = self.current()

        if token.type == TokenType.INTEGER:
            self.advance()
            return ast.NumberLiteral(token.value, line=token.line, column=token.column)

        if token.type == TokenType.FLOAT:
            self.advance()
            return ast.NumberLiteral(token.value, line=token.line, column=token.column)

        if token.type == TokenType.STRING:
            self.advance()
            return ast.StringLiteral(token.value, line=token.line, column=token.column)

        if token.type == TokenType.FSTRING:
            self.advance()
            return ast.FStringLiteral(token.value, line=token.line, column=token.column)

        if token.type == TokenType.YES:
            self.advance()
            return ast.BooleanLiteral(True, line=token.line, column=token.column)

        if token.type == TokenType.NO:
            self.advance()
            return ast.BooleanLiteral(False, line=token.line, column=token.column)

        if token.type == TokenType.NOTHING:
            self.advance()
            return ast.NothingLiteral(line=token.line, column=token.column)

        if token.type == TokenType.ME:
            self.advance()
            return ast.MeExpression(line=token.line, column=token.column)

        if token.type == TokenType.IDENTIFIER:
            self.advance()
            return ast.Identifier(token.value, line=token.line, column=token.column)

        if token.type == TokenType.QUICK:
            return self.parse_quick_function()

        if token.type == TokenType.LPAREN:
            self.advance()
            if self.current().type == TokenType.RPAREN:
                self.advance()
                return ast.TupleLiteral([], line=token.line, column=token.column)
            expr = self.parse_expression()
            if self.match(TokenType.COMMA):
                elements = [expr]
                while self.current().type != TokenType.RPAREN:
                    elements.append(self.parse_expression())
                    if not self.match(TokenType.COMMA):
                        break
                self.expect(TokenType.RPAREN)
                return ast.TupleLiteral(elements, line=token.line, column=token.column)
            self.expect(TokenType.RPAREN, "Expected ')'")
            return expr

        if token.type == TokenType.LBRACKET:
            return self.parse_list_literal()

        if token.type == TokenType.LBRACE:
            return self.parse_dict_literal()

        self.error(f"Unexpected token: {token.type.name} ({token.value!r})")

    def parse_quick_function(self):
        token = self.expect(TokenType.QUICK)
        self.expect(TokenType.LPAREN)
        params = []
        while self.current().type != TokenType.RPAREN:
            params.append(self.expect(TokenType.IDENTIFIER).value)
            if not self.match(TokenType.COMMA):
                break
        self.expect(TokenType.RPAREN)
        self.expect(TokenType.COLON)
        body = self.parse_expression()
        return ast.QuickFunction(params, body, line=token.line, column=token.column)

    def parse_list_literal(self):
        token = self.expect(TokenType.LBRACKET)
        elements = []
        self.skip_newlines()
        while self.current().type != TokenType.RBRACKET:
            elements.append(self.parse_expression())
            self.skip_newlines()
            if not self.match(TokenType.COMMA):
                break
            self.skip_newlines()
        self.expect(TokenType.RBRACKET, "Expected ']'")
        return ast.ListLiteral(elements, line=token.line, column=token.column)

    def parse_dict_literal(self):
        token = self.expect(TokenType.LBRACE)
        pairs = []
        self.skip_newlines()
        while self.current().type != TokenType.RBRACE:
            key = self.parse_expression()
            self.expect(TokenType.COLON, "Expected ':' in dict literal")
            value = self.parse_expression()
            pairs.append((key, value))
            self.skip_newlines()
            if not self.match(TokenType.COMMA):
                break
            self.skip_newlines()
        self.expect(TokenType.RBRACE, "Expected '}'")
        return ast.DictLiteral(pairs, line=token.line, column=token.column)


def parse(source, filename="<stdin>"):
    """Convenience: tokenize + parse source code into AST."""
    from neuron.lexer import tokenize
    tokens = tokenize(source, filename)
    parser = Parser(tokens, source, filename)
    return parser.parse()
