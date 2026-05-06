"""
Neuron Lexer (Tokenizer)
========================
Converts raw Neuron source code into a stream of tokens.
Handles indentation-based scoping, double-hash comments (##),
string interpolation, and all Neuron-unique keywords.
"""

from neuron.tokens import Token, TokenType, KEYWORDS
from neuron.errors import NeuronSyntaxError


class Lexer:
    """Tokenizes Neuron source code."""

    def __init__(self, source: str, filename: str = "<stdin>"):
        self.source = source
        self.filename = filename
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens = []
        self.indent_stack = [0]  # Stack of indentation levels
        self.at_line_start = True
        self.paren_depth = 0  # Track nested parens/brackets/braces
        self.source_lines = source.split('\n')

    def error(self, message):
        source_line = None
        if 0 < self.line <= len(self.source_lines):
            source_line = self.source_lines[self.line - 1]
        raise NeuronSyntaxError(
            message,
            line=self.line,
            column=self.column,
            file=self.filename,
            source_line=source_line,
        )

    def peek(self, offset=0):
        """Look at a character without consuming it."""
        idx = self.pos + offset
        if idx < len(self.source):
            return self.source[idx]
        return '\0'

    def advance(self):
        """Consume the current character and return it."""
        if self.pos >= len(self.source):
            return '\0'
        ch = self.source[self.pos]
        self.pos += 1
        if ch == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return ch

    def match(self, expected):
        """Consume the next character if it matches expected."""
        if self.peek() == expected:
            self.advance()
            return True
        return False

    def add_token(self, token_type, value=None):
        """Add a token to the output list."""
        self.tokens.append(Token(
            type=token_type,
            value=value,
            line=self.line,
            column=self.column,
            file=self.filename,
        ))

    def skip_comment(self):
        """Skip a ## comment to end of line."""
        while self.peek() != '\n' and self.peek() != '\0':
            self.advance()

    def read_string(self, quote_char):
        """Read a string literal, supporting interpolation with {expr}."""
        start_line = self.line
        start_col = self.column
        value = ""
        has_interpolation = False
        interpolation_parts = []
        current_text = ""

        while self.peek() != quote_char:
            if self.peek() == '\0':
                self.error(f"Unterminated string starting at line {start_line}")
            if self.peek() == '\\':
                self.advance()
                escape_char = self.advance()
                escape_map = {
                    'n': '\n', 't': '\t', 'r': '\r',
                    '\\': '\\', "'": "'", '"': '"',
                    '0': '\0', 'a': '\a', 'b': '\b', 'f': '\f',
                }
                if escape_char in escape_map:
                    current_text += escape_map[escape_char]
                else:
                    current_text += '\\' + escape_char
            elif self.peek() == '{':
                self.advance()  # consume {
                has_interpolation = True
                if current_text:
                    interpolation_parts.append(('text', current_text))
                    current_text = ""
                # Read the expression inside {}
                expr = ""
                brace_depth = 1
                while brace_depth > 0:
                    if self.peek() == '\0':
                        self.error("Unterminated string interpolation")
                    if self.peek() == '{':
                        brace_depth += 1
                    elif self.peek() == '}':
                        brace_depth -= 1
                        if brace_depth == 0:
                            self.advance()  # consume closing }
                            break
                    expr += self.advance()
                interpolation_parts.append(('expr', expr))
            else:
                current_text += self.advance()

        self.advance()  # consume closing quote

        if has_interpolation:
            if current_text:
                interpolation_parts.append(('text', current_text))
            self.tokens.append(Token(
                type=TokenType.FSTRING,
                value=interpolation_parts,
                line=start_line,
                column=start_col,
                file=self.filename,
            ))
        else:
            value = current_text
            self.tokens.append(Token(
                type=TokenType.STRING,
                value=value,
                line=start_line,
                column=start_col,
                file=self.filename,
            ))

    def read_number(self):
        """Read an integer or float literal."""
        start = self.pos - 1  # We already consumed the first digit
        start_col = self.column - 1
        is_float = False

        while self.peek().isdigit() or self.peek() == '_':
            self.advance()

        if self.peek() == '.' and self.peek(1).isdigit():
            is_float = True
            self.advance()  # consume .
            while self.peek().isdigit() or self.peek() == '_':
                self.advance()

        # Scientific notation
        if self.peek() in ('e', 'E'):
            is_float = True
            self.advance()
            if self.peek() in ('+', '-'):
                self.advance()
            if not self.peek().isdigit():
                self.error("Invalid scientific notation")
            while self.peek().isdigit():
                self.advance()

        text = self.source[start:self.pos].replace('_', '')

        if is_float:
            self.tokens.append(Token(
                type=TokenType.FLOAT,
                value=float(text),
                line=self.line,
                column=start_col,
                file=self.filename,
            ))
        else:
            self.tokens.append(Token(
                type=TokenType.INTEGER,
                value=int(text),
                line=self.line,
                column=start_col,
                file=self.filename,
            ))

    def read_identifier(self):
        """Read an identifier or keyword."""
        start = self.pos - 1
        start_col = self.column - 1

        while self.peek().isalnum() or self.peek() == '_':
            self.advance()

        text = self.source[start:self.pos]

        # Check if it's a keyword
        if text in KEYWORDS:
            token_type = KEYWORDS[text]
        else:
            token_type = TokenType.IDENTIFIER

        self.tokens.append(Token(
            type=token_type,
            value=text,
            line=self.line,
            column=start_col,
            file=self.filename,
        ))

    def handle_indentation(self):
        """Process indentation at the start of a line."""
        if self.paren_depth > 0:
            # Inside parentheses, brackets, or braces — ignore indentation
            while self.peek() in (' ', '\t'):
                self.advance()
            return

        indent = 0
        while self.peek() in (' ', '\t'):
            if self.peek() == ' ':
                indent += 1
            else:
                indent += 4  # Tab = 4 spaces
            self.advance()

        # Skip blank lines and comment-only lines
        if self.peek() == '\n' or self.peek() == '\0':
            return
        if self.peek() == '#' and self.peek(1) == '#':
            return

        current_indent = self.indent_stack[-1]

        if indent > current_indent:
            self.indent_stack.append(indent)
            self.add_token(TokenType.INDENT, indent)
        elif indent < current_indent:
            while self.indent_stack and self.indent_stack[-1] > indent:
                self.indent_stack.pop()
                self.add_token(TokenType.DEDENT, indent)
            if self.indent_stack[-1] != indent:
                self.error(f"Inconsistent indentation (expected {self.indent_stack[-1]}, got {indent})")

    def tokenize(self):
        """Tokenize the entire source code and return a list of tokens."""
        while self.pos < len(self.source):
            # Handle indentation at start of line
            if self.at_line_start:
                self.handle_indentation()
                self.at_line_start = False

            ch = self.peek()

            # Skip spaces/tabs (not at line start)
            if ch in (' ', '\t'):
                self.advance()
                continue

            # Newline
            if ch == '\n':
                if self.paren_depth == 0:
                    # Don't emit consecutive newlines
                    if self.tokens and self.tokens[-1].type != TokenType.NEWLINE:
                        self.add_token(TokenType.NEWLINE, '\\n')
                self.advance()
                self.at_line_start = True
                continue

            # Carriage return
            if ch == '\r':
                self.advance()
                continue

            # Comments: ##
            if ch == '#' and self.peek(1) == '#':
                self.skip_comment()
                continue

            # Numbers
            if ch.isdigit():
                self.advance()
                self.read_number()
                continue

            # Strings
            if ch in ('"', "'"):
                self.advance()
                self.read_string(ch)
                continue

            # Identifiers and keywords
            if ch.isalpha() or ch == '_':
                self.advance()
                self.read_identifier()
                continue

            # Operators and delimiters
            self.advance()  # consume the character

            if ch == '+':
                if self.match('='):
                    self.add_token(TokenType.PLUSASSIGN, '+=')
                else:
                    self.add_token(TokenType.PLUS, '+')
            elif ch == '-':
                if self.match('>'):
                    self.add_token(TokenType.ARROW, '->')
                elif self.match('='):
                    self.add_token(TokenType.MINUSASSIGN, '-=')
                else:
                    self.add_token(TokenType.MINUS, '-')
            elif ch == '*':
                if self.match('*'):
                    self.add_token(TokenType.DOUBLESTAR, '**')
                elif self.match('='):
                    self.add_token(TokenType.STARASSIGN, '*=')
                else:
                    self.add_token(TokenType.STAR, '*')
            elif ch == '/':
                if self.match('/'):
                    self.add_token(TokenType.DOUBLESLASH, '//')
                elif self.match('='):
                    self.add_token(TokenType.SLASHASSIGN, '/=')
                else:
                    self.add_token(TokenType.SLASH, '/')
            elif ch == '%':
                self.add_token(TokenType.PERCENT, '%')
            elif ch == '=':
                if self.match('='):
                    self.add_token(TokenType.EQUAL, '==')
                else:
                    self.add_token(TokenType.ASSIGN, '=')
            elif ch == '!':
                if self.match('='):
                    self.add_token(TokenType.NOTEQUAL, '!=')
                else:
                    self.error("Unexpected character '!'. Did you mean 'nah' (not)?")
            elif ch == '<':
                if self.match('='):
                    self.add_token(TokenType.LESSEQUAL, '<=')
                else:
                    self.add_token(TokenType.LESS, '<')
            elif ch == '>':
                if self.match('='):
                    self.add_token(TokenType.GREATEREQUAL, '>=')
                else:
                    self.add_token(TokenType.GREATER, '>')
            elif ch == '|':
                if self.match('>'):
                    self.add_token(TokenType.PIPE, '|>')
                else:
                    self.error("Unexpected '|'. Did you mean '|>' (pipeline)?")
            elif ch == '(':
                self.paren_depth += 1
                self.add_token(TokenType.LPAREN, '(')
            elif ch == ')':
                self.paren_depth -= 1
                self.add_token(TokenType.RPAREN, ')')
            elif ch == '[':
                self.paren_depth += 1
                self.add_token(TokenType.LBRACKET, '[')
            elif ch == ']':
                self.paren_depth -= 1
                self.add_token(TokenType.RBRACKET, ']')
            elif ch == '{':
                self.paren_depth += 1
                self.add_token(TokenType.LBRACE, '{')
            elif ch == '}':
                self.paren_depth -= 1
                self.add_token(TokenType.RBRACE, '}')
            elif ch == ',':
                self.add_token(TokenType.COMMA, ',')
            elif ch == ':':
                self.add_token(TokenType.COLON, ':')
            elif ch == '.':
                if self.match('.'):
                    if self.match('.'):
                        self.add_token(TokenType.ELLIPSIS, '...')
                    else:
                        self.error("Unexpected '..' — did you mean '...' (no-op)?")
                else:
                    self.add_token(TokenType.DOT, '.')
            elif ch == '@':
                self.add_token(TokenType.AT, '@')
            else:
                self.error(f"Unexpected character: '{ch}'")

        # Emit final NEWLINE if needed
        if self.tokens and self.tokens[-1].type != TokenType.NEWLINE:
            self.add_token(TokenType.NEWLINE, '\\n')

        # Close remaining indentation levels
        while len(self.indent_stack) > 1:
            self.indent_stack.pop()
            self.add_token(TokenType.DEDENT, 0)

        # EOF token
        self.add_token(TokenType.EOF, None)

        return self.tokens


def tokenize(source: str, filename: str = "<stdin>"):
    """Convenience function to tokenize source code."""
    lexer = Lexer(source, filename)
    return lexer.tokenize()
