"""
Neuron Error Handling
=====================
Custom exception classes with rich, colorful error messages
and source location tracking.
"""

from neuron.utils.colors import Colors, NeuronColors, colorize


class NeuronError(Exception):
    """Base exception for all Neuron errors."""

    def __init__(self, message, line=None, column=None, file=None, source_line=None):
        self.message = message
        self.line = line
        self.column = column
        self.file = file or "<stdin>"
        self.source_line = source_line
        super().__init__(self.format_error())

    def error_type(self):
        return "Error"

    def format_error(self):
        parts = []

        # Header
        header = f"{Colors.BOLD}{NeuronColors.ERROR}>> Neuron {self.error_type()}{Colors.RESET}"
        parts.append(header)

        # Location
        if self.line is not None:
            loc = f"{Colors.DIM}  -> {self.file}:{self.line}"
            if self.column is not None:
                loc += f":{self.column}"
            loc += Colors.RESET
            parts.append(loc)

        # Source line with pointer
        if self.source_line is not None:
            parts.append("")
            line_num = f"{Colors.DIM}{self.line:4d} │{Colors.RESET} "
            parts.append(line_num + self.source_line.rstrip())

            if self.column is not None and self.column > 0:
                pointer = " " * (7 + self.column - 1) + f"{NeuronColors.ERROR}^{Colors.RESET}"
                parts.append(pointer)

        # Message
        parts.append("")
        parts.append(f"  {NeuronColors.ERROR}{self.message}{Colors.RESET}")
        parts.append("")

        return "\n".join(parts)


class NeuronSyntaxError(NeuronError):
    """Raised when there's a syntax error in Neuron code."""

    def error_type(self):
        return "SyntaxError"


class NeuronRuntimeError(NeuronError):
    """Raised when there's a runtime error during execution."""

    def __init__(self, message, line=None, column=None, file=None, source_line=None, stack_trace=None):
        self.stack_trace = stack_trace or []
        super().__init__(message, line, column, file, source_line)

    def error_type(self):
        return "RuntimeError"

    def format_error(self):
        base = super().format_error()

        if self.stack_trace:
            trace_parts = [f"{Colors.DIM}  Stack trace (most recent call last):{Colors.RESET}"]
            for frame in self.stack_trace:
                trace_parts.append(
                    f"{Colors.DIM}    at {frame.get('function', '<module>')} "
                    f"({frame.get('file', '<stdin>')}:{frame.get('line', '?')}){Colors.RESET}"
                )
            return "\n".join(trace_parts) + "\n" + base

        return base


class NeuronTypeError(NeuronError):
    """Raised on type mismatch."""

    def error_type(self):
        return "TypeError"


class NeuronCompileError(NeuronError):
    """Raised when static analysis rejects a program before execution."""

    def error_type(self):
        return "CompileError"


class NeuronNameError(NeuronError):
    """Raised when a variable name is not found."""

    def error_type(self):
        return "NameError"


class NeuronIndexError(NeuronError):
    """Raised on index out of bounds."""

    def error_type(self):
        return "IndexError"


class NeuronKeyError(NeuronError):
    """Raised on missing dictionary key."""

    def error_type(self):
        return "KeyError"


class NeuronImportError(NeuronError):
    """Raised when a module cannot be imported."""

    def error_type(self):
        return "ImportError"


class NeuronAttributeError(NeuronError):
    """Raised when an attribute is not found on an object."""

    def error_type(self):
        return "AttributeError"


class NeuronValueError(NeuronError):
    """Raised when a value is inappropriate."""

    def error_type(self):
        return "ValueError"


class NeuronZeroDivisionError(NeuronError):
    """Raised on division by zero."""

    def error_type(self):
        return "ZeroDivisionError"


class NeuronAssertionError(NeuronError):
    """Raised when an ensure statement fails."""

    def error_type(self):
        return "AssertionError"


# --- Control flow signals (not really errors) ---

class NeuronReturnSignal(Exception):
    """Signal for 'give' (return) statements."""

    def __init__(self, value=None):
        self.value = value
        super().__init__()


class NeuronBreakSignal(Exception):
    """Signal for 'stop' (break) statements."""
    pass


class NeuronContinueSignal(Exception):
    """Signal for 'skip' (continue) statements."""
    pass


class NeuronEmitSignal(Exception):
    """Signal for 'emit' (yield) statements."""

    def __init__(self, value=None):
        self.value = value
        super().__init__()
