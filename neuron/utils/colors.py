"""
Neuron Terminal Colors
======================
Cross-platform terminal color support for error messages,
REPL output, and syntax highlighting.
"""

import os
import sys


def supports_color():
    """Check if the terminal supports ANSI color codes."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            return True
        except Exception:
            return os.environ.get("TERM") is not None
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


_COLOR_ENABLED = supports_color()


class Colors:
    """ANSI color codes for terminal output."""

    # Reset
    RESET = "\033[0m" if _COLOR_ENABLED else ""

    # Regular colors
    BLACK = "\033[30m" if _COLOR_ENABLED else ""
    RED = "\033[31m" if _COLOR_ENABLED else ""
    GREEN = "\033[32m" if _COLOR_ENABLED else ""
    YELLOW = "\033[33m" if _COLOR_ENABLED else ""
    BLUE = "\033[34m" if _COLOR_ENABLED else ""
    MAGENTA = "\033[35m" if _COLOR_ENABLED else ""
    CYAN = "\033[36m" if _COLOR_ENABLED else ""
    WHITE = "\033[37m" if _COLOR_ENABLED else ""

    # Bright colors
    BRIGHT_RED = "\033[91m" if _COLOR_ENABLED else ""
    BRIGHT_GREEN = "\033[92m" if _COLOR_ENABLED else ""
    BRIGHT_YELLOW = "\033[93m" if _COLOR_ENABLED else ""
    BRIGHT_BLUE = "\033[94m" if _COLOR_ENABLED else ""
    BRIGHT_MAGENTA = "\033[95m" if _COLOR_ENABLED else ""
    BRIGHT_CYAN = "\033[96m" if _COLOR_ENABLED else ""
    BRIGHT_WHITE = "\033[97m" if _COLOR_ENABLED else ""

    # Styles
    BOLD = "\033[1m" if _COLOR_ENABLED else ""
    DIM = "\033[2m" if _COLOR_ENABLED else ""
    ITALIC = "\033[3m" if _COLOR_ENABLED else ""
    UNDERLINE = "\033[4m" if _COLOR_ENABLED else ""

    # Background
    BG_RED = "\033[41m" if _COLOR_ENABLED else ""
    BG_GREEN = "\033[42m" if _COLOR_ENABLED else ""
    BG_YELLOW = "\033[43m" if _COLOR_ENABLED else ""
    BG_BLUE = "\033[44m" if _COLOR_ENABLED else ""
    BG_MAGENTA = "\033[45m" if _COLOR_ENABLED else ""
    BG_CYAN = "\033[46m" if _COLOR_ENABLED else ""


# Semantic color aliases for Neuron
class NeuronColors:
    """Semantic colors for Neuron language output."""

    KEYWORD = Colors.BRIGHT_MAGENTA
    STRING = Colors.BRIGHT_GREEN
    NUMBER = Colors.BRIGHT_CYAN
    COMMENT = Colors.DIM + Colors.GREEN
    ERROR = Colors.BRIGHT_RED
    WARNING = Colors.BRIGHT_YELLOW
    SUCCESS = Colors.BRIGHT_GREEN
    INFO = Colors.BRIGHT_BLUE
    PROMPT = Colors.BRIGHT_CYAN
    HEADER = Colors.BOLD + Colors.BRIGHT_WHITE
    MUTED = Colors.DIM
    RESET = Colors.RESET
    BOLD = Colors.BOLD


def colorize(text, color):
    """Wrap text in color codes."""
    return f"{color}{text}{Colors.RESET}"


def error_text(text):
    return colorize(text, NeuronColors.ERROR)


def warning_text(text):
    return colorize(text, NeuronColors.WARNING)


def success_text(text):
    return colorize(text, NeuronColors.SUCCESS)


def info_text(text):
    return colorize(text, NeuronColors.INFO)


def bold_text(text):
    return colorize(text, NeuronColors.BOLD)


def header_text(text):
    return colorize(text, NeuronColors.HEADER)
