"""
Neuron Token Definitions
========================
All token types used by the Neuron lexer and parser.
"""

from enum import Enum, auto


class TokenType(Enum):
    """All token types in the Neuron language."""

    # --- Literals ---
    INTEGER = auto()        # 42
    FLOAT = auto()          # 3.14
    STRING = auto()         # "hello"
    IDENTIFIER = auto()     # variable_name
    FSTRING = auto()        # "hello {name}"

    # --- Keywords (Neuron-unique) ---
    DO = auto()             # do (function definition)
    GIVE = auto()           # give (return)
    SHOW = auto()           # show (print)  -- built-in func, not keyword token
    ASK = auto()            # ask (input)   -- built-in func, not keyword token
    WHEN = auto()           # when (if)
    ORWHEN = auto()         # orwhen (elif)
    OTHERWISE = auto()      # otherwise (else)
    EACH = auto()           # each (for)
    REPEAT = auto()         # repeat (while)
    GRAB = auto()           # grab (import)
    ENTITY = auto()         # entity (class)
    BRAIN = auto()          # brain (AI model class)
    QUANTUM = auto()        # quantum (quantum circuit block)
    QUBIT = auto()          # qubit (quantum register declaration)
    QTENSOR = auto()        # qtensor (quantum register declaration)
    ME = auto()             # me (self)
    CREATE = auto()         # create (constructor name)
    YES = auto()            # yes (True)
    NO = auto()             # no (False)
    NOTHING = auto()        # nothing (None)
    ALSO = auto()           # also (and)
    EITHER = auto()         # either (or)
    NAH = auto()            # nah (not)
    SAFE = auto()           # safe (try)
    RESCUE = auto()         # rescue (except/catch)
    STOP = auto()           # stop (break)
    SKIP = auto()           # skip (continue)
    QUICK = auto()          # quick (lambda)
    WITHIN = auto()         # within (in)
    DROP = auto()           # drop (del)
    ENSURE = auto()         # ensure (assert)
    EMIT = auto()           # emit (yield)
    SHARED = auto()         # shared (global)
    FROM = auto()           # from (inheritance)
    AS = auto()             # as (alias in rescue)
    UPTO = auto()           # upto (range) -- built-in func

    # --- Arithmetic Operators ---
    PLUS = auto()           # +
    MINUS = auto()          # -
    STAR = auto()           # *
    SLASH = auto()          # /
    DOUBLESLASH = auto()    # //
    PERCENT = auto()        # %
    DOUBLESTAR = auto()     # **

    # --- Comparison Operators ---
    EQUAL = auto()          # ==
    NOTEQUAL = auto()       # !=
    LESS = auto()           # <
    GREATER = auto()        # >
    LESSEQUAL = auto()      # <=
    GREATEREQUAL = auto()   # >=

    # --- Assignment Operators ---
    ASSIGN = auto()         # =
    PLUSASSIGN = auto()      # +=
    MINUSASSIGN = auto()     # -=
    STARASSIGN = auto()      # *=
    SLASHASSIGN = auto()     # /=

    # --- Logical Operators ---
    # (also, either, nah are keywords above)

    # --- Special Operators ---
    PIPE = auto()           # |>  (pipeline)
    DOT = auto()            # .   (member access)
    COMMA = auto()          # ,
    COLON = auto()          # :
    ARROW = auto()          # ->  (type hint / return type)
    AT = auto()             # @   (decorator)
    ELLIPSIS = auto()       # ... (pass/no-op)

    # --- Delimiters ---
    LPAREN = auto()         # (
    RPAREN = auto()         # )
    LBRACKET = auto()       # [
    RBRACKET = auto()       # ]
    LBRACE = auto()         # {
    RBRACE = auto()         # }

    # --- Whitespace / Structure ---
    NEWLINE = auto()        # End of line
    INDENT = auto()         # Increase indentation
    DEDENT = auto()         # Decrease indentation

    # --- Special ---
    EOF = auto()            # End of file
    COMMENT = auto()        # ## comment


class Token:
    """Represents a single token produced by the lexer."""

    __slots__ = ('type', 'value', 'line', 'column', 'file')

    def __init__(self, type: TokenType, value, line: int, column: int, file: str = "<stdin>"):
        self.type = type
        self.value = value
        self.line = line
        self.column = column
        self.file = file

    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r}, line={self.line}, col={self.column})"

    def __eq__(self, other):
        if isinstance(other, Token):
            return self.type == other.type and self.value == other.value
        return NotImplemented


# Keyword mapping: string -> TokenType
KEYWORDS = {
    "do": TokenType.DO,
    "give": TokenType.GIVE,
    "when": TokenType.WHEN,
    "orwhen": TokenType.ORWHEN,
    "otherwise": TokenType.OTHERWISE,
    "each": TokenType.EACH,
    "repeat": TokenType.REPEAT,
    "grab": TokenType.GRAB,
    "entity": TokenType.ENTITY,
    "brain": TokenType.BRAIN,
    "quantum": TokenType.QUANTUM,
    "qubit": TokenType.QUBIT,
    "qtensor": TokenType.QTENSOR,
    "me": TokenType.ME,
    "create": TokenType.CREATE,
    "yes": TokenType.YES,
    "no": TokenType.NO,
    "nothing": TokenType.NOTHING,
    "also": TokenType.ALSO,
    "either": TokenType.EITHER,
    "nah": TokenType.NAH,
    "safe": TokenType.SAFE,
    "rescue": TokenType.RESCUE,
    "stop": TokenType.STOP,
    "skip": TokenType.SKIP,
    "quick": TokenType.QUICK,
    "within": TokenType.WITHIN,
    "drop": TokenType.DROP,
    "ensure": TokenType.ENSURE,
    "emit": TokenType.EMIT,
    "shared": TokenType.SHARED,
    "from": TokenType.FROM,
    "as": TokenType.AS,
}

# Reverse mapping for pretty printing
TOKEN_NAMES = {v: k for k, v in KEYWORDS.items()}
