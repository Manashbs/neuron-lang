"""
Neuron Bytecode Compiler
========================
Compiles AST to bytecode and provides a VM for execution.
"""

import struct
import pickle
import os
from enum import IntEnum, auto

from neuron import ast_nodes as ast
from neuron.lexer import Lexer
from neuron.parser import Parser


class OpCode(IntEnum):
    LOAD_CONST = 0
    LOAD_VAR = 1
    STORE_VAR = 2
    LOAD_ATTR = 3
    STORE_ATTR = 4
    BINARY_ADD = 10
    BINARY_SUB = 11
    BINARY_MUL = 12
    BINARY_DIV = 13
    BINARY_MOD = 14
    BINARY_POW = 15
    BINARY_FLOORDIV = 16
    COMPARE_EQ = 20
    COMPARE_NE = 21
    COMPARE_LT = 22
    COMPARE_GT = 23
    COMPARE_LE = 24
    COMPARE_GE = 25
    COMPARE_IN = 26
    UNARY_NEG = 30
    UNARY_NOT = 31
    JUMP = 40
    JUMP_IF_FALSE = 41
    JUMP_IF_TRUE = 42
    CALL_FUNCTION = 50
    RETURN_VALUE = 51
    MAKE_FUNCTION = 52
    MAKE_LIST = 53
    MAKE_DICT = 54
    GET_INDEX = 55
    SET_INDEX = 56
    POP_TOP = 60
    DUP_TOP = 61
    BUILD_CLASS = 62
    PIPE = 63
    IMPORT = 64
    PRINT = 65
    NOP = 99
    HALT = 100


class BytecodeChunk:
    """A compiled chunk of bytecode."""
    def __init__(self, name="<module>"):
        self.name = name
        self.code = []       # List of (opcode, operand) tuples
        self.constants = []  # Constant pool
        self.names = []      # Variable name pool
        self.line_map = {}   # instruction_index -> source_line

    def emit(self, opcode, operand=0, line=0):
        idx = len(self.code)
        self.code.append((opcode, operand))
        self.line_map[idx] = line
        return idx

    def add_constant(self, value):
        if value in self.constants:
            return self.constants.index(value)
        self.constants.append(value)
        return len(self.constants) - 1

    def add_name(self, name):
        if name in self.names:
            return self.names.index(name)
        self.names.append(name)
        return len(self.names) - 1

    def patch_jump(self, offset, target=None):
        """Patch a jump instruction's target."""
        if target is None:
            target = len(self.code)
        opcode, _ = self.code[offset]
        self.code[offset] = (opcode, target)


class Compiler:
    """Compiles Neuron AST to bytecode."""

    def compile_file(self, input_path, output_path=None):
        with open(input_path, 'r', encoding='utf-8') as f:
            source = f.read()

        chunk = self.compile_source(source, input_path)

        if output_path is None:
            output_path = input_path.replace('.nl', '.nlc')

        with open(output_path, 'wb') as f:
            pickle.dump(chunk, f)

        return chunk

    def compile_source(self, source, filename="<stdin>"):
        lexer = Lexer(source, filename)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source, filename)
        program = parser.parse()

        chunk = BytecodeChunk(filename)
        self._compile_node(program, chunk)
        chunk.emit(OpCode.HALT)
        return chunk

    def _compile_node(self, node, chunk):
        method = f"_compile_{type(node).__name__}"
        compiler = getattr(self, method, None)
        if compiler:
            return compiler(node, chunk)
        # Fallback: skip unknown nodes
        return

    def _compile_Program(self, node, chunk):
        for stmt in node.statements:
            self._compile_node(stmt, chunk)

    def _compile_NumberLiteral(self, node, chunk):
        idx = chunk.add_constant(node.value)
        chunk.emit(OpCode.LOAD_CONST, idx, node.line)

    def _compile_StringLiteral(self, node, chunk):
        idx = chunk.add_constant(node.value)
        chunk.emit(OpCode.LOAD_CONST, idx, node.line)

    def _compile_BooleanLiteral(self, node, chunk):
        idx = chunk.add_constant(node.value)
        chunk.emit(OpCode.LOAD_CONST, idx, node.line)

    def _compile_NothingLiteral(self, node, chunk):
        idx = chunk.add_constant(None)
        chunk.emit(OpCode.LOAD_CONST, idx, node.line)

    def _compile_Identifier(self, node, chunk):
        idx = chunk.add_name(node.name)
        chunk.emit(OpCode.LOAD_VAR, idx, node.line)

    def _compile_Assignment(self, node, chunk):
        self._compile_node(node.value, chunk)
        idx = chunk.add_name(node.target)
        chunk.emit(OpCode.STORE_VAR, idx, node.line)

    def _compile_ExpressionStatement(self, node, chunk):
        self._compile_node(node.expression, chunk)
        chunk.emit(OpCode.POP_TOP)

    def _compile_BinaryOp(self, node, chunk):
        self._compile_node(node.left, chunk)
        self._compile_node(node.right, chunk)
        op_map = {
            '+': OpCode.BINARY_ADD, '-': OpCode.BINARY_SUB,
            '*': OpCode.BINARY_MUL, '/': OpCode.BINARY_DIV,
            '%': OpCode.BINARY_MOD, '**': OpCode.BINARY_POW,
            '//': OpCode.BINARY_FLOORDIV,
        }
        chunk.emit(op_map[node.op], 0, node.line)

    def _compile_ComparisonOp(self, node, chunk):
        self._compile_node(node.left, chunk)
        self._compile_node(node.right, chunk)
        op_map = {
            '==': OpCode.COMPARE_EQ, '!=': OpCode.COMPARE_NE,
            '<': OpCode.COMPARE_LT, '>': OpCode.COMPARE_GT,
            '<=': OpCode.COMPARE_LE, '>=': OpCode.COMPARE_GE,
            'within': OpCode.COMPARE_IN,
        }
        chunk.emit(op_map[node.op], 0, node.line)

    def _compile_ReturnStatement(self, node, chunk):
        if node.value:
            self._compile_node(node.value, chunk)
        else:
            idx = chunk.add_constant(None)
            chunk.emit(OpCode.LOAD_CONST, idx)
        chunk.emit(OpCode.RETURN_VALUE, 0, node.line)
