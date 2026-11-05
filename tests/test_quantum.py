import unittest

from neuron.compiler import Compiler
from neuron.errors import NeuronCompileError, NeuronRuntimeError
from neuron.interpreter import Interpreter


class QuantumCompilerTests(unittest.TestCase):
    def setUp(self):
        self.compiler = Compiler()

    def test_compiles_bell_state_to_openqasm_3(self):
        source = (
            "quantum BellState:\n"
            "    qubit alice\n"
            "    qubit bob\n"
            "    h(alice)\n"
            "    cx(alice, bob)\n"
            "    measure(alice)\n"
            "    measure(bob)\n"
        )

        qasm = self.compiler.compile_qasm_source(source)

        self.assertEqual(
            qasm,
            'OPENQASM 3.0;\n'
            'include "stdgates.inc";\n'
            'qubit[2] q;\n'
            'bit[2] c;\n'
            'h q[0];\n'
            'cx q[0], q[1];\n'
            'c[0] = measure q[0];\n'
            'c[1] = measure q[1];\n'
        )

    def test_compiles_qtensor_indices_and_rotations(self):
        source = (
            "quantum RegisterCircuit:\n"
            "    qtensor[2] pair\n"
            "    ry(1.57079632679, pair[0])\n"
            "    cx(pair[0], pair[1])\n"
            "    measure(pair[0])\n"
            "    measure(pair[1])\n"
        )

        qasm = self.compiler.compile_qasm_source(source)

        self.assertIn("ry(1.57079632679) q[0];", qasm)
        self.assertIn("cx q[0], q[1];", qasm)

    def test_rejects_qubit_aliasing_at_compile_time(self):
        source = (
            "quantum InvalidClone:\n"
            "    qubit q0\n"
            "    q1 = q0\n"
        )

        with self.assertRaisesRegex(NeuronCompileError, "No-cloning invariant"):
            self.compiler.compile_qasm_source(source)

    def test_rejects_aliasing_an_entangled_qubit(self):
        source = (
            "quantum InvalidEntangledClone:\n"
            "    qubit q0\n"
            "    qubit q1\n"
            "    h(q0)\n"
            "    cx(q0, q1)\n"
            "    q2 = q0\n"
        )

        with self.assertRaisesRegex(NeuronCompileError, "No-cloning invariant"):
            self.compiler.compile_qasm_source(source)

    def test_rejects_dropping_an_unmeasured_qubit(self):
        source = (
            "quantum InvalidDrop:\n"
            "    qubit q0\n"
            "    drop q0\n"
        )

        with self.assertRaisesRegex(NeuronCompileError, "cannot drop unmeasured qubit"):
            self.compiler.compile_qasm_source(source)

    def test_rejects_use_after_measurement(self):
        source = (
            "quantum InvalidLifetime:\n"
            "    qubit q0\n"
            "    measure(q0)\n"
            "    x(q0)\n"
        )

        with self.assertRaisesRegex(NeuronCompileError, "Measured qubit"):
            self.compiler.compile_qasm_source(source)

    def test_rejects_out_of_range_qtensor_index(self):
        source = (
            "quantum InvalidIndex:\n"
            "    qtensor[2] pair\n"
            "    h(pair[2])\n"
        )

        with self.assertRaisesRegex(NeuronCompileError, "constant qtensor indices"):
            self.compiler.compile_qasm_source(source)

    def test_bytecode_target_rejects_quantum_circuit(self):
        source = "quantum Circuit:\n    qubit q0\n    h(q0)\n"

        with self.assertRaisesRegex(NeuronCompileError, "require the OpenQASM 3 target"):
            self.compiler.compile_source(source)

    def test_interpreter_does_not_silently_ignore_quantum_circuit(self):
        source = "quantum Circuit:\n    qubit q0\n    h(q0)\n"

        with self.assertRaisesRegex(NeuronRuntimeError, "Quantum circuits are compile-only"):
            Interpreter().execute(source)


if __name__ == "__main__":
    unittest.main()
