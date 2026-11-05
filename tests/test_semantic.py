import unittest

import numpy as np

from neuron.compiler import Compiler
from neuron.errors import NeuronCompileError
from neuron.interpreter import Interpreter
from neuron.stdlib.tensor import NeuronTensor


class TensorShapeAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.compiler = Compiler()

    def test_infers_matmul_shape_for_typed_assignment(self):
        source = (
            "tensor[100, 784] x = tensor.random(100, 784)\n"
            "tensor[784, 10] w = tensor.random(784, 10)\n"
            "tensor[100, 10] y = x.matmul(w)\n"
        )

        chunk = self.compiler.compile_source(source)

        self.assertEqual(chunk.names, ["x", "w", "y"])

    def test_rejects_matmul_dimension_mismatch_during_compile(self):
        source = (
            "tensor[100, 784] x = tensor.random(100, 784)\n"
            "tensor[128, 10] w = tensor.random(128, 10)\n"
            "y = x.matmul(w)\n"
        )

        with self.assertRaisesRegex(NeuronCompileError, "matmul shape mismatch"):
            self.compiler.compile_source(source)

    def test_rejects_declared_shape_mismatch_before_interpreting(self):
        source = "tensor[2, 3] x = tensor.random(2, 4)\n"

        with self.assertRaisesRegex(NeuronCompileError, "declared \\[2, 3\\], inferred \\[2, 4\\]"):
            Interpreter().execute(source)

    def test_contiguous_float64_input_shares_cpu_buffer(self):
        source = np.arange(6, dtype=np.float64).reshape(2, 3)

        tensor = NeuronTensor(source)

        self.assertEqual(tensor.data_ptr, source.__array_interface__["data"][0])
        source[0, 0] = 42.0
        self.assertEqual(tensor.data[0, 0], 42.0)

    def test_scalar_tensor_keeps_zero_dimensional_shape(self):
        self.assertEqual(NeuronTensor(1.0).shape, [])


if __name__ == "__main__":
    unittest.main()
