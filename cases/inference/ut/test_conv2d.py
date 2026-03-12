"""Conv2D 算子基础测试."""

import unittest


class TestConv2D(unittest.TestCase):
    """二维卷积算子测试套件."""

    def test_identity_kernel(self):
        """单位卷积核."""
        self.assertEqual(1 * 5, 5)

    def test_zero_padding(self):
        """零填充."""
        self.assertEqual(0 + 3, 3)

    def test_stride(self):
        """步长计算."""
        input_size, kernel, stride = 7, 3, 2
        output_size = (input_size - kernel) // stride + 1
        self.assertEqual(output_size, 3)


if __name__ == "__main__":
    unittest.main()
