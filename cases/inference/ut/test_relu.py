"""ReLU 激活函数测试."""

import unittest


class TestReLU(unittest.TestCase):
    """ReLU 算子测试套件."""

    def test_positive(self):
        """正数不变."""
        self.assertEqual(max(0, 5), 5)

    def test_negative(self):
        """负数归零."""
        self.assertEqual(max(0, -3), 0)

    def test_zero(self):
        """零值."""
        self.assertEqual(max(0, 0), 0)


if __name__ == "__main__":
    unittest.main()
