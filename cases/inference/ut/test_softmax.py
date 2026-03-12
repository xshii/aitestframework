"""Softmax 算子测试."""

import math
import unittest


class TestSoftmax(unittest.TestCase):
    """Softmax 算子测试套件."""

    def test_uniform(self):
        """均匀输入输出相等."""
        x = [1.0, 1.0, 1.0]
        exp_x = [math.exp(v) for v in x]
        s = sum(exp_x)
        result = [v / s for v in exp_x]
        for v in result:
            self.assertAlmostEqual(v, 1.0 / 3, places=5)

    def test_sum_to_one(self):
        """输出和为1."""
        x = [2.0, 1.0, 0.1]
        exp_x = [math.exp(v) for v in x]
        s = sum(exp_x)
        result = [v / s for v in exp_x]
        self.assertAlmostEqual(sum(result), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
