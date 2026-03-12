"""损失函数单元测试."""

import math
import unittest


class TestCrossEntropyLoss(unittest.TestCase):
    """交叉熵损失测试."""

    def test_perfect_prediction(self):
        """完美预测的损失接近 0."""
        prob = 0.999
        loss = -math.log(prob)
        self.assertLess(loss, 0.01)

    def test_worst_prediction(self):
        """最差预测的损失很大."""
        prob = 0.001
        loss = -math.log(prob)
        self.assertGreater(loss, 5.0)

    def test_uniform(self):
        """均匀分布的损失."""
        num_classes = 10
        prob = 1.0 / num_classes
        loss = -math.log(prob)
        self.assertAlmostEqual(loss, math.log(num_classes), places=4)


class TestMSELoss(unittest.TestCase):
    """均方误差损失测试."""

    def test_zero_error(self):
        """预测完全正确."""
        pred = [1.0, 2.0, 3.0]
        target = [1.0, 2.0, 3.0]
        mse = sum((p - t) ** 2 for p, t in zip(pred, target)) / len(pred)
        self.assertEqual(mse, 0.0)

    def test_simple_error(self):
        """简单误差计算."""
        pred = [1.0, 2.0]
        target = [2.0, 4.0]
        mse = sum((p - t) ** 2 for p, t in zip(pred, target)) / len(pred)
        self.assertAlmostEqual(mse, 2.5)


if __name__ == "__main__":
    unittest.main()
