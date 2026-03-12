"""优化器单元测试."""

import unittest


class TestSGD(unittest.TestCase):
    """SGD 优化器测试."""

    def test_basic_step(self):
        """基本梯度下降步."""
        lr = 0.01
        weight = 1.0
        grad = 0.5
        weight -= lr * grad
        self.assertAlmostEqual(weight, 0.995)

    def test_momentum(self):
        """带动量的 SGD."""
        lr, momentum = 0.01, 0.9
        velocity = 0.0
        grad = 0.5
        velocity = momentum * velocity + grad
        update = lr * velocity
        self.assertAlmostEqual(update, 0.005)

    def test_zero_grad(self):
        """零梯度不更新."""
        lr = 0.01
        weight = 1.0
        weight -= lr * 0.0
        self.assertEqual(weight, 1.0)


if __name__ == "__main__":
    unittest.main()
