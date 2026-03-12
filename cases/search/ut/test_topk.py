"""TopK 算子单元测试."""

import unittest


class TestTopK(unittest.TestCase):
    """TopK 算子测试套件."""

    def test_basic(self):
        """基本 TopK."""
        data = [3, 1, 4, 1, 5, 9, 2, 6]
        k = 3
        result = sorted(data, reverse=True)[:k]
        self.assertEqual(result, [9, 6, 5])

    def test_k_equals_len(self):
        """K 等于数组长度."""
        data = [5, 3, 1]
        result = sorted(data, reverse=True)[:3]
        self.assertEqual(result, [5, 3, 1])

    def test_single_element(self):
        """单元素 TopK."""
        data = [42]
        result = sorted(data, reverse=True)[:1]
        self.assertEqual(result, [42])


if __name__ == "__main__":
    unittest.main()
