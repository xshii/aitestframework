"""Sort 算子单元测试."""

import unittest


class TestSort(unittest.TestCase):
    """排序算子测试套件."""

    def test_ascending(self):
        """升序排序."""
        data = [5, 2, 8, 1, 9]
        self.assertEqual(sorted(data), [1, 2, 5, 8, 9])

    def test_descending(self):
        """降序排序."""
        data = [5, 2, 8, 1, 9]
        self.assertEqual(sorted(data, reverse=True), [9, 8, 5, 2, 1])

    def test_already_sorted(self):
        """已排序数组."""
        data = [1, 2, 3, 4, 5]
        self.assertEqual(sorted(data), data)

    def test_empty(self):
        """空数组排序."""
        self.assertEqual(sorted([]), [])


if __name__ == "__main__":
    unittest.main()
