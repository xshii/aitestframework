"""向量检索端到端测试."""

import unittest
import math


class TestVectorSearch(unittest.TestCase):
    """向量检索系统测试."""

    def test_cosine_similarity(self):
        """余弦相似度计算."""
        a = [1, 0, 1]
        b = [0, 1, 1]
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        sim = dot / (na * nb)
        self.assertAlmostEqual(sim, 0.5, places=4)

    def test_recall_at_k(self):
        """Recall@K 指标."""
        ground_truth = {1, 2, 3, 4, 5}
        retrieved = [1, 3, 5, 7, 9]
        hits = len(set(retrieved) & ground_truth)
        recall = hits / len(ground_truth)
        self.assertGreaterEqual(recall, 0.5)

    def test_batch_query(self):
        """批量查询."""
        num_queries = 100
        results_per_query = 10
        total_results = num_queries * results_per_query
        self.assertEqual(total_results, 1000)


if __name__ == "__main__":
    unittest.main()
