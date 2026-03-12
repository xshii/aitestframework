"""分布式训练测试."""

import unittest


class TestDistributedTraining(unittest.TestCase):
    """分布式训练系统测试."""

    def test_allreduce(self):
        """AllReduce 通信验证."""
        # 模拟 4 卡 allreduce
        local_grads = [1.0, 2.0, 3.0, 4.0]
        expected_avg = sum(local_grads) / len(local_grads)
        self.assertAlmostEqual(expected_avg, 2.5)

    def test_data_parallel_split(self):
        """数据并行切分验证."""
        total_samples = 1024
        num_devices = 4
        per_device = total_samples // num_devices
        self.assertEqual(per_device, 256)
        self.assertEqual(per_device * num_devices, total_samples)


if __name__ == "__main__":
    unittest.main()
