#!/usr/bin/env python3
"""
tests/test_memory_state.py
memory_state.py 单元测试
"""

import unittest
from aitf.compare.memory_state import MemoryState


class TestReadWrite(unittest.TestCase):
    def test_unwritten_returns_zero(self):
        self.assertEqual(MemoryState().read(0x1000, 4), bytes(4))

    def test_write_and_read_back(self):
        m = MemoryState()
        m.write(0x1000, bytes([0x11, 0x22, 0x33]))
        self.assertEqual(m.read(0x1000, 3), bytes([0x11, 0x22, 0x33]))

    def test_partial_read_before_written(self):
        m = MemoryState()
        m.write(0x1001, bytes([0xBB]))
        self.assertEqual(m.read(0x1000, 3), bytes([0x00, 0xBB, 0x00]))

    def test_overwrite(self):
        m = MemoryState()
        m.write(0x1000, bytes([0xAA, 0xBB]))
        m.write(0x1000, bytes([0xFF, 0xFF]))
        self.assertEqual(m.read(0x1000, 2), bytes([0xFF, 0xFF]))

    def test_partial_overwrite(self):
        m = MemoryState()
        m.write(0x1000, bytes([0x11, 0x22, 0x33, 0x44]))
        m.write(0x1001, bytes([0xBB, 0xCC]))
        self.assertEqual(m.read(0x1000, 4), bytes([0x11, 0xBB, 0xCC, 0x44]))

    def test_non_contiguous_writes(self):
        m = MemoryState()
        m.write(0x1000, bytes([0xAA]))
        m.write(0x2000, bytes([0xBB]))
        self.assertEqual(m.read(0x1000, 1), bytes([0xAA]))
        self.assertEqual(m.read(0x2000, 1), bytes([0xBB]))
        self.assertEqual(m.read(0x1500, 1), bytes([0x00]))

    def test_large_write(self):
        m = MemoryState()
        data = bytes(range(256))
        m.write(0x5000, data)
        self.assertEqual(m.read(0x5000, 256), data)


class TestApplyWithEnable(unittest.TestCase):
    def test_all_enabled(self):
        m      = MemoryState()
        data   = bytes([0xAA, 0xBB, 0xCC, 0xDD])
        result = m.apply_with_enable(0x1000, data, bytes([0x01] * 4))
        self.assertEqual(result, data)
        self.assertEqual(m.read(0x1000, 4), data)

    def test_all_disabled_no_history(self):
        m = MemoryState()
        result = m.apply_with_enable(0x1000, bytes([0xAA, 0xBB]), bytes([0x00, 0x00]))
        self.assertEqual(result, bytes([0x00, 0x00]))

    def test_all_disabled_with_history(self):
        m = MemoryState()
        m.write(0x1000, bytes([0xDE, 0xAD]))
        result = m.apply_with_enable(0x1000, bytes([0xFF, 0xFF]), bytes([0x00, 0x00]))
        self.assertEqual(result, bytes([0xDE, 0xAD]))
        self.assertEqual(m.read(0x1000, 2), bytes([0xDE, 0xAD]))

    def test_partial_enable(self):
        m = MemoryState()
        m.write(0x1000, bytes([0x11, 0x22, 0x33, 0x44]))
        result = m.apply_with_enable(0x1000,
                                     bytes([0xAA, 0xBB, 0xCC, 0xDD]),
                                     bytes([0x01, 0x00, 0x01, 0x00]))
        self.assertEqual(result, bytes([0xAA, 0x22, 0xCC, 0x44]))
        self.assertEqual(m.read(0x1000, 4), bytes([0xAA, 0x22, 0xCC, 0x44]))

    def test_memory_updated_after_apply(self):
        """两次 apply 同地址，第二次 enable=0 的字节取第一次写入的结果。"""
        m = MemoryState()
        m.apply_with_enable(0x2000, bytes([0x10, 0x20, 0x30, 0x40]),
                                    bytes([0x01, 0x01, 0x00, 0x00]))
        result = m.apply_with_enable(0x2000, bytes([0xFF, 0xFF, 0xFF, 0xFF]),
                                             bytes([0x00, 0x00, 0x01, 0x01]))
        self.assertEqual(result, bytes([0x10, 0x20, 0xFF, 0xFF]))

    def test_sequential_partial_writes_accumulate(self):
        """四次单字节 apply 逐步填满同一地址的 4 字节。"""
        m = MemoryState()
        for i, val in enumerate([0xAA, 0xBB, 0xCC, 0xDD]):
            enable = bytes([0x01 if j == i else 0x00 for j in range(4)])
            m.apply_with_enable(0x3000, bytes([val if j == i else 0x00 for j in range(4)]), enable)
        self.assertEqual(m.read(0x3000, 4), bytes([0xAA, 0xBB, 0xCC, 0xDD]))

    def test_cross_address_independence(self):
        """不同地址的 apply_with_enable 互不影响。"""
        m = MemoryState()
        m.write(0x1000, bytes([0x11]))
        m.write(0x2000, bytes([0x22]))
        m.apply_with_enable(0x1000, bytes([0xFF]), bytes([0x01]))
        self.assertEqual(m.read(0x2000, 1), bytes([0x22]))


class TestSnapshot(unittest.TestCase):
    def test_snapshot_same_as_read(self):
        m = MemoryState()
        m.write(0x1000, bytes([0xAA, 0xBB]))
        self.assertEqual(m.snapshot(0x1000, 2), m.read(0x1000, 2))

    def test_snapshot_does_not_modify(self):
        m = MemoryState()
        before = len(m)
        m.snapshot(0x9999, 4)
        self.assertEqual(len(m), before)


class TestReset(unittest.TestCase):
    def test_reset_clears_all(self):
        m = MemoryState()
        m.write(0x1000, bytes([0xAA, 0xBB]))
        m.reset()
        self.assertEqual(len(m), 0)
        self.assertEqual(m.read(0x1000, 2), bytes(2))


class TestWrittenRanges(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(MemoryState().written_ranges(), [])

    def test_single_range(self):
        m = MemoryState()
        m.write(0x1000, bytes([0xAA, 0xBB, 0xCC]))
        self.assertEqual(m.written_ranges(), [(0x1000, 0x1003)])

    def test_two_separate_ranges(self):
        m = MemoryState()
        m.write(0x1000, bytes([0xAA]))
        m.write(0x2000, bytes([0xBB]))
        ranges = m.written_ranges()
        self.assertEqual(len(ranges), 2)
        self.assertIn((0x1000, 0x1001), ranges)
        self.assertIn((0x2000, 0x2001), ranges)

    def test_adjacent_writes_merge(self):
        m = MemoryState()
        m.write(0x1000, bytes([0xAA]))
        m.write(0x1001, bytes([0xBB]))
        self.assertEqual(m.written_ranges(), [(0x1000, 0x1002)])


class TestDunderMethods(unittest.TestCase):
    def test_len_counts_written_bytes(self):
        m = MemoryState()
        m.write(0x1000, bytes([0x11, 0x22, 0x33]))
        self.assertEqual(len(m), 3)

    def test_len_no_double_count_overwrite(self):
        m = MemoryState()
        m.write(0x1000, bytes([0x11, 0x22]))
        m.write(0x1001, bytes([0xFF]))   # 覆盖 0x1001，总字节数仍为 2
        self.assertEqual(len(m), 2)

    def test_repr(self):
        m = MemoryState()
        m.write(0x1000, bytes([0xAA]))
        r = repr(m)
        self.assertIn("MemoryState", r)
        self.assertIn("bytes=1", r)


class TestExportFromSnapshot(unittest.TestCase):
    def test_export_returns_copy(self):
        m = MemoryState()
        m.write(0x1000, bytes([0xAA, 0xBB]))
        snap = m.export_snapshot()
        snap[0x1000] = 0xFF          # 修改快照
        self.assertEqual(m.read(0x1000, 1), bytes([0xAA]))  # m 不受影响

    def test_from_snapshot_restores(self):
        m = MemoryState()
        m.write(0x1000, bytes([0x11, 0x22, 0x33]))
        m2 = MemoryState.from_snapshot(m.export_snapshot())
        self.assertEqual(m2.read(0x1000, 3), bytes([0x11, 0x22, 0x33]))

    def test_from_snapshot_is_independent(self):
        m = MemoryState()
        m.write(0x1000, bytes([0xAA]))
        m2 = MemoryState.from_snapshot(m.export_snapshot())
        m2.write(0x1000, bytes([0xFF]))  # 修改 m2
        self.assertEqual(m.read(0x1000, 1), bytes([0xAA]))  # m 不受影响

    def test_from_snapshot_empty(self):
        m2 = MemoryState.from_snapshot({})
        self.assertEqual(len(m2), 0)
        self.assertEqual(m2.read(0x1000, 4), bytes(4))


if __name__ == "__main__":
    unittest.main(verbosity=2)
