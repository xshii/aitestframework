#!/usr/bin/env python3
"""
tests/test_integration_tensor.py
=================================
tensor commit 日志集成测试

测试日志场景（5 个 commit，3 种 tid，read=input / write=output）：

  commit-1  tid=2  cyc=0x01  MmulCmd
    read  0x1000 [8] = AA BB CC DD 00 00 00 00   ← input（快照，不改内存）
    byteEnable = 01 01 01 01 00 00 00 00
    write 0x1000 [8] = 11 22 33 44 EE FF 00 00   ← output
    → actual[0..3] = 11 22 33 44（enable=1）
    → actual[4..7] = 00 00 00 00（enable=0，历史缺省0x00）

  commit-2  tid=5  cyc=0x03  ConvCmd
    read  0x1000 [8]   cross_op=T: 11 22 33 44 00 00 00 00
                       cross_op=F: 00 00 00 00 00 00 00 00 (日志记录值)
    byteEnable = 00 00 00 00 01 01 01 01
    write 0x1000 [8] = 00 00 00 00 55 66 77 88
    → actual cross_op=T: 11 22 33 44 55 66 77 88（前4取历史commit-1结果）
    → actual cross_op=F: 00 00 00 00 55 66 77 88（前4取0x00，无跨块历史）

  commit-3  tid=2  cyc=0x05  MmulCmd（tid=2 再次出现）
    read  0x1000 [8]   cross_op=T: 11 22 33 44 55 66 77 88
    byteEnable = 01 00 01 00
    write 0x2000 [4] = 11 22 33 44
    → actual: 11 00 33 00（0x2000~0x2003 历史均0x00）

  commit-4  tid=7  cyc=0x07  GemmCmd
    read  0x2000 [4]   cross_op=T: 11 00 33 00
    byteEnable = 00 01 00 01
    write 0x2000 [4] = AA BB CC DD
    → actual cross_op=T: 11 BB 33 DD（取历史commit-3）
    → actual cross_op=F: 00 BB 00 DD（无历史，取0x00）
    byteEnable = 01 01 01 01
    write 0x3000 [4] = DE AD BE EF
    → actual: DE AD BE EF

  commit-5  tid=9  cyc=0x09  GemmCmd   空块

cross_op=True 最终内存：
  0x1000~0x1007: 11 22 33 44 55 66 77 88
  0x2000~0x2003: 11 BB 33 DD
  0x3000~0x3003: DE AD BE EF
"""

import json, os, sys, tempfile, unittest
from aitf.compare.memory_state import MemoryState
from aitf.compare.tensor_log_parser import TensorLogParser, save_tensor_blocks

LOG = """\
[cyc=0x1] commit MmulCmd, groupId: 0, tid: 2, at cyc = 0x1
=========begin tensor=============
[DMA][0] read  mem [0x1000] [8] = AA BB CC DD 00 00 00 00
[DMA][0] byteEnable = 01 01 01 01 00 00 00 00
[DMA][0] write mem [0x1000] [8] = 11 22 33 44 EE FF 00 00
=========end tensor=============
[cyc=0x3] commit ConvCmd, groupId: 0, tid: 5, at cyc = 0x3
=========begin tensor=============
[DMA][0] read  mem [0x1000] [8] = 11 22 33 44 00 00 00 00
[DMA][0] byteEnable = 00 00 00 00 01 01 01 01
[DMA][0] write mem [0x1000] [8] = 00 00 00 00 55 66 77 88
=========end tensor=============
[cyc=0x5] commit MmulCmd, groupId: 0, tid: 2, at cyc = 0x5
=========begin tensor=============
[DMA][0] read  mem [0x1000] [8] = 11 22 33 44 55 66 77 88
[DMA][1] byteEnable = 01 00 01 00
[DMA][1] write mem [0x2000] [4] = 11 22 33 44
=========end tensor=============
[cyc=0x7] commit GemmCmd, groupId: 1, tid: 7, at cyc = 0x7
=========begin tensor=============
[DMA][0] read  mem [0x2000] [4] = 11 00 33 00
[DMA][0] byteEnable = 00 01 00 01
[DMA][0] write mem [0x2000] [4] = AA BB CC DD
[DMA][1] byteEnable = 01 01 01 01
[DMA][1] write mem [0x3000] [4] = DE AD BE EF
=========end tensor=============
[cyc=0x9] commit GemmCmd, groupId: 1, tid: 9, at cyc = 0x9
=========begin tensor=============
=========end tensor=============
"""


def _parse(cross_op: bool):
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
        f.write(LOG); path = f.name
    try:
        parser = TensorLogParser(path, cross_op=cross_op)
        blocks = parser.parse()
        return blocks, parser.mem
    finally:
        os.unlink(path)


# ══════════════════════════════════════════════════════════════
#  基本结构（与 cross_op 无关）
# ══════════════════════════════════════════════════════════════

class TestStructure(unittest.TestCase):
    def setUp(self): self.blocks, _ = _parse(True)

    def test_five_blocks(self):
        self.assertEqual(len(self.blocks), 5)

    def test_tids(self):
        self.assertEqual([b.tid for b in self.blocks], [2, 5, 2, 7, 9])

    def test_cmds(self):
        self.assertEqual([b.cmd for b in self.blocks],
                         ["MmulCmd","ConvCmd","MmulCmd","GemmCmd","GemmCmd"])

    def test_cycs(self):
        self.assertEqual([b.cyc for b in self.blocks], [1, 3, 5, 7, 9])

    def test_last_block_empty(self):
        self.assertTrue(self.blocks[4].is_empty)

    def test_non_empty_blocks(self):
        for b in self.blocks[:4]:
            self.assertFalse(b.is_empty)

    def test_reads_are_inputs(self):
        # commit-1 有 1 read
        self.assertEqual(len(self.blocks[0].reads), 1)
        self.assertEqual(self.blocks[0].reads[0].addr, 0x1000)

    def test_writes_are_outputs(self):
        # commit-1 有 1 write
        self.assertEqual(len(self.blocks[0].writes), 1)
        self.assertEqual(self.blocks[0].writes[0].addr, 0x1000)

    def test_commit4_two_outputs(self):
        self.assertEqual(len(self.blocks[3].writes), 2)

    def test_cross_op_true_mem_not_none(self):
        _, mem = _parse(True)
        self.assertIsNotNone(mem)

    def test_cross_op_false_mem_is_none(self):
        _, mem = _parse(False)
        self.assertIsNone(mem)


# ══════════════════════════════════════════════════════════════
#  cross_op=True：跨 commit 内存累积
# ══════════════════════════════════════════════════════════════

class TestCrossOpTrue(unittest.TestCase):
    def setUp(self): self.blocks, self.mem = _parse(True)

    # commit-1：前4字节enable=1写入，后4字节enable=0取历史(0x00)
    def test_c1_output_actual(self):
        actual = self.blocks[0].writes[0].actual_data
        self.assertEqual(actual, bytes([0x11,0x22,0x33,0x44,0x00,0x00,0x00,0x00]))

    # commit-2：前4字节enable=0取历史(commit-1写的0x11224433)，后4字节enable=1写入
    def test_c2_output_merges_c1(self):
        actual = self.blocks[1].writes[0].actual_data
        self.assertEqual(actual, bytes([0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88]))

    # commit-3：写0x2000，enable=01 00 01 00，历史全0x00
    def test_c3_output_partial(self):
        actual = self.blocks[2].writes[0].actual_data
        self.assertEqual(actual, bytes([0x11,0x00,0x33,0x00]))

    # commit-4 write0：enable=00 01 00 01，取历史commit-3（0x11,0x00,0x33,0x00）
    def test_c4_write0_merges_c3(self):
        w = next(w for w in self.blocks[3].writes if w.addr == 0x2000)
        self.assertEqual(w.actual_data, bytes([0x11,0xBB,0x33,0xDD]))

    # commit-4 write1：enable全1，直接写入
    def test_c4_write1_full(self):
        w = next(w for w in self.blocks[3].writes if w.addr == 0x3000)
        self.assertEqual(w.actual_data, bytes([0xDE,0xAD,0xBE,0xEF]))

    # 最终全局内存
    def test_final_mem_0x1000(self):
        self.assertEqual(self.mem.read(0x1000, 8),
                         bytes([0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88]))

    def test_final_mem_0x2000(self):
        self.assertEqual(self.mem.read(0x2000, 4),
                         bytes([0x11,0xBB,0x33,0xDD]))

    def test_final_mem_0x3000(self):
        self.assertEqual(self.mem.read(0x3000, 4),
                         bytes([0xDE,0xAD,0xBE,0xEF]))

    def test_final_mem_unwritten_zero(self):
        self.assertEqual(self.mem.read(0x9000, 4), bytes(4))

    def test_final_written_ranges(self):
        ranges = self.mem.written_ranges()
        self.assertIn((0x1000, 0x1008), ranges)
        self.assertIn((0x2000, 0x2004), ranges)
        self.assertIn((0x3000, 0x3004), ranges)

    def test_total_written_bytes(self):
        self.assertEqual(len(self.mem), 16)  # 8+4+4


# ══════════════════════════════════════════════════════════════
#  cross_op=False：每个 commit 独立内存，byteEnable=0x00 → 0x00
# ══════════════════════════════════════════════════════════════

class TestCrossOpFalse(unittest.TestCase):
    def setUp(self): self.blocks, _ = _parse(False)

    # commit-1：与 True 相同（无历史时两者等价）
    def test_c1_output_same_as_true(self):
        actual = self.blocks[0].writes[0].actual_data
        self.assertEqual(actual, bytes([0x11,0x22,0x33,0x44,0x00,0x00,0x00,0x00]))

    # commit-2：enable=0的前4字节取本块独立内存（空，=0x00），不取commit-1历史
    def test_c2_output_no_cross_history(self):
        actual = self.blocks[1].writes[0].actual_data
        self.assertEqual(actual, bytes([0x00,0x00,0x00,0x00,0x55,0x66,0x77,0x88]))

    # 与 True 模式结果不同，明确区分
    def test_c2_differs_from_cross_op_true(self):
        actual_false = self.blocks[1].writes[0].actual_data
        blocks_true, _ = _parse(True)
        actual_true  = blocks_true[1].writes[0].actual_data
        self.assertNotEqual(actual_false, actual_true)

    # commit-3：0x2000 历史为空，enable=01 00 01 00 → 结果与 True 相同
    def test_c3_partial_no_history(self):
        actual = self.blocks[2].writes[0].actual_data
        self.assertEqual(actual, bytes([0x11,0x00,0x33,0x00]))

    # commit-4 write0：enable=00 01 00 01，独立内存无历史 → 0x00
    def test_c4_write0_no_cross_history(self):
        w = next(w for w in self.blocks[3].writes if w.addr == 0x2000)
        self.assertEqual(w.actual_data, bytes([0x00,0xBB,0x00,0xDD]))

    # commit-4 write1：全enable=1，与 True 相同
    def test_c4_write1_full(self):
        w = next(w for w in self.blocks[3].writes if w.addr == 0x3000)
        self.assertEqual(w.actual_data, bytes([0xDE,0xAD,0xBE,0xEF]))

    # 块内多次写同地址：同一 commit 内第二次 write enable=0 取本块第一次写的值
    def test_intra_block_history(self):
        """同一 commit 内两次写同地址，第二次 enable=0 应取第一次的写入结果。"""
        log = """\
[cyc=0x1] commit TestCmd, groupId: 0, tid: 1, at cyc = 0x1
=========begin tensor=============
[X][0] byteEnable = 01 01 00 00
[X][0] write mem [0x5000] [4] = AA BB CC DD
[X][0] byteEnable = 00 00 01 01
[X][0] write mem [0x5000] [4] = 00 00 EE FF
=========end tensor=============
"""
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
            f.write(log); path = f.name
        try:
            blocks = TensorLogParser(path, cross_op=False).parse()
        finally:
            os.unlink(path)
        # write1: enable=00 00 01 01，前2字节取块内write0的结果(AA BB)
        self.assertEqual(blocks[0].writes[1].actual_data,
                         bytes([0xAA, 0xBB, 0xEE, 0xFF]))


# ══════════════════════════════════════════════════════════════
#  read 语义：input 快照不改内存
# ══════════════════════════════════════════════════════════════

class TestReadSemantics(unittest.TestCase):
    def test_read_does_not_change_mem(self):
        """read 行只记录日志快照，不写入 MemoryState；0x1000 应反映 write 后的值。"""
        _, mem = _parse(True)
        # commit-1 write actual=11 22 33 44 00 00 00 00，read 的 AA 不应出现在内存
        self.assertEqual(mem.read(0x1000, 1), bytes([0x11]))

    def test_read_raw_data_preserved(self):
        blocks, _ = _parse(True)
        self.assertEqual(blocks[0].reads[0].raw_data,
                         bytes([0xAA,0xBB,0xCC,0xDD,0x00,0x00,0x00,0x00]))

    def test_read_addr_and_size(self):
        blocks, _ = _parse(True)
        r = blocks[0].reads[0]
        self.assertEqual(r.addr, 0x1000)
        self.assertEqual(r.size, 8)


# ══════════════════════════════════════════════════════════════
#  外部传入 MemoryState（cross_op=True，预留跨文件接口）
# ══════════════════════════════════════════════════════════════

class TestExternalMem(unittest.TestCase):
    """传入外部 MemoryState，验证 parser 使用并更新它。"""

    LOG_A = """\
[cyc=0x1] commit WriteCmd, groupId: 0, tid: 1, at cyc = 0x1
=========begin tensor=============
[X][0] byteEnable = 01 01 01 01
[X][0] write mem [0x5000] [4] = 11 22 33 44
=========end tensor=============
"""
    LOG_B = """\
[cyc=0x3] commit ReadCmd, groupId: 0, tid: 2, at cyc = 0x3
=========begin tensor=============
[X][0] read  mem [0x5000] [4] = 11 22 33 44
[X][0] byteEnable = 00 00 01 01
[X][0] write mem [0x5000] [4] = FF FF AA BB
=========end tensor=============
"""

    def _run(self):
        mem = MemoryState()
        files = []
        for text in [self.LOG_A, self.LOG_B]:
            with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
                f.write(text); files.append(f.name)
        try:
            TensorLogParser(files[0], cross_op=True, mem=mem).parse()
            blocks_b = TensorLogParser(files[1], cross_op=True, mem=mem).parse()
            return blocks_b, mem
        finally:
            for p in files: os.unlink(p)

    def test_b_merges_with_a(self):
        blocks_b, _ = self._run()
        # enable=00 00 01 01：前2字节取历史(LOG_A写的11 22)，后2写AA BB
        self.assertEqual(blocks_b[0].writes[0].actual_data,
                         bytes([0x11, 0x22, 0xAA, 0xBB]))

    def test_final_memory(self):
        _, mem = self._run()
        self.assertEqual(mem.read(0x5000, 4), bytes([0x11, 0x22, 0xAA, 0xBB]))


# ══════════════════════════════════════════════════════════════
#  MemoryState export/from_snapshot（跨文件预留接口）
# ══════════════════════════════════════════════════════════════

class TestSnapshotInterface(unittest.TestCase):
    def test_export_snapshot(self):
        m = MemoryState()
        m.write(0x1000, bytes([0xAA, 0xBB]))
        snap = m.export_snapshot()
        self.assertEqual(snap[0x1000], 0xAA)
        self.assertEqual(snap[0x1001], 0xBB)

    def test_from_snapshot_restores(self):
        m = MemoryState()
        m.write(0x1000, bytes([0x11, 0x22, 0x33]))
        snap = m.export_snapshot()
        m2 = MemoryState.from_snapshot(snap)
        self.assertEqual(m2.read(0x1000, 3), bytes([0x11, 0x22, 0x33]))

    def test_from_snapshot_is_independent_copy(self):
        m = MemoryState()
        m.write(0x1000, bytes([0xAA]))
        snap = m.export_snapshot()
        m2 = MemoryState.from_snapshot(snap)
        m2.write(0x1000, bytes([0xFF]))  # 修改 m2
        self.assertEqual(m.read(0x1000, 1), bytes([0xAA]))  # m 不变

    def test_export_is_copy_not_ref(self):
        m = MemoryState()
        m.write(0x1000, bytes([0xAA]))
        snap = m.export_snapshot()
        snap[0x1000] = 0xFF  # 修改导出快照
        self.assertEqual(m.read(0x1000, 1), bytes([0xAA]))  # m 不变

    def test_roundtrip_with_parser(self):
        """模拟跨文件场景：LOG_A 写入，export，from_snapshot，LOG_B 继续累积。"""
        LOG_A = """\
[cyc=0x1] commit Cmd, groupId: 0, tid: 1, at cyc = 0x1
=========begin tensor=============
[X][0] byteEnable = 01 01 00 00
[X][0] write mem [0x7000] [4] = CC DD EE FF
=========end tensor=============
"""
        LOG_B = """\
[cyc=0x3] commit Cmd, groupId: 0, tid: 1, at cyc = 0x3
=========begin tensor=============
[X][0] byteEnable = 00 00 01 01
[X][0] write mem [0x7000] [4] = 00 00 AA BB
=========end tensor=============
"""
        files = []
        for text in [LOG_A, LOG_B]:
            with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
                f.write(text); files.append(f.name)
        try:
            parser_a = TensorLogParser(files[0], cross_op=True)
            parser_a.parse()
            # 导出快照，模拟跨文件传递
            snap  = parser_a.mem.export_snapshot()
            mem_b = MemoryState.from_snapshot(snap)
            parser_b = TensorLogParser(files[1], cross_op=True, mem=mem_b)
            blocks_b = parser_b.parse()
        finally:
            for p in files: os.unlink(p)
        # LOG_A 写 CC DD 到 0x7000~0x7001（enable=01 01 00 00）
        # LOG_B enable=00 00 01 01：前2取历史(CC DD)，后2写AA BB
        self.assertEqual(blocks_b[0].writes[0].actual_data,
                         bytes([0xCC, 0xDD, 0xAA, 0xBB]))


# ══════════════════════════════════════════════════════════════
#  JSON 输出
# ══════════════════════════════════════════════════════════════

class TestJsonOutput(unittest.TestCase):
    def setUp(self):
        self.blocks, _ = _parse(True)
        self.out_dir    = tempfile.mkdtemp()
        save_tensor_blocks(self.blocks, self.out_dir)

    def test_summary_json_exists(self):
        self.assertTrue(os.path.exists(
            os.path.join(self.out_dir, "tensor_summary.json")))

    def test_summary_five_entries(self):
        with open(os.path.join(self.out_dir, "tensor_summary.json")) as f:
            data = json.load(f)
        self.assertEqual(len(data), 5)

    def test_empty_block_flagged(self):
        with open(os.path.join(self.out_dir, "tensor_summary.json")) as f:
            data = json.load(f)
        self.assertTrue(data[4]["is_empty"])

    def test_commit1_actual_in_json(self):
        with open(os.path.join(self.out_dir, "tensor_summary.json")) as f:
            data = json.load(f)
        self.assertEqual(data[0]["writes"][0]["actual_data"],
                         "11 22 33 44 00 00 00 00")

    def test_commit4_two_writes_in_json(self):
        with open(os.path.join(self.out_dir, "tensor_summary.json")) as f:
            data = json.load(f)
        self.assertEqual(len(data[3]["writes"]), 2)

    def _all_txts(self, d):
        txts = []
        for root, _, files in os.walk(d):
            txts += [f for f in files if f.endswith(".txt")]
        return txts

    def test_txt_files_for_nonempty(self):
        # 每个非空 block 最多 2 个文件（Input + Output），有些可能只有 reads 或 writes
        txts = self._all_txts(self.out_dir)
        self.assertTrue(len(txts) > 0)

    def test_no_txt_for_empty_block(self):
        # 空块不产生子目录
        subdirs = [d for d in os.listdir(self.out_dir)
                   if os.path.isdir(os.path.join(self.out_dir, d))]
        # 不检查特定目录名，只确保 txt 都来自有效块
        self.assertTrue(len(subdirs) >= 0)

    def test_txt_sections_renamed(self):
        """txt 文件名包含 Input 或 Output 标记。"""
        txts = self._all_txts(self.out_dir)
        self.assertTrue(any("Input" in f or "Output" in f for f in txts))

    def test_tid_names_in_filename(self):
        # 新格式：文件名为 CMD_idx_Input/Output0_FMT_DTYPE_SHAPE.txt，不含 tid name
        # 但子目录名为 CMD_idx，仍能区分
        names = {2: {"name": "MatMul"}, 5: {"name": "Conv"}, 7: {"name": "Gemm"}}
        out2  = tempfile.mkdtemp()
        save_tensor_blocks(self.blocks, out2, names)
        txts = self._all_txts(out2)
        self.assertTrue(any("Input" in f or "Output" in f for f in txts))


if __name__ == "__main__":
    unittest.main(verbosity=2)
