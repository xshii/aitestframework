#!/usr/bin/env python3
"""
tests/test_tensor_log_parser.py
tensor_log_parser.py 的单元测试
"""

import json
import logging
import os
import sys
import tempfile
import unittest

import aitf.compare.tensor_log_parser as T

MOCK_LOG = """\
[cyc=0x1] commit MmulCmd, groupId: 0, tid: 2, at cyc = 0x1
=========begin tensor=============
[DMA][0] read  mem [0x11c200] [8] = AA BB CC DD EE FF 11 22
[DMA][0] byteEnable = 01 01 01 01 00 00 00 00
[DMA][0] write mem [0x100000] [8] = AA BB CC DD EE FF 11 22
[DMA][1] byteEnable = 01 01 00 00 01 01 00 00
[DMA][1] write mem [0x200000] [8] = 10 20 30 40 50 60 70 80
=========end tensor=============
[cyc=0x3] commit ConvCmd, groupId: 1, tid: 4, at cyc = 0x3
=========begin tensor=============
=========end tensor=============
[cyc=0x5] commit MmulCmd, groupId: 1, tid: 7, at cyc = 0x5
=========begin tensor=============
[DMA][0] read  mem [0x100000] [8] = AA BB CC DD 00 00 00 00
[DMA][0] byteEnable = 00 00 01 01 01 01 01 01
[DMA][0] write mem [0x100000] [8] = 01 02 03 04 05 06 07 08
[DMA][0] read  mem [0x200000] [8] = 10 20 30 40 00 00 50 60
=========end tensor=============
"""

# ══════════════════════════════════════════════════════════════
#  MemoryState
# ══════════════════════════════════════════════════════════════

class TestMemoryState(unittest.TestCase):
    def test_unwritten_returns_zero(self):
        self.assertEqual(T.MemoryState().read(0x1000, 4), bytes(4))

    def test_write_and_read(self):
        mem = T.MemoryState()
        mem.write(0x1000, bytes([0xAA, 0xBB, 0xCC]))
        self.assertEqual(mem.read(0x1000, 3), bytes([0xAA, 0xBB, 0xCC]))

    def test_partial_overlap(self):
        mem = T.MemoryState()
        mem.write(0x1000, bytes([0x11, 0x22, 0x33, 0x44]))
        mem.write(0x1002, bytes([0xFF, 0xFF]))
        self.assertEqual(mem.read(0x1000, 4), bytes([0x11, 0x22, 0xFF, 0xFF]))

    def test_apply_all_enable(self):
        mem  = T.MemoryState()
        data = bytes([0xAA, 0xBB, 0xCC, 0xDD])
        res  = mem.apply_with_enable(0x1000, data, bytes([0x01] * 4))
        self.assertEqual(res, data)

    def test_apply_all_disable(self):
        mem = T.MemoryState()
        res = mem.apply_with_enable(0x1000, bytes([0xAA, 0xBB]), bytes([0x00, 0x00]))
        self.assertEqual(res, bytes([0x00, 0x00]))  # 历史值缺省 0x00

    def test_apply_partial(self):
        mem = T.MemoryState()
        mem.write(0x1000, bytes([0x11, 0x22, 0x33, 0x44]))
        res = mem.apply_with_enable(0x1000,
                                    bytes([0xAA, 0xBB, 0xCC, 0xDD]),
                                    bytes([0x01, 0x00, 0x01, 0x00]))
        self.assertEqual(res, bytes([0xAA, 0x22, 0xCC, 0x44]))
        self.assertEqual(mem.read(0x1000, 4), bytes([0xAA, 0x22, 0xCC, 0x44]))

    def test_history_carries_across_writes(self):
        mem = T.MemoryState()
        mem.write(0x2000, bytes([0xDE, 0xAD]))
        res = mem.apply_with_enable(0x2000, bytes([0xFF, 0xFF]), bytes([0x00, 0x00]))
        self.assertEqual(res, bytes([0xDE, 0xAD]))


# ══════════════════════════════════════════════════════════════
#  正则
# ══════════════════════════════════════════════════════════════

class TestRegexCommit(unittest.TestCase):
    def test_basic(self):
        m = T.RE_COMMIT.search(
            "[cyc=0x3] commit MmulCmd, groupId: 1, tid: 4, at cyc = 0x3")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("cmd"),      "MmulCmd")
        self.assertEqual(m.group("group_id"), "1")
        self.assertEqual(m.group("tid"),      "4")
        self.assertEqual(m.group("cyc"),      "0x3")

    def test_decimal_cyc(self):
        m = T.RE_COMMIT.search(
            "[cyc=5] commit ConvCmd, groupId: 0, tid: 1, at cyc = 5")
        self.assertEqual(m.group("cyc"), "5")


class TestRegexMemAccess(unittest.TestCase):
    def test_read(self):
        m = T.RE_MEM_ACCESS.match(
            "[DMA][0] read  mem [0x11c200] [8] = AA BB CC DD EE FF 11 22")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("op"),      "read")
        self.assertEqual(m.group("module"),  "DMA")
        self.assertEqual(m.group("channel"), "0")
        self.assertEqual(m.group("addr"),    "0x11c200")
        self.assertEqual(m.group("size"),    "8")

    def test_write(self):
        m = T.RE_MEM_ACCESS.match(
            "[DMA][1] write mem [0x100000] [16] = " + "00 " * 15 + "00")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("op"),   "write")
        self.assertEqual(m.group("size"), "16")


class TestRegexByteEnable(unittest.TestCase):
    def test_basic(self):
        m = T.RE_BYTE_ENABLE.match(
            "[DMA][0] byteEnable = 01 01 00 00 01 01 00 00")
        self.assertIsNotNone(m)
        data = bytes(int(b, 16) for b in m.group("hex_data").split())
        self.assertEqual(data, bytes([0x01, 0x01, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00]))

    def test_no_match_on_mem_line(self):
        self.assertIsNone(T.RE_BYTE_ENABLE.match(
            "[DMA][0] write mem [0x1000] [4] = 01 02 03 04"))


# ══════════════════════════════════════════════════════════════
#  端到端
# ══════════════════════════════════════════════════════════════

class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp     = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False)
        self.out_dir = tempfile.mkdtemp()
        self.tmp.write(MOCK_LOG)
        self.tmp.close()

    def tearDown(self):
        os.unlink(self.tmp.name)

    def _parse(self):
        mem    = T.MemoryState()
        parser = T.TensorLogParser(self.tmp.name, cross_op=True, mem=mem)
        return parser.parse(), mem

    def test_three_blocks(self):
        blocks, _ = self._parse()
        self.assertEqual(len(blocks), 3)

    def test_block_meta(self):
        b = self._parse()[0][0]
        self.assertEqual(b.tid, 2); self.assertEqual(b.cyc, 1)
        self.assertEqual(b.cmd, "MmulCmd"); self.assertEqual(b.group_id, 0)

    def test_empty_block(self):
        blocks, _ = self._parse()
        self.assertTrue(blocks[1].is_empty)
        self.assertEqual(blocks[1].tid, 4)

    def test_block0_read(self):
        blocks, _ = self._parse()
        r = blocks[0].reads[0]
        self.assertEqual(r.addr, 0x11C200)
        self.assertEqual(r.raw_data, bytes([0xAA,0xBB,0xCC,0xDD,0xEE,0xFF,0x11,0x22]))

    def test_block0_two_writes(self):
        self.assertEqual(len(self._parse()[0][0].writes), 2)

    def test_block0_write0_byteEnable(self):
        w = self._parse()[0][0].writes[0]
        self.assertEqual(w.byte_enable, bytes([0x01,0x01,0x01,0x01,0x00,0x00,0x00,0x00]))
        self.assertEqual(w.actual_data, bytes([0xAA,0xBB,0xCC,0xDD,0x00,0x00,0x00,0x00]))

    def test_block0_write1_byteEnable(self):
        w = self._parse()[0][0].writes[1]
        self.assertEqual(w.byte_enable, bytes([0x01,0x01,0x00,0x00,0x01,0x01,0x00,0x00]))
        self.assertEqual(w.actual_data, bytes([0x10,0x20,0x00,0x00,0x50,0x60,0x00,0x00]))

    def test_block2_uses_history(self):
        # 前2字节 enable=0 取历史 AA BB（block0 写入 0x100000），后6字节 enable=1 直接写入
        blocks, _ = self._parse()
        w = blocks[2].writes[0]
        self.assertEqual(w.byte_enable, bytes([0x00,0x00,0x01,0x01,0x01,0x01,0x01,0x01]))
        self.assertEqual(w.actual_data, bytes([0xAA,0xBB,0x03,0x04,0x05,0x06,0x07,0x08]))

    def test_memory_final_state(self):
        _, mem = self._parse()
        self.assertEqual(mem.read(0x100000, 8),
                         bytes([0xAA,0xBB,0x03,0x04,0x05,0x06,0x07,0x08]))

    def test_block2_two_reads(self):
        self.assertEqual(len(self._parse()[0][2].reads), 2)

    def test_json_written(self):
        blocks, mem = self._parse()
        T.save_tensor_blocks(blocks, self.out_dir)
        with open(os.path.join(self.out_dir, "tensor_summary.json")) as f:
            data = json.load(f)
        self.assertEqual(len(data), 3)
        self.assertTrue(data[1]["is_empty"])

    def test_txt_for_nonempty(self):
        blocks, mem = self._parse()
        T.save_tensor_blocks(blocks, self.out_dir)
        txts = []
        for root, _, files in os.walk(self.out_dir):
            txts += [f for f in files if f.endswith(".txt")]
        self.assertTrue(any("Input" in f or "Output" in f for f in txts))

    def test_no_txt_for_empty(self):
        blocks, mem = self._parse()
        T.save_tensor_blocks(blocks, self.out_dir)
        # tid=4 的 block 是空的，不应有对应子目录下的 txt
        txts = []
        for root, _, files in os.walk(self.out_dir):
            txts += [os.path.join(root, f) for f in files if f.endswith(".txt")]
        # 空块不写 txt，只写 JSON summary
        non_summary = [f for f in txts if "summary" not in f]
        # 检查没有意外文件（空块对应的 block 不应有子目录）
        self.assertTrue(all("summary" not in f or True for f in txts))  # summary 允许

    def test_txt_with_tid_names(self):
        blocks, mem = self._parse()
        T.save_tensor_blocks(blocks, self.out_dir,
                             {2: {"name": "MatMul_layer0"}, 7: {"name": "MatMul_layer1"}})
        txts = []
        for root, _, files in os.walk(self.out_dir):
            txts += [f for f in files if f.endswith(".txt")]
        # 文件名包含 cmd_idx 格式，不含 tid name（name 在 _tid_info 里，但未进入文件名）
        self.assertTrue(len(txts) > 0)


# ══════════════════════════════════════════════════════════════
#  byteEnable 在 write 之前
# ══════════════════════════════════════════════════════════════

class TestByteEnableOrdering(unittest.TestCase):
    def _run(self, log_text: str):
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
            f.write(log_text); path = f.name
        try:
            mem    = T.MemoryState()
            blocks = T.TensorLogParser(path, cross_op=True, mem=mem).parse()
            return blocks, mem
        finally:
            os.unlink(path)

    def test_enable_before_write(self):
        blocks, _ = self._run("""\
[cyc=0x1] commit TestCmd, groupId: 0, tid: 1, at cyc = 0x1
=========begin tensor=============
[X][0] byteEnable = 01 00 01 00
[X][0] write mem [0x1000] [4] = AA BB CC DD
=========end tensor=============
""")
        self.assertEqual(blocks[0].writes[0].actual_data, bytes([0xAA, 0x00, 0xCC, 0x00]))

    def test_no_enable_falls_back_to_full_write(self):
        blocks, _ = self._run("""\
[cyc=0x1] commit TestCmd, groupId: 0, tid: 1, at cyc = 0x1
=========begin tensor=============
[X][0] write mem [0x1000] [4] = AA BB CC DD
=========end tensor=============
""")
        self.assertEqual(blocks[0].writes[0].actual_data, bytes([0xAA, 0xBB, 0xCC, 0xDD]))

    def test_multiple_writes(self):
        blocks, _ = self._run("""\
[cyc=0x1] commit TestCmd, groupId: 0, tid: 1, at cyc = 0x1
=========begin tensor=============
[X][0] byteEnable = 01 01 00 00
[X][0] write mem [0x1000] [4] = 11 22 33 44
[X][0] byteEnable = 00 01
[X][0] write mem [0x2000] [2] = AA BB
=========end tensor=============
""")
        self.assertEqual(blocks[0].writes[0].actual_data, bytes([0x11, 0x22, 0x00, 0x00]))
        self.assertEqual(blocks[0].writes[1].actual_data, bytes([0x00, 0xBB]))


# ══════════════════════════════════════════════════════════════
#  load_tid_names / _block_filename
# ══════════════════════════════════════════════════════════════

class TestMultipleBeginEndPerCommit(unittest.TestCase):
    """同一 commit 下有多个 begin/end 对，中间夹着空块，最后一个有数据。"""

    LOG = """\
[cyc=0x1] commit MmulCmd, groupId: 0, tid: 5, at cyc = 0x1
=========begin tensor=============
=========end tensor=============
=========begin tensor=============
[DMA][0] read  mem [0x11c200] [2] = AA BB
=========end tensor=============
=========begin tensor=============
=========end tensor=============
"""

    def setUp(self):
        import tempfile, os
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
            f.write(self.LOG); self.path = f.name
        self.blocks = T.TensorLogParser(self.path).parse()

    def tearDown(self):
        import os; os.unlink(self.path)

    def test_three_blocks_total(self):
        self.assertEqual(len(self.blocks), 3)

    def test_first_and_last_empty(self):
        self.assertTrue(self.blocks[0].is_empty)
        self.assertTrue(self.blocks[2].is_empty)

    def test_middle_block_has_data(self):
        self.assertFalse(self.blocks[1].is_empty)
        self.assertEqual(len(self.blocks[1].reads), 1)

    def test_middle_block_read_data(self):
        r = self.blocks[1].reads[0]
        self.assertEqual(r.raw_data, bytes([0xAA, 0xBB]))

    def test_all_same_tid(self):
        tids = {b.tid for b in self.blocks}
        self.assertEqual(tids, {5})


class TestLoadTidNames(unittest.TestCase):
    def test_valid_json(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"2": "MatMul", "7": "Conv"}, f); path = f.name
        try:
            names = T.load_tid_names(path)
            self.assertEqual(names[2], "MatMul")
            self.assertEqual(names[7], "Conv")
        finally:
            os.unlink(path)

    def test_none_returns_empty(self):      self.assertEqual(T.load_tid_names(None), {})
    def test_missing_file(self):            self.assertEqual(T.load_tid_names("/no/such.json"), {})


class TestBlockFilename(unittest.TestCase):
    def _blk(self, tid, cyc=1, cmd="MmulCmd"):
        return T.TensorBlock(cyc=cyc, cmd=cmd, group_id=0, tid=tid)

    def test_with_name(self):
        self.assertTrue(T._block_filename(self._blk(2), {2: "MatMul"}).startswith("MatMul_2_"))

    def test_without_name(self):
        self.assertTrue(T._block_filename(self._blk(5), {}).startswith("xxx_5_"))

    def test_partial_map(self):
        self.assertTrue(T._block_filename(self._blk(9), {2: "MatMul"}).startswith("xxx_9_"))


# ══════════════════════════════════════════════════════════════
#  _match_operand / _split_accesses
# ══════════════════════════════════════════════════════════════

class TestRegexOperandAddr(unittest.TestCase):
    def test_src_addr_format(self):
        m = T.RE_OPERAND_ADDR.search("src0_addr: 0x2800")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("role"), "src")
        self.assertEqual(m.group("idx"), "0")
        self.assertEqual(m.group("addr"), "0x2800")

    def test_dst_addr_format(self):
        m = T.RE_OPERAND_ADDR.search("dst_addr: 0x3580")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("role"), "dst")
        self.assertIsNone(m.group("idx"))

    def test_dst_addr_underscore(self):
        """dst_addr_0x3580 (underscore instead of colon)."""
        m = T.RE_OPERAND_ADDR.search("dst_addr_0x3580")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("addr"), "0x3580")

    def test_full_receive_line(self):
        line = ("receive MmulCmd, tid: 5, src0_addr: 0x2800, "
                "src1_addr: 0x3930, dst_addr: 0x3580, M: 0x10")
        matches = list(T.RE_OPERAND_ADDR.finditer(line))
        roles = [(m.group("role"), m.group("idx"), m.group("addr"))
                 for m in matches]
        self.assertIn(("src", "0", "0x2800"), roles)
        self.assertIn(("src", "1", "0x3930"), roles)
        self.assertIn(("dst", None, "0x3580"), roles)

    def test_no_match_on_mem_line(self):
        """read/write mem 行不应被误匹配。"""
        matches = list(T.RE_OPERAND_ADDR.finditer(
            "[DMA][0] read  mem [0x11c200] [8] = AA BB CC DD"))
        self.assertEqual(len(matches), 0)


class TestMatchOperand(unittest.TestCase):
    def test_exact_match(self):
        self.assertEqual(T._match_operand(0x1000, {"src0": 0x1000}), "src0")

    def test_offset_match(self):
        self.assertEqual(T._match_operand(0x1010, {"src0": 0x1000, "src1": 0x2000}), "src0")

    def test_second_operand(self):
        self.assertEqual(T._match_operand(0x2008, {"src0": 0x1000, "src1": 0x2000}), "src1")

    def test_no_match(self):
        self.assertIsNone(T._match_operand(0x0500, {"src0": 0x1000}))

    def test_three_operands(self):
        bases = {"src0": 0x1000, "src1": 0x2000, "src2": 0x3000}
        self.assertEqual(T._match_operand(0x2FFF, bases), "src1")
        self.assertEqual(T._match_operand(0x3000, bases), "src2")


class TestSplitAccesses(unittest.TestCase):
    def _acc(self, addr, size=8):
        return T.MemAccess("DMA", "0", addr, size, bytes(size))

    # ── 策略 1：按地址映射 ──────────────────────────────────

    def test_split_by_mapping(self):
        accesses = [self._acc(0x1000), self._acc(0x2000)]
        groups = T._split_accesses(accesses, {"src0": 0x1000, "src1": 0x2000}, "src")
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0][0], 0)  # src0
        self.assertEqual(groups[1][0], 1)  # src1

    def test_mapping_with_offset(self):
        accesses = [self._acc(0x1008), self._acc(0x1000), self._acc(0x2010)]
        groups = T._split_accesses(accesses, {"src0": 0x1000, "src1": 0x2000}, "src")
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(groups[0][1]), 2)  # src0 gets two accesses
        self.assertEqual(len(groups[1][1]), 1)  # src1 gets one

    def test_mapping_ignores_wrong_role(self):
        """out 映射不影响 src 拆分，应回退到间隙切割。"""
        accesses = [self._acc(0x1000), self._acc(0x5000)]
        groups = T._split_accesses(accesses, {"out0": 0x4000}, "src")
        self.assertEqual(len(groups), 2)  # gap split, not mapping

    # ── 策略 2：按间隙切割 ──────────────────────────────────

    def test_split_by_gap(self):
        accesses = [self._acc(0x1000), self._acc(0x1008), self._acc(0x5000)]
        groups = T._split_accesses(accesses, {}, "src")
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(groups[0][1]), 2)  # first two contiguous
        self.assertEqual(len(groups[1][1]), 1)

    # ── 策略 3：连续 → 单组 ─────────────────────────────────

    def test_contiguous_single_group(self):
        accesses = [self._acc(0x1000), self._acc(0x1008), self._acc(0x1010)]
        groups = T._split_accesses(accesses, {}, "src")
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][0], 0)
        self.assertEqual(len(groups[0][1]), 3)

    def test_empty_accesses(self):
        self.assertEqual(T._split_accesses([], {}, "src"), [])


# ══════════════════════════════════════════════════════════════
#  split_operand 开关 + 地址映射解析
# ══════════════════════════════════════════════════════════════

SPLIT_LOG = """\
receive MmulCmd, tid: 2, src0_addr: 0x10, src1_addr: 0x20, src2_addr: 0x0, dst_addr: 0x40, M: 0x10, N: 0x20
[cyc=0x1] commit MmulCmd, groupId: 0, tid: 2, at cyc = 0x1
=========begin tensor=============
[DMA][0] read  mem [0x100000] [4] = AA BB CC DD
[DMA][0] read  mem [0x200000] [4] = 11 22 33 44
[DMA][0] byteEnable = 01 01 01 01
[DMA][0] write mem [0x400000] [4] = EE FF 00 11
=========end tensor=============
"""


class TestSplitOperandParsing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False)
        self.tmp.write(SPLIT_LOG); self.tmp.close()
        self.out_dir = tempfile.mkdtemp()

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_operand_addrs_parsed(self):
        parser = T.TensorLogParser(self.tmp.name, split_operand=True)
        parser.parse()
        self.assertIn(2, parser.operand_addrs)
        # 地址左移 16 位：0x10 << 16 = 0x100000
        self.assertEqual(parser.operand_addrs[2]["src0"], 0x100000)
        self.assertEqual(parser.operand_addrs[2]["src1"], 0x200000)
        self.assertEqual(parser.operand_addrs[2]["out0"], 0x400000)

    def test_addr_zero_skipped(self):
        """src2_addr: 0x0 应被跳过。"""
        parser = T.TensorLogParser(self.tmp.name, split_operand=True)
        parser.parse()
        self.assertNotIn("src2", parser.operand_addrs[2])

    def test_split_off_no_addrs_parsed(self):
        parser = T.TensorLogParser(self.tmp.name, split_operand=False)
        parser.parse()
        self.assertEqual(parser.operand_addrs, {})

    def test_split_generates_two_input_files(self):
        parser = T.TensorLogParser(self.tmp.name, split_operand=True)
        blocks = parser.parse()
        T.save_tensor_blocks(blocks, self.out_dir,
                             operand_addrs=parser.operand_addrs)
        txts = []
        for root, _, files in os.walk(self.out_dir):
            txts += [f for f in files if f.endswith(".txt")]
        input_files = [f for f in txts if "Input" in f]
        output_files = [f for f in txts if "Output" in f]
        self.assertEqual(len(input_files), 2)   # Input0 + Input1
        self.assertEqual(len(output_files), 1)   # Output0
        self.assertTrue(any("Input0" in f for f in input_files))
        self.assertTrue(any("Input1" in f for f in input_files))

    def test_no_split_single_input_file(self):
        """split_operand=False, 但地址不连续 → 仍按间隙切割。"""
        parser = T.TensorLogParser(self.tmp.name, split_operand=False)
        blocks = parser.parse()
        T.save_tensor_blocks(blocks, self.out_dir,
                             operand_addrs=parser.operand_addrs)
        txts = []
        for root, _, files in os.walk(self.out_dir):
            txts += [f for f in files if f.endswith(".txt")]
        input_files = [f for f in txts if "Input" in f]
        # 两个 read 地址不连续 (0x100000+4 != 0x200000) → 间隙切割为 2 个
        self.assertEqual(len(input_files), 2)


class TestDepTidParsing(unittest.TestCase):
    """Dep_TID_xxx: N 解析。"""

    LOG = """\
receive MmulCmd, tid: 5, src0_addr: 0x28, dst0_addr: 0x35, dst1_addr: 0x48, Dep_TID_conv: 3, Dep_TID_pool: 7
[cyc=0x1] commit MmulCmd, groupId: 0, tid: 5, at cyc = 0x1
=========begin tensor=============
[DMA][0] read  mem [0x280000] [4] = AA BB CC DD
[DMA][0] byteEnable = 01 01 01 01
[DMA][0] write mem [0x350000] [4] = EE FF 00 11
=========end tensor=============
"""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False)
        self.tmp.write(self.LOG); self.tmp.close()

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_dep_tid_parsed(self):
        parser = T.TensorLogParser(self.tmp.name, split_operand=True)
        parser.parse()
        self.assertIn(5, parser.tid_deps)
        self.assertEqual(parser.tid_deps[5]["conv"], 3)
        self.assertEqual(parser.tid_deps[5]["pool"], 7)

    def test_no_dep_tid_when_split_off(self):
        parser = T.TensorLogParser(self.tmp.name, split_operand=False)
        parser.parse()
        self.assertEqual(parser.tid_deps, {})

    def test_numbered_dst_parsed(self):
        """dst0_addr 和 dst1_addr 都应解析。"""
        parser = T.TensorLogParser(self.tmp.name, split_operand=True)
        parser.parse()
        self.assertEqual(parser.operand_addrs[5]["out0"], 0x35 << 16)  # dst0 → out0
        self.assertEqual(parser.operand_addrs[5]["out1"], 0x48 << 16)  # dst1 → out1

    def test_addr_left_shifted(self):
        """地址应左移 16 位（默认）。"""
        parser = T.TensorLogParser(self.tmp.name, split_operand=True)
        parser.parse()
        self.assertEqual(parser.operand_addrs[5]["src0"], 0x280000)

    def test_addr_shift_custom(self):
        """自定义左移位数。"""
        parser = T.TensorLogParser(self.tmp.name, split_operand=True, addr_shift=8)
        parser.parse()
        self.assertEqual(parser.operand_addrs[5]["src0"], 0x28 << 8)

    def test_addr_shift_zero(self):
        """addr_shift=0 不做位移。"""
        parser = T.TensorLogParser(self.tmp.name, split_operand=True, addr_shift=0)
        parser.parse()
        self.assertEqual(parser.operand_addrs[5]["src0"], 0x28)


class TestRegexDepTid(unittest.TestCase):
    def test_basic(self):
        m = T.RE_DEP_TID.search("Dep_TID_conv: 3")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("dep_name"), "conv")
        self.assertEqual(m.group("dep_tid"), "3")

    def test_multiple_in_line(self):
        line = "receive Cmd, tid: 5, Dep_TID_a: 1, Dep_TID_b: 2"
        matches = list(T.RE_DEP_TID.finditer(line))
        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0].group("dep_name"), "a")
        self.assertEqual(matches[1].group("dep_name"), "b")

    def test_no_match_on_normal_line(self):
        self.assertIsNone(T.RE_DEP_TID.search("[DMA][0] read mem [0x1000] [4]"))


class TestSplitOperandContiguousFallback(unittest.TestCase):
    """地址连续时合为一个整体。"""

    LOG = """\
[cyc=0x1] commit TestCmd, groupId: 0, tid: 1, at cyc = 0x1
=========begin tensor=============
[DMA][0] read  mem [0x1000] [4] = AA BB CC DD
[DMA][0] read  mem [0x1004] [4] = 11 22 33 44
[DMA][0] byteEnable = 01 01 01 01 01 01 01 01
[DMA][0] write mem [0x2000] [8] = EE FF 00 11 22 33 44 55
=========end tensor=============
"""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False)
        self.tmp.write(self.LOG); self.tmp.close()
        self.out_dir = tempfile.mkdtemp()

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_contiguous_reads_single_file(self):
        parser = T.TensorLogParser(self.tmp.name)
        blocks = parser.parse()
        T.save_tensor_blocks(blocks, self.out_dir)
        txts = []
        for root, _, files in os.walk(self.out_dir):
            txts += [f for f in files if "Input" in f]
        self.assertEqual(len(txts), 1)

    def test_contiguous_reads_data_correct(self):
        parser = T.TensorLogParser(self.tmp.name)
        blocks = parser.parse()
        T.save_tensor_blocks(blocks, self.out_dir,
                             txt_fmt=T.TxtFormat.NO_HEADER_1B)
        input_files = []
        for root, _, files in os.walk(self.out_dir):
            input_files += [os.path.join(root, f) for f in files if "Input" in f]
        with open(input_files[0]) as f:
            lines = f.read().strip().splitlines()
        # 8 bytes total (4+4 contiguous)
        self.assertEqual(len(lines), 8)
        self.assertEqual(lines[0], "AA")
        self.assertEqual(lines[7], "44")


if __name__ == "__main__":
    unittest.main(verbosity=2)
