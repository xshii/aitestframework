#!/usr/bin/env python3
"""
tests/test_integration_op.py
op dump 日志集成测试：3个算子 MMUL → MMULADDR → MMUL

重点验证：
  - 同名 opcode (MMUL) 的 op_idx 正确区分（0 和 2）
  - 中间算子 MMULADDR (op_idx=1) 数据隔离
  - 每个算子 tid 正确绑定
  - copy_data_to_ram 输入/输出路由在多算子场景下独立判断
  - 多字节 RE_INPUT_DATA 展开
  - 多输入多输出
"""

import os, sys, tempfile, unittest
from aitf.compare.op_parser_state import OpStateMachineParser
from aitf.compare.op_log_parser import save_op, _resolve_fmt

# ══════════════════════════════════════════════════════════════
#  完整 mock 日志：3 个算子
# ══════════════════════════════════════════════════════════════
#
#  算子 0 MMUL (tid=2)
#    src0: NCHW INT8  → 输入（data_init）；copy_op0 有 data_init → 丢弃
#    src1: NHWC INT8  → 输入（data_init）
#    src4: NCHW INT8  → 输出（copy_op4，无 data_init）
#
#  算子 1 MMULADDR (tid=5)
#    src0: NCHW FP16  → 输入（data_init）
#    src2: NCHW FP16  → 输出（copy_op2）
#    src3: NCHW FP16  → 输出（copy_op3）
#
#  算子 2 MMUL (tid=9)    ← 同名，op_idx 应为 2
#    src0: NCHW INT8  → 输入（data_init，2 行多字节）
#    src4: NHWC INT8  → 输出（copy_op4）

LOG = """\
some system log line
word0: tid = 2
other irrelevant line
[dump_cfg] opcode = MMUL, m = 2, k = 4, n = 4, loop = 1
[dump_cfg] src0: NCHW, INT8, base_addr = 0x100000
[dump_cfg] src1: NHWC, INT8, base_addr = 0x200000
[dump_cfg] src4: NCHW, INT8, base_addr = 0x300000
[data_init]./xxx.cpp:10: init_op0[loop1][2][4] (INT8) from ram
loop0, addr[0][0] = 0x100000, data = 0x01 02 03 04
loop0, addr[1][0] = 0x100100, data = 0x05 06 07 08
[data_init]./xxx.cpp:11: init_op1[loop1][2][4] (INT8) from ram
loop0, addr[0][0] = 0x200000, data = 0xA1 A2 A3 A4
loop0, addr[1][0] = 0x200100, data = 0xA5 A6 A7 A8
[copy_data_to_ram]./xxx.cpp:12: copy_op0(NCHW), matrix_offset = 0x0, write_length = 1,
loop0, blk0-0, addr[0][0] = 0x100000, data = 0x01(1)
loop0, blk0-0, addr[0][1] = 0x100001, data = 0x02(2)
loop0, blk0-0, addr[1][0] = 0x180100, data = 0x05(5)
loop0, blk0-0, addr[1][1] = 0x180101, data = 0x06(6)
[copy_data_to_ram]./xxx.cpp:13: copy_op4(NCHW), matrix_offset = 0x0, write_length = 1,
loop0, blk0-0, addr[0][0] = 0x300000, data = 0xE0(224)
loop0, blk0-0, addr[0][1] = 0x300001, data = 0xE1(225)
loop0, blk0-0, addr[0][2] = 0x300002, data = 0xE2(226)
loop0, blk0-0, addr[0][3] = 0x300003, data = 0xE3(227)
loop0, blk0-0, addr[0][0] = 0x380000, data = 0x0A(10)
loop0, blk0-0, addr[0][1] = 0x380001, data = 0x0B(11)
loop0, blk0-0, addr[0][2] = 0x380002, data = 0x0C(12)
loop0, blk0-0, addr[0][3] = 0x380003, data = 0x0D(13)
word1: tid = 5
[dump_cfg] opcode = MMULADDR, m = 1, k = 2, n = 2, loop = 1
[dump_cfg] src0: NCHW, FP16, base_addr = 0x400000
[dump_cfg] src2: NCHW, FP16, base_addr = 0x500000
[dump_cfg] src3: NCHW, FP16, base_addr = 0x600000
[data_init]./xxx.cpp:20: init_op0[loop1][1][2] (FP16) from ram
loop0, addr[0][0] = 0x400000, data = 0xBE EF
[copy_data_to_ram]./xxx.cpp:21: copy_op2(NCHW), matrix_offset = 0x0, write_length = 1,
loop0, blk0-0, addr[0][0] = 0x500000, data = 0xAA(170)
loop0, blk0-0, addr[0][1] = 0x500001, data = 0xBB(187)
loop0, blk0-0, addr[0][0] = 0x580000, data = 0x11(17)
loop0, blk0-0, addr[0][1] = 0x580001, data = 0x22(34)
[copy_data_to_ram]./xxx.cpp:22: copy_op3(NCHW), matrix_offset = 0x0, write_length = 1,
loop0, blk0-0, addr[0][0] = 0x600000, data = 0xCC(204)
loop0, blk0-0, addr[0][1] = 0x600001, data = 0xDD(221)
loop0, blk0-0, addr[0][0] = 0x680000, data = 0x33(51)
loop0, blk0-0, addr[0][1] = 0x680001, data = 0x44(68)
word2: tid = 9
[dump_cfg] opcode = MMUL, m = 2, k = 4, n = 2, loop = 1
[dump_cfg] src0: NCHW, INT8, base_addr = 0x700000
[dump_cfg] src4: NHWC, INT8, base_addr = 0x800000
[data_init]./xxx.cpp:30: init_op0[loop1][2][4] (INT8) from ram
loop0, addr[0][0] = 0x700000, data = 0xC1 C2 C3 C4
loop0, addr[1][0] = 0x700100, data = 0xC5 C6 C7 C8
[copy_data_to_ram]./xxx.cpp:31: copy_op4(NHWC), matrix_offset = 0x0, write_length = 1,
loop0, blk0-0, addr[0][0] = 0x800000, data = 0xA0(160)
loop0, blk0-0, addr[0][1] = 0x800001, data = 0xA1(161)
loop0, blk0-0, addr[0][0] = 0x880000, data = 0xD1(209)
loop0, blk0-0, addr[0][1] = 0x880001, data = 0xD2(210)
"""


def _parse():
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
        f.write(LOG); path = f.name
    try:
        return OpStateMachineParser(path).parse()
    finally:
        os.unlink(path)


# ══════════════════════════════════════════════════════════════
#  基本结构
# ══════════════════════════════════════════════════════════════

class TestOpStructure(unittest.TestCase):
    def setUp(self): self.blocks = _parse()

    def test_three_blocks(self):
        self.assertEqual(len(self.blocks), 3)

    def test_opcodes(self):
        self.assertEqual([b.opcode for b in self.blocks], ["MMUL", "MMULADDR", "MMUL"])

    def test_op_idx_sequence(self):
        self.assertEqual([b.op_idx for b in self.blocks], [0, 1, 2])

    def test_tids(self):
        self.assertEqual([b.tid for b in self.blocks], [2, 5, 9])

    def test_second_mmul_is_op_idx_2(self):
        mmuls = [b for b in self.blocks if b.opcode == "MMUL"]
        self.assertEqual(mmuls[0].op_idx, 0)
        self.assertEqual(mmuls[1].op_idx, 2)


# ══════════════════════════════════════════════════════════════
#  算子 0 MMUL (tid=2)
# ══════════════════════════════════════════════════════════════

class TestOp0MMUL(unittest.TestCase):
    def setUp(self): self.op = _parse()[0]

    def test_two_inputs(self):
        self.assertEqual(len(self.op.inputs), 2)

    def test_one_output(self):
        # copy_op0 有 data_init src0 → 丢弃；只有 copy_op4 是输出
        self.assertEqual(len(self.op.outputs), 1)

    def test_input_src_indices(self):
        idxs = {i.src_idx for i in self.op.inputs}
        self.assertEqual(idxs, {0, 1})

    def test_input0_bytes(self):
        inp = next(i for i in self.op.inputs if i.src_idx == 0)
        # data_init 2行，全部为 input
        self.assertEqual(inp.raw_bytes, bytes([0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08]))

    def test_input0_shape(self):
        inp = next(i for i in self.op.inputs if i.src_idx == 0)
        self.assertEqual(inp.shape, [1, 2, 4])

    def test_input1_bytes(self):
        inp = next(i for i in self.op.inputs if i.src_idx == 1)
        # data_init 2行，全部为 input
        self.assertEqual(inp.raw_bytes, bytes([0xA1,0xA2,0xA3,0xA4,0xA5,0xA6,0xA7,0xA8]))

    def test_output_is_src4(self):
        self.assertEqual(self.op.outputs[0].src_idx, 4)

    def test_output4_bytes(self):
        out4 = self.op.outputs[0]
        # copy_op4 8行，后4行为 output
        self.assertEqual(out4.raw_bytes, bytes([0x0A,0x0B,0x0C,0x0D]))

    def test_output4_shape(self):
        out4 = self.op.outputs[0]
        self.assertEqual(out4.shape, [1, 4])

    def test_output4_fmt(self):
        out = next(o for o in self.op.outputs if o.src_idx == 4)
        cfg = self.op.get_cfg(out.src_idx)
        fmt = _resolve_fmt([out.fmt_in_line, cfg.fmt if cfg else None])
        self.assertEqual(fmt, "NCHW")


# ══════════════════════════════════════════════════════════════
#  算子 1 MMULADDR (tid=5)
# ══════════════════════════════════════════════════════════════

class TestOp1MMULADDR(unittest.TestCase):
    def setUp(self): self.op = _parse()[1]

    def test_opcode(self):
        self.assertEqual(self.op.opcode, "MMULADDR")

    def test_one_input(self):
        self.assertEqual(len(self.op.inputs), 1)

    def test_two_outputs(self):
        self.assertEqual(len(self.op.outputs), 2)

    def test_input_src0_bytes(self):
        # data_init 1行，全部为 input
        self.assertEqual(self.op.inputs[0].raw_bytes, bytes([0xBE, 0xEF]))

    def test_input_src0_dtype(self):
        self.assertEqual(self.op.inputs[0].dtype, "FP16")

    def test_output_src_indices(self):
        idxs = {o.src_idx for o in self.op.outputs}
        self.assertEqual(idxs, {2, 3})

    def test_output2_bytes(self):
        out = next(o for o in self.op.outputs if o.src_idx == 2)
        # copy_op2 4行，后2行为 output
        self.assertEqual(out.raw_bytes, bytes([0x11, 0x22]))

    def test_output3_bytes(self):
        out = next(o for o in self.op.outputs if o.src_idx == 3)
        # copy_op3 4行，后2行为 output
        self.assertEqual(out.raw_bytes, bytes([0x33, 0x44]))

    def test_no_data_from_op0_leaking(self):
        inp = self.op.inputs[0]
        self.assertEqual(inp.raw_bytes, bytes([0xBE, 0xEF]))


# ══════════════════════════════════════════════════════════════
#  算子 2 MMUL (tid=9) - 同名第二个
# ══════════════════════════════════════════════════════════════

class TestOp2MMULSecond(unittest.TestCase):
    def setUp(self): self.op = _parse()[2]

    def test_op_idx_is_2(self):
        self.assertEqual(self.op.op_idx, 2)

    def test_tid_is_9(self):
        self.assertEqual(self.op.tid, 9)

    def test_one_input_one_output(self):
        self.assertEqual(len(self.op.inputs), 1)
        self.assertEqual(len(self.op.outputs), 1)

    def test_input_bytes(self):
        # data_init 2行，全部为 input
        self.assertEqual(self.op.inputs[0].raw_bytes,
                         bytes([0xC1,0xC2,0xC3,0xC4,0xC5,0xC6,0xC7,0xC8]))

    def test_output_bytes(self):
        # copy_op4 4行，后2行为 output
        self.assertEqual(self.op.outputs[0].raw_bytes,
                         bytes([0xD1,0xD2]))

    def test_output_fmt_nhwc(self):
        out = self.op.outputs[0]
        cfg = self.op.get_cfg(out.src_idx)
        fmt = _resolve_fmt([out.fmt_in_line, cfg.fmt if cfg else None])
        self.assertEqual(fmt, "NHWC")

    def test_no_data_from_op1_leaking(self):
        # op2 的 src0 数据不应包含 op1 的数据
        self.assertNotIn(0xBE, self.op.inputs[0].raw_bytes)


# ══════════════════════════════════════════════════════════════
#  文件输出
# ══════════════════════════════════════════════════════════════

class TestFileOutput(unittest.TestCase):
    def setUp(self):
        self.blocks  = _parse()
        self.out_dir = tempfile.mkdtemp()

    def test_files_created_for_all_ops(self):
        for op in self.blocks:
            import os
            sub = os.path.join(self.out_dir, f"{op.opcode}_{op.op_idx}")
            save_op(op, sub)
        # op0
        sub0 = os.path.join(self.out_dir, "MMUL_0")
        self.assertTrue(any("Input0"  in f for f in os.listdir(sub0)))
        self.assertTrue(any("Input1"  in f for f in os.listdir(sub0)))
        self.assertFalse(any("Output0" in f for f in os.listdir(sub0)))  # copy_op0 丢弃
        self.assertTrue(any("Output4" in f for f in os.listdir(sub0)))
        # op1
        sub1 = os.path.join(self.out_dir, "MMULADDR_1")
        self.assertTrue(any("Input0" in f for f in os.listdir(sub1)))
        self.assertTrue(any("Output2" in f for f in os.listdir(sub1)))
        self.assertTrue(any("Output3" in f for f in os.listdir(sub1)))
        # op2 - 同名 MMUL 但 idx=2
        sub2 = os.path.join(self.out_dir, "MMUL_2")
        self.assertTrue(any("Input0" in f for f in os.listdir(sub2)))
        self.assertTrue(any("Output4" in f for f in os.listdir(sub2)))

    def test_mmul_0_and_mmul_2_distinct_dirs(self):
        import os
        save_op(self.blocks[0], os.path.join(self.out_dir, "MMUL_0"))
        save_op(self.blocks[2], os.path.join(self.out_dir, "MMUL_2"))
        self.assertTrue(os.path.isdir(os.path.join(self.out_dir, "MMUL_0")))
        self.assertTrue(os.path.isdir(os.path.join(self.out_dir, "MMUL_2")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
