#!/usr/bin/env python3
"""
tests/test_op_parser_state.py
op_parser_state.py 单元测试
"""

import os
import sys
import tempfile
import unittest

from aitf.compare.op_parser_state import (
    OpStateMachineParser, ParserContext, State,
    RE_TID, RE_INPUT_DATA,
)
from aitf.compare.op_log_parser import RE_OUTPUT_DATA, OpBlock


# ══════════════════════════════════════════════════════════════
#  正则测试
# ══════════════════════════════════════════════════════════════

class TestRegexTid(unittest.TestCase):
    def test_word0(self):
        m = RE_TID.search("word0: tid = 2")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("tid"), "2")

    def test_word_multidigit(self):
        m = RE_TID.search("word12: tid = 99")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("tid"), "99")

    def test_with_surrounding_text(self):
        m = RE_TID.search("some prefix word3: tid = 5 suffix")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("tid"), "5")

    def test_no_match_on_other_line(self):
        self.assertIsNone(RE_TID.search("[dump_cfg] opcode = MMUL"))
        self.assertIsNone(RE_TID.search("tid = 3"))  # 无 word\d+: 前缀


class TestRegexInputDataNew(unittest.TestCase):
    def test_single_byte(self):
        m = RE_INPUT_DATA.match("loop0, addr[0][0] = 0x100200, data = 0x01")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("hex_vals"), "01")

    def test_multi_byte(self):
        m = RE_INPUT_DATA.match("loop0, addr[0][0] = 0x100200, data = 0x01 02 03 04")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("hex_vals"), "01 02 03 04")

    def test_loop_row_col(self):
        m = RE_INPUT_DATA.match("loop1, addr[3][5] = 0x200000, data = 0xAA BB")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("loop_idx"), "1")
        self.assertEqual(m.group("row_idx"),  "3")
        self.assertEqual(m.group("col_idx"),  "5")
        self.assertEqual(m.group("address"),  "0x200000")

    def test_no_match_old_format_with_dec(self):
        # 旧格式末尾有 (0)，不应匹配
        self.assertIsNone(
            RE_INPUT_DATA.match("loop0, addr[0][0] = 0x100200, data = 0x00(0)"))

    def test_no_match_output_line(self):
        # 输出行有 blk，不应匹配
        self.assertIsNone(
            RE_INPUT_DATA.match("loop0, blk0-0, addr[0][0] = 0x11C200, data = 0x00 01"))


# ══════════════════════════════════════════════════════════════
#  ParserContext 状态转换
# ══════════════════════════════════════════════════════════════

class TestParserContext(unittest.TestCase):
    def test_initial_state(self):
        ctx = ParserContext()
        self.assertEqual(ctx.state, State.IDLE)
        self.assertIsNone(ctx.pending_tid)
        self.assertIsNone(ctx.current_op)

    def test_transition_logs(self):
        ctx = ParserContext()
        ctx._transition(State.WAIT_OP, "test")
        self.assertEqual(ctx.state, State.WAIT_OP)

    def test_flush_input_empty(self):
        ctx = ParserContext()
        ctx.flush_input()   # 无 current_input，不应崩溃

    def test_flush_op_empty(self):
        ctx = ParserContext()
        ctx.flush_op()      # 无 current_op，不应崩溃


# ══════════════════════════════════════════════════════════════
#  辅助：从字符串创建临时日志文件
# ══════════════════════════════════════════════════════════════

def _parse_log(text: str) -> list[OpBlock]:
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
        f.write(text); path = f.name
    try:
        return OpStateMachineParser(path).parse()
    finally:
        os.unlink(path)


# ══════════════════════════════════════════════════════════════
#  基础端到端
# ══════════════════════════════════════════════════════════════

class TestBasicParsing(unittest.TestCase):
    LOG = """\
word0: tid = 5
[dump_cfg] opcode = MMUL, m = 2, k = 4, n = 2, loop = 1
[dump_cfg] src0: NCHW, INT8, base_addr = 0x100000
[dump_cfg] src4: NCHW, INT8, base_addr = 0x11da00
[data_init]./xxx.cpp:1: init_op0[loop1][2][4] (INT8) from ram
loop0, addr[0][0] = 0x100200, data = 0x01 02 03 04
loop0, addr[1][0] = 0x100300, data = 0x05 06 07 08
[copy_data_to_ram]./xxx.cpp:2: copy_op4(NCHW), matrix_offset = 0x0, write_length = 1,
loop0, blk0-0, addr[0][0] = 0x11C200, data = 0xE0(224)
loop0, blk0-0, addr[0][1] = 0x11C201, data = 0xE1(225)
loop0, blk0-0, addr[0][0] = 0x19C200, data = 0x0A(10)
loop0, blk0-0, addr[0][1] = 0x19C201, data = 0x0B(11)
"""

    def setUp(self):
        self.blocks = _parse_log(self.LOG)

    def test_one_block(self):
        self.assertEqual(len(self.blocks), 1)

    def test_opcode_and_tid(self):
        op = self.blocks[0]
        self.assertEqual(op.opcode, "MMUL")
        self.assertEqual(op.tid,    5)
        self.assertEqual(op.op_idx, 0)

    def test_cfg_params(self):
        p = self.blocks[0].cfg_params
        self.assertEqual(p["m"], 2)
        self.assertEqual(p["k"], 4)

    def test_one_input(self):
        self.assertEqual(len(self.blocks[0].inputs), 1)

    def test_input_src_idx(self):
        self.assertEqual(self.blocks[0].inputs[0].src_idx, 0)

    def test_input_shape(self):
        self.assertEqual(self.blocks[0].inputs[0].shape, [1, 2, 4])

    def test_input_bytes(self):
        # data_init 4行，前2行为 input
        self.assertEqual(self.blocks[0].inputs[0].raw_bytes,
                         bytes([0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08]))

    def test_one_output(self):
        self.assertEqual(len(self.blocks[0].outputs), 1)

    def test_output_src_idx(self):
        self.assertEqual(self.blocks[0].outputs[0].src_idx, 4)

    def test_output_bytes(self):
        # copy_op4 4行，后2行为 output
        self.assertEqual(self.blocks[0].outputs[0].raw_bytes,
                         bytes([0x0A,0x0B]))


# ══════════════════════════════════════════════════════════════
#  多算子，tid 触发 flush
# ══════════════════════════════════════════════════════════════

class TestMultipleOps(unittest.TestCase):
    LOG = """\
word0: tid = 3
[dump_cfg] opcode = MMUL, m = 1, k = 2, n = 2, loop = 1
[dump_cfg] src0: NCHW, INT8, base_addr = 0x100000
[dump_cfg] src4: NCHW, INT8, base_addr = 0x200000
[data_init]./xxx.cpp:1: init_op0[loop1][1][2] (INT8) from ram
loop0, addr[0][0] = 0x100000, data = 0x01 02
[copy_data_to_ram]./xxx.cpp:2: copy_op4(NCHW), matrix_offset = 0x0, write_length = 1,
loop0, blk0-0, addr[0][0] = 0x200000, data = 0xE0(224)
loop0, blk0-0, addr[0][1] = 0x200001, data = 0xE1(225)
loop0, blk0-0, addr[0][0] = 0x280000, data = 0x0A(10)
loop0, blk0-0, addr[0][1] = 0x280001, data = 0x0B(11)
word1: tid = 7
[dump_cfg] opcode = CONV, m = 1, k = 2, n = 1, loop = 1
[dump_cfg] src1: NHWC, FP16, base_addr = 0x300000
[data_init]./xxx.cpp:3: init_op1[loop1][1][2] (FP16) from ram
loop0, addr[0][0] = 0x300000, data = 0xAA BB
[copy_data_to_ram]./xxx.cpp:4: copy_op2(NHWC), matrix_offset = 0x0, write_length = 1,
loop0, blk0-0, addr[0][0] = 0x400000, data = 0xEE(238)
loop0, blk0-0, addr[0][0] = 0x480000, data = 0x11(17)
"""

    def setUp(self):
        self.blocks = _parse_log(self.LOG)

    def test_two_blocks(self):
        self.assertEqual(len(self.blocks), 2)

    def test_block0(self):
        op = self.blocks[0]
        self.assertEqual(op.opcode, "MMUL"); self.assertEqual(op.tid, 3)
        self.assertEqual(len(op.inputs), 1);  self.assertEqual(len(op.outputs), 1)

    def test_block1(self):
        op = self.blocks[1]
        self.assertEqual(op.opcode, "CONV"); self.assertEqual(op.tid, 7)
        self.assertEqual(len(op.inputs), 1);  self.assertEqual(len(op.outputs), 1)

    def test_op_idx_increments(self):
        self.assertEqual(self.blocks[0].op_idx, 0)
        self.assertEqual(self.blocks[1].op_idx, 1)

    def test_block0_input_bytes(self):
        # data_init 2行，前1行为 input
        self.assertEqual(self.blocks[0].inputs[0].raw_bytes, bytes([0x01, 0x02]))

    def test_block1_input_bytes(self):
        self.assertEqual(self.blocks[1].inputs[0].raw_bytes, bytes([0xAA, 0xBB]))

    def test_block0_output_bytes(self):
        # copy_op4 4行，后2行为 output
        self.assertEqual(self.blocks[0].outputs[0].raw_bytes, bytes([0x0A, 0x0B]))

    def test_block1_output_bytes(self):
        # copy_op2 2行，后1行为 output
        self.assertEqual(self.blocks[1].outputs[0].raw_bytes, bytes([0x11]))


# ══════════════════════════════════════════════════════════════
#  copy_data_to_ram 路由：永远是 Output
# ══════════════════════════════════════════════════════════════

class TestCopyRouting(unittest.TestCase):
    """有 data_init 的 src 对应的 copy_data_to_ram 丢弃；无 data_init 的 src 为 Output。"""

    LOG = """\
word0: tid = 1
[dump_cfg] opcode = MMUL, m = 1, k = 2, n = 2, loop = 1
[dump_cfg] src0: NCHW, INT8, base_addr = 0x100000
[dump_cfg] src4: NCHW, INT8, base_addr = 0x200000
[data_init]./xxx.cpp:1: init_op0[loop1][1][2] (INT8) from ram
loop0, addr[0][0] = 0x100000, data = 0xAA BB
[copy_data_to_ram]./xxx.cpp:2: copy_op0(NCHW), matrix_offset = 0x0, write_length = 1,
loop0, blk0-0, addr[0][0] = 0x100000, data = 0xAA(170)
loop0, blk0-0, addr[0][1] = 0x180001, data = 0xBB(187)
[copy_data_to_ram]./xxx.cpp:3: copy_op4(NCHW), matrix_offset = 0x0, write_length = 1,
loop0, blk0-0, addr[0][0] = 0x200000, data = 0xE0(224)
loop0, blk0-0, addr[0][1] = 0x200001, data = 0xE1(225)
loop0, blk0-0, addr[0][0] = 0x280000, data = 0x11(17)
loop0, blk0-0, addr[0][1] = 0x280001, data = 0x22(34)
"""

    def setUp(self):
        self.op = _parse_log(self.LOG)[0]

    def test_one_input_one_output(self):
        # copy_op0 有 data_init → 丢弃；copy_op4 无 data_init → Output
        self.assertEqual(len(self.op.inputs),  1)
        self.assertEqual(len(self.op.outputs), 1)

    def test_input_is_src0(self):
        self.assertEqual(self.op.inputs[0].src_idx, 0)

    def test_input_bytes_from_data_init(self):
        # data_init 1行，全部为 input；copy_op0 的数据行在 INPUT_DISCARD 下不追加
        self.assertEqual(self.op.inputs[0].raw_bytes, bytes([0xAA, 0xBB]))

    def test_input_not_polluted_by_copy_data(self):
        # copy_op0 的数据行在 INPUT_DISCARD 下不追加；data_init "0xAA BB" 展开为2个字节
        self.assertEqual(len(self.op.inputs[0].lines), 2)

    def test_copy_discard_data_not_in_output(self):
        # copy_op0 被丢弃，output 只来自 copy_op4
        self.assertEqual(self.op.outputs[0].src_idx, 4)

    def test_output_is_src4(self):
        self.assertEqual(self.op.outputs[0].src_idx, 4)

    def test_output4_bytes(self):
        # copy_op4 4行，后2行为 output
        self.assertEqual(self.op.outputs[0].raw_bytes, bytes([0x11, 0x22]))


# ══════════════════════════════════════════════════════════════
#  多输入多输出
# ══════════════════════════════════════════════════════════════

class TestMultipleInputsOutputs(unittest.TestCase):
    LOG = """\
word0: tid = 2
[dump_cfg] opcode = GEMM, m = 1, k = 2, n = 2, loop = 1
[dump_cfg] src0: NCHW, INT8, base_addr = 0x100000
[dump_cfg] src1: NHWC, INT8, base_addr = 0x200000
[dump_cfg] src4: NCHW, INT8, base_addr = 0x300000
[dump_cfg] src5: NCHW, INT8, base_addr = 0x400000
[data_init]./xxx.cpp:1: init_op0[loop1][1][2] (INT8) from ram
loop0, addr[0][0] = 0x100000, data = 0x01 02
[data_init]./xxx.cpp:2: init_op1[loop1][1][2] (INT8) from ram
loop0, addr[0][0] = 0x200000, data = 0x03 04
[copy_data_to_ram]./xxx.cpp:3: copy_op4(NCHW), matrix_offset = 0x0, write_length = 1,
loop0, blk0-0, addr[0][0] = 0x300000, data = 0xE0(224)
loop0, blk0-0, addr[0][1] = 0x300001, data = 0xE1(225)
loop0, blk0-0, addr[0][0] = 0x380000, data = 0x0A(10)
loop0, blk0-0, addr[0][1] = 0x380001, data = 0x0B(11)
[copy_data_to_ram]./xxx.cpp:4: copy_op5(NCHW), matrix_offset = 0x0, write_length = 1,
loop0, blk0-0, addr[0][0] = 0x400000, data = 0xE2(226)
loop0, blk0-0, addr[0][1] = 0x400001, data = 0xE3(227)
loop0, blk0-0, addr[0][0] = 0x480000, data = 0x0C(12)
loop0, blk0-0, addr[0][1] = 0x480001, data = 0x0D(13)
"""

    def setUp(self):
        self.op = _parse_log(self.LOG)[0]

    def test_two_inputs(self):
        self.assertEqual(len(self.op.inputs), 2)

    def test_two_outputs(self):
        self.assertEqual(len(self.op.outputs), 2)

    def test_input_src_indices(self):
        src_idxs = {i.src_idx for i in self.op.inputs}
        self.assertEqual(src_idxs, {0, 1})

    def test_output_src_indices(self):
        src_idxs = {o.src_idx for o in self.op.outputs}
        self.assertEqual(src_idxs, {4, 5})

    def test_input0_bytes(self):
        inp0 = next(i for i in self.op.inputs if i.src_idx == 0)
        self.assertEqual(inp0.raw_bytes, bytes([0x01, 0x02]))

    def test_input1_bytes(self):
        inp1 = next(i for i in self.op.inputs if i.src_idx == 1)
        self.assertEqual(inp1.raw_bytes, bytes([0x03, 0x04]))

    def test_output4_bytes(self):
        out4 = next(o for o in self.op.outputs if o.src_idx == 4)
        self.assertEqual(out4.raw_bytes, bytes([0x0A, 0x0B]))

    def test_output5_bytes(self):
        out5 = next(o for o in self.op.outputs if o.src_idx == 5)
        self.assertEqual(out5.raw_bytes, bytes([0x0C, 0x0D]))


# ══════════════════════════════════════════════════════════════
#  tid 与 opcode 之间有无关行
# ══════════════════════════════════════════════════════════════

class TestTidWithGarbageLines(unittest.TestCase):
    LOG = """\
some random log line
word5: tid = 9
another irrelevant line
[INFO] system status OK
[dump_cfg] opcode = MMUL, m = 1, k = 2, n = 2, loop = 1
[dump_cfg] src0: NCHW, INT8, base_addr = 0x100000
[dump_cfg] src4: NCHW, INT8, base_addr = 0x200000
[data_init]./xxx.cpp:1: init_op0[loop1][1][2] (INT8) from ram
loop0, addr[0][0] = 0x100000, data = 0x01 02
[copy_data_to_ram]./xxx.cpp:2: copy_op4(NCHW), matrix_offset = 0x0, write_length = 1,
loop0, blk0-0, addr[0][0] = 0x200000, data = 0xE0(224)
loop0, blk0-0, addr[0][0] = 0x280000, data = 0x0A(10)
"""

    def test_tid_parsed_correctly(self):
        blocks = _parse_log(self.LOG)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].tid, 9)
        self.assertEqual(blocks[0].opcode, "MMUL")


# ══════════════════════════════════════════════════════════════
#  多字节 RE_INPUT_DATA 展开为多个 DataLine
# ══════════════════════════════════════════════════════════════

class TestMultiByteInputData(unittest.TestCase):
    LOG = """\
word0: tid = 1
[dump_cfg] opcode = MMUL, m = 1, k = 4, n = 1, loop = 1
[dump_cfg] src0: NCHW, INT8, base_addr = 0x100000
[data_init]./xxx.cpp:1: init_op0[loop1][1][4] (INT8) from ram
loop0, addr[0][0] = 0x100000, data = 0x11 22 33 44
"""

    def test_four_bytes_expanded(self):
        op  = _parse_log(self.LOG)[0]
        inp = op.inputs[0]
        # data_init 8行（每行展开4 DataLine），前半4 DataLine 为 input
        self.assertEqual(len(inp.lines), 4)
        self.assertEqual(inp.raw_bytes, bytes([0x11, 0x22, 0x33, 0x44]))

    def test_col_idx_increments(self):
        op  = _parse_log(self.LOG)[0]
        inp = op.inputs[0]
        col_idxs = [dl.col_idx for dl in sorted(inp.lines, key=lambda d: d.address)]
        self.assertEqual(col_idxs, [0, 1, 2, 3])

    def test_address_increments(self):
        op  = _parse_log(self.LOG)[0]
        inp = op.inputs[0]
        addrs = [dl.address for dl in sorted(inp.lines, key=lambda d: d.address)]
        self.assertEqual(addrs, [0x100000, 0x100001, 0x100002, 0x100003])


# ══════════════════════════════════════════════════════════════
#  operand_cfg 与 srcX 的对应关系
# ══════════════════════════════════════════════════════════════

class TestOperandCfgMapping(unittest.TestCase):
    LOG = """\
word0: tid = 4
[dump_cfg] opcode = MMUL, m = 1, k = 2, n = 2, loop = 1
[dump_cfg] src0: NCHW, INT8, base_addr = 0x100000
[dump_cfg] src4: FORMAT, FP16, base_addr = 0x200000
[data_init]./xxx.cpp:1: init_op0[loop1][1][2] (INT8) from ram
loop0, addr[0][0] = 0x100000, data = 0x01 02
[copy_data_to_ram]./xxx.cpp:2: copy_op4(NCHW), matrix_offset = 0x0, write_length = 1,
loop0, blk0-0, addr[0][0] = 0x200000, data = 0xE0(224)
loop0, blk0-0, addr[0][0] = 0x280000, data = 0x0A(10)
"""

    def setUp(self):
        self.op = _parse_log(self.LOG)[0]

    def test_cfg_src0_fmt(self):
        cfg = self.op.get_cfg(0)
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.fmt,   "NCHW")
        self.assertEqual(cfg.dtype, "INT8")

    def test_cfg_src4_dtype(self):
        cfg = self.op.get_cfg(4)
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.dtype, "FP16")

    def test_output_fmt_resolved_from_copy_line(self):
        from aitf.compare.op_log_parser import _resolve_fmt
        out = self.op.outputs[0]
        cfg = self.op.get_cfg(out.src_idx)
        fmt = _resolve_fmt([out.fmt_in_line, cfg.fmt if cfg else None])
        # copy_op4(NCHW) 括号内有具体名，优先于 cfg 的 FORMAT
        self.assertEqual(fmt, "NCHW")


if __name__ == "__main__":
    unittest.main(verbosity=2)
