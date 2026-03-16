#!/usr/bin/env python3
"""
tests/test_op_log_parser.py
op_log_parser.py 的单元测试
运行：python -m pytest tests/ 或 python tests/test_op_log_parser.py
"""

import io
import logging
import os
import sys
import tempfile
import unittest

import aitf.compare.op_log_parser as P
from aitf.compare.op_parser_state import OpStateMachineParser

# ══════════════════════════════════════════════════════════════
#  正则测试
# ══════════════════════════════════════════════════════════════

class TestRegexCfgOpcode(unittest.TestCase):
    def test_basic(self):
        line = "[dump_cfg] opcode = MMUL, m = 64, k = 128, n = 96, loop = 1"
        m = P.RE_CFG_OPCODE.search(line)
        self.assertIsNotNone(m)
        self.assertEqual(m.group("opcode"), "MMUL")

    def test_no_match_on_src_line(self):
        self.assertIsNone(P.RE_CFG_OPCODE.search(
            "[dump_cfg] src0: NCHW, INT8, base_addr = 0x102200"))


class TestRegexCfgSrc(unittest.TestCase):
    def test_with_fmt_and_dtype(self):
        m = P.RE_CFG_SRC.match("[dump_cfg] src0: NCHW, INT8, base_addr = 0x102200")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("src_idx"), "0")
        self.assertIn("NCHW", m.group("rest"))

    def test_src4_with_literal_FORMAT(self):
        m = P.RE_CFG_SRC.match("[dump_cfg] src4: FORMAT, INT8, base_addr = 0x11da00")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("src_idx"), "4")


class TestRegexDataInit(unittest.TestCase):
    def test_basic(self):
        line = ("[data_init]./code/ip/src/math/xxx_math_data_init.cpp:819:"
                " init_op0[loop1][64][128] (INT8) from ram")
        m = P.RE_DATA_INIT.search(line)
        self.assertIsNotNone(m)
        self.assertEqual(m.group("src_idx"), "0")
        self.assertEqual(m.group("dims"),    "[loop1][64][128]")
        self.assertEqual(m.group("dtype"),   "INT8")
        self.assertEqual(m.group("source"),  "ram")

    def test_src_idx_extraction(self):
        m = P.RE_DATA_INIT.search(
            "[data_init]./xxx.cpp:1: init_op3[loop1][4][8] (FP16) from ddr")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("src_idx"), "3")
        self.assertEqual(m.group("dtype"),   "FP16")
        self.assertEqual(m.group("source"),  "ddr")


class TestRegexCopyHdr(unittest.TestCase):
    def test_basic(self):
        line = ("[copy_data_to_ram]./code/ip/src/math/xxx_math_copy_data_to_ram.cpp:426:"
                " copy_op4(NCHW), matrix_offset = 0x0, write_length = 1,")
        m = P.RE_COPY_HDR.search(line)
        self.assertIsNotNone(m)
        self.assertEqual(m.group("src_idx"),      "4")
        self.assertEqual(m.group("fmt_in_line"),  "NCHW")
        self.assertEqual(m.group("write_length"), "1")

    def test_fmt_is_literal_FORMAT(self):
        m = P.RE_COPY_HDR.search(
            "[copy_data_to_ram]./xxx.cpp:1: copy_op2(FORMAT),"
            " matrix_offset = 0x10, write_length = 2,")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("fmt_in_line"),  "FORMAT")
        self.assertEqual(m.group("write_length"), "2")


class TestRegexOutputDataLine(unittest.TestCase):
    def test_basic(self):
        m = P.RE_OUTPUT_DATA.match(
            "loop0, blk0-0, addr[0][0] = 0x11C200, data = 0x00(0)")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("address"), "0x11C200")

    def test_nonzero_blk(self):
        m = P.RE_OUTPUT_DATA.match(
            "loop1, blk2-3, addr[2][5] = 0x11C225, data = 0x25(37)")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("loop_idx"), "1")
        self.assertEqual(m.group("row_idx"),  "2")
        self.assertEqual(m.group("col_idx"),  "5")


# ══════════════════════════════════════════════════════════════
#  OperandCfg 解析
# ══════════════════════════════════════════════════════════════

class TestParseOperandCfg(unittest.TestCase):
    def _parse_cfg(self, *src_lines):
        """通过完整解析提取 operand_cfgs。"""
        log = "[dump_cfg] opcode = TEST, loop = 1\n"
        log += "\n".join(src_lines) + "\n"
        tmp = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False)
        tmp.write(log); tmp.close()
        try:
            ops = OpStateMachineParser(tmp.name).parse()
            return ops[0].operand_cfgs
        finally:
            os.unlink(tmp.name)

    def test_full(self):
        cfgs = self._parse_cfg("[dump_cfg] src0: NCHW, INT8, base_addr = 0x102200")
        self.assertEqual(cfgs[0].src_idx,   0)
        self.assertEqual(cfgs[0].fmt,       "NCHW")
        self.assertEqual(cfgs[0].dtype,     "INT8")
        self.assertEqual(cfgs[0].base_addr, 0x102200)

    def test_literal_FORMAT(self):
        cfgs = self._parse_cfg("[dump_cfg] src4: FORMAT, INT8, base_addr = 0x11da00")
        self.assertEqual(cfgs[4].fmt,   "FORMAT")
        self.assertEqual(cfgs[4].dtype, "INT8")

    def test_fp16(self):
        cfgs = self._parse_cfg("[dump_cfg] src1: NHWC, FP16, base_addr = 0xABCDEF")
        self.assertEqual(cfgs[1].dtype, "FP16")
        self.assertEqual(cfgs[1].fmt,   "NHWC")


# ══════════════════════════════════════════════════════════════
#  Shape 推断
# ══════════════════════════════════════════════════════════════

def _make_grid(rows, cols, base_addr=0x1000, src_idx=0):
    lines, val = [], 0
    for r in range(rows):
        for c in range(cols):
            lines.append(P.DataLine(0, r, c, base_addr + r * 0x100 + c, val % 256))
            val += 1
    return lines


class TestInputOperandFinalize(unittest.TestCase):
    def test_shape_2d(self):
        inp = P.InputOperand(0, ["4", "8"], "INT8", "ram", _make_grid(4, 8))
        inp.finalize()
        self.assertEqual(inp.shape, [4, 8])

    def test_shape_3d_loop_dim(self):
        inp = P.InputOperand(0, ["loop1", "4", "8"], "INT8", "ram", _make_grid(4, 8))
        inp.finalize()
        self.assertEqual(inp.shape, [1, 4, 8])

    def test_raw_bytes_order(self):
        lines = [
            P.DataLine(0, 1, 0, 0x1100, 0xBB),
            P.DataLine(0, 0, 0, 0x1000, 0xAA),
            P.DataLine(0, 0, 1, 0x1001, 0xCC),
        ]
        inp = P.InputOperand(0, ["2", "2"], "INT8", "ram", lines)
        inp.finalize()
        self.assertEqual(inp.raw_bytes, bytes([0xAA, 0xCC, 0xBB]))

    def test_addr_range(self):
        inp = P.InputOperand(0, ["2", "2"], "INT8", "ram", _make_grid(2, 2))
        inp.finalize()
        self.assertEqual(inp.addr_min, 0x1000)
        self.assertEqual(inp.addr_max, 0x1101 + 1)


class TestOutputOperandFinalize(unittest.TestCase):
    def _out(self, rows, cols, write_length=1):
        grid = _make_grid(rows, cols, base_addr=0x2000)
        # 后半段：地址相同，值不同（output 保留后半段）
        dst  = [P.DataLine(dl.loop_idx, dl.row_idx, dl.col_idx,
                           dl.address, (dl.value + 0x80) % 256) for dl in grid]
        return P.OutputOperand(
            src_idx=4, op_name="copy_op4", fmt_in_line="NCHW",
            matrix_offset=0, write_length=write_length,
            lines=grid + dst,
        )

    def test_shape_2d(self):
        out = self._out(4, 8)
        out.finalize()
        self.assertEqual(out.shape, [4, 8])

    def test_shape_3d_via_write_length(self):
        # 构造 2 组 4x8 grid，各自翻倍，共 4*32 = 128 行
        g1 = _make_grid(4, 8, 0x2000)
        g2 = [P.DataLine(dl.loop_idx, dl.row_idx, dl.col_idx,
                         dl.address + 0x10000, dl.value) for dl in g1]
        all_src = g1 + g2
        # dst = all_src 地址相同，值为 0xFF
        all_dst = [P.DataLine(dl.loop_idx, dl.row_idx, dl.col_idx,
                              dl.address, 0xFF) for dl in all_src]
        out = P.OutputOperand(4, "copy_op4", "NCHW", 0, 1, all_src + all_dst)
        out.finalize(P.OperandCfg(4, "NCHW", "INT8", 0))
        self.assertEqual(out.shape, [2, 4, 8])


# ══════════════════════════════════════════════════════════════
#  文件名生成 / _resolve_fmt
# ══════════════════════════════════════════════════════════════

class TestMakeFilename(unittest.TestCase):
    def _op(self, opcode="MMUL", idx=0):
        return P.OpBlock(op_idx=idx, opcode=opcode)

    def test_input_filename(self):
        self.assertEqual(
            P.make_filename(self._op(), 0, "Input", "NCHW", "INT8", [1, 64, 128]),
            "MMUL_0_Input0_NCHW_INT8_1X64X128.txt")

    def test_output_filename(self):
        self.assertEqual(
            P.make_filename(self._op(), 4, "Output", "NCHW", "INT8", [4, 8]),
            "MMUL_0_Output4_NCHW_INT8_4X8.txt")

    def test_second_op(self):
        self.assertEqual(
            P.make_filename(self._op("CONV", 1), 1, "Input", "NHWC", "FP16", [1, 32, 64]),
            "CONV_1_Input1_NHWC_FP16_1X32X64.txt")

    def test_unknown_shape(self):
        self.assertEqual(
            P.make_filename(self._op(), 0, "Input", "NCHW", "INT8", []),
            "MMUL_0_Input0_NCHW_INT8_UNKNOWN.txt")


class TestResolveFmt(unittest.TestCase):
    def test_first_real_fmt_wins(self):   self.assertEqual(P._resolve_fmt(["NCHW", "NHWC"]),    "NCHW")
    def test_skip_FORMAT_literal(self):   self.assertEqual(P._resolve_fmt(["FORMAT", "NHWC"]),  "NHWC")
    def test_all_FORMAT(self):            self.assertEqual(P._resolve_fmt(["FORMAT", None]),     "FORMAT")
    def test_none_candidates(self):       self.assertEqual(P._resolve_fmt([None, None]),         "FORMAT")
    def test_copy_overrides_cfg(self):    self.assertEqual(P._resolve_fmt(["NCHW", "FORMAT"]),   "NCHW")


# ══════════════════════════════════════════════════════════════
#  地址连续性检查
# ══════════════════════════════════════════════════════════════

class TestAddrContinuity(unittest.TestCase):
    LOG_NAME = "aitf.compare.op_log_parser"

    def test_no_warning_when_contiguous(self):
        lines = [P.DataLine(0, i, 0, 0x1000 + i, i) for i in range(4)]
        with self.assertLogs(self.LOG_NAME, level="WARNING") as cm:
            logging.getLogger(self.LOG_NAME).warning("dummy")
        self.assertEqual([m for m in cm.output if "不连续" in m], [])
        P._check_addr_continuity(lines, 1, "test")   # should not raise

    def test_warning_on_gap(self):
        lines = [
            P.DataLine(0, 0, 0, 0x1000, 0x00),
            P.DataLine(0, 0, 1, 0x1001, 0x01),
            P.DataLine(0, 0, 3, 0x1003, 0x03),   # gap
            P.DataLine(0, 0, 4, 0x1004, 0x04),
        ]
        with self.assertLogs(self.LOG_NAME, level="WARNING") as cm:
            P._check_addr_continuity(lines, 1, "Input src0")
        self.assertTrue(any("不连续" in m for m in cm.output))

    def test_gap_count(self):
        lines = [
            P.DataLine(0, 0, 0, 0x1000, 0),
            P.DataLine(0, 0, 1, 0x1002, 1),   # gap
            P.DataLine(0, 0, 2, 0x1003, 2),
            P.DataLine(0, 0, 3, 0x1005, 3),   # gap
        ]
        with self.assertLogs(self.LOG_NAME, level="WARNING") as cm:
            P._check_addr_continuity(lines, 1, "src0")
        self.assertIn("2 处 gap", "\n".join(cm.output))

    def test_input_finalize_warns(self):
        # 8行，前4行为 input（含 gap），后4行丢弃
        inp = P.InputOperand(0, ["2", "2"], "INT8", "ram", [
            P.DataLine(0, 0, 0, 0x1000, 0xAA),
            P.DataLine(0, 0, 1, 0x1001, 0xBB),
            P.DataLine(0, 1, 0, 0x1003, 0xCC),   # gap
            P.DataLine(0, 1, 1, 0x1004, 0xDD),
            P.DataLine(0, 0, 0, 0x1000, 0xFF),   # dst（丢弃）
            P.DataLine(0, 0, 1, 0x1001, 0xFF),
            P.DataLine(0, 1, 0, 0x1003, 0xFF),
            P.DataLine(0, 1, 1, 0x1004, 0xFF),
        ])
        with self.assertLogs(self.LOG_NAME, level="WARNING") as cm:
            inp.finalize()
        self.assertTrue(any("不连续" in m for m in cm.output))

    def test_output_finalize_warns(self):
        # 8行，后4行为 output（含 gap）
        out = P.OutputOperand(4, "copy_op4", "NCHW", 0, 1, [
            P.DataLine(0, 0, 0, 0x2000, 0xE0),   # src（丢弃）
            P.DataLine(0, 0, 1, 0x2001, 0xE1),
            P.DataLine(0, 1, 0, 0x2010, 0xE2),
            P.DataLine(0, 1, 1, 0x2011, 0xE3),
            P.DataLine(0, 0, 0, 0x2000, 0x0A),   # dst（保留）
            P.DataLine(0, 0, 1, 0x2001, 0x0B),
            P.DataLine(0, 1, 0, 0x2010, 0x0C),   # gap
            P.DataLine(0, 1, 1, 0x2011, 0x0D),
        ])
        with self.assertLogs(self.LOG_NAME, level="WARNING") as cm:
            out.finalize()
        self.assertTrue(any("不连续" in m for m in cm.output))

    def test_correct_step_no_warning(self):
        lines = [P.DataLine(0, 0, i, 0x1000 + i * 2, i) for i in range(4)]
        handler = logging.StreamHandler(io.StringIO())
        handler.setLevel(logging.WARNING)
        log = logging.getLogger(self.LOG_NAME)
        log.addHandler(handler)
        P._check_addr_continuity(lines, 2, "src1")
        out = handler.stream.getvalue()
        log.removeHandler(handler)
        self.assertEqual(out, "")


# ══════════════════════════════════════════════════════════════
#  端到端
# ══════════════════════════════════════════════════════════════

MOCK_LOG = """\
[dump_cfg] opcode = MMUL, m = 4, k = 8, n = 4, loop = 1
[dump_cfg] src0: NCHW, INT8, base_addr = 0x100000
[dump_cfg] src1: NHWC, INT8, base_addr = 0x200000
[dump_cfg] src4: FORMAT, INT8, base_addr = 0x11da00
[data_init]./code/ip/src/math/xxx_math_data_init.cpp:819: init_op0[loop1][2][4] (INT8) from ram
loop0, addr[0][0] = 0x100200, data = 0x00 01 02 03
loop0, addr[1][0] = 0x100210, data = 0x10 11 12 13
[data_init]./code/ip/src/math/xxx_math_data_init.cpp:820: init_op1[loop1][2][4] (INT8) from ram
loop0, addr[0][0] = 0x200200, data = 0xAA BB CC DD
loop0, addr[1][0] = 0x200210, data = 0xEE FF 11 22
[copy_data_to_ram]./code/ip/src/math/xxx_math_copy_data_to_ram.cpp:426: copy_op4(NCHW), matrix_offset = 0x0, write_length = 1,
loop0, blk0-0, addr[0][0] = 0x11C200, data = 0xA0(160)
loop0, blk0-0, addr[0][1] = 0x11C201, data = 0xA1(161)
loop0, blk0-0, addr[0][2] = 0x11C202, data = 0xA2(162)
loop0, blk0-0, addr[0][3] = 0x11C203, data = 0xA3(163)
loop0, blk0-0, addr[1][0] = 0x11C210, data = 0xA4(164)
loop0, blk0-0, addr[1][1] = 0x11C211, data = 0xA5(165)
loop0, blk0-0, addr[1][2] = 0x11C212, data = 0xA6(166)
loop0, blk0-0, addr[1][3] = 0x11C213, data = 0xA7(167)
loop0, blk0-0, addr[0][0] = 0x19C200, data = 0x0A(10)
loop0, blk0-0, addr[0][1] = 0x19C201, data = 0x0B(11)
loop0, blk0-0, addr[0][2] = 0x19C202, data = 0x0C(12)
loop0, blk0-0, addr[0][3] = 0x19C203, data = 0x0D(13)
loop0, blk0-0, addr[1][0] = 0x19C210, data = 0x1A(26)
loop0, blk0-0, addr[1][1] = 0x19C211, data = 0x1B(27)
loop0, blk0-0, addr[1][2] = 0x19C212, data = 0x1C(28)
loop0, blk0-0, addr[1][3] = 0x19C213, data = 0x1D(29)
"""


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp     = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False)
        self.out_dir = tempfile.mkdtemp()
        self.tmp.write(MOCK_LOG)
        self.tmp.close()

    def tearDown(self):
        os.unlink(self.tmp.name)

    def _parse(self):
        return OpStateMachineParser(self.tmp.name).parse()

    def test_one_block(self):           self.assertEqual(len(self._parse()), 1)
    def test_opcode(self):              self.assertEqual(self._parse()[0].opcode, "MMUL")
    def test_op_idx(self):              self.assertEqual(self._parse()[0].op_idx, 0)
    def test_cfg_params(self):
        p = self._parse()[0].cfg_params
        self.assertEqual(p["m"], 4); self.assertEqual(p["k"], 8)
    def test_two_inputs(self):          self.assertEqual(len(self._parse()[0].inputs), 2)
    def test_one_output(self):          self.assertEqual(len(self._parse()[0].outputs), 1)

    def test_input0_shape(self):
        op = self._parse()[0]
        inp = next(i for i in op.inputs if i.src_idx == 0)
        self.assertEqual(inp.shape, [1, 2, 4])

    def test_input0_bytes(self):
        op = self._parse()[0]
        inp = next(i for i in op.inputs if i.src_idx == 0)
        # data_init 8行，全部为 input
        self.assertEqual(inp.raw_bytes,
                         bytes([0x00,0x01,0x02,0x03,0x10,0x11,0x12,0x13]))

    def test_input0_not_polluted_by_copy(self):
        op = self._parse()[0]
        inp = next(i for i in op.inputs if i.src_idx == 0)
        # INPUT_DISCARD 屏蔽了 copy_op4 的数据行，input 行数应恰好等于 data_init 行数
        self.assertEqual(len(inp.lines), 8)

    def test_output4_shape(self):
        self.assertEqual(self._parse()[0].outputs[0].shape, [2, 4])

    def test_output4_bytes(self):
        # copy_op4 16行，后8行为 output
        self.assertEqual(self._parse()[0].outputs[0].raw_bytes,
                         bytes([0x0A,0x0B,0x0C,0x0D,0x1A,0x1B,0x1C,0x1D]))

    def test_filenames_written(self):
        P.save_op(self._parse()[0], self.out_dir)
        files = os.listdir(self.out_dir)
        self.assertIn("MMUL_0_Input0_NCHW_INT8_1X2X4.txt",  files)
        self.assertIn("MMUL_0_Input1_NHWC_INT8_1X2X4.txt",  files)
        self.assertIn("MMUL_0_Output4_NCHW_INT8_2X4.txt",   files)

    def test_txt_header(self):
        P.save_op(self._parse()[0], self.out_dir, P.TxtFormat.WITH_HEADER_4B)
        with open(os.path.join(self.out_dir, "MMUL_0_Input0_NCHW_INT8_1X2X4.txt")) as f:
            parts = f.readline().split()
        # WITH_HEADER_4B：首行 "0xADDR SIZE"
        self.assertTrue(parts[0].startswith("0x"))
        self.assertEqual(int(parts[1]), 8)

# ══════════════════════════════════════════════════════════════
#  TxtFormat 输出格式
# ══════════════════════════════════════════════════════════════

class TestTxtFormat(unittest.TestCase):
    DATA     = bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08])
    ADDR_MIN = 0x00100200

    def _write(self, fmt):
        import tempfile
        path = os.path.join(tempfile.mkdtemp(), "out.txt")
        P.write_txt(path, self.DATA, self.ADDR_MIN, fmt)
        return open(path).read()

    # ── WITH_HEADER_4B ──────────────────────────────────────

    def test_with_header_4b_header_addr_and_size(self):
        """首行含地址和 raw_bytes 字节数。"""
        lines = self._write(P.TxtFormat.WITH_HEADER_4B).splitlines()
        self.assertEqual(lines[0], "0x00100200 8")

    def test_with_header_4b_four_bytes_per_line(self):
        lines = self._write(P.TxtFormat.WITH_HEADER_4B).splitlines()
        self.assertEqual(len(lines), 3)           # 1 header + 2 data rows
        self.assertEqual(lines[1], "0x01020304")
        self.assertEqual(lines[2], "0x05060708")

    def test_with_header_4b_padding(self):
        """末尾不足 4 字节时用 0x00 填充。"""
        path = os.path.join(__import__("tempfile").mkdtemp(), "out.txt")
        P.write_txt(path, bytes([0xAA, 0xBB, 0xCC]), 0x1000, P.TxtFormat.WITH_HEADER_4B)
        lines = open(path).read().splitlines()
        self.assertEqual(lines[0], "0x00001000 3")
        self.assertEqual(lines[1], "0xAABBCC00")

    # ── NO_HEADER_1B ────────────────────────────────────────

    def test_no_header_1b_no_header(self):
        lines = self._write(P.TxtFormat.NO_HEADER_1B).splitlines()
        self.assertEqual(len(lines), 8)           # 纯数据，无首行
        self.assertEqual(lines[0], "01")

    def test_no_header_1b_last_byte(self):
        lines = self._write(P.TxtFormat.NO_HEADER_1B).splitlines()
        self.assertEqual(lines[-1], "08")

    def test_no_header_1b_no_0x_prefix(self):
        lines = self._write(P.TxtFormat.NO_HEADER_1B).splitlines()
        for line in lines:
            self.assertFalse(line.startswith("0x"))

    # ── 默认格式 ────────────────────────────────────────────

    def test_default_is_with_header_4b(self):
        """write_txt / save_op 默认使用 WITH_HEADER_4B 格式，首行含地址和 size。"""
        lines = self._write(P.TxtFormat.WITH_HEADER_4B).splitlines()
        parts = lines[0].split()
        self.assertTrue(parts[0].startswith("0x"))
        self.assertEqual(int(parts[1]), 8)


# ══════════════════════════════════════════════════════════════
#  dtype 字节数推断
# ══════════════════════════════════════════════════════════════

class TestDtypeElemBytes(unittest.TestCase):
    def _b(self, dtype): return P._dtype_elem_bytes(dtype)

    def test_int8(self):   self.assertEqual(self._b("INT8"),    1)
    def test_uint8(self):  self.assertEqual(self._b("UINT8"),   1)
    def test_fp16(self):   self.assertEqual(self._b("FP16"),    2)
    def test_bf16(self):   self.assertEqual(self._b("BF16"),    2)
    def test_int16(self):  self.assertEqual(self._b("INT16"),   2)
    def test_fp32(self):   self.assertEqual(self._b("FP32"),    4)
    def test_float32(self):self.assertEqual(self._b("FLOAT32"), 4)
    def test_int64(self):  self.assertEqual(self._b("INT64"),   8)
    def test_fp64(self):   self.assertEqual(self._b("FP64"),    8)
    def test_bfloat16(self):self.assertEqual(self._b("BFLOAT16"), 2)  # 不在旧字典里
    def test_fp8(self):    self.assertEqual(self._b("FP8"),     1)    # 新格式
    def test_unknown(self):
        # 未知 dtype 无法推断时，提供 dtype_map 可正常解析
        self.assertEqual(P._dtype_elem_bytes("UNKNOWN", {"UNKNOWN": 3}), 3)

    def test_unknown_no_map_exits(self):
        # 无 dtype_map 时未知 dtype 应报错退出
        with self.assertRaises(SystemExit):
            P._dtype_elem_bytes("NOSUCHTYPE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
