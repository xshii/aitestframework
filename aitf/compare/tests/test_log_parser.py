#!/usr/bin/env python3
"""
tests/test_log_parser.py
log_parser.py（联合入口）集成测试
"""

import json
import os
import sys
import tempfile
import unittest

import aitf.compare.log_parser as log_parser

# tensor 日志只有 write（输出），无 read（输入），这是合法日志
OP_LOG = """\
[dump_cfg] opcode = MMUL, m = 4, k = 8, n = 4, loop = 1
[dump_cfg] src0: NCHW, INT8, base_addr = 0x100000
[dump_cfg] src4: NCHW, INT8, base_addr = 0x11da00
[data_init]./xxx.cpp:1: init_op0[loop1][2][4] (INT8) from ram
loop0, addr[0][0] = 0x100200, data = 0x01 02 03 04
loop0, addr[1][0] = 0x100210, data = 0x05 06 07 08
[copy_data_to_ram]./xxx.cpp:2: copy_op4(NCHW), matrix_offset = 0x0, write_length = 1,
loop0, blk0-0, addr[0][0] = 0x11C200, data = 0x0A(10)
loop0, blk0-0, addr[0][1] = 0x11C201, data = 0x0B(11)
loop0, blk0-0, addr[0][2] = 0x11C202, data = 0x0C(12)
loop0, blk0-0, addr[0][3] = 0x11C203, data = 0x0D(13)
loop0, blk0-0, addr[1][0] = 0x11C210, data = 0x1A(26)
loop0, blk0-0, addr[1][1] = 0x11C211, data = 0x1B(27)
loop0, blk0-0, addr[1][2] = 0x11C212, data = 0x1C(28)
loop0, blk0-0, addr[1][3] = 0x11C213, data = 0x1D(29)
"""

TENSOR_LOG = """\
[cyc=0x1] commit MmulCmd, groupId: 0, tid: 2, at cyc = 0x1
=========begin tensor=============
[DMA][0] byteEnable = 01 01 01 01 00 00 00 00
[DMA][0] write mem [0x300000] [8] = AA BB CC DD EE FF 11 22
=========end tensor=============
[cyc=0x3] commit ConvCmd, groupId: 1, tid: 4, at cyc = 0x3
=========begin tensor=============
=========end tensor=============
"""

TID_NAMES = {"2": "MatMul_layer0"}


class TestRunBothLogs(unittest.TestCase):
    def setUp(self):
        self.out_dir = tempfile.mkdtemp()
        self.op_log  = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False)
        self.op_log.write(OP_LOG); self.op_log.close()
        self.tensor_log = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False)
        self.tensor_log.write(TENSOR_LOG); self.tensor_log.close()
        self.tid_json = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(TID_NAMES, self.tid_json); self.tid_json.close()

    def tearDown(self):
        for f in [self.op_log.name, self.tensor_log.name, self.tid_json.name]:
            os.unlink(f)

    def test_op_output_dir_created(self):
        log_parser.run(op_log=self.op_log.name, out_dir=self.out_dir)
        self.assertTrue(os.path.isdir(os.path.join(self.out_dir, "op")))

    def test_tensor_output_dir_created(self):
        log_parser.run(tensor_log=self.tensor_log.name, out_dir=self.out_dir)
        self.assertTrue(os.path.isdir(os.path.join(self.out_dir, "tensor")))

    def test_op_files_written(self):
        log_parser.run(op_log=self.op_log.name, out_dir=self.out_dir)
        op_dir = os.path.join(self.out_dir, "op", "MMUL_0")
        files  = os.listdir(op_dir)
        self.assertTrue(any("Input0"  in f for f in files))
        self.assertTrue(any("Output4" in f for f in files))

    def test_tensor_summary_json(self):
        log_parser.run(tensor_log=self.tensor_log.name, out_dir=self.out_dir)
        summary = os.path.join(self.out_dir, "tensor", "tensor_summary.json")
        self.assertTrue(os.path.exists(summary))
        with open(summary) as f:
            data = json.load(f)
        self.assertEqual(len(data), 2)

    def test_tensor_txt_named_with_tid_names(self):
        log_parser.run(tensor_log=self.tensor_log.name,
                       out_dir=self.out_dir,
                       tid_names_path=self.tid_json.name)
        txts = []
        for root, _, files in os.walk(os.path.join(self.out_dir, "tensor")):
            txts += [f for f in files if f.endswith(".txt")]
        self.assertTrue(any("Input" in f or "Output" in f for f in txts))

    def test_combined_run_no_error(self):
        log_parser.run(
            op_log         = self.op_log.name,
            tensor_log     = self.tensor_log.name,
            out_dir        = self.out_dir,
            tid_names_path = self.tid_json.name,
        )
        self.assertTrue(os.path.isdir(os.path.join(self.out_dir, "op")))
        self.assertTrue(os.path.isdir(os.path.join(self.out_dir, "tensor")))

    def test_cross_op_false_runs_without_error(self):
        log_parser.run(tensor_log=self.tensor_log.name,
                       out_dir=self.out_dir, cross_op=False)
        self.assertTrue(os.path.isdir(os.path.join(self.out_dir, "tensor")))

    def test_no_logs_no_crash(self):
        log_parser.run(out_dir=self.out_dir)


if __name__ == "__main__":
    unittest.main(verbosity=2)
