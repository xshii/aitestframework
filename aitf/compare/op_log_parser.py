#!/usr/bin/env python3
"""
op_log_parser.py
================
硬件算子 dump 日志的公共数据结构、正则、工具函数。

日志结构：
  word3: tid = 2
  [dump_cfg] opcode = MMUL, m = 64, k = 128, n = 96, loop = 1
  [dump_cfg] src0: NCHW, INT8, base_addr = 0x102200
  [dump_cfg] src4: FORMAT, INT8, base_addr = 0x11da00
  [data_init] .../xxx_math_data_init.cpp:819: init_op0[loop1][64][128] (INT8) from ram
  loop0, addr[0][0] = 0x100200, data = 0x00 01 02 03
  [copy_data_to_ram] .../xxx.cpp:426: copy_op4(NCHW), matrix_offset = 0x0, write_length = 1,
  loop0, blk0-0, addr[0][0] = 0x11C200, data = 0x0A(10)

解析器在 op_parser_state.py（OpStateMachineParser）。

输出文件命名：{OPCODE}_{op_idx}_{role}{src_idx}_{FMT}_{DTYPE}_{S0}X{S1}X{S2}.txt
"""

from __future__ import annotations

import re
import os
import sys
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
#  配置加载
# ══════════════════════════════════════════════════════════════

_DEFAULT_CONFIG = {
    "op_log":    "test-tid_2.log",
    "tensor_log": "thread0.tensor_trace.txt",
    "out_dir":   "./output",
    "txt_fmt":   0,
    "cross_op":  True,
    "tid_names": None,
    "dtype_map": {},
}

def load_config(config_path: Optional[str] = None) -> dict:
    """加载 config.yaml，缺失时返回默认值。"""
    try:
        import yaml
    except ImportError:
        logger.warning("未安装 PyYAML，跳过 config.yaml 加载，使用默认配置")
        return dict(_DEFAULT_CONFIG)

    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

    cfg = dict(_DEFAULT_CONFIG)
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        cfg.update({k: v for k, v in loaded.items() if v is not None or k in loaded})
        logger.debug(f"已加载配置文件: {config_path}")
    else:
        logger.debug(f"未找到配置文件 {config_path}，使用默认配置")
    return cfg

# ══════════════════════════════════════════════════════════════
#  工具：dtype 位宽推断
# ══════════════════════════════════════════════════════════════

_RE_DTYPE_BITS = re.compile(r'(\d+)$')

def _dtype_elem_bytes(dtype: str, dtype_map: Optional[dict] = None) -> int:
    """推断 dtype 的元素字节数。
    优先查 dtype_map（来自 config.yaml），其次从末尾数字推断（INT8→1, FP16→2）。
    两者均无法确定时报错退出。
    """
    if dtype_map:
        key = dtype.upper()
        for k, v in dtype_map.items():
            if k.upper() == key:
                return int(v)
    m = _RE_DTYPE_BITS.search(dtype.upper())
    if m:
        return int(m.group(1)) // 8
    logger.error(
        f"未知 dtype '{dtype}'：无法推断字节数，且 config.yaml dtype_map 中未配置。"
        f"请在 config.yaml 的 dtype_map 中添加 '{dtype}: <字节数>'。"
    )
    sys.exit(1)

# ══════════════════════════════════════════════════════════════
#  数据结构
# ══════════════════════════════════════════════════════════════

@dataclass
class OperandCfg:
    """[dump_cfg] srcX: FMT, DTYPE, base_addr = 0x..."""
    src_idx:   int
    fmt:       Optional[str]
    dtype:     Optional[str]
    base_addr: Optional[int]


@dataclass
class DataLine:
    """单条 addr/data 记录（输入或输出）。"""
    loop_idx: int
    row_idx:  int
    col_idx:  int
    address:  int
    value:    int


@dataclass
class InputOperand:
    """
    输入操作数：[data_init] 行开头，后接若干 DataLine。
    shape：data_init 的 loop_dims（高维）+ addr 索引推断（低两维）。
    """
    src_idx:   int
    loop_dims: list[str]      # 原始维度标记，如 ["loop1", "64", "128"]
    dtype:     str
    source:    str
    lines:     list[DataLine] = field(default_factory=list)

    # finalize() 后填充
    raw_bytes: bytes     = b""
    addr_min:  int       = 0
    addr_max:  int       = 0
    shape:     list[int] = field(default_factory=list)

    def finalize(self) -> None:
        if not self.lines:
            return
        self.lines.sort(key=lambda dl: dl.address)
        # data_init 的数据行就是完整 input，不需要切分
        if not self.lines:
            return
        _check_addr_continuity(self.lines, 1, f"Input src{self.src_idx}")
        self.raw_bytes = bytes(dl.value for dl in self.lines)
        self.addr_min  = self.lines[0].address
        self.addr_max  = self.lines[-1].address + 1

        max_row = max(dl.row_idx for dl in self.lines)
        max_col = max(dl.col_idx for dl in self.lines)
        row_dim, col_dim = max_row + 1, max_col + 1

        dims: list[int] = []
        for d in self.loop_dims:
            try:
                dims.append(int(d))
            except ValueError:
                dims.append(1)

        if len(dims) >= 2:
            dims[-2], dims[-1] = row_dim, col_dim
        elif len(dims) == 1:
            dims[-1] = col_dim
            dims.insert(0, row_dim)
        else:
            dims = [row_dim, col_dim]

        self.shape = dims


@dataclass
class OutputOperand:
    """
    输出操作数：[copy_data_to_ram] 行开头，后接若干 DataLine。
    shape：addr 索引推断低两维，write_length 推算高维。
    """
    src_idx:       int
    op_name:       str
    fmt_in_line:   Optional[str]
    matrix_offset: int
    write_length:  int
    lines:         list[DataLine] = field(default_factory=list)

    # finalize() 后填充
    raw_bytes: bytes     = b""
    addr_min:  int       = 0
    addr_max:  int       = 0
    shape:     list[int] = field(default_factory=list)

    def finalize(self, cfg: Optional[OperandCfg] = None,
                 dtype_map: Optional[dict] = None) -> None:
        if not self.lines:
            return
        self.lines.sort(key=lambda dl: dl.address)
        half = len(self.lines) // 2
        self.lines = self.lines[half:]          # 后半段为 output
        if not self.lines:
            return
        _check_addr_continuity(self.lines, self.write_length, f"Output src{self.src_idx}")
        self.raw_bytes = bytes(dl.value for dl in self.lines)
        self.addr_min  = self.lines[0].address
        self.addr_max  = self.lines[-1].address + self.write_length

        max_row = max(dl.row_idx for dl in self.lines)
        max_col = max(dl.col_idx for dl in self.lines)
        row_dim, col_dim = max_row + 1, max_col + 1

        dtype_str   = (cfg.dtype if cfg else None) or "INT8"
        elem_bytes  = _dtype_elem_bytes(dtype_str, dtype_map)
        total_elems = (len(self.lines) * self.write_length) // elem_bytes
        low2        = row_dim * col_dim

        if low2 > 0 and total_elems > low2 and total_elems % low2 == 0:
            self.shape = [total_elems // low2, row_dim, col_dim]
        else:
            self.shape = [row_dim, col_dim]


@dataclass
class OpBlock:
    """一个完整算子块：opcode、tid、cfg、输入列表、输出列表。"""
    op_idx:       int
    opcode:       str
    tid:          Optional[int]         = None
    cfg_params:   dict[str, int | str]  = field(default_factory=dict)
    operand_cfgs: dict[int, OperandCfg] = field(default_factory=dict)
    inputs:       list[InputOperand]    = field(default_factory=list)
    outputs:      list[OutputOperand]   = field(default_factory=list)

    def get_cfg(self, src_idx: int) -> Optional[OperandCfg]:
        return self.operand_cfgs.get(src_idx)


# ══════════════════════════════════════════════════════════════
#  正则
# ══════════════════════════════════════════════════════════════

RE_CFG_OPCODE = re.compile(r'\[dump_cfg\]\s+opcode\s*=\s*(?P<opcode>\w+)')
RE_KV         = re.compile(r'(?P<key>\w+)\s*=\s*(?P<val>\S+?)(?=\s*,|\s*$)')
RE_CFG_SRC    = re.compile(r'\[dump_cfg\]\s+src(?P<src_idx>\d+)\s*:\s*(?P<rest>.+)')
RE_BASE_ADDR  = re.compile(r'base_addr\s*=\s*(?P<addr>0x[0-9a-fA-F]+)')
RE_DATA_INIT  = re.compile(
    r'\[data_init\][^:]*:[^:]*:\s*'
    r'init_op(?P<src_idx>\d+)(?P<dims>(?:\[\w+\])+)'
    r'\s*\((?P<dtype>\w+)\)\s*from\s+(?P<source>\w+)'
)
RE_DIM_TOKEN  = re.compile(r'\[(?P<dim>\w+)\]')
RE_COPY_HDR   = re.compile(
    r'\[copy_data_to_ram\][^:]*:[^:]*:\s*'
    r'copy_op(?P<src_idx>\d+)\((?P<fmt_in_line>[^)]*)\)'
    r'.*?matrix_offset\s*=\s*(?P<matrix_offset>0x[0-9a-fA-F]+|\d+)'
    r'.*?write_length\s*=\s*(?P<write_length>\d+)'
)
RE_OUTPUT_DATA = re.compile(
    r'loop(?P<loop_idx>\d+)'
    r',\s*blk\d+-\d+'
    r',\s*addr\[(?P<row_idx>\d+)\]\[(?P<col_idx>\d+)\]'
    r'\s*=\s*(?P<address>0x[0-9a-fA-F]+)'
    r',\s*data\s*=\s*0x(?P<hex_vals>[0-9a-fA-F]{2}(?:\s+[0-9a-fA-F]{2})*)(?:\(\d+\))?$'
)
# ══════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════

def _check_addr_continuity(lines: list, step: int, label: str) -> None:
    """检查相邻 DataLine 地址步进是否均为 step，gap 时打 WARNING。"""
    gaps = []
    for i in range(1, len(lines)):
        expected = lines[i - 1].address + step
        actual   = lines[i].address
        if actual != expected:
            gaps.append(
                f"  addr[{lines[i-1].row_idx}][{lines[i-1].col_idx}]"
                f"=0x{lines[i-1].address:08X}"
                f" → addr[{lines[i].row_idx}][{lines[i].col_idx}]"
                f"=0x{actual:08X}"
                f" (expected 0x{expected:08X}, gap={actual - expected:+d})"
            )
    if gaps:
        logger.warning(
            "数据不连续 (%s)，共 %d 处 gap:\n%s",
            label, len(gaps), "\n".join(gaps),
        )


def _shape_str(shape: list[int]) -> str:
    return "X".join(str(s) for s in shape)


def _resolve_fmt(fmt_candidates: list[Optional[str]]) -> str:
    """取第一个非空、非 'FORMAT' 的候选值，否则返回 'FORMAT'。"""
    for c in fmt_candidates:
        if c and c.upper() != "FORMAT":
            return c.upper()
    return "FORMAT"


def make_filename(op: OpBlock, src_idx: int, role: str,
                  fmt: str, dtype: str, shape: list[int]) -> str:
    """生成输出文件名：{OPCODE}_{op_idx}_{role}{src_idx}_{FMT}_{DTYPE}_{shape}.txt"""
    shape_s = _shape_str(shape) if shape else "UNKNOWN"
    return f"{op.opcode}_{op.op_idx}_{role}{src_idx}_{fmt}_{dtype}_{shape_s}.txt"


# ══════════════════════════════════════════════════════════════
#  输出格式
# ══════════════════════════════════════════════════════════════

from enum import Enum

class TxtFormat(Enum):
    """
    输出 txt 文件的格式：

      WITH_HEADER_4B  首行 "0xADDR SIZE"（SIZE = raw_bytes 字节数），之后每行 4 字节带 0x 前缀
                        0x00100200 8
                        0x01020304
                        0x05060708

      NO_HEADER_1B    无首行，每行 1 字节不带 0x 前缀
                        01
                        02
                        ...
    """
    WITH_HEADER_4B = "with_header_4b"
    NO_HEADER_1B   = "no_header_1b"


def write_txt(path: str, raw_bytes: bytes, addr_min: int,
              fmt: TxtFormat = TxtFormat.NO_HEADER_1B) -> None:
    """将 raw_bytes 按指定格式写入 txt 文件。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        if fmt == TxtFormat.WITH_HEADER_4B:
            fh.write(f"0x{addr_min:08X} {len(raw_bytes)}\n")
            for i in range(0, len(raw_bytes), 4):
                chunk = raw_bytes[i:i + 4]
                val   = int.from_bytes(chunk.ljust(4, b'\x00'), "big")
                fh.write(f"0x{val:08X}\n")
        else:  # NO_HEADER_1B
            for b in raw_bytes:
                fh.write(f"{b:02X}\n")


def save_op(op: OpBlock, out_dir: str,
            txt_fmt: TxtFormat = TxtFormat.NO_HEADER_1B) -> list:
    """保存一个 OpBlock 的所有输入/输出 txt 文件，返回 saved 列表。"""
    saved = []
    for inp in op.inputs:
        cfg   = op.get_cfg(inp.src_idx)
        fmt   = _resolve_fmt([cfg.fmt if cfg else None])
        dtype = inp.dtype or (cfg.dtype if cfg else None) or "INT8"
        fname = make_filename(op, inp.src_idx, "Input", fmt, dtype, inp.shape)
        path  = os.path.join(out_dir, fname)
        write_txt(path, inp.raw_bytes, inp.addr_min, txt_fmt)
        saved.append(("Input", inp.src_idx, path, inp.shape, dtype, fmt))

    for out in op.outputs:
        cfg   = op.get_cfg(out.src_idx)
        fmt   = _resolve_fmt([out.fmt_in_line, cfg.fmt if cfg else None])
        dtype = (cfg.dtype if cfg else None) or "INT8"
        fname = make_filename(op, out.src_idx, "Output", fmt, dtype, out.shape)
        path  = os.path.join(out_dir, fname)
        write_txt(path, out.raw_bytes, out.addr_min, txt_fmt)
        saved.append(("Output", out.src_idx, path, out.shape, dtype, fmt))

    return saved
