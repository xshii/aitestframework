#!/usr/bin/env python3
"""
op_parser_state.py
==================
op dump 日志的显式状态机解析器（新格式）。

新格式特征：
  - 日志头部有 tid 行：word\d+: tid = N
  - 输入数据行：loop0, addr[0][0] = 0x100200, data = 0x00 01 02 03
    （多字节空格分隔，无末尾十进制括注）
  - 输出数据行：loop0, blk0-0, addr[0][0] = 0x11C200, data = 0x00(0)
    （单字节，含末尾十进制括注，格式与旧版相同）

状态机：
  IDLE        等待 tid
  WAIT_OP     已见 tid，等待 [dump_cfg] opcode
  IN_OP       已知 opcode，收集 srcX 配置
  FIND_INPUT  正在收集某个 src 的输入数据行
  FIND_OUTPUT 正在收集某个 src 的输出数据行

状态转换：
  任意状态
    RE_TID        → flush 当前 op → WAIT_OP

  WAIT_OP
    RE_CFG_OPCODE → IN_OP（新建 OpBlock，消费 pending_tid）

  IN_OP
    RE_CFG_SRC    → IN_OP
    RE_DATA_INIT  → FIND_INPUT，记录 src_idx 到 input_src_set
    RE_COPY_HDR   → src 在 input_src_set → INPUT_DISCARD（屏蔽数据行）
                 → src 不在 input_src_set → FIND_OUTPUT

  FIND_INPUT
    RE_INPUT_DATA → FIND_INPUT（追加 DataLine）
    RE_DATA_INIT  → flush input → FIND_INPUT（新 src）
    RE_COPY_HDR   → flush input → INPUT_DISCARD 或 FIND_OUTPUT

  FIND_OUTPUT
    RE_OUTPUT_DATA → FIND_OUTPUT（追加 DataLine）
    RE_COPY_HDR    → flush output → INPUT_DISCARD 或 FIND_OUTPUT

  INPUT_DISCARD
    RE_DATA_INIT  → flush（无数据）→ FIND_INPUT（新 src）
    RE_COPY_HDR   → INPUT_DISCARD 或 FIND_OUTPUT
    其余数据行    → 全部忽略
"""

from __future__ import annotations

import re
import logging
from enum import Enum, auto
from typing import Optional

from .op_log_parser import (
    OperandCfg, DataLine, InputOperand, OutputOperand, OpBlock,
    RE_CFG_OPCODE, RE_KV, RE_CFG_SRC, RE_BASE_ADDR,
    RE_DATA_INIT, RE_DIM_TOKEN, RE_COPY_HDR, RE_OUTPUT_DATA,
    _check_addr_continuity, _shape_str, _resolve_fmt, make_filename,
    write_txt, save_op, TxtFormat,
)

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
#  新增正则
# ══════════════════════════════════════════════════════════════

# word3: tid = 2
RE_TID = re.compile(r'word\d+:\s*tid\s*=\s*(?P<tid>\d+)')

# 新格式输入数据行：多字节空格分隔，无末尾十进制括注
# 示例：loop0, addr[0][0] = 0x100200, data = 0x00 01 02 03
RE_INPUT_DATA = re.compile(
    r'loop(?P<loop_idx>\d+)'
    r',\s*addr\[(?P<row_idx>\d+)\]\[(?P<col_idx>\d+)\]'
    r'\s*=\s*(?P<address>0x[0-9a-fA-F]+)'
    r',\s*data\s*=\s*0x(?P<hex_vals>[0-9a-fA-F]{2}(?:\s+[0-9a-fA-F]{2})*)$'
)


# ══════════════════════════════════════════════════════════════
#  状态枚举
# ══════════════════════════════════════════════════════════════

class State(Enum):
    IDLE          = auto()
    WAIT_OP       = auto()
    IN_OP         = auto()
    FIND_INPUT    = auto()
    FIND_OUTPUT   = auto()
    INPUT_DISCARD = auto()   # copy 对应 src 有 data_init，屏蔽后续数据行


# ══════════════════════════════════════════════════════════════
#  解析器上下文
# ══════════════════════════════════════════════════════════════

class ParserContext:
    """
    状态机运行时上下文，持有全部可变状态。

      state          当前状态
      pending_tid    见到 RE_TID 后暂存，等 opcode 行消费
      current_op     当前 OpBlock
      current_input  当前活跃 InputOperand
      current_output 当前活跃 OutputOperand
      input_src_set  本算子已出现 data_init 的 src_idx 集合
                     （有 data_init 的 src 对应的 copy_data_to_ram 丢弃）
      op_counter     自增 op_idx
      completed      已完成的 OpBlock 列表
    """

    def __init__(self, dtype_map: Optional[dict] = None) -> None:
        self.state:          State                   = State.IDLE
        self.pending_tid:    Optional[int]           = None
        self.current_op:     Optional[OpBlock]       = None
        self.current_input:  Optional[InputOperand]  = None
        self.current_output: Optional[OutputOperand] = None
        self.input_src_set:  set[int]                = set()
        self.op_counter:     int                     = 0
        self.completed:      list[OpBlock]           = []
        self.dtype_map:      dict                    = dtype_map or {}

    def _transition(self, new_state: State, reason: str = "") -> None:
        if self.state != new_state:
            logger.debug("状态 %s → %s  %s", self.state.name, new_state.name, reason)
            self.state = new_state

    # ── flush ────────────────────────────────────────────────

    def flush_input(self) -> None:
        if self.current_input is not None and self.current_op is not None:
            self.current_input.finalize()
            self.current_op.inputs.append(self.current_input)
            logger.debug("挂载 Input src%d → %s_%d",
                         self.current_input.src_idx,
                         self.current_op.opcode, self.current_op.op_idx)
        self.current_input = None

    def flush_output(self) -> None:
        if self.current_output is not None and self.current_op is not None:
            cfg = self.current_op.get_cfg(self.current_output.src_idx)
            self.current_output.finalize(cfg, self.dtype_map)
            self.current_op.outputs.append(self.current_output)
            logger.debug("挂载 Output src%d → %s_%d",
                         self.current_output.src_idx,
                         self.current_op.opcode, self.current_op.op_idx)
        self.current_output = None

    def flush_current_operand(self) -> None:
        if self.state == State.FIND_INPUT:
            self.flush_input()
        elif self.state == State.FIND_OUTPUT:
            self.flush_output()

    def flush_op(self) -> None:
        self.flush_current_operand()
        if self.current_op is not None:
            self.completed.append(self.current_op)
            logger.debug("完成算子 %s_%d  inputs=%d outputs=%d",
                         self.current_op.opcode, self.current_op.op_idx,
                         len(self.current_op.inputs), len(self.current_op.outputs))
        self.current_op    = None
        self.input_src_set = set()

    def route_copy(self, src_idx: int) -> State:
        """
        有 data_init 的 src → 对应 copy_data_to_ram 丢弃（FIND_INPUT，数据行忽略）
        无 data_init 的 src → copy_data_to_ram 为输出（FIND_OUTPUT）
        """
        if src_idx in self.input_src_set:
            logger.debug("copy_op%d 有 data_init，丢弃", src_idx)
            return State.INPUT_DISCARD
        logger.debug("copy_op%d 无 data_init，视为输出", src_idx)
        return State.FIND_OUTPUT



# ══════════════════════════════════════════════════════════════
#  状态机解析器
# ══════════════════════════════════════════════════════════════

class OpStateMachineParser:
    """用显式状态机解析新格式 op dump 日志。"""

    def __init__(self, log_path: str, dtype_map: Optional[dict] = None):
        self.log_path  = log_path
        self.dtype_map = dtype_map or {}

    def parse(self) -> list[OpBlock]:
        ctx = ParserContext(self.dtype_map)
        with open(self.log_path, encoding="utf-8", errors="replace") as fh:
            for lineno, raw in enumerate(fh, 1):
                self._dispatch(ctx, raw.rstrip(), lineno)
        ctx.flush_op()
        return ctx.completed

    def _dispatch(self, ctx: ParserContext, line: str, lineno: int) -> None:
        """按优先级依次匹配，命中则调用对应处理器。"""

        # ① tid：任意状态均响应
        if m := RE_TID.search(line):
            self._on_tid(ctx, m); return

        # ② opcode：WAIT_OP / IDLE
        if m := RE_CFG_OPCODE.search(line):
            self._on_opcode(ctx, line, m); return

        # ③ srcX 配置：IN_OP
        if ctx.state == State.IN_OP:
            if m := RE_CFG_SRC.match(line):
                self._on_cfg_src(ctx, m); return

        # ④ data_init：IN_OP / FIND_INPUT / INPUT_DISCARD
        if ctx.state in (State.IN_OP, State.FIND_INPUT, State.INPUT_DISCARD):
            if m := RE_DATA_INIT.search(line):
                self._on_data_init(ctx, m); return

        # ⑤ copy_data_to_ram：IN_OP / FIND_INPUT / FIND_OUTPUT / INPUT_DISCARD
        if ctx.state in (State.IN_OP, State.FIND_INPUT, State.FIND_OUTPUT, State.INPUT_DISCARD):
            if m := RE_COPY_HDR.search(line):
                self._on_copy_hdr(ctx, m); return

        # ⑥ 输入数据行：FIND_INPUT 专属（INPUT_DISCARD 下跳过）
        if ctx.state == State.FIND_INPUT:
            if m := RE_INPUT_DATA.search(line):
                self._on_input_data(ctx, m); return

        # ⑦ 输出数据行：FIND_OUTPUT
        if ctx.state == State.FIND_OUTPUT:
            if m := RE_OUTPUT_DATA.search(line):
                self._on_output_data(ctx, m); return

    # ── 事件处理器 ───────────────────────────────────────────

    def _on_tid(self, ctx: ParserContext, m: re.Match) -> None:
        ctx.flush_op()
        ctx.pending_tid = int(m.group("tid"))
        logger.debug("tid=%d", ctx.pending_tid)
        ctx._transition(State.WAIT_OP, f"tid={ctx.pending_tid}")

    def _on_opcode(self, ctx: ParserContext, line: str, m: re.Match) -> None:
        if ctx.state not in (State.WAIT_OP, State.IDLE):
            logger.warning("状态 %s 遇到 opcode，跳过", ctx.state.name)
            return

        opcode = m.group("opcode")
        params: dict[str, int | str] = {}
        for kv in RE_KV.finditer(line):
            k, v = kv.group("key"), kv.group("val")
            if k == "opcode":
                continue
            try:
                params[k] = int(v)
            except ValueError:
                params[k] = v

        tid = ctx.pending_tid
        if tid is None:
            logger.warning("opcode=%s 无对应 tid", opcode)

        ctx.current_op  = OpBlock(op_idx=ctx.op_counter, opcode=opcode,
                                  tid=tid, cfg_params=params)
        ctx.op_counter += 1
        ctx.pending_tid = None
        ctx._transition(State.IN_OP, f"opcode={opcode} tid={tid}")

    def _on_cfg_src(self, ctx: ParserContext, m: re.Match) -> None:
        src_idx = int(m.group("src_idx"))
        rest    = m.group("rest")

        base_addr: Optional[int] = None
        if ba := RE_BASE_ADDR.search(rest):
            base_addr = int(ba.group("addr"), 16)

        tokens = [t.strip().upper() for t in RE_BASE_ADDR.sub("", rest).split(",")]
        tokens = [t for t in tokens if t]
        fmt   = tokens[0] if len(tokens) > 0 else None
        dtype = tokens[1] if len(tokens) > 1 else None

        ctx.current_op.operand_cfgs[src_idx] = OperandCfg(
            src_idx=src_idx, fmt=fmt, dtype=dtype, base_addr=base_addr)
        logger.debug("cfg src%d: fmt=%s dtype=%s addr=0x%X",
                     src_idx, fmt, dtype, base_addr or 0)

    def _on_data_init(self, ctx: ParserContext, m: re.Match) -> None:
        ctx.flush_input()
        src_idx = int(m.group("src_idx"))
        ctx.current_input = InputOperand(
            src_idx   = src_idx,
            loop_dims = RE_DIM_TOKEN.findall(m.group("dims")),
            dtype     = m.group("dtype"),
            source    = m.group("source"),
        )
        ctx.input_src_set.add(src_idx)
        ctx._transition(State.FIND_INPUT, f"data_init src{src_idx}")

    def _on_copy_hdr(self, ctx: ParserContext, m: re.Match) -> None:
        """有 data_init 的 src → copy 块丢弃；无 data_init → Output。"""
        ctx.flush_current_operand()
        src_idx    = int(m.group("src_idx"))
        next_state = ctx.route_copy(src_idx)

        if next_state == State.FIND_INPUT:
            ctx.current_input  = None
            ctx.current_output = None
        else:
            offset_str = m.group("matrix_offset")
            ctx.current_output = OutputOperand(
                src_idx       = src_idx,
                op_name       = f"copy_op{src_idx}",
                fmt_in_line   = m.group("fmt_in_line") or None,
                matrix_offset = int(offset_str, 16) if offset_str.startswith("0x") else int(offset_str),
                write_length  = int(m.group("write_length")),
            )
            ctx.current_input = None

        ctx._transition(next_state, f"copy_op{src_idx}")

    def _on_input_data(self, ctx: ParserContext, m: re.Match) -> None:
        if ctx.current_input is None:
            return
        base_addr = int(m.group("address"), 16)
        loop_idx  = int(m.group("loop_idx"))
        row_idx   = int(m.group("row_idx")) if m.group("row_idx").isdigit() else 0
        col_idx   = int(m.group("col_idx")) if m.group("col_idx").isdigit() else 0
        for i, hv in enumerate(m.group("hex_vals").split()):
            ctx.current_input.lines.append(DataLine(
                loop_idx=loop_idx, row_idx=row_idx,
                col_idx=col_idx + i, address=base_addr + i, value=int(hv, 16),
            ))

    def _on_output_data(self, ctx: ParserContext, m: re.Match) -> None:
        if ctx.current_output is None:
            return
        base_addr = int(m.group("address"), 16)
        loop_idx  = int(m.group("loop_idx"))
        row_idx   = int(m.group("row_idx")) if m.group("row_idx").isdigit() else 0
        col_idx   = int(m.group("col_idx")) if m.group("col_idx").isdigit() else 0
        for i, hv in enumerate(m.group("hex_vals").split()):
            ctx.current_output.lines.append(DataLine(
                loop_idx=loop_idx, row_idx=row_idx,
                col_idx=col_idx + i, address=base_addr + i, value=int(hv, 16),
            ))
