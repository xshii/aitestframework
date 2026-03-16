#!/usr/bin/env python3
"""
tensor_log_parser.py
====================
解析硬件 tensor commit 日志。

语义：
  read  行 → 输入（快照，不修改内存）
  write 行 → 输出（byteEnable 合并后写入内存）
  输出内容以 end tensor 时全局内存状态（actual_data）为准

byteEnable 规则（byte 级别）：
  0x00     → 保留该地址历史内存值（cross_op=False 时块内历史，缺省 0x00）
  非 0x00  → 写入 write 行对应字节

cross_op 开关：
  True  → 整个文件共享一个 MemoryState，跨 commit 内存状态累积（默认）
  False → 每个 commit 独立一个 MemoryState，byteEnable=0x00 取块内历史（缺省 0x00）

预留接口：
  TensorLogParser(log, cross_op=True, mem=existing_mem) 可传入外部 MemoryState，
  供后续跨文件时序场景接入。

用法：
  python tensor_log_parser.py <log> [out_dir] [tid_names.json] [--no-cross-op]
  from tensor_log_parser import TensorLogParser, save_tensor_blocks
"""

from __future__ import annotations

import re
import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from .memory_state  import MemoryState  # noqa: F401  re-export for backward compat
from .op_log_parser import write_txt, TxtFormat, _shape_str

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  数据结构
# ══════════════════════════════════════════════════════════════

@dataclass
class MemAccess:
    """单条 read/write 内存访问记录（解析阶段原始数据）。"""
    module:   str
    channel:  str
    addr:     int
    size:     int
    raw_data: bytes


@dataclass
class WriteRecord:
    """write + byteEnable 合并后的完整输出记录。"""
    module:      str
    channel:     str
    addr:        int
    size:        int
    raw_data:    bytes   # write 行原始数据
    byte_enable: bytes   # byteEnable 码流（长度 == size）
    actual_data: bytes   # 合并内存历史后实际写入的数据（输出内容）


@dataclass
class TensorBlock:
    """
    一次 commit 的完整记录。
      reads  → 输入（read 行，按出现顺序）
      writes → 输出（write + byteEnable 合并结果，actual_data 为真实输出值）
      is_empty → begin/end 之间无任何 read/write
    """
    cyc:      int
    cmd:      str
    group_id: int
    tid:      int
    reads:    list[MemAccess]   = field(default_factory=list)
    writes:   list[WriteRecord] = field(default_factory=list)
    is_empty: bool              = False


# ══════════════════════════════════════════════════════════════
#  正则
# ══════════════════════════════════════════════════════════════

RE_COMMIT = re.compile(
    r'commit\s+(?P<cmd>\w+)'
    r'.*?groupId\s*:\s*(?P<group_id>\d+)'
    r'.*?tid\s*:\s*(?P<tid>\d+)'
    r'.*?at\s+cyc\s*=\s*(?P<cyc>0x[0-9a-fA-F]+|\d+)'
)
RE_BEGIN      = re.compile(r'={3,}begin tensor={3,}')
RE_END        = re.compile(r'={3,}end tensor={3,}')
RE_MEM_ACCESS = re.compile(
    r'\[(?P<module>[^\]]+)\]\[(?P<channel>[^\]]+)\]'
    r'\s+(?P<op>read|write)\s+mem'
    r'\s+\[(?P<addr>0x[0-9a-fA-F]+)\]'
    r'\s+\[(?P<size>\d+)\]'
    r'\s*=\s*(?P<hex_data>[0-9a-fA-F]{2}(?:\s+[0-9a-fA-F]{2})*)'
)
RE_BYTE_ENABLE = re.compile(
    r'\[(?P<module>[^\]]+)\]\[(?P<channel>[^\]]+)\]'
    r'\s+byteEnable\s*=\s*(?P<hex_data>[0-9a-fA-F]{2}(?:\s+[0-9a-fA-F]{2})*)'
)

# 操作数地址映射
# 匹配 "src0_addr: 0x2800" / "dst_addr: 0x3580" / "src0 = 0x1000" 等 token
RE_OPERAND_ADDR = re.compile(
    r'(?P<role>src|dst|out)(?P<idx>\d+)?(?:_addr)?\s*[:=_]\s*'
    r'(?P<addr>0x[0-9a-fA-F]+)',
    re.IGNORECASE,
)
RE_TID_IN_LINE = re.compile(r'tid\s*[:=]\s*(?P<tid>\d+)')
# receive 行额外参数：M, N, K 等维度信息
RE_RECEIVE_CMD = re.compile(
    r'receive\s+(?P<cmd>\w+)',
    re.IGNORECASE,
)
# Dep_TID_xxx: N  — tid 依赖关系
RE_DEP_TID = re.compile(
    r'Dep_TID_(?P<dep_name>\w+)\s*:\s*(?P<dep_tid>\d+)',
    re.IGNORECASE,
)


# ══════════════════════════════════════════════════════════════
#  操作数地址拆分
# ══════════════════════════════════════════════════════════════

def _match_operand(addr: int, bases: dict[str, int]) -> Optional[str]:
    """
    根据地址找到所属操作数（addr >= base，取最近的）。

    bases 示例: {"src0": 0x1000, "src1": 0x2000}
    addr=0x1010 → "src0"（0x1010 >= 0x1000，且 < 0x2000）
    """
    best_name: Optional[str] = None
    best_base = -1
    for name, base in bases.items():
        if addr >= base and base > best_base:
            best_name, best_base = name, base
    return best_name


def _split_accesses(accesses: list,
                    operand_addrs: dict[str, int],
                    role_prefix: str) -> list[tuple[int, list]]:
    """
    将 reads 或 writes 按操作数拆分，返回 [(operand_idx, accesses), ...]。

    策略优先级：
      1. 有 operand_addrs 中对应 role 的映射 → 按地址归属分组
      2. 无映射但地址不连续 → 按间隙自动切割
      3. 地址连续 → 单组（index=0）
    """
    if not accesses:
        return []

    # ── 策略 1：按地址映射归属 ────────────────────────────────
    relevant = {k: v for k, v in operand_addrs.items()
                if k.lower().startswith(role_prefix)}

    if relevant:
        groups: dict[str, list] = {}
        for acc in accesses:
            name = _match_operand(acc.addr, relevant)
            if name is None:
                name = f"{role_prefix}0"
            groups.setdefault(name, []).append(acc)

        result = []
        for name in sorted(groups.keys()):
            m = re.search(r'\d+', name)
            idx = int(m.group()) if m else 0
            result.append((idx, sorted(groups[name], key=lambda a: a.addr)))
        return result

    # ── 策略 2/3：按地址间隙切割 ──────────────────────────────
    sorted_acc = sorted(accesses, key=lambda a: a.addr)
    chunks: list[list] = [[sorted_acc[0]]]
    for acc in sorted_acc[1:]:
        prev = chunks[-1][-1]
        if acc.addr == prev.addr + prev.size:
            chunks[-1].append(acc)
        else:
            chunks.append([acc])
    return [(i, g) for i, g in enumerate(chunks)]


# ══════════════════════════════════════════════════════════════
#  解析器
# ══════════════════════════════════════════════════════════════

class TensorLogParser:
    """
    解析 tensor commit 日志。

    参数：
      log_path  : 日志文件路径
      cross_op  : True  → 共享 MemoryState，跨 commit 状态累积（默认）
                  False → 每个 commit 独立 MemoryState
      mem       : 外部传入的 MemoryState（cross_op=True 且需要跨文件延续时使用）
    """

    def __init__(self,
                 log_path: str,
                 cross_op: bool = True,
                 mem: Optional[MemoryState] = None,
                 split_operand: bool = False,
                 addr_shift: int = 16):
        self.log_path  = log_path
        self.cross_op  = cross_op
        self.split_operand = split_operand
        self.addr_shift = addr_shift
        self._shared_mem = (mem if mem is not None else MemoryState()) if cross_op else None
        self.blocks: list[TensorBlock] = []
        self.operand_addrs: dict[int, dict[str, int]] = {}
        self.tid_deps: dict[int, dict[str, int]] = {}  # tid → {dep_name: dep_tid}

    @property
    def mem(self) -> Optional[MemoryState]:
        """当前共享内存状态（cross_op=False 时为 None）。"""
        return self._shared_mem

    def parse(self) -> list[TensorBlock]:
        with open(self.log_path, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()

        current_commit: Optional[dict]        = None
        in_tensor:      bool                  = False
        current_block:  Optional[TensorBlock] = None
        block_mem:      Optional[MemoryState] = None

        pending_writes: list[tuple[Optional[bytes], MemAccess]] = []
        pending_enable: Optional[bytes] = None

        for raw in lines:
            line = raw.rstrip()

            if m := RE_COMMIT.search(line):
                current_commit = self._parse_commit(m)
                continue

            # 操作数地址映射行（同一行有 tid 和 src/dst 地址）
            # 示例: receive xxxCmd, tid: 5, src0_addr: 0x2800, dst_addr: 0x3580, M: 0x10
            if self.split_operand and RE_OPERAND_ADDR.search(line):
                if tm := RE_TID_IN_LINE.search(line):
                    tid = int(tm.group("tid"))
                    for om in RE_OPERAND_ADDR.finditer(line):
                        addr = int(om.group("addr"), 16)
                        if addr == 0:
                            continue  # 0x0 表示未使用
                        role = om.group("role").lower()
                        idx  = om.group("idx") or "0"
                        # dst → out（内部统一用 out）
                        if role == "dst":
                            role = "out"
                        key = f"{role}{idx}"
                        # 地址左移还原真实地址
                        self.operand_addrs.setdefault(tid, {})[key] = addr << self.addr_shift
                    # 解析 Dep_TID_xxx: N
                    for dm in RE_DEP_TID.finditer(line):
                        dep_name = dm.group("dep_name")
                        dep_tid  = int(dm.group("dep_tid"))
                        self.tid_deps.setdefault(tid, {})[dep_name] = dep_tid
                    logger.debug("operand_addrs tid=%d: %s", tid,
                                 self.operand_addrs.get(tid, {}))
                    if tid in self.tid_deps:
                        logger.debug("tid_deps tid=%d: %s", tid,
                                     self.tid_deps[tid])
                continue

            if RE_BEGIN.search(line):
                if current_commit is None:
                    logger.warning("发现 begin tensor 但无前置 commit 行，跳过")
                    continue
                in_tensor      = True
                current_block  = TensorBlock(**current_commit)
                # 不清空 current_commit，同一 commit 下可能有多个 begin/end 对
                pending_writes = []
                pending_enable = None
                block_mem = self._shared_mem if self.cross_op else MemoryState()
                continue

            if RE_END.search(line):
                if current_block is None:
                    logger.warning("发现 end tensor 但无对应 begin，跳过")
                    in_tensor = False
                    continue

                in_tensor = False
                if not pending_writes and not current_block.reads:
                    current_block.is_empty = True
                    logger.warning(
                        "空 tensor 块：tid=%d cyc=0x%X cmd=%s",
                        current_block.tid, current_block.cyc, current_block.cmd,
                    )
                else:
                    self._flush_writes(current_block, pending_writes, block_mem)

                self.blocks.append(current_block)
                current_block  = None
                block_mem      = None
                pending_writes = []
                pending_enable = None
                continue

            if not in_tensor or current_block is None:
                continue

            if m := RE_MEM_ACCESS.search(line):
                access = self._parse_mem_access(m)
                if m.group("op") == "read":
                    current_block.reads.append(access)
                else:
                    pending_writes.append((pending_enable, access))
                    pending_enable = None
                continue

            if m := RE_BYTE_ENABLE.search(line):
                pending_enable = bytes(
                    int(b, 16) for b in m.group("hex_data").split()
                )
                continue

        return self.blocks

    @staticmethod
    def _parse_commit(m: re.Match) -> dict:
        cyc_s = m.group("cyc")
        cyc   = int(cyc_s, 16) if cyc_s.startswith("0x") else int(cyc_s)
        return dict(cyc=cyc, cmd=m.group("cmd"),
                    group_id=int(m.group("group_id")), tid=int(m.group("tid")))

    @staticmethod
    def _parse_mem_access(m: re.Match) -> MemAccess:
        return MemAccess(
            module   = m.group("module"),
            channel  = m.group("channel"),
            addr     = int(m.group("addr"), 16),
            size     = int(m.group("size")),
            raw_data = bytes(int(b, 16) for b in m.group("hex_data").split()),
        )

    @staticmethod
    def _flush_writes(block: TensorBlock,
                      pending_writes: list[tuple[Optional[bytes], MemAccess]],
                      mem: Optional[MemoryState]) -> None:
        if mem is None:
            logger.error("_flush_writes: mem 为 None，跳过写入")
            return

        for enable, acc in pending_writes:
            if enable is None:
                logger.warning(
                    "write addr=0x%08X size=%d 无对应 byteEnable，视为全字节写入",
                    acc.addr, acc.size,
                )
                enable = bytes([0x01] * acc.size)

            if len(enable) != acc.size:
                logger.warning(
                    "byteEnable 长度 %d 与 write size %d 不符 addr=0x%08X，视为全字节写入",
                    len(enable), acc.size, acc.addr,
                )
                enable = bytes([0x01] * acc.size)

            actual = mem.apply_with_enable(acc.addr, acc.raw_data, enable)
            block.writes.append(WriteRecord(
                module=acc.module, channel=acc.channel,
                addr=acc.addr, size=acc.size,
                raw_data=acc.raw_data, byte_enable=enable, actual_data=actual,
            ))


# ══════════════════════════════════════════════════════════════
#  输出
# ══════════════════════════════════════════════════════════════

def _hex_str(data: bytes) -> str:
    return " ".join(f"{b:02X}" for b in data)


def load_tid_names(json_path: Optional[str]) -> dict:
    """加载 tid → name/meta 映射 JSON。
    支持旧格式 {"3": "matmul"} 和新格式 {"3": {"name": "matmul", "reads": {...}, "writes": {...}}}。
    路径为 None 或文件不存在时返回空字典。
    """
    if json_path is None:
        return {}
    try:
        with open(json_path, encoding="utf-8") as fh:
            raw = json.load(fh)
        return {int(k): v for k, v in raw.items()}
    except Exception as e:
        logger.warning("无法加载 tid 映射文件 %s：%s", json_path, e)
        return {}


def _tid_info(tid: int, tid_names: dict) -> tuple[str, dict, dict]:
    """从 tid_names 取 name、reads_meta、writes_meta。"""
    entry = tid_names.get(tid) if isinstance(tid_names.get(tid), dict) else {}
    name  = entry.get("name") or (tid_names.get(tid) if isinstance(tid_names.get(tid), str) else None) or "xxx"
    reads_meta  = entry.get("reads",  {}) or {}
    writes_meta = entry.get("writes", {}) or {}
    return name, reads_meta, writes_meta


def _meta_filename(cmd: str, blk_idx: int, role: str,
                   meta: dict, operand_idx: int = 0) -> str:
    """生成与 op 一致的文件名：CMD_idx_Input0_FMT_DTYPE_SHAPE.txt"""
    fmt   = meta.get("fmt")   or "FORMAT"
    dtype = meta.get("dtype") or "DTYPE"
    shape = meta.get("shape") or []
    shape_s = _shape_str(shape) if shape else "UNKNOWN"
    return f"{cmd}_{blk_idx}_{role}{operand_idx}_{fmt}_{dtype}_{shape_s}.txt"


def _reads_raw(blk: TensorBlock) -> bytes:
    """将所有 read 数据拼接为连续 bytes。"""
    return b"".join(r.raw_data for r in blk.reads)


def _writes_raw(blk: TensorBlock) -> bytes:
    """将所有 write actual_data 拼接为连续 bytes。"""
    return b"".join(w.actual_data for w in blk.writes)


def _block_filename(blk: TensorBlock, tid_names: dict) -> str:
    name, _, _ = _tid_info(blk.tid, tid_names)
    return f"{name}_{blk.tid}_cyc{blk.cyc:08X}_{blk.cmd}"


def save_tensor_blocks(blocks: list[TensorBlock],
                       out_dir: str,
                       tid_names: Optional[dict] = None,
                       txt_fmt: TxtFormat = TxtFormat.NO_HEADER_1B,
                       operand_addrs: Optional[dict[int, dict[str, int]]] = None,
                       ) -> None:
    """输出汇总 JSON + 每个非空 block 的 Input/Output txt 文件。

    operand_addrs 不为空时按操作数地址拆分，否则按地址连续性自动切割。
    """
    if tid_names is None:
        tid_names = {}
    if operand_addrs is None:
        operand_addrs = {}
    os.makedirs(out_dir, exist_ok=True)

    # ── 汇总 JSON ────────────────────────────────────────────
    records = []
    for blk in blocks:
        entry: dict = {
            "tid": blk.tid, "cyc": f"0x{blk.cyc:X}",
            "cmd": blk.cmd, "group_id": blk.group_id, "is_empty": blk.is_empty,
        }
        if not blk.is_empty:
            entry["reads"] = [
                {"module": r.module, "channel": r.channel,
                 "addr": f"0x{r.addr:08X}", "size": r.size,
                 "data": _hex_str(r.raw_data)}
                for r in blk.reads
            ]
            entry["writes"] = [
                {"module": w.module, "channel": w.channel,
                 "addr": f"0x{w.addr:08X}", "size": w.size,
                 "raw_data":    _hex_str(w.raw_data),
                 "byte_enable": _hex_str(w.byte_enable),
                 "actual_data": _hex_str(w.actual_data)}
                for w in blk.writes
            ]
        records.append(entry)

    summary_path = os.path.join(out_dir, "tensor_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)
    logger.debug("汇总写入: %s", summary_path)

    # ── 每个非空 block → 按操作数拆分输出 txt ────────────────
    blk_counter: dict[tuple, int] = {}

    for blk in blocks:
        if blk.is_empty:
            continue

        key     = (blk.tid, blk.cmd)
        blk_idx = blk_counter.get(key, 0)
        blk_counter[key] = blk_idx + 1

        name, reads_meta, writes_meta = _tid_info(blk.tid, tid_names)
        sub_dir = os.path.join(out_dir, f"{blk.cmd}_{blk_idx}")
        os.makedirs(sub_dir, exist_ok=True)

        tid_addrs = operand_addrs.get(blk.tid, {})

        # Input（reads） — 按操作数 / 地址间隙拆分
        if blk.reads:
            for op_idx, group in _split_accesses(blk.reads, tid_addrs, "src"):
                raw  = b"".join(a.raw_data for a in group)
                addr = group[0].addr
                fname = _meta_filename(blk.cmd, blk_idx, "Input",
                                       reads_meta, op_idx)
                write_txt(os.path.join(sub_dir, fname), raw, addr, txt_fmt)

        # Output（writes actual_data） — 按操作数 / 地址间隙拆分
        if blk.writes:
            for op_idx, group in _split_accesses(blk.writes, tid_addrs, "out"):
                raw  = b"".join(w.actual_data for w in group)
                addr = group[0].addr
                fname = _meta_filename(blk.cmd, blk_idx, "Output",
                                       writes_meta, op_idx)
                write_txt(os.path.join(sub_dir, fname), raw, addr, txt_fmt)


# ══════════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════════

def main(log_path: str,
         out_dir: str = "./tensor_output",
         tid_names_path: Optional[str] = None,
         cross_op: bool = True,
         split_operand: bool = False,
         addr_shift: int = 16,
         txt_fmt: TxtFormat = TxtFormat.NO_HEADER_1B) -> None:
    print(f"[*] 解析日志       : {log_path}")
    print(f"[*] cross_op       : {cross_op}")
    print(f"[*] split_operand  : {split_operand}")
    if tid_names_path:
        print(f"[*] tid 映射文件   : {tid_names_path}")

    tid_names = load_tid_names(tid_names_path)
    parser    = TensorLogParser(log_path, cross_op=cross_op,
                                split_operand=split_operand,
                                addr_shift=addr_shift)
    blocks    = parser.parse()

    total = len(blocks)
    empty = sum(1 for b in blocks if b.is_empty)
    print(f"[*] 共解析到 {total} 个 tensor 块（{total - empty} 有数据，{empty} 空块）")

    save_tensor_blocks(blocks, out_dir, tid_names, txt_fmt,
                       operand_addrs=parser.operand_addrs)

    for blk in blocks:
        name, _, _ = _tid_info(blk.tid, tid_names)
        status = "[empty]" if blk.is_empty else ""
        print(f"  tid={blk.tid}({name})  cyc=0x{blk.cyc:X}  cmd={blk.cmd}"
              f"  reads={len(blk.reads)}  writes={len(blk.writes)}  {status}")

    print("[✓] 完成")


if __name__ == "__main__":
    import argparse
    from .op_log_parser import load_config
    _cfg = load_config()

    ap = argparse.ArgumentParser(description="解析 tensor commit 日志")
    ap.add_argument("log",      nargs="?", default=_cfg.get("tensor_log"),
                    help=f"日志文件路径（默认: {_cfg.get('tensor_log')}）")
    ap.add_argument("out_dir",  nargs="?", default=_cfg["out_dir"], help="输出目录")
    ap.add_argument("tid_names", nargs="?", default=_cfg.get("tid_names"),
                    help="tid 映射 JSON 路径")
    ap.add_argument("--no-cross-op", dest="cross_op", action="store_false", default=True)
    ap.add_argument("--split-operand", action="store_true", default=False,
                    help="启用地址识别拆分操作数（默认关闭）")
    ap.add_argument("--addr-shift", type=int, default=_cfg.get("addr_shift", 16),
                    help="receive 行地址左移位数（默认 16）")
    ap.add_argument("--fmt", choices=["0", "1"], default=str(_cfg.get("txt_fmt", 1)),
                    help="输出格式：0=WITH_HEADER_4B，1=NO_HEADER_1B")
    ap.add_argument("--config", default=None, metavar="PATH")
    args = ap.parse_args()

    if args.config:
        _cfg = load_config(args.config)

    fmt_map = {"0": TxtFormat.WITH_HEADER_4B, "1": TxtFormat.NO_HEADER_1B}
    main(args.log, args.out_dir, args.tid_names, args.cross_op,
         args.split_operand, args.addr_shift, fmt_map[args.fmt])
