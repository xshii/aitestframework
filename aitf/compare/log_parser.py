#!/usr/bin/env python3
"""
log_parser.py
=============
联合入口：同时解析 op dump 日志和 tensor commit 日志。

用法：
  python log_parser.py \
      --op     <op_log_path>      \
      --tensor <tensor_log_path>  \
      --out    <out_dir>          \
      --tid-names <tid_names.json>   # 可选
      --no-cross-op                  # 关闭跨 commit 内存累积（默认开启）
      --fmt    0                     # 0=WITH_HEADER_4B（默认），1=NO_HEADER_1B
      --config <config.yaml>         # 指定配置文件路径（默认与脚本同目录）
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Optional

from .op_log_parser     import save_op, TxtFormat, load_config
from .op_parser_state   import OpStateMachineParser
from .tensor_log_parser import TensorLogParser, MemoryState, \
                               save_tensor_blocks, load_tid_names

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.DEBUG)
logger = logging.getLogger(__name__)


def run(op_log:         Optional[str]  = None,
        tensor_log:     Optional[str]  = None,
        out_dir:        str            = "./output",
        tid_names_path: Optional[str]  = None,
        tid_names:      Optional[dict] = None,
        cross_op:       bool           = True,
        split_operand:  bool           = False,
        addr_shift:     int            = 16,
        txt_fmt:        TxtFormat      = TxtFormat.NO_HEADER_1B,
        dtype_map:      Optional[dict] = None) -> None:

    if tid_names is None:
        tid_names = load_tid_names(tid_names_path)

    # ── op dump 日志 ──────────────────────────────────────────
    if op_log:
        print(f"\n[op]  解析日志 : {op_log}")
        op_blocks = OpStateMachineParser(op_log, dtype_map=dtype_map).parse()
        print(f"[op]  共解析到 {len(op_blocks)} 个 op 块")

        for op in op_blocks:
            print(f"  {op.opcode}_{op.op_idx}  tid={op.tid}  params={op.cfg_params}")
            sub_dir = os.path.join(out_dir, "op", f"{op.opcode}_{op.op_idx}")
            saved   = save_op(op, sub_dir, txt_fmt)
            for role, idx, path, shape, dtype, fmt in saved:
                print(f"    {role}{idx}  {fmt} {dtype} {shape} → {path}")

    # ── tensor commit 日志 ────────────────────────────────────
    if tensor_log:
        print(f"\n[tensor]  解析日志 : {tensor_log}")
        print(f"[tensor]  cross_op  : {cross_op}")
        if tid_names_path:
            print(f"[tensor]  tid 映射  : {tid_names_path}")

        parser        = TensorLogParser(tensor_log, cross_op=cross_op,
                                         split_operand=split_operand,
                                         addr_shift=addr_shift)
        tensor_blocks = parser.parse()
        total = len(tensor_blocks)
        empty = sum(1 for b in tensor_blocks if b.is_empty)
        print(f"[tensor]  共解析到 {total} 个 tensor 块"
              f"（{total - empty} 有数据，{empty} 空块）")

        tensor_out = os.path.join(out_dir, "tensor")
        save_tensor_blocks(tensor_blocks, tensor_out, tid_names, txt_fmt,
                           operand_addrs=parser.operand_addrs)

        for blk in tensor_blocks:
            name   = tid_names.get(blk.tid)
            tag    = f"({name})" if name else ""
            status = "[empty]" if blk.is_empty else ""
            print(f"  tid={blk.tid}{tag}  cyc=0x{blk.cyc:X}  cmd={blk.cmd}"
                  f"  reads={len(blk.reads)}  writes={len(blk.writes)}  {status}")

    if not op_log and not tensor_log:
        logger.warning("未指定任何日志文件，退出。")
        return

    print("\n[✓] 完成")


def main() -> None:
    # 先解析 --config，以便用 yaml 值填充其他参数默认值
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()
    cfg = load_config(pre_args.config)

    fmt_map = {"0": TxtFormat.WITH_HEADER_4B, "1": TxtFormat.NO_HEADER_1B}

    ap = argparse.ArgumentParser(
        description="联合解析 op dump 日志和 tensor commit 日志"
    )
    ap.add_argument("--op",          metavar="PATH", default=cfg.get("op_log"),
                    help=f"op dump 日志路径（默认: {cfg.get('op_log')}）")
    ap.add_argument("--tensor",      metavar="PATH", default=cfg.get("tensor_log"),
                    help=f"tensor commit 日志路径（默认: {cfg.get('tensor_log')}）")
    ap.add_argument("--out",         metavar="DIR",  default=cfg["out_dir"],
                    help=f"输出目录（默认: {cfg['out_dir']}）")
    ap.add_argument("--tid-names",   metavar="JSON", default=cfg.get("tid_names"),
                    help="tid → name 映射 JSON（可选）")
    ap.add_argument("--no-cross-op", dest="cross_op",
                    action="store_false", default=cfg.get("cross_op", True),
                    help="关闭跨 commit 内存累积（默认开启）")
    ap.add_argument("--split-operand", action="store_true", default=False,
                    help="启用地址识别拆分操作数（默认关闭）")
    ap.add_argument("--addr-shift",  type=int,
                    default=cfg.get("addr_shift", 16),
                    help="receive 行地址左移位数（默认 16）")
    ap.add_argument("--fmt",         choices=["0", "1"],
                    default=str(cfg.get("txt_fmt", 1)),
                    help="输出 txt 格式：0=WITH_HEADER_4B，1=NO_HEADER_1B（默认）")
    ap.add_argument("--config",      default=None, metavar="PATH",
                    help="config.yaml 路径（默认与脚本同目录）")
    args = ap.parse_args()

    run(
        op_log         = args.op,
        tensor_log     = args.tensor,
        out_dir        = args.out,
        tid_names_path = args.tid_names,
        cross_op       = args.cross_op,
        split_operand  = args.split_operand,
        addr_shift     = args.addr_shift,
        txt_fmt        = fmt_map[args.fmt],
        dtype_map      = cfg.get("dtype_map"),
    )


if __name__ == "__main__":
    main()
