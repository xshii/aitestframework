#!/usr/bin/env python3
"""
memory_state.py
===============
字节级稀疏内存状态，供 tensor commit 日志解析使用。

语义：
  - addr → byte 稀疏映射，未写入地址读取返回 0x00
  - byteEnable 写入：enable[i] != 0x00 → 写新值，enable[i] == 0x00 → 保留历史
  - 按操作顺序顺序累积，不做重排

跨文件接口（预留）：
  export_snapshot() / from_snapshot() 供后续时序信息接入时跨文件传递状态，
  当前暂不支持跨文件（无时序信息）。
"""

from __future__ import annotations


class MemoryState:
    """
    字节级内存映射。

      read(addr, size)                      → bytes
      write(addr, data)                     → None
      apply_with_enable(addr, data, enable) → bytes   byteEnable 写入，返回实际写入值
      snapshot(addr, size)                  → bytes   只读，不修改状态
      reset()                               → None
      written_ranges()                      → list[(start, end)]  已写区间（调试用）
      export_snapshot()                     → dict[int,int]       序列化快照
      from_snapshot(snap)                   → MemoryState         从快照恢复（类方法）
    """

    def __init__(self) -> None:
        self._mem: dict[int, int] = {}

    # ── 基础读写 ────────────────────────────────────────────

    def read(self, addr: int, size: int) -> bytes:
        """读取 [addr, addr+size)，未写入地址返回 0x00。"""
        return bytes(self._mem.get(addr + i, 0x00) for i in range(size))

    def write(self, addr: int, data: bytes) -> None:
        """写入 [addr, addr+len(data))，覆盖历史值。"""
        for i, b in enumerate(data):
            self._mem[addr + i] = b

    # ── byteEnable 写入 ─────────────────────────────────────

    def apply_with_enable(self,
                          addr: int,
                          write_data: bytes,
                          byte_enable: bytes) -> bytes:
        """
        按 byteEnable 合并 write_data 与历史值后写入，返回实际写入字节。

          byte_enable[i] != 0x00 → 写入 write_data[i]
          byte_enable[i] == 0x00 → 保留历史值（未写入地址默认 0x00）
        """
        result = bytearray(len(write_data))
        for i in range(len(write_data)):
            result[i] = write_data[i] if byte_enable[i] != 0x00 \
                        else self._mem.get(addr + i, 0x00)
        self.write(addr, bytes(result))
        return bytes(result)

    # ── 工具 ────────────────────────────────────────────────

    def snapshot(self, addr: int, size: int) -> bytes:
        """只读快照，等价于 read，明确表示不修改状态。"""
        return self.read(addr, size)

    def reset(self) -> None:
        """清空所有内存状态。"""
        self._mem.clear()

    def written_ranges(self) -> list[tuple[int, int]]:
        """
        返回已写入地址的连续区间列表 [(start, end), ...]（end 为最后地址+1）。
        主要用于调试和测试断言。
        """
        if not self._mem:
            return []
        addrs = sorted(self._mem)
        ranges: list[tuple[int, int]] = []
        start = prev = addrs[0]
        for a in addrs[1:]:
            if a == prev + 1:
                prev = a
            else:
                ranges.append((start, prev + 1))
                start = prev = a
        ranges.append((start, prev + 1))
        return ranges

    # ── 跨文件预留接口 ──────────────────────────────────────
    # 当前不支持跨文件（无时序信息）。
    # 后续有时序信息时，可用 export/from_snapshot 在文件间传递内存状态。

    def export_snapshot(self) -> dict[int, int]:
        """导出当前内存状态副本（addr → byte 字典）。"""
        return dict(self._mem)

    @classmethod
    def from_snapshot(cls, snapshot: dict[int, int]) -> "MemoryState":
        """
        从快照恢复 MemoryState（独立副本，不共享原始字典）。

        示例（跨文件时序接入）：
          snap     = parser_a.mem.export_snapshot()
          mem_b    = MemoryState.from_snapshot(snap)
          parser_b = TensorLogParser(log_b, cross_op=True, mem=mem_b)
        """
        obj = cls()
        obj._mem = dict(snapshot)
        return obj

    # ── dunder ──────────────────────────────────────────────

    def __len__(self) -> int:
        """已写入的字节数。"""
        return len(self._mem)

    def __repr__(self) -> str:
        ranges = self.written_ranges()
        return (f"MemoryState(bytes={len(self._mem)}, "
                f"ranges={ranges[:4]}{'...' if len(ranges) > 4 else ''})")
