"""Ascend tensor layout & dtype conversion utilities.

Unified conversion: layout and dtype changes are coupled for blocked layouts
(NC1HWC0, FRACTAL_NZ) because block size (C0/tile) depends on precision.
Use `convert()` as the single entry point.

Supported layouts:
- NCHW <-> NC1HWC0  (C split into C1 * C0, C0 depends on precision)
- ND   <-> FRACTAL_NZ (2D matrix tiled into fractal blocks, tile=16)
- NCHW <-> NHWC (axis permutation)

Blocked layouts (C0 depends on precision):
  FP16 / BF16 -> 16
  FP32        -> 16
  INT8 / UINT8 -> 32
  INT16       -> 16
  INT32       -> 8
  INT4        -> 64
  BFP16       -> 16
  BFP8        -> 16

TODO:
  - FRACTAL_Z: 与 FRACTAL_NZ 的区别在于外层按 (M1,N1) 排列，
    且 tile 内按 Z 序排列。目前仅支持 FRACTAL_NZ。
  - INT4 打包/解包 (2 个 INT4 packed per byte)
  - ND <-> NC1HWC0 直接转换（目前需 ND→NCHW→NC1HWC0 两步）
  - 5D tensor (NCDHW) 支持
  - BFP8/BFP16 block floating-point 转换
  - BOOL/BITMAP 语义转换
"""

from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

# C0 size per precision (Ascend cube unit size)
C0_MAP: dict[str, int] = {
    "BFP8": 16, "BFP16": 16,
    "FP16": 16, "FP32": 16, "BF16": 16,
    "INT8": 32, "UINT8": 32, "INT16": 16,
    "INT32": 8, "INT4": 64,
    "BOOL": 32, "BITMAP": 32,
}

# numpy dtype per precision
DTYPE_MAP: dict[str, np.dtype] = {
    "BFP8": np.dtype("uint8"),     # block floating-point 8-bit
    "BFP16": np.dtype("uint16"),   # block floating-point 16-bit
    "FP16": np.dtype("float16"),
    "FP32": np.dtype("float32"),
    "BF16": np.dtype("uint16"),    # no native bf16 in numpy
    "INT8": np.dtype("int8"),
    "UINT8": np.dtype("uint8"),
    "INT16": np.dtype("int16"),
    "INT32": np.dtype("int32"),
    "INT4": np.dtype("uint8"),     # packed 2 per byte
    "BOOL": np.dtype("uint8"),     # 1 byte per element
    "BITMAP": np.dtype("uint8"),   # bit-packed
}

TILE = 16  # fractal tile size

# Flat (non-blocked) layouts — dtype conversion is a simple element-wise cast
_FLAT_LAYOUTS = {"NCHW", "NHWC", "ND"}
# Blocked layouts — dtype change requires unblock → cast → re-block
_BLOCKED_LAYOUTS = {"NC1HWC0", "FRACTAL_NZ"}
# Canonical flat layout for each blocked layout (used as intermediate)
_UNBLOCK_MAP = {"NC1HWC0": "NCHW", "FRACTAL_NZ": "ND"}


def _get_c0(precision: str) -> int:
    return C0_MAP.get(precision, 16)


def _get_dtype(precision: str) -> np.dtype:
    return DTYPE_MAP.get(precision, np.dtype("float16"))


def _read_raw(data: bytes, precision: str, expected: int) -> np.ndarray:
    """Read raw binary data into a numpy array, pad if needed."""
    dt = _get_dtype(precision)
    arr = np.frombuffer(data, dtype=dt)
    if arr.size < expected:
        arr = np.pad(arr, (0, expected - arr.size))
    return arr[:expected]


# ---------------------------------------------------------------------------
# Low-level layout converters (operate on numpy arrays)
# ---------------------------------------------------------------------------

def _nchw_to_nc1hwc0(arr: np.ndarray, n: int, c: int, h: int, w: int,
                      precision: str) -> np.ndarray:
    c0 = _get_c0(precision)
    c1 = math.ceil(c / c0)
    a = arr.reshape(n, c, h, w)
    pad_c = c1 * c0 - c
    if pad_c > 0:
        a = np.pad(a, ((0, 0), (0, pad_c), (0, 0), (0, 0)))
    return a.reshape(n, c1, c0, h, w).transpose(0, 1, 3, 4, 2)  # (N,C1,H,W,C0)


def _nc1hwc0_to_nchw(arr: np.ndarray, n: int, c: int, h: int, w: int,
                      precision: str) -> np.ndarray:
    c0 = _get_c0(precision)
    c1 = math.ceil(c / c0)
    a = arr.reshape(n, c1, h, w, c0).transpose(0, 1, 4, 2, 3)  # (N,C1,C0,H,W)
    return a.reshape(n, c1 * c0, h, w)[:, :c, :, :]


def _nd_to_fractal_nz(arr: np.ndarray, m: int, n: int,
                       precision: str) -> np.ndarray:
    tile = TILE
    a = arr.reshape(m, n)
    m1 = math.ceil(m / tile)
    n1 = math.ceil(n / tile)
    pm, pn = m1 * tile - m, n1 * tile - n
    if pm > 0 or pn > 0:
        a = np.pad(a, ((0, pm), (0, pn)))
    return a.reshape(m1, tile, n1, tile).transpose(2, 0, 1, 3)  # (n1,m1,tile,tile)


def _fractal_nz_to_nd(arr: np.ndarray, m: int, n: int,
                       precision: str) -> np.ndarray:
    tile = TILE
    m1 = math.ceil(m / tile)
    n1 = math.ceil(n / tile)
    a = arr.reshape(n1, m1, tile, tile).transpose(1, 2, 0, 3)  # (m1,tile,n1,tile)
    return a.reshape(m1 * tile, n1 * tile)[:m, :n]


def _nchw_to_nhwc(arr: np.ndarray, n: int, c: int, h: int, w: int,
                   precision: str) -> np.ndarray:
    return arr.reshape(n, c, h, w).transpose(0, 2, 3, 1)


def _nhwc_to_nchw(arr: np.ndarray, n: int, c: int, h: int, w: int,
                   precision: str) -> np.ndarray:
    return arr.reshape(n, h, w, c).transpose(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# Converter registry
# ---------------------------------------------------------------------------

# (src_layout, dst_layout) -> (converter_fn, arg_style)
# arg_style: "4d" needs (n,c,h,w), "2d" needs (m,n)
_LAYOUT_CONVERTERS: dict[tuple[str, str], tuple[callable, str]] = {
    ("NCHW", "NC1HWC0"):   (_nchw_to_nc1hwc0,  "4d"),
    ("NC1HWC0", "NCHW"):   (_nc1hwc0_to_nchw,   "4d"),
    ("ND", "FRACTAL_NZ"):  (_nd_to_fractal_nz,  "2d"),
    ("FRACTAL_NZ", "ND"):  (_fractal_nz_to_nd,  "2d"),
    ("NCHW", "NHWC"):      (_nchw_to_nhwc,      "4d"),
    ("NHWC", "NCHW"):      (_nhwc_to_nchw,      "4d"),
}

# For backward compat (routes.py uses CONVERTERS.keys())
CONVERTERS = {k: v[0] for k, v in _LAYOUT_CONVERTERS.items()}



# ---------------------------------------------------------------------------
# Core dtype cast (numpy array level)
# ---------------------------------------------------------------------------

def _cast_dtype(arr: np.ndarray, src_prec: str, dst_prec: str) -> np.ndarray:
    """Element-wise dtype conversion.

    All types go through FP32 as intermediate representation.
    Handles special encodings: BF16, BFP8, BFP16, BOOL, BITMAP, INT4.

    TODO: 待实现的转换路径 (当前使用近似/fallback):
      - BFP8 ↔ any: block floating-point 需要 shared exponent 解码/编码
      - BFP16 ↔ any: block floating-point 需要 shared exponent 解码/编码
      - INT4 pack/unpack: 2 个 INT4 packed per byte, 需要拆分/合并
      - BITMAP ↔ any: bit-packed 格式, 需要 bit-level 解包/打包
    """
    if src_prec == dst_prec:
        return arr

    # --- Decode source to FP32 intermediate ---
    if src_prec == "BF16":
        fp32 = np.zeros(arr.shape, dtype=np.float32)
        fp32.view(np.uint32)[:] = arr.astype(np.uint32) << 16
        arr = fp32
        src_prec = "FP32"
    elif src_prec == "BFP16":
        # TODO: proper block floating-point decode (shared exponent per block)
        arr = arr.view(np.float16).astype(np.float32)
        src_prec = "FP32"
        logger.warning("BFP16 decode: using FP16 bit-pattern fallback")
    elif src_prec == "BFP8":
        # TODO: proper block floating-point decode
        arr = arr.astype(np.float32)
        src_prec = "FP32"
        logger.warning("BFP8 decode: using UINT8->FP32 fallback")
    elif src_prec == "BOOL":
        arr = arr.astype(np.float32)
        src_prec = "FP32"
    elif src_prec == "BITMAP":
        # TODO: bit-level unpack
        arr = arr.astype(np.float32)
        src_prec = "FP32"
        logger.warning("BITMAP decode: treating raw bytes as UINT8 fallback")
    elif src_prec == "INT4":
        # TODO: unpack 2x INT4 per byte
        arr = arr.astype(np.float32)
        src_prec = "FP32"
        logger.warning("INT4 decode: treating packed bytes as UINT8 fallback")

    # --- Encode to destination ---
    if dst_prec == "BF16":
        fp32 = arr.astype(np.float32)
        return (fp32.view(np.uint32) >> 16).astype(np.uint16)
    if dst_prec == "BFP16":
        # TODO: proper block floating-point encode
        logger.warning("BFP16 encode: using FP16 bit-pattern fallback")
        return arr.astype(np.float16).view(np.uint16)
    if dst_prec == "BFP8":
        # TODO: proper block floating-point encode
        logger.warning("BFP8 encode: using UINT8 clamp fallback")
        return np.clip(arr.astype(np.float32), 0, 255).astype(np.uint8)
    if dst_prec == "BOOL":
        return (arr != 0).astype(np.uint8)
    if dst_prec == "BITMAP":
        # TODO: bit-level pack
        logger.warning("BITMAP encode: using UINT8 fallback (no bit-packing)")
        return (arr != 0).astype(np.uint8)
    if dst_prec == "INT4":
        # TODO: pack 2x INT4 per byte
        logger.warning("INT4 encode: using UINT8 clamp fallback (no packing)")
        return np.clip(arr.astype(np.int32), -8, 7).astype(np.uint8)

    return arr.astype(_get_dtype(dst_prec))


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------

def _extract_4d(shape: dict) -> tuple[int, int, int, int]:
    """Extract (n, c, h, w) from shape dict {loop, m, n, k}."""
    return shape.get("loop", 1), shape.get("m", 1), shape.get("n", 1), shape.get("k", 1)


def _extract_2d(shape: dict) -> tuple[int, int]:
    """Extract (m, n) from shape dict, folding loop and k."""
    return (shape.get("m", 1) * shape.get("loop", 1),
            shape.get("n", 1) * shape.get("k", 1))


def _call_layout_fn(fn, arg_style: str, arr: np.ndarray,
                    shape: dict, precision: str) -> np.ndarray:
    if arg_style == "4d":
        n, c, h, w = _extract_4d(shape)
        return fn(arr, n, c, h, w, precision)
    else:
        m, n = _extract_2d(shape)
        return fn(arr, m, n, precision)


def _element_count(shape: dict, layout: str) -> int:
    """Compute expected element count for a flat layout."""
    if layout in ("NCHW", "NHWC", "NC1HWC0"):
        n, c, h, w = _extract_4d(shape)
        return n * c * h * w
    else:  # ND, FRACTAL_NZ
        m, n = _extract_2d(shape)
        return m * n


# ---------------------------------------------------------------------------
# Unified convert() — single entry point
# ---------------------------------------------------------------------------

def can_convert(src_layout: str, dst_layout: str) -> bool:
    """Check if a layout conversion path exists."""
    return (src_layout.upper(), dst_layout.upper()) in _LAYOUT_CONVERTERS


def _validate_convert(src_layout: str, dst_layout: str,
                      src_precision: str, dst_precision: str) -> None:
    """Pre-validate a conversion request, raise ValueError with clear message."""
    all_layouts = _FLAT_LAYOUTS | _BLOCKED_LAYOUTS
    if src_layout not in all_layouts:
        raise ValueError(f"未知的源分型: {src_layout}，支持: {sorted(all_layouts)}")
    if dst_layout not in all_layouts:
        raise ValueError(f"未知的目标分型: {dst_layout}，支持: {sorted(all_layouts)}")

    all_precs = set(DTYPE_MAP.keys())
    if src_precision not in all_precs:
        raise ValueError(f"未知的源数据类型: {src_precision}，支持: {sorted(all_precs)}")
    if dst_precision not in all_precs:
        raise ValueError(f"未知的目标数据类型: {dst_precision}，支持: {sorted(all_precs)}")

    # Check layout path exists (for layout changes)
    if src_layout != dst_layout:
        if (src_layout, dst_layout) in _LAYOUT_CONVERTERS:
            pass
        elif (src_layout in _BLOCKED_LAYOUTS and dst_layout in _BLOCKED_LAYOUTS):
            flat_s = _UNBLOCK_MAP[src_layout]
            flat_d = _UNBLOCK_MAP[dst_layout]
            if flat_s != flat_d and (flat_s, flat_d) not in _LAYOUT_CONVERTERS:
                raise ValueError(
                    f"不支持的分型转换: {src_layout} -> {dst_layout} "
                    f"(中间格式 {flat_s} -> {flat_d} 无转换路径)")
        else:
            raise ValueError(
                f"不支持的分型转换: {src_layout} -> {dst_layout}，"
                f"支持: {[f'{s}->{d}' for s, d in _LAYOUT_CONVERTERS]}")

    # Warn about precision-dependent blocked layout coupling
    if (src_precision != dst_precision and
            (src_layout in _BLOCKED_LAYOUTS or dst_layout in _BLOCKED_LAYOUTS)):
        src_c0 = _get_c0(src_precision)
        dst_c0 = _get_c0(dst_precision)
        if src_c0 != dst_c0:
            logger.warning(
                "convert: blocked layout dtype change %s->%s causes C0 change "
                "%d->%d, data will be unblocked and re-blocked",
                src_precision, dst_precision, src_c0, dst_c0)


def convert(data: bytes, src_layout: str, dst_layout: str,
            src_precision: str, dst_precision: str,
            shape: dict) -> bytes:
    """Unified layout + dtype conversion (single entry point).

    For blocked layouts (NC1HWC0, FRACTAL_NZ), block size (C0) depends on
    precision. When dtype changes, data is automatically unblocked → cast →
    re-blocked with the new C0. This coupling is handled transparently.

    Args:
        data: Raw binary tensor data.
        src_layout: Source layout (NCHW/NHWC/ND/NC1HWC0/FRACTAL_NZ).
        dst_layout: Target layout.
        src_precision: Source dtype (FP16/FP32/BF16/INT8/...), case-insensitive.
        dst_precision: Target dtype, case-insensitive.
        shape: Dict with keys {loop, m, n, k}.

    Returns:
        Converted binary data.

    Raises:
        ValueError: If conversion path is unsupported (with clear message).
    """
    src_layout = src_layout.upper()
    dst_layout = dst_layout.upper()
    src_precision = src_precision.upper()
    dst_precision = dst_precision.upper()

    layout_change = (src_layout != dst_layout)
    dtype_change = (src_precision != dst_precision)

    if not layout_change and not dtype_change:
        return data

    _validate_convert(src_layout, dst_layout, src_precision, dst_precision)

    logger.info("convert: %s/%s -> %s/%s, shape=%s, data_len=%d",
                src_layout, src_precision, dst_layout, dst_precision, shape, len(data))

    # --- Case 1: dtype-only, flat layout ---
    if not layout_change and src_layout in _FLAT_LAYOUTS:
        arr = np.frombuffer(data, dtype=_get_dtype(src_precision))
        return _cast_dtype(arr, src_precision, dst_precision).tobytes()

    # --- Case 2: dtype-only, blocked layout → unblock/cast/re-block ---
    if not layout_change and src_layout in _BLOCKED_LAYOUTS:
        flat = _UNBLOCK_MAP[src_layout]
        fn_unblock, style = _LAYOUT_CONVERTERS[(src_layout, flat)]
        fn_reblock, _ = _LAYOUT_CONVERTERS[(flat, src_layout)]
        arr = np.frombuffer(data, dtype=_get_dtype(src_precision))
        flat_arr = _call_layout_fn(fn_unblock, style, arr, shape, src_precision)
        cast_arr = _cast_dtype(flat_arr.ravel(), src_precision, dst_precision)
        result = _call_layout_fn(fn_reblock, style, cast_arr, shape, dst_precision)
        return result.tobytes()

    # --- Case 3: layout change ---
    src_blocked = src_layout in _BLOCKED_LAYOUTS
    dst_blocked = dst_layout in _BLOCKED_LAYOUTS

    # flat → flat
    if not src_blocked and not dst_blocked:
        elements = _element_count(shape, src_layout)
        arr = _read_raw(data, src_precision, elements)
        if dtype_change:
            arr = _cast_dtype(arr, src_precision, dst_precision)
        prec = dst_precision if dtype_change else src_precision
        fn, style = _LAYOUT_CONVERTERS[(src_layout, dst_layout)]
        return _call_layout_fn(fn, style, arr, shape, prec).tobytes()

    # blocked → flat
    if src_blocked and not dst_blocked:
        fn, style = _LAYOUT_CONVERTERS[(src_layout, dst_layout)]
        arr = np.frombuffer(data, dtype=_get_dtype(src_precision))
        flat_arr = _call_layout_fn(fn, style, arr, shape, src_precision)
        if dtype_change:
            flat_arr = _cast_dtype(flat_arr.ravel(), src_precision, dst_precision)
        return flat_arr.tobytes()

    # flat → blocked
    if not src_blocked and dst_blocked:
        elements = _element_count(shape, src_layout)
        arr = _read_raw(data, src_precision, elements)
        if dtype_change:
            arr = _cast_dtype(arr, src_precision, dst_precision)
        prec = dst_precision if dtype_change else src_precision
        fn, style = _LAYOUT_CONVERTERS[(src_layout, dst_layout)]
        return _call_layout_fn(fn, style, arr, shape, prec).tobytes()

    # blocked → blocked (via flat intermediate)
    flat_src = _UNBLOCK_MAP[src_layout]
    flat_dst = _UNBLOCK_MAP[dst_layout]
    fn_unblock, style_u = _LAYOUT_CONVERTERS[(src_layout, flat_src)]
    arr = np.frombuffer(data, dtype=_get_dtype(src_precision))
    flat_arr = _call_layout_fn(fn_unblock, style_u, arr, shape, src_precision)
    if dtype_change:
        flat_arr = _cast_dtype(flat_arr.ravel(), src_precision, dst_precision)
    prec = dst_precision if dtype_change else src_precision
    if flat_src != flat_dst:
        fn_mid, style_m = _LAYOUT_CONVERTERS[(flat_src, flat_dst)]
        flat_arr = _call_layout_fn(fn_mid, style_m, flat_arr.ravel(), shape, prec)
    fn_reblock, style_r = _LAYOUT_CONVERTERS[(flat_dst, dst_layout)]
    return _call_layout_fn(fn_reblock, style_r, flat_arr.ravel(), shape, prec).tobytes()


# ---------------------------------------------------------------------------
# txt (hex) <-> bin conversion
# ---------------------------------------------------------------------------

def bin_to_hex_txt(data: bytes) -> str:
    """Convert binary data to hex text format (1 byte per line, no 0x prefix)."""
    return "\n".join(f"{b:02X}" for b in data)


def hex_txt_to_bin(text: str) -> bytes:
    """Convert hex text (1 byte per line) back to binary."""
    lines = text.strip().splitlines()
    result = bytearray()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("0x"):
            line = line[2:]
        result.append(int(line, 16))
    return bytes(result)
