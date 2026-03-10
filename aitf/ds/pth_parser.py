"""Lightweight .pth file parser — no torch dependency.

PyTorch .pth files are zip archives containing pickled state dicts.
This module uses a custom pickle Unpickler with stub classes to extract
tensor metadata (key, dtype, shape) and raw binary data without importing torch.

Supports both new-format (zip) and legacy-format (raw pickle) .pth files.
"""

from __future__ import annotations

import io
import logging
import pickle
import zipfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Torch dtype → our standard precision name + element size in bytes
# ---------------------------------------------------------------------------

_TORCH_DTYPE_MAP: dict[str, tuple[str, int]] = {
    "torch.float16":  ("FP16", 2),
    "torch.float32":  ("FP32", 4),
    "torch.float64":  ("FP32", 8),   # 降精度标记
    "torch.bfloat16": ("BFP16", 2),  # Ascend 实际为 BFP16
    "torch.int8":     ("INT8", 1),
    "torch.uint8":    ("BFP8", 1),   # Ascend 实际为 BFP8
    "torch.int16":    ("INT16", 2),
    "torch.int32":    ("INT32", 4),
    "torch.int64":    ("INT32", 8),   # 降精度标记
    "torch.bool":     ("BOOL", 1),
}


def _dtype_info(dtype_str: str) -> tuple[str, int]:
    """Map torch dtype string to (precision, element_bytes)."""
    return _TORCH_DTYPE_MAP.get(dtype_str, ("FP32", 4))


# ---------------------------------------------------------------------------
# Stub classes for pickle unpickling without torch
# ---------------------------------------------------------------------------

class _FakeStorage:
    """Stub for torch.Storage subclasses."""

    def __init__(self, size: int = 0, dtype_str: str = "torch.float32",
                 data: bytes = b"", location: str = "cpu"):
        self.size = size
        self.dtype_str = dtype_str
        self.data = data
        self.location = location
        self._prec, self._elem_bytes = _dtype_info(dtype_str)

    def element_size(self):
        return self._elem_bytes


class _FakeTensor:
    """Stub for torch.Tensor — captures shape, dtype, and raw data reference."""

    def __init__(self, storage: _FakeStorage, offset: int, shape: tuple,
                 stride: tuple, requires_grad: bool = False,
                 backward_hooks=None, **kwargs):
        self.storage = storage
        self.offset = offset
        self.shape = tuple(shape) if shape else ()
        self.stride = stride
        self.dtype_str = storage.dtype_str
        self.precision = storage._prec

    @property
    def data(self) -> bytes:
        elem = self.storage._elem_bytes
        start = self.offset * elem
        nbytes = 1
        for s in self.shape:
            nbytes *= s
        nbytes *= elem
        return self.storage.data[start:start + nbytes]


# Storage type name → dtype string
_STORAGE_DTYPES = {
    "FloatStorage":   "torch.float32",
    "DoubleStorage":  "torch.float64",
    "HalfStorage":    "torch.float16",
    "BFloat16Storage": "torch.bfloat16",
    "ByteStorage":    "torch.uint8",
    "CharStorage":    "torch.int8",
    "ShortStorage":   "torch.int16",
    "IntStorage":     "torch.int32",
    "LongStorage":    "torch.int64",
    "BoolStorage":    "torch.bool",
}


class _PthUnpickler(pickle.Unpickler):
    """Custom unpickler that replaces torch classes with stubs."""

    def __init__(self, fp, zip_data: dict[str, bytes] | None = None):
        super().__init__(fp)
        self._zip_data = zip_data or {}
        self._storages: dict[str, _FakeStorage] = {}

    def find_class(self, module: str, name: str):
        # torch._utils._rebuild_tensor_v2 — the main tensor reconstruction fn
        if name == "_rebuild_tensor_v2":
            return _FakeTensor
        # torch.storage._TypedStorage / torch.storage.TypedStorage
        if "Storage" in name and ("torch" in module):
            return self._make_storage_class(name)
        # OrderedDict, dict, etc.
        if module == "collections" and name == "OrderedDict":
            from collections import OrderedDict
            return OrderedDict
        # Fallback: return a no-op class for unknown torch internals
        if module.startswith("torch"):
            return _noop_class
        return super().find_class(module, name)

    def persistent_load(self, saved_id):
        """Handle torch storage persistent_id protocol."""
        # saved_id format: ('storage', storage_type, key, location, numel)
        if isinstance(saved_id, tuple) and len(saved_id) >= 5:
            _, storage_type_name, key, location, numel = saved_id[:5]
            dtype_str = "torch.float32"
            if hasattr(storage_type_name, "__name__"):
                sname = storage_type_name.__name__
            else:
                sname = str(storage_type_name)
            for known, dt in _STORAGE_DTYPES.items():
                if known in sname:
                    dtype_str = dt
                    break

            if key not in self._storages:
                data = self._zip_data.get(key, b"")
                self._storages[key] = _FakeStorage(
                    size=numel, dtype_str=dtype_str,
                    data=data, location=location)
            return self._storages[key]

        return _FakeStorage()

    def _make_storage_class(self, name: str):
        dtype_str = _STORAGE_DTYPES.get(name, "torch.float32")

        class _TypedStorageStub(_FakeStorage):
            def __init__(self, *args, **kwargs):
                super().__init__(dtype_str=dtype_str)

            @classmethod
            def _new_with_file(cls, *args, **kwargs):
                return cls()

        _TypedStorageStub.__name__ = name
        return _TypedStorageStub


def _noop_class(*args, **kwargs):
    """Placeholder for unknown torch classes."""
    return None


# ---------------------------------------------------------------------------
# Key → operator name + category heuristics
# ---------------------------------------------------------------------------

# 参数类型后缀 → category 映射
_CATEGORY_PATTERNS: list[tuple[str, str]] = [
    ("weight", "weight"),
    ("bias", "bias"),
    ("mask", "mask"),
    ("index", "index"),
    ("embed", "weight"),
    ("norm", "weight"),
    ("running_mean", "other"),
    ("running_var", "other"),
    ("num_batches_tracked", "other"),
]


def _parse_tensor_key(key: str) -> tuple[str, str, int]:
    """从 tensor key 推断算子名和类别。

    e.g. "model.layers.0.self_attn.q_proj.weight"
         → operator="self_attn_q_proj", category="weight", seq=0

    Returns:
        (operator_name, category, seq_hint)
    """
    parts = key.split(".")

    # 末尾匹配 category
    category = "other"
    param_part = parts[-1] if parts else ""
    for suffix, cat in _CATEGORY_PATTERNS:
        if param_part == suffix or param_part.endswith("_" + suffix):
            category = cat
            break

    # 提取算子名: 去掉常见包装前缀、层号、和末尾参数类型
    _SKIP_PREFIXES = {"model", "module", "state_dict"}
    op_parts = []
    seq_hint = 0
    for p in parts[:-1]:  # 排除末尾的 weight/bias 等
        if p in _SKIP_PREFIXES:
            continue
        if p.isdigit():
            seq_hint = int(p)
            continue
        op_parts.append(p)

    operator = "_".join(op_parts) if op_parts else key.replace(".", "_")
    return operator, category, seq_hint


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def inspect_pth(fileobj) -> list[dict]:
    """Parse a .pth file and return metadata for each tensor.

    Args:
        fileobj: file-like object (BytesIO or opened file) of the .pth file.

    Returns:
        List of dicts with key, dtype, shape, numel, bytes,
        plus inferred operator/category/seq.
    """
    content = fileobj.read()
    logger.info("pth inspect: file size %.1f MB", len(content) / 1048576)
    tensors = _load_state_dict(content)

    result = []
    for key, tensor in tensors.items():
        numel = 1
        for s in tensor.shape:
            numel *= s
        elem_bytes = tensor.storage._elem_bytes
        operator, category, seq = _parse_tensor_key(key)
        result.append({
            "key": key,
            "dtype": tensor.precision,
            "torch_dtype": tensor.dtype_str,
            "shape": "x".join(str(s) for s in tensor.shape),
            "numel": numel,
            "bytes": numel * elem_bytes,
            "operator": operator,
            "category": category,
            "seq": seq,
        })

    logger.info("pth inspect: found %d tensors", len(result))
    return result


def export_tensor(fileobj, key: str) -> tuple[bytes, dict]:
    """Export a single tensor from .pth as raw bytes.

    Args:
        fileobj: file-like object of the .pth file.
        key: tensor key name (e.g. "model.layers.0.weight").

    Returns:
        (raw_bytes, info_dict) where info_dict has dtype/shape metadata.

    Raises:
        KeyError: if key not found in the file.
    """
    content = fileobj.read()
    tensors = _load_state_dict(content)

    if key not in tensors:
        raise KeyError(f"tensor '{key}' not found, available: {list(tensors.keys())[:10]}")

    tensor = tensors[key]
    raw = tensor.data
    numel = 1
    for s in tensor.shape:
        numel *= s

    operator, category, seq = _parse_tensor_key(key)
    info = {
        "key": key,
        "dtype": tensor.precision,
        "torch_dtype": tensor.dtype_str,
        "shape": "x".join(str(s) for s in tensor.shape),
        "numel": numel,
        "operator": operator,
        "category": category,
        "seq": seq,
    }
    logger.info("pth export: key=%s dtype=%s shape=%s bytes=%d",
                key, tensor.precision, info["shape"], len(raw))
    return raw, info


def _load_state_dict(content: bytes) -> dict[str, _FakeTensor]:
    """Load state dict from .pth bytes, returns {key: _FakeTensor}."""
    # Try new format (zip)
    if content[:2] == b"PK":
        return _load_zip_format(content)
    # Legacy format (raw pickle)
    return _load_legacy_format(content)


def _load_zip_format(content: bytes) -> dict[str, _FakeTensor]:
    """Parse new-format .pth (zip archive)."""
    zf = zipfile.ZipFile(io.BytesIO(content))

    # Find the data.pkl entry
    pkl_name = None
    for name in zf.namelist():
        if name.endswith("/data.pkl") or name == "data.pkl":
            pkl_name = name
            break
    if not pkl_name:
        raise ValueError("pth 文件中未找到 data.pkl，格式不支持")

    # Read all data/* files (raw tensor storage)
    prefix = pkl_name.rsplit("data.pkl", 1)[0] + "data/"
    zip_data: dict[str, bytes] = {}
    for name in zf.namelist():
        if name.startswith(prefix) and name != prefix:
            storage_key = name[len(prefix):]
            zip_data[storage_key] = zf.read(name)

    pkl_bytes = zf.read(pkl_name)
    unpickler = _PthUnpickler(io.BytesIO(pkl_bytes), zip_data=zip_data)
    obj = unpickler.load()

    return _extract_tensors(obj)


def _load_legacy_format(content: bytes) -> dict[str, _FakeTensor]:
    """Parse legacy-format .pth (raw pickle with appended storage)."""
    unpickler = _PthUnpickler(io.BytesIO(content))
    obj = unpickler.load()
    return _extract_tensors(obj)


def _extract_tensors(obj, prefix: str = "") -> dict[str, _FakeTensor]:
    """Recursively extract _FakeTensor objects from a nested dict/OrderedDict."""
    result = {}
    if isinstance(obj, _FakeTensor):
        result[prefix] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            full_key = f"{prefix}.{k}" if prefix else k
            result.update(_extract_tensors(v, full_key))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            full_key = f"{prefix}.{i}" if prefix else str(i)
            result.update(_extract_tensors(v, full_key))
    return result
