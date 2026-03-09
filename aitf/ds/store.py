"""Golden data storage layer — four-level management.

Hierarchy: Model → Version → Operator → Data Items
Each operator has multiple data items (input/output/weight/bias...),
each with its own shape, layout, and precision.

Files are stored under ``datastore/store/<model>/<version>/<operator>/``.
"""

from __future__ import annotations

import io
import json
import shutil
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ALLOWED_EXT = (".txt", ".pth", ".bin", ".zip", ".tar", ".tar.gz")

META_FILE = "golden_meta.json"

CATEGORY_CHOICES = ("input", "output", "weight", "bias", "mask", "index", "other")
PRECISION_CHOICES = ("fp16", "fp32", "bf16", "int8", "int4", "uint8", "int16", "int32")
LAYOUT_CHOICES = ("NCHW", "NHWC", "NC1HWC0", "ND", "FRACTAL_NZ", "FRACTAL_Z")


@dataclass
class DataItem:
    """One data category entry within an operator."""

    seq: int = 0
    category: str = "input"
    loop: int = 1
    m: int = 64
    n: int = 64
    k: int = 64
    layout: str = "NCHW"
    precision: str = "fp16"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DataItem:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class OperatorMeta:
    """Metadata for an operator: name + list of data items."""

    name: str = ""
    data: list[DataItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "data": [d.to_dict() for d in self.data]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OperatorMeta:
        name = d.get("name", "")
        items = [DataItem.from_dict(i) for i in d.get("data", [])]
        return cls(name=name, data=items)


@dataclass
class GoldenEntry:
    """A single operator entry with data items and files."""

    model: str
    version: str
    operator: str
    meta: OperatorMeta = field(default_factory=OperatorMeta)
    files: list[str] = field(default_factory=list)
    total_size: int = 0


def parse_golden_filename(filename: str) -> dict | None:
    """Parse golden file name into metadata.

    Format: ``{op}_{category}{seq}_{layout}_{precision}_{shape}.ext``
    Example: ``matmul_input0_NCHW_fp16_128x96.bin``

    Shape rules:
    - 4 numbers ``LxMxNxK``: loop, m, n, k
    - 3 numbers ``MxNxK``: loop=1
    - 2 numbers ``NxK``: loop=1, m=1

    Returns dict with keys: operator, category, seq, layout, precision,
    loop, m, n, k.  Returns None if parsing fails.
    """
    # Strip extension
    base = filename
    for ext in (".tar.gz",) + ALLOWED_EXT:
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    else:
        return None

    parts = base.split("_")
    if len(parts) < 5:
        return None

    # Last part = shape
    shape_str = parts[-1]
    dims = shape_str.split("x")
    try:
        dims = [int(d) for d in dims]
    except ValueError:
        return None

    if len(dims) == 4:
        loop, m_val, n_val, k_val = dims
    elif len(dims) == 3:
        loop, m_val, n_val, k_val = 1, dims[0], dims[1], dims[2]
    elif len(dims) == 2:
        loop, m_val, n_val, k_val = 1, 1, dims[0], dims[1]
    else:
        return None

    # precision = parts[-2], layout = parts[-3]
    precision = parts[-2]
    layout = parts[-3]

    # category+seq = parts[-4]
    cat_seq = parts[-4]
    # Split trailing digits as seq
    i = len(cat_seq)
    while i > 0 and cat_seq[i - 1].isdigit():
        i -= 1
    if i == 0 or i == len(cat_seq):
        return None
    category = cat_seq[:i]
    seq = int(cat_seq[i:])

    # operator = everything before (joined with _)
    operator = "_".join(parts[: -4])
    if not operator:
        return None

    return {
        "operator": operator,
        "category": category,
        "seq": seq,
        "layout": layout,
        "precision": precision,
        "loop": loop,
        "m": m_val,
        "n": n_val,
        "k": k_val,
    }


def build_golden_filename(operator: str, item: DataItem, ext: str = ".bin") -> str:
    """Build a golden filename from operator name and data item metadata.

    Format: ``{op}_{category}{seq}_{layout}_{precision}_{shape}.ext``
    """
    if item.loop == 1 and item.m == 1:
        shape = f"{item.n}x{item.k}"
    elif item.loop == 1:
        shape = f"{item.m}x{item.n}x{item.k}"
    else:
        shape = f"{item.loop}x{item.m}x{item.n}x{item.k}"
    return f"{operator}_{item.category}{item.seq}_{item.layout}_{item.precision}_{shape}{ext}"


class GoldenStore:
    """Manages ``datastore/store/`` with model/version/operator hierarchy."""

    def __init__(self, base_dir: str | Path = "datastore") -> None:
        self._root = Path(base_dir) / "store"

    @property
    def root(self) -> Path:
        return self._root

    # -- validation ----------------------------------------------------------

    @staticmethod
    def allowed(filename: str) -> bool:
        return any(filename.endswith(ext) for ext in ALLOWED_EXT)

    @staticmethod
    def parse_package_name(filename: str) -> tuple[str, str] | None:
        """Parse ``{model}_{version}.ext`` → (model, version), or None."""
        base = filename
        for ext in (".tar.gz",) + ALLOWED_EXT:
            if base.endswith(ext):
                base = base[: -len(ext)]
                break
        else:
            return None
        parts = base.split("_", 1)
        if len(parts) == 2 and parts[0] and parts[1]:
            return parts[0], parts[1]
        return None

    # -- metadata I/O --------------------------------------------------------

    def _meta_path(self, model: str, version: str, operator: str) -> Path:
        return self._root / model / version / operator / META_FILE

    def save_meta(self, model: str, version: str, operator: str, meta: OperatorMeta) -> None:
        p = self._meta_path(model, version, operator)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(meta.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def load_meta(self, model: str, version: str, operator: str) -> OperatorMeta:
        p = self._meta_path(model, version, operator)
        if p.is_file():
            return OperatorMeta.from_dict(json.loads(p.read_text(encoding="utf-8")))
        return OperatorMeta(name=operator)

    # -- queries -------------------------------------------------------------

    def list_models(self) -> list[str]:
        if not self._root.is_dir():
            return []
        return sorted(d.name for d in self._root.iterdir() if d.is_dir())

    def list_versions(self, model: str) -> list[str]:
        d = self._root / model
        if not d.is_dir():
            return []
        return sorted(v.name for v in d.iterdir() if v.is_dir())

    def list_operators(self, model: str, version: str) -> list[str]:
        d = self._root / model / version
        if not d.is_dir():
            return []
        return sorted(o.name for o in d.iterdir() if o.is_dir())

    def list(self, model: str | None = None) -> list[GoldenEntry]:
        if not self._root.is_dir():
            return []
        result: list[GoldenEntry] = []
        for model_dir in sorted(self._root.iterdir()):
            if not model_dir.is_dir():
                continue
            if model and model_dir.name != model:
                continue
            for ver_dir in sorted(model_dir.iterdir()):
                if not ver_dir.is_dir():
                    continue
                for op_dir in sorted(ver_dir.iterdir()):
                    if not op_dir.is_dir():
                        continue
                    meta = self.load_meta(model_dir.name, ver_dir.name, op_dir.name)
                    files = []
                    total = 0
                    for f in sorted(op_dir.iterdir()):
                        if f.is_file() and f.name != META_FILE:
                            files.append(f.name)
                            total += f.stat().st_size
                    result.append(GoldenEntry(
                        model=model_dir.name,
                        version=ver_dir.name,
                        operator=op_dir.name,
                        meta=meta,
                        files=files,
                        total_size=total,
                    ))
        return result

    def get_file(self, model: str, version: str, operator: str, filename: str) -> Path | None:
        fp = self._root / model / version / operator / filename
        if fp.is_file() and fp.name != META_FILE:
            return fp
        return None

    def get_operator_files(self, model: str, version: str, operator: str) -> list[Path]:
        d = self._root / model / version / operator
        if not d.is_dir():
            return []
        return sorted(f for f in d.iterdir() if f.is_file() and f.name != META_FILE)

    def export_version(self, model: str, version: str) -> io.BytesIO:
        """Export all operators under a specific model/version as a zip."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            ver_dir = self._root / model / version
            if ver_dir.is_dir():
                for op_dir in sorted(ver_dir.iterdir()):
                    if not op_dir.is_dir():
                        continue
                    meta_path = op_dir / META_FILE
                    if meta_path.is_file():
                        arcname = f"{model}/{version}/{op_dir.name}/{META_FILE}"
                        zf.write(meta_path, arcname)
                    for f in sorted(op_dir.iterdir()):
                        if f.is_file() and f.name != META_FILE:
                            arcname = f"{model}/{version}/{op_dir.name}/{f.name}"
                            zf.write(f, arcname)
        buf.seek(0)
        return buf

    def export_all(self) -> io.BytesIO:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for entry in self.list():
                op_dir = self._root / entry.model / entry.version / entry.operator
                meta_path = op_dir / META_FILE
                if meta_path.is_file():
                    arcname = f"{entry.model}/{entry.version}/{entry.operator}/{META_FILE}"
                    zf.write(meta_path, arcname)
                for fname in entry.files:
                    fp = op_dir / fname
                    if fp.is_file():
                        arcname = f"{entry.model}/{entry.version}/{entry.operator}/{fname}"
                        zf.write(fp, arcname)
        buf.seek(0)
        return buf

    def import_archive(self, buf: io.BytesIO) -> list[str]:
        imported = []
        with zipfile.ZipFile(buf, "r") as zf:
            for name in zf.namelist():
                parts = name.rstrip("/").split("/")
                if len(parts) < 3:
                    continue
                dest = self._root / name
                if name.endswith("/"):
                    dest.mkdir(parents=True, exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(zf.read(name))
                    imported.append(name)
        return imported

    # -- mutations -----------------------------------------------------------

    def save_file(self, model: str, version: str, operator: str, filename: str) -> Path:
        dest = self._root / model / version / operator
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / filename
        if target.exists():
            target.unlink()
        return target

    def delete_operator(self, model: str, version: str, operator: str) -> bool:
        d = self._root / model / version / operator
        if not d.is_dir():
            return False
        shutil.rmtree(d)
        self._cleanup_empty_parents(d)
        return True

    def delete_version(self, model: str, version: str) -> bool:
        d = self._root / model / version
        if not d.is_dir():
            return False
        shutil.rmtree(d)
        self._cleanup_empty_parents(d)
        return True

    def delete_model(self, model: str) -> bool:
        d = self._root / model
        if not d.is_dir():
            return False
        shutil.rmtree(d)
        return True

    def delete_file(self, model: str, version: str, operator: str, filename: str) -> bool:
        fp = self._root / model / version / operator / filename
        if not fp.is_file() or fp.name == META_FILE:
            return False
        fp.unlink()
        return True

    def _cleanup_empty_parents(self, path: Path) -> None:
        for parent in [path.parent, path.parent.parent]:
            if parent == self._root or not parent.is_dir():
                break
            if not any(parent.iterdir()):
                parent.rmdir()
