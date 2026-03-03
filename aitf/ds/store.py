"""Golden data storage layer.

Each model/version holds exactly ONE file (.pth / .bin / .zip / .tar / .tar.gz).
Re-uploading the same model/version replaces the old file.

Used by both CLI and Web API.
"""

from __future__ import annotations

import io
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

ALLOWED_EXT = (".pth", ".bin", ".zip", ".tar", ".tar.gz")


@dataclass
class GoldenEntry:
    model: str
    version: str
    file: str
    size: int


class GoldenStore:
    """Thin wrapper around the ``datastore/store/`` directory."""

    def __init__(self, base_dir: str | Path = "datastore") -> None:
        self._root = Path(base_dir) / "store"

    @property
    def root(self) -> Path:
        return self._root

    # -- validation ----------------------------------------------------------

    @staticmethod
    def allowed(filename: str) -> bool:
        return any(filename.endswith(ext) for ext in ALLOWED_EXT)

    # -- queries -------------------------------------------------------------

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
                for f in ver_dir.iterdir():
                    if f.is_file():
                        result.append(GoldenEntry(
                            model=model_dir.name,
                            version=ver_dir.name,
                            file=f.name,
                            size=f.stat().st_size,
                        ))
                        break  # one file per version
        return result

    def get_file(self, model: str, version: str) -> Path | None:
        """Return the single file path, or None if not found."""
        ver_dir = self._root / model / version
        if not ver_dir.is_dir():
            return None
        for f in ver_dir.iterdir():
            if f.is_file():
                return f
        return None

    def export_all(self) -> io.BytesIO:
        """Pack all golden files into a zip archive in memory.

        Structure inside zip: ``<model>/<version>/<file>``
        """
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for entry in self.list():
                fp = self.get_file(entry.model, entry.version)
                if fp:
                    arcname = f"{entry.model}/{entry.version}/{entry.file}"
                    zf.write(fp, arcname)
        buf.seek(0)
        return buf

    # -- mutations -----------------------------------------------------------

    def save(self, model: str, version: str, filename: str) -> Path:
        """Prepare dest dir (removing old file), return the path to save to."""
        dest = self._root / model / version
        if dest.is_dir():
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)
        return dest / filename

    def delete(self, model: str, version: str) -> bool:
        """Delete a model/version entry. Returns True if it existed."""
        d = self._root / model / version
        if not d.is_dir():
            return False
        shutil.rmtree(d)
        # Clean up empty model dir
        model_dir = d.parent
        if model_dir.is_dir() and not any(model_dir.iterdir()):
            model_dir.rmdir()
        return True
