"""StubManager — facade for stub code management."""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class StubInfo:
    """Metadata for a single model stub."""

    name: str
    path: Path
    sources: list[str] = field(default_factory=list)
    headers: list[str] = field(default_factory=list)
    last_modified: float = 0.0


@dataclass
class BuildResult:
    """Outcome of a cmake configure + build invocation."""

    success: bool
    output_dir: Path
    duration_s: float
    error_msg: str = ""


class StubManager:
    """Central facade for stub code operations."""

    def __init__(
        self,
        stubs_dir: str | Path | None = None,
        build_dir: str | Path | None = None,
    ) -> None:
        if stubs_dir is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            stubs_dir = project_root / "stubs"
        if build_dir is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            build_dir = project_root / "build"
        self._stubs_dir = Path(stubs_dir).resolve()
        self._build_dir = Path(build_dir).resolve()

    # -- queries -------------------------------------------------------------

    def list_stubs(self) -> list[StubInfo]:
        """Scan ``stubs/models/`` for subdirectories containing CMakeLists.txt."""
        models_dir = self._stubs_dir / "models"
        if not models_dir.is_dir():
            return []
        stubs: list[StubInfo] = []
        for child in sorted(models_dir.iterdir()):
            if not child.is_dir():
                continue
            cmake = child / "CMakeLists.txt"
            if not cmake.is_file():
                continue
            stubs.append(self._scan_stub(child))
        return stubs

    def get_stub(self, name: str) -> StubInfo:
        """Return :class:`StubInfo` for a specific model stub."""
        stub_dir = self._stubs_dir / "models" / name
        if not stub_dir.is_dir() or not (stub_dir / "CMakeLists.txt").is_file():
            raise FileNotFoundError(f"stub not found: {name}")
        return self._scan_stub(stub_dir)

    def list_platforms(self) -> list[str]:
        """Return platform names found under ``stubs/hooks/``."""
        hooks_dir = self._stubs_dir / "hooks"
        if not hooks_dir.is_dir():
            return []
        return sorted(
            d.name for d in hooks_dir.iterdir() if d.is_dir()
        )

    # -- build ---------------------------------------------------------------

    def build(
        self,
        platform: str = "func_sim",
        target: str = "all",
        toolchain: str | None = None,
        extra_cmake_args: list[str] | None = None,
    ) -> BuildResult:
        """Run cmake configure + build and return the result."""
        output_dir = self._build_dir / "stubs" / platform
        output_dir.mkdir(parents=True, exist_ok=True)

        cmake_args = [
            "cmake",
            "-S", str(self._stubs_dir),
            "-B", str(output_dir),
            f"-DPLATFORM={platform}",
        ]
        if toolchain:
            cmake_args.append(f"-DCMAKE_TOOLCHAIN_FILE={toolchain}")
        if extra_cmake_args:
            cmake_args.extend(extra_cmake_args)

        t0 = time.monotonic()
        try:
            # Configure
            subprocess.run(cmake_args, check=True, capture_output=True, text=True)
            # Build
            build_cmd = ["cmake", "--build", str(output_dir)]
            if target != "all":
                build_cmd.extend(["--target", target])
            subprocess.run(build_cmd, check=True, capture_output=True, text=True)
            duration = time.monotonic() - t0
            return BuildResult(
                success=True, output_dir=output_dir, duration_s=duration,
            )
        except subprocess.CalledProcessError as exc:
            duration = time.monotonic() - t0
            msg = (exc.stderr or exc.stdout or str(exc)).strip()
            logger.error("stub build failed: %s", msg)
            return BuildResult(
                success=False, output_dir=output_dir,
                duration_s=duration, error_msg=msg,
            )

    # -- change detection ----------------------------------------------------

    def detect_changes(self, since: str = "HEAD~1") -> list[str]:
        """Use ``git diff`` to find changed files under stubs/ and return affected model names."""
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", since],
                capture_output=True, text=True, check=True,
                cwd=self._stubs_dir.parent,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return []

        prefix = "stubs/"
        changed_models: set[str] = set()
        for line in result.stdout.strip().splitlines():
            if not line.startswith(prefix):
                continue
            # e.g. stubs/models/tdd/tdd_stub.c  → tdd
            parts = line[len(prefix):].split("/")
            if len(parts) >= 2 and parts[0] == "models":
                changed_models.add(parts[1])
        return sorted(changed_models)

    # -- scaffolding ---------------------------------------------------------

    def create_from_template(self, name: str) -> Path:
        """Create a new model stub from a minimal template."""
        stub_dir = self._stubs_dir / "models" / name
        if stub_dir.exists():
            raise FileExistsError(f"stub already exists: {name}")
        stub_dir.mkdir(parents=True)

        cmake_content = (
            f'set(MODEL_NAME     "{name}" PARENT_SCOPE)\n'
            f'set(MODEL_RUN_FN   "{name}_run" PARENT_SCOPE)\n'
            f'set(MODEL_SETUP_FN "{name}_setup" PARENT_SCOPE)\n'
        )
        (stub_dir / "CMakeLists.txt").write_text(cmake_content, encoding="utf-8")

        c_content = (
            f'#include "stub_config.h"\n'
            f'#include "stub_log.h"\n'
            f'#include "platform_api.h"\n'
            f'#include <string.h>\n'
            f'\n'
            f'int {name}_setup(const stub_config_t *cfg)\n'
            f'{{\n'
            f'    (void)cfg;\n'
            f'    return 0;\n'
            f'}}\n'
            f'\n'
            f'int {name}_run(const stub_config_t *cfg)\n'
            f'{{\n'
            f'    (void)cfg;\n'
            f'    LOG_INFO("{name}: OK");\n'
            f'    return 0;\n'
            f'}}\n'
        )
        (stub_dir / f"{name}_stub.c").write_text(c_content, encoding="utf-8")

        return stub_dir

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _scan_stub(stub_dir: Path) -> StubInfo:
        """Build a :class:`StubInfo` from a stub directory."""
        sources = [p.name for p in stub_dir.glob("*.c")]
        headers = [p.name for p in stub_dir.glob("*.h")]
        last_modified = 0.0
        for p in stub_dir.iterdir():
            if p.is_file():
                mt = p.stat().st_mtime
                if mt > last_modified:
                    last_modified = mt
        return StubInfo(
            name=stub_dir.name,
            path=stub_dir,
            sources=sorted(sources),
            headers=sorted(headers),
            last_modified=last_modified,
        )
