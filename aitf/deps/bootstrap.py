"""External artifact acquisition: bootstraps a downloader tool, then
optionally extracts archives using system tools.

The downloader binary is wgetted-once and cached under ``~/.cache/aitf/``
(override with ``$AITF_TOOL_CACHE``). Its command-line interface is
described by a template at the top of this file — edit the constants in
the configuration block to match your real tool.

Extraction uses standard system tools (``tar`` / ``unzip`` / ``unrar``).
They must already be on ``$PATH``.

This module is only invoked when a toolchain's ``acquire:`` block contains
an ``artifact_tool:`` entry. Libraries and repos do not use it.
"""

from __future__ import annotations

import logging
import os
import shutil
import stat
import subprocess
from collections import defaultdict
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from aitf.deps.types import AcquireError

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION — edit these to match your real artifact downloader tool
# =============================================================================

#: Where to cache the downloaded tool binary. Override with $AITF_TOOL_CACHE.
CACHE_DIR = Path(os.environ.get("AITF_TOOL_CACHE", str(Path.home() / ".cache" / "aitf")))

#: HTTP URL of the artifact downloader binary. TODO: fill in real value.
ARTIFACT_TOOL_URL = "https://internal.example.com/aitf/artifact_tool"

#: Filename used inside CACHE_DIR.
ARTIFACT_TOOL_NAME = "artifact_tool"

#: CLI flag → value template. Emitted in dict-insertion order.
#:
#: **Available placeholders**:
#:
#: - ``{version}`` — auto-injected from ``toolchains.<name>.version``.
#: - ``{install_dir}`` — auto-injected; computed by the framework at runtime.
#: - **Anything else** — taken from yaml's free-form fields under
#:   ``acquire.artifact_tool:``. e.g. yaml ``repo_dir: "ddk/v2026"``
#:   makes ``{repo_dir}`` available with no Python changes.
#:
#: **Three special forms**:
#:
#: - **Fixed value** — value with no ``{...}`` placeholder is passed verbatim
#:   (e.g. ``"--target": "linux-x86_64"``).
#: - **Value-less flag** — value of ``None`` emits just the flag, no value
#:   (e.g. ``"--quiet": None``).
#: - **Skip-if-empty** — if a placeholder resolves to an empty string, the
#:   entire flag is omitted from the command line. Logged at DEBUG level so
#:   typos in placeholder names don't fail completely silently.
#:
#: Example real invocation built from the defaults below::
#:
#:     artifact_tool --version 10.3 --repo-dir toolchains/gcc --install-dir /opt/gcc
ARTIFACT_TOOL_FLAGS: dict[str, str | None] = {
    "--version":     "{version}",
    "--repo-dir":    "{repo_dir}",
    "--install-dir": "{install_dir}",
    # "--target":    "linux-x86_64",   # fixed value example
    # "--quiet":     None,              # value-less flag example
}

#: Where to send the artifact tool's (and extractor commands') stdout/stderr.
#: Override per-run via the ``$AITF_TOOL_LOG`` environment variable.
#:
#:   "stream"   → stream live to the script's own stdout/stderr (default)
#:   "capture"  → capture; only print on failure (quiet mode)
#:   Path(...)  → append to a log file (parent dirs auto-created)
#:
#: Anything other than the two strings above is treated as a file path.
TOOL_LOG_OUTPUT: "str | Path" = os.environ.get("AITF_TOOL_LOG") or "stream"

#: Archive-extension → extractor command. Each entry is invoked with
#: ``cwd = install_dir`` and the archive's basename appended at the end,
#: so extraction lands in *install_dir* regardless of which tool is used.
#: Edit / extend this list as your archive types require.
EXTRACTORS: list[tuple[tuple[str, ...], list[str]]] = [
    ((".tar.gz", ".tgz"), ["tar", "-xzf"]),
    ((".tar.xz",),        ["tar", "-xJf"]),
    ((".tar.bz2",),       ["tar", "-xjf"]),
    ((".tar",),           ["tar", "-xf"]),
    ((".zip",),           ["unzip", "-q", "-o"]),
    ((".rar",),           ["unrar", "x", "-y", "-inul"]),
]

# =============================================================================
# Implementation — generally no need to edit below
# =============================================================================


def _ensure_tool() -> Path:
    """Return the cached artifact tool path, downloading on first use."""
    dest = CACHE_DIR / ARTIFACT_TOOL_NAME
    if dest.is_file() and os.access(dest, os.X_OK):
        logger.debug("artifact tool already cached at %s", dest)
        return dest
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    logger.info("downloading artifact tool: %s -> %s", ARTIFACT_TOOL_URL, dest)
    try:
        with urlopen(ARTIFACT_TOOL_URL, timeout=300) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
    except URLError as exc:
        tmp.unlink(missing_ok=True)
        raise AcquireError(f"Failed to download {ARTIFACT_TOOL_URL}: {exc}") from exc
    tmp.replace(dest)
    dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return dest


def _run_capturing(cmd: list[str], cwd: str | None) -> tuple[int, str]:
    """Capture stdout+stderr; return ``(returncode, detail-on-failure)``."""
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return r.returncode, r.stderr.strip() if r.returncode else ""


def _run_streaming(cmd: list[str], cwd: str | None) -> tuple[int, str]:
    """Inherit parent stdio so output appears live."""
    r = subprocess.run(cmd, cwd=cwd)
    return r.returncode, "(see output above)" if r.returncode else ""


def _run_to_file(cmd: list[str], cwd: str | None, log_path: Path) -> tuple[int, str]:
    """Append both stdout and stderr to *log_path*."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as logf:
        logf.write(f"\n=== {' '.join(cmd)} ===\n".encode())
        logf.flush()
        r = subprocess.run(cmd, cwd=cwd, stdout=logf, stderr=subprocess.STDOUT)
    return r.returncode, f"(see {log_path})" if r.returncode else ""


def _run(cmd: list[str], *, cwd: str | None = None) -> None:
    """Run *cmd*, routing its stdout/stderr per :data:`TOOL_LOG_OUTPUT`.

    Three IO modes share one error-raise path: each ``_run_*`` helper
    returns ``(returncode, failure-detail-string)`` and we raise once.
    """
    if TOOL_LOG_OUTPUT == "stream":
        mode, runner = "stream", lambda: _run_streaming(cmd, cwd)
    elif TOOL_LOG_OUTPUT == "capture":
        mode, runner = "capture", lambda: _run_capturing(cmd, cwd)
    else:
        log_path = Path(TOOL_LOG_OUTPUT)
        mode, runner = f"file:{log_path}", lambda: _run_to_file(cmd, cwd, log_path)

    logger.info("[%s] %s", mode, " ".join(cmd))
    rc, detail = runner()
    if rc != 0:
        msg = f"Command failed (exit {rc}): {' '.join(cmd)}"
        if detail:
            msg += f"\n{detail}"
        raise AcquireError(msg)


def _extract_all(install_dir: Path) -> None:
    """Extract every top-level archive in *install_dir* in place using
    the system tools listed in :data:`EXTRACTORS`.
    """
    archives = [
        p for p in sorted(install_dir.iterdir())
        if p.is_file() and any(p.name.lower().endswith(e)
                                for exts, _ in EXTRACTORS for e in exts)
    ]
    if not archives:
        logger.info("no archives to extract in %s", install_dir)
        return
    logger.info("extracting %d archive(s) in %s", len(archives), install_dir)
    for archive in archives:
        name = archive.name.lower()
        for exts, cmd in EXTRACTORS:
            if any(name.endswith(e) for e in exts):
                logger.info("  -> %s (via %s)", archive.name, cmd[0])
                _run([*cmd, archive.name], cwd=str(install_dir))
                break


def _build_tool_cmd(tool: Path, lookup: "defaultdict[str, str]") -> list[str]:
    """Build the artifact tool command line by walking ARTIFACT_TOOL_FLAGS."""
    cmd: list[str] = [str(tool)]
    for flag, tpl in ARTIFACT_TOOL_FLAGS.items():
        if tpl is None:
            cmd.append(flag)            # value-less flag (--quiet style)
            continue
        value = tpl.format_map(lookup)
        if not value:
            logger.debug("skipping flag %s (template %r resolved empty)", flag, tpl)
            continue                     # placeholder resolved empty → skip flag
        cmd.append(flag)
        cmd.append(value)
    return cmd


def fetch_via_tool(*, version: str, install_dir: Path, extract: bool,
                   placeholders: dict[str, str] | None = None) -> None:
    """Download a toolchain via the artifact tool, then optionally extract.

    *version* and *install_dir* are auto-injected by the framework and
    available as ``{version}`` / ``{install_dir}`` in :data:`ARTIFACT_TOOL_FLAGS`.
    *placeholders* carries every other yaml field from
    ``acquire.artifact_tool:`` and merges into the substitution namespace.
    Missing placeholders resolve to an empty string and (combined with
    skip-if-empty) cause the flag to be omitted entirely.
    """
    install_dir.mkdir(parents=True, exist_ok=True)

    # Framework-injected keys go LAST so they always win, even if the
    # user accidentally put a "version" or "install_dir" key under
    # acquire.artifact_tool: in yaml.
    values: dict[str, str] = {
        **(placeholders or {}),
        "version": version,
        "install_dir": str(install_dir),
    }
    lookup: defaultdict[str, str] = defaultdict(str, values)

    logger.info(
        "fetching version=%s into %s (extract=%s, %d placeholder(s))",
        version, install_dir, extract, len(placeholders or {}),
    )
    _run(_build_tool_cmd(_ensure_tool(), lookup))

    if extract:
        _extract_all(install_dir)
