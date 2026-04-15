#!/usr/bin/env python3
"""Standalone deps installer — no CLI, just a Python script.

Lives next to the deps source files inside ``aitf/deps/``. Single source
of truth: there is no separate "standalone" copy.

Usage (in this repo)::

    python aitf/deps/install.py                       # uses ./deps.yaml
    python aitf/deps/install.py path/to/deps.yaml

Deploying to another repository::

    # Whole directory (carries unused files like routes.py / manager.py too)
    cp -r aitf/deps /target/path/deps_pkg
    cd /target/path/deps_pkg && python install.py path/to/deps.yaml

    # Or just the 6 essential files (minimal bundle)
    mkdir target_pkg
    cp aitf/deps/{install,types,config,acquire,repo,bootstrap}.py target_pkg/
    cd target_pkg && python install.py path/to/deps.yaml

Environment variables::

    AITF_LOG_LEVEL=INFO        # less framework chatter (default: DEBUG)
    AITF_TOOL_LOG=capture      # quiet mode (default: stream live)
    AITF_TOOL_LOG=/path/to.log # redirect tool stdout/stderr to a file

How it works: register sibling files into ``sys.modules`` under the
``aitf.deps`` namespace via ``importlib`` injection. Internal absolute
imports (``from aitf.deps.X import Y``) inside the loaded files resolve
through the pre-registered stubs. Works without ``aitf`` being on
sys.path and without executing ``aitf/__init__.py``.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import types as _types
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Default to DEBUG so cache hits, skipped flags, etc. are visible.
# Override with $AITF_LOG_LEVEL (e.g. INFO / WARNING) for less noise.
logging.basicConfig(
    level=os.environ.get("AITF_LOG_LEVEL", "DEBUG").upper(),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# ---- bootstrap sibling files as the aitf.deps package ---------------------
for _pkg in ("aitf", "aitf.deps"):
    if _pkg not in sys.modules:
        _m = _types.ModuleType(_pkg)
        _m.__path__ = []
        sys.modules[_pkg] = _m
sys.modules["aitf.deps"].__path__ = [str(HERE)]

for _short in ("types", "config", "bootstrap", "acquire", "repo"):
    _spec = importlib.util.spec_from_file_location(
        f"aitf.deps.{_short}", HERE / f"{_short}.py"
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[f"aitf.deps.{_short}"] = _mod
    _spec.loader.exec_module(_mod)

# ---- now the sibling files are usable like a regular package -------------
from aitf.deps.acquire import install_library, install_toolchain  # noqa: E402
from aitf.deps.config import load_deps_config  # noqa: E402
from aitf.deps.repo import build_repo, clone_repo  # noqa: E402

PROJECT_ROOT = HERE
CACHE_DIR = PROJECT_ROOT / "build" / "cache"
REPOS_DIR = PROJECT_ROOT / "build" / "repos"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
REPOS_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("install")

cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "deps.yaml"
log.info("loading config: %s", cfg_path)
cfg = load_deps_config(cfg_path)
log.info("found %d toolchain(s), %d library(s), %d repo(s)",
         len(cfg.toolchains), len(cfg.libraries), len(cfg.repos))

total = len(cfg.toolchains) + len(cfg.libraries) + len(cfg.repos)
done = 0
failed: list[tuple[str, str]] = []   # (label, error message) — best-effort install


def _attempt(label: str, fn) -> None:
    """Run *fn*; on failure, log the error and record it without re-raising."""
    try:
        fn()
    except Exception as exc:
        log.error("FAILED: %s — %s", label, exc)
        failed.append((label, str(exc)))


for tc in sorted(cfg.toolchains.values(), key=lambda t: t.order):
    done += 1
    label = f"toolchain {tc.name} {tc.version}"
    log.info("[%d/%d] %s", done, total, label)
    _attempt(label, lambda t=tc: install_toolchain(
        t, cache_dir=CACHE_DIR, project_root=PROJECT_ROOT, install_dir=CACHE_DIR / t.name))

for lib in sorted(cfg.libraries.values(), key=lambda x: x.order):
    done += 1
    label = f"library {lib.name} {lib.version}"
    log.info("[%d/%d] %s", done, total, label)
    _attempt(label, lambda x=lib: install_library(
        x, cache_dir=CACHE_DIR, project_root=PROJECT_ROOT, install_dir=CACHE_DIR / x.name))


def _clone_and_build(rc):
    d = REPOS_DIR / rc.name
    clone_repo(rc, REPOS_DIR, repo_dir=d)
    build_repo(rc, d, d, project_root=PROJECT_ROOT)


for rc in sorted(cfg.repos.values(), key=lambda r: r.order):
    done += 1
    label = f"repo {rc.name} @ {rc.ref}"
    log.info("[%d/%d] %s", done, total, label)
    _attempt(label, lambda r=rc: _clone_and_build(r))

if failed:
    log.error("done with %d failure(s) out of %d:", len(failed), total)
    for label, msg in failed:
        log.error("  - %s: %s", label, msg)
    sys.exit(1)

log.info("done — all %d dep(s) installed", total)
