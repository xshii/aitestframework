#!/usr/bin/env python3
"""Standalone deps installer — no CLI, just a Python script.

Lives next to the deps source files inside ``aitf/deps/``. Single source
of truth: there is no separate "standalone" copy.

------------------------------------------------------------------------
Usage (in this repo)
------------------------------------------------------------------------
    python aitf/deps/install.py                       # uses ./deps.yaml
    python aitf/deps/install.py path/to/deps.yaml

Same-``order`` steps run in parallel; groups run sequentially in
ascending ``order``. Fractional orders (e.g. ``1.5``) are supported.

------------------------------------------------------------------------
Deploying to another repository — two options, both supported
------------------------------------------------------------------------

**Option A — copy the whole directory (easiest, recommended)**

    cp -r aitf/deps /target/path/deps_pkg
    cd /target/path/deps_pkg
    python install.py path/to/deps.yaml

This carries unused files (``manager.py`` / ``bundle.py`` / ``lock.py`` /
``doctor.py`` / ``routes.py`` / ``commands.py`` / ``templates/``) along
for the ride. They are **never imported by install.py** and have zero
runtime effect — they just sit there as dead weight.

**Option B — copy only the 6 essential files (minimal, ~60 KB)**

    mkdir target_pkg
    cp aitf/deps/{install,types,config,acquire,repo,bootstrap}.py target_pkg/
    cd target_pkg
    python install.py path/to/deps.yaml

Both options work identically. Pick A for convenience, B to minimise
what you ship.

------------------------------------------------------------------------
How the absolute-imports trick works — and why it doesn't break
------------------------------------------------------------------------

Sibling files use absolute imports like ``from aitf.deps.types import X``.
Normally those would require the ``aitf`` package to be on
``sys.path``. install.py side-steps that entirely:

1. It injects empty ``aitf`` and ``aitf.deps`` stub packages into
   ``sys.modules`` (pointing ``aitf.deps.__path__`` at the script's own
   directory).
2. It then walks the 5 core sibling files in dependency order
   (``types → config → bootstrap → acquire → repo``) and loads each
   via ``importlib.util.spec_from_file_location``, registering the
   module under ``aitf.deps.<name>``.

After that, every ``from aitf.deps.X import Y`` in the code base
resolves against the pre-registered sys.modules entry — no real
``aitf`` package needed on disk or on PYTHONPATH.

------------------------------------------------------------------------
Gotchas to be aware of
------------------------------------------------------------------------

* **Default config path** — ``install.py`` with no argument reads
  ``deps.yaml`` next to **itself** (``HERE / 'deps.yaml'``), NOT the
  current working directory. Pass an explicit path if that's not what
  you want.

* **build/ is relative to install.py** — extracted toolchains land in
  ``<install.py dir>/build/cache/<name>/`` and cloned repos in
  ``<install.py dir>/build/repos/<name>/``. Override per dep with
  ``acquire.install_dir:`` / ``install_dir:`` in ``deps.yaml``.

* **System tools required at runtime** —
  ``git`` (for repo deps; **≥ 2.27** if you use ``sparse_checkout``),
  ``bash`` (for ``acquire.script``),
  ``unrar`` (only if you use ``.rar`` archives).
  ``tar.gz`` / ``tar.xz`` / ``tar.bz2`` / ``zip`` are handled by Python
  stdlib — no external tool needed.

* **Python version** — requires Python 3.12+ (``tarfile.extractall``'s
  ``filter='data'`` keyword was added in 3.12).

* **Do not run from an installed aitf** — if your target machine has
  a real ``aitf`` package already importable (e.g. pip-installed), the
  sibling-file injection will still override ``aitf.deps.*`` entries
  in ``sys.modules``, which may confuse the real package. Either use
  a clean environment, or use the real package's CLI
  (``aitf deps install``) instead of this script.

* **Mode C ``artifact_tool:`` config is baked into bootstrap.py** —
  ``ARTIFACT_TOOL_URL`` / ``ARTIFACT_TOOL_FLAGS`` / ``EXTRACTORS``
  are module-level constants. Edit them before deploying if you rely
  on that acquisition mode.

------------------------------------------------------------------------
Environment variables
------------------------------------------------------------------------
    AITF_LOG_LEVEL=INFO        # less framework chatter (default: DEBUG)
    AITF_TOOL_LOG=capture      # quiet mode (default: stream live)
    AITF_TOOL_LOG=/path/to.log # redirect tool stdout/stderr to a file
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import types as _types
from pathlib import Path

HERE = Path(__file__).resolve().parent

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

from aitf.deps.acquire import Step, install_toolchain, run_ordered  # noqa: E402
from aitf.deps.config import load_deps_config  # noqa: E402
from aitf.deps.repo import build_repo, clone_repo  # noqa: E402

PROJECT_ROOT = HERE
BUILD_DIR = PROJECT_ROOT / "build"
CACHE_DIR = BUILD_DIR / "cache"
REPOS_DIR = BUILD_DIR / "repos"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
REPOS_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("install")

cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "deps.yaml"
log.info("loading config: %s", cfg_path)
cfg = load_deps_config(cfg_path)
log.info("found %d toolchain(s), %d repo(s)",
         len(cfg.toolchains), len(cfg.repos))


def _install_tc(tc) -> None:
    install_toolchain(
        tc, cache_dir=CACHE_DIR, project_root=PROJECT_ROOT,
        install_dir=tc.install_path(CACHE_DIR, BUILD_DIR),
    )


def _clone_and_build(rc) -> None:
    d = rc.install_path(REPOS_DIR, BUILD_DIR)
    clone_repo(rc, REPOS_DIR, repo_dir=d)
    build_repo(rc, d, d, project_root=PROJECT_ROOT)


steps: list[Step] = [
    *((tc.order, f"toolchain {tc.name} {tc.version}", lambda t=tc: _install_tc(t))
      for tc in cfg.toolchains.values()),
    *((rc.order, f"repo {rc.name} @ {rc.resolved_ref}", lambda r=rc: _clone_and_build(r))
      for rc in cfg.repos.values()),
]

failed: list[tuple[str, str]] = []


def _on_progress(done: int, total: int, label: str) -> None:
    log.info("[%d/%d] %s", done, total, label)


def _on_error(label: str, exc: BaseException) -> None:
    log.error("FAILED: %s — %s", label, exc)
    failed.append((label, str(exc)))


run_ordered(steps, on_progress=_on_progress, on_error=_on_error)

if failed:
    log.error("done with %d failure(s) out of %d:", len(failed), len(steps))
    for label, msg in failed:
        log.error("  - %s: %s", label, msg)
    sys.exit(1)

log.info("done — all %d dep(s) installed", len(steps))
