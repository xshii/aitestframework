"""Test plan parsing and matrix expansion.

A testplan.yaml defines what to test and with which parameters:

.. code-block:: yaml

    name: NPU算子全量验证
    plans:
      - name: conv2d多版本验证
        tests: cases/npu/operators/test_conv2d.py
        matrix:
          bundle: [npu-v2.0, npu-v2.1]
          golden:
            - model: npu_model
              version: v2.0
            - model: npu_model
              version: v3.0
        params:
          rtol: 1e-5

      - name: matmul快速验证
        tests: cases/npu/operators/test_matmul.py
        bundle: npu-v2.1
        golden:
          model: npu_model
          version: v3.0

      - name: 全算子冒烟
        tests: cases/npu/
        bundle: npu-v2.1
        golden:
          model: npu_model
          version: v3.0
        repeat: 3
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from aitf.tc.models import SAFE_FILENAME_RE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plan file I/O helpers (shared by tc/routes and sync/routes)
# ---------------------------------------------------------------------------

def list_plan_files(root: str | Path) -> list[dict]:
    """Scan *root* for testplan YAML files. Returns [{filename, name}, ...]."""
    root = Path(root)
    plans: list[dict] = []
    for pattern in ("testplan*.yaml", "testplan*.yml"):
        for f in sorted(root.glob(pattern)):
            try:
                with open(f, encoding="utf-8") as fh:
                    data = yaml.safe_load(fh) or {}
                plan_name = data.get("name", f.stem)
            except (yaml.YAMLError, OSError):
                plan_name = f.stem
            plans.append({"filename": f.name, "name": plan_name})
    return plans


def save_plan_yaml(
    root: str | Path, filename: str, plan_data,
) -> tuple[str | None, str | None]:
    """Validate filename and save plan_data as YAML.

    Returns ``(saved_filename, None)`` on success or ``(None, error)`` on failure.
    """
    if not filename or not plan_data:
        return None, "filename and plan are required"
    if not filename.endswith((".yaml", ".yml")):
        filename += ".yaml"
    stem = Path(filename).stem
    if not SAFE_FILENAME_RE.match(stem):
        return None, "文件名只能包含字母、数字、下划线、中文"
    filename = stem + ".yaml"
    out = Path(root) / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        yaml.dump(plan_data, fh, allow_unicode=True, default_flow_style=False,
                  sort_keys=False)
    return filename, None


def safe_plan_path(root: str | Path, filename: str) -> Path | None:
    """Resolve plan path safely, preventing path traversal.

    Returns the resolved Path if safe, or None.
    """
    root = Path(root).resolve()
    plan_path = (root / filename).resolve()
    if not plan_path.is_relative_to(root):
        return None
    if not filename.endswith((".yaml", ".yml")):
        return None
    return plan_path


@dataclass
class RunConfig:
    """A single concrete execution configuration (one cell of the matrix)."""

    name: str = ""
    tests: list[str] = field(default_factory=list)   # paths relative to cases/
    filter_k: str | None = None
    bundle: str | None = None
    target: str | None = None
    golden_model: str | None = None
    golden_version: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    repeat: int = 1
    test_timeout: int = 300         # per-test timeout in seconds (0=no timeout)
    retry: int = 0                  # retry failed tests N times before final fail


@dataclass
class TestPlan:
    """Parsed testplan with all matrix combinations expanded."""

    name: str = ""
    configs: list[RunConfig] = field(default_factory=list)
    pipeline: list[str] = field(default_factory=list)


def load_testplan(path: str | Path) -> TestPlan:
    """Load and expand a testplan.yaml into concrete RunConfigs."""
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    plan = TestPlan(name=data.get("name", path.stem))

    # Pipeline: ordered list of platform stages
    pipeline = data.get("pipeline", [])
    if isinstance(pipeline, list):
        plan.pipeline = [str(p) for p in pipeline]

    for item in data.get("plans", []):
        configs = _expand_plan_item(item)
        plan.configs.extend(configs)

    logger.info("testplan %s: %d configs expanded from %s (pipeline=%s)",
                plan.name, len(plan.configs), path,
                plan.pipeline or "none")
    return plan


def _expand_plan_item(item: dict) -> list[RunConfig]:
    """Expand one plan item (possibly with matrix) into RunConfigs."""
    name = item.get("name", "")
    tests = item.get("tests", [])
    if isinstance(tests, str):
        tests = [tests]
    filter_k = item.get("filter_k")
    target = item.get("target")
    params = item.get("params", {})
    repeat = int(item.get("repeat", 1))

    matrix = item.get("matrix")
    if matrix:
        return _expand_matrix(name, tests, filter_k, target, params, repeat, matrix)

    # No matrix — single config (or golden.versions shorthand)
    bundle = item.get("bundle")
    golden = item.get("golden", {})
    golden_model = golden.get("model") if isinstance(golden, dict) else None
    golden_version = golden.get("version") if isinstance(golden, dict) else None

    # golden.versions shorthand → expand into multiple configs
    golden_versions = golden.get("versions") if isinstance(golden, dict) else None
    if golden_versions and isinstance(golden_versions, list):
        configs = []
        for ver in golden_versions:
            label = f"{name} | {golden_model}/{ver}" if golden_model else name
            for _ in range(repeat):
                configs.append(RunConfig(
                    name=label, tests=tests, filter_k=filter_k,
                    bundle=bundle, target=target,
                    golden_model=golden_model, golden_version=ver,
                    params=params,
                ))
        return configs

    configs = []
    for _ in range(repeat):
        configs.append(RunConfig(
            name=name, tests=tests, filter_k=filter_k,
            bundle=bundle, target=target,
            golden_model=golden_model, golden_version=golden_version,
            params=params,
        ))
    return configs


def _expand_matrix(
    name: str, tests: list[str], filter_k: str | None,
    target: str | None, params: dict, repeat: int, matrix: dict,
) -> list[RunConfig]:
    """Expand matrix combinations into RunConfigs."""
    # Collect axes
    bundles = matrix.get("bundle", [None])
    if not isinstance(bundles, list):
        bundles = [bundles]

    goldens = matrix.get("golden", [{}])
    if not isinstance(goldens, list):
        goldens = [goldens]

    targets = matrix.get("target", [target])
    if not isinstance(targets, list):
        targets = [targets]

    configs = []
    for bundle, golden, tgt in itertools.product(bundles, goldens, targets):
        golden_model = golden.get("model") if isinstance(golden, dict) else None
        golden_version = golden.get("version") if isinstance(golden, dict) else None

        label_parts = [name]
        if bundle:
            label_parts.append(bundle)
        if golden_model and golden_version:
            label_parts.append(f"{golden_model}/{golden_version}")

        cfg_name = " | ".join(label_parts)

        for _ in range(repeat):
            configs.append(RunConfig(
                name=cfg_name, tests=tests, filter_k=filter_k,
                bundle=bundle, target=tgt,
                golden_model=golden_model, golden_version=golden_version,
                params=params,
            ))

    return configs
