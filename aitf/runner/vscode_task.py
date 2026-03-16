"""Execute VSCode tasks from a repository's .vscode/tasks.json.

Usage::

    from aitf.runner.vscode_task import run_vscode_task

    result = run_vscode_task("/path/to/repo", "[stub] 01-构建桩代码 (app)")
    assert result.returncode == 0
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from .executor import ExecuteResult, TargetConfig
from .local import LocalExecutor

logger = logging.getLogger(__name__)


def _load_tasks(repo_dir: Path) -> dict[str, dict]:
    """Load .vscode/tasks.json and return a label → task mapping."""
    tasks_path = repo_dir / ".vscode" / "tasks.json"
    if not tasks_path.is_file():
        raise FileNotFoundError(f"未找到 {tasks_path}")

    with open(tasks_path, encoding="utf-8") as fh:
        raw = json.load(fh)

    tasks = {}
    for task in raw.get("tasks", []):
        label = task.get("label")
        if label:
            tasks[label] = task
    return tasks


def _resolve_order(label: str,
                   tasks: dict[str, dict],
                   stack: set[str] | None = None,
                   resolved: set[str] | None = None) -> list[str]:
    """Resolve dependsOn chain, return labels in execution order (deps first).

    Raises ValueError on circular dependency, KeyError on missing label.
    """
    if stack is None:
        stack = set()
    if resolved is None:
        resolved = set()

    if label in stack:
        raise ValueError(f"循环依赖: {label}")

    if label in resolved:
        return []  # already processed in another branch

    if label not in tasks:
        raise KeyError(f"未找到 task label: {label!r}，"
                       f"可用: {list(tasks.keys())}")

    stack.add(label)

    task = tasks[label]
    deps = task.get("dependsOn", [])
    if isinstance(deps, str):
        deps = [deps]

    order: list[str] = []
    for dep_label in deps:
        for sub in _resolve_order(dep_label, tasks, stack, resolved):
            if sub not in order:
                order.append(sub)

    if label not in order:
        order.append(label)

    stack.discard(label)
    resolved.add(label)
    return order


def run_vscode_task(
    repo_dir: str | Path,
    label: str,
    *,
    env_vars: dict[str, str] | None = None,
    timeout: int = 300,
    skip_deps: bool = False,
) -> ExecuteResult:
    """Execute a VSCode task by label.

    Reads ``.vscode/tasks.json`` from *repo_dir*, resolves ``dependsOn``
    chain (unless *skip_deps*), and executes each task's ``command`` in
    order.  Returns the :class:`ExecuteResult` of the **final** (target)
    task.

    Args:
        repo_dir: Repository root directory containing ``.vscode/tasks.json``.
        label: Task label to execute (e.g. ``"[stub] 01-构建桩代码 (app)"``).
        env_vars: Extra environment variables merged into the task's env.
        timeout: Per-task timeout in seconds.
        skip_deps: If True, only run the target task, ignoring dependsOn.

    Returns:
        ExecuteResult of the final task.

    Raises:
        FileNotFoundError: If ``.vscode/tasks.json`` does not exist.
        KeyError: If the label is not found in tasks.json.
        RuntimeError: If a dependency task fails (non-zero returncode).
    """
    repo_dir = Path(repo_dir)
    tasks = _load_tasks(repo_dir)

    if skip_deps:
        order = [label]
    else:
        order = _resolve_order(label, tasks)

    config = TargetConfig(name="vscode-task", type="local")
    executor = LocalExecutor(config)

    result: ExecuteResult | None = None
    for task_label in order:
        task = tasks[task_label]
        command = task.get("command", "")
        if not command:
            logger.warning("task %r 无 command，跳过", task_label)
            continue

        # task 级别的 cwd（相对于 repo_dir）
        task_cwd = str(repo_dir)
        if "options" in task and "cwd" in task["options"]:
            task_cwd = str(repo_dir / task["options"]["cwd"])

        logger.info("执行 VSCode task: %s → %s", task_label, command)
        result = executor.run(command, timeout=timeout,
                              env=env_vars, cwd=task_cwd)

        is_target = (task_label == label)
        if result.returncode != 0 and not is_target:
            raise RuntimeError(
                f"依赖 task {task_label!r} 失败 (rc={result.returncode}):\n"
                f"{result.stderr}"
            )

    executor.close()

    if result is None:
        raise RuntimeError(f"task {label!r} 无可执行命令")
    return result
