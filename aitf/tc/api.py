"""用例编写 API — 测试用例中 import 此模块获取运行时上下文。

提供的主要接口:

- ``ctx``              — 运行时上下文代理（bundle / target / golden_model 等）
- ``load_golden()``       — 加载 golden 数据，返回带 category 的 GoldenItem 列表
- ``load_golden_inputs()``— 只加载 input 类别的 golden 数据
- ``load_golden_outputs()``— 只加载 output 类别的 golden 数据（期望结果）
- ``get_golden_file()``   — 获取单个 golden 数据文件路径
- ``save_golden()``       — 上传文件为 golden 基准数据
- ``ensure_dep()``     — 确保单个依赖已安装（含构建）并返回路径
- ``rebuild_dep()``    — 强制重新构建指定依赖
- ``ensure_bundle()``  — 确保 bundle 所有依赖已安装（含构建）
- ``get_dep_path()``   — 获取已安装依赖路径（不触发安装）
- ``get_dep_env()``    — 获取单个依赖的环境变量
- ``get_bundle_env()`` — 获取 bundle 所有环境变量
- ``run_vscode_task()``— 执行仓库 .vscode/tasks.json 中的 task
- ``run_script()``     — 运行项目下的 shell 脚本
- ``get_last_stats()`` — 获取最近一次执行统计
- ``get_stats()``      — 获取指定执行的统计
- ``get_case_results()``— 获取执行中每个用例的结果
- ``list_executions()``— 列出最近的执行记录

用法示例::

    from aitf.tc.api import ctx, load_golden, ensure_dep, get_bundle_env

    class TestMatMul(unittest.TestCase):
        def test_basic(self):
            # 确保依赖可用
            ensure_dep("cann-toolkit")

            # 加载 golden 对比数据
            files = load_golden(operator="matmul")

            # 获取环境变量
            import os
            os.environ.update(get_bundle_env())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class _EmptyContext:
    """未在 aitf runner 中运行时的默认上下文。"""
    name: str = ""
    bundle: str | None = None
    target: str | None = None
    golden_model: str | None = None
    golden_version: str | None = None
    tests: list[str] | None = None
    filter_k: str | None = None
    params: dict[str, Any] | None = None


class _ContextProxy:
    """代理对象，始终指向当前线程的 RunConfig。

    在 aitf runner 执行期间返回真实的 RunConfig；
    未在 runner 中运行时返回空默认值（不会报错）。
    """

    def __getattr__(self, name: str) -> Any:
        from aitf.tc.runner import get_run_context
        rc = get_run_context()
        if rc is None:
            rc = _EmptyContext()
        return getattr(rc, name)

    def __repr__(self) -> str:
        from aitf.tc.runner import get_run_context
        rc = get_run_context()
        return repr(rc) if rc else "RunContext(not active)"

    @property
    def active(self) -> bool:
        """是否在 aitf runner 中运行。"""
        from aitf.tc.runner import get_run_context
        return get_run_context() is not None


# 单例，用例中直接 import 使用
ctx = _ContextProxy()


# ---------------------------------------------------------------------------
# Golden 数据相关
# ---------------------------------------------------------------------------

def get_golden_store():
    """获取 GoldenStore 实例，用于读取 golden 数据。

    CLIENT 模式下返回 CachedGoldenStore（透明远程拉取 + 本地缓存）；
    其他模式返回普通 GoldenStore。
    返回 None 如果 datastore 模块不可用。
    """
    try:
        from flask import current_app
        cfg = current_app.config.get("AITF_CONFIG")
        sc = current_app.config.get("SYNC_CLIENT")
        base = current_app.config.get("DATASTORE_BASE_DIR", "datastore")
        if cfg and sc:
            from aitf.sync.cache import CachedGoldenStore
            return CachedGoldenStore(base, sc)
        from aitf.ds.store import GoldenStore
        return GoldenStore(base)
    except (ImportError, RuntimeError):
        pass
    try:
        from aitf.ds.store import GoldenStore
        return GoldenStore()
    except (ImportError, OSError) as exc:
        logger.debug("GoldenStore unavailable: %s", exc)
        return None


@dataclass
class GoldenItem:
    """一条 golden 数据项，包含文件路径和解析后的元信息。"""
    path: Path
    category: str       # input / output / weight / bias / mask / index / other
    seq: int            # 同类别下的序号
    layout: str         # NCHW / NHWC / ND / ...
    precision: str      # FP16 / FP32 / INT8 / ...
    shape: tuple[int, ...]  # (loop, m, n, k)

    def read_bytes(self) -> bytes:
        """读取文件原始二进制数据。"""
        return self.path.read_bytes()


def _resolve_model_version(model: str | None,
                           version: str | None) -> tuple[str | None, str | None]:
    if model is None:
        model = ctx.golden_model
    if version is None:
        version = ctx.golden_version
    return model, version


def load_golden(operator: str, model: str | None = None,
                version: str | None = None,
                category: str | None = None) -> list[GoldenItem] | None:
    """加载 golden 数据，返回结构化的 GoldenItem 列表。

    每个 GoldenItem 包含 category（input/output/weight/...）、shape、
    precision 等元信息，可直接区分输入输出。

    Args:
        operator: 算子名（对应 golden 目录下的子目录名）。
        model: golden 模型名，默认从运行上下文获取。
        version: golden 版本，默认从运行上下文获取。
        category: 可选过滤，只返回指定类别（如 "input" / "output"）。

    Returns:
        GoldenItem 列表，按 (category, seq) 排序。
        未配置或无数据时返回 None。

    用法示例::

        items = load_golden("matmul")
        if items:
            inputs  = [g for g in items if g.category == "input"]
            outputs = [g for g in items if g.category == "output"]
            for inp in inputs:
                data = inp.read_bytes()
                print(f"{inp.category}{inp.seq}: {inp.precision} {inp.shape}")
    """
    model, version = _resolve_model_version(model, version)
    if not model or not version:
        return None
    gs = get_golden_store()
    if gs is None:
        return None
    try:
        from aitf.ds.store import parse_golden_filename
        files = gs.get_operator_files(model, version, operator)
        if not files:
            return None
        result: list[GoldenItem] = []
        for fp in files:
            parsed = parse_golden_filename(fp.name)
            if parsed is None:
                continue
            if category and parsed["category"] != category:
                continue
            result.append(GoldenItem(
                path=fp,
                category=parsed["category"],
                seq=parsed["seq"],
                layout=parsed["layout"],
                precision=parsed["precision"],
                shape=(parsed["loop"], parsed["m"], parsed["n"], parsed["k"]),
            ))
        result.sort(key=lambda g: (g.category, g.seq))
        return result if result else None
    except (ImportError, FileNotFoundError, OSError) as exc:
        logger.debug("load_golden(%s) failed: %s", operator, exc)
        return None


def load_golden_inputs(operator: str, **kwargs) -> list[GoldenItem] | None:
    """加载算子的 golden 输入数据。等同于 ``load_golden(op, category="input")``。"""
    return load_golden(operator, category="input", **kwargs)


def load_golden_outputs(operator: str, **kwargs) -> list[GoldenItem] | None:
    """加载算子的 golden 输出数据（期望结果）。等同于 ``load_golden(op, category="output")``。"""
    return load_golden(operator, category="output", **kwargs)


def get_golden_file(operator: str, filename: str,
                    model: str | None = None,
                    version: str | None = None) -> Path | None:
    """获取单个 golden 数据文件路径。

    Args:
        operator: 算子名。
        filename: 文件名（如 ``matmul_input0_NCHW_FP16_128x96.bin``）。
        model: golden 模型名，默认从运行上下文获取。
        version: golden 版本，默认从运行上下文获取。

    Returns:
        文件路径，不存在时返回 None。
    """
    model, version = _resolve_model_version(model, version)
    if not model or not version:
        return None
    gs = get_golden_store()
    if gs is None:
        return None
    try:
        return gs.get_file(model, version, operator, filename)
    except (FileNotFoundError, OSError) as exc:
        logger.debug("get_golden_file(%s/%s) failed: %s", operator, filename, exc)
        return None


# ---------------------------------------------------------------------------
# 依赖包 / Bundle 相关
# ---------------------------------------------------------------------------

def _get_deps_manager():
    """获取 DepsManager 实例。"""
    try:
        from aitf.deps.manager import DepsManager
        root = ctx.params.get("project_root", ".") if ctx.active and ctx.params else "."
        return DepsManager(project_root=root)
    except (ImportError, OSError) as exc:
        logger.debug("DepsManager unavailable: %s", exc)
        return None


def _get_bundle_manager():
    """获取 BundleManager 实例。"""
    try:
        dm = _get_deps_manager()
        if dm is None:
            return None
        from aitf.deps.bundle import BundleManager
        return BundleManager(dm, deps_file=dm.deps_file)
    except (ImportError, OSError) as exc:
        logger.debug("BundleManager unavailable: %s", exc)
        return None


def ensure_dep(name: str) -> Path | None:
    """确保指定依赖已安装并返回安装路径。

    完整流程：下载/克隆 → 解压 → 执行 build_script（如有配置）。
    如果依赖已安装则直接返回路径，不重复安装。

    Args:
        name: 依赖名称（toolchain / library / repo 均可）。

    Returns:
        安装目录路径，失败时返回 None。

    用法示例::

        path = ensure_dep("cann-toolkit")
        if path:
            os.environ["ASCEND_HOME"] = str(path)
    """
    dm = _get_deps_manager()
    if dm is None:
        return None
    try:
        install_dir = dm.get_install_dir(name)
        if install_dir is not None:
            return install_dir
        # 未安装，执行安装（含构建）
        logger.info("Installing dependency: %s", name)
        dm.install(name)
        return dm.get_install_dir(name)
    except Exception as exc:
        logger.warning("Failed to ensure dep '%s': %s", name, exc)
        return None


def rebuild_dep(name: str) -> Path | None:
    """强制重新构建指定依赖（删除后重新安装）。

    适用于源码修改后需要重新编译的场景。

    Args:
        name: 依赖名称。

    Returns:
        安装目录路径，失败时返回 None。

    用法示例::

        # 修改了 repo 源码后重新构建
        path = rebuild_dep("my-custom-lib")
    """
    dm = _get_deps_manager()
    if dm is None:
        return None
    try:
        import shutil
        install_dir = dm.get_install_dir(name)
        if install_dir is not None and install_dir.is_dir():
            logger.info("Removing existing install: %s", install_dir)
            shutil.rmtree(install_dir)
        logger.info("Rebuilding dependency: %s", name)
        dm.install(name)
        return dm.get_install_dir(name)
    except Exception as exc:
        logger.warning("Failed to rebuild dep '%s': %s", name, exc)
        return None


def ensure_bundle(name: str | None = None) -> bool:
    """确保指定 bundle 的所有依赖已安装（含构建）。

    会按 deps.yaml 中定义的顺序安装 bundle 引用的所有
    toolchain / library / repo，并执行各自的 build_script。

    Args:
        name: bundle 名称。默认使用运行上下文中配置的 bundle，
              或 deps.yaml 中的 active bundle。

    Returns:
        True 安装成功，False 失败或不可用。

    用法示例::

        ensure_bundle("npu_test_v2")
    """
    bm = _get_bundle_manager()
    if bm is None:
        return False
    try:
        if name is None:
            name = ctx.bundle if ctx.active else None
        if name is None:
            active = bm.active()
            if active is None:
                return False
            name = active.name
        bm.install(name)
        return True
    except Exception as exc:
        logger.warning("Failed to ensure bundle '%s': %s", name, exc)
        return False


def get_dep_path(name: str) -> Path | None:
    """获取已安装依赖的路径（不触发安装）。

    Args:
        name: 依赖名称。

    Returns:
        安装目录路径，未安装时返回 None。
    """
    dm = _get_deps_manager()
    if dm is None:
        return None
    try:
        return dm.get_install_dir(name)
    except (KeyError, OSError) as exc:
        logger.debug("get_dep_path(%s) failed: %s", name, exc)
        return None


def get_dep_env(name: str) -> dict[str, str]:
    """获取单个依赖配置的环境变量。

    在 deps.yaml 中通过 ``env`` 字段定义的变量，
    其中 ``{install_dir}`` 会被替换为实际安装路径。

    Args:
        name: 依赖名称（toolchain / repo）。

    Returns:
        环境变量字典，未找到时返回空字典。

    用法示例::

        import os
        os.environ.update(get_dep_env("cann-toolkit"))
        # 例如 ASCEND_HOME=/path/to/build/cache/cann-toolkit
    """
    dm = _get_deps_manager()
    if dm is None:
        return {}
    try:
        cfg = dm.config
        for section, base in [(cfg.toolchains, dm.cache_dir), (cfg.repos, dm.repos_dir)]:
            if name in section:
                dep = section[name]
                from aitf.deps.types import resolve_dep_dir
                d = resolve_dep_dir(dep, base, dm._build_dir)
                if d.is_dir():
                    return {k: v.replace("{install_dir}", str(d))
                            for k, v in dep.env.items()}
                return {}
        return {}
    except (KeyError, ImportError, OSError) as exc:
        logger.debug("get_dep_env(%s) failed: %s", name, exc)
        return {}


def get_bundle_env(name: str | None = None) -> dict[str, str]:
    """获取 bundle 关联的环境变量（含所有依赖的 env）。

    Args:
        name: bundle 名称，默认使用当前激活的 bundle。

    Returns:
        环境变量字典（可直接 ``os.environ.update(...)``）。

    用法示例::

        import os
        os.environ.update(get_bundle_env("npu_test_v2"))
    """
    bm = _get_bundle_manager()
    if bm is None:
        return {}
    try:
        if name is None and ctx.active:
            name = ctx.bundle
        return bm.get_bundle_env(name)
    except (KeyError, ImportError, OSError) as exc:
        logger.debug("get_bundle_env(%s) failed: %s", name, exc)
        return {}


# ---------------------------------------------------------------------------
# 执行环境 / 命令执行 (REQ-4.2 / REQ-6)
# ---------------------------------------------------------------------------

def select_env(target_name: str = "local",
               targets_file: str | None = None) -> Any:
    """选择执行环境，返回 Environment 对象。

    Args:
        target_name: 目标名（对应 runner/targets.yaml 中的配置），
                     如 "local", "sim-server", "fpga-board"。
        targets_file: targets.yaml 路径，默认自动查找。

    Returns:
        Environment 对象，封装本地或远程执行能力。

    用法示例::

        env = select_env("sim-server")
        result = execute(env, "ls -la")
        env.cleanup()
    """
    from aitf.runner.config import TargetConfig, load_runner_config
    from aitf.runner.environment import Environment

    if targets_file is None:
        # 尝试多个默认路径
        dm = _get_deps_manager()
        root = Path(dm.project_root) if dm else Path(".")
        for candidate in [root / "runner" / "targets.yaml",
                          root / "targets.yaml",
                          root / "config" / "targets.yaml"]:
            if candidate.is_file():
                targets_file = str(candidate)
                break

    if targets_file and Path(targets_file).is_file():
        targets = load_runner_config(targets_file)
        if target_name not in targets:
            raise KeyError(f"Target '{target_name}' not found in {targets_file}. "
                           f"Available: {list(targets.keys())}")
        config = targets[target_name]
    else:
        # 无配置文件时，"local" 使用默认配置
        if target_name == "local":
            config = TargetConfig(name="local")
        else:
            raise FileNotFoundError(
                f"No targets.yaml found and target '{target_name}' is not 'local'")

    return Environment(target_name, config)


def execute(env: Any, command: str, timeout: int = 300,
            env_vars: dict[str, str] | None = None,
            cwd: str | None = None) -> Any:
    """在指定环境中执行命令。

    Args:
        env: select_env() 返回的 Environment 对象。
        command: 要执行的 shell 命令字符串。
        timeout: 超时秒数，默认 300。
        env_vars: 额外环境变量。
        cwd: 工作目录（仅本地执行有效）。

    Returns:
        ExecuteResult，包含 returncode, stdout, stderr, duration_s。

    用法示例::

        env = select_env("local")
        result = execute(env, "echo hello")
        assert result.returncode == 0
        print(result.stdout)
    """
    return env.run(command, timeout=timeout, env=env_vars, cwd=cwd)


def run_script(script: str, args: list[str] | None = None,
               timeout: int = 600) -> bool:
    """运行项目根目录下的 shell 脚本。

    Args:
        script: 脚本路径（相对于项目根目录）。
        args: 传给脚本的参数列表。
        timeout: 超时秒数，默认 600。

    Returns:
        True 成功，False 失败。

    用法示例::

        # 运行自定义构建脚本
        run_script("scripts/build_model.sh", ["--config", "release"])

        # 运行数据预处理脚本
        run_script("scripts/prepare_data.sh", [str(get_dep_path("dataset"))])
    """
    dm = _get_deps_manager()
    if dm is None:
        return False
    try:
        from aitf.deps.acquire import run_script as _run
        _run(script, args or [], project_root=dm.project_root, timeout=timeout)
        return True
    except Exception as exc:
        logger.warning("Script '%s' failed: %s", script, exc)
        return False


# ---------------------------------------------------------------------------
# VSCode Task 执行
# ---------------------------------------------------------------------------

def run_vscode_task(repo_dir: str | Path, label: str,
                    *,
                    env_vars: dict[str, str] | None = None,
                    timeout: int = 300,
                    skip_deps: bool = False) -> Any:
    """执行仓库 .vscode/tasks.json 中指定 label 的 task.

    自动解析 ``dependsOn`` 依赖链，按顺序执行前置任务。
    如果前置任务失败会抛出 RuntimeError。

    Args:
        repo_dir: 仓库根目录（包含 ``.vscode/tasks.json``）。
        label: task label（如 ``"[stub] 01-构建桩代码 (app)"``）。
        env_vars: 额外环境变量，会与系统环境合并。
        timeout: 每个 task 的超时秒数（默认 300）。
        skip_deps: 为 True 时跳过 dependsOn，仅执行目标 task。

    Returns:
        ExecuteResult，包含 returncode / stdout / stderr / duration_s。

    Raises:
        FileNotFoundError: .vscode/tasks.json 不存在。
        KeyError: label 不存在。
        RuntimeError: 依赖 task 执行失败。

    用法示例::

        from aitf.tc.api import run_vscode_task

        # 执行构建任务（自动先执行 dependsOn 的前置任务）
        result = run_vscode_task("/path/to/repo",
                                 "[stub] 01-构建桩代码 (app)")
        assert result.returncode == 0

        # 带 bundle 环境变量执行
        result = run_vscode_task(
            repo_dir, "[stub] 03-运行桩代码",
            env_vars=get_bundle_env("npu_v2"),
        )
    """
    from aitf.runner.vscode_task import run_vscode_task as _run
    return _run(repo_dir, label, env_vars=env_vars,
                timeout=timeout, skip_deps=skip_deps)


# ---------------------------------------------------------------------------
# Golden 数据上传
# ---------------------------------------------------------------------------

def save_golden(operator: str, files: list[Path],
                model: str | None = None,
                version: str | None = None,
                categories: list[str] | None = None) -> bool:
    """上传文件为 golden 基准数据.

    将指定文件保存到 ``datastore/store/<model>/<version>/<operator>/``，
    并自动生成元数据。

    Args:
        operator: 算子名（对应 golden 目录下的子目录名）。
        files: 要上传的文件路径列表。
        model: golden 模型名，默认从运行上下文获取。
        version: golden 版本，默认从运行上下文获取。
        categories: 每个文件的类别（input/output/...），
                    默认根据文件名中的 Input/Output 关键字推断。

    Returns:
        True 成功，False 失败。

    用法示例::

        # 上传解析后的输出文件为 golden 基准
        save_golden("tdd", [Path("output0.txt"), Path("output1.txt")])

        # 指定 model/version
        save_golden("matmul", parsed_files,
                    model="npu_model", version="v2.0")
    """
    import shutil
    model, version = _resolve_model_version(model, version)
    if not model or not version:
        logger.warning("save_golden: 未指定 model/version，无法上传")
        return False

    gs = get_golden_store()
    if gs is None:
        logger.warning("save_golden: GoldenStore 不可用")
        return False

    try:
        from aitf.ds.store import DataItem, OperatorMeta, build_golden_filename

        meta_items: list[DataItem] = []
        saved_names: list[str] = []

        for i, fp in enumerate(files):
            fp = Path(fp)

            # 推断 category
            if categories and i < len(categories):
                cat = categories[i]
            elif "Input" in fp.name or "input" in fp.name:
                cat = "input"
            elif "Output" in fp.name or "output" in fp.name:
                cat = "output"
            else:
                cat = "other"

            item = DataItem(seq=i, category=cat)
            meta_items.append(item)

            golden_fname = build_golden_filename(operator, item, ext=fp.suffix)
            dest = gs.save_file(model, version, operator, golden_fname)
            shutil.copy2(fp, dest)
            saved_names.append(golden_fname)

        gs.save_meta(model, version, operator,
                     OperatorMeta(name=operator, data=meta_items))
        gs.log("save_golden", model=model, version=version,
               operator=operator, files=saved_names,
               source="auto")
        logger.info("save_golden: %s/%s/%s — %d 文件已上传",
                     model, version, operator, len(saved_names))
        return True
    except Exception as exc:
        logger.warning("save_golden failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# 执行统计查询
# ---------------------------------------------------------------------------

@dataclass
class ExecutionStats:
    """一次执行的统计摘要：通过 / 失败 / 未执行。"""
    execution_id: str
    total: int
    passed: int
    failed: int
    not_run: int

    def summary(self) -> str:
        return f"总计: {self.total}  通过: {self.passed}  失败: {self.failed}  未执行: {self.not_run}"


@dataclass
class CaseStatus:
    """单个用例状态：通过 / 失败 / 未执行。"""
    suite_class: str
    case_method: str
    status: str           # passed / failed / not_run
    failure_reason: str | None = None


def get_last_stats() -> ExecutionStats | None:
    """获取最近一次执行的统计。

    用法示例::

        stats = get_last_stats()
        if stats:
            print(stats.summary())
            # 总计: 20  通过: 18  失败: 2  未执行: 0
    """
    try:
        from aitf.tc import store
        rows = store.list_executions(limit=1)
        if not rows:
            return None
        return _dict_to_stats(rows[0])
    except (ImportError, RuntimeError) as exc:
        logger.debug("get_last_stats failed: %s", exc)
        return None


def get_stats(execution_id: str) -> ExecutionStats | None:
    """获取指定执行的统计。"""
    try:
        from aitf.tc import store
        detail = store.get_execution_detail(execution_id)
        if not detail:
            return None
        return _dict_to_stats(detail)
    except (ImportError, RuntimeError) as exc:
        logger.debug("get_stats(%s) failed: %s", execution_id, exc)
        return None


def get_case_results(execution_id: str) -> list[CaseStatus] | None:
    """获取指定执行中每个用例的结果。

    用法示例::

        results = get_case_results("20260312-143000-abc123")
        for r in results:
            print(f"{r.suite_class}.{r.case_method}: {r.status}")
        failed = [r for r in results if r.status == "failed"]
    """
    try:
        from aitf.tc import store
        detail = store.get_execution_detail(execution_id)
        if not detail or "cases" not in detail:
            return None
        return [
            CaseStatus(
                suite_class=c["suite_class"],
                case_method=c["case_method"],
                status=_simplify_status(c["status"]),
                failure_reason=c.get("failure_reason"),
            )
            for c in detail["cases"]
        ]
    except (ImportError, RuntimeError) as exc:
        logger.debug("get_case_results(%s) failed: %s", execution_id, exc)
        return None


def list_executions(limit: int = 20) -> list[ExecutionStats]:
    """获取最近的执行记录列表。"""
    try:
        from aitf.tc import store
        rows = store.list_executions(limit=limit)
        return [_dict_to_stats(r) for r in rows]
    except (ImportError, RuntimeError) as exc:
        logger.debug("list_executions failed: %s", exc)
        return []


def _simplify_status(raw: str) -> str:
    """内部状态 → 三态：passed / failed / not_run。"""
    from aitf.tc.models import CaseStatus
    if raw == CaseStatus.PASS:
        return "passed"
    if raw in CaseStatus.FAILURE:
        return "failed"
    return "not_run"  # PENDING / RUNNING / SKIP


def _dict_to_stats(d: dict) -> ExecutionStats:
    total = d.get("total", 0)
    passed = d.get("passed", 0)
    # Use pre-computed fields if available (from Execution.to_dict)
    failed = d.get("failed_total",
                    (d.get("failed", 0) + d.get("errored", 0)
                     + d.get("timeout", 0) + d.get("crashed", 0)))
    not_run = d.get("not_run", max(0, total - passed - failed))
    return ExecutionStats(
        execution_id=d["id"],
        total=total,
        passed=passed,
        failed=failed,
        not_run=not_run,
    )
