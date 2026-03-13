"""精确比对 Demo — 端到端用例.

完整流程:
  1. ensure_bundle() 安装依赖（含 clone 外部仓库 repo_a）
  2. run_vscode_task() 执行构建 (build app / build ut)
  3. 修改 filter_case.json 选择要执行的算子
  4. run_vscode_task() 执行运行 (run stub)
  5. 在产出目录下找到日志，用 log_parser 解析为 txt
  6. 按 tid_names 映射，逐算子逐字节精确比对 golden 数据
  7. golden 不存在时 save_golden() 自动上传为基准

deps.yaml 配置示例::

    repos:
      repo_a:
        url: git@example.com:team/repo_a.git
        ref: feature/npu-ops
        depth: 1

    bundles:
      npu_exact:
        description: "精确比对测试环境"
        status: verified
        repos:
          repo_a: feature/npu-ops
        env:
          BUILD_PLATFORM: func_sim

testplan.yaml 配置::

    - name: 精确比对Demo
      tests: [npu/operators/test_exact_compare_demo.py]
      bundle: npu_exact
      target: local
      golden:
        model: npu_model
        version: v1.0
      params:
        repo_name: repo_a
        filter_cases: [tdd]
        addr_shift: 16
        tid_names:
          2: matmul_layer0
          5: conv2d_layer3
          7: matmul_layer5

tid_names 说明:
  key 是 tid（int），value 是算子名（str），同时用作:
    - log_parser 输出文件的命名前缀
    - golden store 中的 operator 目录名
    - 精确比对时查找 golden 的 key

  tid_names 来源优先级:
    1. 测试类的 DEFAULT_TID_NAMES 属性（硬编码优先）
    2. testplan.yaml params.tid_names_by_version[当前version]（按版本区分）
    3. testplan.yaml params.tid_names（所有版本共享，兜底）
"""

import json
import shutil
import unittest
from pathlib import Path

from aitf.tc.api import (
    ctx,
    ensure_bundle,
    get_bundle_env,
    get_dep_path,
    load_golden_outputs,
    run_vscode_task,
    save_golden,
    select_env,
    execute,
)


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def _param(key: str, default=None):
    """从 ctx.params 获取配置，未在 runner 中运行时返回默认值."""
    if ctx.active and ctx.params:
        return ctx.params.get(key, default)
    return default


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _get_tid_names(hardcoded: dict[int, str] | None = None) -> dict[int, str]:
    """获取 tid → 算子名 映射.

    优先级: 硬编码 > testplan params.tid_names_by_version[ver] > params.tid_names

    测试套自身最清楚绑定哪些算子和 tid，因此硬编码优先；
    testplan 仅在测试套未指定时作为补充。

    支持两种 testplan params 格式:
      - tid_names: {2: matmul, 5: conv2d}           — 所有版本共享
      - tid_names_by_version: {v1: {2: matmul}, ...} — 按版本区分
    """
    if hardcoded:
        return hardcoded
    # 优先检查 per-version 映射
    by_version = _param("tid_names_by_version", {})
    if by_version and ctx.active and ctx.golden_version:
        ver_map = by_version.get(ctx.golden_version, {})
        if ver_map:
            return {int(k): str(v) for k, v in ver_map.items()}
    raw = _param("tid_names", {})
    return {int(k): str(v) for k, v in raw.items()}


class TestExactCompareDemo(unittest.TestCase):
    """精确比对 Demo 用例."""

    # =================================================================
    #  Fixture: bundle install + build
    # =================================================================

    @classmethod
    def setUpClass(cls):
        cls.root = _project_root()

        # ── 1. 安装 Bundle（clone 仓库 + 安装工具链） ─────────
        bundle_name = ctx.bundle if ctx.active else _param("bundle", "npu_exact")
        if bundle_name:
            ensure_bundle(bundle_name)
            cls.bundle_env = get_bundle_env(bundle_name)
        else:
            cls.bundle_env = {}

        # 获取仓库路径（deps 系统管理的 repo）
        repo_name = _param("repo_name", "repo_a")
        cls.repo_dir = get_dep_path(repo_name)
        if cls.repo_dir is None:
            cls.repo_dir = Path(_param("repo_dir", str(cls.root)))

        platform = (cls.bundle_env.get("BUILD_PLATFORM")
                     or _param("build_platform", "func_sim"))
        cls.build_dir = cls.repo_dir / "build" / platform
        cls.platform = platform

        # ── 2. 执行构建 VSCode tasks ─────────────────────────
        try:
            r = run_vscode_task(cls.repo_dir, "[stub] 01-构建桩代码 (app)",
                                env_vars=cls.bundle_env, timeout=300)
            if r.returncode != 0:
                raise RuntimeError(f"build app 失败 (rc={r.returncode}):\n{r.stderr}")

            r = run_vscode_task(cls.repo_dir, "[stub] 02-构建单元测试 (ut)",
                                env_vars=cls.bundle_env, timeout=300)
            if r.returncode != 0:
                raise RuntimeError(f"build ut 失败 (rc={r.returncode}):\n{r.stderr}")

        except FileNotFoundError:
            env = select_env("local")
            build_sh = cls.repo_dir / "stubs" / "build.sh"
            if not build_sh.exists():
                build_sh = cls.root / "stubs" / "build.sh"

            r = execute(env, f"{build_sh} -p {platform} -m app",
                        cwd=str(cls.repo_dir), timeout=300,
                        env_vars=cls.bundle_env)
            if r.returncode != 0:
                raise RuntimeError(f"build app 失败:\n{r.stderr}")

            r = execute(env, f"{build_sh} -p {platform} -m ut",
                        cwd=str(cls.repo_dir), timeout=300,
                        env_vars=cls.bundle_env)
            if r.returncode != 0:
                raise RuntimeError(f"build ut 失败:\n{r.stderr}")
            env.cleanup()

    # =================================================================
    #  辅助方法
    # =================================================================

    def _write_filter_case(self, cases: list[str]) -> Path:
        """生成 filter_case.json 选择要执行的算子."""
        filter_path = self.build_dir / "filter_case.json"
        filter_path.parent.mkdir(parents=True, exist_ok=True)
        content = {
            "enabled": True,
            "cases": cases,
            "description": "auto-generated by test_exact_compare_demo",
        }
        filter_path.write_text(
            json.dumps(content, indent=2), encoding="utf-8")
        return filter_path

    def _run_stub(self) -> str:
        """执行桩代码 — 通过 VSCode task 或直接命令."""
        result_dir = self.build_dir / _param("log_subdir", "output")
        result_dir.mkdir(parents=True, exist_ok=True)

        try:
            r = run_vscode_task(self.repo_dir, "[stub] 03-运行桩代码",
                                env_vars=self.bundle_env, timeout=600,
                                skip_deps=True)
        except (FileNotFoundError, KeyError):
            runner = self.build_dir / "stub_runner"
            if not runner.exists():
                candidates = list(self.build_dir.glob("libstub_runner.*"))
                runner = candidates[0] if candidates else runner

            models = _param("filter_cases", ["tdd"])
            cmd = (f"{runner} --models {','.join(models)} "
                   f"--result-dir {result_dir}")

            env = select_env("local")
            r = execute(env, cmd, cwd=str(self.repo_dir),
                        timeout=600, env_vars=self.bundle_env)
            env.cleanup()

        return r.stdout + r.stderr

    def _find_log(self, pattern: str = "*.log") -> Path | None:
        """在构建产物目录下查找最新日志文件."""
        result_dir = self.build_dir / _param("log_subdir", "output")
        if not result_dir.exists():
            return None
        logs = sorted(result_dir.glob(pattern),
                      key=lambda p: p.stat().st_mtime)
        return logs[-1] if logs else None

    def _parse_log(self, log_path: Path,
                   tid_names: dict[int, str]) -> Path:
        """调用 log_parser 解析日志，返回输出目录."""
        from aitf.compare.log_parser import run as log_parser_run
        from aitf.compare.op_log_parser import TxtFormat

        out_dir = self.build_dir / "parsed_output"
        if out_dir.exists():
            shutil.rmtree(out_dir)

        addr_shift = int(_param("addr_shift", 16))
        name_lower = log_path.name.lower()

        log_parser_run(
            op_log=str(log_path) if "op" in name_lower else None,
            tensor_log=str(log_path) if "tensor" in name_lower else None,
            out_dir=str(out_dir),
            tid_names=tid_names,
            cross_op=True,
            split_operand=True,
            addr_shift=addr_shift,
            txt_fmt=TxtFormat.NO_HEADER_1B,
        )
        return out_dir

    # =================================================================
    #  测试方法
    # =================================================================

    # 当 testplan 未配置 tid_names 时使用的硬编码默认值。
    # 子类或具体用例可覆盖此属性来指定自己的 tid 映射。
    DEFAULT_TID_NAMES: dict[int, str] = {}

    def test_exact_compare(self):
        """完整精确比对流程: filter → 运行 → 解析 → 逐算子比对 golden."""
        tid_names = _get_tid_names(hardcoded=self.DEFAULT_TID_NAMES)
        filter_cases = _param("filter_cases", list(tid_names.values()) or ["tdd"])

        # ── 3. 修改 filter_case.json ──────────────────────────
        filter_path = self._write_filter_case(filter_cases)
        self.assertTrue(filter_path.exists(),
                        f"filter_case.json 未写入: {filter_path}")

        # ── 4. 执行运行任务 ───────────────────────────────────
        self._run_stub()

        # ── 5. 查找日志 ──────────────────────────────────────
        log_path = self._find_log() or self._find_log("*.txt")
        self.assertIsNotNone(log_path, "未找到产出日志文件")

        # ── 6. 解析日志 ──────────────────────────────────────
        parsed_dir = self._parse_log(log_path, tid_names)
        self.assertTrue(parsed_dir.exists(),
                        f"解析输出目录不存在: {parsed_dir}")

        # ── 7. 按算子逐个比对 golden ─────────────────────────
        # 收集所有 operator 名（去重，保持顺序）
        operators = list(dict.fromkeys(tid_names.values())) if tid_names else filter_cases

        total_compared = 0
        for operator in operators:
            output_txts = sorted(
                p for p in parsed_dir.rglob("*.txt")
                if "Output" in p.name and "summary" not in p.name
            )
            if not output_txts:
                continue

            golden_items = load_golden_outputs(operator)

            if golden_items is None:
                # 首次运行 → 上传当前 output 为 golden 基准
                print(f"[golden] 算子 '{operator}' 无 golden，"
                      f"上传 {len(output_txts)} 文件为基准")
                save_golden(operator, output_txts)
                continue

            self.assertEqual(
                len(output_txts), len(golden_items),
                f"[{operator}] 输出数 ({len(output_txts)}) != "
                f"golden 数 ({len(golden_items)})")

            for txt_path, golden in zip(output_txts, golden_items):
                actual = txt_path.read_bytes()
                expected = golden.read_bytes()

                self.assertEqual(
                    actual, expected,
                    f"[{operator}] 精确比对失败: "
                    f"{txt_path.name} vs {golden.path.name}\n"
                    f"  {_first_diff(actual, expected)}")
                total_compared += 1

        if total_compared > 0:
            print(f"[OK] 精确比对通过: {total_compared} 个输出文件, "
                  f"{len(operators)} 个算子全部一致")
        else:
            self.skipTest("首次运行，已上传 golden 基准，无比对执行")

    def test_input_files_generated(self):
        """验证输入文件也被正确解析生成."""
        parsed_dir = self.build_dir / "parsed_output"
        if not parsed_dir.exists():
            self.skipTest("需先执行 test_exact_compare 生成解析结果")

        input_txts = sorted(parsed_dir.rglob("*Input*.txt"))
        self.assertGreater(len(input_txts), 0, "未生成任何 Input 文件")
        for txt in input_txts:
            self.assertGreater(txt.stat().st_size, 0,
                               f"Input 文件为空: {txt.name}")


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _first_diff(a: bytes, b: bytes) -> str:
    """找到首个不同字节的位置."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return (f"offset=0x{i:X} (byte {i}): "
                    f"actual=0x{x:02X} expected=0x{y:02X}")
    if len(a) != len(b):
        return f"长度不同: actual={len(a)} expected={len(b)}"
    return "完全相同"


if __name__ == "__main__":
    unittest.main()
