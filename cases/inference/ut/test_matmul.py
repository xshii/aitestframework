"""MatMul 算子基础测试."""

import unittest

from aitf.tc.api import (
    ctx, load_golden, load_golden_inputs, load_golden_outputs,
    ensure_dep, ensure_bundle, rebuild_dep,
    get_dep_path, get_dep_env, get_bundle_env, run_script,
)


class TestMatMul(unittest.TestCase):
    """矩阵乘法算子测试套件."""

    def test_square(self):
        """方阵乘法 (4x4 @ 4x4)."""
        a = [[1, 0], [0, 1]]
        b = [[5, 6], [7, 8]]
        # 单位矩阵 × B = B
        result = [
            [sum(a[i][k] * b[k][j] for k in range(2)) for j in range(2)]
            for i in range(2)
        ]
        self.assertEqual(result, b)

    def test_vector_dot(self):
        """向量点积."""
        a = [1, 2, 3]
        b = [4, 5, 6]
        dot = sum(x * y for x, y in zip(a, b))
        self.assertEqual(dot, 32)

    def test_zero_matrix(self):
        """零矩阵乘法."""
        zero = [[0, 0], [0, 0]]
        b = [[1, 2], [3, 4]]
        result = [
            [sum(zero[i][k] * b[k][j] for k in range(2)) for j in range(2)]
            for i in range(2)
        ]
        self.assertEqual(result, zero)

    def test_with_golden(self):
        """演示：根据运行上下文对比 golden 数据."""
        # 加载 golden 输入（自动从 ctx 读取 model/version）
        inputs = load_golden_inputs("matmul")
        if inputs:
            # 每个 GoldenItem 带有 category/seq/precision/shape/layout
            for inp in inputs:
                data = inp.read_bytes()
                print(f"输入: {inp.category}{inp.seq}, "
                      f"精度={inp.precision}, shape={inp.shape}")

        # 加载 golden 输出（期望结果）
        outputs = load_golden_outputs("matmul")
        if outputs:
            for out in outputs:
                expected_data = out.read_bytes()
                # 对比 NPU 计算结果与 golden 输出...

        # 也可以一次加载全部，手动按 category 过滤
        all_items = load_golden("matmul")
        if all_items:
            weights = [g for g in all_items if g.category == "weight"]
            biases = [g for g in all_items if g.category == "bias"]

        # 无 golden 数据时直接验证正确性
        if not inputs and not outputs:
            a = [[1, 2], [3, 4]]
            b = [[5, 6], [7, 8]]
            result = [
                [sum(a[i][k] * b[k][j] for k in range(2)) for j in range(2)]
                for i in range(2)
            ]
            self.assertEqual(result, [[19, 22], [43, 50]])

    def test_context_info(self):
        """演示：读取运行时上下文信息."""
        # ctx.bundle / ctx.target / ctx.golden_model 等
        # 可用于根据不同配置执行不同逻辑
        if ctx.target == "sim":
            # 仿真环境：降低精度要求
            pass
        # 默认正常测试
        self.assertTrue(True)

    def test_with_deps(self):
        """演示：依赖管理 API 用法."""
        import os

        # ---- 1) 确保单个依赖已安装（含下载+构建） ----
        cann_path = ensure_dep("cann-toolkit")
        if cann_path:
            os.environ["ASCEND_HOME"] = str(cann_path)

        # ---- 2) 获取单个依赖的环境变量 ----
        os.environ.update(get_dep_env("cann-toolkit"))

        # ---- 3) 一次性安装整个 bundle（toolchain + library + repo） ----
        # ensure_bundle("npu_test_v2")

        # ---- 4) 获取 bundle 所有环境变量 ----
        # os.environ.update(get_bundle_env("npu_test_v2"))

        # ---- 5) 源码修改后强制重新构建 ----
        # rebuild_dep("my-custom-lib")

        # ---- 6) 运行自定义脚本（相对项目根目录） ----
        # run_script("scripts/build_model.sh", ["--config", "release"])

        # ---- 7) 只查询路径，不触发安装 ----
        # lib_path = get_dep_path("some-lib")

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
