# 断言与验证模块详细设计 (Assertion & Validation)

## 模块概述

| 属性 | 值 |
|------|-----|
| **模块ID** | ASSERT |
| **模块名称** | 断言与验证 |
| **职责** | 测试结果的断言、比较和验证机制 |
| **需求覆盖** | ASSERT-001 ~ ASSERT-010 |
| **外部依赖** | aidevtools.tools.compare (精度比对) |

### 模块定位

断言与验证模块作为AI测试框架的核心验证层，集成 `aidevtools.tools.compare` 提供的精度比对能力，提供：
- **基础断言**：等值、布尔、空值、包含、异常、类型断言
- **数值断言**：近似相等、相对误差、范围断言
- **张量断言**：形状、数据类型、元素级近似、NaN/Inf检测
- **算子精度验证**：集成三列比对机制 (exact/fuzzy_pure/fuzzy_qnt)
- **性能断言**：延迟、吞吐量、内存断言
- **快照对比**：结果快照保存与对比

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Assertion & Validation Module                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      Test Framework Layer                            │    │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐        │    │
│  │  │  Basic    │  │  Tensor   │  │ Operator  │  │ Snapshot  │        │    │
│  │  │ Assertion │  │ Assertion │  │ Precision │  │  Compare  │        │    │
│  │  └───────────┘  └───────────┘  └───────────┘  └───────────┘        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    aidevtools.tools.compare (精度比对)               │    │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐        │    │
│  │  │ compare_  │  │ compare_  │  │ compare_  │  │ compare_  │        │    │
│  │  │  3col     │  │  isclose  │  │   exact   │  │   full    │        │    │
│  │  └───────────┘  └───────────┘  └───────────┘  └───────────┘        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. 逻辑视图

### 1.1 断言类层次

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Assertion Classes                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      AssertionEngine                                  │  │
│  ├───────────────────────────────────────────────────────────────────────┤  │
│  │ - assertions: Dict[str, IAssertion]                                   │  │
│  │ - soft_mode: bool                                                     │  │
│  │ - failures: List[AssertionError]                                      │  │
│  ├───────────────────────────────────────────────────────────────────────┤  │
│  │ + register(name, assertion)                                           │  │
│  │ + assert_that(actual) -> AssertionBuilder                             │  │
│  │ + collect_failures() -> List                                          │  │
│  │ + enable_soft_mode()                                                  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     Assertion Categories                              │  │
│  │                                                                       │  │
│  │  Basic                 Numeric               Tensor                   │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │  │
│  │  │ assertEqual     │  │ assertAlmostEq  │  │ assertShape     │        │  │
│  │  │ assertTrue      │  │ assertInRange   │  │ assertDtype     │        │  │
│  │  │ assertNone      │  │ assertGreater   │  │ assertAllClose  │        │  │
│  │  │ assertIn        │  │ assertLess      │  │ assertNoNaN     │        │  │
│  │  │ assertRaises    │  │ assertRelError  │  │ assertNoInf     │        │  │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘        │  │
│  │                                                                       │  │
│  │  Classification        Detection             Text                     │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │  │
│  │  │ assertClass     │  │ assertIoU       │  │ assertContains  │        │  │
│  │  │ assertTopK      │  │ assertDetection │  │ assertSimilarity│        │  │
│  │  │ assertProbThresh│  │ assertConfidence│  │ assertLength    │        │  │
│  │  │ assertConfMatrix│  │ assertBBox      │  │ assertFormat    │        │  │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘        │  │
│  │                                                                       │  │
│  │  Performance           Metrics               Snapshot                 │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │  │
│  │  │ assertLatency   │  │ assertAccuracy  │  │ saveSnapshot    │        │  │
│  │  │ assertThroughput│  │ assertPrecision │  │ loadSnapshot    │        │  │
│  │  │ assertMemory    │  │ assertRecall    │  │ compareSnapshot │        │  │
│  │  │ assertRegression│  │ assertF1Score   │  │ updateSnapshot  │        │  │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘        │  │
│  │                                                                       │  │
│  │  Operator Precision (aidevtools集成)                                  │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │  │
│  │  │ assert3Col      │  │ assertIsClose   │  │ assertExact     │        │  │
│  │  │ assertPerfect   │  │ assertQSNR      │  │ assertCosine    │        │  │
│  │  │ assertNoQuantIssue│ │ assertMaxAbs   │  │ assertExceedRatio│       │  │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘        │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 流式断言API

```python
# Fluent Assertion API Design

assert_that(output).has_shape([1, 1000])
assert_that(output).has_dtype(torch.float32)
assert_that(output).is_close_to(expected, atol=1e-5)
assert_that(output).has_no_nan().has_no_inf()

assert_that(predictions).matches_class(labels)
assert_that(predictions).has_top_k_accuracy(k=5, threshold=0.9)

assert_that(latency).is_less_than_ms(10)
assert_that(throughput).is_greater_than(100)

assert_that(text).contains("keyword")
assert_that(text).matches_regex(r"\d{4}-\d{2}-\d{2}")
assert_that(json_output).is_valid_json().matches_schema(schema)
```

### 1.3 数据模型

```python
# assertion/models.py

from dataclasses import dataclass, field
from typing import Any, Optional, List
from enum import Enum

class AssertionStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class AssertionResult:
    """断言结果"""
    status: AssertionStatus
    assertion_type: str
    actual: Any
    expected: Any
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """比较结果"""
    equal: bool
    difference: float
    tolerance: float
    details: str


@dataclass
class SnapshotDiff:
    """快照差异"""
    added: List[str]
    removed: List[str]
    changed: List[tuple]
    unchanged_count: int


# ============================================
# 算子精度比对数据结构 (复用 aidevtools.tools.compare)
# ============================================

# 从 aidevtools 导入核心数据结构
from aidevtools.tools.compare import (
    DiffResult,          # 模糊比对结果
    ExactResult,         # 精确比对结果
    IsCloseResult,       # IsClose 比对结果
    FullCompareResult,   # 三列完整比对结果
    CompareThresholds,   # 比对阈值配置
)


class OpCompareStatus(Enum):
    """算子比对状态 (来自 FullCompareResult.status)"""
    PERFECT = "PERFECT"       # exact 通过 (bit级精确)
    PASS = "PASS"             # exact 不过，但 fuzzy_qnt 通过
    QUANT_ISSUE = "QUANT_ISSUE"  # fuzzy_pure 通过，fuzzy_qnt 不过 (量化问题)
    FAIL = "FAIL"             # 都不过


@dataclass
class OpPrecisionResult:
    """算子精度验证结果"""
    op_name: str
    op_id: int
    status: OpCompareStatus

    # 三列比对结果
    exact: ExactResult           # 精确比对
    fuzzy_pure: DiffResult       # 纯 fp32 模糊比对
    fuzzy_qnt: DiffResult        # 量化感知模糊比对

    # 汇总指标
    max_abs_error: float
    qsnr_db: float               # QSNR (dB)
    cosine_similarity: float

    # 断言配置
    thresholds: CompareThresholds

    @property
    def passed(self) -> bool:
        """是否通过验证"""
        return self.status in (OpCompareStatus.PERFECT, OpCompareStatus.PASS)


@dataclass
class BatchPrecisionResult:
    """批量算子精度验证结果"""
    op_results: List[OpPrecisionResult]

    # 汇总统计
    total_count: int
    perfect_count: int
    pass_count: int
    quant_issue_count: int
    fail_count: int

    @property
    def all_passed(self) -> bool:
        """是否全部通过"""
        return self.fail_count == 0

    @property
    def summary(self) -> str:
        """汇总信息"""
        return (
            f"{self.perfect_count} PERFECT, {self.pass_count} PASS, "
            f"{self.quant_issue_count} QUANT_ISSUE, {self.fail_count} FAIL "
            f"(total: {self.total_count})"
        )
```

---

## 2. 开发视图

### 2.1 包结构

```
aitest/assertion/
├── __init__.py
├── engine.py            # AssertionEngine
├── builder.py           # AssertionBuilder (fluent API)
├── basic.py             # 基础断言
├── numeric.py           # 数值断言
├── tensor.py            # 张量断言
├── classification.py    # 分类断言
├── detection.py         # 检测断言
├── text.py              # 文本断言
├── performance.py       # 性能断言
├── metric.py            # 指标断言
├── snapshot.py          # 快照对比
├── custom.py            # 自定义断言
├── errors.py            # 异常定义
├── matchers.py          # 匹配器
│
├── operator/            # 算子精度验证 (aidevtools集成)
│   ├── __init__.py
│   ├── precision.py     # 精度断言 (三列比对封装)
│   ├── isclose.py       # IsClose 断言
│   ├── thresholds.py    # 阈值配置
│   └── report.py        # 比对报告生成
│
├── soft/                # 软断言支持
│   ├── __init__.py
│   ├── context.py       # SoftAssertContext
│   └── collector.py     # 失败收集器
```

### 2.2 软断言支持 (Soft Assertions)

软断言允许在单个测试用例内收集多个断言失败，而不是在第一个失败时立即停止。这对于批量验证算子精度特别有用。

#### 2.2.1 设计思路

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Soft Assertion Architecture                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   正常断言 (Hard Assert)              软断言 (Soft Assert)                   │
│   ┌─────────────────────┐            ┌─────────────────────┐               │
│   │  assert A           │            │  soft.check(A)      │               │
│   │  ↓ (通过)           │            │  ↓ (通过/记录)      │               │
│   │  assert B           │            │  soft.check(B)      │               │
│   │  ↓ (失败 → 停止!)   │            │  ↓ (失败 → 记录)    │               │
│   │  assert C (不执行)  │            │  soft.check(C)      │               │
│   └─────────────────────┘            │  ↓ (继续执行)       │               │
│                                      │  soft.check(D)      │               │
│                                      │  ↓                  │               │
│                                      │  退出时汇总所有失败 │               │
│                                      └─────────────────────┘               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 2.2.2 SoftAssertContext 实现

```python
# assertion/soft/context.py

"""
软断言上下文

用法:
    with SoftAssertContext() as soft:
        soft.check(a == b, "a should equal b")
        soft.check_close(x, y, atol=1e-5, name="tensor_x")
        soft.check_all_close(output, golden, atol=1e-4)
    # 退出时抛出包含所有失败的异常
"""

from typing import Any, List, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class AssertionFailure:
    """断言失败记录"""
    message: str
    location: str = ""
    actual: Any = None
    expected: Any = None
    details: dict = field(default_factory=dict)


class SoftAssertionError(AssertionError):
    """软断言错误 - 包含多个失败"""

    def __init__(self, failures: List[AssertionFailure]):
        self.failures = failures
        messages = [f.message for f in failures]
        super().__init__(
            f"{len(failures)} assertion(s) failed:\n" +
            "\n".join(f"  - {m}" for m in messages)
        )

    def __len__(self) -> int:
        return len(self.failures)

    def summary(self) -> str:
        """生成失败汇总"""
        lines = [f"Total failures: {len(self.failures)}"]
        for i, f in enumerate(self.failures, 1):
            lines.append(f"  {i}. {f.message}")
        return "\n".join(lines)


class SoftAssertContext:
    """软断言上下文管理器"""

    def __init__(self, raise_on_exit: bool = True):
        """
        Args:
            raise_on_exit: 退出时是否抛出异常 (默认 True)
        """
        self.failures: List[AssertionFailure] = []
        self.raise_on_exit = raise_on_exit
        self._pass_count = 0

    def __enter__(self) -> 'SoftAssertContext':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        # 如果有其他异常，不处理
        if exc_type is not None:
            return False

        # 如果有失败且需要抛出
        if self.raise_on_exit and self.failures:
            raise SoftAssertionError(self.failures)
        return False

    # ========== 基础检查 ==========

    def check(self, condition: bool, message: str) -> bool:
        """基础条件检查"""
        if condition:
            self._pass_count += 1
            return True
        self.failures.append(AssertionFailure(message=message))
        return False

    def check_equal(self, actual: Any, expected: Any, name: str = "") -> bool:
        """相等检查"""
        if actual == expected:
            self._pass_count += 1
            return True
        prefix = f"[{name}] " if name else ""
        self.failures.append(AssertionFailure(
            message=f"{prefix}Expected {expected}, got {actual}",
            actual=actual,
            expected=expected,
        ))
        return False

    # ========== 数值检查 ==========

    def check_close(
        self,
        actual: float,
        expected: float,
        atol: float = 1e-8,
        rtol: float = 1e-5,
        name: str = ""
    ) -> bool:
        """近似相等检查"""
        diff = abs(actual - expected)
        threshold = atol + rtol * abs(expected)
        if diff <= threshold:
            self._pass_count += 1
            return True
        prefix = f"[{name}] " if name else ""
        self.failures.append(AssertionFailure(
            message=f"{prefix}Expected ~{expected}, got {actual}, diff={diff:.2e}",
            actual=actual,
            expected=expected,
            details={"diff": diff, "threshold": threshold},
        ))
        return False

    def check_in_range(
        self,
        value: float,
        min_val: float,
        max_val: float,
        name: str = ""
    ) -> bool:
        """范围检查"""
        if min_val <= value <= max_val:
            self._pass_count += 1
            return True
        prefix = f"[{name}] " if name else ""
        self.failures.append(AssertionFailure(
            message=f"{prefix}Expected in [{min_val}, {max_val}], got {value}",
            actual=value,
        ))
        return False

    # ========== 数组检查 ==========

    def check_all_close(
        self,
        actual: np.ndarray,
        expected: np.ndarray,
        atol: float = 1e-8,
        rtol: float = 1e-5,
        name: str = ""
    ) -> bool:
        """数组近似相等检查"""
        if np.allclose(actual, expected, atol=atol, rtol=rtol):
            self._pass_count += 1
            return True

        diff = np.abs(actual - expected)
        max_diff = float(diff.max())
        max_idx = np.unravel_index(diff.argmax(), diff.shape)

        prefix = f"[{name}] " if name else ""
        self.failures.append(AssertionFailure(
            message=f"{prefix}Arrays not close, max_diff={max_diff:.2e} at {max_idx}",
            details={
                "max_diff": max_diff,
                "max_idx": max_idx,
                "mean_diff": float(diff.mean()),
            },
        ))
        return False

    def check_shape(
        self,
        tensor: np.ndarray,
        expected_shape: List[int],
        name: str = ""
    ) -> bool:
        """形状检查"""
        actual_shape = list(tensor.shape)
        if actual_shape == expected_shape:
            self._pass_count += 1
            return True
        prefix = f"[{name}] " if name else ""
        self.failures.append(AssertionFailure(
            message=f"{prefix}Shape mismatch: {actual_shape} != {expected_shape}",
            actual=actual_shape,
            expected=expected_shape,
        ))
        return False

    def check_no_nan(self, tensor: np.ndarray, name: str = "") -> bool:
        """NaN 检查"""
        nan_count = np.isnan(tensor).sum()
        if nan_count == 0:
            self._pass_count += 1
            return True
        prefix = f"[{name}] " if name else ""
        self.failures.append(AssertionFailure(
            message=f"{prefix}Found {nan_count} NaN values",
            details={"nan_count": int(nan_count)},
        ))
        return False

    def check_no_inf(self, tensor: np.ndarray, name: str = "") -> bool:
        """Inf 检查"""
        inf_count = np.isinf(tensor).sum()
        if inf_count == 0:
            self._pass_count += 1
            return True
        prefix = f"[{name}] " if name else ""
        self.failures.append(AssertionFailure(
            message=f"{prefix}Found {inf_count} Inf values",
            details={"inf_count": int(inf_count)},
        ))
        return False

    # ========== 算子精度检查 (aidevtools 集成) ==========

    def check_isclose(
        self,
        result: np.ndarray,
        golden: np.ndarray,
        atol: float = 1e-5,
        rtol: float = 1e-3,
        max_exceed_ratio: float = 0.0,
        name: str = ""
    ) -> bool:
        """IsClose 精度检查 (集成 aidevtools)"""
        from aidevtools.tools.compare import compare_isclose

        comparison = compare_isclose(
            golden=golden,
            result=result,
            atol=atol,
            rtol=rtol,
            max_exceed_ratio=max_exceed_ratio,
        )

        if comparison.passed:
            self._pass_count += 1
            return True

        prefix = f"[{name}] " if name else ""
        self.failures.append(AssertionFailure(
            message=(
                f"{prefix}IsClose failed: exceed_ratio={comparison.exceed_ratio:.4%} "
                f"(max: {max_exceed_ratio:.4%}), max_abs={comparison.max_abs_error:.2e}"
            ),
            details={
                "exceed_ratio": comparison.exceed_ratio,
                "exceed_count": comparison.exceed_count,
                "max_abs_error": comparison.max_abs_error,
            },
        ))
        return False

    def check_3col(
        self,
        op_name: str,
        op_id: int,
        result: np.ndarray,
        golden_pure: np.ndarray,
        golden_qnt: np.ndarray,
        thresholds = None,
    ) -> bool:
        """三列比对检查 (集成 aidevtools)"""
        from aidevtools.tools.compare import compare_3col, CompareThresholds

        thresholds = thresholds or CompareThresholds()
        comparison = compare_3col(
            op_name=op_name,
            op_id=op_id,
            result=result,
            golden_pure=golden_pure,
            golden_qnt=golden_qnt,
            thresholds=thresholds,
        )

        if comparison.status != "FAIL":
            self._pass_count += 1
            return True

        self.failures.append(AssertionFailure(
            message=(
                f"[{op_name}_{op_id}] 3-col compare FAIL: "
                f"qsnr={comparison.fuzzy_qnt.qsnr:.1f}dB"
            ),
            details={
                "status": comparison.status,
                "exact_mismatch": comparison.exact.mismatch_count,
                "fuzzy_pure_qsnr": comparison.fuzzy_pure.qsnr,
                "fuzzy_qnt_qsnr": comparison.fuzzy_qnt.qsnr,
            },
        ))
        return False

    # ========== 统计属性 ==========

    @property
    def pass_count(self) -> int:
        """通过数量"""
        return self._pass_count

    @property
    def fail_count(self) -> int:
        """失败数量"""
        return len(self.failures)

    @property
    def total_count(self) -> int:
        """总数量"""
        return self._pass_count + len(self.failures)

    def has_failures(self) -> bool:
        """是否有失败"""
        return len(self.failures) > 0

    def get_failures(self) -> List[AssertionFailure]:
        """获取所有失败"""
        return self.failures

    def get_failure_messages(self) -> List[str]:
        """获取所有失败消息"""
        return [f.message for f in self.failures]

    def summary(self) -> str:
        """生成汇总报告"""
        return (
            f"Soft Assert Summary: {self._pass_count} passed, "
            f"{len(self.failures)} failed"
        )
```

#### 2.2.3 使用示例

```python
"""软断言使用示例"""

from aitest.assertion.soft import SoftAssertContext
import numpy as np


# ============================================
# 示例1: 批量算子精度验证
# ============================================

def test_all_operators():
    """验证所有算子精度 - 收集所有失败"""

    ops = ["matmul_0", "layernorm_0", "softmax_0", "add_0"]

    with SoftAssertContext() as soft:
        for op_name in ops:
            dut = np.load(f"outputs/{op_name}_dut.npy")
            golden = np.load(f"outputs/{op_name}_golden.npy")

            soft.check_all_close(
                dut, golden,
                atol=1e-5, rtol=1e-3,
                name=op_name
            )

    # 退出时，如果有失败会汇总报告


# ============================================
# 示例2: 多指标验证
# ============================================

def test_model_metrics(result, golden):
    """验证多个指标 - 不因单个失败而停止"""

    with SoftAssertContext() as soft:
        # 形状检查
        soft.check_shape(result, list(golden.shape), "output")

        # NaN/Inf 检查
        soft.check_no_nan(result, "output")
        soft.check_no_inf(result, "output")

        # 精度检查
        soft.check_all_close(result, golden, atol=1e-4, name="precision")

        # 范围检查
        soft.check_in_range(result.max(), 0, 1, "max_value")

        print(soft.summary())  # "Soft Assert Summary: 4 passed, 1 failed"


# ============================================
# 示例3: 与三列比对集成
# ============================================

def test_transformer_with_3col():
    """Transformer 层三列比对 - 软断言"""

    from aidevtools.tools.compare import CompareThresholds

    thresholds = CompareThresholds(
        fuzzy_atol=1e-5,
        fuzzy_rtol=1e-3,
        fuzzy_min_qsnr=30.0,
    )

    ops = [
        ("MatMul", 0), ("MatMul", 1),
        ("LayerNorm", 0), ("Add", 0),
    ]

    with SoftAssertContext() as soft:
        for op_name, op_id in ops:
            dut = np.load(f"outputs/{op_name}_{op_id}_dut.npy")
            pure = np.load(f"outputs/{op_name}_{op_id}_pure.npy")
            qnt = np.load(f"outputs/{op_name}_{op_id}_qnt.npy")

            soft.check_3col(
                op_name=op_name,
                op_id=op_id,
                result=dut,
                golden_pure=pure,
                golden_qnt=qnt,
                thresholds=thresholds,
            )


# ============================================
# 示例4: 不抛出异常模式 (手动处理)
# ============================================

def test_with_manual_handling():
    """手动处理失败 (不抛出异常)"""

    with SoftAssertContext(raise_on_exit=False) as soft:
        soft.check(True, "Always pass")
        soft.check(False, "Will fail but continue")
        soft.check(True, "Another pass")

    # 手动检查结果
    if soft.has_failures():
        print(f"Found {soft.fail_count} failures:")
        for msg in soft.get_failure_messages():
            print(f"  - {msg}")
    else:
        print("All checks passed!")
```

### 2.3 实现示例

```python
# assertion/builder.py

from typing import Any, TypeVar, Generic
import numpy as np

T = TypeVar('T')


class AssertionBuilder(Generic[T]):
    """流式断言构建器"""

    def __init__(self, actual: T, soft_mode: bool = False):
        self._actual = actual
        self._soft_mode = soft_mode
        self._failures = []

    def _check(self, condition: bool, message: str) -> 'AssertionBuilder':
        """执行检查"""
        if not condition:
            if self._soft_mode:
                self._failures.append(AssertionError(message))
            else:
                raise AssertionError(message)
        return self

    # ========== Basic Assertions ==========

    def equals(self, expected: Any) -> 'AssertionBuilder':
        """相等断言"""
        return self._check(
            self._actual == expected,
            f"Expected {expected}, but got {self._actual}"
        )

    def is_true(self) -> 'AssertionBuilder':
        """真值断言"""
        return self._check(
            bool(self._actual),
            f"Expected True, but got {self._actual}"
        )

    def is_not_none(self) -> 'AssertionBuilder':
        """非空断言"""
        return self._check(
            self._actual is not None,
            "Expected non-None value"
        )

    def is_instance_of(self, cls: type) -> 'AssertionBuilder':
        """类型断言"""
        return self._check(
            isinstance(self._actual, cls),
            f"Expected instance of {cls.__name__}, got {type(self._actual).__name__}"
        )

    # ========== Numeric Assertions ==========

    def is_close_to(self, expected: float, atol: float = 1e-8, rtol: float = 1e-5) -> 'AssertionBuilder':
        """近似相等"""
        diff = abs(self._actual - expected)
        threshold = atol + rtol * abs(expected)
        return self._check(
            diff <= threshold,
            f"Expected {expected} (±{threshold}), got {self._actual}, diff={diff}"
        )

    def is_in_range(self, min_val: float, max_val: float) -> 'AssertionBuilder':
        """范围断言"""
        return self._check(
            min_val <= self._actual <= max_val,
            f"Expected value in [{min_val}, {max_val}], got {self._actual}"
        )

    def is_greater_than(self, value: float) -> 'AssertionBuilder':
        """大于断言"""
        return self._check(
            self._actual > value,
            f"Expected > {value}, got {self._actual}"
        )

    def is_less_than(self, value: float) -> 'AssertionBuilder':
        """小于断言"""
        return self._check(
            self._actual < value,
            f"Expected < {value}, got {self._actual}"
        )

    # ========== Tensor Assertions ==========

    def has_shape(self, expected_shape: list) -> 'AssertionBuilder':
        """形状断言"""
        actual_shape = list(self._actual.shape)
        return self._check(
            actual_shape == expected_shape,
            f"Expected shape {expected_shape}, got {actual_shape}"
        )

    def has_dtype(self, expected_dtype) -> 'AssertionBuilder':
        """数据类型断言"""
        return self._check(
            self._actual.dtype == expected_dtype,
            f"Expected dtype {expected_dtype}, got {self._actual.dtype}"
        )

    def has_no_nan(self) -> 'AssertionBuilder':
        """无NaN断言"""
        has_nan = np.isnan(self._actual).any()
        return self._check(
            not has_nan,
            f"Found NaN values in tensor"
        )

    def has_no_inf(self) -> 'AssertionBuilder':
        """无Inf断言"""
        has_inf = np.isinf(self._actual).any()
        return self._check(
            not has_inf,
            f"Found Inf values in tensor"
        )

    def all_close_to(self, expected: np.ndarray, atol: float = 1e-8, rtol: float = 1e-5) -> 'AssertionBuilder':
        """元素级近似"""
        close = np.allclose(self._actual, expected, atol=atol, rtol=rtol)
        if not close:
            diff = np.abs(self._actual - expected)
            max_diff = diff.max()
            max_idx = np.unravel_index(diff.argmax(), diff.shape)
            msg = f"Arrays not close. Max diff: {max_diff} at index {max_idx}"
            return self._check(False, msg)
        return self


# assertion/tensor.py

class TensorAssertions:
    """张量专用断言"""

    @staticmethod
    def assert_shape(tensor, expected_shape):
        """断言张量形状"""
        actual_shape = list(tensor.shape)
        if actual_shape != expected_shape:
            raise AssertionError(f"Shape mismatch: {actual_shape} != {expected_shape}")

    @staticmethod
    def assert_all_close(actual, expected, atol=1e-8, rtol=1e-5):
        """断言元素级近似"""
        if not np.allclose(actual, expected, atol=atol, rtol=rtol):
            diff = np.abs(actual - expected)
            raise AssertionError(
                f"Arrays not close. Max diff: {diff.max()}, "
                f"mean diff: {diff.mean()}"
            )

    @staticmethod
    def assert_gradients_valid(model):
        """断言梯度有效"""
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    raise AssertionError(f"NaN gradient in {name}")
                if torch.isinf(param.grad).any():
                    raise AssertionError(f"Inf gradient in {name}")


# assertion/snapshot.py

import json
from pathlib import Path
from typing import Any, Dict
import hashlib


class SnapshotManager:
    """快照管理器"""

    def __init__(self, snapshot_dir: Path):
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, data: Any) -> Path:
        """保存快照"""
        path = self.snapshot_dir / f"{name}.json"
        with open(path, 'w') as f:
            json.dump(self._serialize(data), f, indent=2)
        return path

    def load(self, name: str) -> Any:
        """加载快照"""
        path = self.snapshot_dir / f"{name}.json"
        with open(path) as f:
            return self._deserialize(json.load(f))

    def compare(self, name: str, current: Any) -> 'SnapshotDiff':
        """对比快照"""
        stored = self.load(name)
        return self._compute_diff(stored, current)

    def update(self, name: str, data: Any) -> None:
        """更新快照"""
        self.save(name, data)

    def _serialize(self, data: Any) -> Dict:
        """序列化数据"""
        if isinstance(data, np.ndarray):
            return {"__type__": "ndarray", "data": data.tolist(), "dtype": str(data.dtype)}
        return data

    def _deserialize(self, data: Dict) -> Any:
        """反序列化数据"""
        if isinstance(data, dict) and data.get("__type__") == "ndarray":
            return np.array(data["data"], dtype=data["dtype"])
        return data

    def _compute_diff(self, stored: Any, current: Any) -> SnapshotDiff:
        """计算差异"""
        # 简化实现
        if isinstance(stored, np.ndarray) and isinstance(current, np.ndarray):
            equal = np.allclose(stored, current)
            return SnapshotDiff(
                added=[], removed=[],
                changed=[] if equal else [("data", stored, current)],
                unchanged_count=1 if equal else 0
            )
        return SnapshotDiff(
            added=[], removed=[],
            changed=[] if stored == current else [("value", stored, current)],
            unchanged_count=1 if stored == current else 0
        )


# assertion/operator/precision.py

"""
算子精度验证模块 - 封装 aidevtools.tools.compare 的三列比对功能

三列比对机制:
- exact: 精确比对 (bit级或指定误差)
- fuzzy_pure: 纯 fp32 模糊比对
- fuzzy_qnt: 量化感知模糊比对

状态判定:
- PERFECT: exact 通过
- PASS: exact 不过，但 fuzzy_qnt 通过
- QUANT_ISSUE: fuzzy_pure 通过，fuzzy_qnt 不过 (量化问题)
- FAIL: 都不过
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass

# 从 aidevtools 导入核心比对函数
from aidevtools.tools.compare import (
    compare_3col,
    compare_isclose,
    compare_exact,
    compare_full,
    CompareThresholds,
    FullCompareResult,
    IsCloseResult,
    print_compare_table,
)


class OpPrecisionAssertion:
    """算子精度断言"""

    def __init__(self, thresholds: CompareThresholds = None):
        self.thresholds = thresholds or CompareThresholds()

    def assert_3col(
        self,
        op_name: str,
        op_id: int,
        result: np.ndarray,
        golden_pure: np.ndarray,
        golden_qnt: np.ndarray,
    ) -> FullCompareResult:
        """
        三列比对断言

        Args:
            op_name: 算子名称
            op_id: 算子 ID
            result: DUT 输出
            golden_pure: 纯 fp32 golden
            golden_qnt: 量化感知 golden

        Returns:
            FullCompareResult

        Raises:
            AssertionError: 当状态为 FAIL 时
        """
        comparison = compare_3col(
            op_name=op_name,
            op_id=op_id,
            result=result,
            golden_pure=golden_pure,
            golden_qnt=golden_qnt,
            thresholds=self.thresholds,
        )

        if comparison.status == "FAIL":
            raise AssertionError(
                f"Operator {op_name}_{op_id} precision check FAILED\n"
                f"  exact: mismatch={comparison.exact.mismatch_count}\n"
                f"  fuzzy_pure: max_abs={comparison.fuzzy_pure.max_abs:.2e}, "
                f"qsnr={comparison.fuzzy_pure.qsnr:.1f}dB\n"
                f"  fuzzy_qnt: max_abs={comparison.fuzzy_qnt.max_abs:.2e}, "
                f"qsnr={comparison.fuzzy_qnt.qsnr:.1f}dB"
            )

        return comparison

    def assert_perfect(
        self,
        op_name: str,
        result: np.ndarray,
        golden: np.ndarray,
    ):
        """断言 bit 级精确匹配"""
        exact_result = compare_exact(golden, result, max_abs=0.0, max_count=0)

        if not exact_result.passed:
            raise AssertionError(
                f"Operator {op_name} is not bit-exact\n"
                f"  mismatch_count: {exact_result.mismatch_count}\n"
                f"  first_diff_offset: {exact_result.first_diff_offset}\n"
                f"  max_abs: {exact_result.max_abs:.2e}"
            )

    def assert_isclose(
        self,
        result: np.ndarray,
        golden: np.ndarray,
        atol: float = 1e-5,
        rtol: float = 1e-3,
        max_exceed_ratio: float = 0.0,
        name: str = "",
    ) -> IsCloseResult:
        """
        IsClose 断言 - 逐元素误差检查

        判断条件: |result - golden| <= atol + rtol * |golden|

        Args:
            result: DUT 输出
            golden: 参考数据
            atol: 绝对误差门限
            rtol: 相对误差门限
            max_exceed_ratio: 允许的最大超限比例

        Raises:
            AssertionError: 当超限比例超过阈值时
        """
        isclose_result = compare_isclose(
            golden=golden,
            result=result,
            atol=atol,
            rtol=rtol,
            max_exceed_ratio=max_exceed_ratio,
        )

        if not isclose_result.passed:
            name_str = f"[{name}] " if name else ""
            raise AssertionError(
                f"{name_str}IsClose check FAILED\n"
                f"  exceed_ratio: {isclose_result.exceed_ratio:.4%} > "
                f"{max_exceed_ratio:.4%}\n"
                f"  exceed_count: {isclose_result.exceed_count} / "
                f"{isclose_result.total_elements}\n"
                f"  max_abs_error: {isclose_result.max_abs_error:.6e}\n"
                f"  max_rel_error: {isclose_result.max_rel_error:.6e}"
            )

        return isclose_result

    def assert_qsnr(
        self,
        result: np.ndarray,
        golden: np.ndarray,
        min_qsnr_db: float = 30.0,
        name: str = "",
    ):
        """断言 QSNR 不低于阈值"""
        from aidevtools.tools.compare import calc_qsnr

        qsnr = calc_qsnr(golden, result)

        if qsnr < min_qsnr_db:
            name_str = f"[{name}] " if name else ""
            raise AssertionError(
                f"{name_str}QSNR check FAILED\n"
                f"  QSNR: {qsnr:.1f} dB < {min_qsnr_db:.1f} dB"
            )

    def assert_cosine(
        self,
        result: np.ndarray,
        golden: np.ndarray,
        min_cosine: float = 0.999,
        name: str = "",
    ):
        """断言余弦相似度不低于阈值"""
        from aidevtools.tools.compare import calc_cosine

        cosine = calc_cosine(golden, result)

        if cosine < min_cosine:
            name_str = f"[{name}] " if name else ""
            raise AssertionError(
                f"{name_str}Cosine similarity check FAILED\n"
                f"  Cosine: {cosine:.6f} < {min_cosine:.6f}"
            )


def assert_batch_precision(
    results: List[FullCompareResult],
    allow_quant_issue: bool = True,
) -> None:
    """
    批量精度断言

    Args:
        results: 比对结果列表
        allow_quant_issue: 是否允许量化问题 (QUANT_ISSUE 状态)

    Raises:
        AssertionError: 当存在 FAIL 状态，或不允许量化问题时存在 QUANT_ISSUE
    """
    fail_count = sum(1 for r in results if r.status == "FAIL")
    quant_issue_count = sum(1 for r in results if r.status == "QUANT_ISSUE")

    if fail_count > 0:
        failed_ops = [f"{r.op_name}_{r.op_id}" for r in results if r.status == "FAIL"]
        raise AssertionError(
            f"Batch precision check FAILED: {fail_count} operators failed\n"
            f"  Failed ops: {', '.join(failed_ops)}"
        )

    if not allow_quant_issue and quant_issue_count > 0:
        quant_ops = [f"{r.op_name}_{r.op_id}" for r in results if r.status == "QUANT_ISSUE"]
        raise AssertionError(
            f"Batch precision check FAILED: {quant_issue_count} operators have quant issues\n"
            f"  Quant issue ops: {', '.join(quant_ops)}"
        )

    # 打印汇总表格
    print_compare_table(results)
```

---

## 3. 场景视图

### 3.1 使用示例

```python
from aitest import assert_that, assert_accuracy, assert_latency

# 基础断言
assert_that(result).equals(expected)
assert_that(value).is_in_range(0, 1)

# 张量断言
assert_that(output).has_shape([batch_size, num_classes])
assert_that(output).has_no_nan().has_no_inf()
assert_that(output).all_close_to(expected, atol=1e-5)

# 分类断言
assert_that(predictions).matches_class(labels)
assert_that(predictions).has_top_k_accuracy(k=5, min_accuracy=0.9)

# 性能断言
assert_latency(latency.p99).less_than_ms(10)
assert_accuracy(metrics.f1_score).greater_than(0.9)

# 快照对比
snapshot = SnapshotManager("snapshots/")
snapshot.compare("model_output_v1", current_output)
```

### 3.2 算子精度验证示例 (aidevtools集成)

```python
"""算子精度验证用例 - 使用 aidevtools 三列比对"""

import numpy as np
from aitest.assertion.operator import OpPrecisionAssertion, assert_batch_precision
from aidevtools.tools.compare import CompareThresholds, compare_3col


# ============================================
# 用例1: 单算子三列比对
# ============================================
def test_matmul_precision():
    """测试 MatMul 算子精度"""

    # 准备测试数据
    dut_output = np.load("outputs/matmul_0_dut.npy")       # DUT 输出
    golden_pure = np.load("outputs/matmul_0_pure.npy")    # 纯 fp32 golden
    golden_qnt = np.load("outputs/matmul_0_qnt.npy")      # 量化感知 golden

    # 配置比对阈值
    thresholds = CompareThresholds(
        exact_max_abs=0.0,      # 精确比对: bit 级
        exact_max_count=0,
        fuzzy_atol=1e-5,        # 模糊比对: 绝对误差
        fuzzy_rtol=1e-3,        # 模糊比对: 相对误差
        fuzzy_min_qsnr=30.0,    # 最小 QSNR (dB)
        fuzzy_min_cosine=0.999, # 最小余弦相似度
    )

    # 执行三列比对
    assertion = OpPrecisionAssertion(thresholds)
    result = assertion.assert_3col(
        op_name="MatMul",
        op_id=0,
        result=dut_output,
        golden_pure=golden_pure,
        golden_qnt=golden_qnt,
    )

    # 打印结果
    print(f"Status: {result.status}")  # PERFECT / PASS / QUANT_ISSUE / FAIL
    print(f"QSNR: {result.fuzzy_qnt.qsnr:.1f} dB")
    print(f"Cosine: {result.fuzzy_qnt.cosine:.6f}")


# ============================================
# 用例2: IsClose 断言
# ============================================
def test_softmax_isclose():
    """测试 Softmax 算子 IsClose 精度"""

    dut_output = np.load("outputs/softmax_0_dut.npy")
    golden = np.load("outputs/softmax_0_golden.npy")

    assertion = OpPrecisionAssertion()
    result = assertion.assert_isclose(
        result=dut_output,
        golden=golden,
        atol=1e-4,              # 绝对误差门限
        rtol=1e-2,              # 相对误差门限
        max_exceed_ratio=0.01,  # 允许 1% 元素超限
        name="Softmax_0",
    )

    print(f"Exceed ratio: {result.exceed_ratio:.4%}")
    print(f"Max abs error: {result.max_abs_error:.6e}")


# ============================================
# 用例3: 批量算子精度验证
# ============================================
def test_transformer_layer_precision():
    """测试 Transformer 层所有算子精度"""

    # 算子列表
    ops = [
        ("LayerNorm", 0), ("Linear", 0), ("Linear", 1), ("Linear", 2),
        ("Attention", 0), ("Linear", 3), ("Add", 0),
        ("LayerNorm", 1), ("Linear", 4), ("GELU", 0), ("Linear", 5), ("Add", 1),
    ]

    results = []
    for op_name, op_id in ops:
        dut = np.load(f"outputs/{op_name}_{op_id}_dut.npy")
        pure = np.load(f"outputs/{op_name}_{op_id}_pure.npy")
        qnt = np.load(f"outputs/{op_name}_{op_id}_qnt.npy")

        result = compare_3col(
            op_name=op_name,
            op_id=op_id,
            result=dut,
            golden_pure=pure,
            golden_qnt=qnt,
        )
        results.append(result)

    # 批量断言 (允许 QUANT_ISSUE，但不允许 FAIL)
    assert_batch_precision(results, allow_quant_issue=True)


# ============================================
# 用例4: QSNR 和 Cosine 断言
# ============================================
def test_conv_qsnr_cosine():
    """测试 Conv 算子 QSNR 和余弦相似度"""

    dut_output = np.load("outputs/conv_0_dut.npy")
    golden = np.load("outputs/conv_0_golden.npy")

    assertion = OpPrecisionAssertion()

    # QSNR 断言: 不低于 35 dB
    assertion.assert_qsnr(dut_output, golden, min_qsnr_db=35.0, name="Conv_0")

    # 余弦相似度断言: 不低于 0.9999
    assertion.assert_cosine(dut_output, golden, min_cosine=0.9999, name="Conv_0")


# ============================================
# 用例5: Pytest 集成
# ============================================
import pytest

class TestOpPrecision:
    """算子精度测试套件"""

    @pytest.fixture
    def assertion(self):
        return OpPrecisionAssertion(
            CompareThresholds(fuzzy_min_qsnr=30.0, fuzzy_min_cosine=0.999)
        )

    @pytest.mark.parametrize("op_name,op_id", [
        ("MatMul", 0), ("MatMul", 1), ("Add", 0), ("LayerNorm", 0),
    ])
    def test_op_precision(self, assertion, op_name, op_id):
        """参数化测试多个算子"""

        dut = np.load(f"outputs/{op_name}_{op_id}_dut.npy")
        pure = np.load(f"outputs/{op_name}_{op_id}_pure.npy")
        qnt = np.load(f"outputs/{op_name}_{op_id}_qnt.npy")

        result = assertion.assert_3col(
            op_name=op_name,
            op_id=op_id,
            result=dut,
            golden_pure=pure,
            golden_qnt=qnt,
        )

        assert result.status in ("PERFECT", "PASS")
```

### 3.3 需求追溯

| 需求ID | 实现类/方法 | 测试用例 | aidevtools 集成 |
|--------|-------------|----------|-----------------|
| ASSERT-001 | `basic.py` | test_basic_assertions | - |
| ASSERT-002 | `numeric.py` | test_numeric_assertions | - |
| ASSERT-003 | `tensor.py`, `operator/precision.py` | test_tensor_assertions | `compare_isclose`, `compare_full` |
| ASSERT-003-03 | `OpPrecisionAssertion.assert_isclose` | test_isclose | `compare_isclose` |
| ASSERT-003-04 | `OpPrecisionAssertion.assert_3col` | test_nan_inf | `compare_3col` |
| ASSERT-004 | `classification.py` | test_classification_assertions | - |
| ASSERT-005 | `detection.py` | test_detection_assertions | - |
| ASSERT-006 | `text.py` | test_text_assertions | - |
| ASSERT-007 | `performance.py` | test_performance_assertions | - |
| ASSERT-008 | `metric.py` | test_metric_assertions | - |
| ASSERT-009 | `snapshot.py` | test_snapshot | - |
| ASSERT-010 | `custom.py` | test_custom_assertions | - |

### 3.4 算子精度验证需求追溯

| 功能 | aidevtools 函数 | aitestframework 封装 | 说明 |
|------|-----------------|---------------------|------|
| 三列比对 | `compare_3col()` | `OpPrecisionAssertion.assert_3col()` | exact + fuzzy_pure + fuzzy_qnt |
| 精确比对 | `compare_exact()` | `OpPrecisionAssertion.assert_perfect()` | bit 级或指定误差 |
| IsClose比对 | `compare_isclose()` | `OpPrecisionAssertion.assert_isclose()` | 逐元素误差检查 |
| 完整比对 | `compare_full()` | 内部使用 | max_abs, qsnr, cosine |
| QSNR断言 | `calc_qsnr()` | `OpPrecisionAssertion.assert_qsnr()` | 信噪比 (dB) |
| 余弦相似度 | `calc_cosine()` | `OpPrecisionAssertion.assert_cosine()` | [-1, 1] |
| 批量断言 | `print_compare_table()` | `assert_batch_precision()` | 汇总表格输出 |

### 3.5 比对状态说明

| 状态 | 条件 | 含义 |
|------|------|------|
| **PERFECT** | exact 通过 | bit 级精确匹配 |
| **PASS** | exact 不过，fuzzy_qnt 通过 | 数值误差在可接受范围内 |
| **QUANT_ISSUE** | fuzzy_pure 通过，fuzzy_qnt 不过 | 量化精度问题，需检查量化配置 |
| **FAIL** | 都不过 | 算子实现存在问题 |

---

*本文档为断言与验证模块的详细设计，基于4+1视图方法，集成 aidevtools.tools.compare 提供的精度比对能力。*
