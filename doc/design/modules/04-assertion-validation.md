# 断言与验证模块详细设计 (Assertion & Validation)

## 模块概述

| 属性 | 值 |
|------|-----|
| **模块ID** | ASSERT |
| **模块名称** | 断言与验证 |
| **职责** | 测试结果的断言、比较和验证机制 |
| **需求覆盖** | ASSERT-001 ~ ASSERT-010 |

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
└── matchers.py          # 匹配器
```

### 2.2 实现示例

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

### 3.2 需求追溯

| 需求ID | 实现类/方法 | 测试用例 |
|--------|-------------|----------|
| ASSERT-001 | `basic.py` | test_basic_assertions |
| ASSERT-002 | `numeric.py` | test_numeric_assertions |
| ASSERT-003 | `tensor.py` | test_tensor_assertions |
| ASSERT-004 | `classification.py` | test_classification_assertions |
| ASSERT-005 | `detection.py` | test_detection_assertions |
| ASSERT-006 | `text.py` | test_text_assertions |
| ASSERT-007 | `performance.py` | test_performance_assertions |
| ASSERT-008 | `metric.py` | test_metric_assertions |
| ASSERT-009 | `snapshot.py` | test_snapshot |
| ASSERT-010 | `custom.py` | test_custom_assertions |

---

*本文档为断言与验证模块的详细设计，基于4+1视图方法。*
