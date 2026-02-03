# 性能测试模块设计 (Performance Testing Module)

## 模块概述

| 属性 | 说明 |
|------|------|
| **模块ID** | PERF |
| **模块名称** | 性能测试 |
| **英文名称** | Performance Testing |
| **分类** | 性能需求 |
| **职责** | 算子性能分析、Cost Model集成、性能回归测试 |
| **关联需求** | MODEL-005, MODEL-006 |
| **外部依赖** | aidevtools.analysis (Cost Model) |

### 模块定位

性能测试模块作为AI测试框架的性能验证层，集成 `aidevtools.analysis` 提供的 Cost Model 能力，提供：
- **算子级性能分析**：集成 PaperAnalyzer 进行时延、带宽、算力分析
- **性能回归测试**：跨版本性能对比与回归检测
- **性能断言**：时延、吞吐量、利用率等指标的阈值断言
- **芯片对比测试**（可选，低优先级）：多芯片性能对比

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Performance Testing Module                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      Test Framework Layer                            │    │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐        │    │
│  │  │   Perf    │  │   Perf    │  │  Version  │  │   Perf    │        │    │
│  │  │ TestCase  │  │ Assertion │  │  Compare  │  │  Report   │        │    │
│  │  └───────────┘  └───────────┘  └───────────┘  └───────────┘        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    aidevtools.analysis (Cost Model)                  │    │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐        │    │
│  │  │  Paper    │  │   Chip    │  │   Pass    │  │   Model   │        │    │
│  │  │ Analyzer  │  │   Spec    │  │   Chain   │  │  Presets  │        │    │
│  │  └───────────┘  └───────────┘  └───────────┘  └───────────┘        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# 一、逻辑视图 (Logical View)

## 1. 核心类设计

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Performance Testing Class Diagram                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      PerfTestRunner                                  │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  - analyzer: PaperAnalyzer        # from aidevtools.analysis        │    │
│  │  - chip_spec: ChipSpec            # 芯片规格                         │    │
│  │  - baseline_store: BaselineStore  # 基线存储                         │    │
│  │  - config: PerfTestConfig                                           │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  + run_latency_test(profiles) -> LatencyResult                      │    │
│  │  + run_throughput_test(model, config) -> ThroughputResult           │    │
│  │  + run_roofline_analysis(profiles) -> RooflineResult                │    │
│  │  + compare_with_baseline(result, baseline) -> ComparisonResult      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│         ┌──────────────────────────┴───────────────────────────┐           │
│         ▼                                                       ▼           │
│  ┌──────────────────────────┐                ┌──────────────────────────┐   │
│  │     PerfAssertion        │                │     BaselineStore        │   │
│  ├──────────────────────────┤                ├──────────────────────────┤   │
│  │ + assert_latency_us()    │                │ - storage_path: str      │   │
│  │ + assert_throughput()    │                │ - baselines: Dict        │   │
│  │ + assert_bandwidth()     │                ├──────────────────────────┤   │
│  │ + assert_utilization()   │                │ + save_baseline()        │   │
│  │ + assert_no_regression() │                │ + load_baseline()        │   │
│  └──────────────────────────┘                │ + compare_versions()     │   │
│                                              └──────────────────────────┘   │
│                                                                             │
│  ┌──────────────────────────┐                ┌──────────────────────────┐   │
│  │  ChipTestSuite (可选)    │                │   PerfReportGenerator    │   │
│  ├──────────────────────────┤                ├──────────────────────────┤   │
│  │ - chips: List[ChipSpec]  │                │ - results: List          │   │
│  │ - profiles: List         │                │ - template: Template     │   │
│  ├──────────────────────────┤                ├──────────────────────────┤   │
│  │ + run_on_all_chips()     │                │ + generate_html()        │   │
│  │ + compare_chips()        │                │ + generate_xlsx()        │   │
│  │ + find_best_chip()       │                │ + generate_gantt()       │   │
│  └──────────────────────────┘                └──────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. 与 aidevtools.analysis 的集成关系

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Integration with aidevtools.analysis                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  aitestframework (性能测试模块)              aidevtools.analysis             │
│  ┌──────────────────────────┐              ┌──────────────────────────┐     │
│  │     PerfTestRunner       │   imports    │     PaperAnalyzer        │     │
│  │                          │─────────────►│                          │     │
│  │  - run_latency_test()    │              │  - add_profile()         │     │
│  │  - run_roofline_test()   │              │  - analyze()             │     │
│  └──────────────────────────┘              │  - get_summary()         │     │
│                                            └──────────────────────────┘     │
│  ┌──────────────────────────┐              ┌──────────────────────────┐     │
│  │     ChipTestSuite        │   imports    │     ChipSpec             │     │
│  │                          │─────────────►│     load_chip_spec()     │     │
│  │  - test_on_npu_910()     │              │     list_chips()         │     │
│  │  - test_on_gpu_a100()    │              └──────────────────────────┘     │
│  └──────────────────────────┘                                               │
│                                            ┌──────────────────────────┐     │
│  ┌──────────────────────────┐   imports    │     Model Presets        │     │
│  │     ModelPerfTest        │─────────────►│     from_preset()        │     │
│  │                          │              │     transformer_layer()  │     │
│  │  - test_llama_7b()       │              │     llama_layer()        │     │
│  │  - test_bert_base()      │              └──────────────────────────┘     │
│  └──────────────────────────┘                                               │
│                                            ┌──────────────────────────┐     │
│  ┌──────────────────────────┐   imports    │     Pass Chain           │     │
│  │     BottleneckTest       │─────────────►│     RooflinePass         │     │
│  │                          │              │     BandwidthPass        │     │
│  │  - detect_bottleneck()   │              │     PrefetchPass         │     │
│  │  - analyze_memory()      │              └──────────────────────────┘     │
│  └──────────────────────────┘                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 3. 核心数据结构

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

# 复用 aidevtools.analysis 的数据结构
from aidevtools.analysis import (
    LatencyResult,
    LatencyBreakdown,
    AnalysisSummary,
    ChipSpec,
    OpProfile,
    GanttData,
)


@dataclass
class PerfTestConfig:
    """性能测试配置"""
    # 芯片配置
    chip: str = "npu_910"
    chips_to_compare: List[str] = field(default_factory=lambda: ["npu_910", "gpu_a100"])

    # 测试配置
    warmup_runs: int = 3
    test_runs: int = 10
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8, 16])

    # 回归检测配置
    regression_threshold_percent: float = 10.0
    enable_baseline_compare: bool = True

    # 报告配置
    generate_gantt: bool = True
    export_xlsx: bool = True


@dataclass
class PerfTestResult:
    """性能测试结果"""
    # 基本信息
    test_name: str
    chip_spec: ChipSpec
    timestamp: datetime

    # 来自 aidevtools.analysis 的结果
    latency_result: LatencyResult
    summary: AnalysisSummary

    # 测试框架层面的信息
    test_config: PerfTestConfig
    assertion_results: List['AssertionResult']

    # 与基线对比
    baseline_comparison: Optional['BaselineComparison'] = None


@dataclass
class AssertionResult:
    """断言结果"""
    assertion_type: str       # "latency", "throughput", "bandwidth", "utilization"
    metric_name: str
    expected_value: float
    actual_value: float
    threshold: float
    passed: bool
    message: str


@dataclass
class BaselineComparison:
    """基线对比结果"""
    baseline_version: str
    current_version: str

    # 指标对比
    latency_diff_percent: float
    throughput_diff_percent: float
    bandwidth_diff_percent: float

    # 回归检测
    has_regression: bool
    regression_details: List[str]

    # 改进项
    improvements: List[str]


@dataclass
class ChipComparisonResult:
    """芯片对比结果"""
    model_name: str
    batch_size: int

    # 各芯片结果
    chip_results: Dict[str, PerfTestResult]

    # 对比分析
    fastest_chip: str
    latency_ranking: List[tuple]  # [(chip, latency_us), ...]
    throughput_ranking: List[tuple]
    efficiency_ranking: List[tuple]
```

## 4. 核心接口定义

```python
from typing import Protocol, List


class IPerfTestRunner(Protocol):
    """性能测试运行器接口"""

    def run_latency_test(
        self,
        profiles: List[OpProfile],
        chip: str = "npu_910"
    ) -> PerfTestResult:
        """运行时延测试"""
        ...

    def run_throughput_test(
        self,
        model_preset: str,
        batch_sizes: List[int],
        chip: str = "npu_910"
    ) -> List[PerfTestResult]:
        """运行吞吐量测试"""
        ...

    def run_roofline_analysis(
        self,
        profiles: List[OpProfile],
        chip: str = "npu_910"
    ) -> RooflineResult:
        """运行 Roofline 分析"""
        ...


class IPerfAssertion(Protocol):
    """性能断言接口"""

    def assert_latency_us(
        self,
        result: PerfTestResult,
        max_latency_us: float
    ) -> AssertionResult:
        """断言时延不超过阈值"""
        ...

    def assert_throughput_tflops(
        self,
        result: PerfTestResult,
        min_tflops: float
    ) -> AssertionResult:
        """断言吞吐量不低于阈值"""
        ...

    def assert_no_regression(
        self,
        current: PerfTestResult,
        baseline: PerfTestResult,
        threshold_percent: float = 10.0
    ) -> AssertionResult:
        """断言无性能回归"""
        ...


class IBaselineStore(Protocol):
    """基线存储接口"""

    def save_baseline(
        self,
        name: str,
        result: PerfTestResult,
        version: str
    ) -> None:
        """保存基线"""
        ...

    def load_baseline(
        self,
        name: str,
        version: str = "latest"
    ) -> PerfTestResult:
        """加载基线"""
        ...

    def compare_versions(
        self,
        name: str,
        version_a: str,
        version_b: str
    ) -> BaselineComparison:
        """对比版本"""
        ...
```

---

# 二、进程视图 (Process View)

## 1. 性能测试执行流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   Performance Test Execution Flow                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐   │
│  │ Prepare │───►│ Profile │───►│ Analyze │───►│ Assert  │───►│ Report  │   │
│  │ Profiles│    │ Collect │    │(aidevtools)│  │ & Compare│   │ Generate│   │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘   │
│                                                                             │
│  详细流程:                                                                   │
│                                                                             │
│  1. Prepare Profiles                                                        │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  # 方式1: 使用预定义模型                                         │     │
│     │  from aidevtools.analysis import from_preset                    │     │
│     │  profiles = from_preset("llama-7b", batch=4)                    │     │
│     │                                                                  │     │
│     │  # 方式2: 自定义模型                                             │     │
│     │  from aidevtools.analysis import transformer_layer              │     │
│     │  profiles = transformer_layer(batch=4, seq=512, hidden=768)     │     │
│     │                                                                  │     │
│     │  # 方式3: 从实际推理收集                                         │     │
│     │  profiles = collect_profiles_from_inference(model, input)       │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  2. Run Analysis (via aidevtools.analysis.PaperAnalyzer)                    │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  from aidevtools.analysis import PaperAnalyzer, load_chip_spec  │     │
│     │                                                                  │     │
│     │  analyzer = PaperAnalyzer(chip="npu_910")                       │     │
│     │  analyzer.add_profiles(profiles)                                │     │
│     │  result = analyzer.analyze()                                    │     │
│     │                                                                  │     │
│     │  # result 包含:                                                  │     │
│     │  # - breakdowns: 每个算子的时延分解                              │     │
│     │  # - summary: 汇总指标 (总时延、瓶颈统计、吞吐量等)              │     │
│     │  # - gantt_data: Gantt 图数据                                   │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  3. Assert & Compare                                                        │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  # 断言时延                                                      │     │
│     │  assert_latency_us(result, max_us=1000)                         │     │
│     │                                                                  │     │
│     │  # 断言吞吐量                                                    │     │
│     │  assert_throughput_tflops(result, min_tflops=100)               │     │
│     │                                                                  │     │
│     │  # 与基线对比                                                    │     │
│     │  baseline = load_baseline("llama-7b", version="v1.0")           │     │
│     │  comparison = compare_with_baseline(result, baseline)           │     │
│     │  assert_no_regression(comparison, threshold=10%)                │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  4. Generate Report                                                         │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  from aidevtools.analysis import export_xlsx                    │     │
│     │  export_xlsx(result, "perf_report.xlsx")                        │     │
│     │                                                                  │     │
│     │  # 生成 Gantt 图                                                 │     │
│     │  generate_gantt_chart(result.gantt_data, "gantt.html")          │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. 芯片对比测试流程（可选，低优先级）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              Multi-Chip Comparison Test Flow (Optional)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                         ┌─────────────────────┐                             │
│                         │   Test Profiles     │                             │
│                         │  (from_preset)      │                             │
│                         └──────────┬──────────┘                             │
│                                    │                                        │
│              ┌─────────────────────┼─────────────────────┐                  │
│              ▼                     ▼                     ▼                  │
│       ┌─────────────┐       ┌─────────────┐       ┌─────────────┐          │
│       │  NPU 310    │       │  NPU 910    │       │  GPU A100   │          │
│       │ PaperAnalyzer│      │ PaperAnalyzer│      │ PaperAnalyzer│          │
│       └──────┬──────┘       └──────┬──────┘       └──────┬──────┘          │
│              │                     │                     │                  │
│              ▼                     ▼                     ▼                  │
│       ┌─────────────┐       ┌─────────────┐       ┌─────────────┐          │
│       │LatencyResult│       │LatencyResult│       │LatencyResult│          │
│       │  310: 5.2ms │       │  910: 1.1ms │       │ A100: 0.8ms │          │
│       └──────┬──────┘       └──────┬──────┘       └──────┬──────┘          │
│              │                     │                     │                  │
│              └─────────────────────┼─────────────────────┘                  │
│                                    ▼                                        │
│                         ┌─────────────────────┐                             │
│                         │  Comparison Report  │                             │
│                         │                     │                             │
│                         │  Ranking:           │                             │
│                         │  1. A100 (0.8ms)    │                             │
│                         │  2. 910  (1.1ms)    │                             │
│                         │  3. 310  (5.2ms)    │                             │
│                         │                     │                             │
│                         │  Best for:          │                             │
│                         │  - Latency: A100    │                             │
│                         │  - Cost/Perf: 910   │                             │
│                         └─────────────────────┘                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 3. 性能回归检测流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   Performance Regression Detection Flow                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Baseline (v1.0)                              Current (v1.1)               │
│      │                                              │                       │
│      ▼                                              ▼                       │
│  ┌─────────┐                                    ┌─────────┐                 │
│  │ Stored  │                                    │  New    │                 │
│  │ Result  │                                    │ Result  │                 │
│  └────┬────┘                                    └────┬────┘                 │
│       │                                              │                      │
│       │         ┌─────────────────────────┐          │                      │
│       └────────►│   Comparison Engine     │◄─────────┘                      │
│                 │                         │                                 │
│                 │  Metrics to Compare:    │                                 │
│                 │  - total_latency_us     │                                 │
│                 │  - achieved_tflops      │                                 │
│                 │  - achieved_bandwidth   │                                 │
│                 │  - per-op latency       │                                 │
│                 └───────────┬─────────────┘                                 │
│                             │                                               │
│                             ▼                                               │
│                 ┌─────────────────────────┐                                 │
│                 │   Regression Check      │                                 │
│                 │                         │                                 │
│                 │  if latency_diff > 10%: │                                 │
│                 │      REGRESSION ⚠️       │                                 │
│                 │  else:                  │                                 │
│                 │      PASS ✓             │                                 │
│                 └───────────┬─────────────┘                                 │
│                             │                                               │
│                             ▼                                               │
│                 ┌─────────────────────────┐                                 │
│                 │   Detailed Report       │                                 │
│                 │                         │                                 │
│                 │  Latency: 1.1ms → 1.3ms │                                 │
│                 │  Change: +18.2% ⚠️       │                                 │
│                 │                         │                                 │
│                 │  Regressed Ops:         │                                 │
│                 │  - attention: +25%      │                                 │
│                 │  - layernorm: +15%      │                                 │
│                 └─────────────────────────┘                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# 三、开发视图 (Development View)

## 1. 包结构

```
src/aitest/performance/
├── __init__.py                 # 公共API导出
├── runner.py                   # PerfTestRunner 主类
├── config.py                   # 配置定义
├── results.py                  # 结果数据类
│
├── integration/                # aidevtools 集成
│   ├── __init__.py
│   ├── analyzer_wrapper.py    # PaperAnalyzer 封装
│   ├── chip_adapter.py        # ChipSpec 适配
│   └── profile_collector.py   # Profile 收集器
│
├── assertions/                 # 性能断言
│   ├── __init__.py
│   ├── latency.py             # 时延断言
│   ├── throughput.py          # 吞吐量断言
│   ├── bandwidth.py           # 带宽断言
│   ├── utilization.py         # 利用率断言
│   └── regression.py          # 回归检测断言
│
├── baseline/                   # 基线管理
│   ├── __init__.py
│   ├── store.py               # 基线存储
│   ├── comparison.py          # 版本对比
│   └── schema.py              # 存储模式
│
├── reports/                    # 报告生成
│   ├── __init__.py
│   ├── html_report.py         # HTML 报告
│   ├── xlsx_report.py         # Excel 报告
│   └── gantt_chart.py         # Gantt 图
│
└── suites/                     # 测试套件
    ├── __init__.py
    ├── model_benchmark.py     # 模型基准测试套
    ├── regression_suite.py    # 回归测试套
    └── chip_comparison.py     # 芯片对比测试套 (可选，低优先级)
```

## 2. 代码示例

### 2.1 基本性能测试

```python
"""基本性能测试用例 - 使用通用模型配置"""

from aitest import performance_test
from aitest.performance import PerfTestConfig, assert_latency_us, assert_throughput

# 使用 aidevtools 的通用模型层函数
from aidevtools.analysis import transformer_layer, PaperAnalyzer


# ============================================
# 通用 Benchmark 配置 (中小模型)
# ============================================
BENCHMARK_CONFIGS = {
    # 小型模型 - 快速验证
    "small": {"batch": 8, "seq": 128, "hidden": 256, "num_heads": 4},
    # 中型模型 - 标准测试
    "medium": {"batch": 4, "seq": 256, "hidden": 512, "num_heads": 8},
    # 大型模型 - 压力测试
    "large": {"batch": 2, "seq": 512, "hidden": 768, "num_heads": 12},
}


@performance_test(chip="npu_910")
def test_transformer_small_latency():
    """测试小型Transformer层推理时延"""

    config = BENCHMARK_CONFIGS["small"]
    profiles = transformer_layer(**config)

    analyzer = PaperAnalyzer(chip="npu_910")
    analyzer.add_profiles(profiles)
    result = analyzer.analyze()

    # 断言时延
    total_latency_us = result.summary.totals.latency_us
    assert_latency_us(total_latency_us, max_us=500)  # < 500us

    return result


@performance_test(chip="npu_910")
def test_transformer_batch_scaling():
    """测试Transformer层批次扩展性"""

    base_config = {"seq": 256, "hidden": 512, "num_heads": 8}

    results = []
    for batch_size in [1, 2, 4, 8, 16]:
        profiles = transformer_layer(batch=batch_size, **base_config)

        analyzer = PaperAnalyzer(chip="npu_910")
        analyzer.add_profiles(profiles)
        result = analyzer.analyze()

        results.append({
            "batch_size": batch_size,
            "latency_us": result.summary.totals.latency_us,
            "throughput_tflops": result.summary.throughput.achieved_tflops,
        })

    # 验证扩展性
    throughput_1 = results[0]["throughput_tflops"]
    throughput_16 = results[4]["throughput_tflops"]
    scaling_efficiency = throughput_16 / (throughput_1 * 16)

    assert scaling_efficiency > 0.7, \
        f"批次扩展效率过低: {scaling_efficiency:.2f}"

    return results


@performance_test(chip="npu_910")
def test_sweep_hidden_dim():
    """扫描不同hidden维度的性能"""

    results = []
    for hidden in [256, 512, 768, 1024]:
        profiles = transformer_layer(
            batch=4, seq=256, hidden=hidden, num_heads=hidden // 64
        )

        analyzer = PaperAnalyzer(chip="npu_910")
        analyzer.add_profiles(profiles)
        result = analyzer.analyze()

        results.append({
            "hidden": hidden,
            "latency_us": result.summary.totals.latency_us,
            "tflops": result.summary.throughput.achieved_tflops,
        })

    return results
```

### 2.2 芯片对比测试（可选，低优先级）

```python
"""芯片对比测试 - 可选功能，低优先级"""

from aitest.performance import ChipTestSuite
from aidevtools.analysis import transformer_layer, list_chips


class TestChipComparison(ChipTestSuite):
    """芯片对比测试套件"""

    chips = ["npu_310", "npu_910"]
    # 使用通用中型配置
    config = {"batch": 4, "seq": 256, "hidden": 512, "num_heads": 8}

    def test_latency_comparison(self):
        """对比各芯片时延"""

        profiles = transformer_layer(**self.config)
        results = self.run_on_all_chips(profiles)

        comparison = self.compare_chips(results)

        print(f"\n{'=' * 50}")
        print(f"Config: {self.config}")
        print(f"{'=' * 50}")
        for i, (chip, latency) in enumerate(comparison.latency_ranking):
            print(f"  {i+1}. {chip}: {latency:.2f} us")

        return comparison
```

### 2.3 性能回归测试

```python
"""性能回归测试"""

from aitest.performance import BaselineStore, assert_no_regression
from aidevtools.analysis import transformer_layer, PaperAnalyzer


class TestPerformanceRegression:
    """性能回归测试"""

    # 通用benchmark配置
    BENCHMARK_CONFIG = {"batch": 4, "seq": 256, "hidden": 512, "num_heads": 8}

    def setup(self):
        self.baseline_store = BaselineStore("./perf_baselines")
        self.chip = "npu_910"

    def test_transformer_no_regression(self):
        """验证Transformer层无性能回归"""

        # 运行当前版本
        profiles = transformer_layer(**self.BENCHMARK_CONFIG)
        analyzer = PaperAnalyzer(chip=self.chip)
        analyzer.add_profiles(profiles)
        current_result = analyzer.analyze()

        # 加载基线
        baseline = self.baseline_store.load_baseline(
            name="transformer_medium",
            version="v1.0"
        )

        # 断言无回归
        comparison = assert_no_regression(
            current=current_result,
            baseline=baseline,
            threshold_percent=10.0
        )

        if comparison.has_regression:
            print(f"\n⚠️ Performance Regression Detected!")
            for detail in comparison.regression_details:
                print(f"  - {detail}")
            raise AssertionError("Performance regression detected")

        # 如果有改进，打印出来
        if comparison.improvements:
            print(f"\n✓ Performance Improvements:")
            for imp in comparison.improvements:
                print(f"  - {imp}")

    def save_baseline(self, version: str):
        """保存当前结果为基线"""

        profiles = transformer_layer(**self.BENCHMARK_CONFIG)
        analyzer = PaperAnalyzer(chip=self.chip)
        analyzer.add_profiles(profiles)
        result = analyzer.analyze()

        self.baseline_store.save_baseline(
            name="transformer_medium",
            result=result,
            version=version
        )
        print(f"Baseline saved: transformer_medium @ {version}")
```

### 2.4 CI 配置示例

```yaml
# .github/workflows/perf-test.yml

name: Performance Tests

on:
  push:
    branches: [main]
  pull_request:

jobs:
  perf-test:
    runs-on: [self-hosted, npu]

    steps:
      - uses: actions/checkout@v3

      - name: Setup Environment
        run: |
          pip install aidevtools aitestframework

      - name: Run Performance Tests
        run: |
          pytest tests/performance/ -v \
            --chip=npu_910 \
            --baseline-compare \
            --regression-threshold=10

      - name: Upload Performance Report
        uses: actions/upload-artifact@v3
        with:
          name: perf-report
          path: reports/perf_report.xlsx
```

---

# 四、物理视图 (Physical View)

## 1. 部署架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   Performance Testing Deployment                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      Test Execution Environment                        │  │
│  │                                                                        │  │
│  │   ┌───────────────────────────────────────────────────────────────┐   │  │
│  │   │                    aitestframework                             │   │  │
│  │   │                                                                │   │  │
│  │   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │   │  │
│  │   │  │   PerfTest  │  │   Perf      │  │  Baseline   │            │   │  │
│  │   │  │   Runner    │  │ Assertions  │  │   Store     │            │   │  │
│  │   │  └─────────────┘  └─────────────┘  └─────────────┘            │   │  │
│  │   └───────────────────────────────────────────────────────────────┘   │  │
│  │                                    │                                   │  │
│  │                                    ▼                                   │  │
│  │   ┌───────────────────────────────────────────────────────────────┐   │  │
│  │   │                    aidevtools.analysis                         │   │  │
│  │   │                                                                │   │  │
│  │   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │   │  │
│  │   │  │   Paper     │  │   Chip      │  │   Model     │            │   │  │
│  │   │  │  Analyzer   │  │   Specs     │  │  Presets    │            │   │  │
│  │   │  └─────────────┘  └─────────────┘  └─────────────┘            │   │  │
│  │   │                                                                │   │  │
│  │   │  Supported Chips: npu_310, npu_910, gpu_a100                  │   │  │
│  │   └───────────────────────────────────────────────────────────────┘   │  │
│  │                                                                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         Storage                                        │  │
│  │                                                                        │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                    │  │
│  │  │  Baselines  │  │   Reports   │  │    Logs     │                    │  │
│  │  │  (JSON/DB)  │  │ (XLSX/HTML) │  │             │                    │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. 软件依赖

```yaml
# pyproject.toml

[project]
name = "aitestframework"
dependencies = [
    "aidevtools>=0.1.0",  # Cost Model 依赖
    "numpy>=1.20",
    "pandas>=1.3",
    "pytest>=7.0",
]

[project.optional-dependencies]
performance = [
    "aidevtools[analysis]",  # 性能分析模块
    "matplotlib>=3.5",       # Gantt 图绘制
    "openpyxl>=3.0",         # Excel 报告
]
```

---

# 五、场景视图 (Scenarios View)

## 1. 核心用例

### UC-PERF-01: 通用模型性能基准测试

```python
# 用例: 测试通用Transformer层在 NPU 910 上的性能

from aidevtools.analysis import transformer_layer, PaperAnalyzer

def test_transformer_benchmark():
    """通用Transformer性能基准测试"""

    # 中型配置: batch=4, seq=256, hidden=512
    profiles = transformer_layer(batch=4, seq=256, hidden=512, num_heads=8)

    analyzer = PaperAnalyzer(chip="npu_910")
    analyzer.add_profiles(profiles)
    result = analyzer.analyze()

    # 打印摘要
    analyzer.print_summary()

    # 验证指标
    assert result.summary.totals.latency_us < 1000  # < 1ms
    assert result.summary.throughput.achieved_tflops > 10  # > 10 TFLOPS
```

### UC-PERF-02: 芯片适配验证（可选，低优先级）

```python
# 用例: 验证算子在不同芯片上的性能表现 (可选功能)

from aidevtools.analysis import transformer_layer, PaperAnalyzer

def test_multi_chip_compatibility():
    """多芯片兼容性测试"""

    # 通用中型配置
    profiles = transformer_layer(batch=4, seq=256, hidden=512, num_heads=8)

    for chip in ["npu_310", "npu_910"]:
        analyzer = PaperAnalyzer(chip=chip)
        analyzer.add_profiles(profiles)
        result = analyzer.analyze()

        print(f"\n{chip}:")
        print(f"  Latency: {result.summary.totals.latency_us:.2f} us")
        print(f"  TFLOPS: {result.summary.throughput.achieved_tflops:.2f}")

        # 基本可用性验证
        assert result.summary.totals.latency_us > 0
        assert result.summary.throughput.achieved_tflops > 0
```

### UC-PERF-03: 性能回归检测

```yaml
# CI 配置: 性能回归门禁

performance_gate:
  enabled: true
  baseline_version: "v1.0.0"

  thresholds:
    latency_regression: 10%      # 时延增加不超过10%
    throughput_regression: -5%   # 吞吐量下降不超过5%

  benchmarks:
    - name: "transformer_small"
      config: {batch: 8, seq: 128, hidden: 256, num_heads: 4}
      chip: "npu_910"
    - name: "transformer_medium"
      config: {batch: 4, seq: 256, hidden: 512, num_heads: 8}
      chip: "npu_910"

  on_regression: fail  # fail | warn | ignore
```

## 2. 场景验证矩阵

| 场景 | 覆盖需求 | 使用的 aidevtools 组件 | 优先级 |
|------|----------|------------------------|--------|
| 时延测试 | MODEL-005-01 | PaperAnalyzer, LatencyResult | P0 |
| 吞吐量测试 | MODEL-005-02 | AnalysisSummary.throughput | P0 |
| 回归测试 | MODEL-005-06 | BaselineStore, comparison | P0 |
| 通用Benchmark | MODEL-006 | transformer_layer(), 自定义配置 | P1 |
| Roofline分析 | MODEL-005-04 | RooflinePass, bottleneck stats | P1 |
| 芯片对比 | MODEL-005-06 | ChipSpec, load_chip_spec() | P2 (可选) |

---

## 需求追溯

| 需求ID | 需求名称 | 模块功能 | aidevtools 集成点 |
|--------|----------|----------|-------------------|
| MODEL-005 | 推理性能测试 | PerfTestRunner | PaperAnalyzer |
| MODEL-005-01 | 推理延迟测试 | assert_latency_us | LatencyResult |
| MODEL-005-02 | 吞吐量测试 | assert_throughput | AnalysisSummary |
| MODEL-005-04 | GPU利用率测试 | Roofline分析 | RooflinePass |
| MODEL-005-06 | 性能基准对比 | BaselineStore | export_xlsx |
| MODEL-006 | 压力测试 | 批次扩展测试 | transformer_layer() |

---

*本文档为AI测试框架性能测试模块设计，通过集成 aidevtools.analysis 提供的 Cost Model 能力，使用通用 Transformer Benchmark 配置（small/medium/large）实现算子级性能分析和性能回归测试。*
