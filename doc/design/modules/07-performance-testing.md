# 性能测试模块设计 (Performance Testing Module)

## 模块概述

| 属性 | 说明 |
|------|------|
| **模块ID** | PERF |
| **模块名称** | 性能测试 |
| **英文名称** | Performance Testing |
| **分类** | 性能需求 |
| **职责** | 算子性能分析、Cost Model集成、性能回归测试、芯片适配验证 |
| **关联需求** | MODEL-005, MODEL-006 |
| **外部依赖** | aidevtools.analysis (Cost Model) |

### 模块定位

性能测试模块作为AI测试框架的性能验证层，集成 `aidevtools.analysis` 提供的 Cost Model 能力，提供：
- **算子级性能分析**：集成 PaperAnalyzer 进行时延、带宽、算力分析
- **芯片适配验证**：验证算子在不同芯片规格下的性能表现
- **性能回归测试**：跨版本性能对比与回归检测
- **性能断言**：时延、吞吐量、利用率等指标的阈值断言

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
│  │    ChipTestSuite         │                │   PerfReportGenerator    │   │
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

## 2. 芯片对比测试流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Multi-Chip Comparison Test Flow                           │
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
    ├── chip_comparison.py     # 芯片对比测试套
    ├── model_benchmark.py     # 模型基准测试套
    └── regression_suite.py    # 回归测试套
```

## 2. 代码示例

### 2.1 基本性能测试

```python
"""基本性能测试用例"""

from aitest import performance_test
from aitest.performance import PerfTestConfig, assert_latency_us, assert_throughput

# 使用 aidevtools 的预定义模型
from aidevtools.analysis import from_preset, PaperAnalyzer


@performance_test(chip="npu_910")
def test_llama_7b_latency():
    """测试 LLaMA-7B 单层推理时延"""

    # 获取模型 profiles
    profiles = from_preset("llama-7b", batch=1)

    # 使用 PaperAnalyzer 分析
    analyzer = PaperAnalyzer(chip="npu_910")
    analyzer.add_profiles(profiles)
    result = analyzer.analyze()

    # 断言时延
    total_latency_us = result.summary.totals.latency_us
    assert_latency_us(total_latency_us, max_us=2000)  # < 2ms

    # 断言吞吐量
    achieved_tflops = result.summary.throughput.achieved_tflops
    assert_throughput(achieved_tflops, min_tflops=100)

    return result


@performance_test(chip="npu_910")
def test_bert_base_batch_scaling():
    """测试 BERT-Base 批次扩展性"""

    from aidevtools.analysis import bert_layer

    results = []
    for batch_size in [1, 4, 8, 16, 32]:
        profiles = bert_layer(batch=batch_size)

        analyzer = PaperAnalyzer(chip="npu_910")
        analyzer.add_profiles(profiles)
        result = analyzer.analyze()

        results.append({
            "batch_size": batch_size,
            "latency_us": result.summary.totals.latency_us,
            "throughput_tflops": result.summary.throughput.achieved_tflops,
        })

    # 验证线性扩展性
    # batch=32 的吞吐量应该接近 batch=1 的 32 倍
    throughput_1 = results[0]["throughput_tflops"]
    throughput_32 = results[4]["throughput_tflops"]
    scaling_efficiency = throughput_32 / (throughput_1 * 32)

    assert scaling_efficiency > 0.8, \
        f"批次扩展效率过低: {scaling_efficiency:.2f}"

    return results
```

### 2.2 芯片对比测试

```python
"""芯片对比测试"""

from aitest.performance import ChipTestSuite
from aidevtools.analysis import from_preset, list_chips


class TestChipComparison(ChipTestSuite):
    """芯片对比测试套件"""

    chips = ["npu_310", "npu_910", "gpu_a100"]
    model_preset = "llama-7b"
    batch_size = 4

    def test_latency_comparison(self):
        """对比各芯片时延"""

        profiles = from_preset(self.model_preset, batch=self.batch_size)
        results = self.run_on_all_chips(profiles)

        # 生成对比报告
        comparison = self.compare_chips(results)

        print(f"\n{'=' * 50}")
        print(f"Model: {self.model_preset}, Batch: {self.batch_size}")
        print(f"{'=' * 50}")
        print(f"\nLatency Ranking:")
        for i, (chip, latency) in enumerate(comparison.latency_ranking):
            print(f"  {i+1}. {chip}: {latency:.2f} us")

        print(f"\nThroughput Ranking:")
        for i, (chip, tflops) in enumerate(comparison.throughput_ranking):
            print(f"  {i+1}. {chip}: {tflops:.2f} TFLOPS")

        return comparison

    def test_roofline_comparison(self):
        """对比各芯片 Roofline 特性"""

        profiles = from_preset(self.model_preset, batch=self.batch_size)

        for chip in self.chips:
            result = self.run_on_chip(profiles, chip)

            compute_bound = result.summary.bottleneck.compute_bound_ops
            memory_bound = result.summary.bottleneck.memory_bound_ops
            total = compute_bound + memory_bound

            print(f"\n{chip}:")
            print(f"  Compute Bound: {compute_bound}/{total} ops")
            print(f"  Memory Bound: {memory_bound}/{total} ops")
```

### 2.3 性能回归测试

```python
"""性能回归测试"""

from aitest.performance import BaselineStore, assert_no_regression
from aidevtools.analysis import from_preset, PaperAnalyzer


class TestPerformanceRegression:
    """性能回归测试"""

    def setup(self):
        self.baseline_store = BaselineStore("./perf_baselines")
        self.chip = "npu_910"

    def test_llama_7b_no_regression(self):
        """验证 LLaMA-7B 无性能回归"""

        # 运行当前版本
        profiles = from_preset("llama-7b", batch=1)
        analyzer = PaperAnalyzer(chip=self.chip)
        analyzer.add_profiles(profiles)
        current_result = analyzer.analyze()

        # 加载基线
        baseline = self.baseline_store.load_baseline(
            name="llama-7b",
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

        profiles = from_preset("llama-7b", batch=1)
        analyzer = PaperAnalyzer(chip=self.chip)
        analyzer.add_profiles(profiles)
        result = analyzer.analyze()

        self.baseline_store.save_baseline(
            name="llama-7b",
            result=result,
            version=version
        )
        print(f"Baseline saved: llama-7b @ {version}")
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

### UC-PERF-01: 模型性能基准测试

```python
# 用例: 测试 LLaMA-7B 在 NPU 910 上的性能

from aidevtools.analysis import from_preset, PaperAnalyzer

def test_llama_7b_benchmark():
    """LLaMA-7B 性能基准测试"""

    profiles = from_preset("llama-7b", batch=1)

    analyzer = PaperAnalyzer(chip="npu_910")
    analyzer.add_profiles(profiles)
    result = analyzer.analyze()

    # 打印摘要
    analyzer.print_summary()

    # 验证指标
    assert result.summary.totals.latency_us < 2000  # < 2ms
    assert result.summary.throughput.achieved_tflops > 100  # > 100 TFLOPS
```

### UC-PERF-02: 芯片适配验证

```python
# 用例: 验证算子在不同芯片上的性能表现

from aidevtools.analysis import from_preset, PaperAnalyzer, list_chips

def test_multi_chip_compatibility():
    """多芯片兼容性测试"""

    profiles = from_preset("bert-base", batch=8)

    for chip in ["npu_310", "npu_910", "gpu_a100"]:
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

  models:
    - name: "llama-7b"
      chip: "npu_910"
      batch: 1
    - name: "bert-base"
      chip: "npu_910"
      batch: 8

  on_regression: fail  # fail | warn | ignore
```

## 2. 场景验证矩阵

| 场景 | 覆盖需求 | 使用的 aidevtools 组件 |
|------|----------|------------------------|
| 时延测试 | MODEL-005-01 | PaperAnalyzer, LatencyResult |
| 吞吐量测试 | MODEL-005-02 | AnalysisSummary.throughput |
| 芯片对比 | MODEL-005-06 | ChipSpec, load_chip_spec() |
| Roofline分析 | MODEL-005-04 | RooflinePass, bottleneck stats |
| 回归测试 | MODEL-005-06 | BaselineStore, comparison |
| 模型基准 | MODEL-006 | from_preset(), model configs |

---

## 需求追溯

| 需求ID | 需求名称 | 模块功能 | aidevtools 集成点 |
|--------|----------|----------|-------------------|
| MODEL-005 | 推理性能测试 | PerfTestRunner | PaperAnalyzer |
| MODEL-005-01 | 推理延迟测试 | assert_latency_us | LatencyResult |
| MODEL-005-02 | 吞吐量测试 | assert_throughput | AnalysisSummary |
| MODEL-005-04 | GPU利用率测试 | Roofline分析 | RooflinePass |
| MODEL-005-06 | 性能基准对比 | BaselineStore | export_xlsx |
| MODEL-006 | 压力测试 | 批次扩展测试 | from_preset |

---

*本文档为AI测试框架性能测试模块设计，通过集成 aidevtools.analysis 提供的 Cost Model 能力，实现算子级性能分析、芯片适配验证和性能回归测试。*
