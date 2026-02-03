# 性能测试模块设计 (Performance Testing Module)

## 模块概述

| 属性 | 说明 |
|------|------|
| **模块ID** | PERF |
| **模块名称** | 性能测试 |
| **英文名称** | Performance Testing |
| **分类** | 性能需求 |
| **职责** | ESL/FPGA接口版本性能追踪、NPU性能分析、CPU PMU分析、性能版本对比、C语言维测桩 |
| **关联需求** | MODEL-005, MODEL-006 |

### 模块定位

性能测试模块专注于硬件验证场景下的性能分析与追踪，主要包括：
- **ESL/FPGA接口版本追踪**：监控ESL模型和FPGA比特流接口变化导致的性能变更
- **NPU性能分析**：NPU专用性能分析脚本，支持算子级别性能剖析
- **CPU PMU分析**：基于硬件性能计数器(PMU)的CPU性能分析
- **性能版本对比**：跨版本性能元数据管理与对比分析
- **软件维测桩**：C语言性能维测桩设计，支持嵌入式系统性能采集

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Performance Testing Module                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Hardware Performance Stack                        │    │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐        │    │
│  │  │    ESL    │  │   FPGA    │  │    NPU    │  │    CPU    │        │    │
│  │  │ Interface │  │ Bitstream │  │  Profiler │  │    PMU    │        │    │
│  │  └───────────┘  └───────────┘  └───────────┘  └───────────┘        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│  ┌─────────────────────────────────┼─────────────────────────────────────┐  │
│  │                                 ▼                                      │  │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐          │  │
│  │  │  Version  │  │   Perf    │  │    C      │  │  Metadata │          │  │
│  │  │  Tracker  │  │  Compare  │  │   Stub    │  │   Store   │          │  │
│  │  └───────────┘  └───────────┘  └───────────┘  └───────────┘          │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
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
│  │                     PerformanceAnalyzer                              │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  - version_tracker: VersionTracker                                  │    │
│  │  - npu_profiler: NPUProfiler                                        │    │
│  │  - pmu_analyzer: PMUAnalyzer                                        │    │
│  │  - metadata_store: MetadataStore                                    │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  + track_version_change(old, new) -> VersionDiff                    │    │
│  │  + analyze_npu_performance(trace) -> NPUReport                      │    │
│  │  + analyze_pmu_counters(data) -> PMUReport                          │    │
│  │  + compare_versions(v1, v2) -> ComparisonReport                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌──────────────────────────┐      ┌──────────────────────────┐            │
│  │     VersionTracker       │      │     InterfaceMonitor     │            │
│  ├──────────────────────────┤      ├──────────────────────────┤            │
│  │ - esl_versions: Dict     │      │ - interface_registry     │            │
│  │ - fpga_versions: Dict    │      │ - change_listeners       │            │
│  │ - change_history: List   │      │ - diff_engine            │            │
│  ├──────────────────────────┤      ├──────────────────────────┤            │
│  │ + register_esl(version)  │      │ + detect_changes()       │            │
│  │ + register_fpga(version) │      │ + get_interface_diff()   │            │
│  │ + get_version_diff()     │      │ + notify_listeners()     │            │
│  │ + export_history()       │      │ + generate_report()      │            │
│  └──────────────────────────┘      └──────────────────────────┘            │
│                                                                             │
│  ┌──────────────────────────┐      ┌──────────────────────────┐            │
│  │      NPUProfiler         │      │      PMUAnalyzer         │            │
│  ├──────────────────────────┤      ├──────────────────────────┤            │
│  │ - npu_device: NPUDevice  │      │ - pmu_events: List       │            │
│  │ - trace_buffer: Buffer   │      │ - counters: Dict         │            │
│  │ - op_stats: Dict         │      │ - sampling_rate: int     │            │
│  ├──────────────────────────┤      ├──────────────────────────┤            │
│  │ + start_profiling()      │      │ + configure_events()     │            │
│  │ + stop_profiling()       │      │ + start_counting()       │            │
│  │ + get_op_breakdown()     │      │ + stop_counting()        │            │
│  │ + get_memory_traffic()   │      │ + read_counters()        │            │
│  │ + get_utilization()      │      │ + analyze_hotspots()     │            │
│  │ + export_trace()         │      │ + export_report()        │            │
│  └──────────────────────────┘      └──────────────────────────┘            │
│                                                                             │
│  ┌──────────────────────────┐      ┌──────────────────────────┐            │
│  │     MetadataStore        │      │    InstrumentStub        │            │
│  ├──────────────────────────┤      ├──────────────────────────┤            │
│  │ - db_path: str           │      │ - stub_config: Config    │            │
│  │ - schema: Schema         │      │ - output_format: Format  │            │
│  │ - indexes: Dict          │      │ - buffer_size: int       │            │
│  ├──────────────────────────┤      ├──────────────────────────┤            │
│  │ + store_metadata()       │      │ + generate_stub()        │            │
│  │ + query_by_version()     │      │ + insert_probes()        │            │
│  │ + get_perf_history()     │      │ + collect_data()         │            │
│  │ + export_comparison()    │      │ + export_results()       │            │
│  └──────────────────────────┘      └──────────────────────────┘            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. 核心数据结构

### 2.1 ESL/FPGA版本追踪数据结构

```python
@dataclass
class ESLVersion:
    """ESL模型版本信息"""
    version_id: str              # 版本标识
    model_name: str              # 模型名称
    interface_hash: str          # 接口签名哈希
    timestamp: datetime          # 时间戳

    # 接口定义
    input_ports: List[PortDef]   # 输入端口定义
    output_ports: List[PortDef]  # 输出端口定义
    config_params: Dict[str, Any] # 配置参数

    # 性能基准
    cycle_count: int             # 周期数
    latency_ns: float            # 延迟(ns)
    throughput_ops: float        # 吞吐量(ops/s)


@dataclass
class FPGABitstream:
    """FPGA比特流版本信息"""
    version_id: str              # 版本标识
    bitstream_hash: str          # 比特流哈希
    device_family: str           # 设备系列
    timestamp: datetime          # 时间戳

    # 资源使用
    lut_usage: int               # LUT使用量
    ff_usage: int                # 触发器使用量
    bram_usage: int              # BRAM使用量
    dsp_usage: int               # DSP使用量

    # 时序信息
    clock_freq_mhz: float        # 时钟频率
    timing_slack_ns: float       # 时序裕量

    # 接口定义
    axi_interfaces: List[AXIInterface]  # AXI接口


@dataclass
class VersionDiff:
    """版本差异分析结果"""
    old_version: str
    new_version: str
    timestamp: datetime

    # 接口变更
    added_ports: List[PortDef]
    removed_ports: List[PortDef]
    modified_ports: List[PortModification]

    # 性能变更
    perf_changes: Dict[str, PerfChange]

    # 兼容性评估
    is_backward_compatible: bool
    breaking_changes: List[str]

    # 影响分析
    affected_testcases: List[str]
    recommended_actions: List[str]


@dataclass
class PerfChange:
    """性能变更详情"""
    metric_name: str
    old_value: float
    new_value: float
    change_percent: float
    significance: str  # "improvement", "regression", "neutral"
```

### 2.2 NPU性能分析数据结构

```python
@dataclass
class NPUProfileConfig:
    """NPU性能分析配置"""
    # 采集配置
    trace_level: str = "operator"  # "operator", "kernel", "instruction"
    sampling_enabled: bool = True
    sampling_rate: int = 1000      # 采样频率(Hz)

    # 内存追踪
    memory_trace: bool = True
    bandwidth_trace: bool = True

    # 算子级配置
    op_breakdown: bool = True
    op_timeline: bool = True

    # 输出配置
    output_format: str = "json"
    output_path: str = "./npu_trace"


@dataclass
class NPUOperatorStats:
    """NPU算子统计"""
    op_name: str                  # 算子名称
    op_type: str                  # 算子类型
    call_count: int               # 调用次数

    # 时间统计
    total_time_us: float          # 总耗时(us)
    avg_time_us: float            # 平均耗时(us)
    min_time_us: float            # 最小耗时(us)
    max_time_us: float            # 最大耗时(us)

    # 计算效率
    compute_utilization: float    # 计算利用率
    memory_utilization: float     # 内存利用率

    # 数据移动
    input_bytes: int              # 输入数据量
    output_bytes: int             # 输出数据量
    weight_bytes: int             # 权重数据量


@dataclass
class NPUProfileResult:
    """NPU性能分析结果"""
    session_id: str
    timestamp: datetime
    device_info: Dict[str, Any]

    # 算子统计
    operator_stats: List[NPUOperatorStats]

    # 整体指标
    total_time_ms: float
    compute_time_ms: float
    memory_time_ms: float
    idle_time_ms: float

    # 内存分析
    peak_memory_mb: float
    memory_bandwidth_gbps: float

    # 利用率分析
    overall_utilization: float
    compute_bound_ops: List[str]
    memory_bound_ops: List[str]

    # 优化建议
    optimization_hints: List[str]
```

### 2.3 CPU PMU分析数据结构

```python
@dataclass
class PMUEventConfig:
    """PMU事件配置"""
    event_name: str               # 事件名称
    event_code: int               # 事件代码
    umask: int = 0                # 单元掩码
    edge: bool = False            # 边沿触发
    inv: bool = False             # 反转计数
    cmask: int = 0                # 计数器掩码


@dataclass
class PMUCounters:
    """PMU计数器数据"""
    # CPU周期
    cpu_cycles: int
    instructions: int
    ipc: float                    # 每周期指令数

    # 缓存相关
    l1d_cache_hits: int
    l1d_cache_misses: int
    l1i_cache_hits: int
    l1i_cache_misses: int
    l2_cache_hits: int
    l2_cache_misses: int
    l3_cache_hits: int
    l3_cache_misses: int

    # 分支预测
    branch_instructions: int
    branch_misses: int
    branch_miss_rate: float

    # 内存访问
    memory_loads: int
    memory_stores: int
    memory_bandwidth_gbps: float

    # TLB
    dtlb_misses: int
    itlb_misses: int


@dataclass
class PMUAnalysisResult:
    """PMU分析结果"""
    session_id: str
    timestamp: datetime
    duration_seconds: float

    # 计数器数据
    counters: PMUCounters

    # 热点函数
    hotspots: List[FunctionHotspot]

    # 性能瓶颈
    bottlenecks: List[PerformanceBottleneck]

    # 优化建议
    recommendations: List[str]


@dataclass
class FunctionHotspot:
    """函数热点信息"""
    function_name: str
    module_name: str
    address: int

    # 采样数据
    sample_count: int
    percentage: float

    # 性能指标
    cycles_per_call: float
    cache_miss_rate: float


@dataclass
class PerformanceBottleneck:
    """性能瓶颈"""
    bottleneck_type: str          # "cache", "branch", "memory", "compute"
    severity: str                 # "low", "medium", "high"
    description: str
    affected_functions: List[str]
    suggested_fix: str
```

### 2.4 性能元数据存储

```python
@dataclass
class PerformanceMetadata:
    """性能版本元数据"""
    # 版本标识
    version_id: str
    build_id: str
    commit_hash: str
    timestamp: datetime

    # 环境信息
    hardware_config: HardwareConfig
    software_config: SoftwareConfig

    # ESL/FPGA信息
    esl_version: Optional[ESLVersion]
    fpga_bitstream: Optional[FPGABitstream]

    # 性能数据
    npu_metrics: Optional[NPUProfileResult]
    pmu_metrics: Optional[PMUAnalysisResult]

    # 自定义指标
    custom_metrics: Dict[str, float]

    # 标签
    tags: List[str]
    annotations: Dict[str, str]


@dataclass
class VersionComparison:
    """版本对比结果"""
    baseline_version: str
    target_version: str
    comparison_timestamp: datetime

    # ESL/FPGA变更
    interface_changes: VersionDiff

    # 性能对比
    perf_comparison: Dict[str, MetricComparison]

    # 回归分析
    regressions: List[RegressionInfo]
    improvements: List[ImprovementInfo]

    # 总体评估
    overall_status: str  # "pass", "warn", "fail"
    summary: str


@dataclass
class MetricComparison:
    """指标对比"""
    metric_name: str
    baseline_value: float
    target_value: float
    absolute_diff: float
    relative_diff_percent: float
    threshold: Optional[float]
    status: str  # "improved", "regressed", "unchanged"
```

## 3. 核心接口定义

```python
class IVersionTracker(Protocol):
    """版本追踪器接口"""

    def register_esl_version(self, version: ESLVersion) -> None:
        """注册ESL版本"""
        ...

    def register_fpga_version(self, bitstream: FPGABitstream) -> None:
        """注册FPGA比特流版本"""
        ...

    def get_version_diff(
        self,
        old_version: str,
        new_version: str
    ) -> VersionDiff:
        """获取版本差异"""
        ...

    def get_change_history(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[VersionDiff]:
        """获取变更历史"""
        ...


class INPUProfiler(Protocol):
    """NPU性能分析器接口"""

    def configure(self, config: NPUProfileConfig) -> None:
        """配置分析器"""
        ...

    def start_profiling(self) -> None:
        """开始性能分析"""
        ...

    def stop_profiling(self) -> NPUProfileResult:
        """停止分析并返回结果"""
        ...

    def get_operator_breakdown(self) -> List[NPUOperatorStats]:
        """获取算子级分解"""
        ...

    def export_trace(self, path: str, format: str = "json") -> None:
        """导出追踪数据"""
        ...


class IPMUAnalyzer(Protocol):
    """PMU分析器接口"""

    def configure_events(self, events: List[PMUEventConfig]) -> None:
        """配置PMU事件"""
        ...

    def start_counting(self) -> None:
        """开始计数"""
        ...

    def stop_counting(self) -> PMUCounters:
        """停止计数"""
        ...

    def analyze(self, counters: PMUCounters) -> PMUAnalysisResult:
        """分析PMU数据"""
        ...

    def identify_hotspots(self) -> List[FunctionHotspot]:
        """识别热点"""
        ...


class IInstrumentStub(Protocol):
    """维测桩接口"""

    def generate_stub_code(
        self,
        config: StubConfig
    ) -> str:
        """生成桩代码"""
        ...

    def insert_probes(
        self,
        source_file: str,
        probe_points: List[ProbePoint]
    ) -> str:
        """插入探针"""
        ...

    def collect_data(self) -> InstrumentData:
        """收集数据"""
        ...


class IMetadataStore(Protocol):
    """元数据存储接口"""

    def store(self, metadata: PerformanceMetadata) -> str:
        """存储元数据"""
        ...

    def query_by_version(self, version_id: str) -> PerformanceMetadata:
        """按版本查询"""
        ...

    def compare_versions(
        self,
        baseline: str,
        target: str
    ) -> VersionComparison:
        """版本对比"""
        ...

    def get_trend(
        self,
        metric: str,
        start: datetime,
        end: datetime
    ) -> List[Tuple[datetime, float]]:
        """获取趋势数据"""
        ...
```

---

# 二、进程视图 (Process View)

## 1. ESL/FPGA接口版本追踪流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│               ESL/FPGA Interface Version Tracking Flow                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐   │
│  │  Parse  │───►│ Extract │───►│ Compare │───►│ Analyze │───►│ Report  │   │
│  │ Version │    │Interface│    │  Diff   │    │ Impact  │    │ Change  │   │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘   │
│                                                                             │
│  详细流程:                                                                   │
│                                                                             │
│  1. Parse Version                                                           │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  ESL Model File (.cpp/.h)  ──parse──► ESLVersion                │     │
│     │  FPGA Bitstream (.bit)     ──parse──► FPGABitstream             │     │
│     │  版本文件 (version.yaml)   ──parse──► VersionInfo               │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  2. Extract Interface                                                       │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  - 解析端口定义 (input/output ports)                            │     │
│     │  - 提取时序约束 (timing constraints)                            │     │
│     │  - 获取配置参数 (configuration parameters)                      │     │
│     │  - 计算接口签名 (interface signature hash)                      │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  3. Compare Diff                                                            │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  old_interface vs new_interface:                                │     │
│     │  - 新增端口 (added ports)                                       │     │
│     │  - 删除端口 (removed ports)                                     │     │
│     │  - 修改端口 (modified ports: type/width/timing)                 │     │
│     │  - 参数变更 (parameter changes)                                 │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  4. Analyze Impact                                                          │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  - 兼容性评估 (backward compatibility check)                    │     │
│     │  - 性能影响分析 (performance impact analysis)                   │     │
│     │  - 受影响测试用例识别 (affected testcase identification)        │     │
│     │  - 迁移建议生成 (migration recommendation)                      │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. NPU性能分析流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NPU Performance Profiling Flow                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                       ┌─────────────────────┐                               │
│                       │    NPU Profiler     │                               │
│                       │     Controller      │                               │
│                       └──────────┬──────────┘                               │
│                                  │                                          │
│         ┌────────────────────────┼────────────────────────┐                 │
│         ▼                        ▼                        ▼                 │
│  ┌─────────────┐         ┌─────────────┐         ┌─────────────┐           │
│  │   Operator  │         │   Memory    │         │    Util     │           │
│  │   Tracer    │         │   Tracker   │         │   Monitor   │           │
│  └──────┬──────┘         └──────┬──────┘         └──────┬──────┘           │
│         │                       │                       │                   │
│         ▼                       ▼                       ▼                   │
│  ┌─────────────┐         ┌─────────────┐         ┌─────────────┐           │
│  │ Op Timeline │         │ Memory BW   │         │ Compute %   │           │
│  │ Op Duration │         │ Peak Mem    │         │ Memory %    │           │
│  │ Call Count  │         │ Data Move   │         │ Idle %      │           │
│  └──────┬──────┘         └──────┬──────┘         └──────┬──────┘           │
│         │                       │                       │                   │
│         └────────────────────────┼────────────────────────┘                 │
│                                  ▼                                          │
│                       ┌─────────────────────┐                               │
│                       │   Result Analyzer   │                               │
│                       │   ┌─────────────┐   │                               │
│                       │   │ Bottleneck  │   │                               │
│                       │   │ Detection   │   │                               │
│                       │   └─────────────┘   │                               │
│                       │   ┌─────────────┐   │                               │
│                       │   │Optimization │   │                               │
│                       │   │   Hints     │   │                               │
│                       │   └─────────────┘   │                               │
│                       └─────────────────────┘                               │
│                                                                             │
│  分析脚本示例:                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  # npu_profiler.py                                                  │    │
│  │  profiler = NPUProfiler(device="npu0")                              │    │
│  │  profiler.configure(trace_level="operator", memory_trace=True)      │    │
│  │                                                                      │    │
│  │  profiler.start_profiling()                                         │    │
│  │  model.inference(input_data)                                        │    │
│  │  result = profiler.stop_profiling()                                 │    │
│  │                                                                      │    │
│  │  # 分析结果                                                          │    │
│  │  for op in result.operator_stats:                                   │    │
│  │      print(f"{op.op_name}: {op.total_time_us}us, util={op.compute_utilization}")│
│  │                                                                      │    │
│  │  # 导出追踪                                                          │    │
│  │  profiler.export_trace("./trace.json")                              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 3. CPU PMU分析流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CPU PMU Analysis Flow                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     PMU Event Configuration                          │    │
│  │                                                                      │    │
│  │   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  │    │
│  │   │ Cycles  │  │  IPC    │  │  Cache  │  │ Branch  │  │ Memory  │  │    │
│  │   │         │  │         │  │  Miss   │  │  Miss   │  │   BW    │  │    │
│  │   └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  │    │
│  │        │            │            │            │            │        │    │
│  │        └────────────┴────────────┼────────────┴────────────┘        │    │
│  │                                  │                                   │    │
│  └──────────────────────────────────┼───────────────────────────────────┘    │
│                                     ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     Data Collection                                  │    │
│  │                                                                      │    │
│  │    perf_event_open() ──► read_counters() ──► aggregate_samples()   │    │
│  │                                                                      │    │
│  │    支持模式:                                                         │    │
│  │    - Counting Mode: 精确计数，低开销                                  │    │
│  │    - Sampling Mode: 采样分析，识别热点                                │    │
│  └──────────────────────────────────┬───────────────────────────────────┘    │
│                                     ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     Analysis & Report                                │    │
│  │                                                                      │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │  热点分析:                                                     │  │    │
│  │  │  - Top函数排名 (按cycle/sample)                                │  │    │
│  │  │  - 调用路径分析                                                 │  │    │
│  │  │  - 源码级热点定位                                               │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  │                                                                      │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │  瓶颈识别:                                                     │  │    │
│  │  │  - Cache瓶颈: L1/L2/L3 miss rate > threshold                   │  │    │
│  │  │  - 分支瓶颈: branch miss rate > threshold                      │  │    │
│  │  │  - 内存瓶颈: memory bandwidth saturation                       │  │    │
│  │  │  - 前端瓶颈: instruction fetch stalls                          │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  PMU分析脚本示例:                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  # pmu_analyzer.py                                                  │    │
│  │  analyzer = PMUAnalyzer()                                           │    │
│  │                                                                      │    │
│  │  # 配置标准事件集                                                    │    │
│  │  analyzer.configure_events([                                        │    │
│  │      PMUEventConfig("cycles", 0x3C),                                │    │
│  │      PMUEventConfig("instructions", 0xC0),                          │    │
│  │      PMUEventConfig("l3_cache_misses", 0x2E, umask=0x41),           │    │
│  │      PMUEventConfig("branch_misses", 0xC5),                         │    │
│  │  ])                                                                  │    │
│  │                                                                      │    │
│  │  analyzer.start_counting()                                          │    │
│  │  run_benchmark()                                                    │    │
│  │  counters = analyzer.stop_counting()                                │    │
│  │                                                                      │    │
│  │  result = analyzer.analyze(counters)                                │    │
│  │  print(f"IPC: {result.counters.ipc}")                               │    │
│  │  print(f"L3 Miss Rate: {result.counters.l3_cache_misses/result.counters.memory_loads*100:.2f}%")│
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 4. 性能版本对比流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  Performance Version Comparison Flow                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Baseline                                        Target                    │
│   Version                                         Version                   │
│      │                                               │                      │
│      ▼                                               ▼                      │
│  ┌─────────┐                                    ┌─────────┐                 │
│  │Metadata │                                    │Metadata │                 │
│  │  Store  │                                    │  Store  │                 │
│  └────┬────┘                                    └────┬────┘                 │
│       │                                              │                      │
│       │         ┌─────────────────────────┐          │                      │
│       └────────►│   Comparison Engine     │◄─────────┘                      │
│                 │                         │                                 │
│                 │  ┌─────────────────┐    │                                 │
│                 │  │ Interface Diff  │    │                                 │
│                 │  └─────────────────┘    │                                 │
│                 │  ┌─────────────────┐    │                                 │
│                 │  │  Metric Delta   │    │                                 │
│                 │  └─────────────────┘    │                                 │
│                 │  ┌─────────────────┐    │                                 │
│                 │  │ Regression Det  │    │                                 │
│                 │  └─────────────────┘    │                                 │
│                 └───────────┬─────────────┘                                 │
│                             │                                               │
│                             ▼                                               │
│                 ┌─────────────────────────┐                                 │
│                 │   Comparison Report     │                                 │
│                 │                         │                                 │
│                 │  - Interface Changes    │                                 │
│                 │  - Performance Delta    │                                 │
│                 │  - Regression List      │                                 │
│                 │  - Trend Analysis       │                                 │
│                 │  - Recommendations      │                                 │
│                 └─────────────────────────┘                                 │
│                                                                             │
│  元数据Schema:                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  version_metadata:                                                  │    │
│  │    version_id: string                                               │    │
│  │    build_info:                                                      │    │
│  │      commit: string                                                 │    │
│  │      branch: string                                                 │    │
│  │      timestamp: datetime                                            │    │
│  │    esl_info:                                                        │    │
│  │      model_version: string                                          │    │
│  │      interface_hash: string                                         │    │
│  │    fpga_info:                                                       │    │
│  │      bitstream_version: string                                      │    │
│  │      resource_usage: object                                         │    │
│  │    performance:                                                     │    │
│  │      npu_metrics: object                                            │    │
│  │      pmu_metrics: object                                            │    │
│  │      custom_metrics: object                                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# 三、开发视图 (Development View)

## 1. 包结构

```
src/aitest/performance/
├── __init__.py                 # 公共API导出
├── analyzer.py                 # PerformanceAnalyzer 主类
├── config.py                   # 配置定义
├── results.py                  # 结果数据类
│
├── version/                    # 版本追踪
│   ├── __init__.py
│   ├── tracker.py             # VersionTracker 版本追踪器
│   ├── esl_parser.py          # ESL模型解析器
│   ├── fpga_parser.py         # FPGA比特流解析器
│   ├── diff_engine.py         # 差异分析引擎
│   └── impact_analyzer.py     # 影响分析器
│
├── npu/                        # NPU性能分析
│   ├── __init__.py
│   ├── profiler.py            # NPU性能分析器
│   ├── trace_collector.py     # 追踪数据收集
│   ├── op_analyzer.py         # 算子分析
│   ├── memory_analyzer.py     # 内存分析
│   └── report_generator.py    # 报告生成
│
├── pmu/                        # CPU PMU分析
│   ├── __init__.py
│   ├── analyzer.py            # PMU分析器
│   ├── event_config.py        # 事件配置
│   ├── counter_reader.py      # 计数器读取
│   ├── hotspot_detector.py    # 热点检测
│   └── bottleneck_analyzer.py # 瓶颈分析
│
├── metadata/                   # 元数据管理
│   ├── __init__.py
│   ├── store.py               # 元数据存储
│   ├── schema.py              # 数据模式
│   ├── query.py               # 查询接口
│   └── comparison.py          # 版本对比
│
├── stub/                       # C语言维测桩
│   ├── __init__.py
│   ├── generator.py           # 桩代码生成器
│   ├── templates/             # 代码模板
│   │   ├── probe_template.c
│   │   ├── timer_template.c
│   │   └── buffer_template.c
│   ├── inserter.py            # 探针插入
│   └── collector.py           # 数据收集
│
└── scripts/                    # 分析脚本
    ├── npu_profile.py         # NPU性能分析脚本
    ├── pmu_analyze.py         # PMU分析脚本
    ├── version_compare.py     # 版本对比脚本
    └── report_generate.py     # 报告生成脚本
```

## 2. C语言维测桩设计

### 2.1 维测桩架构

```c
/*
 * 性能维测桩设计
 * 用于嵌入式系统的轻量级性能采集
 */

/* ============================================
 * 数据结构定义
 * ============================================ */

/* 性能采样点 */
typedef struct {
    uint32_t probe_id;          /* 探针ID */
    uint64_t timestamp;         /* 时间戳(cycles/ns) */
    uint32_t event_type;        /* 事件类型 */
    uint32_t data[4];           /* 自定义数据 */
} perf_sample_t;

/* 环形缓冲区 */
typedef struct {
    perf_sample_t *buffer;      /* 数据缓冲区 */
    uint32_t size;              /* 缓冲区大小 */
    uint32_t head;              /* 写入位置 */
    uint32_t tail;              /* 读取位置 */
    uint32_t overflow_count;    /* 溢出计数 */
} perf_buffer_t;

/* 性能统计 */
typedef struct {
    uint64_t total_time;        /* 总耗时 */
    uint64_t min_time;          /* 最小耗时 */
    uint64_t max_time;          /* 最大耗时 */
    uint64_t call_count;        /* 调用次数 */
} perf_stats_t;

/* 探针配置 */
typedef struct {
    uint32_t probe_id;          /* 探针ID */
    const char *name;           /* 探针名称 */
    uint8_t enabled;            /* 是否启用 */
    uint8_t level;              /* 采集级别 */
    perf_stats_t stats;         /* 统计数据 */
} probe_config_t;


/* ============================================
 * 核心API
 * ============================================ */

/* 初始化性能采集系统 */
int perf_init(uint32_t buffer_size);

/* 反初始化 */
void perf_deinit(void);

/* 启用/禁用采集 */
void perf_enable(void);
void perf_disable(void);

/* 注册探针 */
int perf_register_probe(uint32_t probe_id, const char *name);

/* 记录事件 */
static inline void perf_record(uint32_t probe_id, uint32_t event_type)
{
    if (!g_perf_enabled) return;

    perf_sample_t sample = {
        .probe_id = probe_id,
        .timestamp = perf_get_timestamp(),
        .event_type = event_type,
    };

    perf_buffer_write(&g_perf_buffer, &sample);
}

/* 函数入口/出口宏 */
#define PERF_FUNC_ENTER(id) \
    uint64_t __perf_enter_ts = perf_get_timestamp(); \
    perf_record(id, PERF_EVENT_ENTER)

#define PERF_FUNC_EXIT(id) \
    do { \
        perf_record(id, PERF_EVENT_EXIT); \
        perf_update_stats(id, perf_get_timestamp() - __perf_enter_ts); \
    } while(0)

/* 区间测量宏 */
#define PERF_BEGIN(id)  perf_record(id, PERF_EVENT_BEGIN)
#define PERF_END(id)    perf_record(id, PERF_EVENT_END)

/* 自定义数据记录 */
#define PERF_DATA(id, d0, d1, d2, d3) \
    perf_record_data(id, d0, d1, d2, d3)


/* ============================================
 * 高精度时间戳
 * ============================================ */

/* 平台相关的时间戳获取 */
#if defined(__ARM_ARCH)
/* ARM: 使用CNTVCT_EL0 */
static inline uint64_t perf_get_timestamp(void)
{
    uint64_t ts;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(ts));
    return ts;
}
#elif defined(__x86_64__)
/* x86_64: 使用RDTSC */
static inline uint64_t perf_get_timestamp(void)
{
    uint32_t lo, hi;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}
#else
/* 通用: 使用clock_gettime */
static inline uint64_t perf_get_timestamp(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}
#endif


/* ============================================
 * 数据导出
 * ============================================ */

/* 导出格式 */
typedef enum {
    PERF_EXPORT_BINARY,         /* 二进制格式 */
    PERF_EXPORT_CSV,            /* CSV格式 */
    PERF_EXPORT_JSON,           /* JSON格式 */
} perf_export_format_t;

/* 导出数据 */
int perf_export(const char *path, perf_export_format_t format);

/* 获取统计信息 */
int perf_get_stats(uint32_t probe_id, perf_stats_t *stats);

/* 打印报告 */
void perf_print_report(void);
```

### 2.2 维测桩使用示例

```c
/* 示例: 在NPU驱动中使用维测桩 */

#include "perf_stub.h"

/* 定义探针ID */
enum {
    PROBE_NPU_SUBMIT = 1,
    PROBE_NPU_WAIT,
    PROBE_NPU_DMA,
    PROBE_NPU_COMPUTE,
};

/* 初始化 */
void npu_driver_init(void)
{
    /* 初始化性能采集 */
    perf_init(4096);  /* 4K采样点缓冲区 */

    /* 注册探针 */
    perf_register_probe(PROBE_NPU_SUBMIT, "npu_submit");
    perf_register_probe(PROBE_NPU_WAIT, "npu_wait");
    perf_register_probe(PROBE_NPU_DMA, "npu_dma");
    perf_register_probe(PROBE_NPU_COMPUTE, "npu_compute");

    perf_enable();
}

/* 任务提交函数 */
int npu_submit_task(npu_task_t *task)
{
    PERF_FUNC_ENTER(PROBE_NPU_SUBMIT);

    /* DMA传输 */
    PERF_BEGIN(PROBE_NPU_DMA);
    npu_dma_transfer(task->input, task->input_size);
    PERF_END(PROBE_NPU_DMA);

    /* 启动计算 */
    PERF_BEGIN(PROBE_NPU_COMPUTE);
    npu_start_compute(task);
    PERF_END(PROBE_NPU_COMPUTE);

    PERF_FUNC_EXIT(PROBE_NPU_SUBMIT);
    return 0;
}

/* 等待完成 */
int npu_wait_complete(npu_handle_t handle)
{
    PERF_FUNC_ENTER(PROBE_NPU_WAIT);

    int ret = npu_poll_status(handle);

    PERF_FUNC_EXIT(PROBE_NPU_WAIT);
    return ret;
}

/* 导出性能数据 */
void npu_export_perf_data(void)
{
    perf_export("/tmp/npu_perf.json", PERF_EXPORT_JSON);
    perf_print_report();
}
```

## 3. Python分析脚本示例

### 3.1 NPU性能分析脚本

```python
#!/usr/bin/env python3
"""
NPU性能分析脚本
用于分析NPU性能数据并生成报告
"""

from aitest.performance.npu import NPUProfiler, NPUProfileConfig
from aitest.performance.metadata import MetadataStore

def analyze_npu_performance(model_path: str, input_data: dict) -> dict:
    """分析NPU性能"""

    # 配置分析器
    config = NPUProfileConfig(
        trace_level="operator",
        memory_trace=True,
        bandwidth_trace=True,
        op_breakdown=True,
        output_format="json"
    )

    profiler = NPUProfiler(device="npu0")
    profiler.configure(config)

    # 加载模型
    model = load_model(model_path)

    # 开始分析
    profiler.start_profiling()

    # 执行推理
    output = model.inference(input_data)

    # 停止分析
    result = profiler.stop_profiling()

    # 分析结果
    print(f"总耗时: {result.total_time_ms:.2f}ms")
    print(f"计算时间: {result.compute_time_ms:.2f}ms")
    print(f"内存传输时间: {result.memory_time_ms:.2f}ms")
    print(f"整体利用率: {result.overall_utilization:.1f}%")

    print("\n算子耗时Top10:")
    sorted_ops = sorted(result.operator_stats,
                        key=lambda x: x.total_time_us,
                        reverse=True)
    for i, op in enumerate(sorted_ops[:10]):
        print(f"  {i+1}. {op.op_name}: {op.total_time_us:.1f}us "
              f"(util={op.compute_utilization:.1f}%)")

    print("\n计算瓶颈算子:")
    for op_name in result.compute_bound_ops:
        print(f"  - {op_name}")

    print("\n内存瓶颈算子:")
    for op_name in result.memory_bound_ops:
        print(f"  - {op_name}")

    print("\n优化建议:")
    for hint in result.optimization_hints:
        print(f"  - {hint}")

    # 导出追踪数据
    profiler.export_trace("./npu_trace.json")

    return result.to_dict()


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "model.bin"
    analyze_npu_performance(model_path, {"input": load_test_data()})
```

### 3.2 CPU PMU分析脚本

```python
#!/usr/bin/env python3
"""
CPU PMU分析脚本
使用硬件性能计数器分析CPU性能
"""

from aitest.performance.pmu import PMUAnalyzer, PMUEventConfig

def analyze_cpu_performance(target_func, *args, **kwargs):
    """使用PMU分析CPU性能"""

    analyzer = PMUAnalyzer()

    # 配置标准事件集
    analyzer.configure_events([
        # 基础事件
        PMUEventConfig("cpu_cycles", event_code=0x3C),
        PMUEventConfig("instructions", event_code=0xC0),

        # 缓存事件
        PMUEventConfig("l1d_cache_misses", event_code=0x51, umask=0x01),
        PMUEventConfig("l2_cache_misses", event_code=0x51, umask=0x02),
        PMUEventConfig("l3_cache_misses", event_code=0x2E, umask=0x41),

        # 分支事件
        PMUEventConfig("branch_instructions", event_code=0xC4),
        PMUEventConfig("branch_misses", event_code=0xC5),

        # 内存事件
        PMUEventConfig("memory_loads", event_code=0xD0, umask=0x81),
        PMUEventConfig("memory_stores", event_code=0xD0, umask=0x82),
    ])

    # 开始计数
    analyzer.start_counting()

    # 执行目标函数
    result = target_func(*args, **kwargs)

    # 停止计数
    counters = analyzer.stop_counting()

    # 分析结果
    analysis = analyzer.analyze(counters)

    # 打印报告
    print("=" * 60)
    print("CPU PMU 分析报告")
    print("=" * 60)

    print(f"\n基础指标:")
    print(f"  CPU Cycles:     {counters.cpu_cycles:,}")
    print(f"  Instructions:   {counters.instructions:,}")
    print(f"  IPC:            {counters.ipc:.2f}")

    print(f"\n缓存性能:")
    l1_miss_rate = counters.l1d_cache_misses / max(counters.memory_loads, 1) * 100
    l3_miss_rate = counters.l3_cache_misses / max(counters.memory_loads, 1) * 100
    print(f"  L1D Miss Rate:  {l1_miss_rate:.2f}%")
    print(f"  L3 Miss Rate:   {l3_miss_rate:.2f}%")

    print(f"\n分支预测:")
    branch_miss_rate = counters.branch_misses / max(counters.branch_instructions, 1) * 100
    print(f"  Branch Miss Rate: {branch_miss_rate:.2f}%")

    # 识别瓶颈
    print(f"\n性能瓶颈:")
    for bottleneck in analysis.bottlenecks:
        print(f"  [{bottleneck.severity.upper()}] {bottleneck.bottleneck_type}: "
              f"{bottleneck.description}")
        print(f"    建议: {bottleneck.suggested_fix}")

    # 热点函数
    if analysis.hotspots:
        print(f"\n热点函数 Top5:")
        for i, hotspot in enumerate(analysis.hotspots[:5]):
            print(f"  {i+1}. {hotspot.function_name} "
                  f"({hotspot.percentage:.1f}%)")

    print("=" * 60)

    return analysis


# 使用示例
if __name__ == "__main__":
    def benchmark_workload():
        """示例工作负载"""
        import numpy as np
        a = np.random.rand(1000, 1000)
        b = np.random.rand(1000, 1000)
        for _ in range(10):
            c = np.dot(a, b)
        return c

    analyze_cpu_performance(benchmark_workload)
```

### 3.3 版本对比脚本

```python
#!/usr/bin/env python3
"""
性能版本对比脚本
对比不同版本间的性能差异
"""

from aitest.performance.metadata import MetadataStore, VersionComparison
from aitest.performance.version import VersionTracker

def compare_versions(baseline_version: str, target_version: str):
    """对比两个版本的性能"""

    store = MetadataStore("./perf_metadata.db")
    tracker = VersionTracker()

    # 获取版本元数据
    baseline = store.query_by_version(baseline_version)
    target = store.query_by_version(target_version)

    # 接口变更分析
    interface_diff = tracker.get_version_diff(
        baseline.esl_version,
        target.esl_version
    )

    # 性能对比
    comparison = store.compare_versions(baseline_version, target_version)

    # 生成报告
    print("=" * 70)
    print(f"性能版本对比报告")
    print(f"基线版本: {baseline_version}")
    print(f"目标版本: {target_version}")
    print("=" * 70)

    # 接口变更
    print("\n[接口变更]")
    if interface_diff.is_backward_compatible:
        print("  ✓ 接口向后兼容")
    else:
        print("  ✗ 接口不兼容!")
        for change in interface_diff.breaking_changes:
            print(f"    - {change}")

    if interface_diff.added_ports:
        print(f"  新增端口: {len(interface_diff.added_ports)}")
    if interface_diff.removed_ports:
        print(f"  删除端口: {len(interface_diff.removed_ports)}")
    if interface_diff.modified_ports:
        print(f"  修改端口: {len(interface_diff.modified_ports)}")

    # 性能变化
    print("\n[性能变化]")
    for metric_name, metric_cmp in comparison.perf_comparison.items():
        status_icon = {
            "improved": "↑",
            "regressed": "↓",
            "unchanged": "→"
        }.get(metric_cmp.status, "?")

        print(f"  {metric_name}:")
        print(f"    基线: {metric_cmp.baseline_value:.2f}")
        print(f"    目标: {metric_cmp.target_value:.2f}")
        print(f"    变化: {metric_cmp.relative_diff_percent:+.1f}% {status_icon}")

    # 回归列表
    if comparison.regressions:
        print("\n[性能回归] ⚠️")
        for reg in comparison.regressions:
            print(f"  - {reg.metric}: {reg.change_percent:+.1f}% "
                  f"(阈值: {reg.threshold}%)")

    # 改进列表
    if comparison.improvements:
        print("\n[性能改进] ✓")
        for imp in comparison.improvements:
            print(f"  - {imp.metric}: {imp.change_percent:+.1f}%")

    # 总体评估
    print(f"\n[总体评估]")
    status_text = {
        "pass": "✓ 通过",
        "warn": "⚠ 警告",
        "fail": "✗ 失败"
    }.get(comparison.overall_status, "未知")
    print(f"  状态: {status_text}")
    print(f"  说明: {comparison.summary}")

    print("=" * 70)

    return comparison


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: version_compare.py <baseline> <target>")
        sys.exit(1)

    compare_versions(sys.argv[1], sys.argv[2])
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
│  │                      Host Machine (x86/ARM)                            │  │
│  │                                                                        │  │
│  │   ┌───────────────────────────────────────────────────────────────┐   │  │
│  │   │                   Performance Framework                        │   │  │
│  │   │                                                                │   │  │
│  │   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │   │  │
│  │   │  │   Version   │  │    NPU      │  │    PMU      │            │   │  │
│  │   │  │   Tracker   │  │  Profiler   │  │  Analyzer   │            │   │  │
│  │   │  └─────────────┘  └─────────────┘  └─────────────┘            │   │  │
│  │   │                                                                │   │  │
│  │   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │   │  │
│  │   │  │  Metadata   │  │  Comparison │  │   Report    │            │   │  │
│  │   │  │   Store     │  │   Engine    │  │  Generator  │            │   │  │
│  │   │  └─────────────┘  └─────────────┘  └─────────────┘            │   │  │
│  │   └───────────────────────────────────────────────────────────────┘   │  │
│  │                                    │                                   │  │
│  │   ┌────────────────────────────────┼────────────────────────────────┐ │  │
│  │   │                                ▼                                 │ │  │
│  │   │  ┌───────────┐  ┌───────────────────────┐  ┌───────────┐       │ │  │
│  │   │  │    ESL    │  │         NPU           │  │   FPGA    │       │ │  │
│  │   │  │  Simulator│  │  ┌─────────────────┐  │  │   Board   │       │ │  │
│  │   │  │           │  │  │ C Instrument    │  │  │           │       │ │  │
│  │   │  │  (SystemC)│  │  │ Stub (embedded) │  │  │ (Xilinx/  │       │ │  │
│  │   │  │           │  │  └─────────────────┘  │  │  Intel)   │       │ │  │
│  │   │  └───────────┘  └───────────────────────┘  └───────────┘       │ │  │
│  │   │                                                                  │ │  │
│  │   │                    Hardware Targets                              │ │  │
│  │   └──────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                        │  │
│  │   ┌───────────────────────────────────────────────────────────────┐   │  │
│  │   │                   Storage & Database                           │   │  │
│  │   │                                                                │   │  │
│  │   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │   │  │
│  │   │  │  Metadata   │  │   Trace     │  │   Report    │            │   │  │
│  │   │  │   (SQLite)  │  │  (JSON/Bin) │  │  (HTML/PDF) │            │   │  │
│  │   │  └─────────────┘  └─────────────┘  └─────────────┘            │   │  │
│  │   └───────────────────────────────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. 硬件与软件要求

### 2.1 主机要求

| 组件 | 要求 |
|------|------|
| CPU | x86_64 或 ARM64，支持PMU |
| 内存 | 16GB+ (大型追踪需32GB+) |
| 存储 | SSD，100GB+ |
| OS | Linux 4.x+ (需要perf_event支持) |

### 2.2 目标硬件

| 硬件类型 | 支持型号 | 说明 |
|----------|----------|------|
| NPU | 自研NPU设备 | 需要驱动支持性能采集接口 |
| FPGA | Xilinx Zynq/VCK, Intel Agilex | 需要Vivado/Quartus工具链 |
| ESL | SystemC模型 | 需要SystemC 2.3+ |

### 2.3 软件依赖

```yaml
# 必需依赖
python: ">=3.8"
dependencies:
  - numpy>=1.20
  - pandas>=1.3
  - sqlalchemy>=1.4
  - pyyaml>=5.4
  - jinja2>=3.0

# 可选依赖
optional:
  npu_profiling:
    - npu-driver>=1.0
    - npu-tools>=1.0

  pmu_analysis:
    - linux-perf>=5.x
    - pyperf>=2.0

  esl_analysis:
    - systemc>=2.3
    - verilator>=4.0

  fpga_analysis:
    - vivado>=2021.x  # Xilinx
    - quartus>=21.x   # Intel
```

---

# 五、场景视图 (Scenarios View)

## 1. 核心用例

### UC-PERF-01: ESL接口版本变更追踪

```python
# 用例: 追踪ESL模型接口变化对性能的影响

from aitest.performance import VersionTracker, PerformanceAnalyzer

def test_esl_interface_change():
    """测试ESL接口变更对性能的影响"""

    tracker = VersionTracker()
    analyzer = PerformanceAnalyzer()

    # 注册基线版本
    baseline_esl = tracker.register_esl_version("esl_model_v1.0")

    # 注册新版本
    new_esl = tracker.register_esl_version("esl_model_v1.1")

    # 获取接口差异
    diff = tracker.get_version_diff(baseline_esl, new_esl)

    # 验证兼容性
    assert diff.is_backward_compatible, \
        f"接口不兼容: {diff.breaking_changes}"

    # 运行性能测试
    baseline_perf = analyzer.run_benchmark("esl_model_v1.0")
    new_perf = analyzer.run_benchmark("esl_model_v1.1")

    # 对比性能
    comparison = analyzer.compare_versions(baseline_perf, new_perf)

    # 验证无性能回归
    for reg in comparison.regressions:
        assert reg.change_percent < 10, \
            f"性能回归: {reg.metric} 下降 {reg.change_percent}%"
```

### UC-PERF-02: NPU算子性能分析

```python
# 用例: 分析NPU模型算子级性能

from aitest.performance.npu import NPUProfiler, NPUProfileConfig

def test_npu_operator_performance():
    """测试NPU算子性能"""

    profiler = NPUProfiler(device="npu0")
    profiler.configure(NPUProfileConfig(
        trace_level="operator",
        op_breakdown=True
    ))

    # 执行带性能分析的推理
    profiler.start_profiling()
    model.inference(test_input)
    result = profiler.stop_profiling()

    # 验证整体利用率
    assert result.overall_utilization > 70, \
        f"NPU利用率过低: {result.overall_utilization}%"

    # 检查内存瓶颈算子
    memory_bound_ratio = len(result.memory_bound_ops) / len(result.operator_stats)
    assert memory_bound_ratio < 0.3, \
        f"内存瓶颈算子过多: {memory_bound_ratio*100}%"

    # 验证关键算子性能
    conv_ops = [op for op in result.operator_stats if "conv" in op.op_type]
    for op in conv_ops:
        assert op.compute_utilization > 80, \
            f"Conv算子利用率低: {op.op_name} = {op.compute_utilization}%"
```

### UC-PERF-03: CPU PMU热点分析

```python
# 用例: 使用PMU分析CPU热点

from aitest.performance.pmu import PMUAnalyzer

def test_cpu_hotspot_analysis():
    """CPU热点分析测试"""

    analyzer = PMUAnalyzer()
    analyzer.configure_standard_events()

    # 执行被测代码
    analyzer.start_counting()
    run_cpu_intensive_task()
    counters = analyzer.stop_counting()

    result = analyzer.analyze(counters)

    # 验证IPC
    assert result.counters.ipc > 1.5, \
        f"IPC过低: {result.counters.ipc}"

    # 验证缓存效率
    l3_miss_rate = result.counters.l3_cache_misses / result.counters.memory_loads
    assert l3_miss_rate < 0.05, \
        f"L3缓存命中率过低: {(1-l3_miss_rate)*100}%"

    # 检查严重瓶颈
    severe_bottlenecks = [b for b in result.bottlenecks if b.severity == "high"]
    assert len(severe_bottlenecks) == 0, \
        f"存在严重性能瓶颈: {[b.description for b in severe_bottlenecks]}"
```

### UC-PERF-04: 跨版本性能对比

```yaml
# CI配置: 版本性能对比

performance_gate:
  # 基线配置
  baseline_version: "${BASELINE_VERSION}"
  target_version: "${BUILD_VERSION}"

  # 回归阈值
  thresholds:
    npu_latency_p99: 10%     # 延迟增加不超过10%
    npu_throughput: -5%       # 吞吐量下降不超过5%
    cpu_ipc: -10%             # IPC下降不超过10%
    memory_bandwidth: -5%     # 带宽下降不超过5%

  # 接口兼容性
  interface_check:
    require_backward_compatible: true
    allow_new_ports: true
    forbid_port_removal: true

  # 报告配置
  report:
    format: html
    output: ./perf_comparison_report.html
    include_trend: true
```

## 2. 场景验证矩阵

| 场景 | 覆盖需求 | 验证指标 |
|------|----------|----------|
| ESL版本追踪 | MODEL-005 | 接口兼容性、性能变化 |
| FPGA版本追踪 | MODEL-005 | 资源使用、时序变化 |
| NPU性能分析 | MODEL-005-01 | 算子耗时、利用率 |
| PMU分析 | MODEL-005-02 | IPC、缓存命中率、分支预测 |
| 版本对比 | MODEL-005-04 | 性能回归检测 |
| C维测桩 | MODEL-006 | 嵌入式性能采集 |

---

## 需求追溯

| 需求ID | 需求名称 | 模块功能 |
|--------|----------|----------|
| MODEL-005 | 推理性能测试 | VersionTracker, NPUProfiler, PMUAnalyzer |
| MODEL-005-01 | 推理延迟测量 | NPU算子级性能分析 |
| MODEL-005-02 | 吞吐量测试 | PMU计数器分析 |
| MODEL-005-03 | 资源使用测量 | 内存带宽、利用率监控 |
| MODEL-005-04 | 性能基线对比 | MetadataStore, VersionComparison |
| MODEL-006 | 压力测试 | C语言维测桩 |

---

*本文档为AI测试框架性能测试模块设计，专注于硬件验证场景下的ESL/FPGA版本追踪、NPU性能分析、CPU PMU分析和C语言维测桩设计。*
