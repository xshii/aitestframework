# 核心框架模块详细设计 (Core Framework)

## 模块概述

| 属性 | 值 |
|------|-----|
| **模块ID** | CORE |
| **模块名称** | 核心框架 |
| **职责** | 测试框架的核心引擎和基础设施 |
| **需求覆盖** | CORE-001 ~ CORE-008 |

---

## 1. 逻辑视图

### 1.1 核心类图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Core Framework Classes                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                          TestEngine                                  │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │ - config: FrameworkConfig                                           │    │
│  │ - discovery: ITestDiscovery                                         │    │
│  │ - scheduler: ITestScheduler                                         │    │
│  │ - executor: ITestExecutor                                           │    │
│  │ - lifecycle: LifecycleManager                                       │    │
│  │ - plugin_manager: PluginManager                                     │    │
│  │ - reporter: ReportEngine                                            │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │ + initialize(config_path: Path) -> None                             │    │
│  │ + discover_tests(paths: List[Path]) -> List[TestCase]               │    │
│  │ + run(tests: List[TestCase]) -> TestResults                         │    │
│  │ + run_single(test: TestCase) -> TestResult                          │    │
│  │ + shutdown() -> None                                                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                       │
│                    ┌────────────────┼────────────────┐                      │
│                    ▼                ▼                ▼                      │
│  ┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐ │
│  │  TestDiscoveryEngine │ │   TestScheduler      │ │    TestExecutor      │ │
│  ├──────────────────────┤ ├──────────────────────┤ ├──────────────────────┤ │
│  │ - collectors: List   │ │ - strategy: Strategy │ │ - context: Context   │ │
│  │ - filters: List      │ │ - dependency_graph   │ │ - hooks: HookManager │ │
│  │ - patterns: List     │ │ - worker_count: int  │ │ - timeout: int       │ │
│  ├──────────────────────┤ ├──────────────────────┤ ├──────────────────────┤ │
│  │ + discover()         │ │ + schedule()         │ │ + execute()          │ │
│  │ + filter()           │ │ + add_dependency()   │ │ + setup()            │ │
│  │ + register_collector │ │ + get_next_batch()   │ │ + teardown()         │ │
│  └──────────────────────┘ └──────────────────────┘ └──────────────────────┘ │
│                                                                             │
│  ┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐ │
│  │  LifecycleManager    │ │   FixtureManager     │ │    ConfigLoader      │ │
│  ├──────────────────────┤ ├──────────────────────┤ ├──────────────────────┤ │
│  │ - hooks: Dict        │ │ - fixtures: Dict     │ │ - config_paths: List │ │
│  │ - state: State       │ │ - scopes: Dict       │ │ - env_prefix: str    │ │
│  ├──────────────────────┤ ├──────────────────────┤ ├──────────────────────┤ │
│  │ + on_suite_start()   │ │ + register()         │ │ + load()             │ │
│  │ + on_test_start()    │ │ + get()              │ │ + merge()            │ │
│  │ + on_test_end()      │ │ + setup_scope()      │ │ + validate()         │ │
│  │ + on_suite_end()     │ │ + teardown_scope()   │ │ + get_value()        │ │
│  └──────────────────────┘ └──────────────────────┘ └──────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 关键接口定义

```python
# core/interfaces.py

from typing import Protocol, List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

class ITestDiscovery(Protocol):
    """测试发现接口"""

    def discover(self, paths: List[Path]) -> List['TestCase']:
        """发现测试用例"""
        ...

    def filter(self, tests: List['TestCase'], criteria: 'FilterCriteria') -> List['TestCase']:
        """过滤测试用例"""
        ...

    def register_collector(self, collector: 'ITestCollector') -> None:
        """注册收集器"""
        ...


class ITestScheduler(Protocol):
    """测试调度接口"""

    def schedule(self, tests: List['TestCase']) -> 'ExecutionPlan':
        """生成执行计划"""
        ...

    def get_next_batch(self) -> List['TestCase']:
        """获取下一批待执行测试"""
        ...

    def set_strategy(self, strategy: 'ScheduleStrategy') -> None:
        """设置调度策略"""
        ...


class ITestExecutor(Protocol):
    """测试执行接口"""

    def execute(self, test: 'TestCase', context: 'TestContext') -> 'TestResult':
        """执行单个测试"""
        ...

    def setup(self, test: 'TestCase') -> None:
        """测试前置"""
        ...

    def teardown(self, test: 'TestCase', result: 'TestResult') -> None:
        """测试后置"""
        ...


class ILifecycleHook(Protocol):
    """生命周期钩子接口"""

    def on_event(self, event: str, context: Dict[str, Any]) -> Optional[Any]:
        """处理生命周期事件"""
        ...
```

### 1.3 数据模型

```python
# core/models.py

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional, Callable

class TestStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class TestCase:
    """测试用例"""
    id: str
    name: str
    path: Path
    function: Callable
    tags: List[str] = field(default_factory=list)
    timeout: Optional[int] = None
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    markers: List[str] = field(default_factory=list)


@dataclass
class TestResult:
    """测试结果"""
    test_id: str
    status: TestStatus
    duration: float
    started_at: datetime
    finished_at: datetime
    error: Optional['ExceptionInfo'] = None
    output: Dict[str, Any] = field(default_factory=dict)
    artifacts: List['Artifact'] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)


@dataclass
class TestContext:
    """测试上下文"""
    test_case: TestCase
    config: 'FrameworkConfig'
    fixtures: Dict[str, Any]
    temp_dir: Path
    artifacts_dir: Path


@dataclass
class FrameworkConfig:
    """框架配置"""
    test_paths: List[Path]
    parallel_workers: int = 1
    timeout: int = 300
    retry_count: int = 0
    log_level: str = "INFO"
    report_formats: List[str] = field(default_factory=lambda: ["console"])
    plugins: List[str] = field(default_factory=list)
```

---

## 2. 进程视图

### 2.1 测试执行流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Test Execution Process Flow                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         Main Process                                 │    │
│  │                                                                      │    │
│  │   [Initialize] ──► [Discover] ──► [Schedule] ──► [Coordinate]       │    │
│  │                                                       │              │    │
│  │                                          ┌────────────┴────────────┐│    │
│  │                                          ▼                         ▼│    │
│  │                                   [Monitor Progress]    [Collect Results]│
│  │                                          │                         │ │    │
│  │                                          └────────────┬────────────┘│    │
│  │                                                       ▼             │    │
│  │                                               [Generate Report]     │    │
│  │                                                                      │    │
│  └──────────────────────────────────┬───────────────────────────────────┘    │
│                                     │                                       │
│                    ┌────────────────┼────────────────┐                      │
│                    │                │                │                      │
│                    ▼                ▼                ▼                      │
│  ┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐ │
│  │   Worker Process 1   │ │   Worker Process 2   │ │   Worker Process N   │ │
│  │                      │ │                      │ │                      │ │
│  │  ┌────────────────┐  │ │  ┌────────────────┐  │ │  ┌────────────────┐  │ │
│  │  │  Task Queue    │  │ │  │  Task Queue    │  │ │  │  Task Queue    │  │ │
│  │  │  Consumer      │  │ │  │  Consumer      │  │ │  │  Consumer      │  │ │
│  │  └───────┬────────┘  │ │  └───────┬────────┘  │ │  └───────┬────────┘  │ │
│  │          │           │ │          │           │ │          │           │ │
│  │          ▼           │ │          ▼           │ │          ▼           │ │
│  │  ┌────────────────┐  │ │  ┌────────────────┐  │ │  ┌────────────────┐  │ │
│  │  │ Test Executor  │  │ │  │ Test Executor  │  │ │  │ Test Executor  │  │ │
│  │  │                │  │ │  │                │  │ │  │                │  │ │
│  │  │ setup()        │  │ │  │ setup()        │  │ │  │ setup()        │  │ │
│  │  │ execute()      │  │ │  │ execute()      │  │ │  │ execute()      │  │ │
│  │  │ teardown()     │  │ │  │ teardown()     │  │ │  │ teardown()     │  │ │
│  │  └───────┬────────┘  │ │  └───────┬────────┘  │ │  └───────┬────────┘  │ │
│  │          │           │ │          │           │ │          │           │ │
│  │          ▼           │ │          ▼           │ │          ▼           │ │
│  │  ┌────────────────┐  │ │  ┌────────────────┐  │ │  ┌────────────────┐  │ │
│  │  │ Result Queue   │  │ │  │ Result Queue   │  │ │  │ Result Queue   │  │ │
│  │  │ Producer       │  │ │  │ Producer       │  │ │  │ Producer       │  │ │
│  │  └────────────────┘  │ │  └────────────────┘  │ │  └────────────────┘  │ │
│  └──────────────────────┘ └──────────────────────┘ └──────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 调度策略

| 策略 | 描述 | 适用场景 |
|------|------|----------|
| `sequential` | 顺序执行 | 有依赖关系的测试 |
| `parallel` | 并行执行 | 独立测试 |
| `priority` | 按优先级 | 快速失败 |
| `dependency` | 依赖拓扑排序 | 复杂依赖 |

---

## 3. 开发视图

### 3.1 包结构

```
aitest/core/
├── __init__.py              # 导出公共API
├── engine.py                # TestEngine 主引擎
├── discovery.py             # 测试发现
├── scheduler.py             # 调度器
├── executor.py              # 执行器
├── lifecycle.py             # 生命周期管理
├── fixture.py               # Fixture机制
├── context.py               # 测试上下文
├── result.py                # 结果模型
├── exceptions.py            # 异常定义
├── decorators.py            # 装饰器
├── markers.py               # 标记系统
└── config/
    ├── __init__.py
    ├── loader.py            # 配置加载
    ├── schema.py            # 配置Schema
    └── defaults.py          # 默认配置
```

### 3.2 核心实现示例

```python
# core/engine.py

from typing import List, Optional
from pathlib import Path
import logging

from .discovery import TestDiscoveryEngine
from .scheduler import TestScheduler
from .executor import TestExecutor
from .lifecycle import LifecycleManager
from .config import ConfigLoader, FrameworkConfig
from .result import TestResults

logger = logging.getLogger(__name__)


class TestEngine:
    """AI测试框架核心引擎"""

    def __init__(self, config: Optional[FrameworkConfig] = None):
        self.config = config or FrameworkConfig()
        self._initialized = False

        # 核心组件
        self.discovery = TestDiscoveryEngine()
        self.scheduler = TestScheduler()
        self.executor = TestExecutor()
        self.lifecycle = LifecycleManager()

    def initialize(self, config_path: Optional[Path] = None) -> None:
        """初始化引擎"""
        if self._initialized:
            return

        logger.info("Initializing TestEngine")

        # 加载配置
        if config_path:
            loader = ConfigLoader()
            self.config = loader.load(config_path)

        # 初始化组件
        self.discovery.initialize(self.config)
        self.scheduler.initialize(self.config)
        self.executor.initialize(self.config)
        self.lifecycle.initialize(self.config)

        self._initialized = True
        logger.info("TestEngine initialized successfully")

    def discover_tests(self, paths: Optional[List[Path]] = None) -> List['TestCase']:
        """发现测试用例"""
        self._ensure_initialized()

        search_paths = paths or self.config.test_paths
        logger.info(f"Discovering tests in: {search_paths}")

        # 触发生命周期钩子
        self.lifecycle.trigger("before_collection", {"paths": search_paths})

        tests = self.discovery.discover(search_paths)

        self.lifecycle.trigger("after_collection", {"tests": tests})

        logger.info(f"Discovered {len(tests)} tests")
        return tests

    def run(self, tests: Optional[List['TestCase']] = None) -> TestResults:
        """运行测试"""
        self._ensure_initialized()

        if tests is None:
            tests = self.discover_tests()

        logger.info(f"Running {len(tests)} tests")

        # 触发生命周期钩子
        self.lifecycle.trigger("before_suite", {"tests": tests})

        # 创建执行计划
        plan = self.scheduler.schedule(tests)

        # 执行测试
        results = TestResults()
        for batch in plan.batches:
            batch_results = self._execute_batch(batch)
            results.extend(batch_results)

        # 触发生命周期钩子
        self.lifecycle.trigger("after_suite", {"results": results})

        return results

    def _execute_batch(self, batch: List['TestCase']) -> List['TestResult']:
        """执行一批测试"""
        results = []
        for test in batch:
            result = self.executor.execute(test)
            results.append(result)
        return results

    def _ensure_initialized(self) -> None:
        """确保已初始化"""
        if not self._initialized:
            self.initialize()

    def shutdown(self) -> None:
        """关闭引擎"""
        logger.info("Shutting down TestEngine")
        self.executor.shutdown()
        self.lifecycle.shutdown()
        self._initialized = False
```

### 3.3 轻量级独立测试运行器 (多底座支持)

针对不同运行底座 (FPGA/NPU/GPU/CPU)，提供不依赖 pytest 的轻量级测试运行器，通过分层抽象支持多底座适配。

#### 3.3.1 设计目标

| 目标 | 说明 |
|------|------|
| **多底座支持** | 支持 FPGA、NPU、GPU、CPU 等不同运行底座 |
| **生命周期分离** | 初始化/执行/结束/分析/切换/报告 各阶段解耦 |
| **公共抽象** | 定义统一接口，底座实现可插拔 |
| **Fail-Continue** | 单个用例失败不影响后续用例执行 |
| **软断言支持** | 支持收集单个用例内的多个失败 |
| **零外部依赖** | 核心运行器仅使用 Python 标准库 |

#### 3.3.2 生命周期分层架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Test Lifecycle Architecture                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    TestRunner (公共编排层)                             │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────┐│  │
│  │  │ 用例    │ │ 用例    │ │ 用例    │ │ 结果    │ │ 用例    │ │ 报告 ││  │
│  │  │ 初始化  │→│ 执行    │→│ 结束    │→│ 分析    │→│ 切换    │→│ 生成 ││  │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └──┬───┘│  │
│  └───────┼──────────┼──────────┼──────────┼──────────┼─────────────┼────┘  │
│          │          │          │          │          │             │        │
│          ▼          ▼          ▼          ▼          ▼             ▼        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    抽象接口层 (ITestLifecycle)                         │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐  │  │
│  │  │ ITestSetup  │ │ ITestExec   │ │ ITestCleanup│ │ IResultAnalyzer │  │  │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────────┘  │  │
│  │  ┌─────────────┐ ┌─────────────┐                                      │  │
│  │  │ ITestSwitch │ │ IReporter   │                                      │  │
│  │  └─────────────┘ └─────────────┘                                      │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│          │          │          │          │          │             │        │
│          ▼          ▼          ▼          ▼          ▼             ▼        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      底座实现层 (Platform Backends)                    │  │
│  │                                                                       │  │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐             │  │
│  │  │  FPGABackend  │  │   NPUBackend  │  │   GPUBackend  │  ...        │  │
│  │  │               │  │               │  │               │             │  │
│  │  │ - fpga_init() │  │ - npu_init()  │  │ - cuda_init() │             │  │
│  │  │ - fpga_exec() │  │ - npu_exec()  │  │ - cuda_exec() │             │  │
│  │  │ - fpga_clean()│  │ - npu_clean() │  │ - cuda_clean()│             │  │
│  │  └───────────────┘  └───────────────┘  └───────────────┘             │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 3.3.3 生命周期阶段定义

| 阶段 | 职责 | 公共行为 | 底座差异 |
|------|------|----------|----------|
| **用例初始化** | 准备测试环境和数据 | 加载配置、分配资源 | 设备初始化、内存分配方式 |
| **用例执行** | 运行测试逻辑 | 异常捕获、超时控制 | 算子调用方式、数据传输 |
| **用例结束** | 清理资源 | 释放临时资源、状态重置 | 设备资源释放、内存回收 |
| **结果分析** | 分析测试结果 | 精度比对、状态判定 | 性能指标采集方式 |
| **用例切换** | 切换到下一个用例 | 状态保存、上下文切换 | 设备状态管理 |
| **用例报告** | 生成测试报告 | 格式化输出、汇总统计 | 平台特定指标展示 |

#### 3.3.4 抽象接口定义

```python
# core/lifecycle/interfaces.py

"""
测试生命周期抽象接口

各底座需要实现这些接口以支持测试执行。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class TestStatus(Enum):
    """测试状态"""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    SKIP = "SKIP"


@dataclass
class TestContext:
    """测试上下文 - 在各阶段间传递"""
    test_name: str
    test_id: str
    config: Dict[str, Any]
    platform: str  # fpga, npu, gpu, cpu

    # 运行时数据
    inputs: Dict[str, Any] = None
    outputs: Dict[str, Any] = None
    golden: Dict[str, Any] = None

    # 结果数据
    status: TestStatus = TestStatus.PENDING
    error: Optional[str] = None
    metrics: Dict[str, Any] = None

    # 底座特定数据
    platform_data: Dict[str, Any] = None


@dataclass
class TestResult:
    """测试结果"""
    test_name: str
    status: TestStatus
    duration_ms: float
    error: Optional[str] = None
    soft_failures: List[str] = None
    metrics: Dict[str, Any] = None
    platform_metrics: Dict[str, Any] = None


# ============================================
# 1. 用例初始化接口
# ============================================

class ITestSetup(ABC):
    """用例初始化接口"""

    @abstractmethod
    def setup_platform(self, ctx: TestContext) -> None:
        """平台/设备初始化"""
        pass

    @abstractmethod
    def setup_data(self, ctx: TestContext) -> None:
        """数据准备 (输入、Golden)"""
        pass

    @abstractmethod
    def setup_resources(self, ctx: TestContext) -> None:
        """资源分配 (内存、流等)"""
        pass

    def setup(self, ctx: TestContext) -> None:
        """完整初始化流程 (模板方法)"""
        self.setup_platform(ctx)
        self.setup_data(ctx)
        self.setup_resources(ctx)


# ============================================
# 2. 用例执行接口
# ============================================

class ITestExecutor(ABC):
    """用例执行接口"""

    @abstractmethod
    def execute(self, ctx: TestContext) -> None:
        """执行测试逻辑"""
        pass

    @abstractmethod
    def get_timeout(self, ctx: TestContext) -> float:
        """获取超时时间 (秒)"""
        pass

    def execute_with_timeout(self, ctx: TestContext) -> None:
        """带超时的执行 (公共实现)"""
        import signal

        timeout = self.get_timeout(ctx)

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Test {ctx.test_name} timed out after {timeout}s")

        # 设置超时 (仅 Unix)
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))

        try:
            self.execute(ctx)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


# ============================================
# 3. 用例结束接口
# ============================================

class ITestCleanup(ABC):
    """用例结束/清理接口"""

    @abstractmethod
    def cleanup_resources(self, ctx: TestContext) -> None:
        """释放资源"""
        pass

    @abstractmethod
    def cleanup_platform(self, ctx: TestContext) -> None:
        """平台清理"""
        pass

    def cleanup(self, ctx: TestContext) -> None:
        """完整清理流程 (模板方法)"""
        self.cleanup_resources(ctx)
        self.cleanup_platform(ctx)


# ============================================
# 4. 结果分析接口
# ============================================

class IResultAnalyzer(ABC):
    """结果分析接口"""

    @abstractmethod
    def analyze_precision(self, ctx: TestContext) -> Dict[str, Any]:
        """精度分析"""
        pass

    @abstractmethod
    def analyze_performance(self, ctx: TestContext) -> Dict[str, Any]:
        """性能分析"""
        pass

    @abstractmethod
    def determine_status(self, ctx: TestContext) -> TestStatus:
        """判定测试状态"""
        pass

    def analyze(self, ctx: TestContext) -> TestResult:
        """完整分析流程 (模板方法)"""
        precision = self.analyze_precision(ctx)
        performance = self.analyze_performance(ctx)
        status = self.determine_status(ctx)

        return TestResult(
            test_name=ctx.test_name,
            status=status,
            duration_ms=performance.get("duration_ms", 0),
            error=ctx.error,
            metrics=precision,
            platform_metrics=performance,
        )


# ============================================
# 5. 用例切换接口
# ============================================

class ITestSwitch(ABC):
    """用例切换接口"""

    @abstractmethod
    def save_state(self, ctx: TestContext) -> Dict[str, Any]:
        """保存当前状态"""
        pass

    @abstractmethod
    def restore_state(self, state: Dict[str, Any], ctx: TestContext) -> None:
        """恢复状态"""
        pass

    @abstractmethod
    def prepare_next(self, current_ctx: TestContext, next_ctx: TestContext) -> None:
        """准备下一个用例"""
        pass


# ============================================
# 6. 报告生成接口
# ============================================

class IReporter(ABC):
    """报告生成接口"""

    @abstractmethod
    def report_test(self, result: TestResult) -> str:
        """单个用例报告"""
        pass

    @abstractmethod
    def report_summary(self, results: List[TestResult]) -> str:
        """汇总报告"""
        pass

    @abstractmethod
    def export(self, results: List[TestResult], path: str, format: str) -> None:
        """导出报告 (json/html/csv)"""
        pass


# ============================================
# 组合接口: 完整生命周期
# ============================================

class ITestLifecycle(ITestSetup, ITestExecutor, ITestCleanup,
                      IResultAnalyzer, ITestSwitch, ABC):
    """完整测试生命周期接口"""

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """底座名称"""
        pass
```

#### 3.3.5 底座实现示例

```python
# core/lifecycle/backends/fpga.py

"""FPGA 底座实现"""

from typing import Any, Dict
from ..interfaces import (
    ITestLifecycle, TestContext, TestResult, TestStatus
)


class FPGABackend(ITestLifecycle):
    """FPGA 底座实现"""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self._device = None
        self._memory_pool = None

    @property
    def platform_name(self) -> str:
        return "fpga"

    # ========== 1. 用例初始化 ==========

    def setup_platform(self, ctx: TestContext) -> None:
        """FPGA 设备初始化"""
        # 打开 FPGA 设备
        self._device = self._open_device(self.device_id)
        # 加载 bitstream (如果需要)
        bitstream = ctx.config.get("bitstream")
        if bitstream:
            self._load_bitstream(bitstream)

        ctx.platform_data = {"device": self._device}

    def setup_data(self, ctx: TestContext) -> None:
        """加载测试数据到 FPGA 内存"""
        import numpy as np

        # 加载输入数据
        input_path = ctx.config.get("input_path")
        ctx.inputs = {"data": np.load(input_path)}

        # 加载 Golden 数据
        golden_path = ctx.config.get("golden_path")
        ctx.golden = {"data": np.load(golden_path)}

        # 将输入数据传输到 FPGA
        ctx.platform_data["input_addr"] = self._alloc_and_copy(
            ctx.inputs["data"]
        )

    def setup_resources(self, ctx: TestContext) -> None:
        """分配 FPGA 资源"""
        output_shape = ctx.config.get("output_shape")
        output_dtype = ctx.config.get("output_dtype", "float32")

        # 分配输出缓冲区
        ctx.platform_data["output_addr"] = self._alloc_buffer(
            output_shape, output_dtype
        )

    # ========== 2. 用例执行 ==========

    def execute(self, ctx: TestContext) -> None:
        """执行 FPGA 算子"""
        import time

        input_addr = ctx.platform_data["input_addr"]
        output_addr = ctx.platform_data["output_addr"]

        start = time.perf_counter()

        # 触发 FPGA 计算
        self._run_kernel(
            kernel_name=ctx.config.get("kernel"),
            input_addr=input_addr,
            output_addr=output_addr,
        )

        # 等待完成
        self._wait_complete()

        duration_ms = (time.perf_counter() - start) * 1000

        # 读取输出
        ctx.outputs = {
            "data": self._read_buffer(output_addr)
        }
        ctx.platform_data["duration_ms"] = duration_ms

    def get_timeout(self, ctx: TestContext) -> float:
        return ctx.config.get("timeout", 60.0)

    # ========== 3. 用例结束 ==========

    def cleanup_resources(self, ctx: TestContext) -> None:
        """释放 FPGA 内存"""
        if ctx.platform_data:
            if "input_addr" in ctx.platform_data:
                self._free_buffer(ctx.platform_data["input_addr"])
            if "output_addr" in ctx.platform_data:
                self._free_buffer(ctx.platform_data["output_addr"])

    def cleanup_platform(self, ctx: TestContext) -> None:
        """FPGA 设备清理"""
        # 可选: 重置设备状态
        pass

    # ========== 4. 结果分析 ==========

    def analyze_precision(self, ctx: TestContext) -> Dict[str, Any]:
        """精度分析"""
        import numpy as np
        from aidevtools.tools.compare import compare_isclose

        result = ctx.outputs["data"]
        golden = ctx.golden["data"]

        comparison = compare_isclose(
            golden=golden,
            result=result,
            atol=ctx.config.get("atol", 1e-5),
            rtol=ctx.config.get("rtol", 1e-3),
        )

        return {
            "passed": comparison.passed,
            "exceed_ratio": comparison.exceed_ratio,
            "max_abs_error": comparison.max_abs_error,
        }

    def analyze_performance(self, ctx: TestContext) -> Dict[str, Any]:
        """性能分析"""
        return {
            "duration_ms": ctx.platform_data.get("duration_ms", 0),
            "platform": "fpga",
            "device_id": self.device_id,
        }

    def determine_status(self, ctx: TestContext) -> TestStatus:
        """判定状态"""
        if ctx.error:
            return TestStatus.ERROR

        precision = self.analyze_precision(ctx)
        if precision["passed"]:
            return TestStatus.PASS
        return TestStatus.FAIL

    # ========== 5. 用例切换 ==========

    def save_state(self, ctx: TestContext) -> Dict[str, Any]:
        """保存 FPGA 状态"""
        return {
            "device_id": self.device_id,
            "test_name": ctx.test_name,
        }

    def restore_state(self, state: Dict[str, Any], ctx: TestContext) -> None:
        """恢复状态"""
        pass  # FPGA 通常不需要状态恢复

    def prepare_next(self, current_ctx: TestContext, next_ctx: TestContext) -> None:
        """准备下一个用例"""
        # 清理当前用例资源
        self.cleanup(current_ctx)

    # ========== 私有方法 (FPGA 特定) ==========

    def _open_device(self, device_id: int):
        """打开 FPGA 设备"""
        # 实际实现依赖具体 FPGA SDK
        pass

    def _load_bitstream(self, path: str):
        """加载 bitstream"""
        pass

    def _alloc_and_copy(self, data):
        """分配内存并拷贝数据"""
        pass

    def _alloc_buffer(self, shape, dtype):
        """分配缓冲区"""
        pass

    def _free_buffer(self, addr):
        """释放缓冲区"""
        pass

    def _run_kernel(self, kernel_name, input_addr, output_addr):
        """运行 kernel"""
        pass

    def _wait_complete(self):
        """等待完成"""
        pass

    def _read_buffer(self, addr):
        """读取缓冲区"""
        pass


# core/lifecycle/backends/npu.py

"""NPU 底座实现"""

class NPUBackend(ITestLifecycle):
    """NPU (昇腾) 底座实现"""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self._context = None
        self._stream = None

    @property
    def platform_name(self) -> str:
        return "npu"

    def setup_platform(self, ctx: TestContext) -> None:
        """NPU 设备初始化"""
        # import acl  # 昇腾 ACL
        # acl.init()
        # acl.rt.set_device(self.device_id)
        # self._context, _ = acl.rt.create_context(self.device_id)
        # self._stream, _ = acl.rt.create_stream()
        pass

    def setup_data(self, ctx: TestContext) -> None:
        """加载数据到 NPU"""
        # 数据加载逻辑
        pass

    def setup_resources(self, ctx: TestContext) -> None:
        """分配 NPU 资源"""
        pass

    def execute(self, ctx: TestContext) -> None:
        """执行 NPU 算子"""
        pass

    def get_timeout(self, ctx: TestContext) -> float:
        return ctx.config.get("timeout", 60.0)

    def cleanup_resources(self, ctx: TestContext) -> None:
        """释放 NPU 资源"""
        pass

    def cleanup_platform(self, ctx: TestContext) -> None:
        """NPU 清理"""
        # if self._stream:
        #     acl.rt.destroy_stream(self._stream)
        # if self._context:
        #     acl.rt.destroy_context(self._context)
        pass

    def analyze_precision(self, ctx: TestContext) -> Dict[str, Any]:
        return {}

    def analyze_performance(self, ctx: TestContext) -> Dict[str, Any]:
        return {"platform": "npu", "device_id": self.device_id}

    def determine_status(self, ctx: TestContext) -> TestStatus:
        return TestStatus.PASS if not ctx.error else TestStatus.ERROR

    def save_state(self, ctx: TestContext) -> Dict[str, Any]:
        return {}

    def restore_state(self, state: Dict[str, Any], ctx: TestContext) -> None:
        pass

    def prepare_next(self, current_ctx: TestContext, next_ctx: TestContext) -> None:
        self.cleanup(current_ctx)


# core/lifecycle/backends/cpu.py

"""CPU 底座实现 (参考实现)"""

class CPUBackend(ITestLifecycle):
    """CPU 底座实现 - 最简单的参考实现"""

    @property
    def platform_name(self) -> str:
        return "cpu"

    def setup_platform(self, ctx: TestContext) -> None:
        pass  # CPU 无需特殊初始化

    def setup_data(self, ctx: TestContext) -> None:
        import numpy as np
        ctx.inputs = {"data": np.load(ctx.config["input_path"])}
        ctx.golden = {"data": np.load(ctx.config["golden_path"])}

    def setup_resources(self, ctx: TestContext) -> None:
        pass

    def execute(self, ctx: TestContext) -> None:
        import time
        import numpy as np

        start = time.perf_counter()

        # 执行 CPU 计算
        func = ctx.config.get("func")
        ctx.outputs = {"data": func(ctx.inputs["data"])}

        ctx.platform_data = {
            "duration_ms": (time.perf_counter() - start) * 1000
        }

    def get_timeout(self, ctx: TestContext) -> float:
        return ctx.config.get("timeout", 300.0)

    def cleanup_resources(self, ctx: TestContext) -> None:
        pass

    def cleanup_platform(self, ctx: TestContext) -> None:
        pass

    def analyze_precision(self, ctx: TestContext) -> Dict[str, Any]:
        import numpy as np

        result = ctx.outputs["data"]
        golden = ctx.golden["data"]

        diff = np.abs(result - golden)
        return {
            "max_abs_error": float(diff.max()),
            "mean_abs_error": float(diff.mean()),
            "passed": np.allclose(result, golden, atol=1e-5, rtol=1e-3),
        }

    def analyze_performance(self, ctx: TestContext) -> Dict[str, Any]:
        return {
            "duration_ms": ctx.platform_data.get("duration_ms", 0),
            "platform": "cpu",
        }

    def determine_status(self, ctx: TestContext) -> TestStatus:
        if ctx.error:
            return TestStatus.ERROR
        precision = self.analyze_precision(ctx)
        return TestStatus.PASS if precision["passed"] else TestStatus.FAIL

    def save_state(self, ctx: TestContext) -> Dict[str, Any]:
        return {}

    def restore_state(self, state: Dict[str, Any], ctx: TestContext) -> None:
        pass

    def prepare_next(self, current_ctx: TestContext, next_ctx: TestContext) -> None:
        pass
```

#### 3.3.6 统一测试运行器

```python
# core/lifecycle/runner.py

"""
统一测试运行器 - 编排测试生命周期

支持:
- 多底座适配
- Fail-Continue
- 软断言
- 可扩展报告
"""

import time
import traceback
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass, field

from .interfaces import (
    ITestLifecycle, IReporter,
    TestContext, TestResult, TestStatus
)


@dataclass
class AssertionFailure:
    """断言失败记录"""
    message: str
    location: str = ""


class SoftAssertionError(AssertionError):
    """软断言错误"""
    def __init__(self, failures: List[AssertionFailure]):
        self.failures = failures
        super().__init__(
            f"{len(failures)} assertion(s) failed:\n" +
            "\n".join(f"  - {f.message}" for f in failures)
        )


class SoftAssertContext:
    """软断言上下文"""

    def __init__(self, raise_on_exit: bool = True):
        self.failures: List[AssertionFailure] = []
        self.raise_on_exit = raise_on_exit

    def __enter__(self) -> 'SoftAssertContext':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is None and self.raise_on_exit and self.failures:
            raise SoftAssertionError(self.failures)
        return False

    def check(self, condition: bool, message: str) -> bool:
        if not condition:
            self.failures.append(AssertionFailure(message=message))
            return False
        return True

    def check_equal(self, actual, expected, name: str = "") -> bool:
        if actual != expected:
            prefix = f"[{name}] " if name else ""
            self.failures.append(AssertionFailure(
                message=f"{prefix}Expected {expected}, got {actual}"
            ))
            return False
        return True

    def check_all_close(self, actual, expected, atol=1e-5, rtol=1e-3, name="") -> bool:
        import numpy as np
        if not np.allclose(actual, expected, atol=atol, rtol=rtol):
            diff = np.abs(actual - expected)
            prefix = f"[{name}] " if name else ""
            self.failures.append(AssertionFailure(
                message=f"{prefix}Arrays not close, max_diff={diff.max():.2e}"
            ))
            return False
        return True

    def has_failures(self) -> bool:
        return len(self.failures) > 0

    def get_failure_messages(self) -> List[str]:
        return [f.message for f in self.failures]


@dataclass
class TestItem:
    """测试项定义"""
    name: str
    config: Dict[str, Any]
    test_func: Optional[Callable[[TestContext], None]] = None


@dataclass
class RunSummary:
    """运行汇总"""
    total: int
    passed: int
    failed: int
    errors: int
    skipped: int
    duration_ms: float
    results: List[TestResult]
    platform: str

    def __str__(self) -> str:
        return (
            f"\n{'='*60}\n"
            f"Platform: {self.platform}\n"
            f"Test Summary: {self.passed}/{self.total} passed\n"
            f"  PASS: {self.passed}, FAIL: {self.failed}, "
            f"ERROR: {self.errors}, SKIP: {self.skipped}\n"
            f"  Duration: {self.duration_ms:.1f} ms\n"
            f"{'='*60}"
        )


class UnifiedTestRunner:
    """
    统一测试运行器

    编排测试生命周期，支持多底座适配。

    Usage:
        # 选择底座
        backend = FPGABackend(device_id=0)
        runner = UnifiedTestRunner(backend)

        # 添加测试
        runner.add_test("test_matmul", {"kernel": "matmul", ...})
        runner.add_test("test_softmax", {"kernel": "softmax", ...})

        # 运行 (fail-continue)
        summary = runner.run_all()
        print(summary)
    """

    def __init__(
        self,
        backend: ITestLifecycle,
        reporter: IReporter = None,
        verbose: bool = True,
    ):
        self.backend = backend
        self.reporter = reporter or ConsoleReporter()
        self.verbose = verbose

        self.tests: List[TestItem] = []
        self.results: List[TestResult] = []

        # 回调
        self.on_test_start: Callable[[TestContext], None] = None
        self.on_test_end: Callable[[TestResult], None] = None

    def add_test(
        self,
        name: str,
        config: Dict[str, Any],
        test_func: Callable[[TestContext], None] = None,
    ) -> 'UnifiedTestRunner':
        """添加测试"""
        self.tests.append(TestItem(name=name, config=config, test_func=test_func))
        return self

    def run_all(self) -> RunSummary:
        """运行所有测试 (fail-continue)"""
        self.results = []
        start_time = time.perf_counter()

        prev_ctx = None
        for i, test in enumerate(self.tests):
            # 创建上下文
            ctx = TestContext(
                test_name=test.name,
                test_id=f"{i}",
                config=test.config,
                platform=self.backend.platform_name,
            )

            # 用例切换
            if prev_ctx:
                self.backend.prepare_next(prev_ctx, ctx)

            # 运行测试
            result = self._run_single(test, ctx)
            self.results.append(result)

            # 报告
            if self.verbose:
                print(self.reporter.report_test(result))

            prev_ctx = ctx

        total_duration = (time.perf_counter() - start_time) * 1000

        return RunSummary(
            total=len(self.results),
            passed=sum(1 for r in self.results if r.status == TestStatus.PASS),
            failed=sum(1 for r in self.results if r.status == TestStatus.FAIL),
            errors=sum(1 for r in self.results if r.status == TestStatus.ERROR),
            skipped=sum(1 for r in self.results if r.status == TestStatus.SKIP),
            duration_ms=total_duration,
            results=self.results,
            platform=self.backend.platform_name,
        )

    def _run_single(self, test: TestItem, ctx: TestContext) -> TestResult:
        """运行单个测试"""
        start_time = time.perf_counter()

        # 回调
        if self.on_test_start:
            self.on_test_start(ctx)

        try:
            # 1. 用例初始化
            self.backend.setup(ctx)

            # 2. 用例执行
            if test.test_func:
                # 使用自定义测试函数
                test.test_func(ctx)
            else:
                # 使用底座默认执行
                self.backend.execute(ctx)

            # 3. 用例结束 (在 finally 中)

            # 4. 结果分析
            result = self.backend.analyze(ctx)
            result.duration_ms = (time.perf_counter() - start_time) * 1000

        except SoftAssertionError as e:
            result = TestResult(
                test_name=test.name,
                status=TestStatus.FAIL,
                duration_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e),
                soft_failures=[f.message for f in e.failures],
            )

        except AssertionError as e:
            result = TestResult(
                test_name=test.name,
                status=TestStatus.FAIL,
                duration_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e),
            )

        except Exception as e:
            result = TestResult(
                test_name=test.name,
                status=TestStatus.ERROR,
                duration_ms=(time.perf_counter() - start_time) * 1000,
                error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            )

        finally:
            # 3. 用例结束 (清理)
            try:
                self.backend.cleanup(ctx)
            except Exception:
                pass  # 忽略清理错误

        # 回调
        if self.on_test_end:
            self.on_test_end(result)

        return result


class ConsoleReporter(IReporter):
    """控制台报告器"""

    def report_test(self, result: TestResult) -> str:
        icon = "✓" if result.status == TestStatus.PASS else "✗"
        line = f"  {icon} {result.test_name} ({result.duration_ms:.1f}ms)"

        if result.error and result.status != TestStatus.PASS:
            # 只显示第一行错误
            first_line = result.error.split('\n')[0][:80]
            line += f"\n    {first_line}"

        return line

    def report_summary(self, results: List[TestResult]) -> str:
        passed = sum(1 for r in results if r.status == TestStatus.PASS)
        failed = sum(1 for r in results if r.status == TestStatus.FAIL)
        errors = sum(1 for r in results if r.status == TestStatus.ERROR)

        return (
            f"\nSummary: {passed}/{len(results)} passed, "
            f"{failed} failed, {errors} errors"
        )

    def export(self, results: List[TestResult], path: str, format: str) -> None:
        import json

        if format == "json":
            data = [
                {
                    "name": r.test_name,
                    "status": r.status.value,
                    "duration_ms": r.duration_ms,
                    "error": r.error,
                    "metrics": r.metrics,
                }
                for r in results
            ]
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
```

#### 3.3.7 使用示例

```python
"""
多底座测试运行示例

展示如何使用 UnifiedTestRunner 在不同底座上运行测试。
"""

from core.lifecycle.runner import UnifiedTestRunner, SoftAssertContext
from core.lifecycle.backends.fpga import FPGABackend
from core.lifecycle.backends.npu import NPUBackend
from core.lifecycle.backends.cpu import CPUBackend
import numpy as np


# ============================================
# 示例1: FPGA 底座测试
# ============================================

def run_fpga_tests():
    """FPGA 算子精度测试"""

    # 1. 选择底座
    backend = FPGABackend(device_id=0)
    runner = UnifiedTestRunner(backend)

    # 2. 添加测试 (配置方式)
    runner.add_test("test_matmul", {
        "kernel": "matmul",
        "input_path": "data/matmul_input.npy",
        "golden_path": "data/matmul_golden.npy",
        "output_shape": [1, 512, 512],
        "atol": 1e-5,
        "rtol": 1e-3,
    })

    runner.add_test("test_softmax", {
        "kernel": "softmax",
        "input_path": "data/softmax_input.npy",
        "golden_path": "data/softmax_golden.npy",
        "output_shape": [1, 1000],
        "atol": 1e-4,
    })

    # 3. 运行 (fail-continue)
    summary = runner.run_all()
    print(summary)

    return summary


# ============================================
# 示例2: 自定义测试函数 + 软断言
# ============================================

def test_transformer_with_soft_assert(ctx):
    """Transformer 层测试 - 软断言收集所有失败"""

    ops = ["layernorm_0", "linear_0", "attention_0", "add_0"]

    with SoftAssertContext() as soft:
        for op_name in ops:
            dut = np.load(f"outputs/{op_name}_dut.npy")
            golden = np.load(f"outputs/{op_name}_golden.npy")

            soft.check_all_close(
                dut, golden,
                atol=1e-5, rtol=1e-3,
                name=op_name
            )


def run_with_custom_func():
    """使用自定义测试函数"""

    backend = FPGABackend(device_id=0)
    runner = UnifiedTestRunner(backend)

    # 使用自定义测试函数
    runner.add_test(
        "test_transformer",
        config={"model": "transformer"},
        test_func=test_transformer_with_soft_assert
    )

    return runner.run_all()


# ============================================
# 示例3: 多底座对比测试
# ============================================

def run_cross_platform_test(test_config):
    """在多个底座上运行相同测试进行对比"""

    results = {}

    for platform, BackendClass in [
        ("fpga", FPGABackend),
        ("npu", NPUBackend),
        ("cpu", CPUBackend),
    ]:
        backend = BackendClass()
        runner = UnifiedTestRunner(backend, verbose=False)
        runner.add_test("precision_test", test_config)

        summary = runner.run_all()
        results[platform] = summary

        print(f"{platform}: {summary.passed}/{summary.total} passed")

    return results


# ============================================
# 示例4: 集成 aidevtools 三列比对
# ============================================

from aidevtools.tools.compare import compare_3col, CompareThresholds


def test_with_3col_compare(ctx):
    """使用 aidevtools 三列比对"""

    thresholds = CompareThresholds(
        fuzzy_atol=1e-5,
        fuzzy_rtol=1e-3,
        fuzzy_min_qsnr=30.0,
    )

    ops = [("MatMul", 0), ("LayerNorm", 0), ("Softmax", 0)]

    with SoftAssertContext() as soft:
        for op_name, op_id in ops:
            dut = ctx.outputs.get(f"{op_name}_{op_id}")
            pure = np.load(f"golden/{op_name}_{op_id}_pure.npy")
            qnt = np.load(f"golden/{op_name}_{op_id}_qnt.npy")

            result = compare_3col(
                op_name=op_name,
                op_id=op_id,
                result=dut,
                golden_pure=pure,
                golden_qnt=qnt,
                thresholds=thresholds,
            )

            soft.check(
                result.status != "FAIL",
                f"{op_name}_{op_id}: {result.status}, qsnr={result.fuzzy_qnt.qsnr:.1f}dB"
            )


# ============================================
# 主程序
# ============================================

if __name__ == "__main__":
    # 运行 FPGA 测试
    summary = run_fpga_tests()

    if summary.failed == 0:
        print("All tests passed!")
    else:
        print(f"Failed: {summary.failed}/{summary.total}")
```

#### 3.3.8 包结构

```
aitest/core/lifecycle/
├── __init__.py
├── interfaces.py          # 抽象接口定义
├── runner.py              # UnifiedTestRunner
├── soft_assert.py         # SoftAssertContext
│
├── backends/              # 底座实现
│   ├── __init__.py
│   ├── base.py            # 基类 (可选)
│   ├── fpga.py            # FPGA 底座
│   ├── npu.py             # NPU 底座
│   ├── gpu.py             # GPU 底座
│   └── cpu.py             # CPU 底座
│
└── reporters/             # 报告生成
    ├── __init__.py
    ├── console.py         # 控制台报告
    ├── json_reporter.py   # JSON 报告
    └── html_reporter.py   # HTML 报告
```

---

## 4. 物理视图

### 4.1 资源需求

| 组件 | CPU | 内存 | 存储 |
|------|-----|------|------|
| Main Process | 2核 | 2GB | 100MB |
| Worker Process | 1核/worker | 1GB/worker | - |
| Log Storage | - | - | 按需 |

### 4.2 部署配置

```yaml
# 核心框架配置示例
core:
  engine:
    parallel_workers: 4
    timeout: 300
    retry:
      enabled: true
      max_attempts: 3
      delay: 1.0

  discovery:
    patterns:
      - "test_*.py"
      - "*_test.py"
    exclude:
      - "__pycache__"
      - ".git"

  scheduler:
    strategy: "parallel"  # sequential, parallel, priority
    max_parallel: 4

  logging:
    level: INFO
    format: "structured"
    output:
      - console
      - file
```

---

## 5. 场景视图

### 5.1 核心用例

**UC-CORE-01: 执行测试套件**

```
参与者: 开发者
前置条件: 已配置测试路径
基本流程:
1. 用户执行 aitest run
2. 引擎初始化配置
3. 发现测试用例
4. 生成执行计划
5. 并行执行测试
6. 收集结果
7. 生成报告
后置条件: 测试完成，报告生成
```

**UC-CORE-02: 过滤测试用例**

```
参与者: 开发者
前置条件: 已有测试用例
基本流程:
1. 用户指定过滤条件 (--tags, --name, --path)
2. 发现所有测试
3. 应用过滤条件
4. 返回匹配的测试
后置条件: 只执行匹配的测试
```

### 5.2 需求追溯

| 需求ID | 实现类/方法 | 测试用例 |
|--------|-------------|----------|
| CORE-001 | `ConfigLoader.load()` | test_config_loading |
| CORE-002 | `TestDiscoveryEngine.discover()` | test_discovery |
| CORE-003 | `TestScheduler.schedule()` | test_scheduling |
| CORE-004 | `LifecycleManager` | test_lifecycle_hooks |
| CORE-005 | `FixtureManager` | test_fixtures |
| CORE-006 | `TestExecutor.execute()` | test_error_handling |
| CORE-007 | `utils/logging.py` | test_logging |
| CORE-008 | `cli/commands/` | test_cli |

---

*本文档为核心框架模块的详细设计，基于4+1视图方法。*
