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
