# 测试套模块设计 (Test Suite Module)

## 模块概述

| 属性 | 说明 |
|------|------|
| **模块ID** | SUITE |
| **模块名称** | 测试套管理 |
| **英文名称** | Test Suite Management |
| **职责** | 测试用例的组织、分组、过滤、选择和执行编排 |
| **关联需求** | CORE-002, CORE-003, CORE-004 |

### 模块定位

测试套模块是连接测试用例发现与测试执行的桥梁，负责将发现的测试用例组织成层级结构，支持灵活的过滤、选择和执行策略。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Test Suite Module Position                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐      ┌─────────────────┐      ┌─────────────┐            │
│   │  Discovery  │─────►│   Test Suite    │─────►│  Executor   │            │
│   │  测试发现    │      │    测试套管理    │      │   执行器    │            │
│   └─────────────┘      └─────────────────┘      └─────────────┘            │
│         │                      │                       │                    │
│         │              ┌───────┴───────┐               │                    │
│         │              ▼               ▼               │                    │
│         │      ┌─────────────┐ ┌─────────────┐        │                    │
│         │      │   Filter    │ │  Organizer  │        │                    │
│         │      │   过滤器    │ │   组织器    │        │                    │
│         │      └─────────────┘ └─────────────┘        │                    │
│         │                                              │                    │
│         └──────────────────────────────────────────────┘                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# 一、逻辑视图 (Logical View)

## 1. 测试套层级结构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Test Suite Hierarchy                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                          ┌─────────────────┐                                │
│                          │   TestSession   │                                │
│                          │    测试会话      │                                │
│                          └────────┬────────┘                                │
│                                   │                                         │
│                    ┌──────────────┼──────────────┐                          │
│                    ▼              ▼              ▼                          │
│             ┌───────────┐  ┌───────────┐  ┌───────────┐                     │
│             │ TestSuite │  │ TestSuite │  │ TestSuite │                     │
│             │  测试套A   │  │  测试套B   │  │  测试套C   │                     │
│             └─────┬─────┘  └─────┬─────┘  └───────────┘                     │
│                   │              │                                          │
│          ┌────────┼────────┐    ┌┴────────┐                                 │
│          ▼        ▼        ▼    ▼         ▼                                 │
│      ┌───────┐┌───────┐┌───────┐┌───────┐┌───────┐                          │
│      │Module1││Module2││Module3││Module4││Module5│                          │
│      │测试模块││测试模块││测试模块││测试模块││测试模块│                          │
│      └───┬───┘└───┬───┘└───────┘└───┬───┘└───────┘                          │
│          │        │                 │                                       │
│     ┌────┴────┐  ┌┴───┐        ┌────┴────┐                                  │
│     ▼         ▼  ▼    ▼        ▼         ▼                                  │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                               │
│  │Class1│ │Class2│ │Class3│ │Class4│ │Class5│                               │
│  │测试类 │ │测试类 │ │测试类 │ │测试类 │ │测试类 │                               │
│  └──┬───┘ └──┬───┘ └──────┘ └──┬───┘ └──────┘                               │
│     │        │                 │                                            │
│  ┌──┴──┐  ┌──┴──┐          ┌───┴───┐                                        │
│  ▼     ▼  ▼     ▼          ▼       ▼                                        │
│ ┌────┐┌────┐┌────┐┌────┐ ┌────┐ ┌────┐                                      │
│ │Case││Case││Case││Case│ │Case│ │Case│                                      │
│ │用例1││用例2││用例3││用例4│ │用例5│ │用例6│                                      │
│ └────┘└────┘└────┘└────┘ └────┘ └────┘                                      │
│                                                                             │
│  层级说明:                                                                   │
│  - Session: 一次测试运行的顶层容器                                            │
│  - Suite:   按功能/模块分组的测试集合                                         │
│  - Module:  对应一个测试文件 (test_*.py)                                      │
│  - Class:   测试类 (可选，支持函数式测试)                                      │
│  - Case:    单个测试用例                                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. 核心类设计

### 2.1 测试套类图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Test Suite Class Diagram                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      <<interface>> ITestItem                         │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │ + id: str                                                           │    │
│  │ + name: str                                                         │    │
│  │ + parent: Optional[ITestItem]                                       │    │
│  │ + children: List[ITestItem]                                         │    │
│  │ + tags: Set[str]                                                    │    │
│  │ + metadata: Dict[str, Any]                                          │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │ + get_path() -> str                                                 │    │
│  │ + iter_cases() -> Iterator[TestCase]                                │    │
│  │ + count_cases() -> int                                              │    │
│  │ + filter(criteria) -> ITestItem                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    △                                        │
│                                    │                                        │
│         ┌──────────────────────────┼──────────────────────────┐             │
│         │                          │                          │             │
│  ┌──────┴──────┐           ┌───────┴───────┐          ┌───────┴───────┐     │
│  │ TestSession │           │   TestSuite   │          │  TestModule   │     │
│  ├─────────────┤           ├───────────────┤          ├───────────────┤     │
│  │ - suites    │           │ - modules     │          │ - classes     │     │
│  │ - config    │           │ - description │          │ - functions   │     │
│  │ - start_time│           │ - priority    │          │ - file_path   │     │
│  ├─────────────┤           ├───────────────┤          ├───────────────┤     │
│  │ + add_suite │           │ + add_module  │          │ + add_class   │     │
│  │ + get_stats │           │ + get_modules │          │ + add_function│     │
│  │ + to_dict   │           │ + set_priority│          │ + get_fixtures│     │
│  └─────────────┘           └───────────────┘          └───────────────┘     │
│                                                                             │
│  ┌─────────────────┐                      ┌─────────────────┐               │
│  │   TestClass     │                      │   TestCase      │               │
│  ├─────────────────┤                      ├─────────────────┤               │
│  │ - methods       │                      │ - function      │               │
│  │ - class_obj     │                      │ - parameters    │               │
│  │ - setup_method  │                      │ - expected      │               │
│  │ - teardown_meth │                      │ - timeout       │               │
│  ├─────────────────┤                      │ - retry_count   │               │
│  │ + get_methods   │                      │ - dependencies  │               │
│  │ + instantiate   │                      ├─────────────────┤               │
│  │ + get_fixtures  │                      │ + run()         │               │
│  └─────────────────┘                      │ + skip()        │               │
│                                           │ + mark_xfail()  │               │
│                                           │ + get_markers() │               │
│                                           └─────────────────┘               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 测试套管理器类图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Test Suite Manager Classes                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        SuiteManager                                  │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  - session: TestSession                                             │    │
│  │  - organizer: SuiteOrganizer                                        │    │
│  │  - filter_chain: FilterChain                                        │    │
│  │  - selector: TestSelector                                           │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  + build_session(discovered: List[TestCase]) -> TestSession         │    │
│  │  + organize_by(strategy: str) -> TestSession                        │    │
│  │  + filter(criteria: FilterCriteria) -> TestSession                  │    │
│  │  + select(selector: str) -> List[TestCase]                          │    │
│  │  + get_execution_plan() -> ExecutionPlan                            │    │
│  │  + reorder(strategy: ReorderStrategy) -> TestSession                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌──────────────────────────┐      ┌──────────────────────────┐            │
│  │     SuiteOrganizer       │      │      FilterChain         │            │
│  ├──────────────────────────┤      ├──────────────────────────┤            │
│  │ - strategies: Dict       │      │ - filters: List[IFilter] │            │
│  ├──────────────────────────┤      ├──────────────────────────┤            │
│  │ + by_directory()         │      │ + add_filter(f)          │            │
│  │ + by_tag()               │      │ + apply(items) -> items  │            │
│  │ + by_marker()            │      │ + reset()                │            │
│  │ + by_priority()          │      └──────────────────────────┘            │
│  │ + by_custom(fn)          │                                              │
│  └──────────────────────────┘      ┌──────────────────────────┐            │
│                                    │      TestSelector        │            │
│  ┌──────────────────────────┐      ├──────────────────────────┤            │
│  │    <<interface>>         │      │ - patterns: List[str]    │            │
│  │      IFilter             │      ├──────────────────────────┤            │
│  ├──────────────────────────┤      │ + by_name(pattern)       │            │
│  │ + match(item) -> bool    │      │ + by_path(glob)          │            │
│  │ + get_name() -> str      │      │ + by_tag(tags)           │            │
│  └──────────────────────────┘      │ + by_marker(markers)     │            │
│              △                     │ + by_id(ids)             │            │
│              │                     │ + exclude(pattern)       │            │
│    ┌─────────┼─────────┐           └──────────────────────────┘            │
│    │         │         │                                                   │
│  ┌─┴──────┐┌─┴──────┐┌─┴──────┐                                            │
│  │TagFilter││NameFilter││PathFilter│                                        │
│  └────────┘└────────┘└────────┘                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 3. 核心数据结构

### 3.1 测试套配置

```python
@dataclass
class SuiteConfig:
    """测试套配置"""

    # 组织策略
    organize_by: str = "directory"  # directory, tag, marker, priority

    # 执行顺序
    execution_order: str = "definition"  # definition, alphabetical, random, dependency
    random_seed: Optional[int] = None

    # 并行配置
    parallel_mode: str = "none"  # none, thread, process, distributed
    max_workers: int = 4

    # 失败处理
    fail_fast: bool = False
    max_failures: Optional[int] = None

    # 重试配置
    retry_failed: bool = False
    retry_count: int = 2

    # 超时配置
    suite_timeout: Optional[float] = None
    default_case_timeout: float = 300.0


@dataclass
class FilterCriteria:
    """过滤条件"""

    # 包含条件
    include_tags: Set[str] = field(default_factory=set)
    include_markers: Set[str] = field(default_factory=set)
    include_paths: List[str] = field(default_factory=list)
    include_names: List[str] = field(default_factory=list)

    # 排除条件
    exclude_tags: Set[str] = field(default_factory=set)
    exclude_markers: Set[str] = field(default_factory=set)
    exclude_paths: List[str] = field(default_factory=list)
    exclude_names: List[str] = field(default_factory=list)

    # 优先级过滤
    min_priority: Optional[int] = None
    max_priority: Optional[int] = None

    # 自定义过滤
    custom_filter: Optional[Callable[[TestCase], bool]] = None


@dataclass
class ExecutionPlan:
    """执行计划"""

    # 执行批次
    batches: List[ExecutionBatch] = field(default_factory=list)

    # 依赖图
    dependency_graph: Dict[str, Set[str]] = field(default_factory=dict)

    # 估算信息
    estimated_duration: Optional[float] = None
    total_cases: int = 0

    # 资源需求
    requires_gpu: bool = False
    requires_network: bool = False


@dataclass
class ExecutionBatch:
    """执行批次"""

    batch_id: str
    cases: List[TestCase]
    can_parallel: bool = True
    dependencies: Set[str] = field(default_factory=set)
    resource_group: Optional[str] = None
```

## 4. 核心接口定义

### 4.1 测试套接口

```python
class ITestSuite(Protocol):
    """测试套接口"""

    @property
    def id(self) -> str:
        """唯一标识"""
        ...

    @property
    def name(self) -> str:
        """套件名称"""
        ...

    @property
    def description(self) -> str:
        """套件描述"""
        ...

    def add_module(self, module: ITestModule) -> None:
        """添加测试模块"""
        ...

    def get_modules(self) -> List[ITestModule]:
        """获取所有模块"""
        ...

    def iter_cases(self) -> Iterator[TestCase]:
        """迭代所有测试用例"""
        ...

    def filter(self, criteria: FilterCriteria) -> 'ITestSuite':
        """过滤并返回新的测试套"""
        ...

    def count(self) -> SuiteCount:
        """统计用例数量"""
        ...


class ISuiteOrganizer(Protocol):
    """测试套组织器接口"""

    def organize(
        self,
        cases: List[TestCase],
        strategy: str
    ) -> TestSession:
        """按策略组织测试用例"""
        ...

    def register_strategy(
        self,
        name: str,
        strategy: Callable
    ) -> None:
        """注册组织策略"""
        ...


class ITestFilter(Protocol):
    """测试过滤器接口"""

    def match(self, item: ITestItem) -> bool:
        """判断是否匹配"""
        ...

    @property
    def name(self) -> str:
        """过滤器名称"""
        ...

    @property
    def description(self) -> str:
        """过滤器描述"""
        ...
```

---

# 二、进程视图 (Process View)

## 1. 测试套构建流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Suite Building Process Flow                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐   │
│  │Discovery│───►│Collected│───►│ Filter  │───►│Organize │───►│Execution│   │
│  │  发现   │    │  Cases  │    │  过滤   │    │  组织   │    │  Plan   │   │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘   │
│                                                                             │
│  详细流程:                                                                   │
│                                                                             │
│  1. Discovery Phase (发现阶段)                                               │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  scan paths ──► parse files ──► extract tests ──► raw cases    │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                    │                                        │
│                                    ▼                                        │
│  2. Filter Phase (过滤阶段)                                                  │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │         ┌─────────┐                                             │     │
│     │  raw ──►│TagFilter│──►┌──────────┐──►┌───────────┐──► filtered  │     │
│     │  cases  └─────────┘   │NameFilter│   │ PathFilter│     cases    │     │
│     │                       └──────────┘   └───────────┘              │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                    │                                        │
│                                    ▼                                        │
│  3. Organization Phase (组织阶段)                                            │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  filtered   ┌──────────────────────────────┐                    │     │
│     │   cases ───►│     Organization Strategy     │───► TestSession   │     │
│     │             │  (by_dir/by_tag/by_priority) │                    │     │
│     │             └──────────────────────────────┘                    │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                    │                                        │
│                                    ▼                                        │
│  4. Planning Phase (规划阶段)                                                │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  TestSession ──► analyze deps ──► create batches ──► exec plan  │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. 测试套执行流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Suite Execution Flow                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        Session Start                                 │    │
│  │  ┌────────────────────────────────────────────────────────────────┐ │    │
│  │  │  load config ──► init resources ──► session_setup hooks        │ │    │
│  │  └────────────────────────────────────────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     For Each Suite                                   │    │
│  │  ┌────────────────────────────────────────────────────────────────┐ │    │
│  │  │  suite_setup ──► execute modules ──► suite_teardown            │ │    │
│  │  └────────────────────────────────────────────────────────────────┘ │    │
│  │                                 │                                    │    │
│  │                                 ▼                                    │    │
│  │  ┌────────────────────────────────────────────────────────────────┐ │    │
│  │  │                  For Each Module                                │ │    │
│  │  │  ┌──────────────────────────────────────────────────────────┐  │ │    │
│  │  │  │ module_setup ──► execute classes/funcs ──► module_teardown│  │ │    │
│  │  │  └──────────────────────────────────────────────────────────┘  │ │    │
│  │  │                              │                                  │ │    │
│  │  │                              ▼                                  │ │    │
│  │  │  ┌──────────────────────────────────────────────────────────┐  │ │    │
│  │  │  │                   For Each Case                           │  │ │    │
│  │  │  │  ┌────────────────────────────────────────────────────┐  │  │ │    │
│  │  │  │  │ setup ──► run test ──► collect result ──► teardown │  │  │ │    │
│  │  │  │  └────────────────────────────────────────────────────┘  │  │ │    │
│  │  │  └──────────────────────────────────────────────────────────┘  │ │    │
│  │  └────────────────────────────────────────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        Session End                                   │    │
│  │  ┌────────────────────────────────────────────────────────────────┐ │    │
│  │  │  session_teardown ──► aggregate results ──► generate report    │ │    │
│  │  └────────────────────────────────────────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 3. 并行执行模型

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Parallel Execution Models                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. Suite-Level Parallel (套件级并行)                                        │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │                      TestSession                                 │     │
│     │        ┌──────────────────┬──────────────────┐                  │     │
│     │        ▼                  ▼                  ▼                  │     │
│     │   ┌─────────┐        ┌─────────┐        ┌─────────┐            │     │
│     │   │Worker 1 │        │Worker 2 │        │Worker 3 │            │     │
│     │   │ Suite A │        │ Suite B │        │ Suite C │            │     │
│     │   └─────────┘        └─────────┘        └─────────┘            │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  2. Module-Level Parallel (模块级并行)                                       │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │                        Suite                                     │     │
│     │        ┌──────────────────┬──────────────────┐                  │     │
│     │        ▼                  ▼                  ▼                  │     │
│     │   ┌─────────┐        ┌─────────┐        ┌─────────┐            │     │
│     │   │Worker 1 │        │Worker 2 │        │Worker 3 │            │     │
│     │   │Module 1 │        │Module 2 │        │Module 3 │            │     │
│     │   └─────────┘        └─────────┘        └─────────┘            │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  3. Case-Level Parallel (用例级并行)                                         │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │                        Module                                    │     │
│     │   ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐    │     │
│     │   ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼      │    │     │
│     │ ┌───┐  ┌───┐  ┌───┐  ┌───┐  ┌───┐  ┌───┐  ┌───┐  ┌───┐   │    │     │
│     │ │W1 │  │W2 │  │W3 │  │W4 │  │W1 │  │W2 │  │W3 │  │W4 │   │    │     │
│     │ │C1 │  │C2 │  │C3 │  │C4 │  │C5 │  │C6 │  │C7 │  │C8 │   │    │     │
│     │ └───┘  └───┘  └───┘  └───┘  └───┘  └───┘  └───┘  └───┘   │    │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  4. Dependency-Aware Parallel (依赖感知并行)                                  │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │                                                                  │     │
│     │   Batch 1 (并行)        Batch 2 (并行)        Batch 3 (并行)     │     │
│     │   ┌───┬───┬───┐        ┌───┬───┐            ┌───┐              │     │
│     │   │C1 │C2 │C3 │ ─────► │C4 │C5 │ ─────────► │C6 │              │     │
│     │   └───┴───┴───┘        └───┴───┘            └───┘              │     │
│     │   (无依赖)             (依赖Batch1)         (依赖Batch2)        │     │
│     │                                                                  │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 4. 状态管理

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Suite State Machine                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                           ┌──────────────┐                                  │
│                           │   CREATED    │                                  │
│                           │    已创建     │                                  │
│                           └──────┬───────┘                                  │
│                                  │ configure                                │
│                                  ▼                                          │
│                           ┌──────────────┐                                  │
│                           │  CONFIGURED  │                                  │
│                           │    已配置     │                                  │
│                           └──────┬───────┘                                  │
│                                  │ build                                    │
│                                  ▼                                          │
│                           ┌──────────────┐                                  │
│                           │    READY     │                                  │
│                           │    就绪      │                                  │
│                           └──────┬───────┘                                  │
│                                  │ start                                    │
│                                  ▼                                          │
│                           ┌──────────────┐                                  │
│              ┌────────────│   RUNNING    │────────────┐                     │
│              │            │    运行中     │            │                     │
│              │            └──────┬───────┘            │                     │
│          pause                   │                 cancel                   │
│              │            ┌──────┴──────┐             │                     │
│              ▼            ▼             ▼             ▼                     │
│       ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│       │  PAUSED  │  │ COMPLETED│  │  FAILED  │  │ CANCELLED│               │
│       │   暂停   │  │   完成   │  │   失败   │  │   取消   │               │
│       └────┬─────┘  └──────────┘  └──────────┘  └──────────┘               │
│            │ resume                                                         │
│            └──────────────────────────┐                                     │
│                                       ▼                                     │
│                                ┌──────────────┐                             │
│                                │   RUNNING    │                             │
│                                └──────────────┘                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# 三、开发视图 (Development View)

## 1. 包结构

```
src/aitest/suite/
├── __init__.py                # 公共API导出
├── session.py                 # TestSession 实现
├── suite.py                   # TestSuite 实现
├── module.py                  # TestModule 实现
├── case.py                    # TestCase 实现
├── manager.py                 # SuiteManager 实现
│
├── organization/              # 组织策略
│   ├── __init__.py
│   ├── base.py               # 基础组织器
│   ├── directory.py          # 按目录组织
│   ├── tag.py                # 按标签组织
│   ├── marker.py             # 按标记组织
│   └── priority.py           # 按优先级组织
│
├── filter/                    # 过滤器
│   ├── __init__.py
│   ├── base.py               # 基础过滤器
│   ├── tag_filter.py         # 标签过滤器
│   ├── name_filter.py        # 名称过滤器
│   ├── path_filter.py        # 路径过滤器
│   ├── marker_filter.py      # 标记过滤器
│   └── chain.py              # 过滤器链
│
├── selector/                  # 选择器
│   ├── __init__.py
│   ├── pattern.py            # 模式选择器
│   ├── expression.py         # 表达式选择器
│   └── interactive.py        # 交互式选择器
│
├── execution/                 # 执行规划
│   ├── __init__.py
│   ├── plan.py               # 执行计划
│   ├── batch.py              # 执行批次
│   ├── dependency.py         # 依赖分析
│   └── order.py              # 执行顺序
│
├── parallel/                  # 并行执行
│   ├── __init__.py
│   ├── strategy.py           # 并行策略
│   ├── worker_pool.py        # 工作池
│   └── resource.py           # 资源管理
│
└── markers/                   # 标记系统
    ├── __init__.py
    ├── builtin.py            # 内置标记
    ├── custom.py             # 自定义标记
    └── registry.py           # 标记注册
```

## 2. 代码示例

### 2.1 测试套定义

```python
# tests/suites/model_tests.py

from aitest import suite, tag, priority, depends_on

@suite(
    name="Model Inference Tests",
    description="测试模型推理正确性",
    tags=["model", "inference"],
    priority=1
)
class ModelInferenceSuite:
    """模型推理测试套件"""

    @classmethod
    def setup_suite(cls):
        """套件级别初始化"""
        cls.model = load_test_model()
        cls.test_data = load_test_dataset()

    @classmethod
    def teardown_suite(cls):
        """套件级别清理"""
        cls.model.cleanup()


@suite(name="Performance Tests", tags=["performance"])
@depends_on(ModelInferenceSuite)  # 依赖关系
class PerformanceSuite:
    """性能测试套件"""
    pass
```

### 2.2 测试过滤与选择

```python
# 使用过滤器
from aitest.suite import SuiteManager, FilterCriteria

manager = SuiteManager()

# 创建过滤条件
criteria = FilterCriteria(
    include_tags={"model", "inference"},
    exclude_tags={"slow"},
    include_paths=["tests/model/*"],
    min_priority=1
)

# 应用过滤
session = manager.build_session(discovered_cases)
filtered_session = manager.filter(criteria)

# 选择特定测试
selected = manager.select("tests/model/test_inference.py::test_accuracy")
```

### 2.3 CLI用法

```bash
# 按标签过滤
aitest run --tag model --tag inference --exclude-tag slow

# 按路径过滤
aitest run tests/model/ --ignore tests/model/slow/

# 按名称模式
aitest run -k "test_accuracy or test_performance"

# 按标记过滤
aitest run -m "not slow and not integration"

# 组合过滤
aitest run tests/ \
    --tag model \
    --exclude-tag slow \
    -k "accuracy" \
    --priority 1

# 并行执行
aitest run tests/ --parallel=auto --workers=4

# 查看测试套结构
aitest list --tree
aitest list --by-tag
aitest list --by-suite
```

### 2.4 执行计划生成

```python
from aitest.suite import SuiteManager, SuiteConfig

config = SuiteConfig(
    organize_by="tag",
    parallel_mode="process",
    max_workers=4,
    fail_fast=False
)

manager = SuiteManager(config)
session = manager.build_session(cases)

# 生成执行计划
plan = manager.get_execution_plan()

print(f"总用例数: {plan.total_cases}")
print(f"执行批次: {len(plan.batches)}")
print(f"预计时长: {plan.estimated_duration}s")

for batch in plan.batches:
    print(f"Batch {batch.batch_id}: {len(batch.cases)} cases")
    print(f"  可并行: {batch.can_parallel}")
    print(f"  依赖: {batch.dependencies}")
```

---

# 四、物理视图 (Physical View)

## 1. 部署架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Suite Execution Deployment                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. 单机模式                                                                 │
│     ┌───────────────────────────────────────────────────────────────────┐   │
│     │                        Local Machine                               │   │
│     │   ┌─────────────┐                                                 │   │
│     │   │Suite Manager│                                                 │   │
│     │   └──────┬──────┘                                                 │   │
│     │          │                                                        │   │
│     │   ┌──────┴──────────────────────────────────┐                     │   │
│     │   ▼              ▼              ▼           ▼                     │   │
│     │ ┌──────┐      ┌──────┐      ┌──────┐    ┌──────┐                 │   │
│     │ │Worker│      │Worker│      │Worker│    │Worker│                 │   │
│     │ │ (P1) │      │ (P2) │      │ (P3) │    │ (P4) │                 │   │
│     │ └──────┘      └──────┘      └──────┘    └──────┘                 │   │
│     └───────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  2. 分布式模式                                                               │
│     ┌───────────────────────────────────────────────────────────────────┐   │
│     │                      Coordinator Node                              │   │
│     │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │   │
│     │   │Suite Manager│  │Task Scheduler│  │Result Agg.  │              │   │
│     │   └──────┬──────┘  └──────┬──────┘  └─────────────┘              │   │
│     └──────────┼────────────────┼───────────────────────────────────────┘   │
│                │                │                                           │
│       ┌────────┴────────┬───────┴───────┬────────────────┐                  │
│       ▼                 ▼               ▼                ▼                  │
│   ┌────────┐       ┌────────┐      ┌────────┐       ┌────────┐             │
│   │Worker 1│       │Worker 2│      │Worker 3│       │Worker 4│             │
│   │ (GPU)  │       │ (GPU)  │      │ (CPU)  │       │ (CPU)  │             │
│   │Suite A │       │Suite B │      │Suite C │       │Suite D │             │
│   └────────┘       └────────┘      └────────┘       └────────┘             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. 资源管理

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Resource Management                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  资源类型:                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  - GPU: 模型推理测试专用                                              │    │
│  │  - CPU: 通用测试执行                                                  │    │
│  │  - Memory: 大数据集测试                                               │    │
│  │  - Network: 分布式和集成测试                                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  资源分配策略:                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                                                                      │    │
│  │   Suite     Resource Req.    Worker Assignment                       │    │
│  │   ─────────────────────────────────────────────────                  │    │
│  │   Model     GPU + 16GB RAM   GPU Worker Pool                         │    │
│  │   Data      32GB RAM         High-Memory Worker                      │    │
│  │   Unit      2GB RAM          Any Worker                              │    │
│  │   Integ     Network          Network-enabled Worker                  │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# 五、场景视图 (Scenarios View)

## 1. 核心用例

### UC-SUITE-01: 按标签过滤执行测试

```
┌─────┐          ┌─────────┐          ┌─────────┐          ┌─────────┐
│User │          │   CLI   │          │ Manager │          │ Filter  │
└──┬──┘          └────┬────┘          └────┬────┘          └────┬────┘
   │                  │                    │                    │
   │ aitest run       │                    │                    │
   │ --tag model      │                    │                    │
   │─────────────────>│                    │                    │
   │                  │                    │                    │
   │                  │  build_session()   │                    │
   │                  │───────────────────>│                    │
   │                  │                    │                    │
   │                  │                    │  apply_filter()    │
   │                  │                    │───────────────────>│
   │                  │                    │                    │
   │                  │                    │   filtered_cases   │
   │                  │                    │<───────────────────│
   │                  │                    │                    │
   │                  │   session          │                    │
   │                  │<───────────────────│                    │
   │                  │                    │                    │
   │   results        │                    │                    │
   │<─────────────────│                    │                    │
```

### UC-SUITE-02: 并行执行测试套

```python
# 配置并行执行
config = SuiteConfig(
    parallel_mode="process",
    max_workers=4,
    organize_by="suite"
)

# 构建会话
manager = SuiteManager(config)
session = manager.build_session(discovered_cases)

# 生成执行计划
plan = manager.get_execution_plan()

# 执行（自动并行）
results = manager.execute(plan)
```

### UC-SUITE-03: 依赖顺序执行

```python
# 定义依赖关系
@suite(name="Setup Suite")
class SetupSuite:
    pass

@suite(name="Model Suite")
@depends_on(SetupSuite)
class ModelSuite:
    pass

@suite(name="Cleanup Suite")
@depends_on(ModelSuite)
class CleanupSuite:
    pass

# 执行时自动处理依赖
# 执行顺序: SetupSuite -> ModelSuite -> CleanupSuite
```

## 2. 场景验证矩阵

| 场景 | 覆盖需求 | 验证方法 |
|------|----------|----------|
| 按标签过滤 | CORE-002-04 | 过滤结果验证 |
| 按路径过滤 | CORE-002-02 | 路径匹配测试 |
| 套件级setup/teardown | CORE-004-01 | 生命周期追踪 |
| 并行执行 | CORE-003-02 | 并发数验证 |
| 依赖顺序 | CORE-003-01 | 执行顺序验证 |
| 执行超时 | CORE-003-04 | 超时中断测试 |

---

## 需求追溯

| 需求ID | 需求名称 | 模块功能 |
|--------|----------|----------|
| CORE-002 | 测试用例发现 | 用例收集、组织到测试套 |
| CORE-002-04 | 测试用例过滤 | FilterChain, TestSelector |
| CORE-003 | 测试执行调度 | ExecutionPlan, 批次管理 |
| CORE-003-01 | 顺序执行 | 依赖分析、顺序排列 |
| CORE-003-02 | 并行执行 | 并行策略、Worker管理 |
| CORE-004-01 | 全局setup/teardown | Session/Suite级别钩子 |
| CORE-004-02 | 模块setup/teardown | Module级别钩子 |

---

*本文档为AI测试框架测试套模块设计，详细描述了测试套的组织、过滤、选择和执行机制。*
