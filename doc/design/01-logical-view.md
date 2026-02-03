# AI测试框架 - 逻辑视图 (Logical View)

## 概述

逻辑视图描述系统的功能分解，关注系统的类、对象、接口以及它们之间的关系。本视图采用面向对象的方法，展示系统的静态结构。

---

## 1. 系统分层架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            表现层 (Presentation Layer)                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │  CLI Interface  │  │   Python API    │  │      REST API Server        │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────────┤
│                            应用层 (Application Layer)                         │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        Test Orchestrator (测试编排器)                   │  │
│  │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐  │  │
│  │   │Discovery │  │Scheduler │  │ Executor │  │  Lifecycle Manager   │  │  │
│  │   └──────────┘  └──────────┘  └──────────┘  └──────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────────┤
│                            领域层 (Domain Layer)                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │  Model Testing  │  │ Data Management │  │   Assertion & Validation    │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────────┤
│                            基础设施层 (Infrastructure Layer)                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │  Logger  │  │  Config  │  │  Plugin  │  │  Report  │  │  Integration │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 核心类设计

### 2.1 核心引擎类图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Core Engine Package                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────┐         ┌───────────────────────┐               │
│  │    <<interface>>      │         │    <<interface>>      │               │
│  │    ITestDiscovery     │         │    ITestScheduler     │               │
│  ├───────────────────────┤         ├───────────────────────┤               │
│  │ + discover(path)      │         │ + schedule(tests)     │               │
│  │ + filter(criteria)    │         │ + get_execution_order │               │
│  │ + collect()           │         │ + set_strategy()      │               │
│  └───────────┬───────────┘         └───────────┬───────────┘               │
│              │                                 │                            │
│              ▼                                 ▼                            │
│  ┌───────────────────────┐         ┌───────────────────────┐               │
│  │  TestDiscoveryEngine  │         │   TestScheduler       │               │
│  ├───────────────────────┤         ├───────────────────────┤               │
│  │ - collectors: List    │         │ - strategy: Strategy  │               │
│  │ - filters: List       │         │ - workers: int        │               │
│  ├───────────────────────┤         ├───────────────────────┤               │
│  │ + discover()          │         │ + schedule()          │               │
│  │ + register_collector()│         │ + add_dependency()    │               │
│  │ + register_filter()   │         │ + parallelize()       │               │
│  └───────────────────────┘         └───────────────────────┘               │
│                                                                             │
│  ┌───────────────────────┐         ┌───────────────────────┐               │
│  │    <<interface>>      │         │      TestEngine       │               │
│  │    ITestExecutor      │         ├───────────────────────┤               │
│  ├───────────────────────┤         │ - discovery: IDisc    │               │
│  │ + execute(test)       │         │ - scheduler: ISched   │               │
│  │ + setup()             │         │ - executor: IExec     │               │
│  │ + teardown()          │         │ - lifecycle: ILife    │               │
│  └───────────┬───────────┘         ├───────────────────────┤               │
│              │                     │ + initialize()        │               │
│              ▼                     │ + run(config)         │               │
│  ┌───────────────────────┐         │ + shutdown()          │               │
│  │    TestExecutor       │         └───────────────────────┘               │
│  ├───────────────────────┤                     ▲                           │
│  │ - context: Context    │                     │                           │
│  │ - hooks: HookManager  │                     │                           │
│  ├───────────────────────┤         ┌───────────────────────┐               │
│  │ + execute()           │         │  LifecycleManager     │               │
│  │ + handle_exception()  │         ├───────────────────────┤               │
│  │ + collect_result()    │         │ - hooks: Dict         │               │
│  └───────────────────────┘         │ - fixtures: Dict      │               │
│                                    ├───────────────────────┤               │
│                                    │ + register_hook()     │               │
│                                    │ + trigger(event)      │               │
│                                    │ + get_fixture()       │               │
│                                    └───────────────────────┘               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 测试用例类图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Test Case Package                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────┐                                  │
│  │           <<abstract>>                │                                  │
│  │           BaseTestCase                │                                  │
│  ├───────────────────────────────────────┤                                  │
│  │ # id: str                             │                                  │
│  │ # name: str                           │                                  │
│  │ # tags: List[str]                     │                                  │
│  │ # timeout: Optional[int]              │                                  │
│  │ # dependencies: List[str]             │                                  │
│  ├───────────────────────────────────────┤                                  │
│  │ + setup()                             │                                  │
│  │ + teardown()                          │                                  │
│  │ + run() -> TestResult                 │                                  │
│  │ + skip(reason: str)                   │                                  │
│  └───────────────┬───────────────────────┘                                  │
│                  │                                                          │
│    ┌─────────────┼─────────────┬─────────────────────┐                      │
│    │             │             │                     │                      │
│    ▼             ▼             ▼                     ▼                      │
│  ┌───────────┐ ┌───────────┐ ┌───────────────┐ ┌─────────────────┐          │
│  │FunctionTC │ │  ClassTC  │ │ParameterizedTC│ │   ModelTestCase │          │
│  ├───────────┤ ├───────────┤ ├───────────────┤ ├─────────────────┤          │
│  │ - func    │ │ - cls     │ │ - params      │ │ - model         │          │
│  │ - module  │ │ - methods │ │ - generator   │ │ - dataset       │          │
│  └───────────┘ └───────────┘ └───────────────┘ └─────────────────┘          │
│                                                                             │
│  ┌───────────────────────────────────────┐                                  │
│  │            TestResult                 │                                  │
│  ├───────────────────────────────────────┤                                  │
│  │ + test_id: str                        │                                  │
│  │ + status: TestStatus                  │                                  │
│  │ + duration: float                     │                                  │
│  │ + error: Optional[Exception]          │                                  │
│  │ + output: Dict                        │                                  │
│  │ + artifacts: List[Artifact]           │                                  │
│  └───────────────────────────────────────┘                                  │
│                                                                             │
│  ┌───────────────────────────────────────┐                                  │
│  │        <<enumeration>>                │                                  │
│  │          TestStatus                   │                                  │
│  ├───────────────────────────────────────┤                                  │
│  │   PASSED                              │                                  │
│  │   FAILED                              │                                  │
│  │   ERROR                               │                                  │
│  │   SKIPPED                             │                                  │
│  │   XFAIL                               │                                  │
│  │   XPASS                               │                                  │
│  └───────────────────────────────────────┘                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 模型测试类图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Model Testing Package                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────┐       ┌─────────────────────────────┐      │
│  │     <<interface>>           │       │     <<interface>>           │      │
│  │     IModelLoader            │       │     IModelInference         │      │
│  ├─────────────────────────────┤       ├─────────────────────────────┤      │
│  │ + load(path) -> Model       │       │ + predict(input) -> Output  │      │
│  │ + supports(format) -> bool  │       │ + batch_predict(inputs)     │      │
│  │ + get_metadata() -> Dict    │       │ + warmup(samples)           │      │
│  └──────────────┬──────────────┘       └─────────────────────────────┘      │
│                 │                                                           │
│    ┌────────────┼────────────┬────────────────┐                             │
│    ▼            ▼            ▼                ▼                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐                    │
│  │PyTorch   │ │TensorFlow│ │  ONNX    │ │ HuggingFace  │                    │
│  │Loader    │ │Loader    │ │ Loader   │ │   Loader     │                    │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘                    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         ModelTestSuite                               │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  - model: LoadedModel                                               │    │
│  │  - dataset: TestDataset                                             │    │
│  │  - metrics: List[Metric]                                            │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  + test_inference_correctness()                                     │    │
│  │  + test_accuracy(threshold: float)                                  │    │
│  │  + test_performance(latency_p99: float, throughput: float)          │    │
│  │  + test_robustness(perturbations: List)                             │    │
│  │  + test_consistency()                                               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌──────────────────────────┐    ┌──────────────────────────┐               │
│  │    PerformanceTester     │    │    AccuracyEvaluator     │               │
│  ├──────────────────────────┤    ├──────────────────────────┤               │
│  │ - warmup_iterations      │    │ - metrics: List[Metric]  │               │
│  │ - test_iterations        │    │ - threshold: float       │               │
│  ├──────────────────────────┤    ├──────────────────────────┤               │
│  │ + measure_latency()      │    │ + evaluate(preds, labels)│               │
│  │ + measure_throughput()   │    │ + compute_metrics()      │               │
│  │ + measure_memory()       │    │ + check_threshold()      │               │
│  │ + profile_gpu()          │    │ + generate_report()      │               │
│  └──────────────────────────┘    └──────────────────────────┘               │
│                                                                             │
│  ┌──────────────────────────┐    ┌──────────────────────────┐               │
│  │     RobustnessTester     │    │      LLMTester           │               │
│  ├──────────────────────────┤    ├──────────────────────────┤               │
│  │ - noise_types: List      │    │ - prompts: List          │               │
│  │ - perturbation_level     │    │ - evaluators: List       │               │
│  ├──────────────────────────┤    ├──────────────────────────┤               │
│  │ + test_noise_injection() │    │ + test_prompt()          │               │
│  │ + test_boundary_values() │    │ + test_context_length()  │               │
│  │ + test_adversarial()     │    │ + test_safety()          │               │
│  │ + generate_perturbations │    │ + evaluate_generation()  │               │
│  └──────────────────────────┘    └──────────────────────────┘               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.4 数据管理类图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Data Management Package                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────┐       ┌─────────────────────────────┐      │
│  │     <<interface>>           │       │     <<interface>>           │      │
│  │     IDataLoader             │       │     IDataTransform          │      │
│  ├─────────────────────────────┤       ├─────────────────────────────┤      │
│  │ + load(source) -> Dataset   │       │ + transform(data) -> data   │      │
│  │ + supports(format) -> bool  │       │ + fit(data)                 │      │
│  │ + stream(source) -> Iterator│       │ + inverse(data) -> data     │      │
│  └──────────────┬──────────────┘       └──────────────┬──────────────┘      │
│                 │                                     │                     │
│    ┌────────────┼────────────┐           ┌───────────┼───────────┐          │
│    ▼            ▼            ▼           ▼           ▼           ▼          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐ ┌─────────┐ ┌────────┐  │
│  │CSV/JSON  │ │  Image   │ │  Remote  │ │Normalize│ │Augment  │ │Tokenize│  │
│  │Loader    │ │ Loader   │ │  Loader  │ │Transform│ │Transform│ │Transform│ │
│  └──────────┘ └──────────┘ └──────────┘ └─────────┘ └─────────┘ └────────┘  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                          DataPipeline                                │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  - steps: List[IDataTransform]                                      │    │
│  │  - cache_enabled: bool                                              │    │
│  │  - parallel_workers: int                                            │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  + add_step(transform: IDataTransform)                              │    │
│  │  + process(data) -> ProcessedData                                   │    │
│  │  + process_batch(batch) -> List[ProcessedData]                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌──────────────────────────┐    ┌──────────────────────────┐               │
│  │       DatasetRegistry    │    │     GoldenDataset        │               │
│  ├──────────────────────────┤    ├──────────────────────────┤               │
│  │ - datasets: Dict         │    │ - inputs: List           │               │
│  │ - versions: Dict         │    │ - expected_outputs: List │               │
│  ├──────────────────────────┤    │ - version: str           │               │
│  │ + register(name, ds)     │    ├──────────────────────────┤               │
│  │ + get(name, version)     │    │ + load(path)             │               │
│  │ + list_datasets()        │    │ + validate_format()      │               │
│  │ + get_versions(name)     │    │ + compare(predictions)   │               │
│  └──────────────────────────┘    └──────────────────────────┘               │
│                                                                             │
│  ┌──────────────────────────┐    ┌──────────────────────────┐               │
│  │      DataSampler         │    │    SyntheticGenerator    │               │
│  ├──────────────────────────┤    ├──────────────────────────┤               │
│  │ - strategy: SampleStrat  │    │ - generators: Dict       │               │
│  │ - seed: int              │    │ - templates: List        │               │
│  ├──────────────────────────┤    ├──────────────────────────┤               │
│  │ + sample(n) -> Subset    │    │ + generate(spec) -> Data │               │
│  │ + stratified_sample()    │    │ + generate_boundary()    │               │
│  │ + weighted_sample()      │    │ + generate_random()      │               │
│  └──────────────────────────┘    └──────────────────────────┘               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.5 断言与验证类图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Assertion & Validation Package                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      AssertionEngine                                 │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  - assertions: Dict[str, IAssertion]                                │    │
│  │  - soft_mode: bool                                                  │    │
│  │  - failures: List[AssertionError]                                   │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  + register(name, assertion)                                        │    │
│  │  + assert_that(actual, matcher)                                     │    │
│  │  + collect_failures() -> List                                       │    │
│  │  + reset()                                                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌─────────────────────────────┐       ┌─────────────────────────────┐      │
│  │     <<interface>>           │       │  BasicAssertions            │      │
│  │     IAssertion              │       ├─────────────────────────────┤      │
│  ├─────────────────────────────┤       │ + assertEqual(a, b)         │      │
│  │ + check(actual, expected)   │       │ + assertTrue(condition)     │      │
│  │ + get_message() -> str      │       │ + assertIn(item, container) │      │
│  └─────────────────────────────┘       │ + assertRaises(exc, func)   │      │
│                                        └─────────────────────────────┘      │
│                                                                             │
│  ┌──────────────────────────┐    ┌──────────────────────────┐               │
│  │   NumericAssertions      │    │    TensorAssertions      │               │
│  ├──────────────────────────┤    ├──────────────────────────┤               │
│  │ + assertAlmostEqual()    │    │ + assertShape(t, shape)  │               │
│  │ + assertInRange()        │    │ + assertDtype(t, dtype)  │               │
│  │ + assertRelativeError()  │    │ + assertAllClose(a, b)   │               │
│  │ + assertGreater()        │    │ + assertNoNaN(t)         │               │
│  │ + assertLess()           │    │ + assertNoInf(t)         │               │
│  └──────────────────────────┘    └──────────────────────────┘               │
│                                                                             │
│  ┌──────────────────────────┐    ┌──────────────────────────┐               │
│  │ ClassificationAssertions │    │  DetectionAssertions     │               │
│  ├──────────────────────────┤    ├──────────────────────────┤               │
│  │ + assertClassMatch()     │    │ + assertIoU(pred, gt)    │               │
│  │ + assertTopK()           │    │ + assertDetectionCount() │               │
│  │ + assertProbThreshold()  │    │ + assertClassDetected()  │               │
│  │ + assertConfusionMatrix()│    │ + assertConfidence()     │               │
│  └──────────────────────────┘    └──────────────────────────┘               │
│                                                                             │
│  ┌──────────────────────────┐    ┌──────────────────────────┐               │
│  │   TextAssertions         │    │   MetricAssertions       │               │
│  ├──────────────────────────┤    ├──────────────────────────┤               │
│  │ + assertContains()       │    │ + assertAccuracy()       │               │
│  │ + assertSimilarity()     │    │ + assertPrecision()      │               │
│  │ + assertLength()         │    │ + assertRecall()         │               │
│  │ + assertMatchesFormat()  │    │ + assertF1Score()        │               │
│  │ + assertValidJSON()      │    │ + assertLatency()        │               │
│  └──────────────────────────┘    └──────────────────────────┘               │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      SnapshotManager                                 │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  - snapshot_dir: Path                                               │    │
│  │  - serializer: ISerializer                                          │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  + save_snapshot(name, data)                                        │    │
│  │  + load_snapshot(name) -> data                                      │    │
│  │  + compare(current, snapshot) -> DiffReport                         │    │
│  │  + update_snapshot(name, data)                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.6 报告生成类图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Report Generation Package                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        ReportEngine                                  │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  - reporters: List[IReporter]                                       │    │
│  │  - aggregators: List[IAggregator]                                   │    │
│  │  - visualizers: List[IVisualizer]                                   │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  + generate(results: TestResults) -> Report                         │    │
│  │  + register_reporter(reporter)                                      │    │
│  │  + export(format: str, path: Path)                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌─────────────────────────────┐       ┌─────────────────────────────┐      │
│  │     <<interface>>           │       │     <<interface>>           │      │
│  │     IReporter               │       │     IVisualizer             │      │
│  ├─────────────────────────────┤       ├─────────────────────────────┤      │
│  │ + format() -> str           │       │ + create_chart(data) -> Fig │      │
│  │ + generate(data) -> Report  │       │ + supports(chart_type)      │      │
│  │ + export(path)              │       │ + export(format)            │      │
│  └──────────────┬──────────────┘       └──────────────┬──────────────┘      │
│                 │                                     │                     │
│    ┌────────────┼────────────┬───────────┐   ┌───────┼───────┐              │
│    ▼            ▼            ▼           ▼   ▼       ▼       ▼              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ ┌──────┐ ┌──────────┐    │
│  │  HTML    │ │   JSON   │ │  JUnit   │ │Markdown│ │ Pie  │ │ Timeline │    │
│  │ Reporter │ │ Reporter │ │ Reporter │ │Reporter│ │Chart │ │  Chart   │    │
│  └──────────┘ └──────────┘ └──────────┘ └────────┘ └──────┘ └──────────┘    │
│                                                                             │
│  ┌──────────────────────────┐    ┌──────────────────────────┐               │
│  │     TestSummary          │    │  PerformanceReport       │               │
│  ├──────────────────────────┤    ├──────────────────────────┤               │
│  │ + total_tests: int       │    │ + latency_stats: Stats   │               │
│  │ + passed: int            │    │ + throughput: float      │               │
│  │ + failed: int            │    │ + memory_usage: Stats    │               │
│  │ + skipped: int           │    │ + gpu_utilization: Stats │               │
│  │ + duration: float        │    ├──────────────────────────┤               │
│  │ + pass_rate: float       │    │ + generate_charts()      │               │
│  └──────────────────────────┘    │ + compare_baseline()     │               │
│                                  └──────────────────────────┘               │
│                                                                             │
│  ┌──────────────────────────┐    ┌──────────────────────────┐               │
│  │    AccuracyReport        │    │   ComparisonReport       │               │
│  ├──────────────────────────┤    ├──────────────────────────┤               │
│  │ + metrics: Dict          │    │ + baseline: Report       │               │
│  │ + confusion_matrix       │    │ + current: Report        │               │
│  │ + per_class_metrics      │    │ + diff: DiffResult       │               │
│  ├──────────────────────────┤    ├──────────────────────────┤               │
│  │ + generate_confusion()   │    │ + compute_diff()         │               │
│  │ + generate_roc_curve()   │    │ + highlight_regressions()│               │
│  │ + list_error_samples()   │    │ + highlight_improvements()│              │
│  └──────────────────────────┘    └──────────────────────────┘               │
│                                                                             │
│  ┌──────────────────────────┐    ┌──────────────────────────┐               │
│  │    ReportDistributor     │    │   RealtimeReporter       │               │
│  ├──────────────────────────┤    ├──────────────────────────┤               │
│  │ - channels: List         │    │ - console: Console       │               │
│  ├──────────────────────────┤    │ - progress_bar: ProgressBar│             │
│  │ + send_email(report)     │    ├──────────────────────────┤               │
│  │ + send_webhook(report)   │    │ + on_test_start(test)    │               │
│  │ + notify_slack(report)   │    │ + on_test_complete(result)│              │
│  │ + archive(report)        │    │ + update_progress()      │               │
│  └──────────────────────────┘    └──────────────────────────┘               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.7 插件与扩展类图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Plugin & Extension Package                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        PluginManager                                 │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  - plugins: Dict[str, IPlugin]                                      │    │
│  │  - hooks: HookRegistry                                              │    │
│  │  - enabled: Set[str]                                                │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  + discover_plugins() -> List[IPlugin]                              │    │
│  │  + register(plugin: IPlugin)                                        │    │
│  │  + enable(plugin_name)                                              │    │
│  │  + disable(plugin_name)                                             │    │
│  │  + get_plugin(name) -> IPlugin                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌─────────────────────────────┐       ┌─────────────────────────────┐      │
│  │     <<interface>>           │       │     <<interface>>           │      │
│  │     IPlugin                 │       │     IHook                   │      │
│  ├─────────────────────────────┤       ├─────────────────────────────┤      │
│  │ + name: str                 │       │ + event: str                │      │
│  │ + version: str              │       │ + priority: int             │      │
│  │ + initialize(config)        │       │ + execute(context)          │      │
│  │ + shutdown()                │       └─────────────────────────────┘      │
│  └──────────────┬──────────────┘                                            │
│                 │                                                           │
│    ┌────────────┼────────────┬────────────────┬───────────────┐             │
│    ▼            ▼            ▼                ▼               ▼             │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐ ┌──────────┐ ┌──────────┐       │
│  │ Model    │ │  Data    │ │  Reporter    │ │ Assertion│ │  Metric  │       │
│  │ Loader   │ │ Loader   │ │   Plugin     │ │  Plugin  │ │  Plugin  │       │
│  │ Plugin   │ │ Plugin   │ │              │ │          │ │          │       │
│  └──────────┘ └──────────┘ └──────────────┘ └──────────┘ └──────────┘       │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        HookRegistry                                  │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  - hooks: Dict[str, List[IHook]]                                    │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  + register(event, hook)                                            │    │
│  │  + unregister(event, hook)                                          │    │
│  │  + trigger(event, context) -> List[Result]                          │    │
│  │  + get_hooks(event) -> List[IHook]                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Hook Events (钩子事件)                            │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  BEFORE_COLLECTION    - 用例收集前                                   │    │
│  │  AFTER_COLLECTION     - 用例收集后                                   │    │
│  │  BEFORE_TEST          - 测试执行前                                   │    │
│  │  AFTER_TEST           - 测试执行后                                   │    │
│  │  ON_TEST_FAILURE      - 测试失败时                                   │    │
│  │  ON_TEST_SUCCESS      - 测试成功时                                   │    │
│  │  BEFORE_REPORT        - 报告生成前                                   │    │
│  │  AFTER_REPORT         - 报告生成后                                   │    │
│  │  ON_EXCEPTION         - 异常发生时                                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心接口定义

### 3.1 测试发现接口

```python
class ITestDiscovery(Protocol):
    """测试用例发现接口"""

    def discover(self, paths: List[Path]) -> List[TestCase]:
        """从指定路径发现测试用例"""
        ...

    def filter(self, tests: List[TestCase], criteria: FilterCriteria) -> List[TestCase]:
        """根据条件过滤测试用例"""
        ...

    def register_collector(self, collector: ITestCollector) -> None:
        """注册测试收集器"""
        ...
```

### 3.2 测试执行接口

```python
class ITestExecutor(Protocol):
    """测试执行器接口"""

    def execute(self, test: TestCase, context: TestContext) -> TestResult:
        """执行单个测试用例"""
        ...

    def setup(self, test: TestCase) -> None:
        """测试前置处理"""
        ...

    def teardown(self, test: TestCase, result: TestResult) -> None:
        """测试后置处理"""
        ...
```

### 3.3 模型加载器接口

```python
class IModelLoader(Protocol):
    """模型加载器接口"""

    def load(self, path: Union[str, Path], **kwargs) -> LoadedModel:
        """加载模型"""
        ...

    def supports(self, format: str) -> bool:
        """检查是否支持指定格式"""
        ...

    def get_metadata(self, model: LoadedModel) -> ModelMetadata:
        """获取模型元数据"""
        ...
```

### 3.4 数据加载器接口

```python
class IDataLoader(Protocol):
    """数据加载器接口"""

    def load(self, source: Union[str, Path, URL]) -> Dataset:
        """加载数据集"""
        ...

    def stream(self, source: Union[str, Path, URL]) -> Iterator[DataItem]:
        """流式加载数据"""
        ...

    def supports(self, format: str) -> bool:
        """检查是否支持指定格式"""
        ...
```

### 3.5 断言接口

```python
class IAssertion(Protocol):
    """断言接口"""

    def check(self, actual: Any, expected: Any, **kwargs) -> AssertionResult:
        """执行断言检查"""
        ...

    def get_message(self, result: AssertionResult) -> str:
        """获取断言失败消息"""
        ...
```

### 3.6 报告器接口

```python
class IReporter(Protocol):
    """报告生成器接口"""

    @property
    def format(self) -> str:
        """报告格式标识"""
        ...

    def generate(self, results: TestResults, config: ReportConfig) -> Report:
        """生成报告"""
        ...

    def export(self, report: Report, path: Path) -> None:
        """导出报告到文件"""
        ...
```

### 3.7 插件接口

```python
class IPlugin(Protocol):
    """插件接口"""

    @property
    def name(self) -> str:
        """插件名称"""
        ...

    @property
    def version(self) -> str:
        """插件版本"""
        ...

    def initialize(self, config: PluginConfig) -> None:
        """初始化插件"""
        ...

    def shutdown(self) -> None:
        """关闭插件"""
        ...
```

---

## 4. 模块间依赖关系

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Module Dependencies                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                        ┌───────────────┐                                    │
│                        │     CLI       │                                    │
│                        └───────┬───────┘                                    │
│                                │                                            │
│                                ▼                                            │
│                        ┌───────────────┐                                    │
│                        │  Core Engine  │                                    │
│                        └───────┬───────┘                                    │
│                                │                                            │
│          ┌─────────────────────┼─────────────────────┐                      │
│          │                     │                     │                      │
│          ▼                     ▼                     ▼                      │
│   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐                 │
│   │Model Testing│◄─────│    Data     │─────►│  Assertion  │                 │
│   └──────┬──────┘      │  Management │      └──────┬──────┘                 │
│          │             └──────┬──────┘             │                        │
│          │                    │                    │                        │
│          └────────────────────┼────────────────────┘                        │
│                               │                                             │
│                               ▼                                             │
│                        ┌─────────────┐                                      │
│                        │   Report    │                                      │
│                        │  Generation │                                      │
│                        └──────┬──────┘                                      │
│                               │                                             │
│                               ▼                                             │
│                        ┌─────────────┐                                      │
│                        │ Integration │                                      │
│                        └─────────────┘                                      │
│                                                                             │
│   横向依赖:                                                                  │
│   ┌─────────────┐                                                           │
│   │ Extensibility│ ←── 被所有模块依赖，提供插件和扩展支持                    │
│   └─────────────┘                                                           │
│                                                                             │
│   图例:  ──► 依赖方向    ◄── 被依赖                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. 关键数据模型

### 5.1 配置数据模型

```python
@dataclass
class FrameworkConfig:
    """框架配置"""
    version: str
    test_paths: List[Path]
    parallel_workers: int = 1
    timeout: int = 300
    log_level: str = "INFO"
    plugins: List[str] = field(default_factory=list)

@dataclass
class ModelTestConfig:
    """模型测试配置"""
    model_path: Path
    framework: str  # pytorch, tensorflow, onnx
    device: str = "cpu"
    batch_size: int = 1
    warmup_iterations: int = 10

@dataclass
class DataConfig:
    """数据配置"""
    dataset_path: Path
    format: str
    preprocessing: List[str] = field(default_factory=list)
    sample_size: Optional[int] = None
```

### 5.2 测试结果数据模型

```python
@dataclass
class TestResult:
    """测试结果"""
    test_id: str
    test_name: str
    status: TestStatus
    duration: float
    started_at: datetime
    finished_at: datetime
    error: Optional[ExceptionInfo] = None
    output: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[Artifact] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class TestSuiteResult:
    """测试套件结果"""
    suite_id: str
    results: List[TestResult]
    total_duration: float
    environment: EnvironmentInfo

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.PASSED)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.FAILED)
```

---

## 6. 需求到类的追溯

| 需求ID | 实现类 | 说明 |
|--------|--------|------|
| CORE-001 | `TestEngine`, `ConfigLoader` | 测试引擎初始化 |
| CORE-002 | `TestDiscoveryEngine`, `TestCollector` | 测试用例发现 |
| CORE-003 | `TestScheduler`, `ParallelExecutor` | 测试执行调度 |
| CORE-004 | `LifecycleManager`, `HookRegistry` | 测试生命周期钩子 |
| CORE-005 | `FixtureManager`, `Fixture` | Fixture机制 |
| MODEL-001 | `PyTorchLoader`, `TensorFlowLoader`, `ONNXLoader` | 模型加载器 |
| MODEL-003 | `InferenceValidator`, `OutputValidator` | 推理正确性测试 |
| MODEL-004 | `AccuracyEvaluator`, `MetricCalculator` | 模型精度测试 |
| MODEL-005 | `PerformanceTester`, `Profiler` | 推理性能测试 |
| DATA-001 | `CSVLoader`, `ImageLoader`, `JSONLoader` | 数据集加载 |
| DATA-005 | `DataPipeline`, `TransformChain` | 数据预处理Pipeline |
| ASSERT-001 | `BasicAssertions` | 基础断言方法 |
| ASSERT-003 | `TensorAssertions` | 张量断言 |
| REPORT-001 | `HTMLReporter`, `JSONReporter`, `JUnitReporter` | 多格式报告输出 |
| EXT-001 | `PluginManager`, `PluginLoader` | 插件架构 |

---

*本文档为AI测试框架逻辑视图设计，详细描述了系统的静态结构和类关系。*
