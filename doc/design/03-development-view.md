# AI测试框架 - 开发视图 (Development View)

## 概述

开发视图描述软件的静态组织结构，关注代码的模块划分、包结构、层次关系和开发规范。本视图为开发人员提供代码组织的蓝图。

---

## 1. 项目结构

### 1.1 顶层目录结构

```
ai-test-framework/
├── src/                          # 源代码目录
│   └── aitest/                   # 主包
│       ├── __init__.py
│       ├── core/                 # 核心引擎模块
│       ├── model/                # 模型测试模块
│       ├── data/                 # 数据管理模块
│       ├── assertion/            # 断言验证模块
│       ├── report/               # 报告生成模块
│       ├── integration/          # 集成部署模块
│       ├── plugin/               # 插件系统模块
│       ├── cli/                  # 命令行接口
│       ├── api/                  # API接口
│       └── utils/                # 工具函数
│
├── tests/                        # 测试代码目录
│   ├── unit/                     # 单元测试
│   ├── integration/              # 集成测试
│   └── e2e/                      # 端到端测试
│
├── docs/                         # 文档目录
│   ├── api/                      # API文档
│   ├── guides/                   # 使用指南
│   └── design/                   # 设计文档
│
├── examples/                     # 示例代码
│   ├── basic/                    # 基础示例
│   ├── model_testing/            # 模型测试示例
│   └── plugins/                  # 插件示例
│
├── configs/                      # 配置文件模板
│   ├── default.yaml              # 默认配置
│   └── ci/                       # CI配置模板
│
├── scripts/                      # 脚本工具
│   ├── build.sh
│   └── release.sh
│
├── pyproject.toml                # 项目配置
├── setup.py                      # 安装脚本
├── requirements.txt              # 依赖列表
├── requirements-dev.txt          # 开发依赖
├── Makefile                      # 构建命令
├── Dockerfile                    # Docker镜像
└── README.md                     # 项目说明
```

### 1.2 核心模块详细结构

```
src/aitest/
│
├── __init__.py                   # 包初始化，导出公共API
├── __version__.py                # 版本信息
│
├── core/                         # 核心引擎 [CORE-001~008]
│   ├── __init__.py
│   ├── engine.py                 # TestEngine 主引擎
│   ├── discovery.py              # 测试发现
│   ├── scheduler.py              # 调度器
│   ├── executor.py               # 执行器
│   ├── lifecycle.py              # 生命周期管理
│   ├── fixture.py                # Fixture机制
│   ├── context.py                # 测试上下文
│   ├── result.py                 # 结果模型
│   ├── exceptions.py             # 异常定义
│   └── config/                   # 配置管理
│       ├── __init__.py
│       ├── loader.py             # 配置加载器
│       ├── schema.py             # 配置Schema
│       └── defaults.py           # 默认配置
│
├── model/                        # 模型测试 [MODEL-001~009]
│   ├── __init__.py
│   ├── loader/                   # 模型加载器
│   │   ├── __init__.py
│   │   ├── base.py               # 基类接口
│   │   ├── pytorch.py            # PyTorch加载器
│   │   ├── tensorflow.py         # TensorFlow加载器
│   │   ├── onnx.py               # ONNX加载器
│   │   └── huggingface.py        # HuggingFace加载器
│   ├── inference.py              # 推理封装
│   ├── accuracy.py               # 精度评估
│   ├── performance.py            # 性能测试
│   ├── robustness.py             # 鲁棒性测试
│   ├── consistency.py            # 一致性测试
│   ├── llm/                      # LLM专项
│   │   ├── __init__.py
│   │   ├── prompt_test.py        # 提示词测试
│   │   ├── generation.py         # 生成评估
│   │   └── safety.py             # 安全性测试
│   └── metrics/                  # 评估指标
│       ├── __init__.py
│       ├── classification.py     # 分类指标
│       ├── regression.py         # 回归指标
│       ├── detection.py          # 检测指标
│       └── nlp.py                # NLP指标
│
├── data/                         # 数据管理 [DATA-001~009]
│   ├── __init__.py
│   ├── loader/                   # 数据加载器
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── file.py               # 文件加载 (CSV/JSON/Parquet)
│   │   ├── image.py              # 图像加载
│   │   ├── text.py               # 文本加载
│   │   └── remote.py             # 远程数据加载
│   ├── transform/                # 数据转换
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── normalize.py          # 归一化
│   │   ├── augment.py            # 数据增强
│   │   └── tokenize.py           # 文本分词
│   ├── pipeline.py               # 预处理Pipeline
│   ├── sampler.py                # 采样器
│   ├── generator.py              # 合成数据生成
│   ├── registry.py               # 数据集注册
│   ├── golden.py                 # 黄金数据集
│   └── validator.py              # 数据验证
│
├── assertion/                    # 断言验证 [ASSERT-001~010]
│   ├── __init__.py
│   ├── engine.py                 # 断言引擎
│   ├── basic.py                  # 基础断言
│   ├── numeric.py                # 数值断言
│   ├── tensor.py                 # 张量断言
│   ├── classification.py         # 分类断言
│   ├── detection.py              # 检测断言
│   ├── text.py                   # 文本断言
│   ├── performance.py            # 性能断言
│   ├── metric.py                 # 指标断言
│   ├── snapshot.py               # 快照对比
│   └── custom.py                 # 自定义断言
│
├── report/                       # 报告生成 [REPORT-001~009]
│   ├── __init__.py
│   ├── engine.py                 # 报告引擎
│   ├── reporter/                 # 报告器
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── console.py            # 控制台报告
│   │   ├── html.py               # HTML报告
│   │   ├── json.py               # JSON报告
│   │   ├── junit.py              # JUnit XML
│   │   └── markdown.py           # Markdown报告
│   ├── visualizer/               # 可视化
│   │   ├── __init__.py
│   │   ├── charts.py             # 图表生成
│   │   └── templates/            # HTML模板
│   ├── aggregator.py             # 结果聚合
│   ├── comparator.py             # 版本对比
│   ├── distributor.py            # 报告分发
│   └── realtime.py               # 实时报告
│
├── integration/                  # 集成部署 [INTEG-001~010]
│   ├── __init__.py
│   ├── ci/                       # CI集成
│   │   ├── __init__.py
│   │   ├── github.py             # GitHub Actions
│   │   ├── gitlab.py             # GitLab CI
│   │   └── jenkins.py            # Jenkins
│   ├── container/                # 容器化
│   │   ├── __init__.py
│   │   ├── docker.py             # Docker支持
│   │   └── kubernetes.py         # K8s支持
│   ├── serving/                  # 模型服务
│   │   ├── __init__.py
│   │   ├── torchserve.py
│   │   ├── tfserving.py
│   │   └── triton.py
│   ├── cloud/                    # 云平台
│   │   ├── __init__.py
│   │   ├── aws.py
│   │   ├── azure.py
│   │   └── gcp.py
│   ├── monitoring/               # 监控
│   │   ├── __init__.py
│   │   ├── prometheus.py
│   │   └── grafana.py
│   └── environment.py            # 环境管理
│
├── plugin/                       # 插件系统 [EXT-001~009]
│   ├── __init__.py
│   ├── manager.py                # 插件管理器
│   ├── discovery.py              # 插件发现
│   ├── registry.py               # 插件注册
│   ├── hooks.py                  # 钩子系统
│   ├── base.py                   # 插件基类
│   └── builtin/                  # 内置插件
│       ├── __init__.py
│       └── ...
│
├── cli/                          # 命令行接口
│   ├── __init__.py
│   ├── main.py                   # CLI入口
│   ├── commands/                 # 命令实现
│   │   ├── __init__.py
│   │   ├── run.py                # run命令
│   │   ├── list.py               # list命令
│   │   ├── config.py             # config命令
│   │   └── plugin.py             # plugin命令
│   └── formatters.py             # 输出格式化
│
├── api/                          # API接口
│   ├── __init__.py
│   ├── public.py                 # 公共Python API
│   ├── rest/                     # REST API
│   │   ├── __init__.py
│   │   ├── server.py
│   │   ├── routes.py
│   │   └── schemas.py
│   └── client.py                 # API客户端
│
└── utils/                        # 工具函数
    ├── __init__.py
    ├── logging.py                # 日志工具
    ├── timing.py                 # 计时工具
    ├── io.py                     # IO工具
    ├── serialization.py          # 序列化
    └── validation.py             # 验证工具
```

---

## 2. 模块分层架构

### 2.1 层次结构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Layer Architecture                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        Interface Layer (接口层)                        │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐    │  │
│  │  │     CLI     │  │  Python API │  │        REST API             │    │  │
│  │  │  cli/*.py   │  │ api/public  │  │       api/rest/*            │    │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────────┘    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      Application Layer (应用层)                        │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                      core/engine.py                              │  │  │
│  │  │                      (Test Orchestration)                        │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │  │
│  │  │   Discovery  │  │   Scheduler  │  │   Executor   │                 │  │
│  │  │core/discovery│  │core/scheduler│  │ core/executor│                 │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                 │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        Domain Layer (领域层)                           │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │  │
│  │  │   Model     │  │    Data     │  │  Assertion  │  │   Report    │   │  │
│  │  │   Testing   │  │  Management │  │ & Validation│  │  Generation │   │  │
│  │  │  model/*    │  │   data/*    │  │ assertion/* │  │  report/*   │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    Infrastructure Layer (基础设施层)                    │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │  │
│  │  │  Config  │  │  Logger  │  │  Plugin  │  │Integration│ │  Utils   │ │  │
│  │  │core/config│ │utils/log │  │ plugin/* │  │integration│ │  utils/* │ │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  Dependencies Direction: ↓ (Upper layers depend on lower layers)            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 层间依赖规则

| 层级 | 可依赖 | 不可依赖 |
|------|--------|----------|
| Interface | Application, Domain, Infrastructure | - |
| Application | Domain, Infrastructure | Interface |
| Domain | Infrastructure | Interface, Application |
| Infrastructure | - | Interface, Application, Domain |

---

## 3. 模块依赖图

### 3.1 包级依赖关系

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Package Dependencies                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                           ┌─────────┐                                       │
│                           │   cli   │                                       │
│                           └────┬────┘                                       │
│                                │                                            │
│                                ▼                                            │
│                           ┌─────────┐                                       │
│                      ┌────│  core   │────┐                                  │
│                      │    └────┬────┘    │                                  │
│                      │         │         │                                  │
│           ┌──────────┼─────────┼─────────┼──────────┐                       │
│           │          │         │         │          │                       │
│           ▼          ▼         ▼         ▼          ▼                       │
│      ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │
│      │  model  │ │  data   │ │assertion│ │ report  │ │ plugin  │           │
│      └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘           │
│           │           │           │           │           │                 │
│           └───────────┴─────┬─────┴───────────┴───────────┘                 │
│                             │                                               │
│                             ▼                                               │
│                        ┌─────────┐                                          │
│                        │  utils  │                                          │
│                        └─────────┘                                          │
│                                                                             │
│   Dependency Types:                                                         │
│   ─────► Direct import dependency                                           │
│   - - -► Optional/Plugin dependency                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 详细依赖矩阵

```
          cli  core  model  data  assert  report  plugin  integ  utils
cli        -    ✓      ○      ○      ○       ✓       ○       ○      ✓
core       ✗    -      ✓      ✓      ✓       ✓       ✓       ○      ✓
model      ✗    ○      -      ✓      ✓       ○       ○       ○      ✓
data       ✗    ○      ○      -      ○       ○       ○       ○      ✓
assert     ✗    ○      ○      ○      -       ○       ○       ○      ✓
report     ✗    ○      ○      ○      ○       -       ○       ○      ✓
plugin     ✗    ✓      ○      ○      ○       ○       -       ○      ✓
integ      ✗    ○      ○      ○      ○       ✓       ○       -      ✓
utils      ✗    ✗      ✗      ✗      ✗       ✗       ✗       ✗      -

Legend: ✓ Required dependency
        ○ Optional dependency
        ✗ No dependency (forbidden)
        - Self
```

---

## 4. 代码组织规范

### 4.1 模块结构模板

```python
# aitest/example_module/__init__.py

"""
Example Module - Brief description

This module provides...

Example:
    >>> from aitest.example import SomeClass
    >>> obj = SomeClass()
"""

from .main_class import MainClass
from .secondary import SecondaryClass
from .exceptions import ModuleError

__all__ = [
    "MainClass",
    "SecondaryClass",
    "ModuleError",
]
```

### 4.2 类文件结构模板

```python
# aitest/example_module/main_class.py

"""
Main class implementation for example module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol

if TYPE_CHECKING:
    from aitest.core import TestContext

from aitest.utils.validation import validate_config

from .exceptions import ModuleError

logger = logging.getLogger(__name__)


# ============================================
# Protocols / Interfaces
# ============================================

class IExample(Protocol):
    """Interface for example implementations."""

    def process(self, data: Any) -> Any:
        """Process the input data."""
        ...


# ============================================
# Data Classes
# ============================================

@dataclass
class ExampleConfig:
    """Configuration for Example."""

    option_a: str
    option_b: int = 10
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExampleResult:
    """Result from Example processing."""

    success: bool
    output: Any
    errors: List[str] = field(default_factory=list)


# ============================================
# Main Implementation
# ============================================

class MainClass:
    """
    Main implementation class.

    This class provides the primary functionality for...

    Attributes:
        config: The configuration for this instance.
        _initialized: Whether the instance has been initialized.

    Example:
        >>> main = MainClass(config)
        >>> result = main.process(data)
    """

    def __init__(self, config: ExampleConfig) -> None:
        """
        Initialize MainClass.

        Args:
            config: Configuration object.

        Raises:
            ModuleError: If configuration is invalid.
        """
        self.config = config
        self._initialized = False
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate the configuration."""
        if not self.config.option_a:
            raise ModuleError("option_a is required")

    def initialize(self) -> None:
        """Initialize resources."""
        if self._initialized:
            return
        logger.info("Initializing MainClass")
        # initialization logic
        self._initialized = True

    def process(self, data: Any) -> ExampleResult:
        """
        Process input data.

        Args:
            data: Input data to process.

        Returns:
            ExampleResult containing the processing outcome.

        Raises:
            ModuleError: If processing fails.
        """
        if not self._initialized:
            self.initialize()

        try:
            # processing logic
            output = self._do_process(data)
            return ExampleResult(success=True, output=output)
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return ExampleResult(success=False, output=None, errors=[str(e)])

    def _do_process(self, data: Any) -> Any:
        """Internal processing implementation."""
        # actual processing
        return data

    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up MainClass")
        self._initialized = False
```

### 4.3 命名规范

| 类型 | 规范 | 示例 |
|------|------|------|
| 包名 | 小写，下划线分隔 | `model_testing`, `data_loader` |
| 模块名 | 小写，下划线分隔 | `pytorch_loader.py`, `tensor_assert.py` |
| 类名 | PascalCase | `TestEngine`, `ModelLoader` |
| 函数名 | snake_case | `load_model`, `run_tests` |
| 常量 | 大写，下划线分隔 | `DEFAULT_TIMEOUT`, `MAX_WORKERS` |
| 私有成员 | 单下划线前缀 | `_internal_method`, `_cache` |
| 接口/协议 | I前缀 + PascalCase | `IModelLoader`, `IReporter` |

---

## 5. 外部依赖管理

### 5.1 依赖分类

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        External Dependencies                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Core Dependencies (必需):                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  pyyaml          - 配置文件解析                                      │    │
│  │  pydantic        - 数据验证                                          │    │
│  │  click           - CLI框架                                           │    │
│  │  rich            - 终端美化                                          │    │
│  │  structlog       - 结构化日志                                        │    │
│  │  numpy           - 数值计算                                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ML Framework Dependencies (可选):                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  torch           - PyTorch模型支持        [extras: pytorch]         │    │
│  │  tensorflow      - TensorFlow模型支持     [extras: tensorflow]      │    │
│  │  onnxruntime     - ONNX推理               [extras: onnx]            │    │
│  │  transformers    - HuggingFace模型        [extras: huggingface]     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Report Dependencies (可选):                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  jinja2          - 模板渲染               [extras: report]          │    │
│  │  matplotlib      - 图表生成               [extras: report]          │    │
│  │  plotly          - 交互式图表             [extras: report-interactive]│   │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Development Dependencies:                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  pytest          - 测试框架                                          │    │
│  │  pytest-cov      - 覆盖率                                            │    │
│  │  mypy            - 类型检查                                          │    │
│  │  ruff            - 代码检查                                          │    │
│  │  black           - 代码格式化                                        │    │
│  │  pre-commit      - Git钩子                                           │    │
│  │  sphinx          - 文档生成                                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 pyproject.toml 配置

```toml
[project]
name = "aitest-framework"
version = "1.0.0"
description = "AI/ML Model Testing Framework"
requires-python = ">=3.9"

dependencies = [
    "pyyaml>=6.0",
    "pydantic>=2.0",
    "click>=8.0",
    "rich>=13.0",
    "structlog>=23.0",
    "numpy>=1.21",
]

[project.optional-dependencies]
pytorch = ["torch>=2.0"]
tensorflow = ["tensorflow>=2.12"]
onnx = ["onnxruntime>=1.15"]
huggingface = ["transformers>=4.30"]
report = ["jinja2>=3.0", "matplotlib>=3.7"]
all = ["aitest-framework[pytorch,tensorflow,onnx,huggingface,report]"]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "mypy>=1.0",
    "ruff>=0.1",
    "black>=23.0",
    "pre-commit>=3.0",
]

[project.scripts]
aitest = "aitest.cli.main:cli"

[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"
```

---

## 6. 构建与测试

### 6.1 构建流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Build Pipeline                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌────────┐ │
│  │   Lint   │───►│   Type   │───►│   Test   │───►│  Build   │───►│Publish │ │
│  │  Check   │    │  Check   │    │          │    │ Package  │    │        │ │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘    └────────┘ │
│                                                                             │
│  Commands:                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  make lint      # ruff check + black --check                        │    │
│  │  make typecheck # mypy src/                                         │    │
│  │  make test      # pytest tests/                                     │    │
│  │  make coverage  # pytest --cov=aitest tests/                        │    │
│  │  make build     # python -m build                                   │    │
│  │  make publish   # twine upload dist/*                               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 测试结构

```
tests/
├── conftest.py                   # 全局fixtures
├── unit/                         # 单元测试
│   ├── conftest.py
│   ├── core/
│   │   ├── test_engine.py
│   │   ├── test_discovery.py
│   │   └── test_scheduler.py
│   ├── model/
│   │   ├── test_pytorch_loader.py
│   │   └── test_accuracy.py
│   ├── data/
│   │   ├── test_file_loader.py
│   │   └── test_pipeline.py
│   └── assertion/
│       ├── test_basic.py
│       └── test_tensor.py
│
├── integration/                  # 集成测试
│   ├── conftest.py
│   ├── test_full_pipeline.py
│   ├── test_model_testing.py
│   └── test_report_generation.py
│
├── e2e/                          # 端到端测试
│   ├── conftest.py
│   ├── test_cli.py
│   └── test_api.py
│
└── fixtures/                     # 测试数据
    ├── models/
    ├── datasets/
    └── configs/
```

### 6.3 Makefile

```makefile
.PHONY: help install dev lint typecheck test coverage build clean

PYTHON := python3
PIP := pip3

help:
	@echo "Available commands:"
	@echo "  install    - Install package"
	@echo "  dev        - Install with dev dependencies"
	@echo "  lint       - Run linters"
	@echo "  typecheck  - Run type checker"
	@echo "  test       - Run tests"
	@echo "  coverage   - Run tests with coverage"
	@echo "  build      - Build package"
	@echo "  clean      - Clean build artifacts"

install:
	$(PIP) install -e .

dev:
	$(PIP) install -e ".[dev,all]"
	pre-commit install

lint:
	ruff check src/ tests/
	black --check src/ tests/

format:
	ruff check --fix src/ tests/
	black src/ tests/

typecheck:
	mypy src/

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

coverage:
	pytest tests/ --cov=aitest --cov-report=html --cov-report=term

build:
	$(PYTHON) -m build

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
```

---

## 7. 配置管理

### 7.1 配置层次结构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Configuration Hierarchy                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Priority (Low → High):                                                      │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  1. Built-in Defaults                                               │    │
│  │     aitest/core/config/defaults.py                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  2. System Config                                                   │    │
│  │     /etc/aitest/config.yaml                                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  3. User Config                                                     │    │
│  │     ~/.aitest/config.yaml                                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  4. Project Config                                                  │    │
│  │     ./aitest.yaml or ./pyproject.toml [tool.aitest]                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  5. Environment Variables                                           │    │
│  │     AITEST_* prefixed variables                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  6. CLI Arguments                                                   │    │
│  │     --config, --option flags                                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 配置Schema

```yaml
# aitest.yaml - 项目配置示例

# 框架基础配置
framework:
  version: "1.0"
  log_level: INFO
  parallel_workers: 4
  timeout: 300

# 测试发现配置
discovery:
  paths:
    - tests/
    - src/**/test_*.py
  patterns:
    - "test_*.py"
    - "*_test.py"
  exclude:
    - "**/fixtures/**"
  tags:
    include: []
    exclude: ["slow"]

# 模型测试配置
model:
  default_device: cuda
  batch_size: 32
  warmup_iterations: 10
  loaders:
    pytorch:
      enabled: true
    tensorflow:
      enabled: false

# 数据配置
data:
  cache_dir: .aitest/cache
  max_cache_size: 10GB

# 报告配置
report:
  formats:
    - console
    - html
    - json
  output_dir: reports/
  html:
    template: default

# 插件配置
plugins:
  enabled:
    - aitest-plugin-example
  config:
    aitest-plugin-example:
      option: value
```

---

## 8. 版本控制与发布

### 8.1 分支策略

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Git Branch Strategy                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  main ─────●─────●─────────────●─────────────●─────────────────────────►    │
│            │     │             │             │                              │
│            │     │   ┌─────────┘             │                              │
│            │     │   │                       │                              │
│  develop ──●─────●───●─────●─────●───────────●─────●───────────────────►    │
│                  │         │     │           │     │                        │
│                  │    ┌────┘     │     ┌─────┘     │                        │
│                  │    │          │     │           │                        │
│  feature/xxx ────●────●          │     │           │                        │
│                       │          │     │           │                        │
│  feature/yyy ─────────●──────────●     │           │                        │
│                                        │           │                        │
│  release/1.0 ──────────────────────────●───────────●                        │
│                                        │                                    │
│  hotfix/bug ───────────────────────────●────────────────────────────────    │
│                                                                             │
│  Branch Types:                                                              │
│  - main:      Production-ready code                                         │
│  - develop:   Integration branch for features                               │
│  - feature/*: New feature development                                       │
│  - release/*: Release preparation                                           │
│  - hotfix/*:  Production bug fixes                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 版本号规范

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Semantic Versioning (SemVer)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Version Format: MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]                     │
│                                                                             │
│  Examples:                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  1.0.0        - First stable release                                │    │
│  │  1.0.1        - Patch release (bug fixes)                           │    │
│  │  1.1.0        - Minor release (new features, backward compatible)   │    │
│  │  2.0.0        - Major release (breaking changes)                    │    │
│  │  2.0.0-alpha  - Pre-release alpha version                           │    │
│  │  2.0.0-beta.1 - Pre-release beta version 1                          │    │
│  │  2.0.0-rc.1   - Release candidate 1                                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Version Bump Rules:                                                         │
│  - MAJOR: Breaking API changes                                              │
│  - MINOR: New features, backward compatible                                 │
│  - PATCH: Bug fixes, backward compatible                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. 需求到代码的追溯

| 需求ID | 模块路径 | 主要类/函数 |
|--------|----------|-------------|
| CORE-001 | `core/engine.py`, `core/config/` | `TestEngine`, `ConfigLoader` |
| CORE-002 | `core/discovery.py` | `TestDiscoveryEngine`, `TestCollector` |
| CORE-003 | `core/scheduler.py`, `core/executor.py` | `TestScheduler`, `TestExecutor` |
| CORE-004 | `core/lifecycle.py` | `LifecycleManager` |
| CORE-005 | `core/fixture.py` | `FixtureManager`, `Fixture` |
| CORE-006 | `core/exceptions.py` | `TestError`, `FrameworkError` |
| CORE-007 | `utils/logging.py` | `setup_logging`, `TestLogger` |
| CORE-008 | `cli/` | `main.py`, `commands/` |
| MODEL-001 | `model/loader/` | `PyTorchLoader`, `TensorFlowLoader` |
| MODEL-003 | `model/inference.py` | `InferenceValidator` |
| MODEL-004 | `model/accuracy.py`, `model/metrics/` | `AccuracyEvaluator` |
| MODEL-005 | `model/performance.py` | `PerformanceTester` |
| DATA-001 | `data/loader/` | `FileLoader`, `ImageLoader` |
| DATA-005 | `data/pipeline.py` | `DataPipeline` |
| ASSERT-001 | `assertion/basic.py` | `BasicAssertions` |
| REPORT-001 | `report/reporter/` | `HTMLReporter`, `JSONReporter` |
| EXT-001 | `plugin/manager.py` | `PluginManager` |

---

*本文档为AI测试框架开发视图设计，详细描述了代码组织结构和开发规范。*
