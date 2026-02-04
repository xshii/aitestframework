# AI Test Framework

一个专为业务模型/小模型测试设计的自动化测试框架，集成 aidevtools 算子验证工具。

## 项目简介

AI Test Framework 旨在为自研芯片的 AI 模型提供全面的测试解决方案，集成 aidevtools 工具集实现：
- 算子级 Golden 生成与精度比对
- 四状态判定（PASS / GOLDEN_SUSPECT / DUT_ISSUE / BOTH_SUSPECT）
- 多种量化格式支持（BFP、GFloat、float16 等）

## 快速开始

```bash
# 1. 安装
./install.sh dev

# 2. 激活环境
source .venv/bin/activate

# 3. 运行 aidevtools Demo
python demos/02_mini_transformer/run.py

# 4. 运行测试
pytest tests/ -v
```

## 功能特性

### aidevtools 集成（核心）
- **四状态比对**：PASS / GOLDEN_SUSPECT / DUT_ISSUE / BOTH_SUSPECT
- **双轨 Golden**：pure (fp32) / quant (量化感知) 模式
- **量化格式**：BFP16/8/4、GFloat16/8/4、float16
- **内置算子**：matmul、softmax、layernorm、gelu、attention 等

### 核心框架
- 测试用例自动发现与收集
- 灵活的测试执行调度（顺序/并行）
- 完整的生命周期钩子
- 强大的 Fixture 机制

### AI 模型测试
- PyTorch/ONNX 模型支持
- 推理正确性验证（四状态比对）
- 精度指标评估（QSNR、cosine、max_abs）
- 性能基准测试（延迟、吞吐量）
- 量化模型一致性测试

### 报告生成
- 多格式输出（HTML、JSON、JUnit XML）
- 四状态汇总统计
- QSNR/cosine 指标详情
- 失败算子详细报告

## 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                    aitestframework                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │ 测试用例    │ │ 执行调度    │ │ 报告生成    │           │
│  │ 发现/管理   │ │ 顺序/并行   │ │ HTML/JSON   │           │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘           │
│         └───────────────┼───────────────┘                   │
│                         │                                   │
│  ┌──────────────────────▼──────────────────────┐           │
│  │           aidevtools 集成层                  │           │
│  │  CompareEngine │ ops │ formats │ xlsx       │           │
│  └──────────────────────┬──────────────────────┘           │
└─────────────────────────┼───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                      aidevtools                             │
│  compare (四状态) │ ops (Golden) │ formats (BFP/GFloat)     │
└─────────────────────────────────────────────────────────────┘
```

## 基础使用

### 算子 Golden 生成

```python
from aidevtools import ops
from aidevtools.ops import _functional as F
import numpy as np

ops.clear()

x = np.random.randn(2, 8, 64).astype(np.float32)
w = np.random.randn(64, 128).astype(np.float32)

y = F.matmul(x, w)
y = F.layer_norm(y, (128,))
y = F.softmax(y, dim=-1)

for r in ops.get_records():
    print(f"{r.op_name}: input={r.input.shape}, golden={r.golden.shape}")
```

### 精度比对

```python
from aidevtools.compare import CompareEngine, CompareConfig

config = CompareConfig(
    fuzzy_min_qsnr=30.0,
    fuzzy_min_cosine=0.999,
)
engine = CompareEngine(config)

result = engine.compare(
    dut_output=dut,
    golden_pure=golden_fp32,
    golden_qnt=golden_qnt,
)
print(f"Status: {result.status.value}")  # PASS / DUT_ISSUE / ...
```

## 四状态判定模型

| DUT vs Golden | Golden 自检 | 判定状态 | 含义 |
|---------------|-------------|----------|------|
| PASS | PASS | **PASS** | DUT 正确，Golden 有效 |
| PASS | FAIL | **GOLDEN_SUSPECT** | DUT 匹配，但 Golden 可疑 |
| FAIL | PASS | **DUT_ISSUE** | Golden 有效，DUT 有问题 |
| FAIL | FAIL | **BOTH_SUSPECT** | 都可疑，需人工排查 |

## 目录结构

```
aitestframework/
├── aitestframework/          # 测试框架主包
│   ├── __init__.py
│   └── cli.py
├── aidevtools/               # 算子验证工具（集成）
│   ├── compare/              # 比对引擎
│   ├── ops/                  # 算子 Golden
│   ├── formats/              # 量化格式
│   ├── frontend/             # 数据生成
│   └── xlsx/                 # Excel 工作流
├── backends/                 # 多后端支持
│   ├── __init__.py           # 后端抽象层
│   ├── common/               # 公共模块
│   ├── cpu/                  # CPU 后端 (Golden 生成)
│   └── npu/                  # NPU 后端 (DUT 验证)
├── tests/                    # 测试用例
├── demos/                    # 示例
├── doc/
│   ├── req/                  # 需求文档
│   └── aidevtools/           # aidevtools 文档
├── pyproject.toml
└── README.md
```

## 多后端架构

支持 CPU 和 NPU 两种后端的验证：

```
┌─────────────────────────────────────────────────────────────┐
│                     测试框架                                 │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌───────────┐   ┌───────────┐   ┌───────────┐
    │  CPU 后端  │   │  NPU 后端  │   │  其他后端  │
    │  (Golden)  │   │   (DUT)   │   │  (预留)   │
    └───────────┘   └───────────┘   └───────────┘
          │               │
          └───────┬───────┘
                  ▼
    ┌─────────────────────────────┐
    │   aidevtools.compare        │
    │   四状态比对                 │
    └─────────────────────────────┘
```

### 验证流程

1. **CPU 后端**生成 Golden（通过 aidevtools.ops）
2. **NPU 后端**执行 DUT 获取输出
3. **CompareEngine** 进行四状态比对
4. 生成验证报告

## 需求文档

详细的需求分析文档位于 `doc/req/` 目录：

| 文件 | 说明 |
|------|------|
| [00-requirements-index.yaml](doc/req/00-requirements-index.yaml) | 需求总览索引 |
| [01-core-framework.yaml](doc/req/01-core-framework.yaml) | 核心框架需求 |
| [02-ai-model-testing.yaml](doc/req/02-ai-model-testing.yaml) | AI模型测试需求 |
| [03-data-management.yaml](doc/req/03-data-management.yaml) | 数据管理需求 |
| [04-assertion-validation.yaml](doc/req/04-assertion-validation.yaml) | 断言与验证需求 |
| [05-report-generation.yaml](doc/req/05-report-generation.yaml) | 报告生成需求 |
| [06-integration-deployment.yaml](doc/req/06-integration-deployment.yaml) | 集成与部署需求 |
| [07-extensibility.yaml](doc/req/07-extensibility.yaml) | 扩展性需求 |
| [08-aidevtools-integration.yaml](doc/req/08-aidevtools-integration.yaml) | aidevtools集成需求 |

### 需求统计

- **总模块数**: 8
- **总需求数**: 68 个主需求，236 个子需求
- **优先级分布**: P0(28) / P1(32) / P2(7) / P3(1)

## 开发路线图

### Phase 1: MVP 版本
实现 P0 优先级的核心功能（28 项需求）
- aidevtools 集成（四状态比对）
- 测试用例发现与执行
- 基础报告生成
- CLI 工具

### Phase 2: 功能完善
补充 P1 优先级的重要功能（32 项需求）

### Phase 3: 增强扩展
添加 P2/P3 优先级的增强功能（8 项需求）

## 开发

```bash
# 安装开发依赖
./install.sh dev

# 编译 C++ Golden
./build_golden.sh

# 运行测试
pytest tests/ -v

# 代码检查
ruff check aidevtools/ aitestframework/
```

## 项目状态

开发中 - aidevtools 已集成，测试框架核心功能开发中

## License

MIT License
