# AI Test Framework 目录结构

基于需求文档 `doc/req/00-requirements-index.yaml` 设计的目录结构。

## 顶层结构

```
aitestframework/                    # 项目根目录
│
├── aitestframework/                # 主框架包（自研代码）
│   ├── core/                       # CORE - 核心框架
│   ├── model/                      # MODEL - AI模型测试
│   ├── data/                       # DATA - 数据管理
│   ├── assertion/                  # ASSERT - 断言与验证
│   ├── report/                     # REPORT - 报告生成
│   ├── integration/                # INTEG - 集成与部署
│   ├── extension/                  # EXT - 扩展性
│   └── adt/                        # ADT - aidevtools集成适配层
│
├── libs/                           # 外部集成库
│   ├── aidevtools/                 # 算子验证工具集
│   └── prettycli/                  # CLI美化工具
│
├── backends/                       # 多后端实现
│   ├── common/                     # 公共模块
│   ├── cpu/                        # CPU后端 (Golden生成)
│   └── npu/                        # NPU后端 (DUT验证)
│
├── demos/                          # 示例代码
├── tests/                          # 测试用例
├── doc/                            # 文档
│   ├── req/                        # 需求文档
│   ├── api/                        # API文档
│   └── guides/                     # 使用指南
│
└── scripts/                        # 脚本工具
```

## 模块说明

### aitestframework/ - 主框架包

按需求模块组织，对应 `doc/req/` 中的需求文档：

| 目录 | 模块ID | 需求文档 | 说明 |
|------|--------|----------|------|
| core/ | CORE | 01-core-framework.yaml | 测试引擎、用例发现、执行调度 |
| model/ | MODEL | 02-ai-model-testing.yaml | 模型加载、推理测试、精度验证 |
| data/ | DATA | 03-data-management.yaml | 数据加载、生成、预处理 |
| assertion/ | ASSERT | 04-assertion-validation.yaml | 断言方法、结果验证 |
| report/ | REPORT | 05-report-generation.yaml | 报告生成、可视化 |
| integration/ | INTEG | 06-integration-deployment.yaml | CI/CD、部署集成 |
| extension/ | EXT | 07-extensibility.yaml | 插件系统、API扩展 |
| adt/ | ADT | 08-aidevtools-integration.yaml | aidevtools适配层 |

### libs/ - 外部集成库

集中管理外部依赖的工具/仓库：

| 目录 | 来源 | 说明 |
|------|------|------|
| aidevtools/ | 内部仓库 | 算子验证工具集（四状态比对、Golden生成） |
| prettycli/ | 内部仓库 | CLI美化工具 |
| [预留] pcit/ | 待集成 | CPU Golden生成工具 |

### backends/ - 多后端支持

| 目录 | 说明 |
|------|------|
| common/ | 后端抽象层、公共接口 |
| cpu/ | CPU后端，生成Golden（通过libs/aidevtools或pcit） |
| npu/ | NPU后端，执行DUT获取输出 |

## 使用方式

### 1. 直接使用aidevtools

```python
# 设置Python路径
import sys
sys.path.insert(0, 'libs')

# 导入aidevtools
from aidevtools.compare import CompareEngine, CompareConfig
from aidevtools.compare.types import CompareStatus
```

### 2. 通过aitestframework适配层

```python
from aitestframework.adt import CompareEngine, CompareStatus

engine = CompareEngine()
result = engine.compare(dut, golden_pure, golden_qnt)
```

### 3. 通过顶层导入

```python
from aitestframework import CompareEngine, CompareStatus
```

## 依赖关系

```
aitestframework
    └── adt (适配层)
            └── libs/aidevtools (实际实现)

backends
    ├── cpu
    │   └── libs/aidevtools (Golden生成)
    └── npu
        └── 厂商SDK (DUT执行)
```

## 扩展新的外部库

1. 将外部仓库放入 `libs/` 目录
2. 在 `aitestframework/` 中创建对应的适配模块
3. 更新 `libs/__init__.py` 的导出列表
