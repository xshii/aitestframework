# 需求文档索引

## 项目边界

```
┌─────────────────────────────────────────────────────────────┐
│                    本仓库：验证框架                          │
├─────────────────────────────────────────────────────────────┤
│  测试框架 (C)      - 断言宏、测试注册、运行器、报告输出       │
│  参考模型          - Golden数据管理、结果比较工具            │
│  平台适配层        - HAL接口、各平台适配桩/Mock              │
│  依赖管理          - 工具链版本管理、配套关系检查            │
│  用例管理          - 用例组织、筛选、配置、标签、平台映射     │
│  结果管理          - 收集、归档、报告、趋势分析              │
│  CI/CD            - 流水线定义、触发策略、通知              │
│  辅助工具          - Python工具、Web界面、数据生成          │
└─────────────────────────────────────────────────────────────┘
                              ↓ 测试
┌─────────────────────────────────────────────────────────────┐
│                    外部：被测对象                            │
├─────────────────────────────────────────────────────────────┤
│  HAL层             - 硬件抽象层（被测）                      │
│  Driver层          - NPU驱动（被测）                        │
│  算子实现          - 硬件算子（被测）                        │
└─────────────────────────────────────────────────────────────┘
                              ↓ 依赖
┌─────────────────────────────────────────────────────────────┐
│                    外部：工具链（二进制下载）                 │
├─────────────────────────────────────────────────────────────┤
│  仿真器 / 编译器 / ESL模型 / FPGA工具                        │
└─────────────────────────────────────────────────────────────┘
```

## 文档结构

```
docs/req/
├── README.md                 # 本文件：需求索引、DDD架构
│
│   # 基础
├── REQ-000-system.md         # REQ-SYS 系统级需求、事件定义
│
│   # 核心域 (Core Domain)
├── REQ-100-framework.md      # REQ-FWK 测试框架需求 (聚合: TestCase)
├── REQ-700-platform.md       # REQ-PLT 平台适配层需求 (聚合: Platform)
├── REQ-200-model.md          # REQ-MDL 参考模型需求 (聚合: Dataset)
│
│   # 支撑域 (Supporting Domain)
├── REQ-400-testmgmt.md       # REQ-TMT 用例管理需求 (聚合: TestSuite)
├── REQ-500-result.md         # REQ-RST 结果管理需求 (聚合: Execution)
│
│   # 通用域 (Generic Domain)
├── REQ-300-deps.md           # REQ-DEP 依赖管理需求
├── REQ-900-quality.md        # REQ-QCK 代码质量需求
├── REQ-600-cicd.md           # REQ-CIC CI/CD需求
└── REQ-B00-devtool.md        # REQ-DVT 开发工具需求 (原TLS+EFF)
│
│   # 已废弃 (合并到DVT)
├── REQ-800-tools.md          # [废弃] 已合并到 REQ-DVT
└── REQ-A00-efficiency.md     # [废弃] 已合并到 REQ-DVT
```

## 需求编号规则

```
REQ-<模块缩写>-<序号>
     │          │
     └──────────┴── 模块缩写(3字符) + 序号(3位)

模块分配 (按DDD域划分):

  # 基础
  SYS - 系统级需求、事件定义

  # 核心域 (Core Domain)
  FWK - 测试框架 (测试执行上下文)
  PLT - 平台适配层 (平台适配上下文)
  MDL - 参考模型 (参考数据上下文)

  # 支撑域 (Supporting Domain)
  TMT - 用例管理 (用例管理上下文)
  RST - 结果管理 (结果管理上下文)

  # 通用域 (Generic Domain)
  DEP - 依赖管理
  QCK - 代码质量
  CIC - CI/CD
  DVT - 开发工具 (原TLS+EFF合并)

  # 已废弃
  TLS - [废弃] 已合并到DVT
  EFF - [废弃] 已合并到DVT
```

## 模块概览

### 限界上下文与模块映射

| 上下文 | 缩写 | 模块名 | 描述 | 优先级 | 文件 |
|--------|------|--------|------|--------|------|
| - | SYS | System | 系统级需求、总体约束、事件定义 | P0 | REQ-000-system.md |
| **核心域** | FWK | Framework | C测试框架核心（聚合：TestCase） | P0 | REQ-100-framework.md |
| **核心域** | PLT | Platform | 平台适配层（聚合：Platform） | P0 | REQ-700-platform.md |
| **核心域** | MDL | Model | Golden数据与比较（聚合：Dataset） | P0 | REQ-200-model.md |
| **支撑域** | TMT | TestMgmt | 用例管理（聚合：TestSuite） | P0 | REQ-400-testmgmt.md |
| **支撑域** | RST | Result | 结果管理（聚合：Execution） | P1 | REQ-500-result.md |
| **通用域** | DEP | Deps | 依赖管理（独立） | P1 | REQ-300-deps.md |
| **通用域** | QCK | Quality | 代码质量检查（独立） | P0 | REQ-900-quality.md |
| **通用域** | CIC | CICD | 流水线（事件订阅者） | P1 | REQ-600-cicd.md |
| **通用域** | DVT | DevTool | 开发工具（原TLS+EFF合并） | P1 | REQ-B00-devtool.md |

**注**：原REQ-TLS和REQ-EFF已合并为REQ-DVT（开发工具），减少模块数量。

## 状态与优先级

**状态**:
| 状态 | 含义 |
|------|------|
| draft | 草稿 |
| approved | 已批准 |
| implemented | 已实现 |
| verified | 已验证 |

**优先级**:
| 级别 | 含义 |
|------|------|
| P0 | 必须，阻塞性 |
| P1 | 重要，应实现 |
| P2 | 可延后 |

## DDD架构设计

### 领域驱动设计原则

本框架采用DDD（领域驱动设计）组织模块，遵循以下原则：

1. **限界上下文（Bounded Context）**：每个上下文有独立的领域模型，通过明确的接口通信
2. **领域事件（Domain Event）**：上下文间通过事件解耦，而非直接调用
3. **聚合（Aggregate）**：每个上下文有明确的聚合根，保证数据一致性
4. **防腐层（ACL）**：上下文间通过适配器转换，防止模型污染

### 限界上下文划分

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         验证框架 - 限界上下文图                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ╔══════════════════════════════════════════════════════════════════╗  │
│   ║                    核心域 (Core Domain)                           ║  │
│   ║   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         ║  │
│   ║   │  测试执行    │    │  平台适配    │    │  参考数据    │         ║  │
│   ║   │  Context    │    │  Context    │    │  Context    │         ║  │
│   ║   │  (FWK)      │◄──►│  (PLT)      │    │  (MDL)      │         ║  │
│   ║   │             │    │             │    │             │         ║  │
│   ║   │ 聚合:       │    │ 聚合:       │    │ 聚合:       │         ║  │
│   ║   │ - TestCase  │    │ - Platform  │    │ - Dataset   │         ║  │
│   ║   │ - TestResult│    │ - HalOps    │    │ - Golden    │         ║  │
│   ║   └──────┬──────┘    └─────────────┘    └──────┬──────┘         ║  │
│   ╚══════════╪══════════════════════════════════════╪════════════════╝  │
│              │           领域事件                    │                   │
│              ▼                                       ▼                   │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │                     事件总线 (Event Bus)                          │  │
│   │  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐       │  │
│   │  │TestCompleted   │ │PlatformReady   │ │GoldenUpdated   │       │  │
│   │  └────────────────┘ └────────────────┘ └────────────────┘       │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│              │                                       │                   │
│              ▼                                       ▼                   │
│   ╔══════════════════════════════════════════════════════════════════╗  │
│   ║                   支撑域 (Supporting Domain)                      ║  │
│   ║   ┌─────────────┐                    ┌─────────────┐             ║  │
│   ║   │  用例管理    │   订阅事件         │  结果管理    │             ║  │
│   ║   │  Context    │◄──────────────────►│  Context    │             ║  │
│   ║   │  (TMT)      │                    │  (RST)      │             ║  │
│   ║   │             │                    │             │             ║  │
│   ║   │ 聚合:       │                    │ 聚合:       │             ║  │
│   ║   │ - TestSuite │                    │ - Execution │             ║  │
│   ║   │ - TestList  │                    │ - Report    │             ║  │
│   ║   └─────────────┘                    └─────────────┘             ║  │
│   ╚══════════════════════════════════════════════════════════════════╝  │
│                                                                          │
│   ╔══════════════════════════════════════════════════════════════════╗  │
│   ║                   通用域 (Generic Domain) - 可替换                ║  │
│   ║   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            ║  │
│   ║   │   DEP   │  │   QCK   │  │   CIC   │  │ DEVTOOL │            ║  │
│   ║   │ 依赖管理 │  │ 代码质量 │  │  CI/CD  │  │ 开发工具 │            ║  │
│   ║   └─────────┘  └─────────┘  └─────────┘  └─────────┘            ║  │
│   ║                     (通过事件订阅与核心域交互)                     ║  │
│   ╚══════════════════════════════════════════════════════════════════╝  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

图例: ◄──► 同步调用（核心域内部）   ───► 事件发布/订阅（跨域）
```

### 上下文映射 (Context Map)

| 上游 | 下游 | 关系类型 | 说明 |
|------|------|----------|------|
| FWK | TMT | 发布-订阅 | FWK发布TestRegistered事件，TMT订阅 |
| FWK | RST | 发布-订阅 | FWK发布TestCompleted事件，RST订阅 |
| FWK | PLT | 共享内核 | 共享HalOps接口定义 |
| MDL | FWK | 客户-供应商 | FWK使用MDL的Golden数据格式 |
| RST | CIC | 发布-订阅 | RST发布ReportGenerated事件，CIC订阅 |
| QCK | CIC | 遵奉者 | CIC遵守QCK定义的检查规则 |

### 领域事件定义

```yaml
# 核心域事件 (由FWK发布)
TestExecutionStarted:
  source: FWK
  payload:
    executionId: STRING
    platform: STRING
    testCount: INT32

TestCaseCompleted:
  source: FWK
  payload:
    executionId: STRING
    testName: STRING
    result: TestResultEnum
    durationMs: UINT32
    errorInfo: STRING (optional)

TestExecutionCompleted:
  source: FWK
  payload:
    executionId: STRING
    summary: TestSummaryStru

# 平台域事件 (由PLT发布)
PlatformInitialized:
  source: PLT
  payload:
    platform: STRING
    capabilities: PlatformCapsStru

# 数据域事件 (由MDL发布)
GoldenDataUpdated:
  source: MDL
  payload:
    dataset: STRING
    version: STRING
    checksum: STRING

# 结果域事件 (由RST发布)
ReportGenerated:
  source: RST
  payload:
    executionId: STRING
    reportPath: STRING
    format: STRING
```

### 聚合定义

| 上下文 | 聚合根 | 包含实体 | 值对象 |
|--------|--------|----------|--------|
| FWK | TestCase | - | TestCaseInfo, TestResult |
| FWK | TestExecution | TestCaseResult[] | TestSummary |
| PLT | Platform | HalOps | PlatformCaps |
| MDL | GoldenDataset | GoldenFile[] | GoldenHeader |
| TMT | TestSuite | TestConfig[] | TestFilter, TestTag |
| RST | Execution | TestResult[] | Report |

### 模块合并

为减少模块数量，将功能相近的模块合并：

| 原模块 | 合并后 | 新编号 | 说明 |
|--------|--------|--------|------|
| TLS + EFF | **DEVTOOL** | REQ-DVT | 辅助工具 + 效率工具 → 开发工具 |

合并后模块数：11 → **10**

### 依赖关系（基于DDD）

```
┌─────────────────────────────────────────────────────────────────┐
│  核心域内部: 同步调用 (Shared Kernel)                            │
│                                                                  │
│    FWK ◄────────► PLT  (共享 HalOps 接口)                       │
│     │                                                            │
│     │ 使用                                                       │
│     ▼                                                            │
│    MDL  (Golden数据格式)                                         │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│  跨域: 事件驱动 (Event-Driven)                                   │
│                                                                  │
│    FWK ──TestCompleted──► RST (结果收集)                        │
│    FWK ──TestCompleted──► CIC (CI报告)                          │
│    RST ──ReportGenerated──► CIC (报告集成)                       │
│    MDL ──GoldenUpdated──► DEVTOOL (数据同步)                    │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│  通用域: 无依赖 (独立运行)                                       │
│                                                                  │
│    DEP  (独立的工具链管理)                                       │
│    QCK  (独立的代码检查)                                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 防腐层 (Anti-Corruption Layer)

各上下文通过防腐层隔离，防止模型污染：

```c
/* RST上下文的防腐层 - 转换FWK的事件 */
/* src/result/acl/fwk_adapter.c */

typedef struct RstTestResultStru {
    /* RST上下文内部模型，与FWK解耦 */
    CHAR testName[256];
    INT32 status;           /* RST内部状态码，非FWK的TestResultEnum */
    UINT32 durationMs;
} RstTestResultStru;

/* 事件适配器：将FWK事件转换为RST内部模型 */
SEC_DDR_TEXT ERRNO_T RST_ACL_ConvertTestResult(
    const TestCaseCompletedEvent *fwkEvent,
    RstTestResultStru *rstResult);
```

---

## 需求汇总表

### REQ-SYS 系统级需求
| ID | 标题 | 优先级 | 状态 | MVP |
|----|------|--------|------|-----|
| REQ-SYS-001 | 验证框架定位 | P0 | draft | ✓ |
| REQ-SYS-002 | 编程语言约束 | P0 | draft | ✓ |
| REQ-SYS-003 | 跨平台验证支持 | P1 | draft | |
| REQ-SYS-004 | 验证流水线阶段 | P1 | draft | |
| REQ-SYS-005 | 框架可扩展性 | P1 | draft | |
| REQ-SYS-006 | 文档要求 | P1 | draft | |
| REQ-SYS-007 | 代码质量要求 | P1 | draft | |
| REQ-SYS-008 | 测试覆盖率要求 | P1 | draft | |
| REQ-SYS-009 | 框架版本兼容性 | P2 | draft | |
| REQ-SYS-010 | 基础类型定义 | P0 | draft | ✓ |
| REQ-SYS-011 | 头文件组织规范 | P1 | draft | |
| REQ-SYS-012 | 跨模块接口契约 | P1 | draft | |
| REQ-SYS-013 | 领域事件机制 | P1 | draft | |

### REQ-FWK 测试框架需求
| ID | 标题 | 优先级 | 状态 | MVP |
|----|------|--------|------|-----|
| REQ-FWK-001 | 测试用例自动注册 | P0 | draft | ✓ |
| REQ-FWK-002 | 断言宏集合 | P0 | draft | ✓ |
| REQ-FWK-003 | 测试结果类型 | P0 | draft | ✓ |
| REQ-FWK-004 | 测试运行器 | P0 | draft | ✓ |
| REQ-FWK-005 | 测试输出格式 | P0 | draft | ✓ |
| REQ-FWK-006 | Setup/Teardown | P1 | draft | |
| REQ-FWK-007 | 测试日志接口 | P1 | draft | |
| REQ-FWK-008 | 参数化测试 | P1 | draft | |
| REQ-FWK-009 | 数据驱动测试 | P1 | draft | |
| REQ-FWK-010 | Mock框架 | P1 | draft | |
| REQ-FWK-011 | 约束随机测试 | P2 | draft | |
| REQ-FWK-012 | 性能基准测试 | P1 | draft | |
| REQ-FWK-013 | 功能覆盖率 | P2 | draft | |
| REQ-FWK-014 | Fixture环境管理 | P1 | draft | |
| REQ-FWK-015 | 失败调试支持 | P2 | draft | |
| REQ-FWK-016 | 测试分片执行 | P2 | draft | |
| REQ-FWK-017 | 测试用例隔离 | P1 | draft | |
| REQ-FWK-018 | 失败重现支持 | P1 | draft | |
| REQ-FWK-019 | 信号处理机制 | P0 | draft | ✓ |
| REQ-FWK-020 | 超时粒度细化 | P1 | draft | |
| REQ-FWK-021 | 测试运行器回调 | P2 | draft | |

### REQ-MDL 参考模型需求
| ID | 标题 | 优先级 | 状态 | MVP |
|----|------|--------|------|-----|
| REQ-MDL-001 | Golden数据管理 | P0 | draft | ✓ |
| REQ-MDL-002 | 结果比较工具 | P0 | draft | ✓ |
| REQ-MDL-003 | 测试数据版本管理 | P2 | draft | |

### REQ-DEP 依赖管理需求
| ID | 标题 | 优先级 | 状态 | MVP |
|----|------|--------|------|-----|
| REQ-DEP-001 | 依赖配置 | P1 | draft | |
| REQ-DEP-002 | 依赖关系刷新 | P1 | draft | |
| REQ-DEP-003 | 命令行接口 | P1 | draft | |

### REQ-TMT 用例管理需求
| ID | 标题 | 优先级 | 状态 | MVP |
|----|------|--------|------|-----|
| REQ-TMT-001 | 用例目录结构 | P0 | draft | ✓ |
| REQ-TMT-002 | 用例平台映射 | P1 | draft | |
| REQ-TMT-003 | 用例标签系统 | P1 | draft | |
| REQ-TMT-004 | 执行配置 | P1 | draft | |
| REQ-TMT-005 | 测试列表定义 | P1 | draft | |
| REQ-TMT-006 | 用例元数据管理 | P2 | draft | |
| REQ-TMT-007 | 自动用例发现 | P0 | draft | ✓ |

### REQ-RST 结果管理需求
| ID | 标题 | 优先级 | 状态 | MVP |
|----|------|--------|------|-----|
| REQ-RST-001 | 测试结果收集 | P0 | draft | ✓ |
| REQ-RST-002 | 结果持久化存储 | P1 | draft | |
| REQ-RST-003 | 历史结果归档 | P2 | draft | |
| REQ-RST-004 | 测试报告生成 | P1 | draft | |
| REQ-RST-005 | 结果趋势分析 | P2 | draft | |
| REQ-RST-006 | 结果差异对比 | P2 | draft | |
| REQ-RST-007 | 结果通知 | P2 | draft | |
| REQ-RST-008 | 分布式日志收集 | P2 | draft | |

### REQ-CIC CI/CD需求

**用例CI/CD（运行NPU验证用例）：**
| ID | 标题 | 优先级 | 状态 | MVP |
|----|------|--------|------|-----|
| REQ-CIC-001 | 验证流水线阶段 | P1 | draft | |
| REQ-CIC-002 | 流水线触发策略 | P1 | draft | |
| REQ-CIC-003 | CI作业定义 | P1 | draft | |
| REQ-CIC-004 | 并行测试执行 | P1 | draft | |
| REQ-CIC-005 | 执行环境管理 | P1 | draft | |
| REQ-CIC-006 | CI平台集成 | P1 | draft | |
| REQ-CIC-007 | 失败处理策略 | P2 | draft | |
| REQ-CIC-008 | CI报告集成 | P2 | draft | |
| REQ-CIC-009 | 资源配额管理 | P2 | draft | |

**框架CI/CD（框架自身构建测试发布）：**
| ID | 标题 | 优先级 | 状态 | MVP |
|----|------|--------|------|-----|
| REQ-CIC-010 | 框架代码PR检查 | P1 | draft | |
| REQ-CIC-011 | 框架版本发布流程 | P1 | draft | |
| REQ-CIC-012 | 文档自动生成发布 | P2 | draft | |
| REQ-CIC-013 | Python工具包发布 | P2 | draft | |

### REQ-PLT 平台适配层需求
| ID | 标题 | 优先级 | 状态 | MVP |
|----|------|--------|------|-----|
| REQ-PLT-001 | HAL统一接口规范 | P0 | draft | ✓ |
| REQ-PLT-002 | LinuxUT Mock实现 | P0 | draft | ✓ |
| REQ-PLT-003 | LinuxST Stub实现 | P1 | draft | |
| REQ-PLT-004 | Simulator适配桩 | P1 | draft | |
| REQ-PLT-005 | ESL适配桩 | P2 | draft | |
| REQ-PLT-006 | FPGA适配桩 | P2 | draft | |
| REQ-PLT-007 | Chip适配桩 | P2 | draft | |
| REQ-PLT-008 | 平台选择机制 | P0 | draft | ✓ |
| REQ-PLT-009 | 平台配置文件 | P1 | draft | |
| REQ-PLT-010 | 跨平台数据兼容 | P1 | draft | |

### REQ-QCK 代码质量需求
| ID | 标题 | 优先级 | 状态 | MVP |
|----|------|--------|------|-----|
| REQ-QCK-001 | 类型命名检查 | P1 | draft | |
| REQ-QCK-002 | 命名规范检查 | P1 | draft | |
| REQ-QCK-003 | 内存段标记检查 | P1 | draft | |
| REQ-QCK-004 | 安全函数检查 | P1 | draft | |
| REQ-QCK-005 | 错误码检查 | P1 | draft | |
| REQ-QCK-006 | 判断宏使用检查 | P2 | draft | |
| REQ-QCK-007 | 静态分析工具集成 | P1 | draft | |
| REQ-QCK-008 | 增量检查 | P2 | draft | |
| REQ-QCK-009 | 规则配置 | P2 | draft | |
| REQ-QCK-010 | 内存安全检测 | P2 | draft | |
| REQ-QCK-011 | 代码复杂度检查 | P2 | draft | |

### REQ-DVT 开发工具需求 (原TLS+EFF合并)
| ID | 标题 | 优先级 | 状态 | MVP |
|----|------|--------|------|-----|
| REQ-DVT-001 | Python测试运行器 | P1 | draft | |
| REQ-DVT-002 | 数据生成工具 | P1 | draft | |
| REQ-DVT-003 | 报告生成工具 | P1 | draft | |
| REQ-DVT-004 | 用例管理Web界面 | P2 | draft | |
| REQ-DVT-005 | 结果归档工具 | P2 | draft | |
| REQ-DVT-006 | 数据格式转换工具 | P1 | draft | |
| REQ-DVT-007 | 环境检查工具 | P1 | draft | |
| REQ-DVT-008 | VSCode插件集成 | P2 | draft | |
| REQ-DVT-009 | 右键菜单与快捷命令 | P2 | draft | |
| REQ-DVT-010 | 代码片段模板 | P2 | draft | |
| REQ-DVT-011 | 一键测试执行 | P2 | draft | |
| REQ-DVT-012 | 问题导航跳转 | P2 | draft | |

---

## 统计

| 模块 | 上下文 | P0 | P1 | P2 | 总计 |
|------|--------|-----|-----|-----|------|
| SYS | - | 3 | 7 | 3 | 13 |
| FWK | 核心域 | 6 | 9 | 6 | 21 |
| PLT | 核心域 | 3 | 4 | 3 | 10 |
| MDL | 核心域 | 2 | 0 | 1 | 3 |
| TMT | 支撑域 | 2 | 4 | 1 | 7 |
| RST | 支撑域 | 1 | 2 | 5 | 8 |
| DEP | 通用域 | 0 | 3 | 0 | 3 |
| QCK | 通用域 | 0 | 5 | 6 | 11 |
| CIC | 通用域 | 0 | 8 | 5 | 13 |
| DVT | 通用域 | 0 | 4 | 8 | 12 |
| **总计** | | **17** | **46** | **38** | **101** |

### MVP统计 (P0需求)

| 域 | 模块 | P0需求 | 说明 |
|----|------|--------|------|
| 基础 | SYS | 3 | 定位、语言、类型定义 |
| 核心域 | FWK | 6 | 注册、断言、结果、运行器、输出、信号处理 |
| 核心域 | PLT | 3 | HAL接口、LinuxUT Mock、平台选择 |
| 核心域 | MDL | 2 | Golden管理、比较工具 |
| 支撑域 | TMT | 2 | 目录结构、用例发现 |
| 支撑域 | RST | 1 | 结果收集 |
| **总计** | | **17** | **MVP核心功能** |

### 优先级分布

```
P0 (MVP必须):  17个 (16.8%) ████
P1 (重要):     46个 (45.5%) ████████████████
P2 (可延后):   38个 (37.6%) ████████████████
```

---

## 技术选型

### 原则

- **C代码**：零外部依赖（纯C99 + POSIX）
- **Python**：最小依赖 + 标准库优先

### C代码

| 功能 | 方案 |
|------|------|
| JSON输出 | 内置（仅输出，~100行） |
| 用例注册 | linker section |
| 信号/Socket/共享内存 | POSIX标准库 |
| CRC32 | 内置（~50行） |

**外部依赖：0**

### Python依赖

```
# 核心 (3个，必须)
pyyaml>=6.0      # 配置解析
httpx>=0.24      # HTTP下载（异步、HTTP/2）
numpy>=1.20      # 数值计算

# 增强 (2个，推荐)
jinja2>=3.0      # HTML报告模板
rich>=13.0       # 终端UI（进度条+表格+着色）

# 可选 (1个)
flask>=2.0       # Web界面（仅DVT-004需要）
```

**标准库替代第三方库：**

| 用标准库 | 不用 |
|----------|------|
| argparse | click/typer |
| json | simplejson |
| sqlite3 | sqlalchemy |
| logging | loguru |
| concurrent.futures | celery |
| subprocess | sh |
| pathlib, tempfile, hashlib | - |

### 模块依赖映射

| 模块 | 依赖 |
|------|------|
| FWK/PLT/SYS | 无（纯C） |
| MDL | numpy |
| TMT/CIC | pyyaml |
| DEP | httpx, pyyaml, rich |
| RST | jinja2, rich |
| QCK | rich + 外部工具(clang-tidy/cppcheck) |
| DVT | numpy, jinja2, rich, flask(可选) |

### 不使用

| 库 | 原因 |
|----|------|
| pandas | 太重，numpy够用 |
| requests | httpx更现代 |
| tqdm/colorama | rich已包含 |
| sqlalchemy | sqlite3够用 |
| click | argparse够用 |

### 统计

| 类型 | 数量 |
|------|------|
| C外部依赖 | 0 |
| Python核心 | 3 |
| Python增强 | 2 |
| Python可选 | 1 |
| **运行时总计** | **5~6** |

---

## 变更记录

| 日期 | 版本 | 变更说明 |
|------|------|----------|
| 2026-02-04 | 2.4 | **设计文档刷新**：根目录README.md更新DDD架构、简化目录结构、同步技术选型 |
| 2026-02-04 | 2.3 | **技术选型**：C代码零依赖；Python核心3个(pyyaml/httpx/numpy)+增强2个(jinja2/rich)；click改用标准库argparse |
| 2026-02-04 | 2.2 | **框架CI/CD**：REQ-CIC新增框架自身CI/CD需求（010~013），区分用例CI/CD和框架CI/CD；总需求97→101 |
| 2026-02-04 | 2.1 | **P0精简**：MVP从54个精简至17个P0需求；通用域（DEP/QCK/CIC/DVT）全部降为P1/P2；确保P0仅包含LinuxUT最小可运行功能 |
| 2026-02-04 | 2.0 | **DDD重构**：引入限界上下文、领域事件、聚合；合并TLS+EFF为DVT；新增REQ-SYS-013领域事件机制 |
| 2026-02-04 | 1.3 | 第四轮评审：优化模块依赖关系，增加依赖层次图，明确跨模块接口，修复EFF/CIC依赖声明 |
| 2026-02-04 | 1.2 | 第三轮评审修订：增加TEST_CRASH状态，补充错误码分段，修复类型引用，补充依赖关系 |
| 2026-02-04 | 1.1 | 需求评审后修订：新增13条需求，调整2条优先级，修复一致性问题 |
| 2026-02-03 | 1.0 | 初始版本 |
