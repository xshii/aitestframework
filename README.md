# AI芯片应用侧验证框架

## 一、项目定位

### 核心需求

- **应用侧验证**：验证AI加速器的软件/驱动层
- **验证代码用C**：纯C99实现，零外部依赖
- **工具链二进制下载**：仿真器、编译器、OS平台软件通过httpx下载
- **辅助工具用Python**：CI/CD、报告生成、数据工具（最小依赖）
- **跨平台复用**：同一套测试代码在不同平台运行

### 架构边界

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
│  HAL层 / Driver层 / 算子实现（被测）                        │
└─────────────────────────────────────────────────────────────┘
                              ↓ 依赖
┌─────────────────────────────────────────────────────────────┐
│                    外部：工具链（httpx下载）                  │
├─────────────────────────────────────────────────────────────┤
│  仿真器 / 编译器 / ESL模型 / FPGA工具                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、DDD架构设计

本框架采用领域驱动设计（DDD），划分为核心域、支撑域、通用域三个层次。

### 2.1 限界上下文

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         验证框架 - 限界上下文图                           │
├─────────────────────────────────────────────────────────────────────────┤
│   ╔══════════════════════════════════════════════════════════════════╗  │
│   ║                    核心域 (Core Domain)                           ║  │
│   ║   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         ║  │
│   ║   │  测试执行    │    │  平台适配    │    │  参考数据    │         ║  │
│   ║   │  (FWK)      │◄──►│  (PLT)      │    │  (MDL)      │         ║  │
│   ║   │ 聚合:TestCase│    │ 聚合:Platform│    │ 聚合:Dataset │         ║  │
│   ║   └──────┬──────┘    └─────────────┘    └──────┬──────┘         ║  │
│   ╚══════════╪══════════════════════════════════════╪════════════════╝  │
│              │           领域事件                    │                   │
│              ▼                                       ▼                   │
│   ╔══════════════════════════════════════════════════════════════════╗  │
│   ║                   支撑域 (Supporting Domain)                      ║  │
│   ║   ┌─────────────┐                    ┌─────────────┐             ║  │
│   ║   │  用例管理    │   订阅事件         │  结果管理    │             ║  │
│   ║   │  (TMT)      │◄──────────────────►│  (RST)      │             ║  │
│   ║   │ 聚合:TestSuite│                   │ 聚合:Execution│            ║  │
│   ║   └─────────────┘                    └─────────────┘             ║  │
│   ╚══════════════════════════════════════════════════════════════════╝  │
│   ╔══════════════════════════════════════════════════════════════════╗  │
│   ║                   通用域 (Generic Domain) - 可替换                ║  │
│   ║   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            ║  │
│   ║   │   DEP   │  │   QCK   │  │   CIC   │  │   DVT   │            ║  │
│   ║   │ 依赖管理 │  │ 代码质量 │  │  CI/CD  │  │ 开发工具 │            ║  │
│   ║   └─────────┘  └─────────┘  └─────────┘  └─────────┘            ║  │
│   ╚══════════════════════════════════════════════════════════════════╝  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 模块概览

| 域 | 缩写 | 模块名 | 描述 | 需求文档 |
|----|------|--------|------|----------|
| 基础 | SYS | System | 系统级需求、总体约束、事件定义 | REQ-000-system.md |
| **核心域** | FWK | Framework | C测试框架核心（聚合：TestCase） | REQ-100-framework.md |
| **核心域** | PLT | Platform | 平台适配层（聚合：Platform） | REQ-700-platform.md |
| **核心域** | MDL | Model | Golden数据与比较（聚合：Dataset） | REQ-200-model.md |
| **支撑域** | TMT | TestMgmt | 用例管理（聚合：TestSuite） | REQ-400-testmgmt.md |
| **支撑域** | RST | Result | 结果管理（聚合：Execution） | REQ-500-result.md |
| **通用域** | DEP | Deps | 依赖管理（独立） | REQ-300-deps.md |
| **通用域** | QCK | Quality | 代码质量检查（独立） | REQ-900-quality.md |
| **通用域** | CIC | CICD | 流水线（事件订阅者） | REQ-600-cicd.md |
| **通用域** | DVT | DevTool | 开发工具 | REQ-B00-devtool.md |

### 2.3 验证流水线

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         验证流水线 Pipeline                              │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────────────┤
│  LinuxUT    │  LinuxST    │  功能验证    │  性能ESL    │   原型/EDA      │
│  单元测试   │  系统测试    │             │             │                 │
├─────────────┴─────────────┼─────────────┼─────────────┼─────────────────┤
│     Host Linux环境        │  Simulator  │  ESL Model  │  FPGA/Chip      │
└───────────────────────────┴─────────────┴─────────────┴─────────────────┘
```

| 阶段 | 环境 | 目标 | 速度 | 精度 |
|------|------|------|------|------|
| **LinuxUT** | Host Linux | 驱动/框架单元测试 | 最快 | 逻辑级 |
| **LinuxST** | Host Linux | 系统集成测试 | 快 | 逻辑级 |
| **功能验证** | Simulator | 算子功能正确性 | 中 | 位精确 |
| **性能ESL** | ESL Model | 性能/时序/带宽 | 中 | 周期级 |
| **原型/EDA** | FPGA/Chip | 真实硬件验证 | 慢 | 硬件级 |

### 2.4 分层架构

```
┌─────────────────────────────────────────────────────────────┐
│              Python辅助层 (CI/CD, 报告, 数据生成)             │
├──────────────────────────┬──────────────────────────────────┤
│   Python参考模型         │        C测试用例                  │
│   (numpy)                │   UT/ST/功能/性能测试             │
├──────────────────────────┴──────────────────────────────────┤
│                  Test Framework (C测试框架)                  │
│    断言宏 | 测试注册 | 结果统计 | 日志输出 | 超时控制          │
├─────────────────────────────────────────────────────────────┤
│                  HAL Layer (C硬件抽象层)                     │
│        reg_read/write | dma_transfer | irq_wait | mem_map   │
├────────────┬────────────┬────────────┬────────────┬─────────┤
│  LinuxUT   │  LinuxST   │ Simulator  │    ESL     │FPGA/Chip│
│  Mock HAL  │  Mock HAL  │  Sim HAL   │  ESL HAL   │ HW HAL  │
└────────────┴────────────┴────────────┴────────────┴─────────┘
```

---

## 三、目录结构

```
aitestframework/
│
├── README.md                       # 本文件：项目设计
├── Makefile                        # 顶层构建
├── requirements.txt                # Python依赖
│
├── docs/                           # 文档
│   ├── req/                        # 需求文档
│   │   ├── README.md               # 需求索引、DDD架构详细说明
│   │   └── REQ-*.md                # 各模块需求
│   └── coding_style.md             # C编码规范
│
├── include/                        # C公开头文件
│   ├── common/
│   │   ├── types.h                 # SYS-010 基础类型
│   │   ├── errno.h                 # 错误码定义
│   │   └── macros.h                # 通用宏
│   ├── framework/
│   │   ├── test_case.h             # FWK-001 用例注册宏
│   │   ├── assert.h                # FWK-002 断言宏
│   │   ├── result.h                # FWK-003 结果类型
│   │   └── runner.h                # FWK-004 运行器接口
│   ├── hal/
│   │   └── hal_ops.h               # PLT-001 HAL接口
│   └── model/
│       └── compare.h               # MDL-002 比较接口
│
├── src/                            # C源代码
│   ├── framework/
│   │   ├── runner.c                # FWK-004 运行器实现
│   │   ├── output.c                # FWK-005 输出格式
│   │   └── signal.c                # FWK-019 信号处理
│   ├── platform/
│   │   ├── linux_ut/
│   │   │   └── ut_hal.c            # PLT-002 LinuxUT Mock
│   │   └── platform.c              # PLT-008 平台选择
│   └── model/
│       └── compare.c               # MDL-002 比较实现
│
├── tests/                          # TMT-001 测试用例
│   ├── unit/                       # 单元测试
│   │   └── framework/
│   └── functional/                 # 功能测试
│       └── sanity/
│
├── testdata/                       # MDL-001 Golden数据
│   ├── inputs/
│   └── golden/
│
├── tools/                          # Python工具（DVT）
│   ├── runner/                     # 测试运行器
│   ├── data/                       # 数据生成工具
│   ├── report/                     # 报告生成
│   └── deps/                       # 依赖管理
│
├── deps/                           # 依赖配置
│   └── config.yaml                 # 依赖版本配置
│
└── build/                          # 构建输出 (gitignore)
    ├── bin/
    └── reports/
```

---

## 四、技术选型

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

### 模块依赖映射

| 模块 | 依赖 |
|------|------|
| FWK/PLT/SYS | 无（纯C） |
| MDL | numpy |
| TMT/CIC | pyyaml |
| DEP | httpx, pyyaml, rich |
| RST | jinja2, rich |
| DVT | numpy, jinja2, rich, flask(可选) |

---

## 五、关键设计

### 5.1 HAL层设计（函数指针多态）

```c
/* include/hal/hal_ops.h */
typedef struct HalOpsStru {
    ERRNO_T (*RegWrite)(VOID *addr, UINT32 value);
    ERRNO_T (*RegRead)(VOID *addr, UINT32 *value);
    ERRNO_T (*MemAlloc)(UINT64 size, VOID **ptr);
    VOID    (*MemFree)(VOID *ptr);
    ERRNO_T (*Init)(VOID);
    ERRNO_T (*Deinit)(VOID);
} HalOpsStru;

extern HalOpsStru *g_hal;
```

### 5.2 测试框架核心

```c
/* include/framework/test_case.h */
typedef ERRNO_T (*TestFunc)(VOID);

typedef struct TestCaseStru {
    const CHAR *suite;
    const CHAR *name;
    TestFunc    func;
    UINT32      timeoutMs;
    const CHAR *tags;
} TestCaseStru;

/* linker section方式注册 */
#define TEST_CASE(suite, name) \
    static ERRNO_T test_##suite##_##name(VOID); \
    static TestCaseStru __test_##suite##_##name \
    __attribute__((section(".testcases"), used)) = { \
        .suite = #suite, .name = #name, \
        .func = test_##suite##_##name, \
        .timeoutMs = 5000 \
    }; \
    static ERRNO_T test_##suite##_##name(VOID)

/* include/framework/assert.h */
#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        g_testCtx.failFile = __FILE__; \
        g_testCtx.failLine = __LINE__; \
        return TEST_FAIL; \
    } \
} while(0)

#define TEST_ASSERT_EQ(exp, act) TEST_ASSERT((exp) == (act))

/* include/framework/result.h */
typedef enum TestResultEnum {
    TEST_PASS    = 0,
    TEST_FAIL    = 1,
    TEST_SKIP    = 2,
    TEST_TIMEOUT = 3,
    TEST_ERROR   = 4,
    TEST_CRASH   = 5,
} TestResultEnum;
```

---

## 六、快速开始

### 环境准备

```bash
# 克隆仓库
git clone <repo_url>
cd aitestframework

# 安装Python依赖
pip install -r requirements.txt
```

### 构建

```bash
# 构建LinuxUT版本
make PLATFORM=linux_ut

# 构建Simulator版本
make PLATFORM=simulator
```

### 运行测试

```bash
# 运行LinuxUT单元测试
./build/bin/test_runner

# 运行指定用例
./build/bin/test_runner --filter "matmul*"

# 查看结果
cat build/reports/result.json
```

---

## 七、需求统计

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

**MVP (P0)**: 17个需求，覆盖LinuxUT平台最小可运行功能

详细需求见 [docs/req/README.md](docs/req/README.md)

---

## 八、文档索引

| 文档 | 说明 |
|------|------|
| [docs/req/README.md](docs/req/README.md) | 需求索引、DDD架构详细说明 |
| [docs/req/REQ-100-framework.md](docs/req/REQ-100-framework.md) | 测试框架需求 |
| [docs/req/REQ-700-platform.md](docs/req/REQ-700-platform.md) | 平台适配层需求 |
| [docs/req/REQ-200-model.md](docs/req/REQ-200-model.md) | 参考模型需求 |
| [docs/coding_style.md](docs/coding_style.md) | C编码规范 |
