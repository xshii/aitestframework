# REQ-SYS 系统级需求

---
id: REQ-SYS
title: 系统级需求
priority: P0
status: draft
---

## 概述

AI芯片应用侧验证框架的系统级需求，定义项目边界、总体目标和约束。

---

## REQ-SYS-001 项目定位

---
id: REQ-SYS-001
title: 验证框架定位
priority: P0
status: draft
parent: REQ-SYS
---

### 描述

本框架是**验证工具**，不是被测对象。被测对象（HAL、Driver、算子）由外部仓库提供。

### 边界定义

| 范围 | 包含 | 不包含 |
|------|------|--------|
| 本仓库 | 测试框架、参考模型、用例管理、结果管理、CI/CD | HAL实现、Driver实现、算子实现 |
| 外部 | 被测代码、工具链二进制 | - |

### 被测代码接口规范

被测代码需要实现以下接口才能被本框架测试：

```c
/* 被测代码必须提供的头文件 */
#include "npu_hal.h"      /* HAL层接口 */
#include "npu_driver.h"   /* 驱动层接口 */
#include "npu_ops.h"      /* 算子接口 */
```

### 配置文件示例

```yaml
# configs/target.yaml - 指定被测代码
target:
  # 被测代码路径
  source_path: "/path/to/npu-sdk/src"
  include_path: "/path/to/npu-sdk/include"
  library_path: "/path/to/npu-sdk/lib"

  # 被测库
  libraries:
    - libnpu_hal.a
    - libnpu_driver.a

  # 接口头文件
  headers:
    hal: "npu_hal.h"
    driver: "npu_driver.h"
    ops: "npu_ops.h"
```

### 验收标准

1. 框架代码与被测代码完全解耦
2. 通过 `configs/target.yaml` 指定被测代码路径和接口
3. 框架可独立编译和运行（使用桩/Mock）
4. 被测代码只需实现规定的接口即可接入测试

---

## REQ-SYS-002 语言约束

---
id: REQ-SYS-002
title: 编程语言约束
priority: P0
status: draft
parent: REQ-SYS
---

### 描述

测试框架核心用C语言，辅助工具用Python。

### 验收标准

| 组件 | 语言 | 标准/版本 | 外部依赖 |
|------|------|-----------|----------|
| 测试框架 | C | C99 | 无 |
| 测试用例 | C | C99 | 无 |
| C参考模型 | C | C99 | 无 |
| Python参考模型 | Python | 3.8+ | numpy |
| 辅助工具 | Python | 3.8+ | 见requirements.txt |

---

## REQ-SYS-003 跨平台验证

---
id: REQ-SYS-003
title: 跨平台验证支持
priority: P1
status: draft
parent: REQ-SYS
---

### 描述

同一套测试用例能在不同验证平台运行。

### 支持平台

| 平台 | 环境 | 用途 | 速度 |
|------|------|------|------|
| LinuxUT | Host Linux + Mock | 单元测试 | 最快 |
| LinuxST | Host Linux + Stub | 系统测试 | 快 |
| Simulator | 功能仿真器 | 功能验证 | 中 |
| ESL | 性能模型 | 性能验证 | 中 |
| FPGA | FPGA原型 | 原型验证 | 慢 |
| Chip | 真实芯片 | 硅后验证 | 慢 |

### LinuxUT vs LinuxST 区别

| 特性 | LinuxUT (单元测试) | LinuxST (系统测试) |
|------|-------------------|-------------------|
| **测试粒度** | 单个函数/模块 | 多模块集成 |
| **外部依赖** | 全部Mock | 部分Stub + 部分真实 |
| **HAL层** | 纯内存模拟，无IO | 可有文件IO、网络 |
| **被测代码** | 隔离测试单个组件 | 测试组件间交互 |
| **数据规模** | 小规模 (<1KB) | 中等规模 (<1MB) |
| **典型用例** | 函数逻辑、边界条件 | API流程、错误处理 |

### Mock vs Stub 定义

| 特性 | Mock（模拟） | Stub（桩） |
|------|-------------|-----------|
| **主要目的** | 验证交互行为 | 提供简化实现 |
| **行为逻辑** | 无真实逻辑，返回预设值 | 有简化的功能逻辑 |
| **调用验证** | 支持验证调用次数、参数 | 不验证调用 |
| **状态管理** | 无状态 | 可有内部状态 |
| **典型用途** | 单元测试隔离 | 集成测试替代 |

```c
/* Mock示例：无行为，可验证 */
UINT32 mock_reg_read(VOID *addr) {
    MOCK_RECORD_CALL(addr);           /* 记录调用 */
    return MOCK_RETURN_VALUE();       /* 返回预设值 */
}
/* 测试中：MOCK_VERIFY_CALL_COUNT(mock_reg_read, 3); */

/* Stub示例：有简化行为，不验证 */
ERRNO_T stub_dma_transfer(VOID *src, VOID *dst, UINT64 size) {
    (VOID)memcpy_s(dst, size, src, size);  /* 实际执行内存拷贝 */
    return ERR_OK;
}
```

### 验收标准

1. 测试用例代码不包含平台特定逻辑
2. 通过配置文件指定用例在哪些平台运行
3. 框架自动跳过不适用当前平台的用例

---

## REQ-SYS-004 验证流水线

---
id: REQ-SYS-004
title: 验证流水线阶段
priority: P1
status: draft
parent: REQ-SYS
depends:
  - REQ-SYS-003
---

### 描述

验证分阶段执行，逐级门控。

### 阶段定义

```
提交 → LinuxUT/ST → Functional → Performance → Prototype
         (分钟)       (小时)        (天)         (周)
```

| 阶段 | 触发 | 平台 | 目标 |
|------|------|------|------|
| LinuxUT/ST | 每次提交 | Host | 快速门控 |
| Functional | 每夜构建 | Simulator | 功能正确性 |
| Performance | 每周构建 | ESL | 性能指标 |
| Prototype | 里程碑 | FPGA/Chip | 硬件实测 |

### 验收标准

1. 前一阶段失败则阻塞后续阶段
2. 可配置跳过某些阶段
3. 阶段结果可追溯

---

## REQ-SYS-005 可扩展性

---
id: REQ-SYS-005
title: 框架可扩展性
priority: P1
status: draft
parent: REQ-SYS
---

### 描述

框架应易于扩展。

### 平台适配层接口

添加新平台需要实现以下函数：

```c
/* platform/<platform_name>/<platform>_adapter.c */

/* 必须实现 */
int platform_init(void);              /* 平台初始化 */
int platform_deinit(void);            /* 平台清理 */
int platform_run_test(test_case_t *tc); /* 执行单个测试 */

/* 可选实现 */
int platform_setup(void);             /* 测试前准备 */
int platform_teardown(void);          /* 测试后清理 */
void platform_log(int level, const char *fmt, ...); /* 日志输出 */
uint64_t platform_get_time_us(void);  /* 获取时间戳 */
```

### 参考模型标准接口

添加新算子参考模型需要实现：

```c
/* Python参考模型 (pymodel/ops/<op_name>.py) */
def forward(inputs: List[np.ndarray], params: dict) -> List[np.ndarray]:
    """算子前向计算"""
    pass

def generate_testcase(config: dict) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """生成测试用例 (inputs, expected_outputs)"""
    pass

/* C参考模型 (src/model/ref_<op_name>.c) */
int ref_<op_name>_f32(const float *input, float *output, const <op>_params_t *params);
int ref_<op_name>_f16(const uint16_t *input, uint16_t *output, const <op>_params_t *params);
```

### 报告生成接口

添加新报告格式需要实现：

```python
# tools/report/<format>_report.py
class <Format>ReportGenerator:
    def generate(self, results: dict, output_path: str) -> None:
        """生成报告文件"""
        pass
```

### 验收标准

1. 添加新测试用例：只需新增1个.c文件，无需修改框架代码
2. 添加新平台：实现平台适配层接口，新增文件 <= 5个，代码量 <= 1000行
3. 添加新算子参考模型：实现 forward() + generate_testcase()，新增文件 <= 2个
4. 添加新报告格式：继承 ReportGenerator 基类，新增文件 1个

---

## REQ-SYS-006 文档要求

---
id: REQ-SYS-006
title: 文档要求
priority: P1
status: draft
parent: REQ-SYS
---

### 描述

提供完整的使用和开发文档。

### 验收标准

1. README：快速开始指南
2. 架构文档：整体设计说明
3. 用例编写指南：如何编写测试用例
4. 平台适配指南：如何适配新平台
5. API文档：框架接口说明

---

## REQ-SYS-007 代码质量要求

---
id: REQ-SYS-007
title: 代码质量要求
priority: P1
status: draft
parent: REQ-SYS
---

### 描述

框架自身代码必须满足质量标准。

### Python代码质量

| 工具 | 要求 | 配置 |
|------|------|------|
| pylint | 评分 >= 9.0 | .pylintrc |
| pylint | 无 Error/Fatal 级别告警 | - |
| flake8 | 无告警 | .flake8 |
| black | 格式化通过 | pyproject.toml |
| mypy | 类型检查通过（可选） | mypy.ini |

### C代码质量

| 工具 | 要求 | 配置 |
|------|------|------|
| cppcheck | 无 error 级别告警 | - |
| clang-format | 格式化通过 | .clang-format |
| gcc -Wall -Werror | 编译无警告 | Makefile |

### 验收标准

1. CI流水线包含代码质量检查
2. 质量检查不通过阻塞合入
3. 提供pre-commit hook自动检查
4. 质量配置文件纳入版本控制

---

## REQ-SYS-008 测试覆盖率要求

---
id: REQ-SYS-008
title: 测试覆盖率要求
priority: P1
status: draft
parent: REQ-SYS
---

### 描述

框架自身代码的测试覆盖率要求。

### 覆盖率目标

| 代码类型 | 行覆盖率 | 分支覆盖率 | 工具 |
|----------|----------|------------|------|
| Python核心代码 | >= 80% | >= 70% | pytest-cov |
| Python工具代码 | >= 60% | >= 50% | pytest-cov |
| C测试框架 | >= 70% | >= 60% | gcov/lcov |

### 代码分类定义

```
Python核心代码（80%覆盖率要求）:
├── pymodel/ops/          # 参考模型算子
├── pymodel/quantize/     # 量化工具
├── pymodel/layers/       # 复合层
├── tools/runner/         # 测试运行器核心
└── deps/scripts/         # 依赖管理核心

Python工具代码（60%覆盖率要求）:
├── tools/report/         # 报告生成
├── tools/testmgmt/       # 用例管理Web
├── tools/archive/        # 归档工具
├── tools/data/           # 数据工具
├── cicd/                 # CI/CD脚本
└── scripts/              # 辅助脚本

C测试框架代码（70%覆盖率要求）:
├── src/framework/        # 测试框架核心
├── src/model/            # C参考模型
└── src/common/           # 通用工具
```

### 覆盖率排除

```ini
# .coveragerc
[run]
omit =
    */tests/*
    */test_*
    */__pycache__/*
    */migrations/*
```

### 验收标准

1. CI流水线生成覆盖率报告
2. 覆盖率不达标阻塞合入（可配置）
3. 覆盖率趋势可追踪
4. 支持覆盖率徽章展示

### 报告要求

```
Coverage Report:
-------------------------------------------------
Module                  Stmts   Miss  Cover
-------------------------------------------------
tools/runner            120     18    85%
tools/report            89      12    87%
pymodel/ops             156     31    80%
-------------------------------------------------
TOTAL                   365     61    83%
-------------------------------------------------

Branch Coverage: 72%
```

### 测量方法

| 语言 | 工具 | 命令 |
|------|------|------|
| Python | pytest-cov | `pytest --cov=tools --cov-branch` |
| C | gcov/lcov | `gcc --coverage && lcov --capture --branch-coverage` |

---

## REQ-SYS-009 版本兼容性

---
id: REQ-SYS-009
title: 框架版本兼容性
priority: P1
status: draft
parent: REQ-SYS
---

### 描述

框架升级时保持向后兼容，确保旧测试用例无需修改即可运行。

### 兼容性规则

| 版本变更类型 | 兼容性要求 |
|-------------|-----------|
| PATCH (x.x.Z) | 完全兼容，仅bug修复 |
| MINOR (x.Y.0) | 向后兼容，新增功能不影响旧用例 |
| MAJOR (X.0.0) | 可不兼容，但提供迁移指南 |

### 版本号定义

```c
/* include/framework/version.h */
#define FWK_VERSION_MAJOR  1
#define FWK_VERSION_MINOR  0
#define FWK_VERSION_PATCH  0
#define FWK_VERSION_STRING "1.0.0"

/* 运行时版本检查 */
SEC_DDR_TEXT ERRNO_T FWK_CheckVersion(INT32 requiredMajor, INT32 requiredMinor);
```

### 验收标准

1. 版本号遵循语义化版本规范(SemVer)
2. MINOR版本升级后，旧测试用例100%通过
3. MAJOR版本升级提供迁移脚本或文档
4. 提供版本兼容性检查API

---

## REQ-SYS-010 基础类型定义

---
id: REQ-SYS-010
title: 基础类型定义
priority: P0
status: draft
parent: REQ-SYS
---

### 描述

定义框架使用的基础类型别名，确保跨平台一致性。

### 类型定义

```c
/* include/common/types.h */

/* 整数类型 */
typedef signed char        INT8;
typedef unsigned char      UINT8;
typedef signed short       INT16;
typedef unsigned short     UINT16;
typedef signed int         INT32;
typedef unsigned int       UINT32;
typedef signed long long   INT64;
typedef unsigned long long UINT64;

/* 浮点类型 */
typedef float              FLOAT32;
typedef double             FLOAT64;

/* 其他类型 */
typedef char               CHAR;
typedef void               VOID;
typedef UINT8              BOOL;

/* 错误码类型 */
typedef INT32              ERRNO_T;

/* 布尔值 */
#define TRUE   1
#define FALSE  0

/* 空指针 */
#ifndef NULL
#define NULL   ((VOID *)0)
#endif
```

### 错误码分段

```c
/* include/common/errno.h */

/* 成功 */
#define ERR_OK           0

/* 错误码分段（每段4096个） */
#define ERR_FWK_BASE     0x1000  /* 0x1000-0x1FFF: 测试框架错误 */
#define ERR_HAL_BASE     0x2000  /* 0x2000-0x2FFF: HAL层错误 */
#define ERR_PLAT_BASE    0x3000  /* 0x3000-0x3FFF: 平台适配错误 */
#define ERR_MDL_BASE     0x4000  /* 0x4000-0x4FFF: 参考模型错误 */
#define ERR_TMT_BASE     0x5000  /* 0x5000-0x5FFF: 用例管理错误 */
#define ERR_RST_BASE     0x6000  /* 0x6000-0x6FFF: 结果管理错误 */
#define ERR_DEP_BASE     0x7000  /* 0x7000-0x7FFF: 依赖管理错误 */
#define ERR_CIC_BASE     0x8000  /* 0x8000-0x8FFF: CI/CD错误 */
#define ERR_QCK_BASE     0x9000  /* 0x9000-0x9FFF: 代码质量错误 */
#define ERR_EFF_BASE     0xA000  /* 0xA000-0xAFFF: 效率工具错误 */
#define ERR_TLS_BASE     0xB000  /* 0xB000-0xBFFF: 辅助工具错误 */

/* 框架通用错误 */
#define ERR_FWK_NULL_PTR      (ERR_FWK_BASE + 0x001)  /* 空指针 */
#define ERR_FWK_INVALID_PARAM (ERR_FWK_BASE + 0x002)  /* 无效参数 */
#define ERR_FWK_TIMEOUT       (ERR_FWK_BASE + 0x003)  /* 超时 */
#define ERR_FWK_NO_MEMORY     (ERR_FWK_BASE + 0x004)  /* 内存不足 */
#define ERR_FWK_NOT_SUPPORT   (ERR_FWK_BASE + 0x005)  /* 不支持 */
```

### 验收标准

1. 所有框架代码使用统一类型别名
2. 禁止直接使用 int/long/char 等原生类型
3. ERRNO_T 统一为 INT32
4. 错误码值全局唯一，不冲突

---

## REQ-SYS-011 头文件组织

---
id: REQ-SYS-011
title: 头文件组织规范
priority: P1
status: draft
parent: REQ-SYS
---

### 描述

定义框架头文件的组织结构和包含规则。

### 目录结构

```
include/
├── common/               # 通用定义
│   ├── types.h           # 基础类型定义
│   ├── errno.h           # 错误码定义
│   └── macros.h          # 通用宏定义
│
├── framework/            # 测试框架公开API
│   ├── test_case.h       # 用例定义宏
│   ├── assert.h          # 断言宏
│   ├── runner.h          # 运行器接口
│   ├── mock.h            # Mock框架
│   ├── param.h           # 参数化测试
│   ├── fixture.h         # Fixture管理
│   ├── log.h             # 日志接口
│   └── version.h         # 版本信息
│
├── hal/                  # HAL层接口
│   └── hal_ops.h         # HAL统一接口
│
├── model/                # 参考模型接口
│   └── compare.h         # 比较工具接口
│
└── internal/             # 框架内部头文件（不对外）
    ├── registry.h        # 用例注册内部实现
    └── platform.h        # 平台内部接口
```

### 包含规则

```c
/* 公开头文件：使用 <> 包含 */
#include <framework/test_case.h>
#include <framework/assert.h>
#include <hal/hal_ops.h>

/* 内部头文件：使用 "" 包含 */
#include "internal/registry.h"

/* 禁止循环包含，使用前向声明 */
```

### 验收标准

1. 公开API和内部实现分离
2. 每个头文件有 include guard
3. 头文件自包含（可独立编译）
4. 无循环依赖

---

## REQ-SYS-012 跨模块接口契约

---
id: REQ-SYS-012
title: 跨模块接口契约
priority: P1
status: draft
parent: REQ-SYS
---

### 描述

定义跨模块通信的标准接口，使模块依赖接口而非实现，降低耦合度。

### 设计原则

1. **接口隔离**：模块只暴露必要的接口
2. **依赖倒置**：上层模块依赖抽象接口，不依赖具体实现
3. **单一职责**：每个接口只负责一类功能

### 公共接口定义

#### 1. 测试结果接口（供RST、CIC、TLS使用）

```c
/* include/common/result_intf.h */

/* 测试结果枚举（定义在FWK，但作为公共接口） */
typedef enum TestResultEnum {
    TEST_PASS    = 0,
    TEST_FAIL    = 1,
    TEST_SKIP    = 2,
    TEST_TIMEOUT = 3,
    TEST_ERROR   = 4,
    TEST_CRASH   = 5,
} TestResultEnum;

/* 测试结果摘要（JSON格式的C结构映射） */
typedef struct TestSummaryStru {
    INT32 total;
    INT32 passed;
    INT32 failed;
    INT32 skipped;
    UINT32 durationMs;
} TestSummaryStru;
```

#### 2. 用例元数据接口（供TMT、RST、CIC使用）

```c
/* include/common/testcase_intf.h */

/* 用例基本信息（不含执行逻辑） */
typedef struct TestCaseInfoStru {
    const CHAR *name;        /* 用例名 */
    const CHAR *suite;       /* 套件名 */
    const CHAR *tags;        /* 标签列表，逗号分隔 */
    const CHAR *platforms;   /* 支持平台，逗号分隔 */
    UINT32 timeoutMs;        /* 超时时间 */
} TestCaseInfoStru;

/* 用例列表迭代器接口 */
typedef struct TestCaseIteratorStru TestCaseIteratorStru;
SEC_DDR_TEXT TestCaseIteratorStru* TCINFO_CreateIterator(const CHAR *filter);
SEC_DDR_TEXT const TestCaseInfoStru* TCINFO_Next(TestCaseIteratorStru *iter);
SEC_DDR_TEXT VOID TCINFO_DestroyIterator(TestCaseIteratorStru *iter);
```

#### 3. 数据格式接口（供MDL、TLS使用）

```c
/* include/common/data_intf.h */

/* Golden数据头（简化版，供读取使用） */
typedef struct GoldenInfoStru {
    UINT32 version;
    UINT32 dtype;
    UINT32 ndim;
    UINT32 shape[8];
    UINT64 dataSize;
} GoldenInfoStru;

/* 数据读取接口 */
SEC_DDR_TEXT ERRNO_T DATA_ReadGoldenInfo(const CHAR *path, GoldenInfoStru *info);
SEC_DDR_TEXT ERRNO_T DATA_ReadGoldenData(const CHAR *path, VOID *buf, UINT64 bufSize);
```

### 模块依赖矩阵

| 接口 | 定义方 | 使用方 |
|------|--------|--------|
| TestResultEnum | FWK | RST, CIC, TLS, EFF |
| TestSummaryStru | FWK | RST, CIC, TLS |
| TestCaseInfoStru | TMT | RST, CIC |
| GoldenInfoStru | MDL | TLS |
| 错误码分段 | SYS | 所有模块 |

### 依赖规则

```
模块依赖接口，不依赖实现：
┌─────────────────────────────────────────────┐
│  依赖方式                                    │
├─────────────────────────────────────────────┤
│  ✓ #include <common/result_intf.h>          │
│  ✓ 使用 TestResultEnum 枚举                 │
│  ✓ 调用 TCINFO_CreateIterator() 接口        │
├─────────────────────────────────────────────┤
│  ✗ #include <framework/internal/runner.h>   │
│  ✗ 直接访问 g_testCaseRegistry 全局变量     │
│  ✗ 依赖 FWK 内部数据结构                    │
└─────────────────────────────────────────────┘
```

### 验收标准

1. 跨模块通信必须通过公共接口
2. 公共接口定义在 include/common/ 目录
3. 接口变更需评审，保持向后兼容
4. 禁止模块直接访问其他模块的内部实现

---

## REQ-SYS-013 领域事件机制

---
id: REQ-SYS-013
title: 领域事件机制
priority: P1
status: draft
parent: REQ-SYS
---

### 描述

定义跨限界上下文的事件驱动通信机制，实现模块解耦。

### 设计原则

1. **发布-订阅模式**：事件生产者与消费者解耦
2. **事件不可变**：事件一旦发布不可修改
3. **最终一致性**：允许异步处理，保证最终一致

### 事件基础结构

```c
/* include/common/event.h */

/* 事件类型枚举 */
typedef enum EventTypeEnum {
    EVENT_TEST_STARTED      = 0x0001,
    EVENT_TEST_COMPLETED    = 0x0002,
    EVENT_PLATFORM_READY    = 0x0003,
    EVENT_GOLDEN_UPDATED    = 0x0004,
    EVENT_REPORT_GENERATED  = 0x0005,
} EventTypeEnum;
typedef UINT16 EVENT_TYPE_ENUM_UINT16;

/* 事件头（所有事件共有） */
typedef struct EventHeaderStru {
    EVENT_TYPE_ENUM_UINT16 type;    /* 事件类型 */
    UINT16 version;                  /* 事件版本 */
    UINT32 timestamp;                /* 时间戳(秒) */
    CHAR source[32];                 /* 来源上下文 */
    CHAR eventId[64];                /* 事件唯一ID */
} EventHeaderStru;

/* 事件处理器函数指针 */
typedef ERRNO_T (*EventHandlerFunc)(const EventHeaderStru *header, const VOID *payload);
```

### 核心事件定义

```c
/* 测试完成事件 */
typedef struct TestCompletedEventStru {
    EventHeaderStru header;
    CHAR executionId[64];
    CHAR testName[256];
    TEST_RESULT_ENUM_UINT8 result;
    UINT32 durationMs;
    CHAR errorInfo[512];
} TestCompletedEventStru;

/* 执行完成事件 */
typedef struct ExecutionCompletedEventStru {
    EventHeaderStru header;
    CHAR executionId[64];
    TestSummaryStru summary;
} ExecutionCompletedEventStru;

/* 平台就绪事件 */
typedef struct PlatformReadyEventStru {
    EventHeaderStru header;
    CHAR platform[32];
    PlatformCapsStru caps;
} PlatformReadyEventStru;
```

### 事件总线接口

```c
/* include/common/event_bus.h */

/* 初始化事件总线 */
SEC_DDR_TEXT ERRNO_T EVT_Init(VOID);

/* 发布事件 */
SEC_DDR_TEXT ERRNO_T EVT_Publish(const EventHeaderStru *event);

/* 订阅事件 */
SEC_DDR_TEXT ERRNO_T EVT_Subscribe(EVENT_TYPE_ENUM_UINT16 type, EventHandlerFunc handler);

/* 取消订阅 */
SEC_DDR_TEXT ERRNO_T EVT_Unsubscribe(EVENT_TYPE_ENUM_UINT16 type, EventHandlerFunc handler);

/* 销毁事件总线 */
SEC_DDR_TEXT ERRNO_T EVT_Deinit(VOID);
```

### 使用示例

```c
/* FWK发布事件 */
static ERRNO_T PublishTestCompleted(const CHAR *testName, TEST_RESULT_ENUM_UINT8 result)
{
    TestCompletedEventStru event = {0};
    event.header.type = EVENT_TEST_COMPLETED;
    event.header.version = 1;
    event.header.timestamp = (UINT32)time(NULL);
    (VOID)strcpy_s(event.header.source, sizeof(event.header.source), "FWK");
    (VOID)strcpy_s(event.testName, sizeof(event.testName), testName);
    event.result = result;

    return EVT_Publish(&event.header);
}

/* RST订阅事件 */
static ERRNO_T OnTestCompleted(const EventHeaderStru *header, const VOID *payload)
{
    const TestCompletedEventStru *event = (const TestCompletedEventStru *)header;
    /* 通过防腐层转换后处理 */
    return RST_RecordResult(event->testName, event->result);
}

ERRNO_T RST_Init(VOID)
{
    return EVT_Subscribe(EVENT_TEST_COMPLETED, OnTestCompleted);
}
```

### 同步与异步模式

| 模式 | 场景 | 实现 |
|------|------|------|
| 同步 | LinuxUT/ST | 直接调用handler |
| 异步 | Simulator/FPGA | 事件队列 + 工作线程 |

```c
/* 配置事件处理模式 */
typedef enum EventModeEnum {
    EVENT_MODE_SYNC  = 0,   /* 同步：发布时立即调用所有handler */
    EVENT_MODE_ASYNC = 1,   /* 异步：放入队列，由工作线程处理 */
} EventModeEnum;

SEC_DDR_TEXT ERRNO_T EVT_SetMode(EventModeEnum mode);
```

### 验收标准

1. 核心域通过事件与支撑域通信
2. 事件发布失败不影响主流程（可配置）
3. 支持同步和异步两种模式
4. 事件可序列化用于分布式场景
