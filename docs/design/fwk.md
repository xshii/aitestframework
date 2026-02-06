# FWK 测试框架模块设计

---
module: FWK
version: 1.0
date: 2026-02-04
status: draft
requirements: REQ-FWK-001~005, REQ-FWK-019
domain: Core
aggregate: TestCase
---

## 1. 模块概述

### 1.1 职责

测试框架核心功能：
- 测试用例自动注册
- 断言宏集合
- 测试运行器
- 结果收集与输出
- 信号处理（崩溃/超时）

### 1.2 DDD定位

- **限界上下文**：测试执行上下文
- **聚合根**：TestCase
- **发布事件**：TestCaseCompleted, TestExecutionCompleted

---

## 2. 文件结构

```
include/framework/
├── test_case.h    # 用例注册宏
├── assert.h       # 断言宏
├── result.h       # 结果类型
└── runner.h       # 运行器接口

src/framework/
├── runner.c       # 运行器实现
├── output.c       # 输出格式化
└── signal.c       # 信号处理
```

---

## 3. 聚合设计

### 3.1 TestCase聚合

```
TestCase (聚合根)
├── suite: STRING          # 测试套件名
├── name: STRING           # 用例名
├── func: TestFunc         # 测试函数指针
├── timeoutMs: UINT32      # 超时时间
├── tags: STRING           # 标签
└── result: TestResult     # 执行结果（值对象）
    ├── status: TestResultEnum
    ├── durationMs: UINT32
    ├── failFile: STRING
    └── failLine: INT32
```

### 3.2 数据结构

```c
/* 测试函数原型 */
typedef ERRNO_T (*TestFunc)(VOID);

/* 测试用例结构体 */
typedef struct TestCaseStru {
    const CHAR *suite;
    const CHAR *name;
    TestFunc    func;
    UINT32      timeoutMs;
    const CHAR *tags;
} TestCaseStru;

/* 测试结果枚举 */
typedef enum TestResultEnum {
    TEST_PASS    = 0,
    TEST_FAIL    = 1,
    TEST_SKIP    = 2,
    TEST_TIMEOUT = 3,
    TEST_ERROR   = 4,
    TEST_CRASH   = 5,
} TestResultEnum;

/* 单个用例结果 */
typedef struct TestResultStru {
    const CHAR     *suite;
    const CHAR     *name;
    TestResultEnum  status;
    UINT32          durationMs;
    const CHAR     *failFile;
    INT32           failLine;
    const CHAR     *failExpr;
} TestResultStru;

/* 执行汇总 */
typedef struct TestSummaryStru {
    UINT32 total;
    UINT32 passed;
    UINT32 failed;
    UINT32 skipped;
    UINT32 timeout;
    UINT32 crashed;
    UINT32 durationMs;
} TestSummaryStru;
```

---

## 4. 用例注册机制

### 4.1 Linker Section方式

```c
#define TEST_CASE(suite, name) \
    static ERRNO_T test_##suite##_##name(VOID); \
    static TestCaseStru __attribute__((section(".testcases"), used)) \
    __test_##suite##_##name = { \
        .suite = #suite, \
        .name = #name, \
        .func = test_##suite##_##name, \
        .timeoutMs = 5000, \
        .tags = NULL \
    }; \
    static ERRNO_T test_##suite##_##name(VOID)
```

### 4.2 变体宏

```c
/* 带标签 */
#define TEST_CASE_TAGS(suite, name, tagStr)

/* 带超时 */
#define TEST_CASE_TIMEOUT(suite, name, timeoutMs)
```

### 4.3 Linker Script

```ld
SECTIONS {
    .testcases : {
        __start_testcases = .;
        KEEP(*(.testcases))
        __stop_testcases = .;
    }
}
```

### 4.4 用例发现

```c
extern TestCaseStru __start_testcases;
extern TestCaseStru __stop_testcases;

UINT32 Runner_GetTestCount(VOID)
{
    return &__stop_testcases - &__start_testcases;
}
```

---

## 5. 断言宏设计

### 5.1 测试上下文

```c
typedef struct TestContextStru {
    const CHAR *failFile;
    INT32       failLine;
    const CHAR *failExpr;
} TestContextStru;

extern TestContextStru g_testCtx;
```

### 5.2 基础断言

```c
#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        g_testCtx.failFile = __FILE__; \
        g_testCtx.failLine = __LINE__; \
        g_testCtx.failExpr = #cond; \
        return AITF_ERR_ASSERT_FAIL; \
    } \
} while (0)
```

### 5.3 断言宏列表

| 宏 | 用途 |
|----|------|
| TEST_ASSERT(cond) | 条件断言 |
| TEST_ASSERT_EQ(exp, act) | 相等 |
| TEST_ASSERT_NE(exp, act) | 不等 |
| TEST_ASSERT_GT(a, b) | 大于 |
| TEST_ASSERT_GE(a, b) | 大于等于 |
| TEST_ASSERT_LT(a, b) | 小于 |
| TEST_ASSERT_LE(a, b) | 小于等于 |
| TEST_ASSERT_NEAR(exp, act, eps) | 浮点近似 |
| TEST_ASSERT_NOT_NULL(ptr) | 非空 |
| TEST_ASSERT_NULL(ptr) | 为空 |
| TEST_ASSERT_TRUE(cond) | 布尔真 |
| TEST_ASSERT_FALSE(cond) | 布尔假 |
| TEST_ASSERT_STR_EQ(exp, act) | 字符串相等 |
| TEST_ASSERT_MEM_EQ(exp, act, sz) | 内存相等 |
| TEST_ASSERT_OK(expr) | 错误码成功 |
| TEST_SKIP(reason) | 跳过测试 |
| TEST_FAIL(reason) | 测试失败 |

---

## 6. 运行器设计

### 6.1 接口

```c
typedef struct RunnerConfigStru {
    const CHAR *filter;
    const CHAR *outputPath;
    BOOL        jsonOutput;
    BOOL        verbose;
    UINT32      defaultTimeoutMs;
} RunnerConfigStru;

ERRNO_T Runner_Init(const RunnerConfigStru *config);
ERRNO_T Runner_RunAll(TestSummaryStru *summary);
UINT32  Runner_GetTestCount(VOID);
VOID    Runner_ListTests(VOID);
VOID    Runner_Cleanup(VOID);
```

### 6.2 执行流程

```
Runner_Init()
    │
    ├── 解析配置
    ├── 安装信号处理
    └── 初始化结果收集器

Runner_RunAll()
    │
    ├── 遍历所有TestCase
    │   │
    │   ├── 过滤检查
    │   ├── HAL_Init()
    │   ├── 设置超时alarm
    │   ├── sigsetjmp (崩溃恢复点)
    │   ├── 执行 TestCase.func()
    │   ├── 取消alarm
    │   ├── HAL_Deinit()
    │   └── 记录结果
    │
    └── 输出汇总
```

### 6.3 过滤机制

```c
/* 通配符匹配 */
BOOL Filter_Match(const CHAR *pattern,
                  const CHAR *suite,
                  const CHAR *name);

/* 支持模式 */
// "matmul*"      → 匹配 matmul.basic, matmul.large
// "sanity.*"     → 匹配 sanity套件下所有
// "*basic*"      → 匹配包含basic的所有
```

---

## 7. 信号处理设计

### 7.1 信号映射

| 信号 | 处理 | 结果 |
|------|------|------|
| SIGSEGV | siglongjmp恢复 | TEST_CRASH |
| SIGBUS | siglongjmp恢复 | TEST_CRASH |
| SIGFPE | siglongjmp恢复 | TEST_CRASH |
| SIGALRM | siglongjmp恢复 | TEST_TIMEOUT |
| SIGINT | 设置中断标志 | 优雅退出 |

### 7.2 实现

```c
typedef struct SignalContextStru {
    sigjmp_buf  jumpBuf;
    INT32       caughtSignal;
    BOOL        inTest;
    BOOL        interrupted;
} SignalContextStru;

static SignalContextStru g_sigCtx;

static void SignalHandler(int sig)
{
    if (g_sigCtx.inTest) {
        g_sigCtx.caughtSignal = sig;
        siglongjmp(g_sigCtx.jumpBuf, 1);
    }
}

/* 执行用例 */
g_sigCtx.inTest = TRUE;
if (sigsetjmp(g_sigCtx.jumpBuf, 1) == 0) {
    alarm(timeoutSec);
    result = testCase->func();
    alarm(0);
} else {
    /* 从信号恢复 */
    result = (g_sigCtx.caughtSignal == SIGALRM)
             ? TEST_TIMEOUT : TEST_CRASH;
}
g_sigCtx.inTest = FALSE;
```

---

## 8. 输出格式

### 8.1 控制台输出

```
[PASS] sanity.hello (1ms)
[FAIL] matmul.large (5012ms)
       at test_matmul_large.c:45
       assertion failed: cosine_sim >= 0.999
[SKIP] perf.latency
       reason: requires real hardware

Summary: 145 passed, 3 failed, 2 skipped (5230ms)
```

### 8.2 JSON输出

```json
{
  "execution_id": "20260204_103000",
  "platform": "linux_ut",
  "summary": {
    "total": 150,
    "passed": 145,
    "failed": 3,
    "skipped": 2,
    "duration_ms": 5230
  },
  "tests": [
    {
      "suite": "sanity",
      "name": "hello",
      "status": "PASS",
      "duration_ms": 1
    }
  ]
}
```

---

## 9. 领域事件

### 9.1 发布的事件

```yaml
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
```

---

## 10. 使用示例

```c
#include "framework/test_case.h"
#include "framework/assert.h"

TEST_CASE(sanity, hello)
{
    INT32 a = 1;
    INT32 b = 1;
    TEST_ASSERT_EQ(a, b);
    return AITF_OK;
}

TEST_CASE_TAGS(matmul, basic, "smoke,matmul")
{
    FLOAT32 result[4];
    FLOAT32 expected[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    /* 执行矩阵乘法 */

    TEST_ASSERT_NEAR(expected[0], result[0], 1e-5);
    return AITF_OK;
}
```

---

## 11. 需求追溯

| 需求ID | 需求标题 | 设计章节 |
|--------|----------|----------|
| REQ-FWK-001 | 测试用例自动注册 | 4 |
| REQ-FWK-002 | 断言宏集合 | 5 |
| REQ-FWK-003 | 测试结果类型 | 3 |
| REQ-FWK-004 | 测试运行器 | 6 |
| REQ-FWK-005 | 测试输出格式 | 8 |
| REQ-FWK-019 | 信号处理机制 | 7 |
