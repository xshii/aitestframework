# TMT 用例管理模块设计

---
module: TMT
version: 1.0
date: 2026-02-04
status: draft
requirements: REQ-TMT-001, REQ-TMT-007
domain: Supporting
aggregate: TestSuite
---

## 1. 模块概述

### 1.1 职责

管理测试用例的组织和发现：
- 用例目录结构规范
- 自动用例发现机制
- 用例过滤和筛选
- 平台映射（P1）
- 标签系统（P1）

### 1.2 DDD定位

- **限界上下文**：用例管理上下文
- **聚合根**：TestSuite
- **订阅事件**：TestRegistered（可选）

---

## 2. 目录结构

### 2.1 规范

```
tests/
├── unit/                       # 单元测试
│   ├── framework/              # 框架自测
│   │   ├── test_assert.c
│   │   └── test_runner.c
│   ├── hal/                    # HAL测试
│   │   └── test_mock.c
│   └── model/                  # 模型测试
│       └── test_compare.c
│
├── functional/                 # 功能测试
│   ├── sanity/                 # 冒烟测试
│   │   ├── test_sanity.c
│   │   └── test_basic_op.c
│   ├── matmul/                 # 矩阵乘法
│   │   ├── test_matmul_basic.c
│   │   ├── test_matmul_shapes.c
│   │   └── test_matmul_precision.c
│   ├── conv/                   # 卷积
│   │   └── test_conv2d.c
│   └── activation/             # 激活函数
│       ├── test_relu.c
│       └── test_gelu.c
│
├── performance/                # 性能测试
│   ├── test_matmul_perf.c
│   └── test_latency.c
│
├── stress/                     # 压力测试
│   └── test_long_run.c
│
└── e2e/                        # 端到端测试
    └── test_resnet_layer.c
```

### 2.2 命名规范

| 项目 | 规范 | 示例 |
|------|------|------|
| 目录 | 小写下划线 | `functional/matmul/` |
| 文件 | `test_<name>.c` | `test_matmul_basic.c` |
| 套件名 | 小写 | `matmul`, `sanity` |
| 用例名 | 小写下划线 | `basic_2x2`, `large_matrix` |

---

## 3. 聚合设计

### 3.1 TestSuite聚合

```
TestSuite (聚合根)
├── name: STRING               # 套件名
├── path: STRING               # 源文件路径
├── testCases: TestCase[]      # 用例列表
└── config: TestConfig         # 配置（值对象）
    ├── defaultTimeout: UINT32
    └── tags: STRING[]
```

---

## 4. 用例发现机制

### 4.1 编译时注册

所有使用 `TEST_CASE` 宏的用例在编译时自动注册到 `.testcases` section。

```c
/* 用例定义 */
TEST_CASE(sanity, hello)
{
    TEST_ASSERT_TRUE(1);
    return AITF_OK;
}

/* 编译后放入 .testcases section */
```

### 4.2 运行时发现

```c
/* linker symbols */
extern TestCaseStru __start_testcases;
extern TestCaseStru __stop_testcases;

/* 遍历所有用例 */
static ERRNO_T DiscoverTests(VOID)
{
    TestCaseStru *start = &__start_testcases;
    TestCaseStru *stop = &__stop_testcases;

    g_testCount = stop - start;

    for (TestCaseStru *tc = start; tc < stop; tc++) {
        /* 注册到运行器 */
        RegisterTestCase(tc);
    }

    return AITF_OK;
}
```

### 4.3 Linker Script

```ld
/* 确保section不被优化掉 */
SECTIONS {
    .testcases : {
        __start_testcases = .;
        KEEP(*(.testcases))
        __stop_testcases = .;
    }
}
```

---

## 5. 用例过滤

### 5.1 过滤模式

| 模式 | 含义 | 示例 |
|------|------|------|
| `*` | 匹配任意字符 | `matmul*` |
| `?` | 匹配单个字符 | `test_?` |
| `.` | 套件/用例分隔 | `sanity.*` |

### 5.2 过滤示例

| 过滤器 | 匹配 |
|--------|------|
| `matmul*` | `matmul.basic`, `matmul.large` |
| `sanity.*` | `sanity.hello`, `sanity.basic` |
| `*basic*` | `matmul.basic`, `sanity.basic_op` |
| `functional.*` | 所有functional套件用例 |

### 5.3 实现

```c
BOOL Filter_Match(const CHAR *pattern,
                  const CHAR *suite,
                  const CHAR *name)
{
    CHAR fullName[256];
    snprintf(fullName, sizeof(fullName), "%s.%s", suite, name);
    return WildcardMatch(pattern, fullName);
}

static BOOL WildcardMatch(const CHAR *pattern, const CHAR *str)
{
    while (*pattern && *str) {
        if (*pattern == '*') {
            pattern++;
            if (*pattern == '\0') return TRUE;
            while (*str) {
                if (WildcardMatch(pattern, str)) return TRUE;
                str++;
            }
            return FALSE;
        } else if (*pattern == '?' || *pattern == *str) {
            pattern++;
            str++;
        } else {
            return FALSE;
        }
    }
    while (*pattern == '*') pattern++;
    return (*pattern == '\0' && *str == '\0');
}
```

---

## 6. 平台映射（P1）

### 6.1 配置格式

```yaml
# tests/testcfg/platform_mapping.yaml
default_platforms: [linux_ut, linux_st, simulator]

overrides:
  # 性能测试只在特定平台运行
  "performance/*":
    platforms: [esl, fpga, chip]
    skip_reason: "需要真实时序"

  # 功耗测试只在芯片上运行
  "performance/test_power.c":
    platforms: [chip]
    skip_reason: "需要真实功耗测量"

  # 长时间运行测试
  "stress/*":
    platforms: [simulator, fpga, chip]
    timeout_ms: 3600000  # 1小时
```

### 6.2 平台检查

```c
BOOL IsPlatformSupported(const CHAR *suite, const CHAR *name)
{
    /* 从配置中获取当前平台支持的用例列表 */
    /* 返回当前用例是否支持当前平台 */
}
```

---

## 7. 标签系统（P1）

### 7.1 标签定义

```c
TEST_CASE_TAGS(matmul, basic_2x2, "smoke,matmul,fast")
{
    /* ... */
}
```

### 7.2 标签过滤

```bash
# 运行所有smoke标签的用例
./test_runner --tags smoke

# 运行matmul且非slow的用例
./test_runner --tags "matmul,!slow"
```

### 7.3 标签解析

```c
BOOL Tags_Match(const CHAR *caseTags, const CHAR *filterTags)
{
    /* 解析filterTags中的包含/排除规则 */
    /* 检查caseTags是否满足 */
}
```

---

## 8. 用例列表输出

### 8.1 命令行

```bash
./test_runner --list
./test_runner --list --filter "matmul*"
```

### 8.2 输出格式

```
Available tests (5):
  sanity.hello
  sanity.basic_op
  matmul.basic_2x2
  matmul.large_matrix
  conv.basic
```

### 8.3 JSON格式

```bash
./test_runner --list --json
```

```json
{
  "count": 5,
  "tests": [
    {"suite": "sanity", "name": "hello", "tags": "smoke"},
    {"suite": "sanity", "name": "basic_op", "tags": "smoke"},
    {"suite": "matmul", "name": "basic_2x2", "tags": "smoke,matmul"},
    {"suite": "matmul", "name": "large_matrix", "tags": "matmul"},
    {"suite": "conv", "name": "basic", "tags": "smoke,conv"}
  ]
}
```

---

## 9. 使用示例

### 9.1 编写测试用例

```c
/* tests/functional/sanity/test_sanity.c */

#include "framework/test_case.h"
#include "framework/assert.h"

TEST_CASE(sanity, hello)
{
    /* 最简单的测试 */
    TEST_ASSERT_TRUE(1);
    return AITF_OK;
}

TEST_CASE_TAGS(sanity, basic_op, "smoke")
{
    INT32 a = 2, b = 3;
    TEST_ASSERT_EQ(5, a + b);
    return AITF_OK;
}
```

### 9.2 运行测试

```bash
# 运行所有测试
./build/bin/test_runner

# 只运行sanity套件
./build/bin/test_runner --filter "sanity.*"

# 列出matmul相关测试
./build/bin/test_runner --list --filter "matmul*"
```

---

## 10. 需求追溯

| 需求ID | 需求标题 | 设计章节 |
|--------|----------|----------|
| REQ-TMT-001 | 用例目录结构 | 2 |
| REQ-TMT-007 | 自动用例发现 | 4 |
| REQ-TMT-002 | 用例平台映射 | 6 (P1) |
| REQ-TMT-003 | 用例标签系统 | 7 (P1) |
