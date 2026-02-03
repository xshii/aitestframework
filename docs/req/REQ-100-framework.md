# REQ-FWK 测试框架需求

---
id: REQ-FWK
title: C测试框架需求
priority: P0
status: draft
parent: REQ-SYS
---

## 概述

纯C实现的测试框架，提供测试注册、断言、运行、报告等核心功能。

---

## REQ-FWK-001 测试用例注册

---
id: REQ-FWK-001
title: 测试用例自动注册
priority: P0
status: draft
parent: REQ-FWK
---

### 描述

测试用例通过宏自动注册，无需手动维护用例列表。

### 接口设计

```c
/* 基本用例定义 */
TEST_CASE(suite_name, test_name)
{
    /* 测试代码 */
    return TEST_PASS;
}

/* 带属性的用例定义 */
TEST_CASE_EX(suite_name, test_name,
    .timeout_ms = 10000,
    .tags = "smoke,matmul",
    .platforms = "simulator,fpga,chip"
)
{
    /* 测试代码 */
    return TEST_PASS;
}
```

### 验收标准

1. 使用linker section实现自动注册
2. 编译时自动收集所有TEST_CASE
3. 支持指定超时、标签、平台等属性
4. 无需修改任何列表文件即可添加新用例

---

## REQ-FWK-002 断言宏

---
id: REQ-FWK-002
title: 断言宏集合
priority: P0
status: draft
parent: REQ-FWK
---

### 描述

提供丰富的断言宏，失败时记录详细信息。

### 接口设计

```c
/* 基本断言 */
TEST_ASSERT(condition)
TEST_ASSERT_MSG(condition, fmt, ...)

/* 比较断言 */
TEST_ASSERT_EQ(expected, actual)      /* == */
TEST_ASSERT_NE(a, b)                   /* != */
TEST_ASSERT_LT(a, b)                   /* < */
TEST_ASSERT_LE(a, b)                   /* <= */
TEST_ASSERT_GT(a, b)                   /* > */
TEST_ASSERT_GE(a, b)                   /* >= */

/* 浮点断言 */
TEST_ASSERT_FLOAT_EQ(expected, actual, epsilon)
TEST_ASSERT_DOUBLE_EQ(expected, actual, epsilon)

/* 数组/张量断言 */
TEST_ASSERT_ARRAY_EQ(expected, actual, count)
TEST_ASSERT_TENSOR_NEAR(expected, actual, count, epsilon)

/* 字符串断言 */
TEST_ASSERT_STR_EQ(expected, actual)

/* 指针断言 */
TEST_ASSERT_NULL(ptr)
TEST_ASSERT_NOT_NULL(ptr)
```

### 验收标准

1. 断言失败时记录文件名、行号、表达式
2. 断言失败时记录期望值和实际值
3. 断言失败后立即返回TEST_FAIL
4. 支持自定义失败消息

---

## REQ-FWK-003 测试结果类型

---
id: REQ-FWK-003
title: 测试结果类型
priority: P0
status: draft
parent: REQ-FWK
---

### 描述

定义测试用例的执行结果类型。

### 接口设计

```c
typedef enum TestResultEnum {
    TEST_PASS    = 0,   /* 通过 */
    TEST_FAIL    = 1,   /* 失败 */
    TEST_SKIP    = 2,   /* 跳过 */
    TEST_TIMEOUT = 3,   /* 超时 */
    TEST_ERROR   = 4,   /* 错误（框架异常） */
} TestResultEnum;
typedef UINT8 TEST_RESULT_ENUM_UINT8;
```

### 验收标准

1. 每个用例返回明确的结果类型
2. SKIP用于平台不支持等场景
3. TIMEOUT由框架自动判定
4. ERROR用于框架本身异常

---

## REQ-FWK-004 测试运行器

---
id: REQ-FWK-004
title: 测试运行器
priority: P0
status: draft
parent: REQ-FWK
depends:
  - REQ-FWK-001
  - REQ-FWK-003
---

### 描述

运行测试用例并收集结果。

### 功能要求

1. **用例发现**：扫描注册的所有用例
2. **用例筛选**：按名称、标签、套件筛选
3. **用例执行**：调用用例函数
4. **超时控制**：超时自动终止
5. **结果收集**：统计pass/fail/skip/timeout

### 接口设计

```c
typedef struct TestRunnerConfigStru {
    const CHAR *filter;       /* 名称过滤 (支持通配符) */
    const CHAR *tags;         /* 标签过滤 */
    const CHAR *suite;        /* 套件过滤 */
    UINT32 defaultTimeout;    /* 默认超时(ms) */
    INT32 verbose;            /* 详细输出 */
} TestRunnerConfigStru;

SEC_DDR_TEXT ERRNO_T TEST_RunnerRun(const TestRunnerConfigStru *config);
```

### 验收标准

1. 支持通配符过滤：`test_matmul_*`
2. 支持标签过滤：`--tags smoke`
3. 支持套件过滤：`--suite functional`
4. 返回失败用例数量

---

## REQ-FWK-005 测试输出格式

---
id: REQ-FWK-005
title: 测试输出格式
priority: P0
status: draft
parent: REQ-FWK
depends:
  - REQ-FWK-004
---

### 描述

支持多种输出格式。

### 格式要求

**控制台输出**：
```
[PASS] functional.matmul.basic_2x2 (12ms)
[FAIL] functional.matmul.large (timeout)
  └─ Assertion failed: result[0] == expected[0]
     at tests/functional/matmul/test_large.c:45
[SKIP] performance.bandwidth (platform: linux_ut)

Summary: 45 passed, 2 failed, 3 skipped (1.23s)
```

**JSON输出**（供工具解析）：
```json
{
  "summary": {"total": 50, "passed": 45, "failed": 2, "skipped": 3},
  "tests": [
    {"name": "functional.matmul.basic_2x2", "status": "PASS", "duration_ms": 12},
    {"name": "functional.matmul.large", "status": "FAIL", "error": "timeout"}
  ]
}
```

### 验收标准

1. 默认输出到stdout
2. 支持`--output json`切换格式
3. 支持`--output-file <path>`输出到文件
4. 失败用例显示详细错误信息

---

## REQ-FWK-006 Setup/Teardown

---
id: REQ-FWK-006
title: Setup和Teardown支持
priority: P1
status: draft
parent: REQ-FWK
---

### 描述

支持用例级和套件级的初始化和清理。

### 接口设计

```c
/* 套件级 */
TEST_SUITE_SETUP(suite_name) { /* 套件开始前 */ }
TEST_SUITE_TEARDOWN(suite_name) { /* 套件结束后 */ }

/* 用例级 */
TEST_SETUP(suite_name) { /* 每个用例前 */ }
TEST_TEARDOWN(suite_name) { /* 每个用例后 */ }
```

### 验收标准

1. Setup失败则跳过对应用例
2. Teardown即使用例失败也要执行
3. 套件Setup失败则跳过整个套件

---

## REQ-FWK-007 日志接口

---
id: REQ-FWK-007
title: 测试日志接口
priority: P1
status: draft
parent: REQ-FWK
---

### 描述

测试用例中的日志输出。

### 接口设计

```c
TEST_LOG_ERROR(fmt, ...)
TEST_LOG_WARN(fmt, ...)
TEST_LOG_INFO(fmt, ...)
TEST_LOG_DEBUG(fmt, ...)

/* 带条件的日志 */
TEST_LOG_IF(condition, level, fmt, ...)
```

### 验收标准

1. 日志级别可配置
2. 日志包含时间戳
3. 失败用例的日志在报告中可见
4. 支持重定向到文件

---

## REQ-FWK-008 参数化测试

---
id: REQ-FWK-008
title: 参数化测试支持
priority: P0
status: draft
parent: REQ-FWK
---

### 描述

同一测试逻辑，使用不同参数多次运行。

### 参数类型定义

```c
/* 定义参数结构体 */
typedef struct MatmulParamStru {
    INT32 M;
    INT32 N;
    INT32 K;
} MatmulParamStru;

/* 参数类型注册宏 */
PARAM_TYPE_DEFINE(MatmulParamStru,
    PARAM_FIELD(INT32, M),
    PARAM_FIELD(INT32, N),
    PARAM_FIELD(INT32, K)
);
```

### 接口设计

```c
/* 方式1：内联参数列表 */
TEST_CASE_PARAM(matmul, shapes, MatmulParamStru,
    {.M = 2,   .N = 2,   .K = 2},
    {.M = 4,   .N = 4,   .K = 4},
    {.M = 128, .N = 256, .K = 512}
)
{
    INT32 M = param.M;
    INT32 N = param.N;
    INT32 K = param.K;
    /* 测试代码 */
    return TEST_PASS;
}

/* 方式2：参数生成函数 */
SEC_DDR_DATA MatmulParamStru g_matmulParamList[] = {{2, 2, 2}, {4, 4, 4}, {128, 256, 512}};

SEC_DDR_TEXT ERRNO_T GenMatmulParams(MatmulParamStru **params, INT32 *count)
{
    RET_IF_PTR_INVALID(params, ERR_FWK_0001);
    RET_IF_PTR_INVALID(count, ERR_FWK_0002);
    *params = g_matmulParamList;
    *count = 3;
    return ERR_OK;
}

TEST_CASE_PARAM_GEN(matmul, dynamic_shapes, MatmulParamStru, GenMatmulParams)
{
    /* param 由生成函数提供 */
    return TEST_PASS;
}
```

### 参数生成函数接口

```c
/**
 * @brief 参数生成函数原型
 * @param[out] params  参数数组指针（由函数分配或指向全局数组）
 * @param[out] count   参数数量
 * @return ERR_OK成功，其他失败
 */
typedef ERRNO_T (*ParamGeneratorFunc)(VOID **params, INT32 *count);
```

### 验收标准

1. 每组参数作为独立用例统计
2. 用例名自动包含参数信息：`suite.name[0]`, `suite.name[1]`
3. 单组参数失败不影响其他组
4. 支持参数生成函数（动态参数）
5. 参数类型需通过 PARAM_TYPE_DEFINE 注册

---

## REQ-FWK-009 数据驱动测试

---
id: REQ-FWK-009
title: 数据驱动测试
priority: P0
status: draft
parent: REQ-FWK
---

### 描述

测试数据与代码分离，从外部文件读取。

### 数据格式

```yaml
# testdata/cases/matmul_cases.yaml
cases:
  - name: small_2x2
    params:
      M: 2
      N: 2
      K: 2
    input_files:
      A: inputs/matmul/small_A.bin
      B: inputs/matmul/small_B.bin
    golden_file: golden/matmul/small_C.bin
    tolerance: 1e-5

  - name: large_1024
    params:
      M: 1024
      N: 1024
      K: 1024
    input_files:
      A: inputs/matmul/large_A.bin
      B: inputs/matmul/large_B.bin
    golden_file: golden/matmul/large_C.bin
    tolerance: 1e-4
```

### 实现机制

**编译时处理**（保持C代码无外部依赖）：

```
YAML数据文件 → Python预处理脚本 → 生成C数组头文件 → C测试引用
```

```bash
# 构建时自动执行
python tools/data/yaml_to_c.py tests/data/matmul_cases.yaml \
    --output build/generated/matmul_cases.h
```

生成的头文件：
```c
/* build/generated/matmul_cases.h (自动生成，勿手动修改) */
static const matmul_testcase_t matmul_cases[] = {
    {.name="small_2x2", .M=2, .N=2, .K=2, .tolerance=1e-5,
     .input_A="inputs/matmul/small_A.bin",
     .input_B="inputs/matmul/small_B.bin",
     .golden="golden/matmul/small_C.bin"},
    {.name="large_1024", .M=1024, .N=1024, .K=1024, .tolerance=1e-4,
     .input_A="inputs/matmul/large_A.bin",
     .input_B="inputs/matmul/large_B.bin",
     .golden="golden/matmul/large_C.bin"},
};
static const int matmul_cases_count = 2;
```

### 接口设计

```c
/* 引用生成的数据 */
#include "build/generated/matmul_cases.h"

TEST_CASE_DATA(matmul, from_yaml, matmul_cases, matmul_cases_count)
{
    /* data 指向当前测试数据项 */
    load_binary(data->input_A, A_buf);
    load_binary(data->input_B, B_buf);
    load_binary(data->golden, golden_buf);

    npu_matmul(A_buf, B_buf, output, data->M, data->N, data->K);
    TEST_ASSERT_TENSOR_NEAR(golden_buf, output, count, data->tolerance);
    return TEST_PASS;
}
```

### 验收标准

1. YAML/JSON 由Python脚本在**编译时**转换为C头文件
2. C测试代码保持**无外部依赖**
3. 二进制数据（input/golden）在**运行时**加载
4. 每条数据作为独立用例
5. 修改YAML后需重新构建（make会自动检测依赖）

---

## REQ-FWK-010 Mock框架

---
id: REQ-FWK-010
title: Mock框架
priority: P1
status: draft
parent: REQ-FWK
---

### 描述

模拟被测对象的外部依赖，用于单元测试隔离。

### 实现机制

采用**函数指针替换**方式（兼容性最好）：

```c
/* 原理：被测代码通过函数指针调用，测试时替换指针 */

/* 1. 被测代码定义函数指针（在HAL层） */
typedef uint32_t (*hal_reg_read_fn)(uint32_t addr);
extern hal_reg_read_fn g_hal_reg_read;  /* 全局函数指针 */

/* 2. 真实实现 */
uint32_t real_hal_reg_read(uint32_t addr) { /* 真实硬件访问 */ }

/* 3. Mock实现 */
uint32_t mock_hal_reg_read(uint32_t addr) { /* 模拟行为 */ }

/* 4. 测试时替换 */
g_hal_reg_read = mock_hal_reg_read;  /* 切换到Mock */
g_hal_reg_read = real_hal_reg_read;  /* 恢复真实 */
```

**备选方案**（需链接器支持）：
- `--wrap` 链接器选项：`gcc -Wl,--wrap=hal_reg_read`
- 适用于无法修改被测代码的情况

### 接口设计

```c
/* 定义Mock函数 */
MOCK_DEFINE(hal_reg_read, uint32_t, (uint32_t addr))
{
    MOCK_RECORD_CALL(addr);  /* 记录调用参数 */
    if (addr == REG_STATUS) return mock_ctx.status_value;
    return MOCK_RETURN_VALUE(hal_reg_read);  /* 返回预设值 */
}

/* 使用Mock */
TEST_CASE(driver, init_sequence)
{
    /* 启用Mock并设置返回值 */
    MOCK_ENABLE(hal_reg_read);
    MOCK_SET_RETURN(hal_reg_read, 0x1234);

    /* 调用被测代码 */
    npu_init();

    /* 验证Mock调用情况 */
    MOCK_VERIFY_CALL_COUNT(hal_reg_read, 3);
    MOCK_VERIFY_CALLED_WITH(hal_reg_read, 0, REG_STATUS);  /* 第0次调用参数 */

    /* 恢复（自动在用例结束时执行） */
    MOCK_DISABLE(hal_reg_read);
    return TEST_PASS;
}
```

### Mock上下文结构

```c
typedef struct MockContextStru {
    INT32 callCount;                     /* 调用次数 */
    VOID *callArgs[MAX_MOCK_CALLS];      /* 调用参数记录 */
    VOID *returnValues[MAX_MOCK_CALLS];  /* 返回值序列 */
    INT32 returnIndex;                   /* 当前返回值索引 */
} MockContextStru;
```

### 验收标准

1. 通过函数指针替换实现Mock（无需特殊链接器）
2. 支持记录调用参数（最多记录 MAX_MOCK_CALLS=100 次）
3. 支持验证调用次数和参数
4. 支持设置返回值序列（多次调用返回不同值）
5. Mock作用域隔离（每个用例独立的mock_context）
6. 用例结束自动恢复原函数

---

## REQ-FWK-011 约束随机测试

---
id: REQ-FWK-011
title: 约束随机测试
priority: P2
status: draft
parent: REQ-FWK
---

### 描述

自动生成满足约束的随机测试输入。

**注意**：完整的constraint solver（如UVM）实现复杂，本框架采用**简化方案**：
- 支持基本随机（范围、枚举）
- 约束通过**拒绝采样**实现（生成后检查，不满足则重新生成）
- 不实现复杂的SAT求解器

### 接口设计

```c
/* 定义随机参数结构 */
typedef struct {
    RAND_INT(M, 1, 2048);           /* M: 1~2048随机整数 */
    RAND_INT(N, 1, 2048);           /* N: 1~2048随机整数 */
    RAND_INT(K, 1, 2048);           /* K: 1~2048随机整数 */
    RAND_ENUM(dtype, 3, DTYPE_FP32, DTYPE_FP16, DTYPE_INT8);  /* 枚举 */
} matmul_rand_t;

/* 约束检查函数（拒绝采样） */
static int matmul_constraint_check(const matmul_rand_t *r) {
    if (r->M * r->K > 1024 * 1024) return 0;  /* 不满足，拒绝 */
    if (r->dtype == DTYPE_INT8 && r->K % 4 != 0) return 0;  /* INT8需对齐 */
    return 1;  /* 满足约束 */
}

/* 随机测试 */
TEST_CASE_RANDOM(matmul, random_shapes, matmul_rand_t, matmul_constraint_check, 100)
{
    /* 运行100次，每次随机参数（满足约束） */
    TEST_LOG_INFO("Testing M=%d, N=%d, K=%d, dtype=%d", rand.M, rand.N, rand.K, rand.dtype);

    float *A = alloc_random_tensor(rand.M, rand.K, rand.dtype);
    float *B = alloc_random_tensor(rand.K, rand.N, rand.dtype);
    /* ... */
    return TEST_PASS;
}
```

### 随机数生成

```c
/* 设置seed（便于复现） */
SEC_DDR_TEXT VOID TEST_RandomSeed(UINT32 seed);

/* 获取当前seed（失败时记录） */
SEC_DDR_TEXT UINT32 TEST_RandomGetSeed(VOID);

/* 随机生成宏（内部使用） */
#define RAND_INT(name, min, max)    INT32 name
#define RAND_ENUM(name, count, ...) INT32 name
```

### 失败时输出

```
[FAIL] matmul.random_shapes[42]
  Seed: 0x12345678  <-- 使用此seed可复现
  Parameters: M=1024, N=512, K=2048, dtype=FP16
  Error: cosine_sim 0.985 < 0.999
```

### 验收标准

1. 支持范围随机（RAND_INT）和枚举随机（RAND_ENUM）
2. 约束通过**拒绝采样**实现（最多重试1000次）
3. 可复现：相同seed产生相同序列
4. 失败时自动记录seed
5. 支持命令行指定seed：`--seed 0x12345678`

---

## REQ-FWK-012 性能基准测试

---
id: REQ-FWK-012
title: 性能基准测试
priority: P1
status: draft
parent: REQ-FWK
---

### 描述

性能测试和基准对比。

### 接口设计

```c
BENCHMARK(matmul, perf_1024x1024)
{
    BENCH_SETUP {
        A = alloc_tensor(1024, 1024);
        B = alloc_tensor(1024, 1024);
        C = alloc_tensor(1024, 1024);
    }

    BENCH_WARMUP(10) {
        npu_matmul(A, B, C, 1024, 1024, 1024);
    }

    BENCH_RUN(100) {
        npu_matmul(A, B, C, 1024, 1024, 1024);
    }

    BENCH_TEARDOWN {
        free_tensor(A);
        free_tensor(B);
        free_tensor(C);
    }
}
```

### 输出格式

```
[BENCH] matmul.perf_1024x1024
  Iterations: 100
  Total time: 123.45ms
  Avg: 1.23ms, Min: 1.20ms, Max: 1.35ms, Std: 0.05ms
  Throughput: 1.72 TFLOPS
  Bandwidth: 12.3 GB/s
```

### 验收标准

1. 支持warmup阶段
2. 统计min/max/avg/stddev
3. 支持自定义指标（TFLOPS、带宽等）
4. 支持与baseline对比，检测性能回归
5. 结果可导出为JSON/CSV

---

## REQ-FWK-013 覆盖率收集

---
id: REQ-FWK-013
title: 功能覆盖率
priority: P2
status: draft
parent: REQ-FWK
---

### 描述

收集功能覆盖率，验证测试充分性。

### 接口设计

```c
/* 定义覆盖组（借鉴UVM covergroup） */
COVERAGE_GROUP(matmul_coverage)
{
    /* 覆盖点 */
    COVERPOINT(M,
        BIN(tiny, 1, 16),
        BIN(small, 17, 128),
        BIN(medium, 129, 512),
        BIN(large, 513, 2048)
    );

    COVERPOINT(dtype,
        BIN(fp32, DTYPE_FP32),
        BIN(fp16, DTYPE_FP16),
        BIN(int8, DTYPE_INT8)
    );

    /* 交叉覆盖 */
    CROSS(M, dtype);
}

/* 在测试中采样 */
TEST_CASE(matmul, various_sizes)
{
    COVERAGE_SAMPLE(matmul_coverage, .M = M, .dtype = dtype);
    /* ... */
}
```

### 验收标准

1. 支持覆盖点（coverpoint）
2. 支持交叉覆盖（cross）
3. 生成覆盖率报告（HTML/JSON）
4. 支持设置覆盖率目标

---

## REQ-FWK-014 Fixture管理

---
id: REQ-FWK-014
title: Fixture环境管理
priority: P1
status: draft
parent: REQ-FWK
---

### 描述

可复用的测试环境配置。

### 接口设计

```c
/* 定义Fixture */
FIXTURE_DEFINE(npu_initialized)
{
    FIXTURE_DATA {
        npu_handle_t *handle;
        npu_buffer_t *scratch;
    };

    FIXTURE_SETUP {
        self->handle = npu_open();
        self->scratch = npu_alloc(1024 * 1024);
    }

    FIXTURE_TEARDOWN {
        npu_free(self->scratch);
        npu_close(self->handle);
    }
}

/* 使用Fixture */
TEST_WITH_FIXTURE(npu_initialized, matmul, basic)
{
    /* fixture->handle 和 fixture->scratch 可用 */
    npu_matmul(fixture->handle, ...);
    return TEST_PASS;
}

/* Fixture组合 */
FIXTURE_COMPOSE(full_env, npu_initialized, memory_pool, logging);
```

### 验收标准

1. Fixture自动setup/teardown
2. 支持Fixture嵌套和组合
3. Teardown即使失败也执行
4. Fixture数据在用例间隔离

---

## REQ-FWK-015 调试支持

---
id: REQ-FWK-015
title: 失败调试支持
priority: P2
status: draft
parent: REQ-FWK
---

### 描述

测试失败时提供调试信息。

### 接口设计

```c
/* 失败时自动dump */
TEST_ON_FAILURE {
    TEST_DUMP_FILE("input.bin", input_buf, input_size);
    TEST_DUMP_FILE("output.bin", output_buf, output_size);
    TEST_DUMP_FILE("golden.bin", golden_buf, golden_size);
    TEST_DUMP_REGS("regs.txt", 0x0, 0x1000);
}

/* 断点条件 */
TEST_BREAK_IF(error_count > 10);

/* 详细diff */
TEST_ASSERT_TENSOR_NEAR_VERBOSE(expected, actual, count, eps);
/* 输出: Mismatch at [123]: expected=1.234, actual=1.567, diff=0.333 */
```

### 验收标准

1. 失败时自动保存输入/输出数据
2. 提供详细的差异信息
3. 支持条件断点
4. Dump文件路径可配置

---

## REQ-FWK-016 测试分片

---
id: REQ-FWK-016
title: 测试分片执行
priority: P2
status: draft
parent: REQ-FWK
---

### 描述

大规模测试分片并行执行。

### 命令行接口

```bash
# 分4片，运行第1片
./test_runner --shard-index 0 --shard-count 4

# 按时间平衡分片
./test_runner --shard-index 0 --shard-count 4 --shard-strategy time
```

### 分片策略

| 策略 | 描述 |
|------|------|
| round-robin | 轮询分配 |
| hash | 按用例名hash |
| time | 按历史耗时平衡 |

### 验收标准

1. 分片间用例不重复
2. 所有分片并集覆盖全部用例
3. 支持按历史耗时平衡分片
4. 结果可合并生成完整报告
