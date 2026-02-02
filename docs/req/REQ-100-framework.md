# REQ-100 测试框架需求

---
id: REQ-100
title: C测试框架需求
priority: P0
status: draft
parent: REQ-000
---

## 概述

纯C实现的测试框架，提供测试注册、断言、运行、报告等核心功能。

---

## REQ-101 测试用例注册

---
id: REQ-101
title: 测试用例自动注册
priority: P0
status: draft
parent: REQ-100
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

## REQ-102 断言宏

---
id: REQ-102
title: 断言宏集合
priority: P0
status: draft
parent: REQ-100
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

## REQ-103 测试结果类型

---
id: REQ-103
title: 测试结果类型
priority: P0
status: draft
parent: REQ-100
---

### 描述

定义测试用例的执行结果类型。

### 接口设计

```c
typedef enum {
    TEST_PASS    = 0,   /* 通过 */
    TEST_FAIL    = 1,   /* 失败 */
    TEST_SKIP    = 2,   /* 跳过 */
    TEST_TIMEOUT = 3,   /* 超时 */
    TEST_ERROR   = 4,   /* 错误（框架异常） */
} test_result_t;
```

### 验收标准

1. 每个用例返回明确的结果类型
2. SKIP用于平台不支持等场景
3. TIMEOUT由框架自动判定
4. ERROR用于框架本身异常

---

## REQ-104 测试运行器

---
id: REQ-104
title: 测试运行器
priority: P0
status: draft
parent: REQ-100
depends:
  - REQ-101
  - REQ-103
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
typedef struct test_runner_config {
    const char *filter;       /* 名称过滤 (支持通配符) */
    const char *tags;         /* 标签过滤 */
    const char *suite;        /* 套件过滤 */
    uint32_t default_timeout; /* 默认超时(ms) */
    int verbose;              /* 详细输出 */
} test_runner_config_t;

int test_runner_run(const test_runner_config_t *config);
```

### 验收标准

1. 支持通配符过滤：`test_matmul_*`
2. 支持标签过滤：`--tags smoke`
3. 支持套件过滤：`--suite functional`
4. 返回失败用例数量

---

## REQ-105 测试输出格式

---
id: REQ-105
title: 测试输出格式
priority: P0
status: draft
parent: REQ-100
depends:
  - REQ-104
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

## REQ-106 Setup/Teardown

---
id: REQ-106
title: Setup和Teardown支持
priority: P1
status: draft
parent: REQ-100
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

## REQ-107 日志接口

---
id: REQ-107
title: 测试日志接口
priority: P1
status: draft
parent: REQ-100
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

## REQ-108 参数化测试

---
id: REQ-108
title: 参数化测试支持
priority: P0
status: draft
parent: REQ-100
---

### 描述

同一测试逻辑，使用不同参数多次运行。

### 接口设计

```c
/* 内联参数 */
TEST_CASE_PARAM(suite, name, param_type,
    {2, 2, 2},
    {4, 4, 4},
    {128, 256, 512}
)
{
    int M = param.a, N = param.b, K = param.c;
    /* 测试代码 */
    return TEST_PASS;
}

/* 参数生成函数 */
TEST_CASE_PARAM_GEN(suite, name, param_type, generator_func)
{
    /* 测试代码 */
}
```

### 验收标准

1. 每组参数作为独立用例统计
2. 用例名自动包含参数信息：`suite.name[0]`, `suite.name[1]`
3. 单组参数失败不影响其他组
4. 支持参数生成函数（动态参数）

---

## REQ-109 数据驱动测试

---
id: REQ-109
title: 数据驱动测试
priority: P0
status: draft
parent: REQ-100
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

### 接口设计

```c
TEST_CASE_DATA(suite, name, "path/to/cases.yaml")
{
    /* data->params, data->inputs, data->golden 自动加载 */
    npu_matmul(data->inputs.A, data->inputs.B, output, &data->params);
    TEST_ASSERT_TENSOR_NEAR(data->golden, output, count, data->tolerance);
    return TEST_PASS;
}
```

### 验收标准

1. 支持YAML/JSON格式数据文件
2. 自动加载输入和golden数据
3. 每条数据作为独立用例
4. 数据文件变更无需重新编译

---

## REQ-110 Mock框架

---
id: REQ-110
title: Mock框架
priority: P1
status: draft
parent: REQ-100
---

### 描述

模拟被测对象的外部依赖，用于单元测试隔离。

### 接口设计

```c
/* 定义Mock函数 */
MOCK_FUNC(hal_reg_read, uint32_t, (uint32_t addr))
{
    MOCK_RECORD_CALL(addr);  /* 记录调用 */
    if (addr == REG_STATUS) return mock_status_value;
    return 0;
}

/* 使用Mock */
TEST_CASE(driver, init_sequence)
{
    MOCK_ENABLE(hal_reg_read);
    MOCK_SET_RETURN(hal_reg_read, 0x1234);  /* 设置返回值 */

    npu_init();

    MOCK_VERIFY_CALLED(hal_reg_read, 3);     /* 验证调用次数 */
    MOCK_VERIFY_CALLED_WITH(hal_reg_read, REG_STATUS);

    MOCK_DISABLE(hal_reg_read);
    return TEST_PASS;
}
```

### 验收标准

1. 支持替换任意函数
2. 支持记录调用参数
3. 支持验证调用次数和顺序
4. 支持设置返回值序列
5. Mock作用域隔离（用例间不干扰）

---

## REQ-111 约束随机测试

---
id: REQ-111
title: 约束随机测试
priority: P1
status: draft
parent: REQ-100
---

### 描述

自动生成满足约束的随机测试输入（借鉴UVM）。

### 接口设计

```c
/* 定义约束 */
RANDOM_CONSTRAINT(matmul_constraint)
{
    RAND_RANGE(M, 1, 2048);
    RAND_RANGE(N, 1, 2048);
    RAND_RANGE(K, 1, 2048);
    RAND_ENUM(dtype, DTYPE_FP32, DTYPE_FP16, DTYPE_INT8);

    /* 约束条件 */
    CONSTRAINT(M * K <= 1024 * 1024);  /* 限制输入大小 */
    CONSTRAINT(dtype == DTYPE_INT8 ? K % 4 == 0 : 1);  /* INT8需对齐 */
}

/* 随机测试 */
TEST_CASE_RANDOM(matmul, random_shapes, matmul_constraint, 100)
{
    /* 运行100次，每次随机参数 */
    float *A = alloc_random_tensor(rand.M, rand.K, rand.dtype);
    float *B = alloc_random_tensor(rand.K, rand.N, rand.dtype);
    /* ... */
    return TEST_PASS;
}
```

### 验收标准

1. 支持范围、枚举、权重分布
2. 支持约束条件（constraint solver）
3. 可复现：相同seed产生相同序列
4. 失败时记录seed，便于复现

---

## REQ-112 性能基准测试

---
id: REQ-112
title: 性能基准测试
priority: P1
status: draft
parent: REQ-100
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

## REQ-113 覆盖率收集

---
id: REQ-113
title: 功能覆盖率
priority: P2
status: draft
parent: REQ-100
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

## REQ-114 Fixture管理

---
id: REQ-114
title: Fixture环境管理
priority: P1
status: draft
parent: REQ-100
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

## REQ-115 调试支持

---
id: REQ-115
title: 失败调试支持
priority: P2
status: draft
parent: REQ-100
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

## REQ-116 测试分片

---
id: REQ-116
title: 测试分片执行
priority: P2
status: draft
parent: REQ-100
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
