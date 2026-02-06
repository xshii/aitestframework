# RST 结果管理模块设计

---
module: RST
version: 1.0
date: 2026-02-04
status: draft
requirements: REQ-RST-001
domain: Supporting
aggregate: Execution
---

## 1. 模块概述

### 1.1 职责

管理测试结果的收集和输出：
- 测试结果收集
- 结果格式化输出
- 结果持久化存储（P1）
- 报告生成（P1）
- 趋势分析（P2）

### 1.2 DDD定位

- **限界上下文**：结果管理上下文
- **聚合根**：Execution
- **订阅事件**：TestCaseCompleted, TestExecutionCompleted
- **发布事件**：ReportGenerated

---

## 2. 聚合设计

### 2.1 Execution聚合

```
Execution (聚合根)
├── executionId: STRING        # 执行ID (如 20260204_103000_abc123)
├── platform: STRING           # 平台名称
├── startedAt: TIMESTAMP       # 开始时间
├── finishedAt: TIMESTAMP      # 结束时间
├── summary: TestSummary       # 汇总（值对象）
│   ├── total: UINT32
│   ├── passed: UINT32
│   ├── failed: UINT32
│   ├── skipped: UINT32
│   └── durationMs: UINT32
└── results: TestCaseResult[]  # 用例结果列表（实体）
    ├── testName: STRING
    ├── status: ResultStatus
    ├── durationMs: UINT32
    └── errorInfo: STRING
```

---

## 3. 结果收集

### 3.1 事件订阅

```c
/* 订阅FWK事件，收集结果 */
VOID RST_OnTestCaseCompleted(const TestCaseCompletedEvent *event)
{
    TestCaseResultStru result = {
        .suite = event->suite,
        .name = event->name,
        .status = ConvertStatus(event->result),
        .durationMs = event->durationMs,
        .failFile = event->failFile,
        .failLine = event->failLine,
        .failExpr = event->failExpr,
    };

    AddResult(&g_execution, &result);
}
```

### 3.2 结果数据结构

```c
typedef struct ExecutionStru {
    CHAR        executionId[64];
    CHAR        platform[32];
    UINT64      startedAt;
    UINT64      finishedAt;
    TestSummaryStru summary;
    TestCaseResultStru *results;
    UINT32      resultCount;
    UINT32      resultCapacity;
} ExecutionStru;

static ExecutionStru g_execution;
```

### 3.3 收集流程

```
TestCaseCompleted事件
        │
        ▼
┌─────────────────┐
│  RST订阅处理    │
│  更新results[]  │
│  更新summary    │
└─────────────────┘
        │
        ▼
TestExecutionCompleted事件
        │
        ▼
┌─────────────────┐
│  输出最终结果   │
│  - stdout       │
│  - JSON文件     │
└─────────────────┘
```

---

## 4. 输出格式

### 4.1 控制台输出

```
=== Test Execution Started ===
Platform: linux_ut
Time: 2026-02-04 10:30:00

[PASS] sanity.hello (1ms)
[PASS] sanity.basic_op (2ms)
[FAIL] matmul.large (5012ms)
       at test_matmul_large.c:45
       assertion failed: cosine_sim >= 0.999
[SKIP] perf.latency
       reason: requires real hardware
[TIMEOUT] stress.long_run (10000ms)
[CRASH] buggy.segfault (15ms)
       signal: SIGSEGV

=== Test Execution Completed ===
Summary: 145 passed, 3 failed, 2 skipped, 1 timeout, 1 crashed
Duration: 5230ms
```

### 4.2 JSON输出

```json
{
  "execution_id": "20260204_103000_abc123",
  "platform": "linux_ut",
  "started_at": "2026-02-04T10:30:00Z",
  "finished_at": "2026-02-04T10:30:05Z",
  "toolchain": {
    "simulator": "2.0.0",
    "compiler": "2.1.0"
  },
  "summary": {
    "total": 152,
    "passed": 145,
    "failed": 3,
    "skipped": 2,
    "timeout": 1,
    "crashed": 1,
    "duration_ms": 5230
  },
  "tests": [
    {
      "suite": "sanity",
      "name": "hello",
      "status": "PASS",
      "duration_ms": 1,
      "tags": ["smoke"]
    },
    {
      "suite": "matmul",
      "name": "large",
      "status": "FAIL",
      "duration_ms": 5012,
      "fail_file": "test_matmul_large.c",
      "fail_line": 45,
      "fail_reason": "cosine_sim >= 0.999",
      "tags": ["matmul"]
    },
    {
      "suite": "stress",
      "name": "long_run",
      "status": "TIMEOUT",
      "duration_ms": 10000
    },
    {
      "suite": "buggy",
      "name": "segfault",
      "status": "CRASH",
      "duration_ms": 15,
      "fail_reason": "SIGSEGV"
    }
  ]
}
```

### 4.3 JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["execution_id", "platform", "summary", "tests"],
  "properties": {
    "execution_id": {"type": "string"},
    "platform": {"type": "string"},
    "started_at": {"type": "string", "format": "date-time"},
    "finished_at": {"type": "string", "format": "date-time"},
    "summary": {
      "type": "object",
      "properties": {
        "total": {"type": "integer"},
        "passed": {"type": "integer"},
        "failed": {"type": "integer"},
        "skipped": {"type": "integer"},
        "timeout": {"type": "integer"},
        "crashed": {"type": "integer"},
        "duration_ms": {"type": "integer"}
      }
    },
    "tests": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["suite", "name", "status"],
        "properties": {
          "suite": {"type": "string"},
          "name": {"type": "string"},
          "status": {"enum": ["PASS", "FAIL", "SKIP", "TIMEOUT", "ERROR", "CRASH"]},
          "duration_ms": {"type": "integer"},
          "fail_file": {"type": "string"},
          "fail_line": {"type": "integer"},
          "fail_reason": {"type": "string"}
        }
      }
    }
  }
}
```

---

## 5. 输出实现

### 5.1 控制台输出

```c
static VOID OutputConsole_TestResult(const TestCaseResultStru *result)
{
    const CHAR *statusStr;
    const CHAR *color;

    switch (result->status) {
        case TEST_PASS:    statusStr = "PASS";    color = "\033[32m"; break;
        case TEST_FAIL:    statusStr = "FAIL";    color = "\033[31m"; break;
        case TEST_SKIP:    statusStr = "SKIP";    color = "\033[33m"; break;
        case TEST_TIMEOUT: statusStr = "TIMEOUT"; color = "\033[35m"; break;
        case TEST_CRASH:   statusStr = "CRASH";   color = "\033[31m"; break;
        default:           statusStr = "ERROR";   color = "\033[31m"; break;
    }

    printf("%s[%s]\033[0m %s.%s (%ums)\n",
           color, statusStr, result->suite, result->name, result->durationMs);

    if (result->status == TEST_FAIL && result->failFile) {
        printf("       at %s:%d\n", result->failFile, result->failLine);
        printf("       assertion failed: %s\n", result->failExpr);
    }
}
```

### 5.2 JSON输出

```c
static ERRNO_T OutputJson_Execution(const ExecutionStru *exec,
                                    const CHAR *outputPath)
{
    FILE *fp = fopen(outputPath, "w");
    RET_IF_NULL(fp);

    fprintf(fp, "{\n");
    fprintf(fp, "  \"execution_id\": \"%s\",\n", exec->executionId);
    fprintf(fp, "  \"platform\": \"%s\",\n", exec->platform);

    /* 汇总 */
    fprintf(fp, "  \"summary\": {\n");
    fprintf(fp, "    \"total\": %u,\n", exec->summary.total);
    fprintf(fp, "    \"passed\": %u,\n", exec->summary.passed);
    fprintf(fp, "    \"failed\": %u,\n", exec->summary.failed);
    fprintf(fp, "    \"skipped\": %u,\n", exec->summary.skipped);
    fprintf(fp, "    \"duration_ms\": %u\n", exec->summary.durationMs);
    fprintf(fp, "  },\n");

    /* 用例结果 */
    fprintf(fp, "  \"tests\": [\n");
    for (UINT32 i = 0; i < exec->resultCount; i++) {
        OutputJson_TestResult(fp, &exec->results[i], i < exec->resultCount - 1);
    }
    fprintf(fp, "  ]\n");
    fprintf(fp, "}\n");

    fclose(fp);
    return AITF_OK;
}
```

---

## 6. 结果存储（P1）

### 6.1 存储路径

```
build/results/
├── 20260204/
│   ├── 103000_abc123/
│   │   ├── result.json
│   │   ├── console.log
│   │   └── artifacts/
│   └── 143000_def456/
│       └── result.json
└── latest -> 20260204/143000_def456/
```

### 6.2 SQLite Schema（可选）

```sql
CREATE TABLE executions (
    id TEXT PRIMARY KEY,
    platform TEXT,
    started_at DATETIME,
    finished_at DATETIME,
    total INTEGER,
    passed INTEGER,
    failed INTEGER,
    skipped INTEGER,
    duration_ms INTEGER
);

CREATE TABLE test_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id TEXT,
    suite TEXT,
    name TEXT,
    status TEXT,
    duration_ms INTEGER,
    fail_file TEXT,
    fail_line INTEGER,
    fail_reason TEXT,
    FOREIGN KEY (execution_id) REFERENCES executions(id)
);

CREATE INDEX idx_execution ON test_results(execution_id);
CREATE INDEX idx_test_name ON test_results(suite, name);
```

---

## 7. 领域事件

### 7.1 订阅的事件

```yaml
TestCaseCompleted:
  handler: RST_OnTestCaseCompleted
  action: 记录单个用例结果

TestExecutionCompleted:
  handler: RST_OnExecutionCompleted
  action: 输出最终结果、生成报告
```

### 7.2 发布的事件

```yaml
ReportGenerated:
  source: RST
  payload:
    executionId: STRING
    reportPath: STRING
    format: STRING  # "json", "html", "junit"
```

---

## 8. 使用示例

```c
/* 初始化结果收集 */
RST_Init("linux_ut");

/* 运行测试（FWK会发布事件） */
Runner_RunAll(&summary);

/* 输出结果 */
RST_OutputConsole();
RST_OutputJson("build/results/result.json");

/* 清理 */
RST_Cleanup();
```

---

## 9. 需求追溯

| 需求ID | 需求标题 | 设计章节 |
|--------|----------|----------|
| REQ-RST-001 | 测试结果收集 | 3, 4, 5 |
| REQ-RST-002 | 结果持久化存储 | 6 (P1) |
| REQ-RST-004 | 测试报告生成 | DVT模块 (P1) |
