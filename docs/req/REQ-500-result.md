# REQ-RST 结果管理需求

---
id: REQ-RST
title: 结果管理需求
priority: P1
status: draft
parent: REQ-SYS
depends:
  - REQ-FWK
  - REQ-TMT
---

## 概述

测试结果的收集、存储、归档、报告和趋势分析。

---

## REQ-RST-001 结果收集

---
id: REQ-RST-001
title: 测试结果收集
priority: P0
status: draft
parent: REQ-RST
---

### 描述

收集测试执行的详细结果。

### 结果结构

```json
{
  "execution_id": "20260202_103000_abc123",
  "started_at": "2026-02-02T10:30:00Z",
  "finished_at": "2026-02-02T10:35:23Z",
  "platform": "simulator",
  "toolchain": {
    "simulator": "1.1.0",
    "compiler": "1.5.0"
  },
  "summary": {
    "total": 150,
    "passed": 145,
    "failed": 3,
    "skipped": 2,
    "duration_ms": 323000
  },
  "tests": [
    {
      "name": "functional.matmul.basic_2x2",
      "suite": "functional.matmul",
      "status": "PASS",
      "duration_ms": 12,
      "tags": ["smoke", "matmul"]
    },
    {
      "name": "functional.matmul.large",
      "suite": "functional.matmul",
      "status": "FAIL",
      "duration_ms": 5000,
      "error": {
        "type": "assertion",
        "message": "cosine_sim 0.985 < threshold 0.999",
        "file": "test_matmul_large.c",
        "line": 45
      },
      "artifacts": ["output.bin", "golden.bin", "diff.log"]
    }
  ]
}
```

### 验收标准

1. 记录每个用例的详细结果
2. 记录失败原因和位置
3. 记录执行环境信息
4. 支持附加artifact文件

---

## REQ-RST-002 结果存储

---
id: REQ-RST-002
title: 结果持久化存储
priority: P0
status: draft
parent: REQ-RST
---

### 描述

持久化存储测试结果。

### 存储方案

```
存储后端:
├── 文件系统 (默认)
│   └── build/results/<date>/<execution_id>/
│       ├── result.json
│       ├── logs/
│       └── artifacts/
│
├── SQLite (本地数据库)
│   └── data/testresults.db
│
└── 远程服务 (可选)
    └── POST /api/results
```

### 数据库Schema

```sql
-- 执行批次
CREATE TABLE executions (
    id TEXT PRIMARY KEY,
    started_at DATETIME,
    finished_at DATETIME,
    platform TEXT,
    toolchain_json TEXT,
    total INTEGER,
    passed INTEGER,
    failed INTEGER,
    skipped INTEGER
);

-- 用例结果
CREATE TABLE test_results (
    id INTEGER PRIMARY KEY,
    execution_id TEXT,
    test_name TEXT,
    status TEXT,
    duration_ms INTEGER,
    error_message TEXT,
    FOREIGN KEY (execution_id) REFERENCES executions(id)
);
```

### 验收标准

1. 默认存储到文件系统
2. 可选SQLite便于查询
3. 支持远程上报
4. 存储失败不影响测试执行

---

## REQ-RST-003 结果归档

---
id: REQ-RST-003
title: 历史结果归档
priority: P1
status: draft
parent: REQ-RST
---

### 描述

归档历史结果，管理存储空间。

### 归档策略

```yaml
# configs/archive_policy.yaml
archive:
  # 保留策略
  retention:
    recent_days: 30      # 最近30天全量保留
    monthly_days: 365    # 每月保留1份，保留1年
    yearly_days: -1      # 每年保留1份，永久

  # 压缩策略
  compression:
    after_days: 7        # 7天后压缩
    format: tar.gz

  # 存储路径
  paths:
    active: build/results/
    archive: /data/archive/testresults/
```

### 验收标准

1. 自动清理过期结果
2. 保留关键节点（发版、里程碑）
3. 归档后可恢复查看
4. 支持自定义保留策略

---

## REQ-RST-004 报告生成

---
id: REQ-RST-004
title: 测试报告生成
priority: P0
status: draft
parent: REQ-RST
---

### 描述

生成多种格式的测试报告。

### 报告格式

**HTML报告**:
```
┌─────────────────────────────────────┐
│  Test Report - 2026-02-02 Nightly   │
├─────────────────────────────────────┤
│  Summary: 145/150 passed (96.7%)    │
│  Duration: 5m 23s                   │
├─────────────────────────────────────┤
│  ● Passed: 145                      │
│  ● Failed: 3                        │
│  ● Skipped: 2                       │
├─────────────────────────────────────┤
│  Failed Tests:                      │
│  ✗ functional.matmul.large          │
│    └─ cosine_sim 0.985 < 0.999      │
│  ✗ functional.conv.stride2          │
│    └─ timeout after 10000ms         │
└─────────────────────────────────────┘
```

**JUnit XML** (CI集成):
```xml
<testsuite name="functional" tests="50" failures="2" time="123.45">
  <testcase name="matmul.basic_2x2" time="0.012"/>
  <testcase name="matmul.large" time="5.0">
    <failure message="cosine_sim 0.985 &lt; 0.999"/>
  </testcase>
</testsuite>
```

### 验收标准

1. 支持HTML、JSON、JUnit XML
2. HTML报告可独立查看（内联CSS/JS）
3. 失败用例突出显示
4. 包含趋势图表

---

## REQ-RST-005 趋势分析

---
id: REQ-RST-005
title: 结果趋势分析
priority: P1
status: draft
parent: REQ-RST
---

### 描述

分析历史结果趋势。

### 分析维度

1. **通过率趋势**：每日/每周通过率曲线
2. **耗时趋势**：用例执行时间变化
3. **失败热点**：频繁失败的用例
4. **Flaky检测**：不稳定用例识别

### API设计

```python
# Python分析接口
from tools.analysis import TrendAnalyzer

analyzer = TrendAnalyzer(db_path="data/results.db")

# 通过率趋势
pass_rate = analyzer.get_pass_rate_trend(
    start_date="2026-01-01",
    end_date="2026-02-01",
    platform="simulator"
)

# 失败热点
hot_failures = analyzer.get_failure_hotspots(
    days=30,
    min_failures=3
)

# Flaky用例
flaky_tests = analyzer.detect_flaky_tests(
    window_size=10,
    flip_threshold=3
)
```

### 验收标准

1. 提供趋势可视化图表
2. 自动检测Flaky用例
3. 支持性能回归检测
4. 结果可导出

---

## REQ-RST-006 结果对比

---
id: REQ-RST-006
title: 结果差异对比
priority: P1
status: draft
parent: REQ-RST
---

### 描述

对比两次执行结果的差异。

### 对比功能

```bash
# 命令行对比
./tools/compare_results.py --base exec_001 --target exec_002

# 输出
Comparison: exec_001 vs exec_002
================================
New Failures (3):
  - functional.matmul.large
  - functional.conv.stride2
  - e2e.resnet.layer1

Fixed (2):
  - functional.activation.gelu
  - stress.long_run.basic

Status Changed (1):
  - functional.precision.fp16: SKIP -> PASS

New Tests (5):
  - functional.matmul.bf16_*

Removed Tests (0):
  (none)
```

### 验收标准

1. 识别新增失败
2. 识别修复的用例
3. 识别新增/删除的用例
4. 支持生成diff报告

---

## REQ-RST-007 通知告警

---
id: REQ-RST-007
title: 结果通知
priority: P2
status: draft
parent: REQ-RST
---

### 描述

测试完成后发送通知。

### 通知渠道

```yaml
# configs/notification.yaml
notifications:
  email:
    enabled: true
    recipients:
      - team@example.com
    on: [failure, recovery]

  slack:
    enabled: true
    webhook: "https://hooks.slack.com/..."
    channel: "#npu-test"
    on: [failure]

  webhook:
    enabled: true
    url: "https://api.example.com/notify"
    on: [always]
```

### 通知内容

```
[FAILED] Nightly Test - 2026-02-02

Summary: 145/150 passed (96.7%)
Platform: simulator
Duration: 5m 23s

Failed Tests:
- functional.matmul.large
- functional.conv.stride2
- e2e.resnet.layer1

Report: https://ci.example.com/report/20260202
```

### 验收标准

1. 支持邮件、Slack、Webhook
2. 可配置触发条件
3. 通知内容可定制
4. 失败重试后成功发recovery通知