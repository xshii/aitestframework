# REQ-5 日志分析工具

---
version: 1.1
date: 2026-02-27
status: draft
---

## 1. 概述

解析两类日志：(1) 测试框架自身的执行日志；(2) 仿真器/FPGA/芯片产生的硬件日志。提取关键信息（执行结果、错误原因、性能数据），支持可配置的解析规则。

**架构原则：** 所有日志解析逻辑实现在 Core API 层（`aitestframework.logparser`），CLI 和 Web 均为薄壳调用层。

## 2. 需求详情

### REQ-5.1 测试框架日志解析 [P0]

**描述：** 解析测试框架运行过程中产生的日志，提取结构化信息。

**日志来源：** 框架执行器（REQ-6）产生的标准输出/文件日志

**提取信息：**

| 信息 | 说明 |
|------|------|
| 用例结果 | 每个用例的 PASS/FAIL/TIMEOUT/CRASH |
| 耗时 | 每个用例的执行时间 |
| 失败原因 | FAIL 用例的比较结果、CRASH 用例的信号信息 |
| 环境信息 | 平台、工具链版本、git commit |
| 汇总统计 | 总数、通过数、失败数、通过率 |

**框架日志格式（自定义结构化格式）：**

```
[2026-02-27 14:30:01] [INFO] === Execution Start ===
[2026-02-27 14:30:01] [INFO] Platform: npu, Git: abc1234
[2026-02-27 14:30:02] [INFO] [PASS] conv2d_fp32_3x3_basic (2.5s)
[2026-02-27 14:30:04] [FAIL] conv2d_fp16_3x3_basic (2.1s) max_abs_error=0.15 > atol=1e-4
[2026-02-27 14:30:07] [TIMEOUT] matmul_int8_large (60.0s) killed by timeout
[2026-02-27 14:30:07] [INFO] === Execution End: 3 total, 1 pass, 1 fail, 1 timeout ===
```

**验收标准：**
- 解析后输出结构化的 JSON 或 Python 对象
- 支持从文件或 stdin 读取日志
- 解析失败的行记录为 warning，不中断整体解析

### REQ-5.2 仿真器日志解析 [P0]

**描述：** 解析仿真器运行日志，提取关键事件和错误信息。

**典型仿真器日志事件：**

| 事件类型 | 示例 |
|----------|------|
| 启动/结束 | `Simulation started / finished` |
| 错误 | `ERROR: DMA timeout at cycle 12345` |
| 警告 | `WARNING: register X value mismatch` |
| 性能数据 | `Total cycles: 50000, Throughput: 128 GOPS` |
| 断言失败 | `ASSERTION FAILED: expected 0x1234, got 0x5678 at addr 0xFF00` |

**解析规则配置：**

```yaml
# logparser/rules/simulator.yaml
name: simulator_v3
patterns:
  - name: sim_error
    pattern: 'ERROR:\s+(.+?)(?:\s+at\s+cycle\s+(\d+))?'
    fields:
      message: "$1"
      cycle: "$2"
    level: error

  - name: sim_assertion
    pattern: 'ASSERTION FAILED:\s+expected\s+(0x[\da-fA-F]+),\s+got\s+(0x[\da-fA-F]+)\s+at\s+addr\s+(0x[\da-fA-F]+)'
    fields:
      expected: "$1"
      actual: "$2"
      address: "$3"
    level: error

  - name: sim_perf
    pattern: 'Total cycles:\s+(\d+),\s+Throughput:\s+([\d.]+)\s+GOPS'
    fields:
      total_cycles: "$1"
      throughput_gops: "$2"
    level: info

  - name: sim_warning
    pattern: 'WARNING:\s+(.+)'
    fields:
      message: "$1"
    level: warning
```

**验收标准：**
- 解析规则通过 YAML 配置，不需要改代码即可支持新格式
- 正则表达式中的命名捕获组映射到字段
- 输出解析后的事件列表，每个事件包含：时间（如有）、级别、类型、字段

### REQ-5.3 硬件日志解析 [P1]

**描述：** 解析 FPGA/芯片产生的硬件级日志。

**典型硬件日志内容：**
- 寄存器读写记录
- 中断事件
- DMA 传输状态
- 内存访问模式

**解析规则：** 与 REQ-5.2 共用规则引擎，通过不同的 rules YAML 文件配置。

**验收标准：**
- 支持二进制日志转文本后解析（转文本工具由用户提供）
- 解析规则文件按硬件平台分别配置

### REQ-5.4 可配置解析规则 [P1]

**描述：** 日志解析规则通过 YAML 配置文件管理，支持用户自行扩展。

**规则文件组织：**

```
logparser/
├── rules/
│   ├── framework.yaml      # 测试框架日志规则
│   ├── simulator.yaml      # 仿真器日志规则
│   ├── fpga.yaml           # FPGA日志规则
│   └── custom/             # 用户自定义规则
│       └── my_format.yaml
└── ...
```

**规则结构：**

```yaml
name: rule_set_name
version: "1.0"
description: "规则集描述"
log_format: text              # text / json / csv
timestamp_pattern: '\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]'
patterns:
  - name: pattern_name
    pattern: 'regex pattern'
    fields:
      field_name: "$capture_group"
    level: error | warning | info
    category: optional_category
```

**验收标准：**
- 新增日志格式只需添加 YAML 规则文件
- 规则文件有版本号，避免格式不兼容
- 正则表达式语法错误在加载时报错
- 提供规则测试命令：`aitf log test-rule --rule my.yaml --input sample.log`

### REQ-5.5 错误摘要与告警 [P2]

**描述：** 自动汇总日志中的错误和警告，生成摘要。

**摘要内容：**
- 错误分类统计（按 error pattern 分组计数）
- 首次出现和最后出现的时间
- 高频错误 Top N
- 严重错误高亮

**输出格式：**

```
=== Log Analysis Summary ===
File: sim_20260227.log (12,345 lines)
Duration: 14:30:00 - 14:35:00

Errors (5):
  [3x] DMA timeout at various cycles (first: cycle 1000, last: cycle 8000)
  [1x] ASSERTION FAILED at addr 0xFF00
  [1x] Memory access violation at addr 0x1000

Warnings (12):
  [8x] register value mismatch
  [4x] clock frequency deviation

Performance:
  Total cycles: 50000
  Throughput: 128 GOPS
```

**验收标准：**
- 摘要支持终端输出和 JSON 格式
- 严重错误（CRASH、ASSERTION）置顶显示
- 摘要数据可推送到 Dashboard 展示

## 3. 技术选型

| 决策 | 选型 | 备选 | 理由 |
|------|------|------|------|
| 正则引擎 | Python re | regex | 标准库足够，无额外依赖 |
| 规则配置 | YAML | JSON, DSL | 可读性好，支持注释 |
| 日志读取 | 逐行流式读取 | 全量加载 | 支持大文件，内存可控 |
| 输出格式 | JSON + rich终端 | 纯文本 | 结构化存储+美观终端输出 |

## 4. 数据模型

```python
@dataclass
class ParseRule:
    """单条解析规则"""
    name: str
    pattern: str            # 正则表达式
    fields: dict[str, str]  # 字段名 -> 捕获组映射
    level: str              # error / warning / info
    category: str | None = None

@dataclass
class RuleSet:
    """规则集"""
    name: str
    version: str
    description: str
    log_format: str         # text / json / csv
    timestamp_pattern: str | None
    patterns: list[ParseRule]

@dataclass
class LogEvent:
    """解析出的日志事件"""
    timestamp: datetime | None
    level: str
    rule_name: str
    fields: dict[str, str]
    raw_line: str
    line_number: int

@dataclass
class LogSummary:
    """日志分析摘要"""
    file_path: str
    total_lines: int
    time_range: tuple[datetime, datetime] | None
    events: list[LogEvent]
    error_count: int
    warning_count: int
    error_groups: dict[str, int]    # rule_name -> count
    performance: dict[str, str]     # key -> value
```

## 5. 对外接口

```python
class LogParser:
    def __init__(self, ruleset: RuleSet): ...
    def parse_file(self, path: str) -> list[LogEvent]: ...
    def parse_stream(self, stream: IO) -> Iterator[LogEvent]: ...
    def summarize(self, events: list[LogEvent]) -> LogSummary: ...

class RuleLoader:
    def load(self, path: str) -> RuleSet: ...
    def validate(self, ruleset: RuleSet) -> list[str]: ...  # 返回错误列表
```

## 6. 依赖

- REQ-6（执行框架）：执行过程产生日志
- REQ-7（Dashboard）：摘要数据在页面展示
