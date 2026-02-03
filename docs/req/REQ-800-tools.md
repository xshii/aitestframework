# REQ-TLS 辅助工具需求

---
id: REQ-TLS
title: 辅助工具需求
priority: P1
status: draft
parent: REQ-SYS
depends:
  - REQ-MDL
  - REQ-RST
---

## 概述

Python实现的辅助工具集，包括测试运行器、数据生成、报告生成、用例管理Web界面等。

```
tools/
├── runner/           # 测试运行器
├── data/             # 数据生成工具
├── report/           # 报告生成工具
├── testmgmt/         # 用例管理Web
└── archive/          # 结果归档工具
```

---

## REQ-TLS-001 Python测试运行器

---
id: REQ-TLS-001
title: Python测试运行器
priority: P0
status: draft
parent: REQ-TLS
---

### 描述

Python封装的测试运行器，提供比C运行器更丰富的功能。

### 功能

1. **编排执行**：调用C测试程序，收集结果
2. **并行执行**：多进程并行运行测试
3. **结果聚合**：合并多次/多平台结果
4. **重试机制**：失败用例自动重试

### 命令行接口

```bash
# 基本运行
python -m tools.runner --platform simulator --list sanity

# 并行运行
python -m tools.runner --platform simulator --parallel 4

# 指定用例
python -m tools.runner --platform linux_ut --filter "matmul*"

# 失败重试
python -m tools.runner --platform simulator --retry 2

# 输出格式
python -m tools.runner --platform simulator --output json --output-file results.json
```

### Python接口

```python
from tools.runner import TestRunner

runner = TestRunner(
    platform="simulator",
    test_binary="build/bin/simulator/test_runner",
    config_path="configs/platforms/simulator.yaml"
)

# 运行测试
results = runner.run(
    testlist="sanity",
    parallel=4,
    timeout=3600,
    retry_failed=2
)

# 获取结果
print(f"Passed: {results.passed}, Failed: {results.failed}")
for test in results.failed_tests:
    print(f"  {test.name}: {test.error}")
```

### 验收标准

1. 支持调用各平台C测试程序
2. 支持并行执行（进程级）
3. 支持失败重试
4. 结果输出JSON格式
5. 支持超时控制

---

## REQ-TLS-002 数据生成工具

---
id: REQ-TLS-002
title: 测试数据生成工具
priority: P0
status: draft
parent: REQ-TLS
depends:
  - REQ-MDL-002
---

### 描述

基于Python参考模型生成测试输入和Golden数据。

### 功能

1. **随机数据生成**：生成随机输入张量
2. **Golden计算**：调用参考模型计算期望输出
3. **批量生成**：根据配置批量生成测试数据
4. **数据导出**：导出为二进制文件

### 命令行接口

```bash
# 生成单个算子数据
python -m tools.data.generate --op matmul --config data_config.yaml

# 批量生成
python -m tools.data.generate --batch testdata/generators/matmul_cases.yaml

# 指定输出目录
python -m tools.data.generate --op conv2d --output testdata/golden/conv2d/
```

### 配置文件

```yaml
# testdata/generators/matmul_cases.yaml
operator: matmul
dtype: float32

cases:
  - name: small_2x2
    params: {M: 2, N: 2, K: 2}

  - name: medium_128
    params: {M: 128, N: 128, K: 128}

  - name: large_1024
    params: {M: 1024, N: 1024, K: 1024}

# 随机生成
random_cases:
  count: 10
  M: {min: 16, max: 512, step: 16}
  N: {min: 16, max: 512, step: 16}
  K: {min: 16, max: 512, step: 16}
```

### Python接口

```python
from tools.data import DataGenerator
from pymodel.ops import matmul

gen = DataGenerator(output_dir="testdata")

# 生成单个用例
inputs, golden = gen.generate_case(
    op=matmul,
    params={"M": 128, "N": 128, "K": 128},
    dtype="float32"
)

# 保存为二进制
gen.save_binary("inputs/matmul/case001_A.bin", inputs["A"])
gen.save_binary("inputs/matmul/case001_B.bin", inputs["B"])
gen.save_binary("golden/matmul/case001_C.bin", golden)
```

### YAML到C头文件转换

```bash
# 将YAML数据配置转换为C头文件（供C测试引用）
python -m tools.data.yaml_to_c testdata/cases/matmul.yaml \
    --output build/generated/matmul_cases.h
```

### 验收标准

1. 支持各算子的数据生成
2. 支持批量生成配置
3. 支持随机参数生成
4. 输出符合Golden数据格式规范（REQ-MDL-004）
5. 支持YAML转C头文件

---

## REQ-TLS-003 报告生成工具

---
id: REQ-TLS-003
title: 测试报告生成工具
priority: P0
status: draft
parent: REQ-TLS
depends:
  - REQ-RST-004
---

### 描述

生成多种格式的测试报告。

### 支持格式

| 格式 | 用途 | 文件 |
|------|------|------|
| HTML | 人工查看 | report.html |
| JSON | 工具解析 | report.json |
| JUnit XML | CI集成 | report.xml |
| Markdown | 文档嵌入 | report.md |

### 命令行接口

```bash
# 生成HTML报告
python -m tools.report --input results.json --format html --output report.html

# 生成多种格式
python -m tools.report --input results.json --format html,json,junit --output-dir reports/

# 包含趋势图
python -m tools.report --input results.json --format html --trend-data history/ --output report.html
```

### HTML报告内容

```
┌─────────────────────────────────────────────────────────┐
│  Test Report - 2026-02-02 Nightly                       │
├─────────────────────────────────────────────────────────┤
│  Summary                                                 │
│  ├─ Total: 150                                          │
│  ├─ Passed: 145 (96.7%)                                 │
│  ├─ Failed: 3                                           │
│  ├─ Skipped: 2                                          │
│  └─ Duration: 5m 23s                                    │
├─────────────────────────────────────────────────────────┤
│  Failed Tests                                           │
│  ├─ functional.matmul.large                             │
│  │   └─ cosine_sim 0.985 < threshold 0.999              │
│  ├─ functional.conv.stride2                             │
│  │   └─ timeout after 10000ms                           │
├─────────────────────────────────────────────────────────┤
│  Trend (Last 7 Days)                                    │
│  [趋势图表]                                              │
├─────────────────────────────────────────────────────────┤
│  All Tests (按Suite分组，可展开)                         │
└─────────────────────────────────────────────────────────┘
```

### Python接口

```python
from tools.report import ReportGenerator

gen = ReportGenerator()

# 加载结果
gen.load_results("results.json")

# 加载历史数据（可选，用于趋势）
gen.load_history("history/")

# 生成报告
gen.generate_html("report.html")
gen.generate_junit("report.xml")
```

### 验收标准

1. HTML报告可独立查看（内联CSS/JS）
2. 失败用例突出显示
3. 支持按Suite分组
4. 支持趋势图表（可选）
5. JUnit XML兼容主流CI

---

## REQ-TLS-004 用例管理Web界面

---
id: REQ-TLS-004
title: 用例管理Web界面
priority: P2
status: draft
parent: REQ-TLS
---

### 描述

简洁的Web界面用于浏览和管理测试用例及结果。

### 功能

1. **用例浏览**：按目录/标签/平台浏览用例
2. **结果查看**：查看历史执行结果
3. **趋势展示**：通过率/耗时趋势图
4. **Flaky管理**：标记和管理不稳定用例

### 技术选型

```
后端: Flask (轻量级)
前端: 简单HTML + Bootstrap (无复杂框架)
数据库: SQLite (本地部署)
```

### 页面设计

```
┌─────────────────────────────────────────────────────────┐
│  NPU Test Management                    [Search] [User] │
├───────────┬─────────────────────────────────────────────┤
│ Navigation│  Dashboard                                   │
│           │  ┌─────────────────────────────────────────┐│
│ Dashboard │  │  Pass Rate: 96.7%  │  Total: 150       ││
│ Tests     │  │  Failed: 3         │  Flaky: 2         ││
│ Results   │  └─────────────────────────────────────────┘│
│ Trends    │                                             │
│ Settings  │  Recent Executions                          │
│           │  ┌─────────────────────────────────────────┐│
│           │  │ 2026-02-02 Nightly  │ 145/150 │ PASS   ││
│           │  │ 2026-02-01 Nightly  │ 148/150 │ PASS   ││
│           │  │ 2026-01-31 Nightly  │ 140/150 │ FAIL   ││
│           │  └─────────────────────────────────────────┘│
└───────────┴─────────────────────────────────────────────┘
```

### API设计

```python
# RESTful API
GET  /api/tests                    # 用例列表
GET  /api/tests/<id>               # 用例详情
GET  /api/results                  # 结果列表
GET  /api/results/<exec_id>        # 执行详情
GET  /api/trends?days=30           # 趋势数据
POST /api/tests/<id>/flaky         # 标记Flaky
```

### 验收标准

1. 单命令启动：`python -m tools.testmgmt.server`
2. 无需复杂部署（SQLite + Flask）
3. 支持用例浏览和搜索
4. 支持结果查看
5. 响应式设计（支持移动端）

---

## REQ-TLS-005 结果归档工具

---
id: REQ-TLS-005
title: 结果归档工具
priority: P1
status: draft
parent: REQ-TLS
depends:
  - REQ-RST-003
---

### 描述

管理测试结果的归档、清理和恢复。

### 功能

1. **自动归档**：按策略归档历史结果
2. **空间清理**：清理过期结果
3. **归档恢复**：从归档恢复结果查看

### 命令行接口

```bash
# 执行归档（按策略）
python -m tools.archive run --config configs/archive_policy.yaml

# 手动归档指定日期
python -m tools.archive create --date 2026-01-15 --output /data/archive/

# 清理过期
python -m tools.archive clean --keep-days 30

# 恢复归档
python -m tools.archive restore --archive /data/archive/2026-01-15.tar.gz --output build/results/

# 列出归档
python -m tools.archive list --path /data/archive/
```

### 归档策略配置

```yaml
# configs/archive_policy.yaml
archive:
  # 保留策略
  retention:
    recent_days: 30       # 最近30天全量保留
    weekly_keep: 12       # 保留12周的周报告
    monthly_keep: 12      # 保留12个月的月报告

  # 压缩
  compression:
    enabled: true
    format: tar.gz
    after_days: 7         # 7天后压缩

  # 路径
  paths:
    active: build/results/
    archive: /data/archive/testresults/

  # 特殊保留（不清理）
  keep_forever:
    - pattern: "*release*"
    - pattern: "*milestone*"
```

### Python接口

```python
from tools.archive import Archiver

archiver = Archiver(config_path="configs/archive_policy.yaml")

# 执行归档
archiver.run()

# 获取归档列表
archives = archiver.list_archives()
for a in archives:
    print(f"{a.date}: {a.size_mb}MB, {a.test_count} tests")

# 恢复
archiver.restore(archive_path, output_dir)
```

### 验收标准

1. 支持按策略自动归档
2. 支持压缩存储
3. 支持保留重要结果（release/milestone）
4. 支持从归档恢复
5. 提供空间占用统计

---

## REQ-TLS-006 YAML/JSON数据转换

---
id: REQ-TLS-006
title: 数据格式转换工具
priority: P1
status: draft
parent: REQ-TLS
---

### 描述

在YAML、JSON、C头文件等格式间转换测试数据。

### 功能

1. **YAML → C头文件**：测试参数转C数组
2. **JSON → Binary**：JSON描述转二进制数据
3. **Binary → JSON**：二进制数据转JSON（调试用）

### YAML到C转换

```bash
# 转换测试用例配置
python -m tools.data.yaml_to_c testdata/cases/matmul.yaml \
    --output build/generated/matmul_cases.h \
    --struct-name matmul_testcase_t
```

输入YAML：
```yaml
cases:
  - name: small
    M: 2
    N: 2
    K: 2
    tolerance: 1e-5
  - name: large
    M: 1024
    N: 1024
    K: 1024
    tolerance: 1e-4
```

输出C头文件：
```c
/* Auto-generated, do not edit */
static const matmul_testcase_t matmul_cases[] = {
    {.name = "small", .M = 2, .N = 2, .K = 2, .tolerance = 1e-5f},
    {.name = "large", .M = 1024, .N = 1024, .K = 1024, .tolerance = 1e-4f},
};
static const int matmul_cases_count = 2;
```

### 二进制数据工具

```bash
# 查看二进制文件信息
python -m tools.data.binview testdata/golden/matmul/case001.bin

# 输出：
# Magic: GOLD (0x474F4C44)
# Version: 2
# Dtype: float32
# Shape: [128, 128]
# Data size: 65536 bytes
# Checksum: 0xABCD1234

# 导出为文本（调试用）
python -m tools.data.bin2txt testdata/golden/matmul/case001.bin --output case001.txt
```

### 验收标准

1. YAML转C头文件保持数据精度
2. 支持查看二进制文件元信息
3. 支持二进制导出为可读文本
4. 转换错误有明确提示

---

## REQ-TLS-007 环境检查工具

---
id: REQ-TLS-007
title: 环境检查工具
priority: P1
status: draft
parent: REQ-TLS
---

### 描述

检查测试环境是否满足运行要求。

### 检查项

1. **Python环境**：版本、依赖包
2. **C工具链**：编译器、版本
3. **平台工具**：仿真器、FPGA驱动
4. **资源**：磁盘空间、内存

### 命令行接口

```bash
# 完整检查
python -m tools.check_env

# 指定平台检查
python -m tools.check_env --platform simulator

# 输出JSON
python -m tools.check_env --format json
```

### 输出示例

```
Environment Check Report
========================

[OK] Python: 3.10.5 (>= 3.8 required)
[OK] numpy: 1.24.0
[OK] pyyaml: 6.0

[OK] GCC: 11.3.0 (>= 9.0 required)
[OK] Make: 4.3

[OK] Simulator: 1.1.0 (path: /opt/toolchain/simulator)
[WARN] ESL: not installed (optional)

[OK] Disk space: 50GB available (>= 10GB required)
[OK] Memory: 16GB available (>= 8GB required)

Summary: 9 OK, 1 WARN, 0 FAIL
```

### 验收标准

1. 检查所有必要依赖
2. 区分必须/可选项
3. 给出修复建议
4. 支持JSON输出（供CI使用）
