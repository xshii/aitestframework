# REQ-TMT 用例管理需求

---
id: REQ-TMT
title: 用例管理需求
priority: P0
status: draft
parent: REQ-SYS
depends:
  - REQ-FWK
---

## 概述

测试用例的组织、筛选、配置和平台映射。

---

## REQ-TMT-001 用例组织结构

---
id: REQ-TMT-001
title: 用例目录结构
priority: P0
status: draft
parent: REQ-TMT
---

### 描述

按功能而非平台组织用例。

### 目录结构

```
tests/
├── unit/                    # 单元测试
│   ├── framework/           # 框架自身测试
│   └── model/               # 参考模型测试
│
├── functional/              # 功能测试
│   ├── sanity/              # 冒烟测试
│   ├── matmul/              # 矩阵乘法
│   ├── conv/                # 卷积
│   ├── activation/          # 激活函数
│   └── precision/           # 精度测试
│
├── performance/             # 性能测试
│   ├── throughput/          # 吞吐量
│   ├── latency/             # 延迟
│   └── bandwidth/           # 带宽
│
├── stress/                  # 压力测试
│   ├── long_run/            # 长时间运行
│   └── concurrent/          # 并发测试
│
├── e2e/                     # 端到端测试
│   ├── resnet/
│   └── transformer/
│
└── testcfg/                 # 测试配置
    ├── platform_mapping.yaml
    └── test_execution.yaml
```

### 验收标准

1. 按功能分类，不按平台分
2. 目录层级不超过3级
3. 命名规范：`test_<功能>_<场景>.c`

---

## REQ-TMT-002 平台映射

---
id: REQ-TMT-002
title: 用例平台映射
priority: P0
status: draft
parent: REQ-TMT
---

### 描述

配置用例在哪些平台运行。

### 配置格式

```yaml
# tests/testcfg/platform_mapping.yaml

# 默认：所有用例在所有平台运行
default_platforms: [linux_ut, linux_st, simulator, esl, fpga, chip]

# 用例级别覆盖
testcases:
  "performance/*":
    platforms: [esl, fpga, chip]
    skip_reason: "性能测试需要真实时序"

  "performance/power/*":
    platforms: [chip]
    skip_reason: "功耗测试只能在芯片"

  "stress/*":
    skip_platforms: [linux_ut]
    skip_reason: "压力测试需要完整环境"

# 阶段定义
stages:
  linux_ut:
    platforms: [linux_ut]
    includes:
      - "unit/*"
      - "functional/sanity/*"

  functional:
    platforms: [simulator]
    includes:
      - "functional/*"
      - "e2e/*"
```

### 配置优先级

当同一用例出现在多处配置时，按以下优先级（高→低）：

```
1. 代码内定义 (.platforms = "chip")     ← 最高
2. testcases 精确匹配 ("test_power.c")
3. testcases 通配符 ("performance/*")
4. stages 配置
5. default_platforms                     ← 最低
```

**示例**：
```yaml
default_platforms: [linux_ut, simulator, chip]

testcases:
  "performance/*":
    platforms: [esl, fpga, chip]  # 覆盖default

stages:
  linux_ut:
    includes: ["unit/*"]  # 不影响testcases配置
```

用例 `performance/test_latency.c`：
- 最终平台 = `[esl, fpga, chip]`（testcases覆盖default）

### 验收标准

1. 支持通配符匹配（`*`匹配任意字符，`**`匹配多级目录）
2. 支持include和exclude
3. 运行时自动跳过不适用平台
4. 跳过时显示原因
5. 配置冲突时按优先级解决

---

## REQ-TMT-003 用例标签

---
id: REQ-TMT-003
title: 用例标签系统
priority: P0
status: draft
parent: REQ-TMT
---

### 描述

灵活的标签系统用于筛选用例。

### 标签定义

```c
/* 在代码中定义 */
TEST_CASE_EX(matmul, basic,
    .tags = "smoke,p0,matmul,quick"
)
{ ... }

/* 或在配置中定义 */
```

```yaml
# tests/testcfg/tags.yaml
tag_definitions:
  smoke: "冒烟测试，快速验证"
  p0: "最高优先级"
  p1: "高优先级"
  slow: "耗时较长"
  flaky: "不稳定，可能失败"

testcase_tags:
  "functional/matmul/*":
    - matmul
    - compute
  "e2e/*":
    - slow
    - e2e
```

### 命令行使用

```bash
# 运行smoke标签
./test_runner --tags smoke

# 运行p0且非slow
./test_runner --tags "p0 & !slow"

# 运行matmul或conv
./test_runner --tags "matmul | conv"
```

### 验收标准

1. 支持多标签
2. 支持标签表达式（AND/OR/NOT）
3. 代码和配置两种定义方式
4. 标签可有描述

---

## REQ-TMT-004 测试执行配置

---
id: REQ-TMT-004
title: 执行配置
priority: P1
status: draft
parent: REQ-TMT
---

### 描述

配置用例的执行环境和检查方式。

### 配置格式

```yaml
# tests/testcfg/test_execution.yaml

# 启动流程定义
setup_flows:
  minimal:
    steps: [log_init]

  standard:
    steps: [log_init, framework_init, load_golden]

  full:
    steps: [log_init, framework_init, load_golden, warmup]

# 检查器定义
checkers:
  exact_match:
    func: check_exact
  float_tolerance:
    func: check_float_near
    default_epsilon: 1e-5
  cosine_similarity:
    func: check_cosine
    default_threshold: 0.999

# 用例配置
testcase_configs:
  "unit/*":
    setup: minimal
    checker: exact_match
    timeout_ms: 1000

  "functional/matmul/*":
    setup: standard
    checker: float_tolerance
    checker_params:
      epsilon: 1e-4

  "functional/precision/*":
    setup: standard
    checker: float_tolerance
    checker_params:
      epsilon: 1e-6

  "e2e/*":
    setup: full
    checker: cosine_similarity
    timeout_ms: 60000
```

### 验收标准

1. 支持不同setup流程
2. 支持不同检查器
3. 支持检查器参数配置
4. 配置可继承和覆盖

---

## REQ-TMT-005 测试列表

---
id: REQ-TMT-005
title: 测试列表定义
priority: P0
status: draft
parent: REQ-TMT
---

### 描述

预定义的测试列表，用于不同场景。

### 列表格式

```yaml
# configs/testlists/sanity.yaml
name: sanity
description: "冒烟测试，每次提交运行"
timeout_minutes: 10

include:
  - "functional/sanity/*"
  - tags: smoke

exclude:
  - tags: flaky
```

```yaml
# configs/testlists/nightly.yaml
name: nightly
description: "每夜完整测试"
timeout_minutes: 240

include:
  - "unit/*"
  - "functional/*"
  - "e2e/*"

exclude:
  - "performance/*"
  - tags: manual
```

### 命令行使用

```bash
./test_runner --list sanity
./test_runner --list nightly
./test_runner --list configs/testlists/custom.yaml
```

### 验收标准

1. 支持按路径和标签include/exclude
2. 支持列表组合
3. 列表可指定总超时
4. 内置sanity/nightly/full列表

---

## REQ-TMT-006 用例元数据

---
id: REQ-TMT-006
title: 用例元数据管理
priority: P2
status: draft
parent: REQ-TMT
---

### 描述

管理用例的元数据信息。

### 元数据字段

```c
TEST_CASE_EX(matmul, basic,
    .tags = "smoke,matmul",
    .platforms = "all",
    .timeout_ms = 5000,
    .owner = "team-compute",
    .jira = "NPU-1234",
    .description = "基础2x2矩阵乘法测试"
)
```

### 验收标准

1. 支持owner、jira等扩展字段
2. 元数据可导出为报告
3. 支持按owner筛选
4. 支持与外部系统关联

---

## REQ-TMT-007 用例发现

---
id: REQ-TMT-007
title: 自动用例发现
priority: P0
status: draft
parent: REQ-TMT
---

### 描述

自动发现和注册测试用例。

### 发现机制

1. **编译时**：linker section收集TEST_CASE
2. **运行时**：扫描section获取用例列表
3. **配置合并**：与yaml配置合并

### 命令行

```bash
# 列出所有用例
./test_runner --list-tests

# 列出匹配的用例
./test_runner --list-tests --filter "matmul*"

# 导出用例列表
./test_runner --list-tests --format json > tests.json
```

### 验收标准

1. 新增.c文件自动发现
2. 支持列出用例不执行
3. 显示用例元数据
4. 支持多种输出格式