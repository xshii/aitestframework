# REQ-DVT 开发工具需求

---
id: REQ-DVT
title: 开发工具需求
priority: P1
status: draft
parent: REQ-SYS
context: Generic  # 通用域，可替换
---

## 概述

开发工具上下文（DevTool Context）提供开发效率提升工具，包括Python辅助工具和IDE集成。

**由原REQ-TLS（辅助工具）和REQ-EFF（效率提升）合并而来。**

### DDD定位

- **限界上下文**：通用域（Generic Domain）
- **聚合根**：无（工具类模块）
- **通信方式**：订阅核心域事件，不直接依赖
- **可替换性**：高（可用第三方工具替代）

### 事件订阅

```yaml
订阅事件:
  - GoldenUpdated → 触发数据同步
  - TestCompleted → 更新IDE状态
  - ReportGenerated → 刷新报告视图
```

### 目录结构

```
tools/
├── runner/           # 测试运行器
├── data/             # 数据生成工具
├── report/           # 报告生成工具
├── testmgmt/         # 用例管理Web
├── archive/          # 结果归档工具
└── vscode/           # VSCode插件
    ├── extension/    # 插件源码
    └── snippets/     # 代码片段
```

### 技术选型

详见 README.md "技术选型" 章节。本模块使用：

| 功能 | 库 |
|------|-----|
| 数值计算 | numpy |
| 模板渲染 | jinja2 |
| 终端UI | rich |
| 命令行 | argparse（标准库） |
| Web界面 | flask（可选） |

---

## REQ-DVT-001 Python测试运行器

---
id: REQ-DVT-001
title: Python测试运行器
priority: P0
status: draft
parent: REQ-DVT
subscribes:
  - TestCompleted
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
python -m tools.runner --platform simulator --list sanity
python -m tools.runner --platform simulator --parallel 4
python -m tools.runner --platform linux_ut --filter "matmul*"
python -m tools.runner --platform simulator --retry 2
```

### 事件集成

```python
# 订阅TestCompleted事件更新进度
class TestRunner:
    def on_test_completed(self, event: TestCompletedEvent):
        self.update_progress(event.test_name, event.result)
```

### 验收标准

1. 支持调用各平台C测试程序
2. 支持并行执行（进程级）
3. 支持失败重试
4. 结果输出JSON格式

---

## REQ-DVT-002 数据生成工具

---
id: REQ-DVT-002
title: 测试数据生成工具
priority: P0
status: draft
parent: REQ-DVT
subscribes:
  - GoldenUpdated
---

### 描述

基于Python参考模型生成测试输入和Golden数据。

### 命令行接口

```bash
python -m tools.data.generate --op matmul --config data_config.yaml
python -m tools.data.generate --batch testdata/generators/matmul_cases.yaml
python -m tools.data.yaml_to_c testdata/cases/matmul.yaml --output build/generated/
```

### 事件发布

```python
# 生成完成后发布GoldenUpdated事件
def generate_golden(op, config):
    # ... 生成数据 ...
    event_bus.publish(GoldenUpdatedEvent(
        dataset=op,
        version=config.version,
        checksum=compute_checksum(output_path)
    ))
```

### 验收标准

1. 支持各算子的数据生成
2. 输出符合Golden数据格式规范（REQ-MDL-001）
3. 生成完成后发布事件通知

---

## REQ-DVT-003 报告生成工具

---
id: REQ-DVT-003
title: 测试报告生成工具
priority: P0
status: draft
parent: REQ-DVT
subscribes:
  - ExecutionCompleted
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
python -m tools.report --input results.json --format html --output report.html
python -m tools.report --input results.json --format html,json,junit --output-dir reports/
```

### 验收标准

1. HTML报告可独立查看（内联CSS/JS）
2. 失败用例突出显示
3. JUnit XML兼容主流CI

---

## REQ-DVT-004 用例管理Web界面

---
id: REQ-DVT-004
title: 用例管理Web界面
priority: P2
status: draft
parent: REQ-DVT
---

### 描述

简洁的Web界面用于浏览和管理测试用例及结果。

### 技术选型

```
后端: Flask (轻量级)
前端: 简单HTML + Bootstrap
数据库: SQLite (本地部署)
```

### 验收标准

1. 单命令启动：`python -m tools.testmgmt.server`
2. 无需复杂部署
3. 响应式设计

---

## REQ-DVT-005 结果归档工具

---
id: REQ-DVT-005
title: 结果归档工具
priority: P1
status: draft
parent: REQ-DVT
---

### 描述

管理测试结果的归档、清理和恢复。

### 命令行接口

```bash
python -m tools.archive run --config configs/archive_policy.yaml
python -m tools.archive clean --keep-days 30
python -m tools.archive restore --archive /data/archive/2026-01-15.tar.gz
```

### 验收标准

1. 支持按策略自动归档
2. 支持压缩存储
3. 支持从归档恢复

---

## REQ-DVT-006 数据格式转换工具

---
id: REQ-DVT-006
title: 数据格式转换工具
priority: P1
status: draft
parent: REQ-DVT
---

### 描述

在YAML、JSON、C头文件等格式间转换测试数据。

### 命令行接口

```bash
python -m tools.data.yaml_to_c testdata/cases/matmul.yaml --output build/generated/
python -m tools.data.binview testdata/golden/matmul/case001.bin
```

### 验收标准

1. YAML转C头文件保持数据精度
2. 支持查看二进制文件元信息

---

## REQ-DVT-007 环境检查工具

---
id: REQ-DVT-007
title: 环境检查工具
priority: P1
status: draft
parent: REQ-DVT
---

### 描述

检查测试环境是否满足运行要求。

### 命令行接口

```bash
python -m tools.check_env
python -m tools.check_env --platform simulator
python -m tools.check_env --format json
```

### 验收标准

1. 检查所有必要依赖
2. 区分必须/可选项
3. 给出修复建议

---

## REQ-DVT-008 VSCode插件集成

---
id: REQ-DVT-008
title: VSCode插件集成
priority: P0
status: draft
parent: REQ-DVT
---

### 描述

开发VSCode插件，集成框架的常用功能。

### 功能列表

| 功能 | 描述 |
|------|------|
| 代码质量检查 | 实时显示检查结果 |
| 测试执行 | 运行/调试测试用例 |
| 依赖管理 | 查看/刷新依赖状态 |
| 错误码导航 | 点击错误码跳转定义 |
| 模板生成 | 快速创建用例/模块 |

### 插件配置

```json
{
  "aitestframework.remoteSshHost": "dev@192.168.1.100",
  "aitestframework.qualityCheck.enabled": true,
  "aitestframework.qualityCheck.onSave": true
}
```

### 验收标准

1. 插件可通过 .vsix 安装
2. 支持本地和Remote-SSH模式

---

## REQ-DVT-009 右键菜单与快捷命令

---
id: REQ-DVT-009
title: 右键菜单与快捷命令
priority: P0
status: draft
parent: REQ-DVT
---

### 描述

在VSCode中添加右键菜单项和命令面板命令。

### 编辑器右键菜单

```
AI Test Framework
├── Run This Test
├── Debug This Test
├── Run Tests in File
├── Check Code Quality
└── Generate Mock
```

### 命令面板

| 命令 | 描述 |
|------|------|
| `AITF: Run All Tests` | 运行所有测试 |
| `AITF: Run Current File Tests` | 运行当前文件测试 |
| `AITF: Check All Quality` | 全量代码质量检查 |
| `AITF: Refresh Dependencies` | 刷新依赖 |

### 验收标准

1. 菜单项根据上下文智能显示/隐藏
2. 常用命令支持快捷键

---

## REQ-DVT-010 代码片段模板

---
id: REQ-DVT-010
title: 代码片段模板
priority: P1
status: draft
parent: REQ-DVT
---

### 描述

提供符合代码规范的代码片段模板。

### 模板列表

| 前缀 | 描述 |
|------|------|
| `tfunc` | 测试函数 |
| `tstru` | 结构体 |
| `tenum` | 枚举 |
| `terr` | 错误码 |
| `tmock` | Mock函数 |

### 验收标准

1. 模板符合coding_style.md规范
2. 支持Tab跳转和占位符

---

## REQ-DVT-011 一键测试执行

---
id: REQ-DVT-011
title: 一键测试执行
priority: P0
status: draft
parent: REQ-DVT
---

### 描述

支持多种方式快速执行测试。

### 执行方式

| 方式 | 触发 | 作用域 |
|------|------|--------|
| CodeLens | 点击函数上方 "Run Test" | 单个测试 |
| 快捷键 | Ctrl+Shift+T | 当前文件所有测试 |
| 右键 | Run This Test | 光标处测试 |

### 验收标准

1. 支持运行和调试模式
2. 测试输出实时显示
3. 失败用例可点击跳转

---

## REQ-DVT-012 问题导航跳转

---
id: REQ-DVT-012
title: 问题导航跳转
priority: P1
status: draft
parent: REQ-DVT
---

### 描述

支持从问题列表、测试报告、错误日志跳转到源码。

### 跳转场景

| 场景 | 来源 | 目标 |
|------|------|------|
| 质量问题 | Problems面板 | 违规代码行 |
| 测试失败 | 测试输出 | 失败断言位置 |
| 错误码 | 日志/报告 | 错误码定义处 |

### 验收标准

1. 终端输出支持Ctrl+Click跳转
2. HTML报告支持链接跳转
