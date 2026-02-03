# REQ-EFF 效率提升需求

---
id: REQ-EFF
title: 效率提升需求
priority: P1
status: draft
parent: REQ-SYS
---

## 概述

提供开发效率提升工具，重点支持VSCode集成，通过右键菜单、快捷命令等方式简化日常开发操作。

**设计原则：**
- 减少重复操作，一键完成常见任务
- 与代码质量检查工具无缝集成
- 支持远程开发场景

---

## REQ-EFF-001 VSCode插件集成

---
id: REQ-EFF-001
title: VSCode插件集成
priority: P0
status: draft
parent: REQ-EFF
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
// .vscode/settings.json
{
  "aitestframework.remoteSshHost": "dev@192.168.1.100",
  "aitestframework.remotePath": "/home/dev/project",
  "aitestframework.qualityCheck.enabled": true,
  "aitestframework.qualityCheck.onSave": true
}
```

### 验收标准

1. 插件可通过 .vsix 安装
2. 支持本地和Remote-SSH模式
3. 配置项可在设置界面修改

---

## REQ-EFF-002 右键菜单扩展

---
id: REQ-EFF-002
title: 右键菜单扩展
priority: P0
status: draft
parent: REQ-EFF
---

### 描述

在VSCode编辑器和文件资源管理器中添加右键菜单项。

### 编辑器右键菜单

```
┌─────────────────────────────┐
│ AI Test Framework           │
│ ├── Run This Test          │ ← 光标在测试函数内
│ ├── Debug This Test        │
│ ├── Run Tests in File      │
│ ├── Check Code Quality     │
│ └── Generate Mock          │ ← 光标在函数调用处
└─────────────────────────────┘
```

### 文件资源管理器右键菜单

```
┌─────────────────────────────┐
│ AI Test Framework           │
│ ├── Run Tests              │
│ ├── Check Quality          │
│ ├── New Test Case          │
│ ├── New Test Module        │
│ └── Refresh Dependencies   │
└─────────────────────────────┘
```

### 验收标准

1. 菜单项根据上下文智能显示/隐藏
2. 操作结果在输出面板显示
3. 支持快捷键绑定

---

## REQ-EFF-003 代码片段模板

---
id: REQ-EFF-003
title: 代码片段模板
priority: P1
status: draft
parent: REQ-EFF
---

### 描述

提供符合代码规范的代码片段模板。

### 模板列表

| 前缀 | 描述 | 生成内容 |
|------|------|----------|
| `tfunc` | 测试函数 | 完整测试用例结构 |
| `tstru` | 结构体 | 含SEC标记的结构体 |
| `tenum` | 枚举 | 含类型别名的枚举 |
| `terr` | 错误码 | ERR_MODULE_XXXX格式 |
| `tret` | 返回检查 | RET_IF系列宏 |
| `tmock` | Mock函数 | Mock上下文结构 |

### 模板示例

```json
{
  "Test Function": {
    "prefix": "tfunc",
    "body": [
      "SEC_DDR_TEXT VOID TEST_${1:Module}_${2:FunctionName}(VOID)",
      "{",
      "    /* Arrange */",
      "    $3",
      "",
      "    /* Act */",
      "    $4",
      "",
      "    /* Assert */",
      "    $5",
      "}"
    ]
  }
}
```

### 验收标准

1. 模板符合coding_style.md规范
2. 支持Tab跳转和占位符
3. 自动生成唯一错误码

---

## REQ-EFF-004 快捷命令面板

---
id: REQ-EFF-004
title: 快捷命令面板
priority: P1
status: draft
parent: REQ-EFF
---

### 描述

在VSCode命令面板(Ctrl+Shift+P)中提供快捷命令。

### 命令列表

| 命令 | 描述 |
|------|------|
| `AITF: Run All Tests` | 运行所有测试 |
| `AITF: Run Current File Tests` | 运行当前文件测试 |
| `AITF: Check All Quality` | 全量代码质量检查 |
| `AITF: Check Current File` | 检查当前文件 |
| `AITF: Refresh Dependencies` | 刷新依赖 |
| `AITF: Install Dependencies` | 安装依赖 |
| `AITF: Generate Error Code` | 生成新错误码 |
| `AITF: Open Test Report` | 打开测试报告 |

### 验收标准

1. 命令支持模糊搜索
2. 常用命令支持快捷键
3. 命令执行状态有提示

---

## REQ-EFF-005 一键测试执行

---
id: REQ-EFF-005
title: 一键测试执行
priority: P0
status: draft
parent: REQ-EFF
---

### 描述

支持多种方式快速执行测试。

### 执行方式

| 方式 | 触发 | 作用域 |
|------|------|--------|
| CodeLens | 点击函数上方 "Run Test" | 单个测试 |
| 快捷键 | Ctrl+Shift+T | 当前文件所有测试 |
| 右键 | Run This Test | 光标处测试 |
| 资源管理器 | 右键 Run Tests | 选中文件/目录 |

### 远程执行

```yaml
# 本地开发，远程执行
1. VSCode通过Remote-SSH连接远程服务器
2. 测试命令在远程服务器执行
3. 结果实时回显到VSCode终端
4. 测试报告可在本地浏览器查看
```

### 验收标准

1. 支持运行和调试模式
2. 测试输出实时显示
3. 失败用例可点击跳转

---

## REQ-EFF-006 问题导航跳转

---
id: REQ-EFF-006
title: 问题导航跳转
priority: P1
status: draft
parent: REQ-EFF
---

### 描述

支持从问题列表、测试报告、错误日志跳转到源码。

### 跳转场景

| 场景 | 来源 | 目标 |
|------|------|------|
| 质量问题 | Problems面板 | 违规代码行 |
| 测试失败 | 测试输出 | 失败断言位置 |
| 错误码 | 日志/报告 | 错误码定义处 |
| Golden数据 | 比较报告 | 测试用例文件 |

### 实现方式

```
# 终端输出格式（可点击）
src/hal.c:45:12: error: 使用原生类型 int [TYP001]

# 测试报告链接
Test FAILED: TEST_Hal_RegWrite
  at src/test_hal.c:123
  Expected: 0, Actual: -1
```

### 验收标准

1. 终端输出支持Ctrl+Click跳转
2. HTML报告支持链接跳转
3. 错误码支持F12跳转定义
