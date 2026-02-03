# REQ-DEP 依赖管理需求

---
id: REQ-DEP
title: 依赖管理需求
priority: P1
status: draft
parent: REQ-SYS
---

## 概述

管理外部工具链（仿真器、编译器、ESL模型等）的版本依赖。

**设计原则：**
- 只需配置一整套依赖版本
- 每次刷新时重新生成依赖关系表
- 保持简单，不做复杂的版本兼容性计算

---

## REQ-DEP-001 依赖配置

---
id: REQ-DEP-001
title: 依赖配置
priority: P0
status: draft
parent: REQ-DEP
---

### 描述

定义当前使用的一整套依赖版本。

### 配置格式

```yaml
# deps/config.yaml
version: "1.0"
generated_at: "2026-02-03T10:00:00Z"

dependencies:
  simulator:
    version: "2.0.0"
    url: "https://release.example.com/sim/npu-sim-2.0.0.tar.gz"
    sha256: "abc123..."

  compiler:
    version: "2.1.0"
    url: "https://release.example.com/compiler/npu-cc-2.1.0.tar.gz"
    sha256: "def456..."

  esl_model:
    version: "1.5.0"
    url: "https://release.example.com/esl/npu-esl-1.5.0.tar.gz"
    sha256: "ghi789..."
```

### 验收标准

1. 单一配置文件定义所有依赖
2. 包含下载URL和SHA256校验
3. 配置文件纳入版本控制

---

## REQ-DEP-002 依赖刷新

---
id: REQ-DEP-002
title: 依赖关系刷新
priority: P0
status: draft
parent: REQ-DEP
---

### 描述

刷新依赖配置时重新生成依赖关系表。

### 刷新流程

```
1. 编辑 deps/config.yaml 更新版本
2. 执行 deps refresh 命令
3. 生成新的依赖关系表 deps/resolved.yaml
4. 下载并校验新版本
```

### 生成的关系表

```yaml
# deps/resolved.yaml (自动生成)
resolved_at: "2026-02-03T10:30:00Z"
source: "deps/config.yaml"

components:
  simulator:
    version: "2.0.0"
    sha256: "abc123..."
    install_path: "build/toolchain/simulator"
    status: installed

  compiler:
    version: "2.1.0"
    sha256: "def456..."
    install_path: "build/toolchain/compiler"
    status: installed
```

### 验收标准

1. 刷新时重新生成完整关系表
2. 关系表记录安装状态
3. 支持增量更新（只下载变更的组件）

---

## REQ-DEP-003 命令行接口

---
id: REQ-DEP-003
title: 命令行接口
priority: P0
status: draft
parent: REQ-DEP
---

### 描述

依赖管理CLI工具。

### 命令设计

```bash
# 刷新并安装
deps refresh              # 刷新依赖关系表
deps install              # 安装所有依赖

# 查询
deps list                 # 列出已安装
deps status               # 检查安装状态

# 清理
deps clean                # 清理安装目录
```

### 验收标准

1. 安装带进度显示
2. SHA256校验失败则终止
3. 支持离线安装（本地缓存）
