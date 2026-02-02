# REQ-300 依赖管理需求

---
id: REQ-300
title: 依赖管理需求
priority: P1
status: draft
parent: REQ-000
---

## 概述

管理外部工具链（仿真器、编译器、ESL模型等）的版本和配套关系。

---

## REQ-301 依赖清单

---
id: REQ-301
title: 依赖清单定义
priority: P0
status: draft
parent: REQ-300
---

### 描述

定义所有外部依赖组件及其版本。

### 清单格式

```yaml
# deps/manifests/simulator.yaml
name: simulator
description: "NPU功能仿真器"

versions:
  "1.0.0":
    url: "https://release.example.com/sim/npu-sim-1.0.0.tar.gz"
    sha256: "abc123..."
    release_date: "2025-06-01"
    min_os: "ubuntu20.04"

  "2.0.0":
    url: "https://release.example.com/sim/npu-sim-2.0.0.tar.gz"
    sha256: "def456..."
    release_date: "2026-01-01"
    min_os: "ubuntu22.04"
    requires:
      compiler: ">=2.0.0"
```

### 验收标准

1. 每个组件独立清单文件
2. 包含下载URL和SHA256校验
3. 支持版本间依赖声明
4. 支持平台要求声明

---

## REQ-302 配套关系矩阵

---
id: REQ-302
title: 版本配套关系
priority: P0
status: draft
parent: REQ-300
---

### 描述

定义组件间的版本配套关系。

### 配套矩阵

```yaml
# deps/compatibility/matrix.yaml
compatibility_groups:
  gen1:  # 第一代配套
    simulator: ["1.0.x", "1.1.x"]
    compiler: ["1.x"]
    os_sdk: ["1.x"]

  gen2:  # 第二代配套
    simulator: ["2.x"]
    compiler: ["2.x"]
    os_sdk: ["2.x"]

rules:
  - name: "sim2需要compiler2"
    when:
      simulator: ">=2.0.0"
    requires:
      compiler: ">=2.0.0"

  - name: "compiler2需要os_sdk2"
    when:
      compiler: ">=2.0.0"
    requires:
      os_sdk: ">=2.0.0"
```

### 验收标准

1. 支持配套组定义
2. 支持版本约束规则
3. 安装时自动检查兼容性
4. 不兼容时给出明确提示

---

## REQ-303 版本Profile

---
id: REQ-303
title: 预定义版本组合
priority: P1
status: draft
parent: REQ-300
---

### 描述

预定义经过验证的版本组合。

### Profile定义

```yaml
# deps/profiles/stable.yaml
name: stable
description: "生产环境推荐"
date: "2026-01-15"

components:
  simulator: "1.1.0"
  compiler: "1.5.0"
  os_sdk: "1.2.0"

validation:
  sanity: PASS
  nightly: PASS
  full: PASS
```

### 验收标准

1. 提供stable/latest/版本号等profile
2. 记录profile验证状态
3. 一键安装整个profile
4. 支持自定义profile

---

## REQ-304 锁定文件

---
id: REQ-304
title: 版本锁定
priority: P1
status: draft
parent: REQ-300
---

### 描述

锁定实际安装的版本，确保可复现。

### 锁定文件

```yaml
# deps/lock/deps.lock.yaml
generated_at: "2026-02-02T10:30:00Z"
profile: stable

resolved:
  simulator:
    version: "1.1.0"
    sha256: "def456..."
    install_path: "build/toolchain/simulator"

  compiler:
    version: "1.5.0"
    sha256: "..."
    install_path: "build/toolchain/compiler"
```

### 验收标准

1. 安装后自动生成锁定文件
2. 锁定文件纳入版本控制
3. 支持从锁定文件精确恢复
4. 检测锁定文件与实际不一致

---

## REQ-305 依赖管理命令

---
id: REQ-305
title: 命令行接口
priority: P0
status: draft
parent: REQ-300
depends:
  - REQ-301
  - REQ-302
  - REQ-303
  - REQ-304
---

### 描述

依赖管理CLI工具。

### 命令设计

```bash
# 安装
deps install                     # 安装锁定版本
deps install --profile stable    # 安装指定profile
deps install simulator==2.0.0    # 安装指定版本

# 检查
deps check                       # 检查兼容性
deps outdated                    # 检查过期版本

# 查询
deps list                        # 列出已安装
deps list --available            # 列出可用版本
deps info simulator              # 组件详情

# 更新
deps update                      # 更新锁定文件
deps upgrade simulator           # 升级组件
```

### 验收标准

1. 安装带进度显示
2. 安装失败可回滚
3. 支持离线安装（本地缓存）
4. 支持代理设置

---

## REQ-306 下载与校验

---
id: REQ-306
title: 安全下载
priority: P0
status: draft
parent: REQ-300
---

### 描述

安全下载和校验工具链。

### 验收标准

1. HTTPS下载
2. SHA256校验，不匹配则失败
3. 下载失败自动重试
4. 支持断点续传
5. 本地缓存避免重复下载