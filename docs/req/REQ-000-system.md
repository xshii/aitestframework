# REQ-000 系统级需求

---
id: REQ-000
title: 系统级需求
priority: P0
status: draft
---

## 概述

AI芯片应用侧验证框架的系统级需求，定义项目边界、总体目标和约束。

---

## REQ-001 项目定位

---
id: REQ-001
title: 验证框架定位
priority: P0
status: draft
parent: REQ-000
---

### 描述

本框架是**验证工具**，不是被测对象。被测对象（HAL、Driver、算子）由外部仓库提供。

### 边界定义

| 范围 | 包含 | 不包含 |
|------|------|--------|
| 本仓库 | 测试框架、参考模型、用例管理、结果管理、CI/CD | HAL实现、Driver实现、算子实现 |
| 外部 | 被测代码、工具链二进制 | - |

### 验收标准

1. 框架代码与被测代码完全解耦
2. 通过配置指定被测代码路径/接口
3. 框架可独立编译和运行（使用桩/Mock）

---

## REQ-002 语言约束

---
id: REQ-002
title: 编程语言约束
priority: P0
status: draft
parent: REQ-000
---

### 描述

测试框架核心用C语言，辅助工具用Python。

### 验收标准

| 组件 | 语言 | 标准/版本 | 外部依赖 |
|------|------|-----------|----------|
| 测试框架 | C | C99 | 无 |
| 测试用例 | C | C99 | 无 |
| C参考模型 | C | C99 | 无 |
| Python参考模型 | Python | 3.8+ | numpy |
| 辅助工具 | Python | 3.8+ | 见requirements.txt |

---

## REQ-003 跨平台验证

---
id: REQ-003
title: 跨平台验证支持
priority: P0
status: draft
parent: REQ-000
---

### 描述

同一套测试用例能在不同验证平台运行。

### 支持平台

| 平台 | 环境 | 用途 | 速度 |
|------|------|------|------|
| LinuxUT | Host Linux + Mock | 单元测试 | 最快 |
| LinuxST | Host Linux + Stub | 系统测试 | 快 |
| Simulator | 功能仿真器 | 功能验证 | 中 |
| ESL | 性能模型 | 性能验证 | 中 |
| FPGA | FPGA原型 | 原型验证 | 慢 |
| Chip | 真实芯片 | 硅后验证 | 慢 |

### 验收标准

1. 测试用例代码不包含平台特定逻辑
2. 通过配置文件指定用例在哪些平台运行
3. 框架自动跳过不适用当前平台的用例

---

## REQ-004 验证流水线

---
id: REQ-004
title: 验证流水线阶段
priority: P0
status: draft
parent: REQ-000
depends:
  - REQ-003
---

### 描述

验证分阶段执行，逐级门控。

### 阶段定义

```
提交 → LinuxUT/ST → Functional → Performance → Prototype
         (分钟)       (小时)        (天)         (周)
```

| 阶段 | 触发 | 平台 | 目标 |
|------|------|------|------|
| LinuxUT/ST | 每次提交 | Host | 快速门控 |
| Functional | 每夜构建 | Simulator | 功能正确性 |
| Performance | 每周构建 | ESL | 性能指标 |
| Prototype | 里程碑 | FPGA/Chip | 硬件实测 |

### 验收标准

1. 前一阶段失败则阻塞后续阶段
2. 可配置跳过某些阶段
3. 阶段结果可追溯

---

## REQ-005 可扩展性

---
id: REQ-005
title: 框架可扩展性
priority: P1
status: draft
parent: REQ-000
---

### 描述

框架应易于扩展。

### 验收标准

1. 添加新测试用例：只需新增.c文件，无需修改框架
2. 添加新平台：实现平台适配层接口即可
3. 添加新算子参考模型：实现标准接口即可
4. 添加新报告格式：实现报告生成接口即可

---

## REQ-006 文档要求

---
id: REQ-006
title: 文档要求
priority: P1
status: draft
parent: REQ-000
---

### 描述

提供完整的使用和开发文档。

### 验收标准

1. README：快速开始指南
2. 架构文档：整体设计说明
3. 用例编写指南：如何编写测试用例
4. 平台适配指南：如何适配新平台
5. API文档：框架接口说明
