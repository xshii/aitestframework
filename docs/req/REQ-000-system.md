# REQ-SYS 系统级需求

---
id: REQ-SYS
title: 系统级需求
priority: P0
status: draft
---

## 概述

AI芯片应用侧验证框架的系统级需求，定义项目边界、总体目标和约束。

---

## REQ-SYS-001 项目定位

---
id: REQ-SYS-001
title: 验证框架定位
priority: P0
status: draft
parent: REQ-SYS
---

### 描述

本框架是**验证工具**，不是被测对象。被测对象（HAL、Driver、算子）由外部仓库提供。

### 边界定义

| 范围 | 包含 | 不包含 |
|------|------|--------|
| 本仓库 | 测试框架、参考模型、用例管理、结果管理、CI/CD | HAL实现、Driver实现、算子实现 |
| 外部 | 被测代码、工具链二进制 | - |

### 被测代码接口规范

被测代码需要实现以下接口才能被本框架测试：

```c
/* 被测代码必须提供的头文件 */
#include "npu_hal.h"      /* HAL层接口 */
#include "npu_driver.h"   /* 驱动层接口 */
#include "npu_ops.h"      /* 算子接口 */
```

### 配置文件示例

```yaml
# configs/target.yaml - 指定被测代码
target:
  # 被测代码路径
  source_path: "/path/to/npu-sdk/src"
  include_path: "/path/to/npu-sdk/include"
  library_path: "/path/to/npu-sdk/lib"

  # 被测库
  libraries:
    - libnpu_hal.a
    - libnpu_driver.a

  # 接口头文件
  headers:
    hal: "npu_hal.h"
    driver: "npu_driver.h"
    ops: "npu_ops.h"
```

### 验收标准

1. 框架代码与被测代码完全解耦
2. 通过 `configs/target.yaml` 指定被测代码路径和接口
3. 框架可独立编译和运行（使用桩/Mock）
4. 被测代码只需实现规定的接口即可接入测试

---

## REQ-SYS-002 语言约束

---
id: REQ-SYS-002
title: 编程语言约束
priority: P0
status: draft
parent: REQ-SYS
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

## REQ-SYS-003 跨平台验证

---
id: REQ-SYS-003
title: 跨平台验证支持
priority: P0
status: draft
parent: REQ-SYS
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

### LinuxUT vs LinuxST 区别

| 特性 | LinuxUT (单元测试) | LinuxST (系统测试) |
|------|-------------------|-------------------|
| **测试粒度** | 单个函数/模块 | 多模块集成 |
| **外部依赖** | 全部Mock | 部分Stub + 部分真实 |
| **HAL层** | 纯内存模拟，无IO | 可有文件IO、网络 |
| **被测代码** | 隔离测试单个组件 | 测试组件间交互 |
| **数据规模** | 小规模 (<1KB) | 中等规模 (<1MB) |
| **典型用例** | 函数逻辑、边界条件 | API流程、错误处理 |

### Mock vs Stub 定义

```
Mock（模拟）:
- 完全在内存中模拟，无真实行为
- 可验证调用次数、参数
- 示例：mock_reg_read() 直接返回预设值

Stub（桩）:
- 简化实现，有基本行为逻辑
- 不验证调用，只提供功能
- 示例：stub_dma_transfer() 实际拷贝内存
```

### 验收标准

1. 测试用例代码不包含平台特定逻辑
2. 通过配置文件指定用例在哪些平台运行
3. 框架自动跳过不适用当前平台的用例

---

## REQ-SYS-004 验证流水线

---
id: REQ-SYS-004
title: 验证流水线阶段
priority: P0
status: draft
parent: REQ-SYS
depends:
  - REQ-SYS-003
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

## REQ-SYS-005 可扩展性

---
id: REQ-SYS-005
title: 框架可扩展性
priority: P1
status: draft
parent: REQ-SYS
---

### 描述

框架应易于扩展。

### 平台适配层接口

添加新平台需要实现以下函数：

```c
/* platform/<platform_name>/<platform>_adapter.c */

/* 必须实现 */
int platform_init(void);              /* 平台初始化 */
int platform_deinit(void);            /* 平台清理 */
int platform_run_test(test_case_t *tc); /* 执行单个测试 */

/* 可选实现 */
int platform_setup(void);             /* 测试前准备 */
int platform_teardown(void);          /* 测试后清理 */
void platform_log(int level, const char *fmt, ...); /* 日志输出 */
uint64_t platform_get_time_us(void);  /* 获取时间戳 */
```

### 参考模型标准接口

添加新算子参考模型需要实现：

```c
/* Python参考模型 (pymodel/ops/<op_name>.py) */
def forward(inputs: List[np.ndarray], params: dict) -> List[np.ndarray]:
    """算子前向计算"""
    pass

def generate_testcase(config: dict) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """生成测试用例 (inputs, expected_outputs)"""
    pass

/* C参考模型 (src/model/ref_<op_name>.c) */
int ref_<op_name>_f32(const float *input, float *output, const <op>_params_t *params);
int ref_<op_name>_f16(const uint16_t *input, uint16_t *output, const <op>_params_t *params);
```

### 报告生成接口

添加新报告格式需要实现：

```python
# tools/report/<format>_report.py
class <Format>ReportGenerator:
    def generate(self, results: dict, output_path: str) -> None:
        """生成报告文件"""
        pass
```

### 验收标准

1. 添加新测试用例：只需新增.c文件，无需修改框架
2. 添加新平台：实现上述平台适配层接口（3个必须 + 4个可选）
3. 添加新算子参考模型：实现 forward() 和 generate_testcase()
4. 添加新报告格式：继承 ReportGenerator 基类

---

## REQ-SYS-006 文档要求

---
id: REQ-SYS-006
title: 文档要求
priority: P1
status: draft
parent: REQ-SYS
---

### 描述

提供完整的使用和开发文档。

### 验收标准

1. README：快速开始指南
2. 架构文档：整体设计说明
3. 用例编写指南：如何编写测试用例
4. 平台适配指南：如何适配新平台
5. API文档：框架接口说明

---

## REQ-SYS-007 代码质量要求

---
id: REQ-SYS-007
title: 代码质量要求
priority: P0
status: draft
parent: REQ-SYS
---

### 描述

框架自身代码必须满足质量标准。

### Python代码质量

| 工具 | 要求 | 配置 |
|------|------|------|
| pylint | 评分 >= 9.0 | .pylintrc |
| pylint | 无 Error/Fatal 级别告警 | - |
| flake8 | 无告警 | .flake8 |
| black | 格式化通过 | pyproject.toml |
| mypy | 类型检查通过（可选） | mypy.ini |

### C代码质量

| 工具 | 要求 | 配置 |
|------|------|------|
| cppcheck | 无 error 级别告警 | - |
| clang-format | 格式化通过 | .clang-format |
| gcc -Wall -Werror | 编译无警告 | Makefile |

### 验收标准

1. CI流水线包含代码质量检查
2. 质量检查不通过阻塞合入
3. 提供pre-commit hook自动检查
4. 质量配置文件纳入版本控制

---

## REQ-SYS-008 测试覆盖率要求

---
id: REQ-SYS-008
title: 测试覆盖率要求
priority: P0
status: draft
parent: REQ-SYS
---

### 描述

框架自身代码的测试覆盖率要求。

### 覆盖率目标

| 代码类型 | 行覆盖率 | 分支覆盖率 | 工具 |
|----------|----------|------------|------|
| Python核心代码 | >= 80% | >= 70% | pytest-cov |
| Python工具代码 | >= 60% | >= 50% | pytest-cov |
| C测试框架 | >= 70% | >= 60% | gcov/lcov |

### 代码分类定义

```
Python核心代码（80%覆盖率要求）:
├── pymodel/ops/          # 参考模型算子
├── pymodel/quantize/     # 量化工具
├── pymodel/layers/       # 复合层
├── tools/runner/         # 测试运行器核心
└── deps/scripts/         # 依赖管理核心

Python工具代码（60%覆盖率要求）:
├── tools/report/         # 报告生成
├── tools/testmgmt/       # 用例管理Web
├── tools/archive/        # 归档工具
├── tools/data/           # 数据工具
├── cicd/                 # CI/CD脚本
└── scripts/              # 辅助脚本

C测试框架代码（70%覆盖率要求）:
├── src/framework/        # 测试框架核心
├── src/model/            # C参考模型
└── src/common/           # 通用工具
```

### 覆盖率排除

```ini
# .coveragerc
[run]
omit =
    */tests/*
    */test_*
    */__pycache__/*
    */migrations/*
```

### 验收标准

1. CI流水线生成覆盖率报告
2. 覆盖率不达标阻塞合入（可配置）
3. 覆盖率趋势可追踪
4. 支持覆盖率徽章展示

### 报告要求

```
Coverage Report:
-------------------------------------------------
Module                  Stmts   Miss  Cover
-------------------------------------------------
tools/runner            120     18    85%
tools/report            89      12    87%
pymodel/ops             156     31    80%
-------------------------------------------------
TOTAL                   365     61    83%
-------------------------------------------------

Branch Coverage: 72%
```
