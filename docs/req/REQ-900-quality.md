# REQ-QCK 代码质量需求

---
id: REQ-QCK
title: 代码质量需求
priority: P1
status: draft
parent: REQ-SYS
---

## 概述

基于 `docs/coding_style.md` 定义的代码规范，提供静态分析工具进行自动化检查。

**设计原则：**
- 规则可配置，支持开关和严重级别
- CI集成，提交时自动检查
- 增量检查，只分析变更文件

---

## REQ-QCK-001 类型命名检查

---
id: REQ-QCK-001
title: 类型命名检查
priority: P1
status: draft
parent: REQ-QCK
---

### 描述

检查代码是否使用项目规定的类型别名。

### 检查规则

| 规则ID | 检查项 | 严重级别 |
|--------|--------|----------|
| TYP001 | 禁止原生类型 int/long/char等 | ERROR |
| TYP002 | 必须使用 INT8/INT32/UINT32等 | ERROR |
| TYP003 | 自定义类型需大驼峰+后缀 | WARNING |
| TYP004 | 枚举类型别名需 _ENUM_UINTx | WARNING |

### 验收标准

1. 检测到原生类型使用时报错
2. 支持白名单排除第三方头文件
3. 报告包含文件、行号、违规内容

---

## REQ-QCK-002 命名规范检查

---
id: REQ-QCK-002
title: 命名规范检查
priority: P1
status: draft
parent: REQ-QCK
---

### 描述

检查变量、函数、宏的命名是否符合规范。

### 检查规则

| 规则ID | 检查项 | 严重级别 |
|--------|--------|----------|
| NAM001 | 全局变量需 g_ 前缀 | ERROR |
| NAM002 | 局部变量需小驼峰 | WARNING |
| NAM003 | 对外函数需 MODULE_ 前缀 | WARNING |
| NAM004 | 宏常量需全大写 | WARNING |
| NAM005 | 避免 static 变量 | WARNING |

### 验收标准

1. 能区分全局变量和局部变量
2. 支持模块名白名单配置
3. 能识别对外接口（头文件声明）

---

## REQ-QCK-003 内存段标记检查

---
id: REQ-QCK-003
title: 内存段标记检查
priority: P1
status: draft
parent: REQ-QCK
---

### 描述

检查全局变量和函数是否有正确的内存段标记。

### 检查规则

| 规则ID | 检查项 | 严重级别 |
|--------|--------|----------|
| SEC001 | 全局变量需 SEC_xxx_BSS/DATA | ERROR |
| SEC002 | 函数需 SEC_xxx_TEXT 标记 | ERROR |
| SEC003 | const 数据需 SEC_xxx_RODATA | WARNING |
| SEC004 | 段标记格式需符合 SEC_<层级>_<类型> | ERROR |

### 验收标准

1. 检测缺失段标记的全局变量/函数
2. 验证段标记格式正确性
3. 支持排除特定文件（如平台无关代码）

---

## REQ-QCK-004 安全函数检查

---
id: REQ-QCK-004
title: 安全函数检查
priority: P1
status: draft
parent: REQ-QCK
---

### 描述

检查是否使用安全函数替代不安全函数。

### 检查规则

| 规则ID | 检查项 | 严重级别 |
|--------|--------|----------|
| SAF001 | 禁止 memset，使用 memset_s | ERROR |
| SAF002 | 禁止 memcpy，使用 memcpy_s | ERROR |
| SAF003 | 禁止 strcpy，使用 strcpy_s | ERROR |
| SAF004 | 禁止 sprintf，使用 snprintf_s | ERROR |
| SAF005 | 安全函数返回值需检查或(void) | WARNING |

### 验收标准

1. 检测不安全函数调用
2. 识别 (void) 显式忽略返回值
3. 检查 destMax != srcLen 时的返回值校验

---

## REQ-QCK-005 错误码检查

---
id: REQ-QCK-005
title: 错误码检查
priority: P1
status: draft
parent: REQ-QCK
---

### 描述

检查错误码定义和使用规范。

### 检查规则

| 规则ID | 检查项 | 严重级别 |
|--------|--------|----------|
| ERR001 | 错误码格式需 ERR_MODULE_XXXX | ERROR |
| ERR002 | 错误码值需全局唯一 | ERROR |
| ERR003 | 函数返回类型需 ERRNO_T | WARNING |
| ERR004 | 禁止魔鬼数字作为错误返回 | ERROR |

### 验收标准

1. 扫描所有错误码定义，检查唯一性
2. 检测硬编码错误返回值
3. 生成错误码使用报告

---

## REQ-QCK-006 判断宏使用检查

---
id: REQ-QCK-006
title: 判断宏使用检查
priority: P2
status: draft
parent: REQ-QCK
---

### 描述

检查是否正确使用判断语法糖宏。

### 检查规则

| 规则ID | 检查项 | 严重级别 |
|--------|--------|----------|
| MAC001 | 空指针判断需使用 RET_IF_PTR_INVALID | WARNING |
| MAC002 | 返回值判断需使用 RET_IF_NOT_OK | WARNING |
| MAC003 | 数组下标需校验后使用 | ERROR |

### 验收标准

1. 检测裸 if + return 模式，建议使用宏
2. 检测数组访问前是否有边界检查
3. 提供自动修复建议

---

## REQ-QCK-007 静态分析工具集成

---
id: REQ-QCK-007
title: 静态分析工具集成
priority: P1
status: draft
parent: REQ-QCK
---

### 描述

集成静态分析工具实现自动化检查。

### 工具选型

```yaml
tools:
  clang-tidy:
    purpose: "C/C++ 静态分析"
    custom_checks: true

  cppcheck:
    purpose: "补充检查"
    custom_rules: true

  custom_checker:
    purpose: "项目特有规则"
    language: "Python/C"
```

### 验收标准

1. 支持命令行单独执行
2. 支持IDE集成（VSCode、CLion）
3. 支持CI流水线集成
4. 检查结果可导出为标准格式（SARIF）

---

## REQ-QCK-008 增量检查

---
id: REQ-QCK-008
title: 增量检查
priority: P2
status: draft
parent: REQ-QCK
---

### 描述

支持只检查变更文件，提升检查效率。

### 命令设计

```bash
# 全量检查
quality check --all

# 增量检查（基于git diff）
quality check --diff HEAD~1

# 检查指定文件
quality check --files src/hal.c src/plat.c

# 生成报告
quality check --report html
```

### 验收标准

1. 正确识别git变更文件
2. 增量检查结果与全量一致
3. 支持指定基准分支

---

## REQ-QCK-009 规则配置

---
id: REQ-QCK-009
title: 规则配置
priority: P2
status: draft
parent: REQ-QCK
---

### 描述

支持规则的开关和严重级别配置。

### 配置格式

```yaml
# .quality.yaml
version: "1.0"

rules:
  TYP001:
    enabled: true
    severity: error

  NAM002:
    enabled: true
    severity: warning

  SEC003:
    enabled: false  # 禁用此规则

excludes:
  - "third_party/**"
  - "generated/**"

includes:
  - "src/**/*.c"
  - "include/**/*.h"
```

### 验收标准

1. 支持规则级别的开关控制
2. 支持文件/目录排除
3. 支持严重级别调整

---

## REQ-QCK-010 内存安全检测

---
id: REQ-QCK-010
title: 内存安全检测
priority: P2
status: draft
parent: REQ-QCK
---

### 描述

集成内存安全检测工具，检测内存泄漏、越界访问等问题。

### 检测工具

| 工具 | 用途 | 平台 |
|------|------|------|
| AddressSanitizer (ASan) | 内存越界、UAF | LinuxUT/ST |
| LeakSanitizer (LSan) | 内存泄漏 | LinuxUT/ST |
| Valgrind | 内存错误全面检测 | LinuxUT/ST |
| MemorySanitizer (MSan) | 未初始化内存 | LinuxUT/ST |

### 编译选项

```makefile
# Makefile
ifeq ($(SANITIZER),asan)
    CFLAGS += -fsanitize=address -fno-omit-frame-pointer
    LDFLAGS += -fsanitize=address
endif

ifeq ($(SANITIZER),lsan)
    CFLAGS += -fsanitize=leak
    LDFLAGS += -fsanitize=leak
endif

ifeq ($(SANITIZER),msan)
    CFLAGS += -fsanitize=memory -fno-omit-frame-pointer
    LDFLAGS += -fsanitize=memory
endif
```

### 命令行使用

```bash
# ASan编译运行
make PLATFORM=linux_ut SANITIZER=asan
./build/bin/linux_ut/test_runner

# Valgrind运行（无需重新编译）
valgrind --leak-check=full --error-exitcode=1 ./test_runner

# CI配置
./test_runner --sanitizer-report report.txt
```

### 验收标准

1. CI流水线包含ASan检测阶段
2. 内存泄漏导致测试失败
3. 错误信息包含调用栈
4. 支持抑制已知问题（suppressions文件）

---

## REQ-QCK-011 代码复杂度检查

---
id: REQ-QCK-011
title: 代码复杂度检查
priority: P2
status: draft
parent: REQ-QCK
---

### 描述

检查代码复杂度指标，防止过于复杂的代码。

### 检查指标

| 指标 | 阈值 | 说明 |
|------|------|------|
| 圈复杂度 | <= 15 | 单函数圈复杂度 |
| 函数行数 | <= 100 | 单函数代码行数 |
| 文件行数 | <= 1000 | 单文件代码行数 |
| 嵌套深度 | <= 4 | 最大嵌套层数 |

### 验收标准

1. 超过阈值报WARNING
2. 严重超标（2倍）报ERROR
3. 支持配置例外
