# QCK 代码质量模块设计

---
module: QCK
version: 1.0
date: 2026-02-04
status: draft
requirements: REQ-QCK-001~007
domain: Generic
priority: P1
---

## 1. 模块概述

### 1.1 职责

代码质量检查和规范验证：
- 类型命名检查
- 命名规范检查
- 内存段标记检查
- 安全函数检查
- 静态分析集成

### 1.2 DDD定位

- **限界上下文**：通用域（可替换）
- **独立运行**：不依赖核心域
- **可替代**：可用其他静态分析工具替代

---

## 2. 文件结构

```
tools/quality/
├── __init__.py
├── checker.py          # 检查器主入口
├── rules/
│   ├── __init__.py
│   ├── naming.py       # 命名规范检查
│   ├── types.py        # 类型检查
│   ├── memory.py       # 内存段检查
│   └── security.py     # 安全函数检查
├── reporters/
│   ├── __init__.py
│   ├── console.py      # 控制台输出
│   └── json.py         # JSON输出
└── config/
    └── rules.yaml      # 规则配置
```

---

## 3. 检查规则

### 3.1 类型命名检查 (REQ-QCK-001)

| 类型 | 规范 | 示例 | 反例 |
|------|------|------|------|
| 整数 | 大写 | INT32, UINT8 | int, uint8_t |
| 浮点 | 大写 | FLOAT32 | float |
| 通用 | 大写 | VOID, CHAR | void, char |
| 返回值 | ERRNO_T | ERRNO_T | int |

```python
# rules/types.py
TYPE_RULES = {
    r'\bint\b': 'INT32',
    r'\bunsigned\s+int\b': 'UINT32',
    r'\bchar\b': 'CHAR',
    r'\bvoid\b': 'VOID',
    r'\bfloat\b': 'FLOAT32',
    r'\bdouble\b': 'FLOAT64',
    r'\buint8_t\b': 'UINT8',
    r'\bint32_t\b': 'INT32',
}

def check_types(content: str, filepath: str) -> list[Issue]:
    issues = []
    for pattern, replacement in TYPE_RULES.items():
        for match in re.finditer(pattern, content):
            issues.append(Issue(
                filepath=filepath,
                line=get_line_number(content, match.start()),
                rule="type-naming",
                message=f"Use {replacement} instead of {match.group()}",
                severity="warning",
            ))
    return issues
```

### 3.2 命名规范检查 (REQ-QCK-002)

| 项目 | 规范 | 正则 | 示例 |
|------|------|------|------|
| 结构体 | PascalCase + Stru | `^[A-Z][a-zA-Z0-9]*Stru$` | TestCaseStru |
| 枚举 | PascalCase + Enum | `^[A-Z][a-zA-Z0-9]*Enum$` | TestResultEnum |
| 函数 | Module_Action | `^[A-Z][a-z]+_[A-Z][a-zA-Z0-9]*$` | Runner_Init |
| 宏 | UPPER_CASE | `^[A-Z][A-Z0-9_]*$` | TEST_ASSERT |
| 全局变量 | g_前缀 | `^g_[a-z][a-zA-Z0-9]*$` | g_hal |
| 静态变量 | s_前缀（可选） | `^s_[a-z][a-zA-Z0-9]*$` | s_instance |

```python
# rules/naming.py
NAMING_RULES = {
    "struct": (r'typedef\s+struct\s+(\w+)', r'^[A-Z][a-zA-Z0-9]*Stru$'),
    "enum": (r'typedef\s+enum\s+(\w+)', r'^[A-Z][a-zA-Z0-9]*Enum$'),
    "function": (r'^(?:static\s+)?(?:\w+\s+)+(\w+)\s*\(', r'^[A-Z][a-z]+_[A-Z][a-zA-Z0-9]*$'),
    "macro": (r'#define\s+(\w+)', r'^[A-Z][A-Z0-9_]*$'),
    "global": (r'^(?:extern\s+)?(?:\w+\s+)+(\w+)\s*;', r'^g_[a-z][a-zA-Z0-9]*$'),
}
```

### 3.3 内存段标记检查 (REQ-QCK-003)

```python
# rules/memory.py
MEMORY_SECTIONS = ['SEC_DDR_TEXT', 'SEC_DDR_DATA', 'SEC_DDR_BSS', 'SEC_DDR_RODATA']

def check_memory_sections(content: str, filepath: str) -> list[Issue]:
    """检查函数和全局变量是否有内存段标记"""
    issues = []

    # 检查函数定义
    for match in re.finditer(r'^(\w+\s+)+(\w+)\s*\([^)]*\)\s*{', content, re.MULTILINE):
        func_start = match.start()
        # 检查函数前是否有SEC_前缀
        prefix = content[max(0, func_start-50):func_start]
        if not any(sec in prefix for sec in MEMORY_SECTIONS):
            issues.append(Issue(
                filepath=filepath,
                line=get_line_number(content, func_start),
                rule="memory-section",
                message=f"Function {match.group(2)} missing memory section marker",
                severity="info",
            ))

    return issues
```

### 3.4 安全函数检查 (REQ-QCK-004)

| 禁用函数 | 替代方案 |
|----------|----------|
| strcpy | strncpy / strlcpy |
| strcat | strncat / strlcat |
| sprintf | snprintf |
| gets | fgets |
| scanf | fgets + sscanf |

```python
# rules/security.py
UNSAFE_FUNCTIONS = {
    'strcpy': 'strncpy or strlcpy',
    'strcat': 'strncat or strlcat',
    'sprintf': 'snprintf',
    'vsprintf': 'vsnprintf',
    'gets': 'fgets',
    'scanf': 'fgets + sscanf',
}

def check_unsafe_functions(content: str, filepath: str) -> list[Issue]:
    issues = []
    for func, replacement in UNSAFE_FUNCTIONS.items():
        pattern = rf'\b{func}\s*\('
        for match in re.finditer(pattern, content):
            issues.append(Issue(
                filepath=filepath,
                line=get_line_number(content, match.start()),
                rule="unsafe-function",
                message=f"Use {replacement} instead of {func}",
                severity="error",
            ))
    return issues
```

### 3.5 错误码检查 (REQ-QCK-005)

```python
# rules/errno.py
def check_errno_return(content: str, filepath: str) -> list[Issue]:
    """检查函数返回值是否使用ERRNO_T"""
    issues = []

    # 查找返回int的函数（应该返回ERRNO_T）
    pattern = r'^int\s+(\w+)\s*\([^)]*\)\s*{'
    for match in re.finditer(pattern, content, re.MULTILINE):
        func_name = match.group(1)
        # 排除main函数
        if func_name != 'main':
            issues.append(Issue(
                filepath=filepath,
                line=get_line_number(content, match.start()),
                rule="errno-return",
                message=f"Function {func_name} should return ERRNO_T instead of int",
                severity="warning",
            ))

    return issues
```

---

## 4. 静态分析集成 (REQ-QCK-007)

### 4.1 支持的工具

| 工具 | 用途 | 配置文件 |
|------|------|----------|
| clang-tidy | C/C++静态分析 | .clang-tidy |
| cppcheck | C/C++检查 | cppcheck.cfg |
| flake8 | Python检查 | .flake8 |
| mypy | Python类型检查 | mypy.ini |

### 4.2 clang-tidy配置

```yaml
# .clang-tidy
Checks: >
  clang-analyzer-*,
  bugprone-*,
  misc-*,
  -misc-unused-parameters,
  modernize-*,
  performance-*,
  readability-*,
  -readability-magic-numbers

WarningsAsErrors: ''

CheckOptions:
  - key: readability-identifier-naming.FunctionCase
    value: CamelCase
  - key: readability-identifier-naming.GlobalVariablePrefix
    value: g_
  - key: readability-identifier-naming.MacroDefinitionCase
    value: UPPER_CASE
```

### 4.3 集成运行

```python
# tools/quality/external.py
import subprocess

def run_clang_tidy(files: list[str]) -> list[Issue]:
    """运行clang-tidy"""
    cmd = ['clang-tidy'] + files
    result = subprocess.run(cmd, capture_output=True, text=True)
    return parse_clang_tidy_output(result.stdout)

def run_cppcheck(files: list[str]) -> list[Issue]:
    """运行cppcheck"""
    cmd = ['cppcheck', '--enable=all', '--xml'] + files
    result = subprocess.run(cmd, capture_output=True, text=True)
    return parse_cppcheck_output(result.stderr)
```

---

## 5. 命令行接口

### 5.1 命令

```bash
# 检查所有代码
python -m tools.quality check

# 检查指定文件
python -m tools.quality check src/framework/runner.c

# 检查指定目录
python -m tools.quality check --path src/

# 只运行特定规则
python -m tools.quality check --rules naming,types

# 输出JSON格式
python -m tools.quality check --format json --output report.json

# 集成外部工具
python -m tools.quality check --with-clang-tidy --with-cppcheck
```

### 5.2 输出示例

```
$ python -m tools.quality check src/

Code Quality Check
==================

src/framework/runner.c:45: [warning] type-naming
  Use UINT32 instead of unsigned int

src/framework/runner.c:78: [error] unsafe-function
  Use snprintf instead of sprintf

src/platform/linux_ut/ut_hal.c:23: [warning] naming-function
  Function name 'init' should follow Module_Action pattern

Summary:
  Files checked: 15
  Issues found: 12 (3 errors, 7 warnings, 2 info)
```

---

## 6. 规则配置

### 6.1 配置文件

```yaml
# tools/quality/config/rules.yaml
rules:
  type-naming:
    enabled: true
    severity: warning

  naming-struct:
    enabled: true
    severity: warning
    pattern: "^[A-Z][a-zA-Z0-9]*Stru$"

  naming-function:
    enabled: true
    severity: warning
    pattern: "^[A-Z][a-z]+_[A-Z][a-zA-Z0-9]*$"
    exclude:
      - "main"
      - "test_*"  # 测试函数例外

  unsafe-function:
    enabled: true
    severity: error

  memory-section:
    enabled: false  # 默认关闭，仅嵌入式平台需要

# 排除路径
exclude:
  - "build/"
  - "third_party/"
  - "*_test.c"  # 测试文件可放宽
```

---

## 7. 数据模型

```python
from dataclasses import dataclass
from enum import Enum

class Severity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

@dataclass
class Issue:
    filepath: str
    line: int
    rule: str
    message: str
    severity: Severity
    column: int | None = None
    suggestion: str | None = None

@dataclass
class CheckResult:
    files_checked: int
    issues: list[Issue]
    error_count: int
    warning_count: int
    info_count: int

    @property
    def has_errors(self) -> bool:
        return self.error_count > 0
```

---

## 8. 输出格式

### 8.1 JSON输出

```json
{
  "files_checked": 15,
  "summary": {
    "errors": 3,
    "warnings": 7,
    "info": 2
  },
  "issues": [
    {
      "filepath": "src/framework/runner.c",
      "line": 45,
      "column": 5,
      "rule": "type-naming",
      "severity": "warning",
      "message": "Use UINT32 instead of unsigned int",
      "suggestion": "UINT32"
    }
  ]
}
```

### 8.2 VSCode问题格式

```
src/framework/runner.c:45:5: warning: Use UINT32 instead of unsigned int [type-naming]
```

---

## 9. CI集成

### 9.1 GitHub Actions

```yaml
- name: Code Quality Check
  run: |
    python -m tools.quality check --format json --output quality-report.json
    if [ $(jq '.summary.errors' quality-report.json) -gt 0 ]; then
      echo "Quality check failed with errors"
      exit 1
    fi
```

### 9.2 Pre-commit Hook

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: quality-check
        name: Code Quality Check
        entry: python -m tools.quality check
        language: python
        types: [c]
```

---

## 10. 需求追溯

| 需求ID | 需求标题 | 设计章节 |
|--------|----------|----------|
| REQ-QCK-001 | 类型命名检查 | 3.1 |
| REQ-QCK-002 | 命名规范检查 | 3.2 |
| REQ-QCK-003 | 内存段标记检查 | 3.3 |
| REQ-QCK-004 | 安全函数检查 | 3.4 |
| REQ-QCK-005 | 错误码检查 | 3.5 |
| REQ-QCK-007 | 静态分析工具集成 | 4 |
