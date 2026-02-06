# DVT 开发工具模块设计

---
module: DVT
version: 1.0
date: 2026-02-04
status: draft
requirements: REQ-DVT-001~012
domain: Generic
priority: P1
---

## 1. 模块概述

### 1.1 职责

提供开发效率提升工具：
- Python测试运行器
- 数据生成工具
- 报告生成工具
- 数据格式转换
- 环境检查工具
- VSCode插件（P2）

### 1.2 DDD定位

- **限界上下文**：通用域（可替换）
- **订阅事件**：GoldenUpdated, TestCompleted
- **独立运行**：可用第三方工具替代

### 1.3 技术选型

| 功能 | 库 |
|------|-----|
| 数值计算 | numpy |
| 模板渲染 | jinja2 |
| 终端UI | rich |
| 命令行 | argparse（标准库） |
| Web界面 | flask（可选） |

---

## 2. 文件结构

```
tools/
├── __init__.py
├── runner/                 # 测试运行器
│   ├── __init__.py
│   ├── runner.py           # 主运行器
│   ├── parallel.py         # 并行执行
│   └── result_parser.py    # 结果解析
├── data/                   # 数据工具
│   ├── __init__.py
│   ├── generate.py         # 数据生成
│   ├── yaml_to_c.py        # YAML转C
│   └── binview.py          # 二进制查看
├── report/                 # 报告生成
│   ├── __init__.py
│   ├── generator.py        # 报告生成器
│   └── templates/
│       ├── html_report.html
│       └── junit.xml
├── testmgmt/               # 用例管理Web（可选）
│   ├── __init__.py
│   ├── server.py
│   └── templates/
├── archive/                # 归档工具
│   ├── __init__.py
│   └── archiver.py
├── check_env.py            # 环境检查
└── vscode/                 # VSCode插件
    ├── extension/
    └── snippets/
```

---

## 3. Python测试运行器 (REQ-DVT-001)

### 3.1 功能

- 编排执行C测试程序
- 多进程并行运行
- 结果聚合
- 失败重试

### 3.2 命令行接口

```bash
# 列出测试
python -m tools.runner --platform linux_ut --list

# 运行所有测试
python -m tools.runner --platform linux_ut

# 并行运行
python -m tools.runner --platform linux_ut --parallel 4

# 过滤运行
python -m tools.runner --platform linux_ut --filter "matmul*"

# 失败重试
python -m tools.runner --platform linux_ut --retry 2

# 指定测试列表
python -m tools.runner --platform simulator --list sanity
```

### 3.3 实现

```python
# tools/runner/runner.py
import subprocess
import json
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from rich.progress import Progress
from rich.console import Console

@dataclass
class RunConfig:
    platform: str
    filter: str | None = None
    parallel: int = 1
    retry: int = 0
    test_list: str | None = None
    output_dir: Path = Path("build/reports")

class TestRunner:
    def __init__(self, config: RunConfig):
        self.config = config
        self.console = Console()

    def run(self) -> dict:
        """运行测试并返回结果"""
        test_binary = self._get_test_binary()

        cmd = [str(test_binary), "--json"]
        if self.config.filter:
            cmd.extend(["--filter", self.config.filter])

        if self.config.parallel > 1:
            return self._run_parallel(cmd)
        else:
            return self._run_single(cmd)

    def _run_single(self, cmd: list[str]) -> dict:
        """单进程运行"""
        result = subprocess.run(cmd, capture_output=True, text=True)
        return json.loads(result.stdout)

    def _run_parallel(self, base_cmd: list[str]) -> dict:
        """多进程并行运行"""
        # 获取测试列表
        tests = self._list_tests()

        # 分片
        shards = self._shard_tests(tests, self.config.parallel)

        # 并行执行
        results = []
        with ProcessPoolExecutor(max_workers=self.config.parallel) as executor:
            futures = [
                executor.submit(self._run_shard, base_cmd, shard)
                for shard in shards
            ]
            for future in futures:
                results.append(future.result())

        # 合并结果
        return self._merge_results(results)

    def _retry_failed(self, result: dict, cmd: list[str]) -> dict:
        """重试失败用例"""
        failed_tests = [t for t in result["tests"] if t["status"] == "FAIL"]

        for _ in range(self.config.retry):
            if not failed_tests:
                break

            for test in failed_tests:
                filter_pattern = f"{test['suite']}.{test['name']}"
                retry_result = self._run_single(cmd + ["--filter", filter_pattern])

                # 更新结果
                if retry_result["tests"][0]["status"] == "PASS":
                    self._update_test_result(result, retry_result["tests"][0])
                    failed_tests.remove(test)

        return result
```

### 3.4 并行执行

```python
# tools/runner/parallel.py
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.progress import Progress, TaskID

def run_parallel_tests(
    tests: list[str],
    runner_path: str,
    workers: int = 4,
) -> list[dict]:
    """并行执行测试"""
    results = []

    with Progress() as progress:
        task = progress.add_task("Running tests...", total=len(tests))

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(run_single_test, runner_path, test): test
                for test in tests
            }

            for future in as_completed(futures):
                test = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({
                        "test": test,
                        "status": "ERROR",
                        "error": str(e)
                    })
                progress.advance(task)

    return results
```

---

## 4. 数据生成工具 (REQ-DVT-002)

### 4.1 功能

- 基于Python参考模型生成测试数据
- 支持批量生成
- 输出Golden格式

### 4.2 命令行接口

```bash
# 生成单个算子数据
python -m tools.data.generate --op matmul --config data_config.yaml

# 批量生成
python -m tools.data.generate --batch testdata/generators/matmul_cases.yaml

# 指定输出目录
python -m tools.data.generate --op conv --output testdata/golden/conv/
```

### 4.3 配置格式

```yaml
# testdata/generators/matmul_cases.yaml
operator: matmul
dtype: float32
cases:
  - name: basic_2x2
    inputs:
      A: {shape: [2, 2], init: random}
      B: {shape: [2, 2], init: random}

  - name: large_matrix
    inputs:
      A: {shape: [256, 512], init: random}
      B: {shape: [512, 256], init: random}

  - name: edge_case_zeros
    inputs:
      A: {shape: [4, 4], init: zeros}
      B: {shape: [4, 4], init: ones}
```

### 4.4 实现

```python
# tools/data/generate.py
import numpy as np
from pathlib import Path
import yaml
from dataclasses import dataclass

@dataclass
class GenerateConfig:
    operator: str
    dtype: str
    cases: list[dict]
    output_dir: Path

class DataGenerator:
    def __init__(self, config: GenerateConfig):
        self.config = config
        self.dtype = getattr(np, config.dtype)

    def generate_all(self):
        """生成所有用例数据"""
        for case in self.config.cases:
            self.generate_case(case)

    def generate_case(self, case: dict):
        """生成单个用例"""
        name = case["name"]
        inputs = {}

        # 生成输入
        for input_name, spec in case["inputs"].items():
            inputs[input_name] = self._create_tensor(spec)

        # 计算参考输出
        output = self._compute_reference(inputs)

        # 保存
        self._save_golden(name, inputs, output)

    def _create_tensor(self, spec: dict) -> np.ndarray:
        shape = spec["shape"]
        init = spec.get("init", "random")

        if init == "random":
            return np.random.randn(*shape).astype(self.dtype)
        elif init == "zeros":
            return np.zeros(shape, dtype=self.dtype)
        elif init == "ones":
            return np.ones(shape, dtype=self.dtype)
        else:
            raise ValueError(f"Unknown init: {init}")

    def _compute_reference(self, inputs: dict) -> np.ndarray:
        """调用参考模型计算"""
        if self.config.operator == "matmul":
            return np.matmul(inputs["A"], inputs["B"])
        elif self.config.operator == "conv2d":
            # ... conv实现
            pass
        else:
            raise ValueError(f"Unknown operator: {self.config.operator}")

    def _save_golden(self, name: str, inputs: dict, output: np.ndarray):
        """保存为Golden格式"""
        output_dir = self.config.output_dir / name
        output_dir.mkdir(parents=True, exist_ok=True)

        for input_name, data in inputs.items():
            save_golden_file(output_dir / f"input_{input_name}.bin", data)

        save_golden_file(output_dir / "golden.bin", output)
```

---

## 5. 报告生成工具 (REQ-DVT-003)

### 5.1 支持格式

| 格式 | 用途 | 文件 |
|------|------|------|
| HTML | 人工查看 | report.html |
| JSON | 工具解析 | report.json |
| JUnit XML | CI集成 | report.xml |
| Markdown | 文档嵌入 | report.md |

### 5.2 命令行接口

```bash
# 生成HTML报告
python -m tools.report --input results.json --format html --output report.html

# 多格式输出
python -m tools.report --input results.json --format html,json,junit --output-dir reports/
```

### 5.3 HTML模板

```html
<!-- tools/report/templates/html_report.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Test Report - {{ execution_id }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .summary { background: #f5f5f5; padding: 15px; border-radius: 5px; }
        .pass { color: green; }
        .fail { color: red; }
        .skip { color: orange; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background: #4CAF50; color: white; }
        tr:nth-child(even) { background: #f9f9f9; }
    </style>
</head>
<body>
    <h1>Test Report</h1>
    <p>Execution ID: {{ execution_id }}</p>
    <p>Platform: {{ platform }}</p>
    <p>Time: {{ started_at }} - {{ finished_at }}</p>

    <div class="summary">
        <h2>Summary</h2>
        <p>Total: {{ summary.total }}</p>
        <p class="pass">Passed: {{ summary.passed }}</p>
        <p class="fail">Failed: {{ summary.failed }}</p>
        <p class="skip">Skipped: {{ summary.skipped }}</p>
        <p>Duration: {{ summary.duration_ms }}ms</p>
    </div>

    {% if failed_tests %}
    <h2>Failed Tests</h2>
    <table>
        <tr>
            <th>Test</th>
            <th>Duration</th>
            <th>Error</th>
        </tr>
        {% for test in failed_tests %}
        <tr>
            <td>{{ test.suite }}.{{ test.name }}</td>
            <td>{{ test.duration_ms }}ms</td>
            <td>{{ test.fail_reason }}<br>at {{ test.fail_file }}:{{ test.fail_line }}</td>
        </tr>
        {% endfor %}
    </table>
    {% endif %}

    <h2>All Tests</h2>
    <table>
        <tr>
            <th>Test</th>
            <th>Status</th>
            <th>Duration</th>
        </tr>
        {% for test in tests %}
        <tr>
            <td>{{ test.suite }}.{{ test.name }}</td>
            <td class="{{ test.status | lower }}">{{ test.status }}</td>
            <td>{{ test.duration_ms }}ms</td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>
```

### 5.4 JUnit XML生成

```python
# tools/report/generator.py
from xml.etree import ElementTree as ET

def generate_junit_xml(results: dict, output_path: str):
    """生成JUnit XML格式报告"""
    testsuites = ET.Element("testsuites")

    # 按套件分组
    suites = {}
    for test in results["tests"]:
        suite_name = test["suite"]
        if suite_name not in suites:
            suites[suite_name] = []
        suites[suite_name].append(test)

    for suite_name, tests in suites.items():
        testsuite = ET.SubElement(testsuites, "testsuite")
        testsuite.set("name", suite_name)
        testsuite.set("tests", str(len(tests)))
        testsuite.set("failures", str(sum(1 for t in tests if t["status"] == "FAIL")))
        testsuite.set("time", str(sum(t["duration_ms"] for t in tests) / 1000))

        for test in tests:
            testcase = ET.SubElement(testsuite, "testcase")
            testcase.set("name", test["name"])
            testcase.set("classname", suite_name)
            testcase.set("time", str(test["duration_ms"] / 1000))

            if test["status"] == "FAIL":
                failure = ET.SubElement(testcase, "failure")
                failure.set("message", test.get("fail_reason", ""))
                failure.text = f"at {test.get('fail_file', '')}:{test.get('fail_line', '')}"
            elif test["status"] == "SKIP":
                ET.SubElement(testcase, "skipped")

    tree = ET.ElementTree(testsuites)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
```

---

## 6. 数据格式转换 (REQ-DVT-006)

### 6.1 YAML转C

```bash
python -m tools.data.yaml_to_c testdata/cases/matmul.yaml --output build/generated/
```

```python
# tools/data/yaml_to_c.py
def yaml_to_c_header(yaml_path: str, output_dir: str):
    """将YAML测试数据转换为C头文件"""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    output_path = Path(output_dir) / f"{Path(yaml_path).stem}.h"

    with open(output_path, "w") as f:
        f.write(f"/* Auto-generated from {yaml_path} */\n")
        f.write("#ifndef TEST_DATA_H\n")
        f.write("#define TEST_DATA_H\n\n")

        for case in data["cases"]:
            name = case["name"]
            for var_name, values in case.items():
                if isinstance(values, list):
                    f.write(f"static const FLOAT32 {name}_{var_name}[] = {{\n")
                    f.write("    " + ", ".join(f"{v}f" for v in values) + "\n")
                    f.write("};\n\n")

        f.write("#endif /* TEST_DATA_H */\n")
```

### 6.2 二进制查看

```bash
python -m tools.data.binview testdata/golden/matmul/case001.bin
```

```python
# tools/data/binview.py
def view_golden_file(path: str):
    """查看Golden文件内容"""
    with open(path, "rb") as f:
        header = read_golden_header(f)

    print(f"File: {path}")
    print(f"Magic: {hex(header.magic)}")
    print(f"Version: {header.version}")
    print(f"Dtype: {DTYPE_NAMES[header.dtype]}")
    print(f"Shape: {header.shape[:header.ndim]}")
    print(f"Elements: {np.prod(header.shape[:header.ndim])}")
    print(f"Data offset: {header.data_offset}")
```

---

## 7. 环境检查工具 (REQ-DVT-007)

### 7.1 命令行接口

```bash
# 检查所有环境
python -m tools.check_env

# 检查指定平台
python -m tools.check_env --platform simulator

# JSON输出
python -m tools.check_env --format json
```

### 7.2 输出示例

```
$ python -m tools.check_env

Environment Check
=================

[OK] Python 3.10.6
[OK] GCC 11.3.0
[OK] Make 4.3

Dependencies:
[OK] numpy 1.24.0
[OK] pyyaml 6.0
[OK] httpx 0.24.0
[OK] jinja2 3.1.2
[OK] rich 13.0.0
[--] flask (optional, not installed)

Toolchain:
[OK] simulator 2.0.0 at /opt/toolchain/simulator
[OK] compiler 2.1.0 at /opt/toolchain/compiler
[!!] esl_model not found (required for ESL platform)

Summary: 9 passed, 1 warning, 1 missing
```

### 7.3 实现

```python
# tools/check_env.py
import shutil
import subprocess
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table

@dataclass
class CheckResult:
    name: str
    status: str  # "ok", "warning", "error"
    version: str | None = None
    path: str | None = None
    message: str | None = None

def check_python_version() -> CheckResult:
    import sys
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return CheckResult("Python", "ok", version)

def check_gcc() -> CheckResult:
    gcc_path = shutil.which("gcc")
    if not gcc_path:
        return CheckResult("GCC", "error", message="not found")

    result = subprocess.run(["gcc", "--version"], capture_output=True, text=True)
    version = result.stdout.split("\n")[0].split()[-1]
    return CheckResult("GCC", "ok", version, gcc_path)

def check_python_package(name: str, required: bool = True) -> CheckResult:
    try:
        module = __import__(name)
        version = getattr(module, "__version__", "unknown")
        return CheckResult(name, "ok", version)
    except ImportError:
        status = "error" if required else "warning"
        return CheckResult(name, status, message="not installed")

def check_toolchain_component(name: str, config_path: str) -> CheckResult:
    # 从deps/resolved.yaml读取安装状态
    pass

def run_all_checks(platform: str | None = None) -> list[CheckResult]:
    results = []

    # 基础工具
    results.append(check_python_version())
    results.append(check_gcc())
    results.append(check_make())

    # Python依赖
    results.append(check_python_package("numpy"))
    results.append(check_python_package("pyyaml"))
    results.append(check_python_package("httpx"))
    results.append(check_python_package("jinja2"))
    results.append(check_python_package("rich"))
    results.append(check_python_package("flask", required=False))

    # 工具链
    if platform in [None, "simulator"]:
        results.append(check_toolchain_component("simulator", "deps/resolved.yaml"))
        results.append(check_toolchain_component("compiler", "deps/resolved.yaml"))

    return results
```

---

## 8. VSCode插件 (REQ-DVT-008~012)

### 8.1 功能列表

| 功能 | 描述 | 优先级 |
|------|------|--------|
| 代码片段 | 测试用例模板 | P1 |
| 右键菜单 | 运行测试、检查质量 | P2 |
| CodeLens | 函数上方Run Test链接 | P2 |
| 问题导航 | 点击跳转到源码 | P2 |

### 8.2 代码片段

```json
// tools/vscode/snippets/c.json
{
  "Test Case": {
    "prefix": "tfunc",
    "body": [
      "TEST_CASE(${1:suite}, ${2:name})",
      "{",
      "    ${3:// Test code}",
      "    return AITF_OK;",
      "}"
    ],
    "description": "Create a test case"
  },
  "Assert Equal": {
    "prefix": "teq",
    "body": "TEST_ASSERT_EQ(${1:expected}, ${2:actual});",
    "description": "Assert equal"
  },
  "Assert Near": {
    "prefix": "tnear",
    "body": "TEST_ASSERT_NEAR(${1:expected}, ${2:actual}, ${3:1e-5});",
    "description": "Assert float near"
  }
}
```

### 8.3 插件配置

```json
// tools/vscode/extension/package.json
{
  "name": "aitestframework",
  "displayName": "AI Test Framework",
  "version": "1.0.0",
  "engines": {"vscode": "^1.80.0"},
  "categories": ["Testing"],
  "contributes": {
    "commands": [
      {"command": "aitf.runTest", "title": "AITF: Run Test"},
      {"command": "aitf.runAllTests", "title": "AITF: Run All Tests"},
      {"command": "aitf.checkQuality", "title": "AITF: Check Code Quality"}
    ],
    "menus": {
      "editor/context": [
        {"command": "aitf.runTest", "when": "resourceExtname == .c"}
      ]
    },
    "configuration": {
      "title": "AI Test Framework",
      "properties": {
        "aitf.platform": {
          "type": "string",
          "default": "linux_ut",
          "description": "Default test platform"
        },
        "aitf.qualityCheck.onSave": {
          "type": "boolean",
          "default": true,
          "description": "Run quality check on save"
        }
      }
    }
  }
}
```

---

## 9. 需求追溯

| 需求ID | 需求标题 | 设计章节 |
|--------|----------|----------|
| REQ-DVT-001 | Python测试运行器 | 3 |
| REQ-DVT-002 | 数据生成工具 | 4 |
| REQ-DVT-003 | 报告生成工具 | 5 |
| REQ-DVT-006 | 数据格式转换工具 | 6 |
| REQ-DVT-007 | 环境检查工具 | 7 |
| REQ-DVT-008 | VSCode插件集成 | 8 |
| REQ-DVT-009 | 右键菜单与快捷命令 | 8.3 |
| REQ-DVT-010 | 代码片段模板 | 8.2 |
