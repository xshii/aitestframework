# REQ-4 用例管理

---
version: 2.0
date: 2026-02-27
status: draft
---

## 1. 概述

用例采用 **unittest 风格**（类似 nosetest）编写——每个测试套是一个 Python 类，每个用例是类中的一个方法。框架提供一套基础 API，用例通过调用这些 API 完成环境选择、构建、执行、结果收集和比较。

**核心设计原则：**
- 用例是 Python 代码，不是 YAML 声明
- 构建操作在 `setUpClass` 中完成，整个测试套只构建一次
- 每个用例方法自行调用执行脚本和比较 API
- 用例执行状态实时推送到 Dashboard

**架构原则：** 所有用例管理逻辑实现在 Core API 层（`aitestframework.api` / `aitestframework.cases`），CLI 和 Web 均为薄壳调用层。

## 2. 需求详情

### REQ-4.1 用例编写规范 [P0]

**描述：** 用例采用 unittest.TestCase 子类的方式编写，框架自动发现和收集。

**用例示例：**

```python
# cases/npu/operators/test_conv2d.py

import unittest
from aitestframework import api

class TestConv2dNPU(unittest.TestCase):
    """Conv2D NPU 算子测试套"""

    @classmethod
    def setUpClass(cls):
        """整个套件只执行一次：选环境、选工具链、编译"""
        cls.env = api.select_env("sim-server")        # 选择执行环境
        api.use_bundle("npu-v2.1")                     # 选择配置集/工具链
        cls.build = api.build("stubs/npu/operators/conv2d")  # 编译桩代码（只编译一次）

    def test_fp32_3x3_basic(self):
        """Conv2D FP32 3x3 基本测试"""
        result = api.execute(self.env, "sim_run --lib libconv2d.so --case fp32_3x3")
        golden = api.load_golden("npu/conv2d/fp32_3x3_output")
        api.compare_fuzzy(result.output, golden, rtol=1e-5, atol=1e-6)

    def test_fp16_3x3_basic(self):
        """Conv2D FP16 3x3 基本测试"""
        result = api.execute(self.env, "sim_run --lib libconv2d.so --case fp16_3x3")
        golden = api.load_golden("npu/conv2d/fp16_3x3_output")
        api.compare_fuzzy(result.output, golden, rtol=1e-3, atol=1e-4)

    def test_int8_3x3_quantized(self):
        """Conv2D INT8 3x3 量化测试"""
        result = api.execute(self.env, "sim_run --lib libconv2d.so --case int8_3x3")
        golden = api.load_golden("npu/conv2d/int8_3x3_output")
        api.compare_exact(result.output, golden)

    @classmethod
    def tearDownClass(cls):
        """清理"""
        cls.env.cleanup()
```

**桩代码UT示例（本地快速验证）：**

```python
# cases/npu/operators/test_conv2d_ut.py

import unittest
from aitestframework import api

class TestConv2dStubUT(unittest.TestCase):
    """Conv2D 桩代码单元测试（本地编译运行）"""

    @classmethod
    def setUpClass(cls):
        cls.env = api.select_env("local")
        api.use_bundle("npu-v2.1")
        cls.build = api.build("stubs/npu/operators/conv2d", target="ut")

    def test_stub_compiles(self):
        """桩代码可以编译通过"""
        self.assertTrue(self.build.success)

    def test_stub_ut_basic(self):
        """桩代码UT基本功能"""
        result = api.execute(self.env, f"{self.build.bin_dir}/conv2d_ut --gtest_filter=Basic.*")
        self.assertEqual(result.returncode, 0)
```

**用例目录组织：**

```
cases/
├── npu/
│   ├── operators/
│   │   ├── test_conv2d.py
│   │   ├── test_conv2d_ut.py
│   │   ├── test_matmul.py
│   │   └── test_relu.py
│   └── models/
│       ├── test_resnet.py
│       └── test_bert.py
├── gpu/
│   └── operators/
│       └── test_gemm.py
├── cpu/
│   └── preprocess/
│       └── test_resize.py
└── conftest.py                  # 公共 fixture（如有需要）
```

**命名规范：**
- 文件名：`test_<算子或模型名>.py`
- 类名：`Test<Name><Platform>`（如 `TestConv2dNPU`）
- 方法名：`test_<具体场景>`

**验收标准：**
- 框架自动发现 `cases/` 目录下所有 `test_*.py` 文件中的 `unittest.TestCase` 子类
- 用例类和方法的 docstring 作为描述展示在 Dashboard 上
- `setUpClass` 失败时该套件所有用例标记为 ERROR

### REQ-4.2 框架基础 API [P0]

**描述：** 框架提供 `aitestframework.api` 模块，用例通过调用这些 API 完成全部操作。

**API 清单：**

```python
# aitestframework/api.py

# === 环境选择 ===
def select_env(target_name: str) -> Environment:
    """选择执行环境（对应 runner/targets.yaml 中的配置）
    Args:
        target_name: 目标名，如 "local", "sim-server", "fpga-board"
    Returns:
        Environment 对象，封装本地或远程执行能力
    """

# === 工具链/配置集 ===
def use_bundle(bundle_name: str) -> BundleInfo:
    """激活指定的配置集（工具链+库版本组合）
    Args:
        bundle_name: 配置集名，如 "npu-v2.1"
    Returns:
        BundleInfo，包含工具链路径和环境变量
    """

# === 构建 ===
def build(stub_path: str, target: str = "lib") -> BuildResult:
    """编译桩代码
    Args:
        stub_path: 桩代码路径，如 "stubs/npu/operators/conv2d"
        target: 构建目标，"lib"=编译为库, "ut"=编译为UT可执行程序
    Returns:
        BuildResult，包含 success、bin_dir、lib_dir 等信息
    """

# === 执行 ===
def execute(env: Environment, command: str,
            timeout: int = 300) -> ExecuteResult:
    """在指定环境中执行命令
    Args:
        env: select_env() 返回的环境对象
        command: 要执行的命令（shell命令字符串）
        timeout: 超时秒数
    Returns:
        ExecuteResult，包含 returncode、stdout、stderr、output 文件路径
    """

# === Golden 数据 ===
def load_golden(golden_id: str) -> np.ndarray:
    """加载 Golden 基准数据
    Args:
        golden_id: Golden 注册表中的 ID，如 "npu/conv2d/fp32_3x3_output"
    Returns:
        numpy ndarray
    """

# === 结果比较 ===
def compare_exact(actual: np.ndarray, expected: np.ndarray) -> None:
    """精确比较，不匹配时抛出 AssertionError"""

def compare_fuzzy(actual: np.ndarray, expected: np.ndarray,
                  rtol: float = 1e-5, atol: float = 1e-6) -> CompareResult:
    """模糊比较（容差匹配），不通过时抛出 AssertionError"""

def compare_cosine(actual: np.ndarray, expected: np.ndarray,
                   threshold: float = 0.999) -> CompareResult:
    """余弦相似度比较，不通过时抛出 AssertionError"""

def compare_topk(actual: np.ndarray, expected: np.ndarray,
                 k: int = 5) -> CompareResult:
    """Top-K 准确率比较，不通过时抛出 AssertionError"""
```

**API 设计原则：**
- 比较方法失败时抛出 `AssertionError`，与 unittest 断言机制一致
- `execute()` 自动捕获 stdout/stderr 并保存日志
- `execute()` 在远程环境自动处理文件上传/下载
- 所有 API 调用自动记录到执行日志

**验收标准：**
- 用例只需 `from aitestframework import api` 即可使用全部功能
- API 有完整的类型注解和 docstring
- API 调用错误时给出清晰的错误信息

### REQ-4.3 执行状态跟踪与实时推送 [P0]

**描述：** 跟踪每个用例的执行状态，并实时推送到 Dashboard。

**状态定义：**

| 状态 | 含义 |
|------|------|
| PENDING | 待执行 |
| RUNNING | 正在执行中 |
| PASS | 执行通过 |
| FAIL | 执行失败（比较不匹配或断言失败） |
| TIMEOUT | 执行超时 |
| CRASH | 执行崩溃（进程异常退出） |
| SKIP | 被跳过 |
| ERROR | 框架级错误（setUpClass 失败等） |

**实时推送机制：**
- 用例状态变更时，框架自动写入 SQLite
- Dashboard 页面通过轮询（每5秒）或页面刷新获取最新状态
- 无需 WebSocket，保持简单

**验收标准：**
- 用例开始执行时状态变为 RUNNING
- 用例结束后状态立即更新
- Dashboard 刷新即可看到最新状态

### REQ-4.4 执行报告 [P0]

**描述：** 生成测试执行报告，支持 JSON 和 HTML 两种格式。

**JSON 报告结构：**

```json
{
  "execution_id": "20260227-143000-abc123",
  "start_time": "2026-02-27T14:30:00Z",
  "end_time": "2026-02-27T14:35:00Z",
  "duration_s": 300,
  "bundle": "npu-v2.1",
  "git_commit": "abc1234",
  "summary": {
    "total": 100,
    "pass": 85,
    "fail": 10,
    "timeout": 2,
    "crash": 1,
    "skip": 2,
    "error": 0
  },
  "pass_rate": 0.85,
  "suites": [
    {
      "class": "TestConv2dNPU",
      "module": "cases.npu.operators.test_conv2d",
      "setup_duration_s": 15.0,
      "cases": [
        {
          "name": "test_fp32_3x3_basic",
          "status": "PASS",
          "duration_s": 2.5,
          "compare_result": {
            "method": "fuzzy",
            "max_abs_error": 1.2e-6,
            "max_rel_error": 8.5e-6
          }
        }
      ]
    }
  ]
}
```

**验收标准：**
- JSON 报告自动生成到 `build/reports/` 目录
- HTML 报告使用 Jinja2 模板渲染
- 报告包含环境信息（配置集、工具链版本、git commit）
- 报告按 suite 分组展示

### REQ-4.5 用例发现与筛选 [P1]

**描述：** 自动发现用例，支持多维度筛选。

**发现机制：** 扫描 `cases/` 目录下所有 `test_*.py` 文件，收集 `unittest.TestCase` 子类。

**筛选方式：**

```bash
# 运行全部用例
aitf run

# 按目录/模块筛选
aitf run cases/npu/operators/
aitf run cases/npu/operators/test_conv2d.py

# 按类名筛选
aitf run -k "TestConv2dNPU"

# 按方法名筛选
aitf run -k "test_fp32"

# 按平台目录筛选
aitf run cases/npu/
aitf run cases/gpu/

# 组合
aitf run cases/npu/ -k "test_fp32"
```

**验收标准：**
- `-k` 支持子串匹配（与 pytest -k 行为一致）
- 可按文件路径、目录路径精确控制执行范围
- `aitf run --collect-only` 只列出用例不执行

### REQ-4.6 执行历史 [P1]

**描述：** 记录用例的历史执行结果，全部保留不自动清理。

**存储：** SQLite 表 `execution_history`

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 自增主键 |
| execution_id | TEXT | 批次ID |
| suite_class | TEXT | 测试套类名 |
| case_method | TEXT | 测试方法名 |
| status | TEXT | 执行状态 |
| duration_s | REAL | 耗时 |
| started_at | DATETIME | 开始时间 |
| finished_at | DATETIME | 结束时间 |
| bundle | TEXT | 使用的配置集 |
| target | TEXT | 执行环境 |
| git_commit | TEXT | 代码版本 |
| detail_json | TEXT | 详细结果JSON（比较数据等） |
| stdout_path | TEXT | stdout日志路径 |
| stderr_path | TEXT | stderr日志路径 |

**验收标准：**
- 每次执行自动写入历史记录
- 历史数据全部保留，不自动清理
- 可查询指定用例的全部历史执行结果
- 可按时间段查询趋势数据（供 Dashboard 图表使用）

### REQ-4.7 结果比较 [P1]

**描述：** 框架提供多种比较方法，用例通过 API 调用。

**比较方法：**

| API | 说明 | 失败行为 |
|-----|------|----------|
| `api.compare_exact(actual, expected)` | 逐元素精确匹配 | 抛出 AssertionError |
| `api.compare_fuzzy(actual, expected, rtol, atol)` | 浮点容差匹配 | 抛出 AssertionError |
| `api.compare_cosine(actual, expected, threshold)` | 余弦相似度 | 抛出 AssertionError |
| `api.compare_topk(actual, expected, k)` | Top-K准确率 | 抛出 AssertionError |

**fuzzy 匹配规则：** `|actual - expected| <= atol + rtol * |expected|`

**比较结果数据：**

```python
@dataclass
class CompareResult:
    method: str
    passed: bool
    max_abs_error: float | None = None
    mean_abs_error: float | None = None
    max_rel_error: float | None = None
    mean_rel_error: float | None = None
    cosine_similarity: float | None = None
    mismatch_count: int = 0
    total_elements: int = 0
    mismatch_rate: float = 0.0
```

**验收标准：**
- 比较失败时的 AssertionError 信息中包含 max_abs_error 等关键指标
- 不匹配时输出前 10 个不匹配元素的 index/expected/actual
- 比较结果自动记录到执行历史

## 3. 技术选型

| 决策 | 选型 | 备选 | 理由 |
|------|------|------|------|
| 用例框架 | unittest (标准库) | pytest, nose2 | 标准库无额外依赖，类继承方式清晰 |
| 用例发现 | unittest + 自定义 TestLoader | nose2, pytest | 保持简单，自定义 loader 支持目录扫描 |
| 用例执行 | unittest.TextTestRunner + 自定义 TestResult | pytest | 自定义 TestResult 实现状态推送和日志收集 |
| 数据存储 | SQLite + SQLAlchemy | 纯文件 | 支持复杂查询和历史记录 |
| 报告模板 | Jinja2 | — | 与 Flask 共用 |
| 数值比较 | numpy | 手写 | numpy.allclose 成熟可靠 |

## 4. 框架 API 数据模型

```python
@dataclass
class Environment:
    """执行环境（本地或远程）"""
    name: str
    type: str                       # local / remote
    host: str | None = None
    _ssh: SSHClient | None = None   # 远程时持有连接

    def run(self, command: str, timeout: int = 300) -> ExecuteResult: ...
    def upload(self, local: str, remote: str) -> None: ...
    def download(self, remote: str, local: str) -> None: ...
    def cleanup(self) -> None: ...

@dataclass
class BuildResult:
    """构建结果"""
    success: bool
    bin_dir: str                    # 可执行文件目录
    lib_dir: str                    # 库文件目录
    duration_s: float
    error_msg: str | None = None

@dataclass
class ExecuteResult:
    """命令执行结果"""
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    output: np.ndarray | None = None  # 如果产生了输出数据文件
    output_path: str | None = None

@dataclass
class BundleInfo:
    """激活的配置集信息"""
    name: str
    toolchains: dict[str, str]      # name -> version
    libraries: dict[str, str]
    env: dict[str, str]             # 设置到环境变量的值

@dataclass
class CompareResult:
    """比较结果"""
    method: str
    passed: bool
    max_abs_error: float | None = None
    mean_abs_error: float | None = None
    max_rel_error: float | None = None
    mean_rel_error: float | None = None
    cosine_similarity: float | None = None
    mismatch_count: int = 0
    total_elements: int = 0
    mismatch_rate: float = 0.0
```

## 5. 自定义 TestResult（状态推送）

```python
class AitfTestResult(unittest.TestResult):
    """自定义 TestResult，负责状态推送和日志收集"""

    def startTest(self, test):
        """用例开始 → 状态写入 RUNNING → Dashboard 可见"""
        super().startTest(test)
        db.update_status(test.id(), "RUNNING")

    def addSuccess(self, test):
        """用例通过 → 状态写入 PASS"""
        super().addSuccess(test)
        db.update_status(test.id(), "PASS")

    def addFailure(self, test, err):
        """用例失败 → 状态写入 FAIL"""
        super().addFailure(test, err)
        db.update_status(test.id(), "FAIL", error=err)

    def addError(self, test, err):
        """用例异常 → 状态写入 ERROR/CRASH/TIMEOUT"""
        super().addError(test, err)
        status = classify_error(err)  # 区分 TIMEOUT / CRASH / ERROR
        db.update_status(test.id(), status, error=err)
```

## 6. 依赖

- REQ-1（桩代码）：`api.build()` 编译桩代码
- REQ-2（Golden数据）：`api.load_golden()` 加载基准数据
- REQ-3（构建管理）：`api.use_bundle()` 选择配置集
- REQ-6（执行框架）：`api.select_env()` + `api.execute()` 的底层实现
- REQ-7（Dashboard）：`AitfTestResult` 推送状态到 SQLite，Dashboard 读取展示
