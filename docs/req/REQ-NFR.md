# REQ-NFR 非功能需求

---
version: 1.1
date: 2026-02-27
status: draft
---

## 1. 概述

定义框架在代码质量、性能、可维护性等方面的要求。

**架构总纲：Core API 唯一逻辑，CLI 和 Web 都是薄壳调用层。**

```
用例代码 / Jenkins / 终端用户  ──→  CLI (argparse薄壳)  ──┐
                                                          ├──→  Core API (唯一业务逻辑)  ──→  SQLite
Web Dashboard 用户             ──→  Flask路由 (薄壳)     ──┘
```

## 2. 需求详情

### NFR-1 测试覆盖率 [P1]

**描述：** 框架自身的 Python 代码需有单元测试。

**要求：**
- 行覆盖率 > 80%
- 核心模块（comparator、runner、logparser）覆盖率 > 90%
- 使用 pytest + pytest-cov

**验证命令：**

```bash
pytest tests/ --cov=aitestframework --cov-report=term-missing --cov-fail-under=80
```

### NFR-2 代码风格 [P0]

**描述：** Python 代码风格统一，使用自动化工具检查。

**工具链：**

| 工具 | 用途 | 配置 |
|------|------|------|
| ruff | lint + format | pyproject.toml [tool.ruff] |
| mypy | 类型检查 | pyproject.toml [tool.mypy] |

**ruff 配置：**

```toml
[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "A", "SIM"]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

**验证命令：**

```bash
ruff check .
ruff format --check .
mypy aitestframework/
```

### NFR-3 文档与类型注解 [P1]

**描述：** 关键模块需有 docstring 和类型注解。

**要求：**
- 所有公开类和函数需有 docstring（Google 风格）
- 所有函数参数和返回值需有类型注解
- 数据类使用 `@dataclass` 或 Pydantic model

**示例：**

```python
def compare_fuzzy(
    actual: np.ndarray,
    expected: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> CompareResult:
    """Compare two arrays with relative and absolute tolerance.

    Args:
        actual: The actual output array from test execution.
        expected: The expected golden reference array.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Returns:
        CompareResult with match status and error metrics.
    """
```

### NFR-4 Python版本 [P0]

**描述：** 支持 Python 3.10 及以上版本。

**要求：**
- 最低版本 Python 3.10
- CI 中测试 3.10 和 3.12 两个版本
- 不使用 3.10 以下不支持的语法特性（如 3.12 的 type 语句）

### NFR-5 Web性能 [P1]

**描述：** Dashboard 页面响应时间要求。

**要求：**

| 指标 | 要求 |
|------|------|
| 页面首次加载 | < 2s |
| 表格筛选响应 | < 1s |
| 图表数据加载 | < 1s |
| Webhook处理 | < 500ms |

**措施：**
- SQLite 表添加必要索引（case_name、execution_id、platform、status）
- 图表数据做服务端聚合，不传原始数据到前端
- 历史数据按天聚合存储，避免查询全量记录

### NFR-6 零外部服务依赖 [P0]

**描述：** 框架运行不需要额外的数据库/消息队列等外部服务。

**要求：**
- 数据存储使用 SQLite（单文件）
- 无需 Redis、PostgreSQL、RabbitMQ 等外部服务
- `pip install` 后即可运行
- 数据库文件默认存储在项目目录下：`data/aitf.db`

## 3. 项目结构规范

```
aitestframework/               # Python 包根目录
├── __init__.py
├── api.py                     # 框架对外 API 入口（用例调用此模块）
├── cli.py                     # 统一命令行入口（薄壳，调用 Core API）
├── core/                      # 公共基础
│   ├── config.py              # 配置加载
│   ├── db.py                  # SQLAlchemy 初始化
│   ├── log.py                 # 日志配置
│   └── models.py              # 数据库模型
├── stubs/                     # REQ-1 桩代码管理
│   └── manager.py
├── golden/                    # REQ-2 Golden数据
│   └── manager.py
├── deps/                      # REQ-3 依赖管理
│   ├── manager.py             # DepsManager
│   └── bundle.py              # BundleManager
├── build/                     # REQ-3 构建管理
│   └── builder.py             # BuildManager
├── cases/                     # REQ-4 用例管理
│   ├── loader.py              # 用例发现和加载
│   ├── comparator.py          # 比较方法
│   ├── result.py              # AitfTestResult
│   └── report.py              # 报告生成
├── logparser/                 # REQ-5 日志分析
│   ├── parser.py
│   └── rules/
├── runner/                    # REQ-6 执行框架
│   ├── executor.py            # 执行调度
│   ├── local.py               # LocalExecutor
│   └── remote.py              # RemoteExecutor
└── web/                       # REQ-7 Dashboard（薄壳，调用 Core API）
    ├── app.py
    ├── views/
    ├── templates/
    └── static/
```

## 4. 开发工作流

```bash
# 初始化开发环境
git clone <repo>
cd aitestframework
pip install -e ".[dev]"

# 开发循环
ruff check .                  # lint
ruff format .                 # format
mypy aitestframework/         # type check
pytest tests/                 # test

# 启动 Dashboard
aitf web --debug
```
