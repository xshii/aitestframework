# REQ-7 Flask CI Dashboard

---
version: 1.2
date: 2026-02-27
status: draft
---

## 1. 概述

基于 Flask 的多页签 Web 界面，作为 CI 流水线的结果展示面板。服务端渲染为主，使用 ECharts 绘制图表。无需用户认证，定位为团队内部 Dashboard 工具。

**架构原则：** Web 层是薄壳调用层。Flask 路由和 API 端点仅负责请求解析和响应渲染，所有业务逻辑调用 Core API 层（`aitestframework.*`）。

**环境约束：**
- 内网环境，无法访问外网 CDN
- 所有前端静态资源（Bootstrap、ECharts）打包在仓库内，本地引用

## 2. 需求详情

### REQ-7.1 概览页 [P0]

**描述：** Dashboard 首页，展示测试整体状态的关键指标。

**页面内容：**

```
┌─────────────────────────────────────────────────────────┐
│  aitestframework Dashboard                    最后更新: ...│
├──────┬──────┬──────┬──────┬──────┬──────────────────────┤
│ 总数 │ PASS │ FAIL │TMOUT │CRASH │      通过率           │
│ 100  │  85  │  10  │  3   │  2   │  ████████░░ 85%      │
├──────┴──────┴──────┴──────┴──────┴──────────────────────┤
│                                                          │
│  最近构建状态              │   通过率趋势（最近30天）      │
│  ┌────────────────────┐    │   ┌─────────────────────┐   │
│  │ #123 ● PASS  2m ago│    │   │     ___/\___        │   │
│  │ #122 ● FAIL  1h ago│    │   │   /        \___     │   │
│  │ #121 ● PASS  3h ago│    │   │ /                   │   │
│  │ ...                 │    │   └─────────────────────┘   │
│  └────────────────────┘    │                              │
│                                                          │
│  失败用例 Top 10                                          │
│  ┌───────────────────────────────────────────────────┐   │
│  │ 用例名              │ 状态 │ 连续失败次数 │ 平台    │   │
│  │ matmul_int8_large   │ FAIL │    5        │ npu    │   │
│  │ conv2d_bf16_5x5     │ TMOUT│    3        │ npu    │   │
│  │ ...                 │      │             │        │   │
│  └───────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

**验收标准：**
- 页面加载后展示最新一次执行的摘要数据
- 通过率趋势图使用 ECharts 折线图
- 数据从 SQLite 查询，页面刷新即更新

### REQ-7.2 用例管理页 [P0]

**描述：** 浏览测试套和用例，支持从页面触发执行。

**页面内容：**

```
┌──────────────────────────────────────────────────────────────┐
│  测试套列表                                    [刷新]         │
├──────────────────────────────────────────────────────────────┤
│ 筛选: [平台 ▼] [状态 ▼]                                     │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ▶ TestConv2dNPU (cases/npu/operators/test_conv2d.py)       │
│    4 用例 | 最近: 3 PASS / 1 FAIL | 2026-02-27 14:30       │
│    [▶ 执行此套件]                                            │
│                                                              │
│  ▶ TestMatmulNPU (cases/npu/operators/test_matmul.py)       │
│    6 用例 | 最近: 6 PASS | 2026-02-27 14:28                 │
│    [▶ 执行此套件]                                            │
│                                                              │
│  ▶ TestResnetNPU (cases/npu/models/test_resnet.py)          │
│    2 用例 | 未执行                                           │
│    [▶ 执行此套件]                                            │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│  批量操作: [☑ 全选] [▶ 执行选中套件]                          │
│  执行参数: 环境 [sim-server ▼]  配置集 [npu-v2.1 ▼]          │
└──────────────────────────────────────────────────────────────┘
```

**展开套件后显示用例列表：**

```
│  ▼ TestConv2dNPU (cases/npu/operators/test_conv2d.py)       │
│    │ test_fp32_3x3_basic        PASS    2.5s                │
│    │ test_fp16_3x3_basic        FAIL    2.1s  [查看日志]     │
│    │ test_int8_3x3_quantized    PASS    1.8s                │
│    │ test_bf16_5x5_large        TIMEOUT 60.0s [查看日志]     │
│    [▶ 执行此套件]                                            │
```

**触发执行流程：**

1. 用户在页面上选择测试套（单个或多个）
2. 选择执行环境（下拉选 targets.yaml 中的目标）和配置集（下拉选 bundles.yaml）
3. 点击"执行"按钮
4. 后端创建一个后台线程，调用 Core API（`aitestframework.runner`）执行选中的测试套
5. 页面跳转到执行详情页，通过轮询（每5秒刷新）展示实时状态
6. 执行过程中各用例状态依次从 PENDING → RUNNING → PASS/FAIL/...

**后端接口：**

```
POST /api/run
Content-Type: application/json

{
  "suites": [
    "cases/npu/operators/test_conv2d.py",
    "cases/npu/operators/test_matmul.py"
  ],
  "target": "sim-server",
  "bundle": "npu-v2.1"
}

Response:
{
  "status": "ok",
  "execution_id": "20260227-150000-def456"
}
```

**测试套发现机制：**
- Dashboard 启动时扫描 `cases/` 目录，解析所有 `test_*.py` 文件
- 提取 `unittest.TestCase` 子类名、方法名、docstring
- 信息缓存到 SQLite，页面直接查询
- 提供"刷新"按钮重新扫描（代码更新后使用）

**验收标准：**
- 页面展示以测试套（类）为粒度，可展开查看用例（方法）
- 支持选择单个或多个测试套触发执行
- 执行时必须选择目标环境和配置集
- 执行触发后跳转到执行详情页，实时看到进度
- 正在执行的套件显示进度条或 spinner
- 表格支持分页（每页50条）
- 筛选条件通过 URL query string 传递
- 状态颜色：PASS=绿、FAIL=红、TIMEOUT=橙、CRASH=紫

### REQ-7.3 执行记录页 [P0]

**描述：** 历史执行批次列表和详情。

**列表页内容：**
- 执行批次表格：execution_id、开始时间、平台、总数/通过/失败、通过率、触发方式（手动/CI）

**详情页内容：**
- 执行摘要统计
- 用例结果列表（可筛选状态）
- 失败用例的详细信息：失败原因、比较结果、日志链接
- 日志内容查看（嵌入终端风格的日志展示区域）

**验收标准：**
- 列表按时间倒序排列
- 详情页可查看单条用例的 stdout/stderr 日志内容
- 日志展示区域支持滚动，错误行红色高亮

### REQ-7.4 Golden管理页 [P1]

**描述：** Golden 数据注册表的浏览界面。

**页面内容：**
- Golden 列表表格：ID、名称、平台、算子、数据类型、shape、版本、校验状态
- 筛选：按平台、算子
- 校验状态：最近一次 checksum 校验结果（通过/失败/未校验）

**验收标准：**
- 表格可按列排序
- 校验失败的条目红色高亮

### REQ-7.5 日志分析页 [P1]

**描述：** 日志解析结果的展示界面。

**页面内容：**
- 选择日志文件（从历史执行记录中选择，或手动指定路径）
- 解析结果展示：事件列表、错误统计、性能数据
- 错误行高亮，可跳转到原始日志对应行

**验收标准：**
- 错误事件红色高亮，警告事件橙色
- 统计区域显示错误/警告分组计数
- 大日志文件分页加载（每页1000行）

### REQ-7.6 构建管理页 [P1]

**描述：** 工具链和依赖的状态展示。

**页面内容：**
- 工具链列表：名称、配置版本、已安装版本、状态（已安装/未安装/版本不匹配）
- 第三方库列表：名称、版本、状态
- 最近构建历史：平台、时间、耗时、结果

**验收标准：**
- 版本不匹配时橙色警告
- 未安装时灰色标识

### REQ-7.7 趋势图表 [P1]

**描述：** 使用 ECharts 展示测试趋势数据。

**图表列表：**

| 图表 | 类型 | 数据 | 位置 |
|------|------|------|------|
| 通过率趋势 | 折线图 | 最近30天每次执行的通过率 | 概览页 |
| 状态分布 | 饼图 | 最近执行的状态分布 | 概览页 |
| 执行耗时趋势 | 折线图 | 最近30天总耗时变化 | 执行记录页 |
| 平台对比 | 柱状图 | 各平台通过率对比 | 概览页 |
| 算子覆盖率 | 热力图 | 各算子在各平台上的测试覆盖 | 用例管理页 |

**技术方案：**
- ECharts 本地引入：`<script src="/static/vendor/echarts/echarts.min.js">`（文件存放于 `web/static/vendor/echarts/`）
- Flask 接口返回图表数据（JSON），前端 JS 初始化 ECharts 实例
- 图表数据接口：`/api/charts/<chart_name>?days=30`

**验收标准：**
- 图表数据通过 API 接口获取，与页面渲染分离
- 图表支持基本交互（hover显示数值、点击跳转详情）
- 无数据时显示空状态提示

### REQ-7.8 CI Webhook 接口 [P0]

**描述：** 接收 Jenkins 推送的构建/测试结果。

**接口设计：**

```
POST /api/webhook

Content-Type: application/json
X-Webhook-Token: <配置的token>

Body:
{
  "event": "execution_complete",
  "execution_id": "20260227-143000-abc123",
  "jenkins_build": {
    "job_name": "aitf-npu-smoke",
    "build_number": 123,
    "build_url": "http://jenkins.internal/job/aitf-npu-smoke/123/"
  },
  "git_commit": "abc1234",
  "platform": "npu",
  "summary": {
    "total": 100,
    "pass": 85,
    "fail": 15
  },
  "report_path": "/path/to/result.json"
}
```

**响应：**

```json
{"status": "ok", "execution_id": "20260227-143000-abc123"}
```

**webhook 处理逻辑：**
1. 校验 token
2. 解析 JSON body
3. 如果 `report_path` 指向本地文件，导入详细结果到 SQLite
4. 更新 Dashboard 数据

**验收标准：**
- token 不匹配返回 401
- body 格式错误返回 400 及错误信息
- 成功处理返回 200
- 处理后 Dashboard 数据立即更新

### REQ-7.9 页面布局 [P0]

**描述：** 简洁的多页签布局，无需用户认证。

**布局结构：**

```
┌──────────────────────────────────────────────────────────┐
│  aitestframework              [概览][用例][执行][Golden]   │
│                                [日志][构建]                │
├──────────────────────────────────────────────────────────┤
│                                                          │
│                     页面内容区域                           │
│                                                          │
│                                                          │
│                                                          │
├──────────────────────────────────────────────────────────┤
│  v0.1.0 | SQLite | 数据更新: 2026-02-27 14:30:00         │
└──────────────────────────────────────────────────────────┘
```

**技术方案：**
- CSS 框架：Bootstrap 5（本地静态文件，存放于 `web/static/vendor/bootstrap/`）
- 导航栏：Bootstrap navbar，多页签通过不同的 Flask route 实现
- 每个页签对应一个 Flask Blueprint

**样式原则：零自定义 CSS，全部使用 Bootstrap 内置类**
- 所有布局、颜色、间距、表格、卡片等**只用 Bootstrap 自带的 utility class 和组件**
- 不编写自定义 CSS 文件（不需要 app.css）
- 状态颜色直接使用 Bootstrap 语义色：`text-success`(PASS)、`text-danger`(FAIL)、`text-warning`(TIMEOUT)、`text-purple`→`badge`(CRASH)
- 表格使用 `table table-striped table-hover`
- 卡片使用 `card`，统计数字使用 `card` + `display-6`
- 如确需微调（如个别间距），在 Jinja2 模板中用 Bootstrap 的 `m-*` `p-*` class 解决，不另写 CSS
- **目标：后期维护只需改 Jinja2 模板中的 HTML 和 Bootstrap class，不涉及任何 CSS 知识**

**Flask 路由：**

| 路由 | 方法 | Blueprint | 页面 |
|------|------|-----------|------|
| `/` | GET | overview | 概览页 |
| `/cases` | GET | cases | 用例管理页（测试套列表） |
| `/cases/<suite>` | GET | cases | 测试套详情（展开用例列表） |
| `/executions` | GET | executions | 执行记录页 |
| `/executions/<id>` | GET | executions | 执行详情页（实时状态） |
| `/golden` | GET | golden | Golden管理页 |
| `/logs` | GET | logs | 日志分析页 |
| `/build` | GET | build | 构建管理页 |
| `/api/run` | POST | api | 触发测试套执行 |
| `/api/suites/refresh` | POST | api | 重新扫描cases/目录 |
| `/api/executions/<id>/status` | GET | api | 轮询执行状态（JSON） |
| `/api/webhook` | POST | api | Jenkins Webhook接口 |
| `/api/charts/<name>` | GET | api | 图表数据API |

**验收标准：**
- 页面首次加载时间 < 2s
- 页签切换通过普通链接跳转（非 SPA）
- 在 1920x1080 和 1366x768 分辨率下均可正常使用
- 无需登录即可访问
- **项目中不存在自定义 CSS 文件**，所有样式由 Bootstrap class 完成

## 3. 技术选型

| 决策 | 选型 | 备选 | 理由 |
|------|------|------|------|
| Web框架 | Flask 3.x | Django, FastAPI | 轻量，模板渲染为主，够用 |
| 模板引擎 | Jinja2 | mako | Flask 默认，无需额外配置 |
| CSS框架 | Bootstrap 5（本地静态文件） | Tailwind, Bulma | 组件丰富，文档好，单文件引入无需构建 |
| 图表库 | ECharts 5（本地静态文件） | Chart.js, Highcharts | 功能强大，中文支持好，单文件引入 |
| ORM | SQLAlchemy 2.x | 裸SQL, peewee | 成熟稳定，类型安全 |
| 数据库 | SQLite | PostgreSQL | 零部署，单文件，适合内部工具 |

## 4. 数据模型（SQLAlchemy）

```python
class Execution(Base):
    __tablename__ = "executions"
    id = Column(String, primary_key=True)        # execution_id
    started_at = Column(DateTime)
    finished_at = Column(DateTime)
    platform = Column(String)
    git_commit = Column(String)
    trigger = Column(String)                      # manual / jenkins
    jenkins_job = Column(String, nullable=True)
    jenkins_build = Column(Integer, nullable=True)
    jenkins_url = Column(String, nullable=True)
    total = Column(Integer)
    passed = Column(Integer)
    failed = Column(Integer)
    timeout = Column(Integer)
    crashed = Column(Integer)
    skipped = Column(Integer)
    pass_rate = Column(Float)
    report_json_path = Column(String, nullable=True)

class CaseResult(Base):
    __tablename__ = "case_results"
    id = Column(Integer, primary_key=True, autoincrement=True)
    execution_id = Column(String, ForeignKey("executions.id"))
    case_name = Column(String)
    suite = Column(String)
    platform = Column(String)
    status = Column(String)
    duration_s = Column(Float)
    failure_reason = Column(String, nullable=True)
    compare_detail = Column(Text, nullable=True)   # JSON
    stdout_path = Column(String, nullable=True)
    stderr_path = Column(String, nullable=True)

class SuiteInfo(Base):
    """从 cases/ 目录扫描发现的测试套信息"""
    __tablename__ = "suite_info"
    id = Column(Integer, primary_key=True, autoincrement=True)
    module_path = Column(String, unique=True)       # cases/npu/operators/test_conv2d.py
    class_name = Column(String)                     # TestConv2dNPU
    docstring = Column(String, nullable=True)       # 类 docstring
    platform = Column(String)                       # npu / gpu / cpu（从目录推断）
    category = Column(String)                       # operators / models / preprocess
    case_count = Column(Integer)                    # 方法数量
    case_names = Column(Text)                       # JSON: ["test_fp32_3x3_basic", ...]
    scanned_at = Column(DateTime)                   # 最近扫描时间
    last_execution_id = Column(String, nullable=True)
    last_status_summary = Column(Text, nullable=True)  # JSON: {"pass":3,"fail":1}
```

## 5. 对外接口

```python
# Flask app factory
def create_app(config: dict | None = None) -> Flask: ...

# API endpoints
# GET  /api/charts/pass_rate_trend?days=30
# GET  /api/charts/status_distribution?execution_id=xxx
# POST /api/webhook
```

**启动命令：**

```bash
# 开发模式
aitf web --debug --port 5000

# 生产模式（gunicorn）
gunicorn "aitestframework.web:create_app()" --bind 0.0.0.0:5000 --workers 2
```

## 6. 静态资源管理

内网环境无法访问外网 CDN，所有第三方前端资源打包在仓库内：

```
web/static/
├── vendor/                         # 第三方库（git追踪）
│   ├── bootstrap/
│   │   ├── css/bootstrap.min.css   # Bootstrap 5.3.x
│   │   └── js/bootstrap.bundle.min.js
│   └── echarts/
│       └── echarts.min.js          # ECharts 5.x
└── js/
    └── charts.js                   # ECharts 图表初始化脚本（仅此一个JS文件）
```

**说明：**
- 不存在自定义 CSS 文件，所有样式通过 Bootstrap class 在模板中完成
- `charts.js` 仅负责调用 ECharts API 渲染图表，逻辑简单固定
- 获取方式：开发者从外网下载后提交到仓库，或从团队共享盘获取，版本在 README 中注明

## 7. 依赖

- REQ-4（用例管理）：用例定义和状态数据
- REQ-5（日志分析）：日志解析结果展示
- REQ-6（执行框架）：执行记录和报告
- REQ-2（Golden数据）：Golden注册表数据
- REQ-3（构建管理）：工具链和依赖状态
