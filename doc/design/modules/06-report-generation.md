# 报告生成模块详细设计 (Report Generation)

## 模块概述

| 属性 | 值 |
|------|-----|
| **模块ID** | REPORT |
| **模块名称** | 报告生成 |
| **职责** | 测试结果的报告生成、可视化和导出 |
| **需求覆盖** | REPORT-001 ~ REPORT-009 |

---

## 1. 逻辑视图

### 1.1 报告器类层次

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Report Generation Classes                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        ReportEngine                                   │  │
│  ├───────────────────────────────────────────────────────────────────────┤  │
│  │ - reporters: List[IReporter]                                          │  │
│  │ - visualizers: List[IVisualizer]                                      │  │
│  │ - aggregators: List[IAggregator]                                      │  │
│  ├───────────────────────────────────────────────────────────────────────┤  │
│  │ + generate(results: TestResults) -> Report                            │  │
│  │ + export(format: str, path: Path)                                     │  │
│  │ + register_reporter(reporter)                                         │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         Reporter Hierarchy                            │  │
│  │                                                                       │  │
│  │              ┌─────────────────────────────┐                          │  │
│  │              │     <<interface>>           │                          │  │
│  │              │       IReporter             │                          │  │
│  │              ├─────────────────────────────┤                          │  │
│  │              │ + format: str               │                          │  │
│  │              │ + generate(data) -> Report  │                          │  │
│  │              │ + export(report, path)      │                          │  │
│  │              └──────────────┬──────────────┘                          │  │
│  │                             │                                         │  │
│  │       ┌───────────────┬─────┴─────┬───────────────┬───────────────┐   │  │
│  │       ▼               ▼           ▼               ▼               ▼   │  │
│  │  ┌─────────┐   ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌─────────┐   │  │
│  │  │Console  │   │  HTML   │  │  JSON   │  │  JUnit   │  │Markdown │   │  │
│  │  │Reporter │   │Reporter │  │Reporter │  │ Reporter │  │Reporter │   │  │
│  │  └─────────┘   └─────────┘  └─────────┘  └──────────┘  └─────────┘   │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        Visualizer Hierarchy                           │  │
│  │                                                                       │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │  │
│  │  │   PieChart      │  │   BarChart      │  │   LineChart     │        │  │
│  │  │   Visualizer    │  │   Visualizer    │  │   Visualizer    │        │  │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘        │  │
│  │                                                                       │  │
│  │  ┌─────────────────┐  ┌─────────────────┐                             │  │
│  │  │ ConfusionMatrix │  │   HeatMap       │                             │  │
│  │  │   Visualizer    │  │   Visualizer    │                             │  │
│  │  └─────────────────┘  └─────────────────┘                             │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        Report Components                              │  │
│  │                                                                       │  │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐     │  │
│  │  │   TestSummary    │  │ PerformanceReport│  │  AccuracyReport  │     │  │
│  │  ├──────────────────┤  ├──────────────────┤  ├──────────────────┤     │  │
│  │  │ + total          │  │ + latency_stats  │  │ + metrics        │     │  │
│  │  │ + passed         │  │ + throughput     │  │ + confusion_matrix    │  │
│  │  │ + failed         │  │ + memory_usage   │  │ + per_class_metrics   │  │
│  │  │ + skipped        │  │ + charts         │  │ + error_samples  │     │  │
│  │  │ + pass_rate      │  └──────────────────┘  └──────────────────┘     │  │
│  │  │ + duration       │                                                 │  │
│  │  └──────────────────┘                                                 │  │
│  │                                                                       │  │
│  │  ┌──────────────────┐  ┌──────────────────┐                           │  │
│  │  │ComparisonReport  │  │   Distributor    │                           │  │
│  │  ├──────────────────┤  ├──────────────────┤                           │  │
│  │  │ + baseline       │  │ + send_email()   │                           │  │
│  │  │ + current        │  │ + send_webhook() │                           │  │
│  │  │ + diff           │  │ + notify_slack() │                           │  │
│  │  │ + regressions    │  │ + archive()      │                           │  │
│  │  └──────────────────┘  └──────────────────┘                           │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 数据模型

```python
# report/models.py

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

@dataclass
class TestSummary:
    """测试摘要"""
    total: int
    passed: int
    failed: int
    skipped: int
    errors: int
    duration: float
    started_at: datetime
    finished_at: datetime

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0


@dataclass
class TestCaseReport:
    """用例报告"""
    test_id: str
    test_name: str
    status: str
    duration: float
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    artifacts: List[Path] = field(default_factory=list)


@dataclass
class PerformanceReport:
    """性能报告"""
    latency_mean: float
    latency_p50: float
    latency_p90: float
    latency_p99: float
    throughput_qps: float
    memory_peak_mb: float
    charts: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccuracyReport:
    """精度报告"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: Optional[List[List[int]]] = None
    per_class_metrics: Dict[str, Dict] = field(default_factory=dict)
    error_samples: List[Dict] = field(default_factory=list)


@dataclass
class Report:
    """完整报告"""
    title: str
    summary: TestSummary
    test_cases: List[TestCaseReport]
    environment: Dict[str, Any]
    performance: Optional[PerformanceReport] = None
    accuracy: Optional[AccuracyReport] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

---

## 2. 开发视图

### 2.1 包结构

```
aitest/report/
├── __init__.py
├── engine.py                # ReportEngine
├── reporter/
│   ├── __init__.py
│   ├── base.py              # IReporter接口
│   ├── console.py           # 控制台报告
│   ├── html.py              # HTML报告
│   ├── json.py              # JSON报告
│   ├── junit.py             # JUnit XML
│   ├── markdown.py          # Markdown报告
│   └── pdf.py               # PDF报告
├── visualizer/
│   ├── __init__.py
│   ├── base.py              # IVisualizer接口
│   ├── charts.py            # 图表生成
│   ├── confusion.py         # 混淆矩阵
│   └── timeline.py          # 时间线
├── templates/
│   ├── html/
│   │   ├── base.html
│   │   ├── summary.html
│   │   └── details.html
│   └── markdown/
│       └── report.md
├── aggregator.py            # 结果聚合
├── comparator.py            # 版本对比
├── distributor.py           # 报告分发
├── realtime.py              # 实时报告
└── models.py                # 数据模型
```

### 2.2 实现示例

```python
# report/reporter/html.py

from pathlib import Path
from typing import Optional
from jinja2 import Environment, PackageLoader
import json

from .base import IReporter
from ..models import Report


class HTMLReporter(IReporter):
    """HTML报告生成器"""

    format = "html"

    def __init__(self, template_dir: Optional[Path] = None):
        self.env = Environment(
            loader=PackageLoader('aitest.report', 'templates/html')
        )
        self.template = self.env.get_template('base.html')

    def generate(self, report: Report) -> str:
        """生成HTML报告"""
        return self.template.render(
            title=report.title,
            summary=report.summary,
            test_cases=report.test_cases,
            performance=report.performance,
            accuracy=report.accuracy,
            environment=report.environment,
            charts=self._generate_charts(report)
        )

    def export(self, report: Report, path: Path) -> None:
        """导出HTML报告"""
        html_content = self.generate(report)
        path.write_text(html_content)

    def _generate_charts(self, report: Report) -> dict:
        """生成图表数据"""
        charts = {}

        # 通过率饼图数据
        charts['pass_rate_pie'] = {
            'labels': ['Passed', 'Failed', 'Skipped', 'Errors'],
            'values': [
                report.summary.passed,
                report.summary.failed,
                report.summary.skipped,
                report.summary.errors
            ]
        }

        # 性能趋势图数据
        if report.performance:
            charts['latency_distribution'] = {
                'labels': ['Mean', 'P50', 'P90', 'P99'],
                'values': [
                    report.performance.latency_mean,
                    report.performance.latency_p50,
                    report.performance.latency_p90,
                    report.performance.latency_p99
                ]
            }

        return charts


# report/reporter/junit.py

import xml.etree.ElementTree as ET
from pathlib import Path

from .base import IReporter
from ..models import Report, TestCaseReport


class JUnitReporter(IReporter):
    """JUnit XML报告生成器"""

    format = "junit"

    def generate(self, report: Report) -> str:
        """生成JUnit XML"""
        root = ET.Element('testsuites')

        testsuite = ET.SubElement(root, 'testsuite', {
            'name': report.title,
            'tests': str(report.summary.total),
            'failures': str(report.summary.failed),
            'errors': str(report.summary.errors),
            'skipped': str(report.summary.skipped),
            'time': str(report.summary.duration)
        })

        for tc in report.test_cases:
            testcase = ET.SubElement(testsuite, 'testcase', {
                'name': tc.test_name,
                'classname': tc.test_id,
                'time': str(tc.duration)
            })

            if tc.status == 'failed':
                failure = ET.SubElement(testcase, 'failure', {
                    'message': tc.error_message or ''
                })
                if tc.stack_trace:
                    failure.text = tc.stack_trace

            elif tc.status == 'skipped':
                ET.SubElement(testcase, 'skipped')

            elif tc.status == 'error':
                error = ET.SubElement(testcase, 'error', {
                    'message': tc.error_message or ''
                })
                if tc.stack_trace:
                    error.text = tc.stack_trace

        return ET.tostring(root, encoding='unicode', xml_declaration=True)

    def export(self, report: Report, path: Path) -> None:
        """导出JUnit XML"""
        xml_content = self.generate(report)
        path.write_text(xml_content)


# report/realtime.py

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.live import Live


class RealtimeReporter:
    """实时进度报告"""

    def __init__(self):
        self.console = Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
        )
        self.stats = {'passed': 0, 'failed': 0, 'skipped': 0}
        self.task_id = None

    def start(self, total_tests: int) -> None:
        """开始测试"""
        self.task_id = self.progress.add_task("Running tests", total=total_tests)
        self.progress.start()

    def on_test_complete(self, result: 'TestResult') -> None:
        """测试完成回调"""
        self.progress.advance(self.task_id)

        if result.status == 'passed':
            self.stats['passed'] += 1
        elif result.status == 'failed':
            self.stats['failed'] += 1
            self.console.print(f"[red]FAIL[/red] {result.test_name}: {result.error_message}")
        elif result.status == 'skipped':
            self.stats['skipped'] += 1

    def finish(self) -> None:
        """完成测试"""
        self.progress.stop()
        self._print_summary()

    def _print_summary(self) -> None:
        """打印摘要"""
        table = Table(title="Test Results")
        table.add_column("Status", style="bold")
        table.add_column("Count", justify="right")

        table.add_row("[green]Passed[/green]", str(self.stats['passed']))
        table.add_row("[red]Failed[/red]", str(self.stats['failed']))
        table.add_row("[yellow]Skipped[/yellow]", str(self.stats['skipped']))

        self.console.print(table)
```

---

## 3. 场景视图

### 3.1 报告生成示例

```python
from aitest.report import ReportEngine, HTMLReporter, JUnitReporter

# 创建报告引擎
engine = ReportEngine()
engine.register_reporter(HTMLReporter())
engine.register_reporter(JUnitReporter())

# 生成报告
report = engine.generate(test_results)

# 导出多种格式
engine.export(report, "reports/result.html", format="html")
engine.export(report, "reports/result.xml", format="junit")
```

### 3.2 需求追溯

| 需求ID | 实现类/方法 | 测试用例 |
|--------|-------------|----------|
| REPORT-001 | `HTMLReporter`, `JSONReporter`, `JUnitReporter` | test_report_formats |
| REPORT-002 | `TestSummary` | test_summary |
| REPORT-003 | `TestCaseReport` | test_case_details |
| REPORT-004 | `PerformanceReport` | test_perf_report |
| REPORT-005 | `AccuracyReport` | test_accuracy_report |
| REPORT-006 | `visualizer/*` | test_visualizations |
| REPORT-007 | `ComparisonReport` | test_comparison |
| REPORT-008 | `Distributor` | test_distribution |
| REPORT-009 | `RealtimeReporter` | test_realtime |

---

*本文档为报告生成模块的详细设计，基于4+1视图方法。*
