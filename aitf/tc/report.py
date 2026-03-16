"""Report generation for test executions (REQ-4.4)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from aitf.tc import store

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "templates" / "tc"


def generate_json_report(execution_id: str, output_dir: str | Path) -> Path:
    """Generate JSON report file. Returns path to the generated file."""
    detail = store.get_execution_detail(execution_id)
    if detail is None:
        raise ValueError(f"Execution not found: {execution_id}")

    out = Path(output_dir) / execution_id
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "result.json"
    report_path.write_text(json.dumps(detail, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("JSON report written: %s", report_path)
    return report_path


def generate_html_report(execution_id: str, output_dir: str | Path) -> Path:
    """Generate HTML report. Returns path to generated file."""
    detail = store.get_execution_detail(execution_id)
    if detail is None:
        raise ValueError(f"Execution not found: {execution_id}")

    # Group cases by suite
    suites: dict[str, list[dict]] = {}
    for case in detail.get("cases", []):
        suite_name = case.get("suite_class", "Unknown")
        suites.setdefault(suite_name, []).append(case)

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=True,
    )
    template = env.get_template("report.html")
    html = template.render(execution=detail, suites=suites)

    out = Path(output_dir) / execution_id
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "report.html"
    report_path.write_text(html, encoding="utf-8")
    logger.info("HTML report written: %s", report_path)
    return report_path
