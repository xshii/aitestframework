"""Report generation for test executions (REQ-4.4)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring

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


def generate_junit_xml(execution_id: str, output_dir: str | Path) -> Path:
    """Generate JUnit XML report for Jenkins integration."""
    detail = store.get_execution_detail(execution_id)
    if detail is None:
        raise ValueError(f"Execution not found: {execution_id}")

    # Group cases by suite
    suites: dict[str, list[dict]] = {}
    for case in detail.get("cases", []):
        suites.setdefault(case.get("suite_class", "Unknown"), []).append(case)

    root = Element("testsuites", name=execution_id,
                   tests=str(detail.get("total", 0)),
                   failures=str(detail.get("failed", 0)),
                   errors=str(detail.get("errored", 0)),
                   time=str(_duration(detail)))

    for suite_name, cases in suites.items():
        ts = SubElement(root, "testsuite", name=suite_name,
                        tests=str(len(cases)),
                        failures=str(sum(1 for c in cases if c["status"] == "FAIL")),
                        errors=str(sum(1 for c in cases if c["status"] in ("ERROR", "TIMEOUT", "CRASH"))))
        for c in cases:
            tc = SubElement(ts, "testcase",
                            classname=suite_name,
                            name=c["case_method"],
                            time=str(c.get("duration_s") or 0))
            if c["status"] == "FAIL":
                f = SubElement(tc, "failure", message=_last_line(c.get("failure_reason", "")))
                f.text = c.get("failure_reason", "")
            elif c["status"] in ("ERROR", "TIMEOUT", "CRASH"):
                e = SubElement(tc, "error", message=c["status"])
                e.text = c.get("failure_reason", "")
            elif c["status"] == "SKIP":
                SubElement(tc, "skipped", message=c.get("failure_reason", ""))
            if c.get("stdout"):
                SubElement(tc, "system-out").text = c["stdout"]
            if c.get("stderr"):
                SubElement(tc, "system-err").text = c["stderr"]

    out = Path(output_dir) / execution_id
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "junit.xml"
    report_path.write_bytes(b'<?xml version="1.0" encoding="UTF-8"?>\n' +
                            tostring(root, encoding="unicode").encode("utf-8"))
    logger.info("JUnit XML written: %s", report_path)
    return report_path


def _duration(detail: dict) -> float:
    if detail.get("started_at") and detail.get("finished_at"):
        from datetime import datetime
        try:
            s = datetime.fromisoformat(detail["started_at"])
            f = datetime.fromisoformat(detail["finished_at"])
            return (f - s).total_seconds()
        except (ValueError, TypeError):
            pass
    return 0.0


def _last_line(text: str) -> str:
    lines = [l for l in text.split("\n") if l.strip()]
    return lines[-1] if lines else ""
