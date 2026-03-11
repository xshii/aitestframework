"""CLI sub-commands for test-case management (REQ-4.5 / REQ-6.7)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

COMMAND = "run"
HELP = "运行测试用例"


def register(subparsers) -> None:
    p = subparsers.add_parser(COMMAND, help=HELP)
    p.add_argument("paths", nargs="*", default=None,
                   help="文件或目录路径 (相对于 cases/)")
    p.add_argument("-k", dest="filter_k", default=None,
                   help="子串匹配筛选用例名")
    p.add_argument("--bundle", default=None, help="使用的配置集")
    p.add_argument("--target", default=None, help="执行环境")
    p.add_argument("--collect-only", action="store_true",
                   help="只列出用例，不执行")
    p.add_argument("-v", "--verbose", action="count", default=1,
                   help="输出详细程度")


def dispatch(args: argparse.Namespace) -> bool:
    if args.command != COMMAND:
        return False

    from aitf.config import load_config
    cfg = load_config()
    cases_dir = cfg.project_root / "cases"
    db_path = cfg.build_root / "aitf.db"

    if not cases_dir.is_dir():
        print(f"Error: cases directory not found: {cases_dir}", file=sys.stderr)
        sys.exit(2)

    if args.collect_only:
        _collect_only(cases_dir, args.paths, args.filter_k)
        return True

    from aitf.tc.runner import run_tests
    execution_id, all_passed = run_tests(
        cases_dir, db_path,
        paths=args.paths or None,
        filter_k=args.filter_k,
        bundle=args.bundle,
        target=args.target,
        verbosity=args.verbose,
    )

    if not execution_id:
        print("No tests found.")
        sys.exit(0)

    sys.exit(0 if all_passed else 1)


def _collect_only(cases_dir: Path, paths: list[str] | None, filter_k: str | None):
    """List discovered tests without running them."""
    from aitf.tc.runner import _load_suites, _iter_tests

    suite = _load_suites(cases_dir, paths, filter_k)
    count = 0
    current_class = None
    for test in _iter_tests(suite):
        cls_name = test.__class__.__name__
        if cls_name != current_class:
            current_class = cls_name
            mod = test.__class__.__module__
            print(f"\n  {cls_name} ({mod})")
        print(f"    {test._testMethodName}")
        count += 1
    print(f"\n{count} test(s) collected.")
