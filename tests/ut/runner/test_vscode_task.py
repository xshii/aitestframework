"""Tests for aitf.runner.vscode_task."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from aitf.runner.vscode_task import (
    _load_tasks,
    _resolve_order,
    run_vscode_task,
)


def _make_repo(tasks: list[dict]) -> Path:
    """Create a temp repo dir with .vscode/tasks.json."""
    repo = Path(tempfile.mkdtemp())
    vscode = repo / ".vscode"
    vscode.mkdir()
    (vscode / "tasks.json").write_text(json.dumps({
        "version": "2.0.0",
        "tasks": tasks,
    }))
    return repo


class TestLoadTasks(unittest.TestCase):
    def test_basic(self):
        repo = _make_repo([
            {"label": "build", "type": "shell", "command": "echo build"},
            {"label": "test", "type": "shell", "command": "echo test"},
        ])
        tasks = _load_tasks(repo)
        self.assertIn("build", tasks)
        self.assertIn("test", tasks)
        self.assertEqual(tasks["build"]["command"], "echo build")

    def test_missing_file(self):
        repo = Path(tempfile.mkdtemp())
        with self.assertRaises(FileNotFoundError):
            _load_tasks(repo)

    def test_empty_tasks(self):
        repo = _make_repo([])
        tasks = _load_tasks(repo)
        self.assertEqual(tasks, {})


class TestResolveOrder(unittest.TestCase):
    def test_no_deps(self):
        tasks = {"a": {"command": "echo a"}}
        self.assertEqual(_resolve_order("a", tasks), ["a"])

    def test_single_dep(self):
        tasks = {
            "build": {"command": "echo build"},
            "run": {"command": "echo run", "dependsOn": "build"},
        }
        order = _resolve_order("run", tasks)
        self.assertEqual(order, ["build", "run"])

    def test_dep_as_list(self):
        tasks = {
            "a": {"command": "echo a"},
            "b": {"command": "echo b"},
            "c": {"command": "echo c", "dependsOn": ["a", "b"]},
        }
        order = _resolve_order("c", tasks)
        self.assertEqual(order, ["a", "b", "c"])

    def test_chain(self):
        tasks = {
            "a": {"command": "echo a"},
            "b": {"command": "echo b", "dependsOn": "a"},
            "c": {"command": "echo c", "dependsOn": "b"},
        }
        order = _resolve_order("c", tasks)
        self.assertEqual(order, ["a", "b", "c"])

    def test_missing_label(self):
        with self.assertRaises(KeyError):
            _resolve_order("missing", {"a": {}})

    def test_circular(self):
        tasks = {
            "a": {"command": "echo a", "dependsOn": "b"},
            "b": {"command": "echo b", "dependsOn": "a"},
        }
        with self.assertRaises(ValueError):
            _resolve_order("a", tasks)

    def test_dedup(self):
        """Diamond dependency: c depends on a and b, both depend on base."""
        tasks = {
            "base": {"command": "echo base"},
            "a": {"command": "echo a", "dependsOn": "base"},
            "b": {"command": "echo b", "dependsOn": "base"},
            "c": {"command": "echo c", "dependsOn": ["a", "b"]},
        }
        order = _resolve_order("c", tasks)
        self.assertEqual(order.count("base"), 1)
        self.assertIn("c", order)
        self.assertTrue(order.index("base") < order.index("a"))
        self.assertTrue(order.index("base") < order.index("b"))


class TestRunVscodeTask(unittest.TestCase):
    def test_simple_echo(self):
        repo = _make_repo([
            {"label": "hello", "type": "shell", "command": "echo hello-world"},
        ])
        result = run_vscode_task(repo, "hello")
        self.assertEqual(result.returncode, 0)
        self.assertIn("hello-world", result.stdout)

    def test_with_deps(self):
        repo = _make_repo([
            {"label": "step1", "type": "shell", "command": "echo step1"},
            {"label": "step2", "type": "shell", "command": "echo step2",
             "dependsOn": "step1"},
        ])
        result = run_vscode_task(repo, "step2")
        self.assertEqual(result.returncode, 0)
        self.assertIn("step2", result.stdout)

    def test_skip_deps(self):
        repo = _make_repo([
            {"label": "step1", "type": "shell", "command": "echo step1"},
            {"label": "step2", "type": "shell", "command": "echo step2",
             "dependsOn": "step1"},
        ])
        result = run_vscode_task(repo, "step2", skip_deps=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("step2", result.stdout)

    def test_dep_failure_raises(self):
        repo = _make_repo([
            {"label": "fail", "type": "shell", "command": "exit 1"},
            {"label": "run", "type": "shell", "command": "echo ok",
             "dependsOn": "fail"},
        ])
        with self.assertRaises(RuntimeError):
            run_vscode_task(repo, "run")

    def test_env_vars_passed(self):
        repo = _make_repo([
            {"label": "env", "type": "shell",
             "command": "echo $MY_TEST_VAR"},
        ])
        result = run_vscode_task(repo, "env",
                                  env_vars={"MY_TEST_VAR": "42"})
        self.assertEqual(result.returncode, 0)
        self.assertIn("42", result.stdout)

    def test_missing_label_raises(self):
        repo = _make_repo([
            {"label": "a", "type": "shell", "command": "echo a"},
        ])
        with self.assertRaises(KeyError):
            run_vscode_task(repo, "nonexistent")

    def test_missing_tasks_json_raises(self):
        repo = Path(tempfile.mkdtemp())
        with self.assertRaises(FileNotFoundError):
            run_vscode_task(repo, "anything")


if __name__ == "__main__":
    unittest.main(verbosity=2)
