"""Tests for deps.config — deps.yaml parsing."""

from __future__ import annotations

import pytest
import yaml

from aitf.deps.config import DepsConfig, detect_platform, load_deps_config, save_deps_config
from aitf.deps.types import DepsConfigError


class TestDetectPlatform:
    def test_returns_string(self):
        plat = detect_platform()
        assert isinstance(plat, str)
        assert "-" in plat  # e.g. "linux-x86_64"


class TestLoadDepsConfig:
    def test_load_full_config(self, deps_yaml, project_root):
        cfg = load_deps_config(deps_yaml)

        assert "npu-compiler" in cfg.toolchains
        tc = cfg.toolchains["npu-compiler"]
        assert tc.version == "2.1.0"
        assert tc.acquire.local_dir == "deps/uploads/"

        assert "npu-runtime" in cfg.repos
        repo = cfg.repos["npu-runtime"]
        assert repo.branch == "main"
        assert repo.resolved_ref == "main"
        assert repo.depth == 1

        assert "npu-v2.1" in cfg.bundles
        b = cfg.bundles["npu-v2.1"]
        assert b.status == "verified"
        assert b.toolchains == {"npu-compiler": "2.1.0"}

        assert cfg.active_bundle == "npu-v2.1"

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(DepsConfigError, match="not found"):
            load_deps_config(tmp_path / "nonexistent.yaml")

    def test_load_invalid_yaml(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("[ invalid: yaml: {")
        with pytest.raises(DepsConfigError, match="Failed to parse"):
            load_deps_config(bad)

    def test_load_non_dict(self, tmp_path):
        bad = tmp_path / "list.yaml"
        bad.write_text("- item1\n- item2\n")
        with pytest.raises(DepsConfigError, match="Expected a YAML mapping"):
            load_deps_config(bad)

    def test_load_empty_file(self, tmp_path):
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        cfg = load_deps_config(empty)
        assert cfg.toolchains == {}
        assert cfg.repos == {}
        assert cfg.bundles == {}

    def test_load_minimal(self, tmp_path):
        minimal = tmp_path / "deps.yaml"
        with open(minimal, "w") as fh:
            yaml.dump({"toolchains": {"cc": {"version": "1.0"}}}, fh)
        cfg = load_deps_config(minimal)
        assert len(cfg.toolchains) == 1
        assert cfg.toolchains["cc"].version == "1.0"

    def test_load_with_acquire_script(self, tmp_path):
        cfg_data = {
            "toolchains": {
                "cc": {
                    "version": "1.0",
                    "acquire": {"script": "fetch.sh", "install_dir": "opt/cc"},
                },
            },
        }
        p = tmp_path / "deps.yaml"
        with open(p, "w") as fh:
            yaml.dump(cfg_data, fh)
        cfg = load_deps_config(p)
        assert cfg.toolchains["cc"].acquire.script == "fetch.sh"
        assert cfg.toolchains["cc"].acquire.install_dir == "opt/cc"

    def test_load_with_srcpkg(self, tmp_path):
        cfg_data = {
            "toolchains": {
                "cc": {
                    "version": "1.0",
                    "acquire": {
                        "local_dir": "deps/uploads",
                        "srcpkg": "custom-name.tar.gz",
                    },
                },
            },
        }
        p = tmp_path / "deps.yaml"
        with open(p, "w") as fh:
            yaml.dump(cfg_data, fh)
        cfg = load_deps_config(p)
        assert cfg.toolchains["cc"].acquire.srcpkg == "custom-name.tar.gz"

    def test_repo_requires_ref(self, tmp_path):
        cfg_data = {"repos": {"r": {"url": "git@host:r.git"}}}
        p = tmp_path / "deps.yaml"
        with open(p, "w") as fh:
            yaml.dump(cfg_data, fh)
        with pytest.raises(DepsConfigError, match="branch / tag / commitno"):
            load_deps_config(p)

    def test_fractional_order(self, tmp_path):
        cfg_data = {
            "toolchains": {
                "a": {"version": "1.0", "order": 1.5},
                "b": {"version": "1.0", "order": 1.5},
            },
        }
        p = tmp_path / "deps.yaml"
        with open(p, "w") as fh:
            yaml.dump(cfg_data, fh)
        cfg = load_deps_config(p)
        assert cfg.toolchains["a"].order == 1.5
        assert cfg.toolchains["b"].order == 1.5


class TestSaveDepsConfig:
    def test_round_trip(self, deps_yaml, project_root):
        cfg = load_deps_config(deps_yaml)

        out = project_root / "deps_out.yaml"
        save_deps_config(cfg, out)

        cfg2 = load_deps_config(out)
        assert cfg2.toolchains.keys() == cfg.toolchains.keys()
        assert cfg2.repos.keys() == cfg.repos.keys()
        assert cfg2.bundles.keys() == cfg.bundles.keys()
        assert cfg2.active_bundle == cfg.active_bundle

    def test_save_empty_config(self, tmp_path):
        cfg = DepsConfig()
        out = tmp_path / "empty.yaml"
        save_deps_config(cfg, out)
        cfg2 = load_deps_config(out)
        assert cfg2.toolchains == {}
