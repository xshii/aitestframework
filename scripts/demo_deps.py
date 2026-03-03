#!/usr/bin/env python3
"""生成 deps demo 数据并启动 web 服务，方便预览 UI。

用法:
    python scripts/demo_deps.py          # 使用临时目录
    python scripts/demo_deps.py --keep   # 数据保留在 /tmp/aitf-demo/
"""
from __future__ import annotations

import io
import os
import sys
import tarfile
from pathlib import Path

import yaml


def _make_tar_gz(dest: Path, inner_name: str = "readme.txt", content: bytes = b"demo"):
    """创建一个最小 tar.gz 归档。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name=inner_name)
        info.size = len(content)
        tf.addfile(info, io.BytesIO(content))
    dest.write_bytes(buf.getvalue())


def setup_demo(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "build" / "cache").mkdir(parents=True, exist_ok=True)
    (root / "build" / "repos").mkdir(parents=True, exist_ok=True)
    uploads = root / "deps" / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)

    # -- 归档文件 (模拟多版本) --
    archives = [
        "npu-compiler-2.0.0.tar.gz",
        "npu-compiler-2.1.0.tar.gz",
        "npu-compiler-2.2.0.tar.gz",
        "json-c-0.16.tar.gz",
        "json-c-0.17.tar.gz",
        "libprotobuf-3.21.0.tar.gz",
        "libprotobuf-3.25.1.tar.gz",
        "opencv-4.8.0.tar.gz",
        "opencv-4.9.0.tar.gz",
        "gtest-1.14.0.tar.gz",
    ]
    for name in archives:
        _make_tar_gz(uploads / name)

    # -- deps.yaml --
    cfg = {
        "server": "http://10.0.0.100:5000",
        "toolchains": {
            "npu-compiler": {
                "version": "2.1.0",
                "sha256": {},
                "bin_dir": "bin",
                "env": {"NPU_CC": "{install_dir}/bin/npu-gcc"},
                "acquire": {"local_dir": "deps/uploads/"},
            },
        },
        "libraries": {
            "json-c": {
                "version": "0.17",
                "sha256": "",
                "build_system": "cmake",
                "acquire": {"local_dir": "deps/uploads/"},
            },
            "libprotobuf": {
                "version": "3.25.1",
                "sha256": "",
                "build_system": "cmake",
                "acquire": {"local_dir": "deps/uploads/"},
            },
            "opencv": {
                "version": "4.9.0",
                "sha256": "",
                "build_system": "cmake",
                "acquire": {"local_dir": "deps/uploads/"},
            },
            "gtest": {
                "version": "1.14.0",
                "sha256": "",
                "build_system": "cmake",
                "acquire": {"local_dir": "deps/uploads/"},
            },
        },
        "repos": {
            "npu-runtime": {
                "url": "git@10.0.0.1:hw/npu-runtime.git",
                "ref": "main",
                "depth": 1,
            },
            "npu-driver": {
                "url": "git@10.0.0.1:hw/npu-driver.git",
                "ref": "v3.2.0",
                "depth": 1,
            },
            "test-vectors": {
                "url": "git@10.0.0.1:qa/test-vectors.git",
                "ref": "develop",
                "depth": 1,
            },
        },
        "bundles": {
            "npu-v2.1": {
                "description": "NPU 测试环境 v2.1",
                "status": "verified",
                "toolchains": {"npu-compiler": "2.1.0"},
                "libraries": {"json-c": "0.17", "libprotobuf": "3.25.1"},
                "repos": {"npu-runtime": "main", "npu-driver": "v3.2.0"},
                "env": {"NPU_SDK_VERSION": "2.1"},
            },
            "npu-v2.0-legacy": {
                "description": "NPU 测试环境 v2.0 (旧版)",
                "status": "deprecated",
                "toolchains": {"npu-compiler": "2.0.0"},
                "libraries": {"json-c": "0.16"},
                "repos": {"npu-runtime": "main"},
                "env": {"NPU_SDK_VERSION": "2.0"},
            },
            "cv-bench": {
                "description": "CV 性能基准测试集",
                "status": "testing",
                "toolchains": {"npu-compiler": "2.2.0"},
                "libraries": {"opencv": "4.9.0", "gtest": "1.14.0"},
                "repos": {"test-vectors": "develop"},
                "env": {},
            },
        },
        "active": "npu-v2.1",
    }
    with open(root / "deps.yaml", "w") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # -- config.yaml (standalone demo) --
    with open(root / "config.yaml", "w") as fh:
        fh.write(
            "# AITF 全局配置\n"
            "server: ''          # 服务器 IP（空 = 单机模式）\n"
            "port: 5000\n"
            "# build_root: build         # 构建根目录（默认: <项目>/build）\n"
            "# datastore_dir: datastore  # 数据存储目录（默认: <项目>/datastore）\n"
        )

    print(f"Demo 数据已生成: {root}")
    print(f"  config.yaml      — standalone 模式 (server 为空)")
    print(f"  deps.yaml        — 1 工具链, 4 库, 3 仓库")
    print(f"  deps/uploads/    — {len(archives)} 个归档")
    print(f"  bundles          — 3 个配置集")


def main():
    keep = "--keep" in sys.argv

    if keep:
        root = Path("/tmp/aitf-demo")
    else:
        import tempfile
        root = Path(tempfile.mkdtemp(prefix="aitf-demo-"))

    setup_demo(root)

    # 启动 web
    print(f"\n启动 Web 服务 (项目根: {root}) ...")
    print("打开浏览器访问: http://127.0.0.1:5000/#tab-deps\n")

    from aitf.config import load_config
    from aitf.deps.manager import DepsManager
    from aitf.web.app import create_app

    aitf_cfg = load_config(project_root=root)
    app = create_app(aitf_config=aitf_cfg)
    app.config["deps_manager"] = DepsManager(
        project_root=str(root), deps_file="deps.yaml", build_dir="build",
    )
    app.run(host=aitf_cfg.bind_host, port=aitf_cfg.port, debug=False)


if __name__ == "__main__":
    main()
