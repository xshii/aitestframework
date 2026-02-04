# DEP 依赖管理模块设计

---
module: DEP
version: 1.0
date: 2026-02-04
status: draft
requirements: REQ-DEP-001~003
domain: Generic
---

## 1. 模块概述

### 1.1 职责

管理外部工具链的版本和下载：
- 依赖配置管理
- 版本下载与校验
- 安装状态跟踪

### 1.2 DDD定位

- **限界上下文**：通用域（可替换）
- **独立运行**：不依赖核心域

### 1.3 设计原则

- 只需配置一整套依赖版本
- 每次刷新时重新生成依赖关系表
- 保持简单，不做复杂的版本兼容性计算

---

## 2. 文件结构

```
deps/
├── config.yaml         # 依赖配置（用户编辑）
├── resolved.yaml       # 解析后的状态（自动生成）
└── scripts/
    ├── __init__.py
    ├── manager.py      # 主入口
    ├── downloader.py   # 下载器
    └── installer.py    # 安装器

tools/deps/             # 命令行工具
└── __main__.py
```

---

## 3. 配置格式

### 3.1 deps/config.yaml

```yaml
# 依赖配置文件
version: "1.0"
generated_at: "2026-02-04T10:00:00Z"

# 安装路径
install_base: "build/toolchain"

# 依赖定义
dependencies:
  simulator:
    version: "2.0.0"
    url: "https://release.example.com/sim/npu-sim-2.0.0.tar.gz"
    sha256: "abc123def456..."
    extract: true
    bin_path: "bin/simulator"

  compiler:
    version: "2.1.0"
    url: "https://release.example.com/compiler/npu-cc-2.1.0.tar.gz"
    sha256: "789ghi012..."
    extract: true
    bin_path: "bin/npu-cc"

  esl_model:
    version: "1.5.0"
    url: "https://release.example.com/esl/npu-esl-1.5.0.tar.gz"
    sha256: "jkl345mno..."
    extract: true
    optional: true  # 可选依赖

# 下载配置
download:
  timeout: 300        # 秒
  retries: 3
  concurrent: 2       # 并发下载数
```

### 3.2 deps/resolved.yaml（自动生成）

```yaml
# 自动生成，不要手动编辑
resolved_at: "2026-02-04T10:30:00Z"
source: "deps/config.yaml"
source_sha256: "..."

components:
  simulator:
    version: "2.0.0"
    sha256: "abc123def456..."
    install_path: "build/toolchain/simulator"
    bin_path: "build/toolchain/simulator/bin/simulator"
    status: installed  # pending | downloading | installed | failed

  compiler:
    version: "2.1.0"
    sha256: "789ghi012..."
    install_path: "build/toolchain/compiler"
    bin_path: "build/toolchain/compiler/bin/npu-cc"
    status: installed

  esl_model:
    version: "1.5.0"
    sha256: "jkl345mno..."
    install_path: "build/toolchain/esl_model"
    status: pending  # 可选，未安装
```

---

## 4. 命令行接口

### 4.1 命令设计

```bash
# 刷新依赖关系表
python -m tools.deps refresh

# 安装所有依赖
python -m tools.deps install

# 安装指定依赖
python -m tools.deps install simulator compiler

# 查看状态
python -m tools.deps status

# 列出已安装
python -m tools.deps list

# 清理安装目录
python -m tools.deps clean

# 检查更新
python -m tools.deps check
```

### 4.2 输出示例

```
$ python -m tools.deps status

Dependency Status
=================
Component    Version   Status      Path
---------    -------   ------      ----
simulator    2.0.0     installed   build/toolchain/simulator
compiler     2.1.0     installed   build/toolchain/compiler
esl_model    1.5.0     pending     (optional)

Summary: 2 installed, 1 pending (optional)
```

---

## 5. 下载实现

### 5.1 使用httpx异步下载

```python
# deps/scripts/downloader.py
import asyncio
import hashlib
from pathlib import Path
import httpx
from rich.progress import Progress, TaskID

class Downloader:
    def __init__(self, timeout: float = 300.0, retries: int = 3):
        self.timeout = timeout
        self.retries = retries

    async def download(
        self,
        url: str,
        dest: Path,
        sha256: str,
        progress: Progress | None = None,
        task_id: TaskID | None = None,
    ) -> None:
        """下载文件并校验SHA256"""

        for attempt in range(self.retries):
            try:
                await self._download_once(url, dest, sha256, progress, task_id)
                return
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                if attempt == self.retries - 1:
                    raise
                wait = 2 ** attempt  # 指数退避
                await asyncio.sleep(wait)

    async def _download_once(
        self,
        url: str,
        dest: Path,
        sha256: str,
        progress: Progress | None,
        task_id: TaskID | None,
    ) -> None:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            follow_redirects=True,
            http2=True,
        ) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()

                total = int(response.headers.get("content-length", 0))
                if progress and task_id:
                    progress.update(task_id, total=total)

                hasher = hashlib.sha256()
                dest.parent.mkdir(parents=True, exist_ok=True)

                with open(dest, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        f.write(chunk)
                        hasher.update(chunk)
                        if progress and task_id:
                            progress.advance(task_id, len(chunk))

                # 校验SHA256
                actual_sha256 = hasher.hexdigest()
                if actual_sha256 != sha256:
                    dest.unlink()
                    raise ValueError(
                        f"SHA256 mismatch: expected {sha256}, got {actual_sha256}"
                    )
```

### 5.2 并发下载

```python
async def download_all(
    components: list[ComponentConfig],
    concurrent: int = 2,
) -> dict[str, bool]:
    """并发下载多个组件"""
    semaphore = asyncio.Semaphore(concurrent)
    downloader = Downloader()
    results = {}

    async def download_one(comp: ComponentConfig) -> tuple[str, bool]:
        async with semaphore:
            try:
                await downloader.download(
                    comp.url,
                    comp.download_path,
                    comp.sha256,
                )
                return comp.name, True
            except Exception as e:
                print(f"Failed to download {comp.name}: {e}")
                return comp.name, False

    tasks = [download_one(comp) for comp in components]
    for coro in asyncio.as_completed(tasks):
        name, success = await coro
        results[name] = success

    return results
```

---

## 6. 安装实现

### 6.1 解压安装

```python
# deps/scripts/installer.py
import tarfile
import zipfile
from pathlib import Path

class Installer:
    def install(self, archive_path: Path, install_path: Path) -> None:
        """解压安装"""
        install_path.mkdir(parents=True, exist_ok=True)

        if archive_path.suffix == ".gz" or archive_path.name.endswith(".tar.gz"):
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(install_path)
        elif archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(install_path)
        else:
            raise ValueError(f"Unsupported archive format: {archive_path}")

    def verify(self, install_path: Path, bin_path: str | None) -> bool:
        """验证安装"""
        if not install_path.exists():
            return False

        if bin_path:
            full_bin_path = install_path / bin_path
            return full_bin_path.exists()

        return True
```

---

## 7. 进度显示

### 7.1 使用rich

```python
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, DownloadColumn
from rich.table import Table

def show_download_progress(components: list[ComponentConfig]):
    """显示下载进度"""
    with Progress(
        TextColumn("[bold blue]{task.fields[name]}"),
        BarColumn(),
        DownloadColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        tasks = {}
        for comp in components:
            tasks[comp.name] = progress.add_task(
                comp.name,
                name=comp.name,
                total=None,
            )

        # 执行下载...

def show_status_table(components: list[ComponentStatus]):
    """显示状态表格"""
    console = Console()
    table = Table(title="Dependency Status")

    table.add_column("Component", style="cyan")
    table.add_column("Version")
    table.add_column("Status")
    table.add_column("Path")

    for comp in components:
        status_style = {
            "installed": "green",
            "pending": "yellow",
            "failed": "red",
        }.get(comp.status, "white")

        table.add_row(
            comp.name,
            comp.version,
            f"[{status_style}]{comp.status}[/]",
            str(comp.install_path) if comp.install_path else "-",
        )

    console.print(table)
```

---

## 8. 数据模型

### 8.1 Python类定义

```python
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

class ComponentStatus(Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    INSTALLED = "installed"
    FAILED = "failed"

@dataclass
class ComponentConfig:
    name: str
    version: str
    url: str
    sha256: str
    extract: bool = True
    bin_path: str | None = None
    optional: bool = False

@dataclass
class ResolvedComponent:
    name: str
    version: str
    sha256: str
    install_path: Path
    bin_path: Path | None
    status: ComponentStatus

@dataclass
class DepsConfig:
    version: str
    install_base: Path
    dependencies: dict[str, ComponentConfig]
    download_timeout: float = 300.0
    download_retries: int = 3
    download_concurrent: int = 2
```

---

## 9. 错误处理

### 9.1 错误类型

```python
class DepsError(Exception):
    """依赖管理错误基类"""
    pass

class ConfigError(DepsError):
    """配置错误"""
    pass

class DownloadError(DepsError):
    """下载错误"""
    pass

class ChecksumError(DepsError):
    """校验错误"""
    pass

class InstallError(DepsError):
    """安装错误"""
    pass
```

### 9.2 错误处理策略

| 错误 | 处理 |
|------|------|
| 网络超时 | 重试（最多3次，指数退避） |
| SHA256不匹配 | 删除文件，报错 |
| 解压失败 | 清理目录，报错 |
| 可选依赖失败 | 警告，继续 |
| 必须依赖失败 | 终止 |

---

## 10. 使用示例

### 10.1 命令行使用

```bash
# 首次安装
$ python -m tools.deps install

Downloading dependencies...
simulator   ████████████████████████ 100% 45.2 MB
compiler    ████████████████████████ 100% 128.7 MB

Installing...
simulator   [OK] build/toolchain/simulator
compiler    [OK] build/toolchain/compiler

All dependencies installed successfully.

# 检查状态
$ python -m tools.deps status

Component    Version   Status      Path
---------    -------   ------      ----
simulator    2.0.0     installed   build/toolchain/simulator
compiler     2.1.0     installed   build/toolchain/compiler
```

### 10.2 Python API使用

```python
from deps.scripts.manager import DepsManager

manager = DepsManager("deps/config.yaml")

# 刷新解析
manager.refresh()

# 安装所有
manager.install_all()

# 获取组件路径
sim_path = manager.get_bin_path("simulator")
```

---

## 11. 需求追溯

| 需求ID | 需求标题 | 设计章节 |
|--------|----------|----------|
| REQ-DEP-001 | 依赖配置 | 3 |
| REQ-DEP-002 | 依赖关系刷新 | 5, 6 |
| REQ-DEP-003 | 命令行接口 | 4 |
