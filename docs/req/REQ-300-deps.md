# REQ-DEP 依赖管理需求

---
id: REQ-DEP
title: 依赖管理需求
priority: P1
status: draft
parent: REQ-SYS
---

## 概述

管理外部工具链（仿真器、编译器、ESL模型等）的版本依赖。

**设计原则：**
- 只需配置一整套依赖版本
- 每次刷新时重新生成依赖关系表
- 保持简单，不做复杂的版本兼容性计算

**技术选型：**
| 功能 | 库 | 说明 |
|------|-----|------|
| HTTP下载 | httpx | 异步、HTTP/2、连接池 |
| 进度显示 | rich | 进度条+状态表格 |
| 配置解析 | pyyaml | YAML解析 |
| 命令行 | argparse | 标准库 |

---

## REQ-DEP-001 依赖配置

---
id: REQ-DEP-001
title: 依赖配置
priority: P1
status: draft
parent: REQ-DEP
---

### 描述

定义当前使用的一整套依赖版本。

### 配置格式

```yaml
# deps/config.yaml
version: "1.0"
generated_at: "2026-02-03T10:00:00Z"

dependencies:
  simulator:
    version: "2.0.0"
    url: "https://release.example.com/sim/npu-sim-2.0.0.tar.gz"
    sha256: "abc123..."

  compiler:
    version: "2.1.0"
    url: "https://release.example.com/compiler/npu-cc-2.1.0.tar.gz"
    sha256: "def456..."

  esl_model:
    version: "1.5.0"
    url: "https://release.example.com/esl/npu-esl-1.5.0.tar.gz"
    sha256: "ghi789..."
```

### 验收标准

1. 单一配置文件定义所有依赖
2. 包含下载URL和SHA256校验
3. 配置文件纳入版本控制

---

## REQ-DEP-002 依赖刷新

---
id: REQ-DEP-002
title: 依赖关系刷新
priority: P1
status: draft
parent: REQ-DEP
---

### 描述

刷新依赖配置时重新生成依赖关系表。

### 刷新流程

```
1. 编辑 deps/config.yaml 更新版本
2. 执行 deps refresh 命令
3. 生成新的依赖关系表 deps/resolved.yaml
4. 下载并校验新版本
```

### 生成的关系表

```yaml
# deps/resolved.yaml (自动生成)
resolved_at: "2026-02-03T10:30:00Z"
source: "deps/config.yaml"

components:
  simulator:
    version: "2.0.0"
    sha256: "abc123..."
    install_path: "build/toolchain/simulator"
    status: installed

  compiler:
    version: "2.1.0"
    sha256: "def456..."
    install_path: "build/toolchain/compiler"
    status: installed
```

### 下载实现（httpx）

```python
# deps/scripts/downloader.py
import httpx
import hashlib
from pathlib import Path

async def download_component(url: str, dest: Path, sha256: str,
                             timeout: float = 300.0) -> None:
    """
    使用httpx下载组件并校验SHA256

    Args:
        url: 下载地址
        dest: 目标路径
        sha256: 期望的SHA256值
        timeout: 超时时间（秒）
    """
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout),
        follow_redirects=True,
        http2=True,
    ) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()

            hasher = hashlib.sha256()
            total = int(response.headers.get("content-length", 0))

            with open(dest, "wb") as f:
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    f.write(chunk)
                    hasher.update(chunk)
                    # 更新进度条...

            # SHA256校验
            actual_sha256 = hasher.hexdigest()
            if actual_sha256 != sha256:
                dest.unlink()
                raise ValueError(
                    f"SHA256 mismatch: expected {sha256}, got {actual_sha256}"
                )
```

### httpx配置

```python
# 连接配置
client_config = {
    "timeout": httpx.Timeout(
        connect=10.0,      # 连接超时
        read=30.0,         # 读取超时
        write=10.0,        # 写入超时
        pool=5.0,          # 连接池超时
    ),
    "limits": httpx.Limits(
        max_connections=10,           # 最大连接数
        max_keepalive_connections=5,  # 保持连接数
    ),
    "http2": True,         # 启用HTTP/2
    "follow_redirects": True,
    "verify": True,        # SSL验证
}

# 重试配置
retry_config = {
    "max_retries": 3,
    "retry_on": [httpx.TimeoutException, httpx.NetworkError],
    "backoff_factor": 1.0,  # 重试间隔: 1s, 2s, 4s
}
```

### 验收标准

1. 刷新时重新生成完整关系表
2. 关系表记录安装状态
3. 支持增量更新（只下载变更的组件）
4. 使用httpx异步下载，支持HTTP/2
5. 下载超时可配置，默认300秒
6. 支持断点续传（如服务器支持Range）

---

## REQ-DEP-003 命令行接口

---
id: REQ-DEP-003
title: 命令行接口
priority: P1
status: draft
parent: REQ-DEP
---

### 描述

依赖管理CLI工具。

### 命令设计

```bash
# 刷新并安装
deps refresh              # 刷新依赖关系表
deps install              # 安装所有依赖

# 查询
deps list                 # 列出已安装
deps status               # 检查安装状态

# 清理
deps clean                # 清理安装目录
```

### 验收标准

1. 安装带进度显示（rich进度条）
2. SHA256校验失败则终止
3. 支持离线安装（本地缓存）
4. 下载失败自动重试（最多3次）
5. 支持并发下载多个组件
