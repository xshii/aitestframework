# REQ-2 测试数据管理

---
version: 2.0
date: 2026-02-28
status: draft
---

## 1. 概述

按用例管理测试所需的全部二进制数据资产，支持在本地和远程服务器之间上传/下载。每个用例包含四类数据：

| 数据类型 | 方向 | 说明 |
|---------|------|------|
| 权重 (weights) | 服务器 → 本地 | 模型权重文件，运行前拉取 |
| 输入 (inputs) | 服务器 → 本地 | 模型输入数据，运行前拉取 |
| 预期输出 (golden) | 服务器 → 本地 | 已确认正确的基准结果，用于外部比对工具 |
| 实际输出 (artifacts) | 本地 → 服务器 | 模型运行产出的实际结果，运行后归档 |

**不包含**数据比对逻辑，比对由外部工具或其他工程完成。本模块只负责数据的存储、元信息维护和搬运。

**架构原则：** 所有逻辑实现在 Core API 层（`aitestframework.datastore`），CLI 和 Web 均为薄壳调用层。

## 2. 需求详情

### REQ-2.1 用例数据包 [P0]

**描述：** 每个用例对应一个数据包，包含该用例运行所需的全部二进制文件。

**数据包结构：**

```
<case_id>/
├── weights/          # 权重文件
│   ├── manifest.txt  # 权重 manifest（与 stubs weight 模块兼容）
│   ├── conv_w.bin
│   ├── conv_b.bin
│   └── fc_w.bin
├── inputs/           # 输入数据
│   ├── input_0.bin
│   └── input_1.bin
├── golden/           # 预期输出（基准）
│   └── output.bin
└── artifacts/        # 实际输出（运行后产生）
    └── output.bin
```

**用例 ID 格式：** `<platform>/<model>/<variant>`，例如 `npu/tdd/fp32_basic`

**验收标准：**
- 四类数据目录结构固定，缺失的类型允许目录为空
- weights/manifest.txt 格式与 stubs 框架的权重 manifest 一致（多组、base_addr + swap）
- artifacts 目录在运行前可为空，运行后由框架填充

### REQ-2.2 数据注册表 [P0]

**描述：** 维护注册表，记录每个用例数据包的元信息。

**注册表字段：**

| 字段 | 类型 | 说明 |
|------|------|------|
| case_id | str | 用例唯一标识：`<platform>/<model>/<variant>` |
| name | str | 可读名称 |
| platform | str | npu / gpu / cpu |
| model | str | 模型名 |
| variant | str | 变体（数据类型、配置等） |
| version | str | 数据版本号 |
| files | dict | 各类型文件清单及 checksum |
| source | str | 来源说明 |
| created_at | datetime | 创建时间 |
| updated_at | datetime | 更新时间 |

**files 字段结构：**

```yaml
files:
  weights:
    - path: weights/conv_w.bin
      size: 4096
      checksum: sha256:a1b2c3...
    - path: weights/conv_b.bin
      size: 256
      checksum: sha256:d4e5f6...
  inputs:
    - path: inputs/input_0.bin
      size: 8192
      checksum: sha256:...
  golden:
    - path: golden/output.bin
      size: 1024
      checksum: sha256:...
  artifacts: []  # 运行后填充
```

**注册表存储：** YAML 清单文件为主（git 可追踪），SQLite 作为运行时查询缓存（从 YAML 自动重建）

**YAML 清单示例：**

```yaml
# datastore/registry/npu_tdd.yaml
- case_id: npu/tdd/fp32_basic
  name: "TDD FP32 基础用例"
  platform: npu
  model: tdd
  variant: fp32_basic
  version: "1.0"
  files:
    weights:
      - path: weights/tdd_conv_w.bin
        size: 32
        checksum: sha256:a1b2c3d4...
      - path: weights/tdd_conv_b.bin
        size: 8
        checksum: sha256:e5f6a7b8...
      - path: weights/tdd_fc_w.bin
        size: 16
        checksum: sha256:c9d0e1f2...
    inputs:
      - path: inputs/input_0.bin
        size: 256
        checksum: sha256:...
    golden:
      - path: golden/output.bin
        size: 56
        checksum: sha256:...
    artifacts: []
  source: "simulator verified 2026-02-28"
```

**验收标准：**
- 注册表支持增删改查
- SQLite 缓存可从 YAML 自动重建（`aitf data rebuild-cache`）
- 注册时自动计算文件 size 和 SHA256 checksum

### REQ-2.3 本地文件存储 [P0]

**描述：** 数据文件按用例组织在本地文件系统。

**目录结构：**

```
datastore/
├── registry/                          # YAML 注册表
│   ├── npu_tdd.yaml
│   ├── npu_fdd.yaml
│   └── ...
└── store/                             # 数据文件
    └── <platform>/
        └── <model>/
            └── <variant>/
                ├── weights/
                │   ├── manifest.txt
                │   ├── conv_w.bin
                │   └── ...
                ├── inputs/
                │   └── input_0.bin
                ├── golden/
                │   └── output.bin
                └── artifacts/
                    └── output.bin
```

**验收标准：**
- 数据文件按 `store/<platform>/<model>/<variant>/` 组织
- 大文件（>50MB）在 .gitignore 中排除，仅在远程服务器存储
- 注册表 YAML 始终可 git 追踪

### REQ-2.4 远程同步 [P0]

**描述：** 支持在本地和远程服务器之间按用例上传/下载数据。

**同步方式：** paramiko SFTP（与 REQ-6 远程执行器复用同一 SSH 基础设施）

**配置：**

```yaml
# datastore/remote.yaml
remotes:
  lab-server:
    host: 192.168.1.100
    user: test
    path: /data/datastore/
    auth:
      method: key
      key_file: ~/.ssh/id_rsa
  ci-server:
    host: ci.internal.com
    user: jenkins
    path: /var/lib/datastore/
    auth:
      method: key
      key_file: ~/.ssh/ci_key
```

**操作：**

```bash
# 拉取用例数据（weights + inputs + golden）
aitf data pull --remote lab-server --case npu/tdd/fp32_basic

# 拉取指定平台/模型的全部用例
aitf data pull --remote lab-server --platform npu --model tdd

# 归档实际输出到远程
aitf data push-artifacts --remote lab-server --case npu/tdd/fp32_basic

# 上传完整用例数据包（注册新用例时）
aitf data push --remote lab-server --case npu/tdd/fp32_basic

# 全量同步
aitf data sync --remote lab-server
```

**典型工作流：**

```
运行前：aitf data pull  → 拉取 weights + inputs + golden
运行中：stubs 执行      → 产出 artifacts
运行后：aitf data push-artifacts → 归档实际输出到服务器
```

**验收标准：**
- pull 拉取 weights/inputs/golden 三类文件，不拉取 artifacts
- push-artifacts 仅上传 artifacts 目录
- push 上传完整数据包（四类全部）
- 拉取后自动校验 checksum
- 支持按用例/模型/平台粒度操作
- 网络中断后可重试续传

### REQ-2.5 版本管理 [P1]

**描述：** 用例数据包支持版本化，可回退到历史版本。

**机制：**
- 注册表 YAML 随 git 版本管理
- 数据文件通过目录区分版本：`store/npu/tdd/fp32_basic/v1/`, `store/npu/tdd/fp32_basic/v2/`
- 注册表中 version 字段标识当前使用的版本

**命令：**

```bash
# 查看版本列表
aitf data versions --case npu/tdd/fp32_basic

# 切换版本
aitf data switch-version --case npu/tdd/fp32_basic --version 2.0
```

**验收标准：**
- 可查看指定用例的历史版本列表
- 可切换当前使用的数据版本
- 版本切换不影响其他用例

### REQ-2.6 完整性校验 [P1]

**描述：** 对数据文件进行完整性校验。

**机制：**
- 注册时自动计算 SHA256 checksum
- pull/push 后自动校验
- 提供手动校验命令

**命令：**

```bash
# 校验全部
aitf data verify

# 校验指定用例
aitf data verify --case npu/tdd/fp32_basic

# 校验指定模型的全部用例
aitf data verify --platform npu --model tdd
```

**验收标准：**
- 校验失败时输出具体的文件路径和期望/实际 checksum
- 校验结果可输出为 JSON 格式

## 3. 技术选型

| 决策 | 选型 | 备选 | 理由 |
|------|------|------|------|
| 注册表存储 | YAML 为主 + SQLite 缓存 | 纯 YAML, 纯 SQLite | YAML 可 git 追踪，SQLite 缓存自动重建加速查询 |
| 远程同步 | paramiko SFTP | rsync, scp | 与 REQ-6 复用 SSH 基础设施，无需远端额外安装 |
| 校验算法 | SHA256 | MD5, CRC32 | 安全性和碰撞率的平衡 |

## 4. 数据模型

```python
@dataclass
class FileEntry:
    """单个文件的元信息"""
    path: str               # 相对于用例目录的路径
    size: int               # 文件大小（字节）
    checksum: str           # sha256:...

@dataclass
class CaseData:
    """用例数据包注册表条目"""
    case_id: str            # npu/tdd/fp32_basic
    name: str
    platform: str
    model: str
    variant: str
    version: str
    files: dict[str, list[FileEntry]]  # weights / inputs / golden / artifacts
    source: str
    created_at: datetime
    updated_at: datetime

@dataclass
class RemoteConfig:
    """远程服务器配置"""
    name: str
    host: str
    user: str
    path: str
    ssh_key: str | None = None
```

## 5. 对外接口

```python
class DataStoreManager:
    """测试数据资产管理器"""

    # --- 注册表 ---
    def register(self, case_id: str, local_path: str) -> CaseData: ...
    def get(self, case_id: str) -> CaseData: ...
    def delete(self, case_id: str) -> None: ...
    def list(self, platform: str | None = None,
             model: str | None = None) -> list[CaseData]: ...
    def rebuild_cache(self) -> None: ...

    # --- 完整性 ---
    def verify(self, case_id: str | None = None) -> list[VerifyResult]: ...

    # --- 远程同步 ---
    def pull(self, remote: str, case_id: str | None = None,
             platform: str | None = None,
             model: str | None = None) -> SyncResult: ...
    def push(self, remote: str, case_id: str) -> SyncResult: ...
    def push_artifacts(self, remote: str, case_id: str,
                       artifacts_dir: str) -> SyncResult: ...

    # --- 版本 ---
    def list_versions(self, case_id: str) -> list[str]: ...
    def switch_version(self, case_id: str, version: str) -> None: ...

    # --- 本地路径解析（供 stubs 等消费方使用） ---
    def get_weights_dir(self, case_id: str) -> str: ...
    def get_inputs_dir(self, case_id: str) -> str: ...
    def get_golden_dir(self, case_id: str) -> str: ...
    def get_artifacts_dir(self, case_id: str) -> str: ...
```

## 6. 与 stubs 框架集成

```
aitf data pull --case npu/tdd/fp32_basic
    ↓
datastore/store/npu/tdd/fp32_basic/
├── weights/manifest.txt  →  stubs --weight-manifest
├── weights/*.bin         →  stubs --weight-dir
├── inputs/*.bin          →  模型 run 中读取
└── golden/*.bin          →  stubs --golden-dir（或外部比对工具）

stubs 运行 → result_export → artifacts/output.bin
    ↓
aitf data push-artifacts --case npu/tdd/fp32_basic
```

路径解析示例：

```python
dm = DataStoreManager("datastore/")
case = "npu/tdd/fp32_basic"

# 获取 stubs 运行所需的路径
weight_manifest = f"{dm.get_weights_dir(case)}/manifest.txt"
weight_dir = dm.get_weights_dir(case)
golden_dir = dm.get_golden_dir(case)
result_dir = dm.get_artifacts_dir(case)
```

## 7. 依赖

- REQ-1（stubs 框架）：weights/manifest.txt 格式兼容，result_export 产出归档为 artifacts
- REQ-4（用例管理）：用例引用 case_id 关联数据包
- REQ-6（执行框架）：执行前 pull、执行后 push-artifacts
