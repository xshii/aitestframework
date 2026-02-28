# REQ-2 Golden数据维护

---
version: 1.1
date: 2026-02-27
status: draft
---

## 1. 概述

管理已确认正确的基准参考数据（Golden数据），用于测试时与实际输出进行比较。Golden数据可能存储在本地或远程服务器上，需要版本化管理和分发能力。

**架构原则：** 所有Golden数据管理逻辑实现在 Core API 层（`aitestframework.golden`），CLI 和 Web 均为薄壳调用层。

## 2. 需求详情

### REQ-2.1 数据注册表 [P0]

**描述：** 维护一个注册表，记录每份Golden数据的元信息。

**注册表字段：**

| 字段 | 类型 | 说明 |
|------|------|------|
| id | str | 唯一标识，格式：`<platform>/<operator>/<variant>` |
| name | str | 可读名称 |
| platform | str | npu / gpu / cpu |
| operator | str | 算子名或模型名 |
| dtype | str | 数据类型：fp32, fp16, int8, bf16, 自定义量化类型 |
| shape | list[int] | 张量维度 |
| format | str | 存储格式：raw, npy, npz, custom |
| file_path | str | 数据文件的相对路径 |
| checksum | str | SHA256校验和 |
| version | str | 数据版本号 |
| source | str | 来源说明 |
| created_at | datetime | 创建时间 |
| updated_at | datetime | 更新时间 |

**注册表存储：** YAML 清单文件为主（git 可追踪），SQLite 作为运行时查询缓存（从 YAML 自动重建，无需手动维护）

**YAML 清单示例：**

```yaml
# golden/registry/npu_conv2d.yaml
- id: npu/conv2d/fp32_3x3
  name: "Conv2D FP32 3x3 kernel"
  platform: npu
  operator: conv2d
  dtype: fp32
  shape: [1, 64, 112, 112]
  format: npy
  file_path: store/npu/conv2d/fp32_3x3_output.npy
  checksum: sha256:a1b2c3d4...
  version: "1.0"
  source: "simulator verified 2026-02-20"
```

**验收标准：**
- 注册表支持增删改查
- SQLite 缓存可从 YAML 自动重建（`aitf golden rebuild-cache`）
- 注册表中的 file_path 对应的文件必须存在且 checksum 匹配

### REQ-2.2 本地文件存储 [P0]

**描述：** Golden数据文件存储在本地文件系统。

**目录结构：**

```
golden/
├── registry/                    # YAML注册表
│   ├── npu_conv2d.yaml
│   ├── npu_matmul.yaml
│   └── ...
└── store/                       # 数据文件
    ├── npu/
    │   ├── conv2d/
    │   │   ├── fp32_3x3_input.npy
    │   │   ├── fp32_3x3_output.npy
    │   │   └── fp32_3x3_weight.npy
    │   └── matmul/
    │       └── ...
    ├── gpu/
    └── cpu/
```

**验收标准：**
- 数据文件按 `store/<platform>/<operator>/` 组织
- 文件命名规范：`<variant>_<role>.{npy,bin,npz}`
- 大文件（>50MB）在 .gitignore 中排除，仅在远程服务器存储

### REQ-2.3 远程同步 [P0]

**描述：** 支持在本地和远程服务器之间同步Golden数据。

**同步方式：** paramiko SFTP（默认，与 REQ-6 远程执行器复用同一 SSH 基础设施）

**配置：**

```yaml
# golden/remote.yaml
remotes:
  lab-server:
    host: 192.168.1.100
    user: test
    path: /data/golden/
    auth:
      method: key
      key_file: ~/.ssh/id_rsa
  ci-server:
    host: ci.internal.com
    user: jenkins
    path: /var/lib/golden/
    auth:
      method: key
      key_file: ~/.ssh/ci_key
```

**命令：**

```bash
# 从远程拉取
aitf golden pull --remote lab-server --operator conv2d

# 推送到远程
aitf golden push --remote lab-server --operator conv2d

# 全量同步
aitf golden sync --remote lab-server
```

**验收标准：**
- 拉取后本地数据 checksum 与注册表一致
- 推送前检查本地数据完整性
- 支持按算子/平台粒度的增量同步
- 网络中断后可重试续传

### REQ-2.4 版本管理 [P1]

**描述：** Golden数据支持版本化，可回退到历史版本。

**机制：**
- 注册表 YAML 文件随 git 版本管理
- 数据文件本身通过文件名后缀区分版本：`conv2d_fp32_3x3_output_v1.npy`
- 或通过目录区分：`store/npu/conv2d/v1/`, `store/npu/conv2d/v2/`
- 注册表中 version 字段标识当前使用的版本

**验收标准：**
- 可查看指定算子的Golden历史版本列表
- 可切换当前使用的Golden版本
- 版本切换不影响其他算子

### REQ-2.5 完整性校验 [P1]

**描述：** 对Golden数据文件进行完整性校验。

**机制：**
- 注册时自动计算 SHA256 checksum
- 同步/拉取后自动校验
- 提供手动校验命令

**命令：**

```bash
# 校验全部
aitf golden verify

# 校验指定算子
aitf golden verify --operator conv2d
```

**验收标准：**
- 校验失败时输出具体的文件路径和期望/实际 checksum
- 校验结果可输出为JSON格式

### REQ-2.6 多格式支持 [P1]

**描述：** 支持多种数据存储格式。

**格式列表：**

| 格式 | 后缀 | 说明 | 读写库 |
|------|------|------|--------|
| NumPy | .npy / .npz | 标准浮点张量 | numpy |
| Raw binary | .bin | 原始二进制，配合 dtype+shape 元信息 | struct |
| 自定义量化 | .qdata | 自定义量化格式（如 BFP、MX 格式） | 自研 |

**验收标准：**
- 注册表中 format 字段标识存储格式
- 所有格式都可加载为 numpy ndarray 进行比较
- 自定义格式通过插件机制注册

## 3. 技术选型

| 决策 | 选型 | 备选 | 理由 |
|------|------|------|------|
| 注册表存储 | YAML为主 + SQLite缓存 | 纯YAML, 纯SQLite | YAML可git追踪，SQLite缓存自动重建加速查询 |
| 远程同步 | paramiko SFTP | rsync, scp | 与REQ-6复用SSH基础设施，无需远端额外安装rsync |
| 校验算法 | SHA256 | MD5, CRC32 | 安全性和碰撞率的平衡 |
| 数据加载 | numpy | torch | numpy是基础依赖，torch过重 |

## 4. 数据模型

```python
@dataclass
class GoldenEntry:
    """Golden数据注册表条目"""
    id: str                 # npu/conv2d/fp32_3x3
    name: str
    platform: str
    operator: str
    dtype: str
    shape: list[int]
    format: str             # npy / bin / qdata
    file_path: str          # 相对于 golden/store/ 的路径
    checksum: str           # sha256:...
    version: str
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
class GoldenManager:
    def register(self, entry: GoldenEntry) -> None: ...
    def get(self, golden_id: str) -> GoldenEntry: ...
    def list(self, platform: str | None = None, operator: str | None = None) -> list[GoldenEntry]: ...
    def load_data(self, golden_id: str) -> np.ndarray: ...
    def verify(self, golden_id: str | None = None) -> list[VerifyResult]: ...
    def pull(self, remote: str, operator: str | None = None) -> SyncResult: ...
    def push(self, remote: str, operator: str | None = None) -> SyncResult: ...
    def list_versions(self, golden_id: str) -> list[str]: ...
    def switch_version(self, golden_id: str, version: str) -> None: ...
```

## 6. 依赖

- REQ-4（用例管理）：用例引用 Golden ID 作为期望结果
- REQ-6（执行框架）：执行时加载 Golden 数据进行比较
