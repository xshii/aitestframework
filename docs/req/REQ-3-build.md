# REQ-3 版本构建管理

---
version: 2.0
date: 2026-02-27
status: draft
---

## 1. 概述

统一管理桩代码编译所需的工具链（编译器、仿真器）、第三方C/C++库、代码仓库依赖以及Python依赖。

**环境约束：**
- 内网环境，无法访问外网
- 禁止使用 Docker / 虚拟机
- 只有 SSH 服务器可用（无 HTTP 文件服务器、无 NFS）

**架构原则：** 所有依赖管理和构建逻辑实现在 Core API 层（`aitestframework.deps` / `aitestframework.build`），CLI 和 Web 均为薄壳调用层。

## 2. 需求详情

### REQ-3.1 统一依赖配置 [P0]

**描述：** 所有依赖（工具链、C/C++库、代码仓库）在项目根目录的单一文件 `deps.yaml` 中声明。

**配置文件：**

```yaml
# deps.yaml（项目根目录）

# ============ 工具链 ============
toolchains:
  npu-compiler:
    version: "2.1.0"
    sha256:
      linux-x86_64: "abc123..."
      linux-aarch64: "def456..."
    bin_dir: "bin"
    env:
      NPU_CC: "{install_dir}/bin/npu-gcc"
      NPU_CXX: "{install_dir}/bin/npu-g++"
    acquire:
      local_dir: "deps/uploads/"           # 方式1: 本地上传
      script: "scripts/fetch_npu_compiler.sh"  # 方式2: 获取脚本

  simulator:
    version: "3.4.1"
    sha256:
      linux-x86_64: "789abc..."
    bin_dir: "bin"
    env:
      SIM_PATH: "{install_dir}"
    acquire:
      script: "scripts/fetch_simulator.sh"

# ============ C/C++ 库 ============
libraries:
  json-c:
    version: "0.17"
    sha256: "abc..."
    build_system: cmake
    cmake_args:
      - "-DBUILD_SHARED_LIBS=OFF"
      - "-DBUILD_TESTING=OFF"
    acquire:
      script: "scripts/fetch_json_c.sh"
      local_dir: "deps/uploads/"

  unity:
    version: "2.6.0"
    sha256: "def..."
    build_system: cmake
    build_script: "scripts/build_unity.sh"   # 自定义构建脚本
    acquire:
      local_dir: "deps/uploads/"

# ============ 代码仓库 ============
repos:
  npu-runtime:
    url: "git@10.0.0.1:hw/npu-runtime.git"
    ref: "main"
    depth: 1
    build_script: "scripts/build_npu_runtime.sh"
    env:
      NPU_RUNTIME_DIR: "{install_dir}"

  hal-driver:
    url: "git@10.0.0.1:hw/hal-driver.git"
    ref: "v1.2.3"
    depth: 1
    # 无 build_script = 仅 clone，不编译（头文件库）
    env:
      HAL_INCLUDE: "{install_dir}/include"

  model-zoo:
    url: "git@10.0.0.1:ai/model-zoo.git"
    ref: "abc1234"
    depth: 1
    sparse_checkout:                        # 大仓库只拉取需要的子目录
      - "models/resnet/"
      - "models/bert/"
```

**验收标准：**
- 所有依赖在单一 `deps.yaml` 中声明，无需维护多个配置文件
- 工具链和库支持两种获取方式：local_dir / script
- 代码仓库通过 git clone 获取
- 每个工具链/库声明 sha256 校验值

### REQ-3.2 依赖获取（两种方式） [P0]

**描述：** 支持两种方式获取工具链和C/C++库，适配内网环境。

#### 方式1: 本地上传

用户手动将压缩包放到指定目录，框架自动识别并注册。

```
deps/uploads/
├── npu-compiler-2.1.0-linux-x86_64.tar.gz
├── simulator-3.4.1-linux-x86_64.tar.gz
└── json-c-0.17.tar.gz
```

**命名规范：** `<name>-<version>[-<platform>].tar.gz`

**流程：**
1. 用户将压缩包拷贝到 `deps/uploads/` 目录
2. 执行 `aitf deps install`
3. 框架扫描目录，根据文件名匹配配置
4. 校验 sha256
5. 解压到 `build/cache/<name>-<version>/`

#### 方式2: Bash 获取脚本

用户自行编写的 shell 脚本，从内网任意来源获取依赖（scp、HTTP、NFS 拷贝等均可）。

**脚本接口约定：**

```bash
#!/bin/bash
# scripts/fetch_npu_compiler.sh
# 参数: $1 = 版本号
#       $2 = 输出目录
# 退出码: 0=成功, 非0=失败
# 脚本需将压缩包下载到 $2/

VERSION=$1
OUTPUT_DIR=$2

# 示例: 从远程服务器 scp
scp "admin@10.0.0.1:/tools/npu-compiler-${VERSION}-linux-x86_64.tar.gz" "${OUTPUT_DIR}/"

# 示例: 从内网 HTTP 服务器下载
# wget -q "http://release.internal/npu-compiler/v${VERSION}/linux-x86_64.tar.gz" \
#      -O "${OUTPUT_DIR}/npu-compiler-${VERSION}-linux-x86_64.tar.gz"

# 示例: 从 NFS 共享盘拷贝
# cp "/mnt/shared/tools/npu-compiler-${VERSION}-linux-x86_64.tar.gz" "${OUTPUT_DIR}/"
```

**框架调用方式：**

```bash
bash scripts/fetch_npu_compiler.sh "2.1.0" "deps/uploads/"
```

**获取优先级：**

```
1. local_dir（检查上传目录是否已有匹配文件）
2. script（本地没有时执行获取脚本）
```

**验收标准：**
- 脚本接口统一：`$1=版本号, $2=输出目录`
- 脚本退出码非0时框架报错
- 脚本由用户自行编写维护，框架只负责调用和后续校验
- 已缓存的依赖不重复获取（通过版本号+sha256判断）
- 获取完成后自动校验 sha256，校验失败删除文件并报错

### REQ-3.3 代码仓库依赖 [P0]

**描述：** 依赖项为代码仓库时，通过 `git clone` 获取。支持 shallow clone、指定节点和 sparse checkout。

**框架执行的 git 命令：**

```bash
# depth=1 的 shallow clone + 指定分支
git clone --depth 1 --branch main git@10.0.0.1:hw/npu-runtime.git build/repos/npu-runtime

# 指定 commit hash（需要先 clone 再 checkout）
git clone --depth 1 git@10.0.0.1:ai/model-zoo.git build/repos/model-zoo
cd build/repos/model-zoo && git fetch --depth 1 origin abc1234 && git checkout abc1234

# sparse checkout（大仓库只拉取部分目录）
git clone --depth 1 --filter=blob:none --sparse git@10.0.0.1:ai/model-zoo.git build/repos/model-zoo
cd build/repos/model-zoo && git sparse-checkout set models/resnet/ models/bert/

# 完整 clone（depth 不设置时）
git clone git@10.0.0.1:hw/hal-driver.git build/repos/hal-driver
git -C build/repos/hal-driver checkout v1.2.3
```

**自定义构建脚本：**

clone 后需要编译的仓库通过 `build_script` 指定用户自维护的构建脚本：

```bash
#!/bin/bash
# scripts/build_npu_runtime.sh
# 参数: $1 = 仓库本地路径, $2 = 安装目录
REPO_DIR=$1
INSTALL_DIR=$2

cd "$REPO_DIR"
cmake -B build -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"
cmake --build build -j$(nproc)
cmake --install build
```

无 `build_script` 的仓库仅 clone，不编译（适用于头文件库、数据仓库等）。

**更新机制：**
- 仓库已存在时执行 `git fetch + git checkout <ref>` 更新到指定节点
- `depth: 1` 的仓库更新时使用 `git fetch --depth 1`

**验收标准：**
- 支持 SSH 协议（`git@host:repo.git`）
- ref 支持分支名、tag、commit hash 三种形式
- sparse_checkout 只拉取指定子目录，节省磁盘空间
- clone 失败时有清晰的错误信息（SSH key问题、网络不通等）
- 已存在的仓库增量更新，不重新 clone

### REQ-3.4 依赖锁文件 [P1]

**描述：** 自动生成 `deps.lock.yaml` 锁定精确版本，确保可重复构建。

**锁文件内容：**

```yaml
# deps.lock.yaml（自动生成，提交到 git）
generated_at: "2026-02-27T14:30:00"
platform: "linux-x86_64"

toolchains:
  npu-compiler:
    version: "2.1.0"
    sha256: "abc123..."
    installed_at: "2026-02-27T10:00:00"

  simulator:
    version: "3.4.1"
    sha256: "789abc..."
    installed_at: "2026-02-27T10:05:00"

libraries:
  json-c:
    version: "0.17"
    sha256: "abc..."
    installed_at: "2026-02-27T10:10:00"

repos:
  npu-runtime:
    ref: "main"
    commit: "a1b2c3d4e5f6..."       # 实际 checkout 的 commit hash
    cloned_at: "2026-02-27T10:15:00"

  hal-driver:
    ref: "v1.2.3"
    commit: "f6e5d4c3b2a1..."
    cloned_at: "2026-02-27T10:16:00"

  model-zoo:
    ref: "abc1234"
    commit: "abc1234..."
    sparse_checkout:
      - "models/resnet/"
      - "models/bert/"
    cloned_at: "2026-02-27T10:17:00"
```

**命令：**

```bash
# 生成/更新锁文件
aitf deps lock

# 按锁文件安装（CI 场景，确保精确复现）
aitf deps install --locked
```

**验收标准：**
- `aitf deps install` 后自动更新 `deps.lock.yaml`
- `--locked` 模式下严格按锁文件的 commit hash checkout 代码仓库
- 锁文件提交到 git，团队共享

### REQ-3.5 配置集（Bundle）管理 [P0]

**描述：** 将一组工具链+库+代码仓库打包为一个"配置集"（Bundle），确保版本组合经过验证，按套管理和切换。

**配置集在 `deps.yaml` 中定义：**

```yaml
# deps.yaml 中追加
bundles:
  npu-v2.1:
    description: "NPU验证环境 v2.1（2026Q1发布）"
    status: verified               # verified / testing / deprecated
    toolchains:
      npu-compiler: "2.1.0"
      simulator: "3.4.1"
    libraries:
      json-c: "0.17"
      unity: "2.6.0"
    repos:
      npu-runtime: "main"
      hal-driver: "v1.2.3"
    env:
      NPU_SDK_VERSION: "2.1"

  npu-v2.0:
    description: "NPU验证环境 v2.0（稳定版本）"
    status: verified
    toolchains:
      npu-compiler: "2.0.3"
      simulator: "3.3.0"
    libraries:
      json-c: "0.16"
      unity: "2.5.2"
    repos:
      npu-runtime: "v2.0.0"
      hal-driver: "v1.1.0"
    env:
      NPU_SDK_VERSION: "2.0"

  fpga-v1.0:
    description: "FPGA验证环境 v1.0"
    status: verified
    toolchains:
      fpga-tools: "1.0.0"
    libraries:
      unity: "2.6.0"
    repos:
      hal-driver: "v1.2.3"

# 当前激活的配置集
active: npu-v2.1
```

**命令：**

```bash
# 查看所有配置集
aitf bundle list

# 查看配置集详情（含安装状态）
aitf bundle show npu-v2.1

# 切换配置集（自动安装缺失依赖）
aitf bundle use npu-v2.0

# 安装指定配置集的全部依赖
aitf bundle install npu-v2.1

# 导出配置集为离线压缩包
aitf bundle export npu-v2.1 --output npu-v2.1-bundle.tar.gz

# 从离线包导入配置集
aitf bundle import npu-v2.1-bundle.tar.gz
```

**导出包结构：**

```
npu-v2.1-bundle.tar.gz
├── bundle.yaml                         # 配置集元信息
├── toolchains/
│   ├── npu-compiler-2.1.0-linux-x86_64.tar.gz
│   └── simulator-3.4.1-linux-x86_64.tar.gz
├── libraries/
│   ├── json-c-0.17.tar.gz
│   └── unity-2.6.0.tar.gz
└── repos/                              # 代码仓库的 tar.gz 快照
    ├── npu-runtime-main.tar.gz
    └── hal-driver-v1.2.3.tar.gz
```

**验收标准：**
- 配置集切换后，代码仓库自动 checkout 到指定 ref
- 切换前检查依赖是否齐全，不齐全时提示并可自动安装
- 导出的离线包可在另一台机器上导入完成部署
- 代码仓库导出为 tar.gz 快照（非 git bundle，更通用）
- 不允许切换到 `status: deprecated` 的配置集（除非 `--force`）

### REQ-3.6 依赖诊断 [P1]

**描述：** 提供 `aitf deps doctor` 命令，检查依赖环境的完整性和一致性。

**检查项：**

| 检查项 | 说明 |
|--------|------|
| 配置完整性 | deps.yaml 格式正确，字段无缺失 |
| 安装状态 | 所有声明的依赖是否已安装 |
| 版本一致 | 安装的版本与 deps.yaml 声明一致 |
| SHA256 校验 | 已安装的工具链/库校验值正确 |
| 仓库状态 | 代码仓库 HEAD 与声明的 ref 对应 |
| 锁文件同步 | deps.lock.yaml 与实际安装状态一致 |
| 脚本可用 | 获取脚本存在且有执行权限 |
| 构建工具 | cmake、git 等必要工具是否可用 |

**命令：**

```bash
aitf deps doctor
# 输出:
#   ✓ deps.yaml 配置完整
#   ✓ npu-compiler 2.1.0 已安装，SHA256匹配
#   ✓ simulator 3.4.1 已安装，SHA256匹配
#   ✗ json-c 0.17 未安装
#   ✓ npu-runtime HEAD=a1b2c3d ref=main 一致
#   ✗ deps.lock.yaml 与实际状态不一致，运行 aitf deps lock 更新
```

**验收标准：**
- 所有检查项有 pass/fail 状态
- 失败项给出修复建议
- 退出码：0=全部通过, 1=有问题

### REQ-3.7 Python 依赖管理 [P0]

**描述：** Python 依赖通过 pyproject.toml 管理。内网环境下通过内部 PyPI 镜像或离线 wheel 安装。

**pyproject.toml 示例：**

```toml
[project]
name = "aitestframework"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "pyyaml>=6.0",
    "paramiko>=3.0",
    "numpy>=1.24",
    "flask>=3.0",
    "sqlalchemy>=2.0",
    "jinja2>=3.1",
    "rich>=13.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.4",
    "mypy>=1.10",
    "pytest-cov>=5.0",
]

[project.scripts]
aitf = "aitestframework.cli:main"
```

**内网安装方式：**

```bash
# 方式1: 内部 PyPI 镜像
pip install -e . --index-url http://pypi.internal/simple/ --trusted-host pypi.internal

# 方式2: 离线 wheel 包目录
pip install -e . --no-index --find-links /path/to/wheels/
```

**验收标准：**
- `pip install -e .` 安装项目及全部依赖
- `pip install -e ".[dev]"` 安装开发依赖
- `aitf` 命令可用作统一 CLI 入口

### REQ-3.8 构建产物隔离 [P1]

**描述：** 不同平台的构建产物和依赖缓存分目录隔离。

**目录结构：**

```
deps/
├── uploads/                    # 本地上传目录
│   └── npu-compiler-2.1.0-linux-x86_64.tar.gz
└── scripts/                    # 获取脚本和构建脚本
    ├── fetch_npu_compiler.sh
    ├── fetch_simulator.sh
    └── build_npu_runtime.sh

build/
├── cache/                      # 工具链和库的解压缓存
│   ├── npu-compiler-2.1.0/
│   ├── simulator-3.4.1/
│   ├── json-c-0.17/
│   └── .downloads/             # 脚本下载的原始压缩包
├── repos/                      # Git 仓库依赖
│   ├── npu-runtime/
│   ├── hal-driver/
│   └── model-zoo/
├── npu/                        # NPU 平台构建产物
│   ├── bin/
│   └── lib/
├── gpu/                        # GPU 平台构建产物
├── cpu/                        # CPU 平台构建产物
└── reports/                    # 测试报告
```

**验收标准：**
- 各平台构建互不干扰
- `build/` 在 .gitignore 中
- `deps/uploads/` 中的大文件在 .gitignore 中

### REQ-3.9 构建缓存 [P2]

**描述：** 支持增量编译，未变更的源文件不重新编译。

**机制：** CMake 原生增量编译 + ccache 加速（可选）

**验收标准：**
- 修改单个文件后 rebuild 只编译受影响目标
- 配置 ccache 后编译速度有可测量的提升

## 3. CLI 命令汇总

```bash
# 依赖管理
aitf deps install                    # 安装全部依赖
aitf deps install npu-compiler       # 安装指定依赖
aitf deps install --locked           # 按锁文件精确安装
aitf deps list                       # 查看依赖状态
aitf deps lock                       # 生成/更新锁文件
aitf deps clean                      # 清理缓存
aitf deps doctor                     # 诊断检查

# 配置集管理
aitf bundle list                     # 查看所有配置集
aitf bundle show <name>              # 查看详情
aitf bundle use <name>               # 切换配置集
aitf bundle install <name>           # 安装配置集依赖
aitf bundle export <name> -o <file>  # 导出离线包
aitf bundle import <file>            # 导入离线包
```

## 4. 技术选型

| 决策 | 选型 | 备选 | 理由 |
|------|------|------|------|
| 构建系统 | CMake 3.16+ | Make, Meson | 跨平台交叉编译支持好 |
| 依赖获取 | 本地上传 + Bash脚本 | Conan, Docker | 适配内网，用户自维护脚本最灵活 |
| 代码仓库 | git clone（shallow + sparse） | git submodule, repo | submodule耦合太紧，自管理更灵活 |
| C/C++包管理 | 自管理（脚本获取+cmake/自定义构建） | Conan, vcpkg | 避免引入重型包管理器 |
| Python包格式 | pyproject.toml | setup.py | PEP 621 标准 |
| 配置集管理 | 自研 YAML + CLI | 无现成工具 | 需求特殊，自研最灵活 |

## 5. 数据模型

```python
@dataclass
class AcquireConfig:
    """获取方式配置"""
    local_dir: str | None = None    # 本地上传目录
    script: str | None = None       # 获取脚本路径

@dataclass
class ToolchainConfig:
    """工具链配置"""
    name: str
    version: str
    sha256: dict[str, str]          # platform -> sha256
    bin_dir: str | None = None
    env: dict[str, str] | None = None
    acquire: AcquireConfig | None = None

@dataclass
class LibraryConfig:
    """第三方库配置"""
    name: str
    version: str
    sha256: str
    build_system: str               # cmake / make / header_only
    cmake_args: list[str] | None = None
    build_script: str | None = None # 自定义构建脚本路径
    acquire: AcquireConfig | None = None

@dataclass
class RepoConfig:
    """代码仓库依赖配置"""
    name: str
    url: str                        # git仓库地址（SSH协议）
    ref: str                        # 分支名 / tag / commit hash
    depth: int | None = None        # shallow clone深度
    sparse_checkout: list[str] | None = None  # 只拉取的子目录
    build_script: str | None = None # 自定义构建脚本路径
    env: dict[str, str] | None = None

@dataclass
class BundleConfig:
    """配置集"""
    name: str
    description: str
    status: str                     # verified / testing / deprecated
    toolchains: dict[str, str]      # name -> version
    libraries: dict[str, str]       # name -> version
    repos: dict[str, str]           # name -> ref
    env: dict[str, str] | None = None

@dataclass
class BuildResult:
    """构建结果"""
    platform: str
    bundle: str                     # 使用的配置集名称
    success: bool
    artifacts: list[str]
    duration_s: float
    error_msg: str | None = None
```

## 6. 对外接口

```python
class DepsManager:
    def install(self, name: str | None = None, locked: bool = False) -> None: ...
    def list_installed(self) -> list[ToolchainConfig | LibraryConfig | RepoConfig]: ...
    def lock(self) -> None: ...
    def clean(self) -> None: ...
    def doctor(self) -> list[DiagResult]: ...
    def get_env(self) -> dict[str, str]: ...

class BundleManager:
    def list_bundles(self) -> list[BundleConfig]: ...
    def show(self, name: str) -> BundleConfig: ...
    def use(self, name: str, force: bool = False) -> None: ...
    def install(self, name: str) -> None: ...
    def export(self, name: str, output: str) -> str: ...
    def import_bundle(self, path: str) -> None: ...
    def active(self) -> BundleConfig: ...

class BuildManager:
    def configure(self, platform: str) -> None: ...
    def build(self, platform: str, target: str | None = None) -> BuildResult: ...
    def clean(self, platform: str | None = None) -> None: ...
```

## 7. 依赖

- REQ-1（桩代码管理）：被编译的源码
- REQ-6（执行框架）：构建产物供执行框架使用
