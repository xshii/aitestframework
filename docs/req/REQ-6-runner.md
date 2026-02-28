# REQ-6 自动化执行框架

---
version: 2.0
date: 2026-02-27
status: draft
---

## 1. 概述

执行框架是 `api.select_env()` 和 `api.execute()` 的底层实现。它管理本地和远程 SSH 执行环境，处理文件传输、命令执行、超时控制和结果收集。

用例通过框架 API 间接使用执行框架，不直接调用本模块。

**架构原则：** 所有执行逻辑实现在 Core API 层（`aitestframework.runner`），CLI 和 Web 均为薄壳调用层。

**架构关系：**

```
用例代码 (test_conv2d.py)
    │
    │ 调用 api.select_env() / api.execute()
    ▼
框架 API (aitestframework.api)
    │
    │ 委托给
    ▼
执行框架 (aitestframework.runner)
    ├── LocalExecutor     ← 本地 subprocess
    └── RemoteExecutor    ← paramiko SSH
```

## 2. 需求详情

### REQ-6.1 本地执行器 [P0]

**描述：** 在本机通过 subprocess 执行命令。

**实现：**

```python
class LocalExecutor:
    def run(self, command: str, timeout: int, env: dict,
            cwd: str | None = None) -> ExecuteResult:
        """执行本地命令
        - 通过 subprocess.run 执行
        - 捕获 stdout/stderr 并保存到日志文件
        - 超时杀进程
        """
```

**验收标准：**
- 命令通过 `subprocess.run(shell=True)` 执行
- stdout/stderr 实时写入日志文件同时捕获到内存
- 超时后 SIGTERM → 等待5s → SIGKILL

### REQ-6.2 远程执行器 [P0]

**描述：** 通过 SSH 连接远程服务器，传输文件并执行命令。

**实现：**

```python
class RemoteExecutor:
    def connect(self, config: TargetConfig) -> None:
        """建立SSH连接"""

    def upload(self, local_path: str, remote_path: str) -> None:
        """SFTP上传文件/目录"""

    def download(self, remote_path: str, local_path: str) -> None:
        """SFTP下载文件/目录"""

    def run(self, command: str, timeout: int, env: dict) -> ExecuteResult:
        """在远程执行命令，实时回传stdout/stderr"""

    def close(self) -> None:
        """关闭连接"""
```

**`api.execute()` 远程执行的完整流程：**

```
1. 检查 SSH 连接（断开则重连）
2. 执行 pre_commands（如 source /opt/env.sh）
3. 上传构建产物和输入数据到 remote_dir
4. 执行用例指定的命令
5. 下载输出文件到本地
6. 执行 post_commands（可选清理）
7. 返回 ExecuteResult
```

**验收标准：**
- 支持密码和密钥两种认证方式
- SSH 连接超时可配置（默认30s）
- 传输大文件时显示进度
- 连接断开时自动重试（最多3次）

### REQ-6.3 执行环境配置 [P0]

**描述：** 通过 YAML 配置文件定义可用的执行环境。

**配置文件：**

```yaml
# runner/targets.yaml
targets:
  local:
    type: local
    build_dir: build/
    env:
      LD_LIBRARY_PATH: "{build_dir}/lib"

  sim-server:
    type: remote
    host: 192.168.1.100
    port: 22
    user: test
    auth:
      method: key
      key_file: ~/.ssh/id_rsa
    remote_dir: /tmp/aitf_run
    env:
      SIM_PATH: /opt/simulator/bin
    pre_commands:
      - "source /opt/env.sh"
    post_commands:
      - "rm -rf /tmp/aitf_run/output"

  fpga-board:
    type: remote
    host: 10.0.0.50
    port: 22
    user: fpga
    auth:
      method: key
      key_file: ~/.ssh/fpga_key
    remote_dir: /home/fpga/test
    upload_extra:
      - "config/fpga_params.bin"
```

**`api.select_env("sim-server")` 的行为：**
1. 从 `targets.yaml` 读取 `sim-server` 配置
2. 创建 `RemoteExecutor` 并建立 SSH 连接
3. 返回 `Environment` 对象，封装该执行器

**验收标准：**
- 配置支持多个目标
- 敏感信息支持环境变量引用：`password: ${REMOTE_PASSWORD}`
- `select_env()` 时自动测试连接可达性，不可达时报错

### REQ-6.4 并行执行 [P1]

**描述：** 支持串行和并行两种执行模式。

**命令：**

```bash
# 串行执行（默认）
aitf run cases/npu/

# 并行执行（4个worker，每个worker独立SSH连接）
aitf run cases/npu/ --parallel 4

# 分片模式（多机并行时使用）
aitf run cases/npu/ --shard 1/4
```

**实现：** `concurrent.futures.ThreadPoolExecutor`

- 每个 worker 线程独立持有 `Environment` 对象
- 远程并行时每个 worker 独立 SSH 连接
- 每个用例的日志写到独立文件，不交叉

**验收标准：**
- 并行执行时日志互不干扰
- 分片模式下所有分片合并后覆盖全部用例
- `setUpClass` 中的构建操作有锁保护，不重复编译

### REQ-6.5 超时与异常恢复 [P1]

**描述：** 用例执行超时或崩溃后自动恢复，继续执行后续用例。

**异常处理：**

| 异常 | 检测方式 | 处理 |
|------|----------|------|
| 超时 | subprocess timeout / SSH channel timeout | 杀进程，标记 TIMEOUT |
| 崩溃 | returncode < 0（被信号杀死） | 标记 CRASH，记录信号 |
| 异常退出 | returncode > 0 | 由用例自行判断（可能是正常失败） |
| SSH断连 | paramiko.SSHException | 重试连接，3次失败后标记 ERROR |
| setUpClass失败 | 异常抛出 | 该套件全部用例标记 ERROR |

**验收标准：**
- 超时用例在 timeout + 5s 内一定终止
- 单个用例异常不影响后续用例执行
- 异常信息完整记录到日志和数据库

### REQ-6.6 结果收集 [P0]

**描述：** 自动收集执行产生的日志和输出文件。

**日志目录结构：**

```
build/reports/
├── 20260227-143000/                     # execution_id
│   ├── result.json                       # 汇总报告
│   ├── report.html                       # HTML报告
│   └── logs/
│       ├── TestConv2dNPU/
│       │   ├── test_fp32_3x3_basic/
│       │   │   ├── stdout.log
│       │   │   ├── stderr.log
│       │   │   └── output.bin
│       │   └── test_fp16_3x3_basic/
│       │       ├── stdout.log
│       │       └── stderr.log
│       └── TestMatmulNPU/
│           └── ...
```

**验收标准：**
- 远程执行的输出文件自动回传到本地
- 每次执行有独立的 execution_id 目录
- 日志文件全部保留，不自动清理
- `api.execute()` 返回的 `ExecuteResult.output_path` 指向本地文件

### REQ-6.7 CLI 与 Jenkins 集成 [P0]

**描述：** 提供命令行接口，支持 Jenkins 流水线调用。

**CLI 命令：**

```bash
# 运行用例
aitf run [path] [-k filter] [--parallel N] [--shard K/N]

# 指定执行环境（覆盖用例中的 select_env）
aitf run cases/npu/ --target sim-server

# 指定配置集（覆盖用例中的 use_bundle）
aitf run cases/npu/ --bundle npu-v2.1

# 输出格式
aitf run cases/npu/ --output build/reports/ci --format json,html

# 执行完成后 webhook 通知 Dashboard
aitf run cases/npu/ --webhook http://dashboard:5000/api/webhook

# 退出码
#   0 = 全部通过
#   1 = 有用例失败
#   2 = 框架级错误
```

**Jenkinsfile 示例：**

```groovy
// Jenkinsfile
pipeline {
    agent { label 'linux' }
    stages {
        stage('Setup') {
            steps {
                sh 'pip install -e .'
                sh 'aitf bundle install npu-v2.1'
                sh 'aitf deps install'
            }
        }
        stage('Test NPU Operators') {
            steps {
                sh '''
                    aitf run cases/npu/operators/ \
                        --target sim-server \
                        --bundle npu-v2.1 \
                        --output build/reports/npu-operators \
                        --format json,html \
                        --webhook http://dashboard:5000/api/webhook
                '''
            }
        }
        stage('Test NPU Models') {
            steps {
                sh '''
                    aitf run cases/npu/models/ \
                        --target sim-server \
                        --bundle npu-v2.1 \
                        --output build/reports/npu-models \
                        --format json,html \
                        --webhook http://dashboard:5000/api/webhook
                '''
            }
        }
    }
    post {
        always {
            archiveArtifacts 'build/reports/**'
        }
    }
}
```

**验收标准：**
- 退出码语义明确，Jenkins 可据此判断流水线状态
- `--target` 和 `--bundle` 可从命令行覆盖用例中的默认值
- webhook 失败不影响测试结果和退出码
- 项目提供 Jenkinsfile 模板

## 3. 技术选型

| 决策 | 选型 | 备选 | 理由 |
|------|------|------|------|
| SSH库 | paramiko | fabric, asyncssh | 成熟稳定，功能完整 |
| 并行框架 | ThreadPoolExecutor | multiprocessing, asyncio | 轻量，subprocess为主要瓶颈，线程足够 |
| 进程管理 | subprocess | os.system | 标准库，支持超时和信号 |
| 文件传输 | SFTP (paramiko) | rsync, scp | paramiko内置，无需额外工具 |
| CLI框架 | argparse | click, typer | 标准库，无额外依赖 |

## 4. 数据模型

```python
@dataclass
class TargetConfig:
    """执行目标配置"""
    name: str
    type: str                       # local / remote
    host: str | None = None
    port: int = 22
    user: str | None = None
    auth: dict | None = None
    remote_dir: str | None = None
    build_dir: str | None = None
    env: dict[str, str] | None = None
    pre_commands: list[str] | None = None
    post_commands: list[str] | None = None
    upload_extra: list[str] | None = None
```

## 5. 依赖

- REQ-1（桩代码）：`api.build()` 编译的源码
- REQ-2（Golden数据）：`api.load_golden()` 加载的数据
- REQ-3（构建管理）：`api.use_bundle()` 使用的配置集和工具链
- REQ-4（用例管理）：用例通过 API 调用本模块
- REQ-5（日志分析）：日志文件供日志解析器分析
- REQ-7（Dashboard）：webhook 推送执行结果
