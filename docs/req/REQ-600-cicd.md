# REQ-CIC CI/CD需求

---
id: REQ-CIC
title: CI/CD需求
priority: P1
status: draft
parent: REQ-SYS
depends:
  - REQ-FWK
  - REQ-DEP
  - REQ-TMT
  - REQ-RST
---

## 概述

持续集成和持续交付流水线定义。

---

## REQ-CIC-001 流水线阶段

---
id: REQ-CIC-001
title: 验证流水线阶段
priority: P0
status: draft
parent: REQ-CIC
---

### 描述

定义验证流水线的各阶段。

### 阶段定义

```
┌─────────────────────────────────────────────────────────────────────┐
│                         验证流水线                                   │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────────┤
│   Stage 1   │   Stage 2   │   Stage 3   │   Stage 4   │   Stage 5   │
│  Commit     │  Nightly    │  Weekly     │  Release    │  Post-Si    │
├─────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│  LinuxUT/ST │  Simulator  │    ESL      │    FPGA     │    Chip     │
│  每次提交   │   每夜      │   每周      │   里程碑    │   流片后    │
│  ~5分钟     │   ~2小时    │   ~8小时    │   ~24小时   │   按需      │
└─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘
```

### 阶段详情

| 阶段 | 触发条件 | 平台 | 用例集 | 超时 | 门控 |
|------|----------|------|--------|------|------|
| Commit | 每次push | Host | sanity | 10min | 阻塞合入 |
| Nightly | 每天0点 | Simulator | nightly | 4h | 邮件通知 |
| Weekly | 每周日 | ESL | full+perf | 12h | 邮件通知 |
| Release | 手动/tag | FPGA | full | 24h | 阻塞发版 |
| Post-Si | 手动 | Chip | full+stress | - | 报告归档 |

### 验收标准

1. 各阶段独立可配置
2. 前置阶段失败可阻塞后续
3. 支持跳过指定阶段
4. 阶段间传递artifact

---

## REQ-CIC-002 触发策略

---
id: REQ-CIC-002
title: 流水线触发策略
priority: P0
status: draft
parent: REQ-CIC
---

### 描述

定义流水线触发条件。

### 触发类型

```yaml
# cicd/triggers.yaml
triggers:
  # 代码推送
  push:
    branches: [main, develop, "feature/*"]
    paths_include: ["src/**", "tests/**", "include/**"]
    paths_exclude: ["docs/**", "*.md"]

  # Pull Request
  pull_request:
    branches: [main]
    stages: [commit]

  # 定时触发
  schedule:
    - cron: "0 0 * * *"      # 每天0点
      stages: [nightly]
    - cron: "0 0 * * 0"      # 每周日0点
      stages: [weekly]

  # 手动触发
  manual:
    stages: [release, post_si]

  # Tag触发
  tag:
    pattern: "v*"
    stages: [release]
```

### 验收标准

1. 支持分支/路径过滤
2. 支持定时触发
3. 支持手动触发带参数
4. 支持Tag触发

---

## REQ-CIC-003 作业定义

---
id: REQ-CIC-003
title: CI作业定义
priority: P0
status: draft
parent: REQ-CIC
---

### 描述

定义可复用的CI作业。

### 作业类型

```yaml
# cicd/jobs/build.yaml
name: build
description: "编译测试程序"
inputs:
  platform: string
  build_type: [debug, release]

steps:
  - name: Setup toolchain
    run: python deps/scripts/dep_manager.py install

  - name: Build
    run: make PLATFORM=${{ inputs.platform }} BUILD=${{ inputs.build_type }}

  - name: Upload artifacts
    artifacts:
      - build/bin/${{ inputs.platform }}/*
```

```yaml
# cicd/jobs/test.yaml
name: test
description: "运行测试"
inputs:
  platform: string
  testlist: string
  timeout_minutes: integer

steps:
  - name: Download build
    download: build-${{ inputs.platform }}

  - name: Run tests
    run: |
      ./build/bin/${{ inputs.platform }}/test_runner \
        --list ${{ inputs.testlist }} \
        --output json \
        --output-file results.json
    timeout: ${{ inputs.timeout_minutes }}m

  - name: Upload results
    artifacts:
      - results.json
      - logs/
```

### 验收标准

1. 作业可参数化
2. 作业可复用
3. 支持artifact传递
4. 支持超时控制

---

## REQ-CIC-004 并行执行

---
id: REQ-CIC-004
title: 并行测试执行
priority: P1
status: draft
parent: REQ-CIC
---

### 描述

并行执行测试以加速。

### 并行策略

```yaml
# cicd/pipelines/nightly.yaml
stages:
  - name: build
    jobs:
      - build-simulator
      - build-esl
    parallel: true

  - name: test
    jobs:
      - job: test-simulator
        matrix:
          shard: [0, 1, 2, 3]  # 4分片并行
        run: |
          ./test_runner --shard-index ${{ matrix.shard }} --shard-count 4

  - name: report
    needs: [test]
    jobs:
      - merge-results
      - generate-report
```

### 验收标准

1. 支持作业级并行
2. 支持矩阵构建
3. 支持测试分片
4. 合并分片结果

---

## REQ-CIC-005 环境管理

---
id: REQ-CIC-005
title: 执行环境管理
priority: P1
status: draft
parent: REQ-CIC
---

### 描述

管理CI执行环境。

### 环境配置

```yaml
# cicd/environments.yaml
environments:
  # 固定远程服务器（禁止使用Docker/虚拟机）
  build-server:
    type: remote
    host: "dev@192.168.1.100"
    ssh_key: "${CI_SSH_KEY}"
    work_dir: "/home/dev/ci-workspace"
    resources:
      cpu: 8
      memory: 32G

  simulator-server:
    type: remote
    host: "sim@192.168.1.101"
    ssh_key: "${CI_SSH_KEY}"
    work_dir: "/opt/npu-sim/workspace"
    labels: [simulator, license]
    resources:
      cpu: 16
      memory: 64G

  fpga-server:
    type: remote
    host: "fpga@192.168.1.102"
    ssh_key: "${CI_SSH_KEY}"
    work_dir: "/opt/fpga/workspace"
    labels: [fpga, hardware]
    resources:
      fpga_board: "xilinx_u250"
```

### 远程执行流程

```
本地/CI触发
    │
    ▼
SSH连接远程服务器（含端口转发）
    │
    ▼
rsync同步代码到work_dir
    │
    ▼
远程执行编译/测试
    │
    ▼
rsync回传结果文件
    │
    ▼
本地解析生成报告
```

### SSH端口转发

由于远端服务器为Docker容器，不支持新增端口暴露，需通过SSH本地端口转发访问服务。

```bash
# 端口转发配置
# ~/.ssh/config
Host dev-server
    HostName 192.168.1.100
    User dev
    # 报告服务器端口转发
    LocalForward 8080 localhost:8080
    # 调试服务器端口转发
    LocalForward 5678 localhost:5678
    # GDB Server端口转发
    LocalForward 1234 localhost:1234

# 或通过命令行
ssh -L 8080:localhost:8080 -L 5678:localhost:5678 dev@192.168.1.100
```

**端口用途**:
| 本地端口 | 远程端口 | 用途 |
|----------|----------|------|
| 8080 | 8080 | 测试报告Web服务 |
| 5678 | 5678 | Python调试器 |
| 1234 | 1234 | GDB远程调试 |

### 验收标准

1. 通过SSH连接远程服务器执行
2. 支持多台服务器标签调度
3. 工作目录自动清理和复用
4. 支持断点续传和失败重连

---

## REQ-CIC-006 CI平台集成

---
id: REQ-CIC-006
title: CI平台集成
priority: P0
status: draft
parent: REQ-CIC
---

### 描述

支持主流CI平台。

### 支持平台

**Jenkins (远程服务器执行)**:
```groovy
// cicd/jenkins/Jenkinsfile
pipeline {
    agent { label 'ci-controller' }
    environment {
        REMOTE_HOST = 'dev@192.168.1.100'
        REMOTE_DIR = '/home/dev/ci-workspace/${BUILD_NUMBER}'
    }
    stages {
        stage('Sync') {
            steps {
                sh 'rsync -avz --delete ./ ${REMOTE_HOST}:${REMOTE_DIR}/'
            }
        }
        stage('Build') {
            steps {
                sh 'ssh ${REMOTE_HOST} "cd ${REMOTE_DIR} && make PLATFORM=simulator"'
            }
        }
        stage('Test') {
            steps {
                sh 'ssh ${REMOTE_HOST} "cd ${REMOTE_DIR} && ./test_runner --list nightly"'
            }
        }
        stage('Collect') {
            steps {
                sh 'rsync -avz ${REMOTE_HOST}:${REMOTE_DIR}/results/ ./results/'
            }
        }
    }
    post {
        always { junit 'results/*.xml' }
        cleanup {
            sh 'ssh ${REMOTE_HOST} "rm -rf ${REMOTE_DIR}"'
        }
    }
}
```

**GitLab CI (远程服务器执行)**:
```yaml
# cicd/gitlab/.gitlab-ci.yml
stages: [sync, build, test, collect, cleanup]

variables:
  REMOTE_HOST: "dev@192.168.1.100"
  REMOTE_DIR: "/home/dev/ci-workspace/${CI_PIPELINE_ID}"

sync:
  stage: sync
  script:
    - rsync -avz --delete ./ ${REMOTE_HOST}:${REMOTE_DIR}/

build:
  stage: build
  script:
    - ssh ${REMOTE_HOST} "cd ${REMOTE_DIR} && make PLATFORM=simulator"

test:
  stage: test
  script:
    - ssh ${REMOTE_HOST} "cd ${REMOTE_DIR} && ./test_runner --list nightly --output junit"

collect:
  stage: collect
  script:
    - rsync -avz ${REMOTE_HOST}:${REMOTE_DIR}/results/ ./results/
  artifacts:
    reports:
      junit: results/results.xml

cleanup:
  stage: cleanup
  when: always
  script:
    - ssh ${REMOTE_HOST} "rm -rf ${REMOTE_DIR}"
```

### 验收标准

1. 提供Jenkins/GitLab/GitHub配置模板
2. 结果集成CI报告系统
3. 支持CI特定功能（缓存、artifact等）

---

## REQ-CIC-007 失败处理

---
id: REQ-CIC-007
title: 失败处理策略
priority: P1
status: draft
parent: REQ-CIC
---

### 描述

处理测试失败和异常情况。

### 策略定义

```yaml
# cicd/failure_handling.yaml
failure_handling:
  # 失败重试
  retry:
    enabled: true
    max_attempts: 2
    retry_on: [timeout, flaky]

  # 失败分类
  classification:
    - pattern: "timeout after *"
      category: timeout
      action: retry

    - pattern: "SIGSEGV"
      category: crash
      action: collect_dump

    - pattern: "cosine_sim * < *"
      category: precision
      action: log_and_continue

  # Flaky处理
  flaky:
    detection: auto  # 自动检测flaky
    action: quarantine  # 隔离flaky用例
    notify: true
```

### 验收标准

1. 支持失败重试
2. 自动分类失败原因
3. 自动检测和隔离flaky
4. 收集crash dump

---

## REQ-CIC-008 报告集成

---
id: REQ-CIC-008
title: CI报告集成
priority: P1
status: draft
parent: REQ-CIC
---

### 描述

测试报告与CI系统集成。

### 集成功能

1. **状态徽章**: `![Tests](https://ci.example.com/badge/tests)`
2. **PR评论**: 自动评论测试结果摘要
3. **趋势图**: 展示历史趋势
4. **失败链接**: 直接跳转到失败详情

### PR评论模板

```markdown
## Test Results

| Platform | Passed | Failed | Skipped | Time |
|----------|--------|--------|---------|------|
| simulator | 145 | 3 | 2 | 5m23s |

### Failed Tests
- `functional.matmul.large` - cosine_sim 0.985 < 0.999
- `functional.conv.stride2` - timeout

[Full Report](https://ci.example.com/report/12345)
```

### 验收标准

1. 生成状态徽章
2. PR自动评论
3. 支持结果趋势展示
4. 失败可追溯到具体用例