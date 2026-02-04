# REQ-CIC CI/CD需求

---
id: REQ-CIC
title: CI/CD需求
priority: P1
status: draft
parent: REQ-SYS
context: Generic  # 通用域
subscribes:       # 订阅的领域事件
  - TestExecutionCompleted
  - ReportGenerated
  - QualityCheckCompleted
uses:             # 使用的接口（非直接依赖）
  - TestRunner CLI (FWK)
  - TestList format (TMT)
  - Report format (RST)
  - Quality rules (QCK)
  - Toolchain config (DEP)
---

## 概述

持续集成和持续交付流水线定义，包含两类CI/CD：

### CI/CD分类

| 类型 | 目的 | 触发 | 需求覆盖 |
|------|------|------|----------|
| **用例CI/CD** | 运行NPU验证用例 | 代码提交/定时/手动 | REQ-CIC-001 ~ 009 |
| **框架CI/CD** | 框架自身构建测试发布 | 框架代码变更/发版 | REQ-CIC-010 ~ 013 |

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CI/CD 流水线体系                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              用例CI/CD (验证被测对象)                         │    │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │    │
│  │  │LinuxUT  │→│Simulator│→│  ESL    │→│FPGA/Chip│           │    │
│  │  │每次提交 │ │ 每夜    │ │ 每周    │ │ 发版    │           │    │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘           │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              框架CI/CD (验证框架自身)                         │    │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │    │
│  │  │PR检查   │→│单元测试 │→│版本发布 │→│文档发布 │           │    │
│  │  │每次PR   │ │ 合入后  │ │ Tag触发 │ │ 发版后  │           │    │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘           │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### DDD定位

- **限界上下文**：通用域（Generic Domain）
- **角色**：编排者（Orchestrator），不包含核心业务逻辑
- **通信方式**：订阅领域事件 + 调用模块CLI接口
- **可替换性**：高（可换用Jenkins/GitLab原生流水线）

### 事件订阅

```yaml
订阅事件:
  TestExecutionCompleted:
    handler: 收集结果，更新状态
  ReportGenerated:
    handler: 发布报告到CI系统
  QualityCheckCompleted:
    handler: 根据结果决定是否阻塞
```

### 与其他模块的关系

```
CIC不直接依赖其他模块代码，而是：
1. 调用各模块提供的CLI命令
2. 订阅各模块发布的领域事件
3. 使用约定的数据格式（JSON/YAML）

┌─────────────────────────────────────────────┐
│                   CIC                        │
│  ┌─────────────────────────────────────┐    │
│  │         事件订阅器                    │    │
│  │  - on TestExecutionCompleted        │    │
│  │  - on ReportGenerated               │    │
│  └─────────────────────────────────────┘    │
│                    │                         │
│                    ▼                         │
│  ┌─────────────────────────────────────┐    │
│  │         CLI调用适配器                 │    │
│  │  - ./test_runner (FWK)              │    │
│  │  - python -m tools.report (RST/DVT) │    │
│  │  - python -m quality check (QCK)    │    │
│  │  - deps install (DEP)               │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

---

## REQ-CIC-001 流水线阶段

---
id: REQ-CIC-001
title: 验证流水线阶段
priority: P1
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
priority: P1
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
priority: P1
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

当远程服务器无法直接暴露端口时（如Docker容器、防火墙限制），可通过SSH本地端口转发访问服务。

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
priority: P1
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
priority: P2
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
priority: P2
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

---

## REQ-CIC-009 资源配额管理

---
id: REQ-CIC-009
title: 资源配额管理
priority: P2
status: draft
parent: REQ-CIC
---

### 描述

管理CI执行时的资源限制，防止资源耗尽。

### 资源限制

```yaml
# cicd/resource_limits.yaml
limits:
  # 单作业限制
  job:
    cpu_cores: 8           # 最大CPU核数
    memory_gb: 32          # 最大内存
    disk_gb: 100           # 最大磁盘使用
    timeout_minutes: 120   # 最大执行时间

  # 并发限制
  concurrency:
    max_parallel_jobs: 4   # 最大并行作业数
    max_parallel_tests: 16 # 最大并行测试数

  # 网络限制
  network:
    download_timeout_s: 300
    max_retry: 3
```

### 监控指标

| 指标 | 告警阈值 | 说明 |
|------|----------|------|
| CPU使用率 | > 90% 持续5分钟 | 可能需要扩容 |
| 内存使用率 | > 85% | 有OOM风险 |
| 磁盘使用率 | > 80% | 需要清理 |
| 作业队列长度 | > 10 | 资源不足 |

### 超限处理

```yaml
# 超限策略
on_limit_exceeded:
  cpu: throttle          # 限流
  memory: kill_job       # 终止作业
  disk: fail_job         # 作业失败
  timeout: fail_job      # 作业失败
```

### 验收标准

1. 支持配置各类资源限制
2. 超限时有明确错误信息
3. 提供资源使用统计报告
4. 支持告警通知

---

# 框架CI/CD需求

以下需求定义框架自身的持续集成和持续交付流程。

---

## REQ-CIC-010 框架PR检查

---
id: REQ-CIC-010
title: 框架代码PR检查
priority: P1
status: draft
parent: REQ-CIC
---

### 描述

框架代码提交PR时的自动化检查流程。

### 检查项

```yaml
# cicd/framework/pr-check.yaml
pr_check:
  stages:
    - name: lint
      parallel: true
      jobs:
        - c_lint:
            run: |
              clang-format --dry-run --Werror src/**/*.c include/**/*.h
              cppcheck --error-exitcode=1 src/
        - python_lint:
            run: |
              black --check tools/ pymodel/
              flake8 tools/ pymodel/
              pylint tools/ pymodel/ --fail-under=9.0

    - name: build
      jobs:
        - build_linux_ut:
            run: make PLATFORM=linux_ut BUILD=debug
        - build_linux_st:
            run: make PLATFORM=linux_st BUILD=debug

    - name: test
      jobs:
        - framework_unit_test:
            run: |
              ./build/bin/linux_ut/test_runner --filter "unit.framework.*"
              pytest tests/unit/tools/ -v --cov=tools --cov-fail-under=60
        - model_unit_test:
            run: pytest tests/unit/pymodel/ -v --cov=pymodel --cov-fail-under=80

    - name: quality
      jobs:
        - code_quality:
            run: python -m quality check --diff origin/main
```

### 检查矩阵

| 检查项 | 工具 | 阻塞合入 | 超时 |
|--------|------|----------|------|
| C代码格式 | clang-format | 是 | 1min |
| C静态分析 | cppcheck | 是 | 5min |
| Python格式 | black, flake8 | 是 | 1min |
| Python质量 | pylint >= 9.0 | 是 | 2min |
| C编译 | gcc -Wall -Werror | 是 | 5min |
| 框架单测 | test_runner | 是 | 5min |
| Python单测 | pytest | 是 | 5min |
| 代码规范 | quality check | 是 | 3min |

### 验收标准

1. PR检查全部通过才能合入
2. 检查结果在PR页面展示
3. 失败时提供详细错误信息
4. 支持跳过检查（需审批）

---

## REQ-CIC-011 框架版本发布

---
id: REQ-CIC-011
title: 框架版本发布流程
priority: P1
status: draft
parent: REQ-CIC
---

### 描述

框架版本发布的自动化流程。

### 发布流程

```
1. 创建Release分支 → 2. 更新版本号 → 3. 生成Changelog → 4. 创建Tag → 5. 构建发布包 → 6. 发布
```

### 版本号规范

```yaml
# 语义化版本 (SemVer)
version_format: "MAJOR.MINOR.PATCH"

# 版本文件
version_files:
  - include/framework/version.h:
      pattern: '#define FWK_VERSION_STRING "(\d+\.\d+\.\d+)"'
  - pyproject.toml:
      pattern: 'version = "(\d+\.\d+\.\d+)"'
  - setup.py:
      pattern: "version='(\d+\.\d+\.\d+)'"
```

### 发布配置

```yaml
# cicd/framework/release.yaml
release:
  trigger:
    - tag: "v*"

  stages:
    - name: validate
      jobs:
        - check_version:
            run: |
              # 检查版本号一致性
              python scripts/check_version.py --tag $TAG
        - check_changelog:
            run: |
              # 检查CHANGELOG.md已更新
              grep -q "## \[$TAG\]" CHANGELOG.md

    - name: build
      jobs:
        - build_all_platforms:
            matrix:
              platform: [linux_ut, linux_st, simulator]
            run: make PLATFORM=${{ matrix.platform }} BUILD=release

    - name: test
      jobs:
        - full_test:
            run: ./test_runner --list full

    - name: package
      jobs:
        - create_tarball:
            run: |
              tar -czvf npu-test-framework-$TAG.tar.gz \
                build/bin/ include/ configs/ docs/
        - create_wheel:
            run: |
              cd tools && python -m build

    - name: publish
      jobs:
        - github_release:
            run: |
              gh release create $TAG \
                --title "Release $TAG" \
                --notes-file CHANGELOG.md \
                npu-test-framework-$TAG.tar.gz
```

### Changelog生成

```bash
# 自动生成Changelog（基于commit message）
python scripts/generate_changelog.py --from v1.0.0 --to v1.1.0

# Commit message规范
# feat: 新功能
# fix: 修复bug
# docs: 文档更新
# refactor: 重构
# test: 测试相关
# chore: 构建/工具变更
```

### 验收标准

1. Tag触发自动发布流程
2. 版本号在所有文件中一致
3. 自动生成Changelog
4. 发布包包含二进制和文档
5. 发布到GitHub Releases

---

## REQ-CIC-012 文档自动生成

---
id: REQ-CIC-012
title: 文档自动生成发布
priority: P2
status: draft
parent: REQ-CIC
---

### 描述

框架文档的自动生成和发布。

### 文档类型

| 类型 | 工具 | 来源 | 输出 |
|------|------|------|------|
| API文档(C) | Doxygen | include/*.h | docs/api/c/ |
| API文档(Python) | Sphinx | tools/, pymodel/ | docs/api/python/ |
| 用户指南 | MkDocs | docs/*.md | docs/guide/ |
| 需求文档 | 自定义 | docs/req/*.md | docs/requirements/ |

### 生成配置

```yaml
# cicd/framework/docs.yaml
docs:
  trigger:
    - push:
        branches: [main]
        paths: ["docs/**", "include/**", "tools/**", "pymodel/**"]
    - tag: "v*"

  stages:
    - name: generate
      jobs:
        - doxygen:
            run: doxygen Doxyfile
        - sphinx:
            run: |
              cd docs && sphinx-build -b html source build/python
        - mkdocs:
            run: mkdocs build

    - name: combine
      jobs:
        - merge_docs:
            run: |
              mkdir -p public
              cp -r docs/api/c public/
              cp -r docs/api/python public/
              cp -r docs/guide public/

    - name: publish
      jobs:
        - deploy_pages:
            run: |
              # 发布到GitHub Pages / GitLab Pages
              gh-pages -d public
```

### Doxygen配置

```
# Doxyfile (关键配置)
PROJECT_NAME           = "NPU Test Framework"
INPUT                  = include/ src/
FILE_PATTERNS          = *.h *.c
EXTRACT_ALL            = YES
GENERATE_HTML          = YES
GENERATE_LATEX         = NO
```

### 验收标准

1. main分支更新自动重新生成
2. 版本发布时生成对应版本文档
3. 文档发布到GitHub/GitLab Pages
4. API文档覆盖所有公开接口

---

## REQ-CIC-013 Python包发布

---
id: REQ-CIC-013
title: Python工具包发布
priority: P2
status: draft
parent: REQ-CIC
---

### 描述

Python辅助工具包的打包和发布。

### 包结构

```
tools/                      # 主包
├── pyproject.toml          # 包配置
├── setup.py                # 兼容旧版pip
├── npu_test_tools/         # 包源码
│   ├── __init__.py
│   ├── runner/             # 测试运行器
│   ├── report/             # 报告生成
│   ├── data/               # 数据工具
│   └── quality/            # 质量检查
└── tests/                  # 包测试

pymodel/                    # 参考模型包
├── pyproject.toml
├── npu_pymodel/
│   ├── __init__.py
│   ├── ops/                # 算子实现
│   └── quantize/           # 量化工具
└── tests/
```

### 发布配置

```yaml
# cicd/framework/pypi.yaml
pypi:
  trigger:
    - tag: "v*"

  packages:
    - name: npu-test-tools
      path: tools/
      registry: internal  # 内部PyPI

    - name: npu-pymodel
      path: pymodel/
      registry: internal

  stages:
    - name: build
      jobs:
        - build_wheel:
            run: |
              cd ${{ package.path }}
              python -m build

    - name: test
      jobs:
        - test_install:
            run: |
              pip install dist/*.whl
              python -c "import ${{ package.name.replace('-', '_') }}"

    - name: publish
      jobs:
        - upload:
            run: |
              twine upload --repository ${{ package.registry }} dist/*
```

### pyproject.toml示例

```toml
# tools/pyproject.toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "npu-test-tools"
version = "1.0.0"
description = "NPU Test Framework Python Tools"
requires-python = ">=3.8"
dependencies = [
    "pyyaml>=6.0",
    "jinja2>=3.0",
    "click>=8.0",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-cov", "black", "flake8", "pylint"]

[project.scripts]
npu-report = "npu_test_tools.report.cli:main"
npu-data = "npu_test_tools.data.cli:main"
```

### 验收标准

1. 版本发布时自动打包上传
2. 支持内部PyPI仓库
3. 包可通过pip install安装
4. 提供命令行入口点