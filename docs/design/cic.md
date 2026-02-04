# CIC CI/CD模块设计

---
module: CIC
version: 1.0
date: 2026-02-04
status: draft
requirements: REQ-CIC-001~013
domain: Generic
priority: P1
---

## 1. 模块概述

### 1.1 职责

CI/CD流水线定义与执行：
- 用例CI/CD（运行NPU验证用例）
- 框架CI/CD（框架自身构建测试发布）

### 1.2 CI/CD分类

```
┌─────────────────────────────────────────────────────────────────┐
│                        CI/CD 分类                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  用例CI/CD (REQ-CIC-001~009)                                    │
│  ├─ 目的: 运行NPU验证用例                                        │
│  ├─ 触发: 驱动/算子代码变更                                      │
│  ├─ 流水线: LinuxUT → Simulator → ESL → FPGA                   │
│  └─ 产出: 测试报告、回归结果                                     │
│                                                                  │
│  框架CI/CD (REQ-CIC-010~013)                                    │
│  ├─ 目的: 框架自身的质量保证                                     │
│  ├─ 触发: 框架代码PR、Tag发布                                    │
│  ├─ 流水线: Lint → Build → Test → Package                       │
│  └─ 产出: 框架发布包、文档、Python wheel                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 DDD定位

- **限界上下文**：通用域（可替换）
- **订阅事件**：ReportGenerated

---

## 2. 文件结构

```
cicd/
├── pipelines/
│   ├── sanity.yaml         # 冒烟测试流水线
│   ├── nightly.yaml        # 每日构建流水线
│   ├── release.yaml        # 发布流水线
│   └── framework.yaml      # 框架CI流水线
├── jobs/
│   ├── build.yaml          # 构建作业
│   ├── test.yaml           # 测试作业
│   └── report.yaml         # 报告作业
├── github/
│   └── workflows/
│       ├── pr-check.yml    # PR检查
│       ├── nightly.yml     # 每日构建
│       └── release.yml     # 发布
├── gitlab/
│   └── .gitlab-ci.yml
└── jenkins/
    └── Jenkinsfile
```

---

## 3. 用例CI/CD流水线

### 3.1 流水线阶段 (REQ-CIC-001)

```
┌─────────────────────────────────────────────────────────────────┐
│                       验证流水线                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Stage 1: LinuxUT          Stage 2: Simulator                   │
│  ┌─────────────────┐       ┌─────────────────┐                  │
│  │ • 框架单元测试   │ ──►   │ • 功能验证      │                  │
│  │ • HAL Mock测试   │       │ • 算子正确性    │                  │
│  │ • 快速反馈       │       │ • 位精确        │                  │
│  └─────────────────┘       └─────────────────┘                  │
│          │                          │                            │
│          ▼                          ▼                            │
│  Stage 3: ESL              Stage 4: FPGA/Chip                   │
│  ┌─────────────────┐       ┌─────────────────┐                  │
│  │ • 性能建模       │ ──►   │ • 硬件验证      │                  │
│  │ • 时序分析       │       │ • 实际性能      │                  │
│  │ • 带宽评估       │       │ • 功耗测量      │                  │
│  └─────────────────┘       └─────────────────┘                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 触发策略 (REQ-CIC-002)

| 触发方式 | 流水线 | 平台 |
|----------|--------|------|
| PR提交 | sanity | LinuxUT |
| 合并main | nightly | LinuxUT + Simulator |
| 定时(每日) | nightly | LinuxUT + Simulator + ESL |
| 手动/Tag | full | 所有平台 |

### 3.3 流水线定义

```yaml
# cicd/pipelines/nightly.yaml
name: nightly-pipeline
description: 每日构建流水线

trigger:
  schedule:
    cron: "0 2 * * *"  # 每天凌晨2点
  branches:
    - main

stages:
  - name: build
    jobs:
      - build-linux-ut
      - build-simulator

  - name: test-linux-ut
    depends_on: build
    jobs:
      - test-unit
      - test-functional-sanity

  - name: test-simulator
    depends_on: build
    parallel: true
    jobs:
      - test-functional-matmul
      - test-functional-conv
      - test-functional-activation

  - name: report
    depends_on:
      - test-linux-ut
      - test-simulator
    jobs:
      - generate-report
      - notify

artifacts:
  - build/bin/*
  - build/reports/*
```

### 3.4 作业定义 (REQ-CIC-003)

```yaml
# cicd/jobs/test.yaml
jobs:
  test-unit:
    runner: linux
    steps:
      - name: Checkout
        uses: checkout

      - name: Build
        run: make PLATFORM=linux_ut

      - name: Run Tests
        run: |
          ./build/bin/test_runner --filter "unit.*" --json --output unit-results.json

      - name: Upload Results
        uses: upload-artifact
        with:
          name: unit-results
          path: unit-results.json

  test-functional-sanity:
    runner: linux
    steps:
      - name: Run Sanity Tests
        run: |
          ./build/bin/test_runner --filter "sanity.*" --json --output sanity-results.json
```

---

## 4. 框架CI/CD流水线

### 4.1 PR检查 (REQ-CIC-010)

```yaml
# cicd/github/workflows/pr-check.yml
name: PR Check

on:
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Code Quality Check
        run: python -m tools.quality check --format json

      - name: Python Lint
        run: |
          pip install flake8 mypy
          flake8 tools/
          mypy tools/

  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build LinuxUT
        run: make PLATFORM=linux_ut

      - name: Build Check
        run: |
          test -f build/bin/test_runner

  test:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build
        run: make PLATFORM=linux_ut

      - name: Run Framework Tests
        run: |
          ./build/bin/test_runner --filter "unit.framework.*"

      - name: Run Python Tests
        run: |
          pip install pytest
          pytest tools/ -v
```

### 4.2 版本发布 (REQ-CIC-011)

```yaml
# cicd/github/workflows/release.yml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        platform: [linux_ut, simulator]

    steps:
      - uses: actions/checkout@v4

      - name: Build
        run: make PLATFORM=${{ matrix.platform }}

      - name: Package
        run: |
          mkdir -p dist
          tar -czf dist/aitestframework-${{ matrix.platform }}-${{ github.ref_name }}.tar.gz \
            build/bin/ include/ docs/

      - name: Upload Artifact
        uses: actions/upload-artifact@v4
        with:
          name: package-${{ matrix.platform }}
          path: dist/*.tar.gz

  release:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Download Artifacts
        uses: actions/download-artifact@v4

      - name: Create Release
        uses: softprops/action-gh-release@v1
        with:
          files: |
            package-*/*.tar.gz
          body: |
            ## Changes
            See CHANGELOG.md for details.

  docs:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build Docs
        run: |
          pip install mkdocs mkdocs-material
          mkdocs build

      - name: Deploy to Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./site
```

### 4.3 文档生成 (REQ-CIC-012)

```yaml
# 文档工具配置
# mkdocs.yml
site_name: AI Test Framework
theme:
  name: material

nav:
  - Home: index.md
  - Requirements:
    - Overview: req/README.md
    - Framework: req/REQ-100-framework.md
    - Platform: req/REQ-700-platform.md
  - Design:
    - Overview: design/README.md
    - Modules: design/
  - API Reference: api/

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          paths: [tools]
```

### 4.4 Python包发布 (REQ-CIC-013)

```yaml
# cicd/github/workflows/pypi.yml
name: Publish Python Package

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Build Package
        run: |
          pip install build
          python -m build

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
```

---

## 5. 并行执行 (REQ-CIC-004)

### 5.1 测试分片

```yaml
# 并行执行配置
parallel:
  strategy: shard
  shards: 4

# 分片算法
# 按测试时间均衡分配
jobs:
  test-shard-1:
    tests: ["matmul.*", "conv.basic"]
  test-shard-2:
    tests: ["conv.advanced", "activation.*"]
  test-shard-3:
    tests: ["precision.*", "e2e.layer1"]
  test-shard-4:
    tests: ["e2e.layer2", "stress.*"]
```

### 5.2 结果合并

```python
# tools/cicd/merge_results.py
def merge_shard_results(shard_files: list[str], output: str):
    """合并分片测试结果"""
    merged = {
        "execution_id": generate_id(),
        "summary": {"total": 0, "passed": 0, "failed": 0},
        "tests": []
    }

    for shard_file in shard_files:
        with open(shard_file) as f:
            shard = json.load(f)
            merged["tests"].extend(shard["tests"])
            for key in ["total", "passed", "failed", "skipped"]:
                merged["summary"][key] += shard["summary"][key]

    with open(output, "w") as f:
        json.dump(merged, f, indent=2)
```

---

## 6. 执行环境管理 (REQ-CIC-005)

### 6.1 Runner配置

```yaml
# 执行器配置
runners:
  linux-ut:
    type: docker
    image: aitf-linux:latest
    resources:
      cpu: 2
      memory: 4Gi

  simulator:
    type: shell
    requirements:
      - simulator >= 2.0.0
      - compiler >= 2.1.0
    setup:
      - source /opt/toolchain/env.sh

  fpga:
    type: remote
    host: fpga-server.example.com
    requirements:
      - fpga-tools >= 1.0.0
```

### 6.2 Docker镜像

```dockerfile
# cicd/docker/Dockerfile.linux-ut
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    gcc \
    make \
    python3 \
    python3-pip

COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt

WORKDIR /workspace
```

---

## 7. 失败处理 (REQ-CIC-007)

### 7.1 重试策略

```yaml
retry:
  max_attempts: 3
  backoff: exponential
  on:
    - network_error
    - timeout
    - flaky_test
```

### 7.2 失败通知

```yaml
notifications:
  on_failure:
    - type: email
      recipients: [team@example.com]
    - type: slack
      channel: "#ci-alerts"
      webhook: ${{ secrets.SLACK_WEBHOOK }}

  on_recovery:
    - type: slack
      channel: "#ci-alerts"
      message: "Build recovered: ${{ pipeline.name }}"
```

---

## 8. CI平台集成 (REQ-CIC-006)

### 8.1 GitHub Actions

见上述示例。

### 8.2 GitLab CI

```yaml
# cicd/gitlab/.gitlab-ci.yml
stages:
  - build
  - test
  - report

build-linux-ut:
  stage: build
  script:
    - make PLATFORM=linux_ut
  artifacts:
    paths:
      - build/

test-unit:
  stage: test
  needs: [build-linux-ut]
  script:
    - ./build/bin/test_runner --filter "unit.*" --json --output results.json
  artifacts:
    reports:
      junit: results.xml

generate-report:
  stage: report
  needs: [test-unit]
  script:
    - python -m tools.report --format html --output report.html
  artifacts:
    paths:
      - report.html
```

### 8.3 Jenkins

```groovy
// cicd/jenkins/Jenkinsfile
pipeline {
    agent any

    stages {
        stage('Build') {
            steps {
                sh 'make PLATFORM=linux_ut'
            }
        }

        stage('Test') {
            parallel {
                stage('Unit Tests') {
                    steps {
                        sh './build/bin/test_runner --filter "unit.*"'
                    }
                }
                stage('Functional Tests') {
                    steps {
                        sh './build/bin/test_runner --filter "functional.*"'
                    }
                }
            }
        }

        stage('Report') {
            steps {
                sh 'python -m tools.report --format html'
                publishHTML([reportDir: 'build/reports', reportFiles: 'report.html'])
            }
        }
    }

    post {
        failure {
            emailext subject: 'Build Failed',
                     body: 'Check console output',
                     recipientProviders: [developers()]
        }
    }
}
```

---

## 9. 需求追溯

| 需求ID | 需求标题 | 设计章节 |
|--------|----------|----------|
| REQ-CIC-001 | 验证流水线阶段 | 3.1 |
| REQ-CIC-002 | 流水线触发策略 | 3.2 |
| REQ-CIC-003 | CI作业定义 | 3.4 |
| REQ-CIC-004 | 并行测试执行 | 5 |
| REQ-CIC-005 | 执行环境管理 | 6 |
| REQ-CIC-006 | CI平台集成 | 8 |
| REQ-CIC-007 | 失败处理策略 | 7 |
| REQ-CIC-010 | 框架代码PR检查 | 4.1 |
| REQ-CIC-011 | 框架版本发布流程 | 4.2 |
| REQ-CIC-012 | 文档自动生成发布 | 4.3 |
| REQ-CIC-013 | Python工具包发布 | 4.4 |
