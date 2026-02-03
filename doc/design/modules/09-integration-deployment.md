# 集成与部署模块详细设计 (Integration & Deployment)

## 模块概述

| 属性 | 值 |
|------|-----|
| **模块ID** | INTEG |
| **模块名称** | 集成与部署 |
| **职责** | CI/CD集成、环境配置、容器化部署 |
| **需求覆盖** | INTEG-001 ~ INTEG-010 |

---

## 1. 逻辑视图

### 1.1 集成组件类图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Integration & Deployment Classes                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         CI Integration                                │  │
│  │                                                                       │  │
│  │              ┌─────────────────────────────┐                          │  │
│  │              │     <<interface>>           │                          │  │
│  │              │      ICIAdapter             │                          │  │
│  │              ├─────────────────────────────┤                          │  │
│  │              │ + configure(config)         │                          │  │
│  │              │ + get_template() -> str     │                          │  │
│  │              │ + upload_report(report)     │                          │  │
│  │              │ + set_status(status)        │                          │  │
│  │              └──────────────┬──────────────┘                          │  │
│  │                             │                                         │  │
│  │       ┌─────────────────────┼─────────────────────┐                   │  │
│  │       ▼                     ▼                     ▼                   │  │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐            │  │
│  │  │GitHubActions │    │  GitLabCI    │    │   Jenkins    │            │  │
│  │  │  Adapter     │    │   Adapter    │    │   Adapter    │            │  │
│  │  └──────────────┘    └──────────────┘    └──────────────┘            │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      Container Support                                │  │
│  │                                                                       │  │
│  │  ┌──────────────────────┐    ┌──────────────────────┐                 │  │
│  │  │   DockerManager      │    │  KubernetesManager   │                 │  │
│  │  ├──────────────────────┤    ├──────────────────────┤                 │  │
│  │  │ + build_image()      │    │ + deploy()           │                 │  │
│  │  │ + push_image()       │    │ + scale(replicas)    │                 │  │
│  │  │ + run_container()    │    │ + get_pods()         │                 │  │
│  │  │ + get_logs()         │    │ + get_logs()         │                 │  │
│  │  └──────────────────────┘    └──────────────────────┘                 │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      Model Serving Integration                        │  │
│  │                                                                       │  │
│  │  ┌──────────────────────┐    ┌──────────────────────┐                 │  │
│  │  │  TorchServeClient    │    │ TFServingClient      │                 │  │
│  │  ├──────────────────────┤    ├──────────────────────┤                 │  │
│  │  │ + predict(input)     │    │ + predict(input)     │                 │  │
│  │  │ + health_check()     │    │ + health_check()     │                 │  │
│  │  │ + get_model_info()   │    │ + get_model_info()   │                 │  │
│  │  └──────────────────────┘    └──────────────────────┘                 │  │
│  │                                                                       │  │
│  │  ┌──────────────────────┐    ┌──────────────────────┐                 │  │
│  │  │   TritonClient       │    │    GenericClient     │                 │  │
│  │  ├──────────────────────┤    ├──────────────────────┤                 │  │
│  │  │ + predict(input)     │    │ + predict(input)     │                 │  │
│  │  │ + health_check()     │    │ + health_check()     │                 │  │
│  │  │ + get_model_status() │    │ + configure(config)  │                 │  │
│  │  └──────────────────────┘    └──────────────────────┘                 │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      Environment Management                           │  │
│  │                                                                       │  │
│  │  ┌──────────────────────┐    ┌──────────────────────┐                 │  │
│  │  │ EnvironmentManager   │    │  DependencyManager   │                 │  │
│  │  ├──────────────────────┤    ├──────────────────────┤                 │  │
│  │  │ + get_config(env)    │    │ + install(deps)      │                 │  │
│  │  │ + switch_env(env)    │    │ + check_versions()   │                 │  │
│  │  │ + get_secrets()      │    │ + lock_versions()    │                 │  │
│  │  └──────────────────────┘    └──────────────────────┘                 │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 开发视图

### 2.1 包结构

```
aitest/integration/
├── __init__.py
├── ci/
│   ├── __init__.py
│   ├── base.py              # ICIAdapter接口
│   ├── github.py            # GitHub Actions
│   ├── gitlab.py            # GitLab CI
│   ├── jenkins.py           # Jenkins
│   └── templates/           # CI配置模板
│       ├── github_actions.yml
│       ├── gitlab_ci.yml
│       └── Jenkinsfile
├── container/
│   ├── __init__.py
│   ├── docker.py            # Docker管理
│   └── kubernetes.py        # K8s管理
├── serving/
│   ├── __init__.py
│   ├── base.py              # IServingClient接口
│   ├── torchserve.py        # TorchServe客户端
│   ├── tfserving.py         # TF Serving客户端
│   └── triton.py            # Triton客户端
├── cloud/
│   ├── __init__.py
│   ├── aws.py               # AWS集成
│   ├── azure.py             # Azure集成
│   └── gcp.py               # GCP集成
├── monitoring/
│   ├── __init__.py
│   ├── prometheus.py        # Prometheus导出
│   └── grafana.py           # Grafana面板
├── environment.py           # 环境管理
└── dependency.py            # 依赖管理
```

### 2.2 CI模板示例

```yaml
# integration/ci/templates/github_actions.yml

name: AI Test Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest]
        python-version: ['3.9', '3.10', '3.11']

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install AI Test Framework
        run: pip install aitest-framework[all]

      - name: Run Tests
        run: |
          aitest run tests/ \
            --report-format junit,html \
            --output-dir reports/

      - name: Upload Test Results
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: test-reports-${{ matrix.python-version }}
          path: reports/

      - name: Publish Test Results
        uses: EnricoMi/publish-unit-test-result-action@v2
        if: always()
        with:
          files: reports/junit.xml

  gpu-test:
    runs-on: [self-hosted, gpu]
    needs: test
    steps:
      - uses: actions/checkout@v4

      - name: Run GPU Tests
        run: |
          aitest run tests/gpu/ \
            --device cuda \
            --report-format junit
```

### 2.3 实现示例

```python
# integration/ci/github.py

from typing import Optional, Dict, Any
from pathlib import Path
import os

from .base import ICIAdapter


class GitHubActionsAdapter(ICIAdapter):
    """GitHub Actions适配器"""

    def __init__(self):
        self.is_ci = os.environ.get('GITHUB_ACTIONS') == 'true'
        self.run_id = os.environ.get('GITHUB_RUN_ID')
        self.repository = os.environ.get('GITHUB_REPOSITORY')

    def configure(self, config: Dict[str, Any]) -> None:
        """配置适配器"""
        self.config = config

    def get_template(self) -> str:
        """获取工作流模板"""
        template_path = Path(__file__).parent / 'templates' / 'github_actions.yml'
        return template_path.read_text()

    def upload_report(self, report_path: Path) -> str:
        """上传报告到Artifacts"""
        if not self.is_ci:
            return str(report_path)

        # 使用 @actions/artifact 上传
        print(f"::set-output name=report_path::{report_path}")
        return f"https://github.com/{self.repository}/actions/runs/{self.run_id}"

    def set_status(self, status: str, message: str = "") -> None:
        """设置检查状态"""
        if status == 'success':
            print(f"::notice::{message}")
        elif status == 'failure':
            print(f"::error::{message}")
        elif status == 'warning':
            print(f"::warning::{message}")

    def add_summary(self, content: str) -> None:
        """添加Job Summary"""
        summary_file = os.environ.get('GITHUB_STEP_SUMMARY')
        if summary_file:
            with open(summary_file, 'a') as f:
                f.write(content + '\n')


# integration/serving/triton.py

import numpy as np
from typing import Dict, Any, List, Optional
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient

from .base import IServingClient


class TritonClient(IServingClient):
    """NVIDIA Triton Inference Server客户端"""

    def __init__(
        self,
        url: str,
        model_name: str,
        protocol: str = "http",
        verbose: bool = False
    ):
        self.url = url
        self.model_name = model_name
        self.protocol = protocol

        if protocol == "http":
            self.client = httpclient.InferenceServerClient(url, verbose=verbose)
        else:
            self.client = grpcclient.InferenceServerClient(url, verbose=verbose)

    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """执行推理"""
        infer_inputs = []
        for name, data in inputs.items():
            infer_input = self._create_input(name, data)
            infer_inputs.append(infer_input)

        results = self.client.infer(
            model_name=self.model_name,
            inputs=infer_inputs
        )

        outputs = {}
        for output in self._get_output_names():
            outputs[output] = results.as_numpy(output)

        return outputs

    def health_check(self) -> bool:
        """健康检查"""
        try:
            return self.client.is_server_live()
        except Exception:
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        metadata = self.client.get_model_metadata(self.model_name)
        return {
            'name': metadata.name,
            'versions': metadata.versions,
            'inputs': [{'name': i.name, 'shape': i.shape} for i in metadata.inputs],
            'outputs': [{'name': o.name, 'shape': o.shape} for o in metadata.outputs]
        }

    def _create_input(self, name: str, data: np.ndarray):
        """创建推理输入"""
        if self.protocol == "http":
            infer_input = httpclient.InferInput(name, data.shape, "FP32")
            infer_input.set_data_from_numpy(data)
        else:
            infer_input = grpcclient.InferInput(name, data.shape, "FP32")
            infer_input.set_data_from_numpy(data)
        return infer_input

    def _get_output_names(self) -> List[str]:
        """获取输出名称"""
        metadata = self.client.get_model_metadata(self.model_name)
        return [o.name for o in metadata.outputs]
```

---

## 3. 物理视图

### 3.1 部署架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Deployment Architecture                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         CI/CD Pipeline                              │    │
│  │                                                                     │    │
│  │   GitHub/GitLab ──► Build ──► Test ──► Deploy ──► Monitor          │    │
│  │                                                                     │    │
│  └──────────────────────────────┬──────────────────────────────────────┘    │
│                                 │                                           │
│                                 ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      Kubernetes Cluster                             │    │
│  │                                                                     │    │
│  │   ┌───────────────┐  ┌───────────────┐  ┌───────────────┐          │    │
│  │   │  Test Runner  │  │ Model Server  │  │   Reporting   │          │    │
│  │   │    Pods       │  │    Pods       │  │    Service    │          │    │
│  │   └───────────────┘  └───────────────┘  └───────────────┘          │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      External Services                              │    │
│  │                                                                     │    │
│  │   ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐       │    │
│  │   │  S3/GCS   │  │Prometheus │  │  Grafana  │  │   Slack   │       │    │
│  │   │ (Storage) │  │(Metrics)  │  │(Dashboard)│  │ (Notify)  │       │    │
│  │   └───────────┘  └───────────┘  └───────────┘  └───────────┘       │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 场景视图

### 4.1 需求追溯

| 需求ID | 实现类/方法 | 测试用例 |
|--------|-------------|----------|
| INTEG-001 | `GitHubActionsAdapter`, `GitLabCIAdapter` | test_ci_integration |
| INTEG-002 | `ICIAdapter.upload_report()` | test_report_upload |
| INTEG-003 | `EnvironmentManager` | test_environment |
| INTEG-004 | `DependencyManager` | test_dependencies |
| INTEG-005 | `DockerManager` | test_docker |
| INTEG-006 | `KubernetesManager` | test_kubernetes |
| INTEG-007 | `TorchServeClient`, `TritonClient` | test_serving |
| INTEG-008 | `cloud/*` | test_cloud |
| INTEG-009 | `PrometheusExporter` | test_monitoring |
| INTEG-010 | `setup.py`, `pyproject.toml` | test_packaging |

---

*本文档为集成与部署模块的详细设计，基于4+1视图方法。*
