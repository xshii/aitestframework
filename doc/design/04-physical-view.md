# AI测试框架 - 物理视图 (Physical View)

## 概述

物理视图描述系统的部署架构，关注软件到硬件的映射、网络拓扑、环境配置和基础设施。本视图为运维人员和系统管理员提供部署蓝图。

---

## 1. 部署架构概览

### 1.1 部署模式

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Deployment Modes                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. Local Mode (本地模式)                                                    │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         Developer Machine                              │  │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │  │
│  │   │  AI Test    │  │    Model    │  │   Test      │                   │  │
│  │   │  Framework  │  │    Files    │  │   Data      │                   │  │
│  │   └─────────────┘  └─────────────┘  └─────────────┘                   │  │
│  │                                                                        │  │
│  │   Resources: CPU/GPU, Local Storage                                   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  2. CI/CD Mode (持续集成模式)                                                │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         CI Runner                                      │  │
│  │   ┌─────────────┐       ┌─────────────┐                               │  │
│  │   │  AI Test    │◄─────►│  Artifact   │                               │  │
│  │   │  Container  │       │   Storage   │                               │  │
│  │   └─────────────┘       └─────────────┘                               │  │
│  │          │                                                            │  │
│  │          ▼                                                            │  │
│  │   ┌─────────────┐                                                     │  │
│  │   │   Report    │                                                     │  │
│  │   │   Upload    │                                                     │  │
│  │   └─────────────┘                                                     │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  3. Distributed Mode (分布式模式)                                            │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                                                                        │  │
│  │   ┌─────────────┐                                                     │  │
│  │   │ Coordinator │                                                     │  │
│  │   │   Node      │                                                     │  │
│  │   └──────┬──────┘                                                     │  │
│  │          │                                                            │  │
│  │    ┌─────┴─────┬─────────────┬─────────────┐                          │  │
│  │    │           │             │             │                          │  │
│  │    ▼           ▼             ▼             ▼                          │  │
│  │ ┌──────┐   ┌──────┐     ┌──────┐     ┌──────┐                        │  │
│  │ │Worker│   │Worker│     │Worker│     │Worker│                        │  │
│  │ │Node 1│   │Node 2│     │Node 3│     │Node N│                        │  │
│  │ │(GPU) │   │(GPU) │     │(CPU) │     │(CPU) │                        │  │
│  │ └──────┘   └──────┘     └──────┘     └──────┘                        │  │
│  │                                                                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 典型部署拓扑

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Enterprise Deployment Topology                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        User Access Layer                             │    │
│  │                                                                      │    │
│  │   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────────┐    │    │
│  │   │Developer│  │   CI    │  │ API     │  │   Monitoring        │    │    │
│  │   │   CLI   │  │ System  │  │ Client  │  │   Dashboard         │    │    │
│  │   └────┬────┘  └────┬────┘  └────┬────┘  └─────────┬───────────┘    │    │
│  │        │            │            │                 │                │    │
│  └────────┼────────────┼────────────┼─────────────────┼────────────────┘    │
│           │            │            │                 │                     │
│           └────────────┴──────┬─────┴─────────────────┘                     │
│                               │                                             │
│  ┌────────────────────────────▼────────────────────────────────────────┐    │
│  │                      Load Balancer / API Gateway                     │    │
│  └────────────────────────────┬────────────────────────────────────────┘    │
│                               │                                             │
│  ┌────────────────────────────▼────────────────────────────────────────┐    │
│  │                       Application Layer                              │    │
│  │                                                                      │    │
│  │   ┌─────────────────────────────────────────────────────────────┐   │    │
│  │   │                 Kubernetes Cluster                           │   │    │
│  │   │                                                              │   │    │
│  │   │   ┌───────────┐  ┌───────────┐  ┌───────────────────────┐   │   │    │
│  │   │   │Coordinator│  │  REST API │  │    Report Service     │   │   │    │
│  │   │   │  Service  │  │  Service  │  │                       │   │   │    │
│  │   │   │ (1 pod)   │  │ (2 pods)  │  │      (2 pods)         │   │   │    │
│  │   │   └───────────┘  └───────────┘  └───────────────────────┘   │   │    │
│  │   │                                                              │   │    │
│  │   │   ┌─────────────────────────────────────────────────────┐   │   │    │
│  │   │   │              Worker Pool (Auto-scaling)              │   │   │    │
│  │   │   │   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐       │   │   │    │
│  │   │   │   │Worker 1│ │Worker 2│ │Worker 3│ │Worker N│       │   │   │    │
│  │   │   │   │ (GPU)  │ │ (GPU)  │ │ (CPU)  │ │ (CPU)  │       │   │   │    │
│  │   │   │   └────────┘ └────────┘ └────────┘ └────────┘       │   │   │    │
│  │   │   └─────────────────────────────────────────────────────┘   │   │    │
│  │   │                                                              │   │    │
│  │   └──────────────────────────────────────────────────────────────┘   │    │
│  │                                                                      │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                               │                                             │
│  ┌────────────────────────────▼────────────────────────────────────────┐    │
│  │                        Data Layer                                    │    │
│  │                                                                      │    │
│  │   ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────────┐    │    │
│  │   │  Object   │  │  Message  │  │  Cache    │  │   Database    │    │    │
│  │   │  Storage  │  │   Queue   │  │  (Redis)  │  │  (PostgreSQL) │    │    │
│  │   │  (S3/GCS) │  │(RabbitMQ) │  │           │  │               │    │    │
│  │   └───────────┘  └───────────┘  └───────────┘  └───────────────┘    │    │
│  │                                                                      │    │
│  │   Models, Datasets, Reports    Task Queue    Model Cache   Results  │    │
│  │                                                                      │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                               │                                             │
│  ┌────────────────────────────▼────────────────────────────────────────┐    │
│  │                     Monitoring Layer                                 │    │
│  │                                                                      │    │
│  │   ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────────┐    │    │
│  │   │Prometheus │  │  Grafana  │  │   Loki    │  │   AlertManager│    │    │
│  │   │           │  │           │  │           │  │               │    │    │
│  │   └───────────┘  └───────────┘  └───────────┘  └───────────────┘    │    │
│  │                                                                      │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 硬件要求

### 2.1 节点规格

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Node Specifications                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Coordinator Node (协调节点)                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Minimum:                    │  Recommended:                        │    │
│  │  - CPU: 4 cores              │  - CPU: 8 cores                      │    │
│  │  - RAM: 8 GB                 │  - RAM: 16 GB                        │    │
│  │  - Disk: 100 GB SSD          │  - Disk: 500 GB SSD                  │    │
│  │  - Network: 1 Gbps           │  - Network: 10 Gbps                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  CPU Worker Node (CPU工作节点)                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Minimum:                    │  Recommended:                        │    │
│  │  - CPU: 8 cores              │  - CPU: 16+ cores                    │    │
│  │  - RAM: 16 GB                │  - RAM: 64 GB                        │    │
│  │  - Disk: 200 GB SSD          │  - Disk: 1 TB NVMe                   │    │
│  │  - Network: 1 Gbps           │  - Network: 10 Gbps                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  GPU Worker Node (GPU工作节点)                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Minimum:                    │  Recommended:                        │    │
│  │  - CPU: 8 cores              │  - CPU: 16+ cores                    │    │
│  │  - RAM: 32 GB                │  - RAM: 128 GB                       │    │
│  │  - GPU: 1x NVIDIA T4 (16GB)  │  - GPU: 4x NVIDIA A100 (40/80GB)     │    │
│  │  - Disk: 500 GB NVMe         │  - Disk: 2 TB NVMe                   │    │
│  │  - Network: 10 Gbps          │  - Network: 100 Gbps + RDMA          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Storage Node (存储节点)                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Model Storage:              │  Dataset Storage:                    │    │
│  │  - Fast NVMe SSD             │  - High capacity HDD/SSD             │    │
│  │  - Low latency access        │  - Sequential read optimized         │    │
│  │  - ~10-100 GB per model      │  - ~100 GB - 10 TB per dataset       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 GPU支持矩阵

| GPU型号 | 显存 | 支持场景 | 推荐并发 |
|---------|------|----------|----------|
| NVIDIA T4 | 16GB | 中小型模型推理 | 2-4 |
| NVIDIA V100 | 32GB | 大型模型推理 | 4-8 |
| NVIDIA A10 | 24GB | 通用推理测试 | 4-6 |
| NVIDIA A100 | 40/80GB | 超大模型/LLM | 8-16 |
| NVIDIA H100 | 80GB | 最新大模型 | 16-32 |

---

## 3. 容器化部署

### 3.1 Docker镜像架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Docker Image Hierarchy                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                    ┌─────────────────────────────────┐                      │
│                    │      aitest/base:latest         │                      │
│                    │  Python 3.11 + Core Deps        │                      │
│                    └───────────────┬─────────────────┘                      │
│                                    │                                        │
│           ┌────────────────────────┼────────────────────────┐               │
│           │                        │                        │               │
│           ▼                        ▼                        ▼               │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│  │ aitest/cpu:1.0  │    │aitest/cuda:1.0  │    │ aitest/full:1.0 │          │
│  │                 │    │                 │    │                 │          │
│  │ CPU-only        │    │ CUDA 12.1       │    │ All frameworks  │          │
│  │ PyTorch CPU     │    │ PyTorch GPU     │    │ PyTorch+TF+ONNX│          │
│  │ TensorFlow CPU  │    │ TensorFlow GPU  │    │ HuggingFace     │          │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘          │
│                                    │                                        │
│                                    ▼                                        │
│                         ┌─────────────────┐                                 │
│                         │aitest/cuda-dev  │                                 │
│                         │                 │                                 │
│                         │ + Dev tools     │                                 │
│                         │ + Debug symbols │                                 │
│                         └─────────────────┘                                 │
│                                                                             │
│  Image Tags:                                                                 │
│  - latest     : Most recent stable                                          │
│  - x.y.z      : Specific version                                            │
│  - x.y.z-cuda : CUDA version                                                │
│  - nightly    : Development build                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Dockerfile示例

```dockerfile
# aitest/cuda:1.0 Dockerfile

# Base image with CUDA support
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    AITEST_HOME=/opt/aitest

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Create application directory
WORKDIR ${AITEST_HOME}

# Install AI Test Framework
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . .
RUN pip install -e ".[pytorch,report]"

# Create non-root user
RUN useradd -m -s /bin/bash aitest && \
    chown -R aitest:aitest ${AITEST_HOME}
USER aitest

# Default command
ENTRYPOINT ["aitest"]
CMD ["--help"]
```

### 3.3 Docker Compose配置

```yaml
# docker-compose.yml

version: '3.8'

services:
  coordinator:
    image: aitest/cuda:1.0
    container_name: aitest-coordinator
    command: aitest serve --mode coordinator
    ports:
      - "8000:8000"
    environment:
      - AITEST_MODE=coordinator
      - AITEST_WORKERS=4
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://postgres:password@db:5432/aitest
    volumes:
      - ./config:/opt/aitest/config:ro
      - ./reports:/opt/aitest/reports
    depends_on:
      - redis
      - db
    networks:
      - aitest-network

  worker-gpu:
    image: aitest/cuda:1.0
    command: aitest serve --mode worker
    deploy:
      replicas: 2
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - AITEST_MODE=worker
      - COORDINATOR_URL=http://coordinator:8000
      - AITEST_DEVICE=cuda
    volumes:
      - ./models:/opt/aitest/models:ro
      - ./data:/opt/aitest/data:ro
    depends_on:
      - coordinator
    networks:
      - aitest-network

  worker-cpu:
    image: aitest/cpu:1.0
    command: aitest serve --mode worker
    deploy:
      replicas: 4
    environment:
      - AITEST_MODE=worker
      - COORDINATOR_URL=http://coordinator:8000
      - AITEST_DEVICE=cpu
    volumes:
      - ./models:/opt/aitest/models:ro
      - ./data:/opt/aitest/data:ro
    depends_on:
      - coordinator
    networks:
      - aitest-network

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    networks:
      - aitest-network

  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=aitest
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - aitest-network

networks:
  aitest-network:
    driver: bridge

volumes:
  redis-data:
  postgres-data:
```

---

## 4. Kubernetes部署

### 4.1 K8s架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Kubernetes Deployment Architecture                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         Namespace: aitest                              │  │
│  │                                                                        │  │
│  │   ┌──────────────────────────────────────────────────────────────┐    │  │
│  │   │                    Ingress Controller                         │    │  │
│  │   │              (nginx-ingress / traefik)                        │    │  │
│  │   └────────────────────────────┬─────────────────────────────────┘    │  │
│  │                                │                                      │  │
│  │          ┌─────────────────────┼─────────────────────┐                │  │
│  │          │                     │                     │                │  │
│  │          ▼                     ▼                     ▼                │  │
│  │   ┌─────────────┐       ┌─────────────┐       ┌─────────────┐        │  │
│  │   │   Service   │       │   Service   │       │   Service   │        │  │
│  │   │ coordinator │       │   api       │       │   report    │        │  │
│  │   │ ClusterIP   │       │ ClusterIP   │       │ ClusterIP   │        │  │
│  │   └──────┬──────┘       └──────┬──────┘       └──────┬──────┘        │  │
│  │          │                     │                     │                │  │
│  │          ▼                     ▼                     ▼                │  │
│  │   ┌─────────────┐       ┌─────────────┐       ┌─────────────┐        │  │
│  │   │ Deployment  │       │ Deployment  │       │ Deployment  │        │  │
│  │   │coordinator  │       │    api      │       │   report    │        │  │
│  │   │ replicas: 1 │       │ replicas: 2 │       │ replicas: 2 │        │  │
│  │   └─────────────┘       └─────────────┘       └─────────────┘        │  │
│  │                                                                        │  │
│  │   ┌────────────────────────────────────────────────────────────────┐  │  │
│  │   │                    Worker Pool (HPA enabled)                    │  │  │
│  │   │                                                                  │  │  │
│  │   │   ┌───────────────────────────┐   ┌───────────────────────────┐ │  │  │
│  │   │   │     DaemonSet: GPU        │   │   Deployment: CPU         │ │  │  │
│  │   │   │     (GPU nodes only)      │   │   replicas: 2-10 (HPA)    │ │  │  │
│  │   │   │                           │   │                           │ │  │  │
│  │   │   │   ┌─────┐ ┌─────┐        │   │   ┌─────┐ ┌─────┐ ┌─────┐ │ │  │  │
│  │   │   │   │Pod 1│ │Pod 2│        │   │   │Pod 1│ │Pod 2│ │Pod N│ │ │  │  │
│  │   │   │   │GPU:1│ │GPU:1│        │   │   │     │ │     │ │     │ │ │  │  │
│  │   │   │   └─────┘ └─────┘        │   │   └─────┘ └─────┘ └─────┘ │ │  │  │
│  │   │   └───────────────────────────┘   └───────────────────────────┘ │  │  │
│  │   └────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                        │  │
│  │   ┌────────────────────────────────────────────────────────────────┐  │  │
│  │   │                     ConfigMaps & Secrets                        │  │  │
│  │   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │  │  │
│  │   │   │ ConfigMap:   │  │   Secret:    │  │   Secret:            │ │  │  │
│  │   │   │ aitest-config│  │ aitest-creds │  │   registry-creds     │ │  │  │
│  │   │   └──────────────┘  └──────────────┘  └──────────────────────┘ │  │  │
│  │   └────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                        │  │
│  │   ┌────────────────────────────────────────────────────────────────┐  │  │
│  │   │                   Persistent Storage                            │  │  │
│  │   │   ┌──────────────────┐      ┌──────────────────────────────┐   │  │  │
│  │   │   │ PVC: models      │      │ PVC: reports                  │   │  │  │
│  │   │   │ StorageClass: ssd│      │ StorageClass: standard        │   │  │  │
│  │   │   │ Size: 500Gi      │      │ Size: 100Gi                   │   │  │  │
│  │   │   └──────────────────┘      └──────────────────────────────┘   │  │  │
│  │   └────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Helm Chart结构

```
aitest-chart/
├── Chart.yaml
├── values.yaml
├── templates/
│   ├── _helpers.tpl
│   ├── deployment-coordinator.yaml
│   ├── deployment-api.yaml
│   ├── deployment-worker-cpu.yaml
│   ├── daemonset-worker-gpu.yaml
│   ├── service-coordinator.yaml
│   ├── service-api.yaml
│   ├── ingress.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── pvc-models.yaml
│   ├── pvc-reports.yaml
│   ├── hpa-worker-cpu.yaml
│   └── serviceaccount.yaml
└── README.md
```

### 4.3 values.yaml示例

```yaml
# Helm values.yaml

replicaCount:
  coordinator: 1
  api: 2
  workerCpu: 4
  report: 2

image:
  repository: aitest/cuda
  tag: "1.0"
  pullPolicy: IfNotPresent

coordinator:
  resources:
    requests:
      cpu: "2"
      memory: "4Gi"
    limits:
      cpu: "4"
      memory: "8Gi"

workerGpu:
  enabled: true
  nodeSelector:
    nvidia.com/gpu: "true"
  resources:
    requests:
      cpu: "4"
      memory: "16Gi"
      nvidia.com/gpu: 1
    limits:
      cpu: "8"
      memory: "32Gi"
      nvidia.com/gpu: 1

workerCpu:
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 10
    targetCPUUtilizationPercentage: 70
  resources:
    requests:
      cpu: "2"
      memory: "8Gi"
    limits:
      cpu: "4"
      memory: "16Gi"

persistence:
  models:
    enabled: true
    size: 500Gi
    storageClass: ssd
  reports:
    enabled: true
    size: 100Gi
    storageClass: standard

ingress:
  enabled: true
  className: nginx
  hosts:
    - host: aitest.example.com
      paths:
        - path: /api
          pathType: Prefix
          service: api
        - path: /reports
          pathType: Prefix
          service: report

redis:
  enabled: true
  architecture: standalone

postgresql:
  enabled: true
  auth:
    database: aitest
```

---

## 5. CI/CD集成

### 5.1 GitHub Actions工作流

```yaml
# .github/workflows/aitest.yml

name: AI Test Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  AITEST_IMAGE: ghcr.io/${{ github.repository }}/aitest

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e ".[dev,pytorch]"

      - name: Run AI Tests
        run: |
          aitest run tests/model_tests/ \
            --report-format junit,html \
            --output-dir reports/

      - name: Upload Test Report
        uses: actions/upload-artifact@v4
        with:
          name: test-reports
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
          docker run --gpus all \
            -v $(pwd):/workspace \
            ${AITEST_IMAGE}:cuda \
            aitest run tests/gpu_tests/ \
              --device cuda \
              --report-format junit

      - name: Upload GPU Test Results
        uses: actions/upload-artifact@v4
        with:
          name: gpu-test-reports
          path: reports/
```

### 5.2 GitLab CI配置

```yaml
# .gitlab-ci.yml

stages:
  - test
  - gpu-test
  - deploy

variables:
  AITEST_IMAGE: ${CI_REGISTRY_IMAGE}/aitest

test:
  stage: test
  image: python:3.11
  script:
    - pip install -e ".[dev,pytorch]"
    - aitest run tests/ --report-format junit,html
  artifacts:
    reports:
      junit: reports/junit.xml
    paths:
      - reports/
    expire_in: 1 week

gpu-test:
  stage: gpu-test
  tags:
    - gpu
  image: ${AITEST_IMAGE}:cuda
  script:
    - aitest run tests/gpu_tests/ --device cuda
  needs:
    - test
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"

deploy-report:
  stage: deploy
  script:
    - aws s3 sync reports/ s3://aitest-reports/${CI_COMMIT_SHA}/
  needs:
    - test
  rules:
    - if: $CI_COMMIT_BRANCH == "main"
```

---

## 6. 云平台部署

### 6.1 AWS部署架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AWS Deployment                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                              VPC                                       │  │
│  │                                                                        │  │
│  │   Public Subnet                    Private Subnet                      │  │
│  │   ┌─────────────────┐             ┌──────────────────────────────┐    │  │
│  │   │                 │             │                              │    │  │
│  │   │  ┌───────────┐  │             │   ┌────────────────────────┐│    │  │
│  │   │  │    ALB    │──┼─────────────┼──►│      EKS Cluster       ││    │  │
│  │   │  └───────────┘  │             │   │                        ││    │  │
│  │   │                 │             │   │  ┌──────┐  ┌──────┐    ││    │  │
│  │   │  ┌───────────┐  │             │   │  │ CPU  │  │ GPU  │    ││    │  │
│  │   │  │  NAT GW   │  │             │   │  │Nodes │  │Nodes │    ││    │  │
│  │   │  └───────────┘  │             │   │  │(EC2) │  │(P4d) │    ││    │  │
│  │   │                 │             │   │  └──────┘  └──────┘    ││    │  │
│  │   └─────────────────┘             │   └────────────────────────┘│    │  │
│  │                                   │                              │    │  │
│  │                                   │   ┌──────────┐  ┌──────────┐│    │  │
│  │                                   │   │ElastiCache  │   RDS    ││    │  │
│  │                                   │   │ (Redis)  │  │(Postgres)││    │  │
│  │                                   │   └──────────┘  └──────────┘│    │  │
│  │                                   │                              │    │  │
│  │                                   └──────────────────────────────┘    │  │
│  │                                                                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  External Services:                                                          │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────────────────┐    │
│  │    S3     │  │    ECR    │  │CloudWatch │  │      SageMaker        │    │
│  │  Models/  │  │  Images   │  │Logs/Metrics│ │ (Optional: Inference) │    │
│  │  Data     │  │           │  │           │  │                       │    │
│  └───────────┘  └───────────┘  └───────────┘  └───────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 云服务集成

| 云平台 | 计算 | 存储 | GPU | 容器 |
|--------|------|------|-----|------|
| **AWS** | EC2, Lambda | S3, EFS | P4d, G5 | EKS, ECS |
| **Azure** | VM, Functions | Blob, Files | NC, ND | AKS |
| **GCP** | Compute, Functions | GCS, Filestore | A100, T4 | GKE |
| **阿里云** | ECS, FC | OSS, NAS | V100, T4 | ACK |

---

## 7. 监控与告警

### 7.1 监控架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Monitoring Architecture                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    AI Test Framework Nodes                             │  │
│  │                                                                        │  │
│  │   ┌──────────────────┐    ┌──────────────────┐                        │  │
│  │   │  Prometheus      │    │     Loki         │                        │  │
│  │   │  Exporter        │    │  Log Agent       │                        │  │
│  │   │  (metrics)       │    │  (logs)          │                        │  │
│  │   └────────┬─────────┘    └────────┬─────────┘                        │  │
│  │            │                       │                                   │  │
│  └────────────┼───────────────────────┼───────────────────────────────────┘  │
│               │                       │                                      │
│               ▼                       ▼                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    Monitoring Stack                                    │  │
│  │                                                                        │  │
│  │   ┌──────────────────┐    ┌──────────────────┐                        │  │
│  │   │   Prometheus     │    │      Loki        │                        │  │
│  │   │   (Time Series)  │    │  (Log Storage)   │                        │  │
│  │   └────────┬─────────┘    └────────┬─────────┘                        │  │
│  │            │                       │                                   │  │
│  │            └───────────┬───────────┘                                   │  │
│  │                        │                                               │  │
│  │                        ▼                                               │  │
│  │            ┌──────────────────────┐                                   │  │
│  │            │       Grafana        │                                   │  │
│  │            │    (Visualization)   │                                   │  │
│  │            └──────────┬───────────┘                                   │  │
│  │                       │                                                │  │
│  │                       ▼                                                │  │
│  │            ┌──────────────────────┐                                   │  │
│  │            │    AlertManager      │                                   │  │
│  │            │     (Alerting)       │                                   │  │
│  │            └──────────────────────┘                                   │  │
│  │                       │                                                │  │
│  └───────────────────────┼────────────────────────────────────────────────┘  │
│                          │                                                   │
│                          ▼                                                   │
│             ┌────────────────────────────┐                                   │
│             │     Notification Channels  │                                   │
│             │  Slack / Email / PagerDuty │                                   │
│             └────────────────────────────┘                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 关键监控指标

| 类别 | 指标 | 告警阈值 |
|------|------|----------|
| **系统** | CPU使用率 | > 80% |
| | 内存使用率 | > 85% |
| | 磁盘使用率 | > 90% |
| | GPU使用率 | < 20% (闲置告警) |
| **应用** | 测试执行时间 | > 基准 * 1.5 |
| | 测试失败率 | > 10% |
| | Worker存活数 | < 最小副本数 |
| **队列** | 任务队列深度 | > 100 |
| | 处理延迟 | > 60s |

---

## 8. 安全配置

### 8.1 网络安全

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Network Security                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Network Policies:                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                                                                     │    │
│  │   Internet ─────► Ingress ─────► API Service                        │    │
│  │                     │               │                               │    │
│  │                     │ (HTTPS only)  │                               │    │
│  │                     │               ▼                               │    │
│  │                     │         ┌───────────┐                         │    │
│  │                     │         │Coordinator│                         │    │
│  │                     │         └─────┬─────┘                         │    │
│  │                     │               │                               │    │
│  │                     │               │ (Internal only)               │    │
│  │                     │               ▼                               │    │
│  │                     │         ┌───────────┐                         │    │
│  │                     │         │  Workers  │                         │    │
│  │                     │         └─────┬─────┘                         │    │
│  │                     │               │                               │    │
│  │                     │               │ (Internal only)               │    │
│  │                     │               ▼                               │    │
│  │                     │    ┌─────────────────────┐                    │    │
│  │                     │    │  Database / Cache   │                    │    │
│  │                     │    │  (No external access)│                   │    │
│  │                     │    └─────────────────────┘                    │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Security Groups / Firewall Rules:                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Component      │ Inbound              │ Outbound                   │    │
│  ├─────────────────┼──────────────────────┼────────────────────────────┤    │
│  │  Load Balancer  │ 443 (HTTPS)          │ 8000 (API), 8001 (Report)  │    │
│  │  API Service    │ 8000 (Internal)      │ Redis, PostgreSQL          │    │
│  │  Workers        │ None                 │ Model Storage, Coordinator │    │
│  │  Database       │ 5432 (Internal only) │ None                       │    │
│  │  Redis          │ 6379 (Internal only) │ None                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 密钥管理

| 密钥类型 | 存储方式 | 访问控制 |
|----------|----------|----------|
| API密钥 | K8s Secret / Vault | RBAC |
| 数据库密码 | K8s Secret / Vault | 仅应用Pod |
| 模型访问Token | K8s Secret | Worker Pod |
| TLS证书 | cert-manager | Ingress Controller |
| 云服务凭证 | IAM Role / Workload Identity | Pod Service Account |

---

## 9. 需求到部署的映射

| 需求ID | 部署组件 | 资源需求 | 部署模式 |
|--------|----------|----------|----------|
| INTEG-001 | CI Runner | CPU: 2核, 内存: 4GB | CI/CD |
| INTEG-005 | Docker镜像 | 基础镜像 ~2GB | 容器化 |
| INTEG-006 | K8s集群 | 按需扩展 | 分布式 |
| INTEG-007 | 模型服务连接器 | 网络访问 | 服务集成 |
| INTEG-008 | 云平台SDK | IAM权限 | 云平台 |
| INTEG-009 | Prometheus/Grafana | 监控资源 | 监控 |
| CORE-003-03 | 多节点Worker | GPU节点 | 分布式 |

---

*本文档为AI测试框架物理视图设计，详细描述了系统的部署架构和基础设施配置。*
