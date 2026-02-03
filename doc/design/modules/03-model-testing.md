# AI模型测试模块详细设计 (Model Testing)

## 模块概述

| 属性 | 值 |
|------|-----|
| **模块ID** | MODEL |
| **模块名称** | AI模型测试 |
| **职责** | 针对AI/ML模型的专项测试能力 |
| **需求覆盖** | MODEL-001 ~ MODEL-009 |

---

## 1. 逻辑视图

### 1.1 模块类图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Model Testing Classes                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        Model Loader Hierarchy                         │  │
│  │                                                                       │  │
│  │              ┌─────────────────────────────┐                          │  │
│  │              │     <<interface>>           │                          │  │
│  │              │      IModelLoader           │                          │  │
│  │              ├─────────────────────────────┤                          │  │
│  │              │ + load(path) -> LoadedModel │                          │  │
│  │              │ + supports(format) -> bool  │                          │  │
│  │              │ + get_metadata() -> Dict    │                          │  │
│  │              └──────────────┬──────────────┘                          │  │
│  │                             │                                         │  │
│  │       ┌─────────────────────┼─────────────────────┐                   │  │
│  │       │                     │                     │                   │  │
│  │       ▼                     ▼                     ▼                   │  │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐            │  │
│  │  │PyTorchLoader │    │TensorFlow    │    │ ONNXLoader   │            │  │
│  │  │              │    │Loader        │    │              │            │  │
│  │  └──────────────┘    └──────────────┘    └──────────────┘            │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        Testing Components                             │  │
│  │                                                                       │  │
│  │  ┌──────────────────────┐    ┌──────────────────────┐                 │  │
│  │  │  InferenceValidator  │    │  AccuracyEvaluator   │                 │  │
│  │  ├──────────────────────┤    ├──────────────────────┤                 │  │
│  │  │ - model: LoadedModel │    │ - model: LoadedModel │                 │  │
│  │  │ - validators: List   │    │ - dataset: Dataset   │                 │  │
│  │  ├──────────────────────┤    │ - metrics: List      │                 │  │
│  │  │ + validate_output()  │    ├──────────────────────┤                 │  │
│  │  │ + validate_shape()   │    │ + evaluate() -> Metrics               │  │
│  │  │ + validate_range()   │    │ + compute_confusion() │                │  │
│  │  │ + validate_determinism()   + per_class_metrics()  │                │  │
│  │  └──────────────────────┘    └──────────────────────┘                 │  │
│  │                                                                       │  │
│  │  ┌──────────────────────┐    ┌──────────────────────┐                 │  │
│  │  │  PerformanceTester   │    │  RobustnessTester    │                 │  │
│  │  ├──────────────────────┤    ├──────────────────────┤                 │  │
│  │  │ - warmup_iters: int  │    │ - perturbations: List│                 │  │
│  │  │ - test_iters: int    │    │ - noise_levels: List │                 │  │
│  │  │ - profiler: Profiler │    ├──────────────────────┤                 │  │
│  │  ├──────────────────────┤    │ + test_noise()       │                 │  │
│  │  │ + measure_latency()  │    │ + test_boundary()    │                 │  │
│  │  │ + measure_throughput()     + test_adversarial()  │                 │  │
│  │  │ + measure_memory()   │    │ + generate_perturb() │                 │  │
│  │  │ + profile_gpu()      │    └──────────────────────┘                 │  │
│  │  └──────────────────────┘                                             │  │
│  │                                                                       │  │
│  │  ┌──────────────────────┐    ┌──────────────────────┐                 │  │
│  │  │ ConsistencyTester    │    │     LLMTester        │                 │  │
│  │  ├──────────────────────┤    ├──────────────────────┤                 │  │
│  │  │ - devices: List      │    │ - generation_config  │                 │  │
│  │  │ - batch_sizes: List  │    │ - evaluators: List   │                 │  │
│  │  ├──────────────────────┤    ├──────────────────────┤                 │  │
│  │  │ + test_cpu_gpu()     │    │ + test_prompt()      │                 │  │
│  │  │ + test_batch_size()  │    │ + test_context()     │                 │  │
│  │  │ + test_precision()   │    │ + test_safety()      │                 │  │
│  │  └──────────────────────┘    │ + evaluate_quality() │                 │  │
│  │                              └──────────────────────┘                 │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                          Metrics Package                              │  │
│  │                                                                       │  │
│  │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐          │  │
│  │  │Classification   │ │ Regression      │ │ Detection       │          │  │
│  │  │Metrics          │ │ Metrics         │ │ Metrics         │          │  │
│  │  ├─────────────────┤ ├─────────────────┤ ├─────────────────┤          │  │
│  │  │ accuracy()      │ │ mse()           │ │ mAP()           │          │  │
│  │  │ precision()     │ │ mae()           │ │ iou()           │          │  │
│  │  │ recall()        │ │ rmse()          │ │ precision@k()   │          │  │
│  │  │ f1_score()      │ │ r2_score()      │ │ recall@k()      │          │  │
│  │  └─────────────────┘ └─────────────────┘ └─────────────────┘          │  │
│  │                                                                       │  │
│  │  ┌─────────────────┐ ┌─────────────────┐                              │  │
│  │  │ NLP Metrics     │ │ Performance     │                              │  │
│  │  │                 │ │ Metrics         │                              │  │
│  │  ├─────────────────┤ ├─────────────────┤                              │  │
│  │  │ bleu()          │ │ LatencyStats    │                              │  │
│  │  │ rouge()         │ │ ThroughputStats │                              │  │
│  │  │ perplexity()    │ │ MemoryStats     │                              │  │
│  │  └─────────────────┘ └─────────────────┘                              │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 关键接口

```python
# model/interfaces.py

from typing import Protocol, Union, Dict, Any, List
from pathlib import Path
import numpy as np

class IModelLoader(Protocol):
    """模型加载器接口"""

    def load(self, path: Union[str, Path], **kwargs) -> 'LoadedModel':
        """加载模型"""
        ...

    def supports(self, format: str) -> bool:
        """检查是否支持指定格式"""
        ...

    def get_metadata(self, model: 'LoadedModel') -> Dict[str, Any]:
        """获取模型元数据"""
        ...


class IModelInference(Protocol):
    """模型推理接口"""

    def predict(self, input: np.ndarray) -> np.ndarray:
        """单样本推理"""
        ...

    def batch_predict(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        """批量推理"""
        ...

    def warmup(self, samples: int = 10) -> None:
        """预热"""
        ...


class IMetric(Protocol):
    """评估指标接口"""

    def compute(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """计算指标"""
        ...

    @property
    def name(self) -> str:
        """指标名称"""
        ...

    @property
    def higher_is_better(self) -> bool:
        """是否越高越好"""
        ...
```

### 1.3 数据模型

```python
# model/models.py

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum

class ModelFramework(Enum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"
    HUGGINGFACE = "huggingface"


@dataclass
class LoadedModel:
    """已加载的模型"""
    model: Any
    framework: ModelFramework
    device: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def predict(self, input: Any) -> Any:
        """执行推理"""
        ...

    def to_device(self, device: str) -> 'LoadedModel':
        """移动到指定设备"""
        ...


@dataclass
class ModelMetadata:
    """模型元数据"""
    name: str
    version: str
    framework: ModelFramework
    input_shape: List[int]
    output_shape: List[int]
    parameters_count: int
    file_size_mb: float
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccuracyMetrics:
    """精度指标"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    per_class: Dict[str, 'ClassMetrics'] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None


@dataclass
class PerformanceMetrics:
    """性能指标"""
    latency_mean: float
    latency_p50: float
    latency_p90: float
    latency_p99: float
    latency_std: float
    throughput_qps: float
    memory_peak_mb: float
    gpu_memory_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None


@dataclass
class LLMGenerationResult:
    """LLM生成结果"""
    prompt: str
    response: str
    tokens_generated: int
    generation_time: float
    finish_reason: str
```

---

## 2. 进程视图

### 2.1 模型加载流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Model Loading Process                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌───────────────┐    ┌───────────────┐    ┌─────────────┐ │
│  │  Request │    │ Model Loader  │    │   Model       │    │   Model     │ │
│  │  Load    │───►│   Factory     │───►│   Cache       │───►│  Instance   │ │
│  └──────────┘    └───────────────┘    └───────────────┘    └─────────────┘ │
│                          │                   │                             │
│                          ▼                   ▼                             │
│                  ┌───────────────┐    ┌───────────────┐                    │
│                  │ Detect Format │    │ Check Cache   │                    │
│                  │ (pt/h5/onnx)  │    │ (LRU)         │                    │
│                  └───────────────┘    └───────────────┘                    │
│                          │                   │                             │
│                          ▼                   │                             │
│                  ┌───────────────┐           │                             │
│                  │Select Loader  │           │                             │
│                  │(PyTorch/TF/   │           │                             │
│                  │ ONNX)         │           │                             │
│                  └───────┬───────┘           │                             │
│                          │                   │                             │
│                          ▼                   │                             │
│                  ┌───────────────┐           │                             │
│                  │ Load to       │◄──────────┘                             │
│                  │ Device        │                                         │
│                  │ (CPU/GPU)     │                                         │
│                  └───────────────┘                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 推理测试流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Inference Testing Process                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Main Thread                                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │  [Load Model] ──► [Load Dataset] ──► [Create Batches] ──►           │   │
│  │                                                                      │   │
│  │                    ┌──────────────────────────────────┐              │   │
│  │                    │     Inference Loop               │              │   │
│  │                    │                                  │              │   │
│  │                    │  for batch in batches:           │              │   │
│  │                    │    predictions = model(batch)    │              │   │
│  │                    │    validate(predictions)         │              │   │
│  │                    │    collect_metrics()             │              │   │
│  │                    │                                  │              │   │
│  │                    └──────────────────────────────────┘              │   │
│  │                                       │                              │   │
│  │                                       ▼                              │   │
│  │                    ┌──────────────────────────────────┐              │   │
│  │                    │     Compute Metrics              │              │   │
│  │                    │                                  │              │   │
│  │                    │  accuracy = compute_accuracy()   │              │   │
│  │                    │  latency = compute_latency()     │              │   │
│  │                    │                                  │              │   │
│  │                    └──────────────────────────────────┘              │   │
│  │                                       │                              │   │
│  │                                       ▼                              │   │
│  │                    [Generate Results] ──► [Return]                   │   │
│  │                                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 性能测试并发模型

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Performance Testing Concurrency                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Latency Test (Sequential):                                                  │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                                   │
│  │ I1  │─│ I2  │─│ I3  │─│ I4  │─│ IN  │  (measure each)                   │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘                                   │
│                                                                             │
│  Throughput Test (Concurrent):                                               │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                                           │
│  │ T1  │ │ T2  │ │ T3  │ │ T4  │  (parallel workers)                       │
│  │─────│ │─────│ │─────│ │─────│                                           │
│  │ I1  │ │ I2  │ │ I3  │ │ I4  │                                           │
│  │ I5  │ │ I6  │ │ I7  │ │ I8  │                                           │
│  │ ... │ │ ... │ │ ... │ │ ... │                                           │
│  └─────┘ └─────┘ └─────┘ └─────┘                                           │
│      │       │       │       │                                             │
│      └───────┴───────┴───────┘                                             │
│              │                                                             │
│              ▼                                                             │
│       QPS = total_requests / duration                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 开发视图

### 3.1 包结构

```
aitest/model/
├── __init__.py
├── loader/
│   ├── __init__.py
│   ├── base.py                  # IModelLoader接口
│   ├── pytorch.py               # PyTorch加载器
│   ├── tensorflow.py            # TensorFlow加载器
│   ├── onnx.py                  # ONNX加载器
│   ├── huggingface.py           # HuggingFace加载器
│   └── factory.py               # 加载器工厂
├── inference.py                 # 推理验证
├── accuracy.py                  # 精度评估
├── performance.py               # 性能测试
├── robustness.py                # 鲁棒性测试
├── consistency.py               # 一致性测试
├── llm/
│   ├── __init__.py
│   ├── tester.py                # LLM测试器
│   ├── prompt.py                # 提示词测试
│   ├── generation.py            # 生成评估
│   ├── safety.py                # 安全性测试
│   └── evaluators.py            # 评估器
├── metrics/
│   ├── __init__.py
│   ├── base.py                  # IMetric接口
│   ├── classification.py        # 分类指标
│   ├── regression.py            # 回归指标
│   ├── detection.py             # 检测指标
│   ├── nlp.py                   # NLP指标
│   └── performance.py           # 性能指标
├── decorators.py                # @model_test等装饰器
└── cache.py                     # 模型缓存
```

### 3.2 实现示例

```python
# model/loader/pytorch.py

import torch
from pathlib import Path
from typing import Union, Dict, Any
import logging

from .base import IModelLoader
from ..models import LoadedModel, ModelFramework, ModelMetadata

logger = logging.getLogger(__name__)


class PyTorchLoader(IModelLoader):
    """PyTorch模型加载器"""

    SUPPORTED_EXTENSIONS = ['.pt', '.pth', '.pkl']

    def __init__(self, default_device: str = "cpu"):
        self.default_device = default_device

    def supports(self, format: str) -> bool:
        """检查是否支持指定格式"""
        return format.lower() in ['pytorch', 'pt', 'pth']

    def load(
        self,
        path: Union[str, Path],
        device: str = None,
        map_location: str = None,
        **kwargs
    ) -> LoadedModel:
        """加载PyTorch模型"""
        path = Path(path)
        device = device or self.default_device

        logger.info(f"Loading PyTorch model from {path}")

        # 加载模型
        try:
            if map_location:
                model = torch.load(path, map_location=map_location)
            else:
                model = torch.load(path, map_location=device)

            # 如果是state_dict，需要实例化模型
            if isinstance(model, dict) and 'state_dict' in model:
                raise ValueError("Model state_dict requires model class to load")

            # 移动到目标设备
            if hasattr(model, 'to'):
                model = model.to(device)

            # 设置为评估模式
            if hasattr(model, 'eval'):
                model.eval()

            return LoadedModel(
                model=model,
                framework=ModelFramework.PYTORCH,
                device=device,
                metadata=self._extract_metadata(model, path)
            )

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def get_metadata(self, model: LoadedModel) -> Dict[str, Any]:
        """获取模型元数据"""
        return model.metadata

    def _extract_metadata(self, model: torch.nn.Module, path: Path) -> Dict[str, Any]:
        """提取模型元数据"""
        param_count = sum(p.numel() for p in model.parameters())
        file_size = path.stat().st_size / (1024 * 1024)  # MB

        return {
            "parameters_count": param_count,
            "file_size_mb": file_size,
            "model_class": model.__class__.__name__,
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad)
        }


# model/accuracy.py

from typing import List, Dict, Optional
import numpy as np

from .metrics import ClassificationMetrics
from .models import AccuracyMetrics, LoadedModel


class AccuracyEvaluator:
    """精度评估器"""

    def __init__(
        self,
        model: LoadedModel,
        dataset: 'Dataset',
        metrics: Optional[List['IMetric']] = None,
        batch_size: int = 32
    ):
        self.model = model
        self.dataset = dataset
        self.metrics = metrics or ClassificationMetrics.default_metrics()
        self.batch_size = batch_size

    def evaluate(self) -> AccuracyMetrics:
        """执行评估"""
        all_predictions = []
        all_labels = []

        # 批量推理
        for batch in self.dataset.batches(self.batch_size):
            inputs, labels = batch
            predictions = self.model.predict(inputs)

            all_predictions.append(predictions)
            all_labels.append(labels)

        predictions = np.concatenate(all_predictions)
        labels = np.concatenate(all_labels)

        # 计算指标
        results = AccuracyMetrics(
            accuracy=self._compute_accuracy(predictions, labels),
            precision=self._compute_precision(predictions, labels),
            recall=self._compute_recall(predictions, labels),
            f1_score=self._compute_f1(predictions, labels),
            confusion_matrix=self._compute_confusion_matrix(predictions, labels)
        )

        # 计算每类指标
        results.per_class = self._compute_per_class_metrics(predictions, labels)

        return results

    def _compute_accuracy(self, preds: np.ndarray, labels: np.ndarray) -> float:
        """计算准确率"""
        pred_classes = np.argmax(preds, axis=-1)
        return np.mean(pred_classes == labels)

    # ... 其他指标计算方法
```

---

## 4. 物理视图

### 4.1 资源需求

| 测试类型 | GPU | 内存 | 典型时长 |
|----------|-----|------|----------|
| 推理正确性 | 推荐 | 模型大小 * 2 | 分钟级 |
| 精度评估 | 推荐 | 模型 + 数据集 | 分钟~小时 |
| 性能测试 | 必需 | 模型 * 并发数 | 分钟级 |
| LLM测试 | 必需 | 模型大小 | 小时级 |

### 4.2 GPU配置

```yaml
model:
  default_device: cuda
  gpu_memory_fraction: 0.9
  allow_growth: true

  loader:
    pytorch:
      map_location: auto
      strict: true
    tensorflow:
      allow_soft_placement: true
    onnx:
      providers: ["CUDAExecutionProvider", "CPUExecutionProvider"]

  performance:
    warmup_iterations: 50
    test_iterations: 1000
    concurrent_workers: 4
```

---

## 5. 场景视图

### 5.1 核心用例

**UC-MODEL-01: 验证模型推理正确性**

```python
@model_test(model="models/resnet50.pt", framework="pytorch", device="cuda")
def test_inference_correctness(model, dataset):
    for sample in dataset:
        output = model.predict(sample.input)

        assert_that(output).has_shape([1, 1000])
        assert_that(output).has_no_nan()
        assert_that(output.argmax()).equals(sample.expected_class)
```

**UC-MODEL-02: 评估模型精度**

```python
@accuracy_test(
    model="models/bert.pt",
    dataset="data/validation.json",
    batch_size=32
)
def test_classification_accuracy(evaluator):
    metrics = evaluator.evaluate()

    assert_accuracy(metrics.accuracy).greater_than(0.90)
    assert_accuracy(metrics.f1_score).greater_than(0.88)
```

**UC-MODEL-03: 性能基准测试**

```python
@performance_test(
    model="models/yolo.pt",
    warmup_iterations=50,
    test_iterations=1000
)
def test_latency(perf_tester):
    latency = perf_tester.measure_latency()

    assert_latency(latency.p99).less_than_ms(20)
    assert_throughput(latency.qps).greater_than(100)
```

### 5.2 需求追溯

| 需求ID | 实现类/方法 | 测试用例 |
|--------|-------------|----------|
| MODEL-001 | `PyTorchLoader`, `TensorFlowLoader`, `ONNXLoader` | test_model_loading |
| MODEL-002 | `ModelRegistry`, `ModelVersion` | test_model_versioning |
| MODEL-003 | `InferenceValidator` | test_inference_correctness |
| MODEL-004 | `AccuracyEvaluator` | test_accuracy_evaluation |
| MODEL-005 | `PerformanceTester` | test_performance |
| MODEL-006 | `StressTester` | test_stress |
| MODEL-007 | `RobustnessTester` | test_robustness |
| MODEL-008 | `ConsistencyTester` | test_consistency |
| MODEL-009 | `LLMTester` | test_llm |

---

*本文档为AI模型测试模块的详细设计，基于4+1视图方法。*
