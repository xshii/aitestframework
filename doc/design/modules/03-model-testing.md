# AI模型测试模块详细设计 (Model Testing)

## 模块概述

| 属性 | 值 |
|------|-----|
| **模块ID** | MODEL |
| **模块名称** | AI模型测试 |
| **职责** | 针对AI/ML模型的专项测试能力 |
| **需求覆盖** | MODEL-001 ~ MODEL-009 |
| **外部依赖** | aidevtools.tools.compare (算子精度比对), aidevtools.golden (Golden生成) |

### 模块定位

AI模型测试模块作为测试框架的核心模块，集成 `aidevtools` 提供的算子验证能力，实现：
- **模型加载**：支持 PyTorch、TensorFlow、ONNX 等多框架模型
- **推理验证**：单样本/批量推理正确性验证
- **算子级验证**：集成三列比对机制，验证每个算子的精度
- **精度评估**：分类、回归、检测等任务精度指标
- **性能测试**：延迟、吞吐量、内存占用测试
- **鲁棒性测试**：边界值、异常输入、噪声注入测试

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AI Model Testing Module                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      Test Framework Layer                            │    │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐        │    │
│  │  │  Model    │  │ Inference │  │ Operator  │  │ Accuracy  │        │    │
│  │  │  Loader   │  │ Validator │  │ Verifier  │  │ Evaluator │        │    │
│  │  └───────────┘  └───────────┘  └───────────┘  └───────────┘        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    aidevtools (算子验证)                              │    │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐        │    │
│  │  │  Golden   │  │ compare_  │  │ compare_  │  │  Formats  │        │    │
│  │  │ Generator │  │   3col    │  │  isclose  │  │ (BFP/GF)  │        │    │
│  │  └───────────┘  └───────────┘  └───────────┘  └───────────┘        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

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
│  │                                                                       │  │
│  │  ┌──────────────────────┐    ┌──────────────────────┐                 │  │
│  │  │  OperatorVerifier    │    │     GoldenManager    │                 │  │
│  │  │  (aidevtools集成)    │    │   (aidevtools集成)   │                 │  │
│  │  ├──────────────────────┤    ├──────────────────────┤                 │  │
│  │  │ - thresholds: Config │    │ - golden_dir: Path   │                 │  │
│  │  │ - precision_assert   │    │ - generator: Golden  │                 │  │
│  │  ├──────────────────────┤    ├──────────────────────┤                 │  │
│  │  │ + verify_op_3col()   │    │ + generate_golden()  │                 │  │
│  │  │ + verify_op_isclose()│    │ + load_golden()      │                 │  │
│  │  │ + verify_model_ops() │    │ + update_golden()    │                 │  │
│  │  │ + generate_report()  │    │ + compare_with_dut() │                 │  │
│  │  └──────────────────────┘    └──────────────────────┘                 │  │
│  │                                                                       │  │
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
│
├── operator/                    # 算子级验证 (aidevtools集成)
│   ├── __init__.py
│   ├── verifier.py              # OperatorVerifier 主类
│   ├── golden.py                # GoldenManager
│   ├── hooks.py                 # 推理 Hook (收集中间结果)
│   ├── report.py                # 验证报告生成
│   └── config.py                # 验证配置
│
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


# model/operator/verifier.py

"""
算子级验证模块 - 集成 aidevtools 的精度比对能力

功能:
1. 收集模型推理过程中每个算子的输入输出
2. 与 Golden 数据进行三列比对 (exact/fuzzy_pure/fuzzy_qnt)
3. 生成详细的算子精度验证报告
"""

from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np

# 从 aidevtools 导入比对函数
from aidevtools.tools.compare import (
    compare_3col,
    compare_isclose,
    CompareThresholds,
    FullCompareResult,
    IsCloseResult,
    print_compare_table,
)


@dataclass
class OpVerifyConfig:
    """算子验证配置"""
    # 比对阈值
    thresholds: CompareThresholds = None
    # Golden 路径
    golden_dir: Path = None
    # 是否允许量化问题
    allow_quant_issue: bool = True
    # 是否生成详细报告
    generate_report: bool = True

    def __post_init__(self):
        if self.thresholds is None:
            self.thresholds = CompareThresholds(
                exact_max_abs=0.0,
                exact_max_count=0,
                fuzzy_atol=1e-5,
                fuzzy_rtol=1e-3,
                fuzzy_min_qsnr=30.0,
                fuzzy_min_cosine=0.999,
            )


class OperatorVerifier:
    """算子级验证器"""

    def __init__(self, config: OpVerifyConfig = None):
        self.config = config or OpVerifyConfig()
        self.results: List[FullCompareResult] = []

    def verify_op(
        self,
        op_name: str,
        op_id: int,
        dut_output: np.ndarray,
        golden_pure: np.ndarray,
        golden_qnt: np.ndarray = None,
    ) -> FullCompareResult:
        """
        验证单个算子

        Args:
            op_name: 算子名称
            op_id: 算子 ID
            dut_output: DUT 输出
            golden_pure: 纯 fp32 Golden
            golden_qnt: 量化感知 Golden (可选，默认使用 golden_pure)

        Returns:
            FullCompareResult
        """
        if golden_qnt is None:
            golden_qnt = golden_pure

        result = compare_3col(
            op_name=op_name,
            op_id=op_id,
            result=dut_output,
            golden_pure=golden_pure,
            golden_qnt=golden_qnt,
            thresholds=self.config.thresholds,
        )

        self.results.append(result)
        return result

    def verify_isclose(
        self,
        name: str,
        dut_output: np.ndarray,
        golden: np.ndarray,
        atol: float = 1e-5,
        rtol: float = 1e-3,
        max_exceed_ratio: float = 0.0,
    ) -> IsCloseResult:
        """
        IsClose 验证

        Args:
            name: 验证名称
            dut_output: DUT 输出
            golden: Golden 数据
            atol: 绝对误差门限
            rtol: 相对误差门限
            max_exceed_ratio: 允许的最大超限比例
        """
        return compare_isclose(
            golden=golden,
            result=dut_output,
            atol=atol,
            rtol=rtol,
            max_exceed_ratio=max_exceed_ratio,
        )

    def verify_model_ops(
        self,
        op_outputs: Dict[str, np.ndarray],
        golden_dir: Path,
    ) -> List[FullCompareResult]:
        """
        验证模型所有算子

        Args:
            op_outputs: 算子输出字典 {op_name_id: output}
            golden_dir: Golden 数据目录

        Returns:
            List[FullCompareResult]
        """
        self.results = []

        for op_key, dut_output in op_outputs.items():
            # 解析算子名称和 ID
            parts = op_key.rsplit("_", 1)
            op_name = parts[0]
            op_id = int(parts[1]) if len(parts) > 1 else 0

            # 加载 Golden
            pure_path = golden_dir / f"{op_key}_pure.npy"
            qnt_path = golden_dir / f"{op_key}_qnt.npy"

            if not pure_path.exists():
                print(f"Warning: Golden not found for {op_key}")
                continue

            golden_pure = np.load(pure_path)
            golden_qnt = np.load(qnt_path) if qnt_path.exists() else golden_pure

            self.verify_op(op_name, op_id, dut_output, golden_pure, golden_qnt)

        return self.results

    def check_all_passed(self) -> bool:
        """检查所有算子是否通过验证"""
        for r in self.results:
            if r.status == "FAIL":
                return False
            if not self.config.allow_quant_issue and r.status == "QUANT_ISSUE":
                return False
        return True

    def print_summary(self):
        """打印汇总表格"""
        print_compare_table(self.results)

    def get_summary(self) -> Dict:
        """获取汇总统计"""
        status_counts = {"PERFECT": 0, "PASS": 0, "QUANT_ISSUE": 0, "FAIL": 0}
        for r in self.results:
            if r.status in status_counts:
                status_counts[r.status] += 1

        return {
            "total": len(self.results),
            "passed": status_counts["PERFECT"] + status_counts["PASS"],
            **status_counts,
        }


class GoldenManager:
    """Golden 数据管理器"""

    def __init__(self, golden_dir: Path):
        self.golden_dir = Path(golden_dir)
        self.golden_dir.mkdir(parents=True, exist_ok=True)

    def save_golden(
        self,
        op_name: str,
        op_id: int,
        pure_output: np.ndarray,
        qnt_output: np.ndarray = None,
    ):
        """保存 Golden 数据"""
        key = f"{op_name}_{op_id}"
        np.save(self.golden_dir / f"{key}_pure.npy", pure_output)
        if qnt_output is not None:
            np.save(self.golden_dir / f"{key}_qnt.npy", qnt_output)

    def load_golden(
        self,
        op_name: str,
        op_id: int,
    ) -> tuple:
        """加载 Golden 数据"""
        key = f"{op_name}_{op_id}"
        pure_path = self.golden_dir / f"{key}_pure.npy"
        qnt_path = self.golden_dir / f"{key}_qnt.npy"

        pure = np.load(pure_path) if pure_path.exists() else None
        qnt = np.load(qnt_path) if qnt_path.exists() else pure

        return pure, qnt

    def list_ops(self) -> List[str]:
        """列出所有已保存的算子"""
        ops = set()
        for f in self.golden_dir.glob("*_pure.npy"):
            op_key = f.stem.replace("_pure", "")
            ops.add(op_key)
        return sorted(ops)
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

**UC-MODEL-04: 算子级精度验证 (aidevtools集成)**

```python
"""算子级精度验证用例"""

from pathlib import Path
from aitest.model.operator import OperatorVerifier, GoldenManager, OpVerifyConfig
from aidevtools.tools.compare import CompareThresholds


class TestOperatorPrecision:
    """算子精度测试套件"""

    def setup(self):
        # 配置验证参数
        self.config = OpVerifyConfig(
            thresholds=CompareThresholds(
                fuzzy_atol=1e-5,
                fuzzy_rtol=1e-3,
                fuzzy_min_qsnr=30.0,
                fuzzy_min_cosine=0.999,
            ),
            golden_dir=Path("./golden_data"),
            allow_quant_issue=True,
        )
        self.verifier = OperatorVerifier(self.config)

    def test_single_op_precision(self):
        """测试单个算子精度"""
        import numpy as np

        # 加载 DUT 输出和 Golden
        dut_output = np.load("outputs/MatMul_0_dut.npy")
        golden_pure = np.load("golden_data/MatMul_0_pure.npy")
        golden_qnt = np.load("golden_data/MatMul_0_qnt.npy")

        # 三列比对
        result = self.verifier.verify_op(
            op_name="MatMul",
            op_id=0,
            dut_output=dut_output,
            golden_pure=golden_pure,
            golden_qnt=golden_qnt,
        )

        # 验证状态
        assert result.status in ("PERFECT", "PASS"), \
            f"MatMul_0 precision check failed: {result.status}"

        # 验证 QSNR
        assert result.fuzzy_qnt.qsnr >= 30.0, \
            f"QSNR too low: {result.fuzzy_qnt.qsnr:.1f} dB"

    def test_model_all_ops(self):
        """测试模型所有算子"""

        # 收集所有算子输出 (通过推理 Hook 或文件加载)
        op_outputs = {}
        for f in Path("outputs").glob("*_dut.npy"):
            op_key = f.stem.replace("_dut", "")
            op_outputs[op_key] = np.load(f)

        # 批量验证
        results = self.verifier.verify_model_ops(
            op_outputs=op_outputs,
            golden_dir=Path("golden_data"),
        )

        # 打印汇总表格
        self.verifier.print_summary()

        # 验证通过率
        summary = self.verifier.get_summary()
        assert summary["FAIL"] == 0, \
            f"{summary['FAIL']} operators failed precision check"

        print(f"\nPrecision Summary: {summary['passed']}/{summary['total']} passed")
        print(f"  PERFECT: {summary['PERFECT']}")
        print(f"  PASS: {summary['PASS']}")
        print(f"  QUANT_ISSUE: {summary['QUANT_ISSUE']}")


@model_test(model="models/transformer.pt", framework="pytorch")
def test_transformer_op_precision(model):
    """端到端算子精度验证"""

    # 1. 运行推理并收集算子输出
    from aitest.model.operator.hooks import register_op_hooks, collect_outputs

    hooks = register_op_hooks(model)
    input_data = torch.randn(1, 512, 768)
    output = model(input_data)
    op_outputs = collect_outputs(hooks)

    # 2. 验证每个算子
    verifier = OperatorVerifier()
    for op_name, dut_output in op_outputs.items():
        golden_pure, golden_qnt = GoldenManager("./golden").load_golden(op_name, 0)
        verifier.verify_op(op_name, 0, dut_output.numpy(), golden_pure, golden_qnt)

    # 3. 检查结果
    assert verifier.check_all_passed(), "Some operators failed precision check"
    verifier.print_summary()
```

### 5.2 需求追溯

| 需求ID | 实现类/方法 | 测试用例 | aidevtools 集成 |
|--------|-------------|----------|-----------------|
| MODEL-001 | `PyTorchLoader`, `TensorFlowLoader`, `ONNXLoader` | test_model_loading | - |
| MODEL-002 | `ModelRegistry`, `ModelVersion` | test_model_versioning | - |
| MODEL-003 | `InferenceValidator`, `OperatorVerifier` | test_inference_correctness | `compare_3col`, `compare_isclose` |
| MODEL-003-01 | `OperatorVerifier.verify_op` | test_single_op_precision | `compare_3col` |
| MODEL-003-03 | `OperatorVerifier.verify_model_ops` | test_model_all_ops | `print_compare_table` |
| MODEL-004 | `AccuracyEvaluator` | test_accuracy_evaluation | - |
| MODEL-005 | `PerformanceTester` | test_performance | - |
| MODEL-006 | `StressTester` | test_stress | - |
| MODEL-007 | `RobustnessTester` | test_robustness | - |
| MODEL-008 | `ConsistencyTester` | test_consistency | - |
| MODEL-009 | `LLMTester` | test_llm | - |

### 5.3 算子验证功能追溯

| 功能 | aidevtools 函数 | aitestframework 封装 | 说明 |
|------|-----------------|---------------------|------|
| 三列比对 | `compare_3col()` | `OperatorVerifier.verify_op()` | exact + fuzzy_pure + fuzzy_qnt |
| IsClose 验证 | `compare_isclose()` | `OperatorVerifier.verify_isclose()` | 逐元素误差检查 |
| 批量验证 | `print_compare_table()` | `OperatorVerifier.verify_model_ops()` | 模型所有算子验证 |
| Golden 管理 | - | `GoldenManager` | Golden 数据存储/加载 |
| 验证配置 | `CompareThresholds` | `OpVerifyConfig` | 比对阈值配置 |

---

*本文档为AI模型测试模块的详细设计，基于4+1视图方法，集成 aidevtools 提供的算子精度比对能力。*
