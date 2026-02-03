# 数据管理模块详细设计 (Data Management)

## 模块概述

| 属性 | 值 |
|------|-----|
| **模块ID** | DATA |
| **模块名称** | 数据管理 |
| **职责** | 测试数据的加载、生成、存储和管理 |
| **需求覆盖** | DATA-001 ~ DATA-009 |

---

## 1. 逻辑视图

### 1.1 模块类图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Data Management Classes                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        Data Loader Hierarchy                          │  │
│  │                                                                       │  │
│  │              ┌─────────────────────────────┐                          │  │
│  │              │     <<interface>>           │                          │  │
│  │              │      IDataLoader            │                          │  │
│  │              ├─────────────────────────────┤                          │  │
│  │              │ + load(source) -> Dataset   │                          │  │
│  │              │ + stream(source) -> Iterator│                          │  │
│  │              │ + supports(format) -> bool  │                          │  │
│  │              └──────────────┬──────────────┘                          │  │
│  │                             │                                         │  │
│  │       ┌─────────────────────┼─────────────────────┐                   │  │
│  │       │          │          │          │          │                   │  │
│  │       ▼          ▼          ▼          ▼          ▼                   │  │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐              │  │
│  │  │  CSV   │ │  JSON  │ │ Image  │ │  Text  │ │ Remote │              │  │
│  │  │ Loader │ │ Loader │ │ Loader │ │ Loader │ │ Loader │              │  │
│  │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘              │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        Transform Pipeline                             │  │
│  │                                                                       │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐│  │
│  │  │                      DataPipeline                                ││  │
│  │  ├──────────────────────────────────────────────────────────────────┤│  │
│  │  │ - steps: List[ITransform]                                        ││  │
│  │  │ - cache_enabled: bool                                            ││  │
│  │  ├──────────────────────────────────────────────────────────────────┤│  │
│  │  │ + add_step(transform)                                            ││  │
│  │  │ + process(data) -> ProcessedData                                 ││  │
│  │  │ + fit(data)                                                      ││  │
│  │  │ + fit_transform(data)                                            ││  │
│  │  └──────────────────────────────────────────────────────────────────┘│  │
│  │                                                                       │  │
│  │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐          │  │
│  │  │  Normalizer     │ │   Augmenter     │ │   Tokenizer     │          │  │
│  │  ├─────────────────┤ ├─────────────────┤ ├─────────────────┤          │  │
│  │  │ - method        │ │ - transforms    │ │ - vocab         │          │  │
│  │  │ - params        │ │ - probability   │ │ - max_length    │          │  │
│  │  └─────────────────┘ └─────────────────┘ └─────────────────┘          │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        Data Generation                                │  │
│  │                                                                       │  │
│  │  ┌──────────────────────┐    ┌──────────────────────┐                 │  │
│  │  │  SyntheticGenerator  │    │     DataSampler      │                 │  │
│  │  ├──────────────────────┤    ├──────────────────────┤                 │  │
│  │  │ - generators: Dict   │    │ - strategy: Strategy │                 │  │
│  │  │ - templates: List    │    │ - seed: int          │                 │  │
│  │  ├──────────────────────┤    ├──────────────────────┤                 │  │
│  │  │ + generate_random()  │    │ + sample(n) -> Data  │                 │  │
│  │  │ + generate_boundary()│    │ + stratified_sample()│                 │  │
│  │  │ + from_template()    │    │ + weighted_sample()  │                 │  │
│  │  └──────────────────────┘    └──────────────────────┘                 │  │
│  │                                                                       │  │
│  │  ┌──────────────────────┐    ┌──────────────────────┐                 │  │
│  │  │   DatasetRegistry    │    │    GoldenDataset     │                 │  │
│  │  ├──────────────────────┤    ├──────────────────────┤                 │  │
│  │  │ - datasets: Dict     │    │ - inputs: List       │                 │  │
│  │  │ - versions: Dict     │    │ - expected: List     │                 │  │
│  │  ├──────────────────────┤    │ - version: str       │                 │  │
│  │  │ + register(name, ds) │    ├──────────────────────┤                 │  │
│  │  │ + get(name, version) │    │ + load(path)         │                 │  │
│  │  │ + list_datasets()    │    │ + compare(preds)     │                 │  │
│  │  └──────────────────────┘    └──────────────────────┘                 │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 数据模型

```python
# data/models.py

from dataclasses import dataclass, field
from typing import List, Dict, Any, Iterator, Optional, Union
from pathlib import Path
import numpy as np

@dataclass
class Dataset:
    """数据集"""
    name: str
    data: Union[np.ndarray, List[Any]]
    labels: Optional[Union[np.ndarray, List[Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> 'DataItem':
        label = self.labels[idx] if self.labels is not None else None
        return DataItem(self.data[idx], label)

    def batches(self, batch_size: int) -> Iterator['Batch']:
        """生成批次"""
        for i in range(0, len(self), batch_size):
            batch_data = self.data[i:i + batch_size]
            batch_labels = self.labels[i:i + batch_size] if self.labels else None
            yield Batch(batch_data, batch_labels)


@dataclass
class DataItem:
    """数据项"""
    input: Any
    label: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Batch:
    """数据批次"""
    inputs: np.ndarray
    labels: Optional[np.ndarray] = None


@dataclass
class GoldenData:
    """黄金数据"""
    input: Any
    expected_output: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransformConfig:
    """转换配置"""
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
```

---

## 2. 进程视图

### 2.1 数据加载流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Data Loading Pipeline                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌────────┐ │
│  │ Source  │───►│  Loader  │───►│Transform │───►│  Cache   │───►│Dataset │ │
│  │(File/URL)    │ (Format) │    │ Pipeline │    │ (Optional)    │        │ │
│  └─────────┘    └──────────┘    └──────────┘    └──────────┘    └────────┘ │
│                                                                             │
│  Pipeline Detail:                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                                                                     │    │
│  │  [Raw Data] ──► [Normalize] ──► [Augment] ──► [Tokenize] ──►       │    │
│  │                                                                     │    │
│  │  ──► [Batch] ──► [ToTensor] ──► [ToDevice] ──► [Ready]             │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Streaming Mode:                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                                                                     │    │
│  │  Source ──► [Read Chunk] ──► [Transform] ──► yield ──► [Next Chunk]│    │
│  │     ▲                                                      │        │    │
│  │     └──────────────────────────────────────────────────────┘        │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 并行预处理

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Parallel Preprocessing                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      DataLoader with Workers                          │  │
│  │                                                                       │  │
│  │   Main Thread                                                         │  │
│  │   ┌────────────────────────────────────────────────────────────────┐ │  │
│  │   │  [Prefetch Queue] ◄──────────────────────────────────────────┐ │ │  │
│  │   │                                                               │ │ │  │
│  │   │  for batch in prefetch_queue:                                 │ │ │  │
│  │   │      yield batch                                              │ │ │  │
│  │   └────────────────────────────────────────────────────────────────┘ │ │  │
│  │                                                                       │ │  │
│  │   Worker Threads (num_workers=4)                                      │ │  │
│  │   ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐        │ │  │
│  │   │  Worker 1  │ │  Worker 2  │ │  Worker 3  │ │  Worker 4  │        │ │  │
│  │   │  [Load]    │ │  [Load]    │ │  [Load]    │ │  [Load]    │        │ │  │
│  │   │  [Transform] │ [Transform]│ │  [Transform] │ [Transform]│        │ │  │
│  │   │  [Push]────┼─┼──[Push]────┼─┼──[Push]────┼─┼──[Push]────┼────────┘ │  │
│  │   └────────────┘ └────────────┘ └────────────┘ └────────────┘          │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 开发视图

### 3.1 包结构

```
aitest/data/
├── __init__.py
├── loader/
│   ├── __init__.py
│   ├── base.py              # IDataLoader接口
│   ├── csv.py               # CSV加载器
│   ├── json.py              # JSON加载器
│   ├── parquet.py           # Parquet加载器
│   ├── image.py             # 图像加载器
│   ├── text.py              # 文本加载器
│   ├── remote.py            # 远程数据加载
│   └── factory.py           # 加载器工厂
├── transform/
│   ├── __init__.py
│   ├── base.py              # ITransform接口
│   ├── normalize.py         # 归一化
│   ├── augment.py           # 数据增强
│   ├── image_transforms.py  # 图像变换
│   ├── text_transforms.py   # 文本变换
│   └── tokenize.py          # 分词
├── pipeline.py              # DataPipeline
├── sampler.py               # 采样器
├── generator.py             # 合成数据生成
├── registry.py              # 数据集注册
├── golden.py                # 黄金数据集
├── validator.py             # 数据验证
├── cache.py                 # 数据缓存
└── models.py                # 数据模型
```

### 3.2 实现示例

```python
# data/pipeline.py

from typing import List, Any, Optional
import logging

from .transform.base import ITransform

logger = logging.getLogger(__name__)


class DataPipeline:
    """数据预处理Pipeline"""

    def __init__(self):
        self._steps: List[ITransform] = []
        self._fitted = False
        self._cache = {}
        self.cache_enabled = False

    def add_step(self, transform: ITransform) -> 'DataPipeline':
        """添加预处理步骤"""
        self._steps.append(transform)
        self._fitted = False
        return self

    def fit(self, data: Any) -> 'DataPipeline':
        """拟合参数"""
        current = data
        for step in self._steps:
            if hasattr(step, 'fit'):
                step.fit(current)
            current = step.transform(current)
        self._fitted = True
        return self

    def transform(self, data: Any) -> Any:
        """执行转换"""
        cache_key = self._get_cache_key(data)
        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        current = data
        for step in self._steps:
            current = step.transform(current)

        if self.cache_enabled:
            self._cache[cache_key] = current

        return current

    def fit_transform(self, data: Any) -> Any:
        """拟合并转换"""
        self.fit(data)
        return self.transform(data)

    def _get_cache_key(self, data: Any) -> str:
        """生成缓存键"""
        return str(hash(str(data)))


# data/transform/normalize.py

import numpy as np
from typing import Optional

from .base import ITransform


class MinMaxNormalizer(ITransform):
    """最小-最大归一化"""

    def __init__(self, feature_range: tuple = (0, 1)):
        self.feature_range = feature_range
        self.min_val: Optional[float] = None
        self.max_val: Optional[float] = None

    def fit(self, data: np.ndarray) -> 'MinMaxNormalizer':
        self.min_val = data.min()
        self.max_val = data.max()
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.min_val is None or self.max_val is None:
            raise ValueError("Normalizer not fitted")

        scale = self.max_val - self.min_val
        if scale == 0:
            return np.zeros_like(data)

        normalized = (data - self.min_val) / scale
        a, b = self.feature_range
        return normalized * (b - a) + a

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        a, b = self.feature_range
        normalized = (data - a) / (b - a)
        return normalized * (self.max_val - self.min_val) + self.min_val


class ZScoreNormalizer(ITransform):
    """Z-Score标准化"""

    def __init__(self):
        self.mean: Optional[float] = None
        self.std: Optional[float] = None

    def fit(self, data: np.ndarray) -> 'ZScoreNormalizer':
        self.mean = data.mean()
        self.std = data.std()
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.std == 0:
            return np.zeros_like(data)
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean
```

---

## 4. 场景视图

### 4.1 核心用例

**UC-DATA-01: 加载数据集**

```python
from aitest.data import load_dataset

# 加载CSV数据
dataset = load_dataset("data/train.csv", format="csv")

# 加载图像数据
images = load_dataset("data/images/", format="image")

# 流式加载大数据
for batch in load_dataset("data/large.parquet", streaming=True, batch_size=1000):
    process(batch)
```

**UC-DATA-02: 数据预处理Pipeline**

```python
from aitest.data import DataPipeline
from aitest.data.transform import MinMaxNormalizer, ImageAugmenter

pipeline = DataPipeline()
pipeline.add_step(MinMaxNormalizer())
pipeline.add_step(ImageAugmenter(rotate=True, flip=True))

processed_data = pipeline.fit_transform(raw_data)
```

**UC-DATA-03: 黄金数据集管理**

```python
from aitest.data import GoldenDataset

golden = GoldenDataset.load("golden/classification_v1.json")

for item in golden:
    prediction = model.predict(item.input)
    assert prediction == item.expected_output
```

### 4.2 需求追溯

| 需求ID | 实现类/方法 | 测试用例 |
|--------|-------------|----------|
| DATA-001 | `CSVLoader`, `JSONLoader`, `ImageLoader` | test_data_loading |
| DATA-002 | `DatasetRegistry` | test_registry |
| DATA-003 | `SyntheticGenerator` | test_generation |
| DATA-004 | `ImageAugmenter`, `TextAugmenter` | test_augmentation |
| DATA-005 | `DataPipeline` | test_pipeline |
| DATA-006 | `MinMaxNormalizer`, `ZScoreNormalizer` | test_normalization |
| DATA-007 | `DataSampler` | test_sampling |
| DATA-008 | `GoldenDataset` | test_golden_data |
| DATA-009 | `DataValidator` | test_validation |

---

*本文档为数据管理模块的详细设计，基于4+1视图方法。*
