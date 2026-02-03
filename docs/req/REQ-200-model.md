# REQ-MDL 参考模型需求

---
id: REQ-MDL
title: 参考模型需求
priority: P0
status: draft
parent: REQ-SYS
---

## 概述

本模块负责golden数据与用例关系的管理，以及结果比较工具。

**本模块职责：**
- Golden数据文件格式定义
- Golden数据与测试用例的关联管理
- 结果比较工具

**不在本模块范围：**
- 参考模型实现（由外部库提供）
- Golden数据生成（由外部工具完成）

---

## REQ-MDL-001 Golden数据管理

---
id: REQ-MDL-001
title: Golden数据管理
priority: P0
status: draft
parent: REQ-MDL
---

### 描述

管理测试用的golden数据与用例的关联关系。

### 目录结构

```
testdata/
├── inputs/           # 输入数据
│   └── matmul/
│       ├── case_001_A.bin
│       └── case_001_B.bin
└── golden/           # 期望输出
    └── matmul/
        └── case_001_C.bin
```

### 文件格式

```c
/* 二进制格式头 */
#define GOLDEN_MAX_DIMS  8

typedef struct GoldenHeaderStru {
    UINT32 magic;                   /* 0x474F4C44 "GOLD" */
    UINT32 version;                 /* 格式版本 */
    UINT32 dtype;                   /* 数据类型 */
    UINT32 ndim;                    /* 维度数 */
    UINT32 shape[GOLDEN_MAX_DIMS];  /* 各维度大小 */
    UINT32 dataOffset;              /* 数据偏移 */
} GoldenHeaderStru;
```

### 版本管理

```c
/* 版本定义 */
#define GOLDEN_VERSION_1  1  /* 初始版本 */
#define GOLDEN_VERSION_2  2  /* 增加checksum字段 */
#define GOLDEN_CURRENT    GOLDEN_VERSION_2
```

### 验收标准

1. 统一的二进制格式，C可读写
2. 包含元信息（magic, version, shape, dtype）
3. header.version字段标识格式版本，支持向后兼容

---

## REQ-MDL-002 比较工具

---
id: REQ-MDL-002
title: 结果比较工具
priority: P0
status: draft
parent: REQ-MDL
---

### 描述

提供多种比较方法验证结果正确性。

### 比较方法

| 方法 | 用途 | 参数 |
|------|------|------|
| exact_match | 整数精确匹配 | - |
| float_near | 浮点容差 | epsilon |
| relative_error | 相对误差 | threshold |
| cosine_similarity | 余弦相似度 | min_similarity |
| mse | 均方误差 | max_mse |

### 精度要求

| 数据类型 | 绝对误差容限 (eps) | 相对误差容限 | 说明 |
|----------|-------------------|-------------|------|
| FP32 | 1e-5 | 1e-4 | 单精度浮点 |
| FP16 | 1e-3 | 1e-2 | 半精度浮点 |
| BF16 | 1e-2 | 5e-2 | Brain Float 16 |
| INT8 | 1 | - | 整数允许±1 |

比较公式：`|actual - expected| < eps + rel_eps * |expected|`

### C接口

```c
typedef struct CompareResultStru {
    bool     passed;
    FLOAT32  maxError;
    FLOAT32  avgError;
    FLOAT32  cosineSim;
    INT32    mismatchCount;
    INT32    firstMismatchIdx;
} CompareResultStru;

SEC_DDR_TEXT ERRNO_T CMP_FloatNear(const FLOAT32 *expected, const FLOAT32 *actual,
                                    INT32 count, FLOAT32 epsilon,
                                    CompareResultStru *result);

SEC_DDR_TEXT ERRNO_T CMP_CosineSim(const FLOAT32 *expected, const FLOAT32 *actual,
                                    INT32 count, FLOAT32 threshold,
                                    CompareResultStru *result);
```

### 验收标准

1. 比较结果包含详细统计信息
2. 支持配置不同算子使用不同比较方法
