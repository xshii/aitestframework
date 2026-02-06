# MDL 参考模型模块设计

---
module: MDL
version: 1.0
date: 2026-02-04
status: draft
requirements: REQ-MDL-001, REQ-MDL-002
domain: Core
aggregate: GoldenDataset
---

## 1. 模块概述

### 1.1 职责

管理测试参考数据：
- Golden数据格式定义与管理
- 结果比较算法
- 数据加载与校验

### 1.2 DDD定位

- **限界上下文**：参考数据上下文
- **聚合根**：GoldenDataset
- **发布事件**：GoldenUpdated

### 1.3 双实现策略

| 实现 | 语言 | 用途 |
|------|------|------|
| Python参考模型 | Python+numpy | 生成Golden数据、离线验证 |
| C比较工具 | C | 嵌入式在线比对 |

---

## 2. 文件结构

```
include/model/
├── golden.h       # Golden数据格式
└── compare.h      # 比较接口

src/model/
├── golden.c       # Golden加载实现
└── compare.c      # 比较算法实现

testdata/
├── inputs/        # 输入数据
│   └── matmul/
│       └── case001_input.bin
└── golden/        # Golden数据
    └── matmul/
        └── case001_golden.bin
```

---

## 3. 聚合设计

### 3.1 GoldenDataset聚合

```
GoldenDataset (聚合根)
├── name: STRING               # 数据集名称（如 "matmul"）
├── version: STRING            # 版本
├── basePath: STRING           # 基础路径
└── files: GoldenFile[]        # Golden文件列表（实体）
    ├── path: STRING
    ├── checksum: STRING
    └── header: GoldenHeader   # 值对象
        ├── dtype: GoldenDtypeEnum
        ├── ndim: UINT32
        └── shape[8]: UINT32
```

---

## 4. Golden数据格式

### 4.1 文件结构

```
┌────────────────────────────────────┐
│  GoldenHeaderStru (64 bytes)       │
├────────────────────────────────────┤
│  Padding (对齐到64字节)             │
├────────────────────────────────────┤
│  Raw Data                          │
│  (dtype × shape[0] × ... × shape[n])│
└────────────────────────────────────┘
```

### 4.2 GoldenHeaderStru

```c
#define GOLDEN_MAGIC     0x444C4F47  /* "GOLD" */
#define GOLDEN_VERSION   1
#define GOLDEN_MAX_NDIM  8

typedef enum GoldenDtypeEnum {
    GOLDEN_DTYPE_FLOAT32 = 0,
    GOLDEN_DTYPE_FLOAT64 = 1,
    GOLDEN_DTYPE_INT8    = 2,
    GOLDEN_DTYPE_INT16   = 3,
    GOLDEN_DTYPE_INT32   = 4,
    GOLDEN_DTYPE_UINT8   = 5,
    GOLDEN_DTYPE_UINT16  = 6,
    GOLDEN_DTYPE_UINT32  = 7,
} GoldenDtypeEnum;

typedef struct GoldenHeaderStru {
    UINT32 magic;                      /* 魔数 0x444C4F47 */
    UINT32 version;                    /* 格式版本 */
    UINT32 dtype;                      /* 数据类型 */
    UINT32 ndim;                       /* 维度数 */
    UINT32 shape[GOLDEN_MAX_NDIM];     /* 各维度大小 */
    UINT32 dataOffset;                 /* 数据起始偏移 */
    UINT32 checksum;                   /* CRC32校验（0表示不校验） */
    UINT32 reserved[2];                /* 保留 */
} GoldenHeaderStru;  /* 64 bytes */
```

### 4.3 数据类型大小

```c
static inline UINT32 Golden_GetDtypeSize(GoldenDtypeEnum dtype)
{
    switch (dtype) {
        case GOLDEN_DTYPE_INT8:
        case GOLDEN_DTYPE_UINT8:   return 1;
        case GOLDEN_DTYPE_INT16:
        case GOLDEN_DTYPE_UINT16:  return 2;
        case GOLDEN_DTYPE_INT32:
        case GOLDEN_DTYPE_UINT32:
        case GOLDEN_DTYPE_FLOAT32: return 4;
        case GOLDEN_DTYPE_FLOAT64: return 8;
        default:                   return 0;
    }
}
```

---

## 5. Golden数据接口

### 5.1 加载接口

```c
/**
 * @brief 加载Golden数据
 * @param path 文件路径
 * @param header 输出文件头
 * @param data 输出数据指针（需调用Golden_Free释放）
 * @return AITF_OK成功
 */
ERRNO_T Golden_Load(const CHAR *path,
                    GoldenHeaderStru *header,
                    VOID **data);

/**
 * @brief 释放Golden数据
 */
VOID Golden_Free(VOID *data);

/**
 * @brief 计算元素总数
 */
UINT64 Golden_GetElementCount(const GoldenHeaderStru *header);

/**
 * @brief 计算数据总字节数
 */
UINT64 Golden_GetDataSize(const GoldenHeaderStru *header);
```

### 5.2 加载实现

```c
ERRNO_T Golden_Load(const CHAR *path,
                    GoldenHeaderStru *header,
                    VOID **data)
{
    RET_IF_NULL(path);
    RET_IF_NULL(header);
    RET_IF_NULL(data);

    FILE *fp = fopen(path, "rb");
    if (fp == NULL) {
        return AITF_ERR_GOLDEN_NOT_FOUND;
    }

    /* 读取头部 */
    if (fread(header, sizeof(GoldenHeaderStru), 1, fp) != 1) {
        fclose(fp);
        return AITF_ERR_IO;
    }

    /* 校验魔数 */
    if (header->magic != GOLDEN_MAGIC) {
        fclose(fp);
        return AITF_ERR_GOLDEN_FORMAT;
    }

    /* 计算数据大小 */
    UINT64 dataSize = Golden_GetDataSize(header);

    /* 分配内存 */
    *data = malloc(dataSize);
    if (*data == NULL) {
        fclose(fp);
        return AITF_ERR_OUT_OF_MEMORY;
    }

    /* 定位到数据起始位置 */
    fseek(fp, header->dataOffset, SEEK_SET);

    /* 读取数据 */
    if (fread(*data, 1, dataSize, fp) != dataSize) {
        free(*data);
        *data = NULL;
        fclose(fp);
        return AITF_ERR_IO;
    }

    fclose(fp);
    return AITF_OK;
}
```

---

## 6. 比较接口设计

### 6.1 比较结果

```c
typedef struct CompareResultStru {
    BOOL    match;               /* 是否匹配 */
    UINT64  totalCount;          /* 总元素数 */
    UINT64  mismatchCount;       /* 不匹配数 */
    UINT64  firstMismatchIdx;    /* 第一个不匹配的索引 */
    FLOAT64 maxAbsDiff;          /* 最大绝对误差 */
    FLOAT64 maxRelDiff;          /* 最大相对误差 */
    FLOAT64 cosineSim;           /* 余弦相似度 */
} CompareResultStru;
```

### 6.2 比较配置

```c
typedef struct CompareConfigStru {
    FLOAT64 absTol;              /* 绝对容差 */
    FLOAT64 relTol;              /* 相对容差 */
    FLOAT64 cosineThreshold;     /* 余弦相似度阈值 */
    BOOL    checkAll;            /* 是否检查所有元素 */
} CompareConfigStru;

#define COMPARE_CONFIG_DEFAULT { \
    .absTol = 1e-5, \
    .relTol = 1e-4, \
    .cosineThreshold = 0.999, \
    .checkAll = FALSE \
}
```

### 6.3 比较函数

```c
/**
 * @brief 比较FLOAT32数组
 */
ERRNO_T Compare_Float32(const FLOAT32 *expected,
                        const FLOAT32 *actual,
                        UINT64 count,
                        const CompareConfigStru *config,
                        CompareResultStru *result);

/**
 * @brief 比较FLOAT64数组
 */
ERRNO_T Compare_Float64(const FLOAT64 *expected,
                        const FLOAT64 *actual,
                        UINT64 count,
                        const CompareConfigStru *config,
                        CompareResultStru *result);

/**
 * @brief 比较INT32数组（精确匹配）
 */
ERRNO_T Compare_Int32(const INT32 *expected,
                      const INT32 *actual,
                      UINT64 count,
                      CompareResultStru *result);

/**
 * @brief 比较UINT8数组（精确匹配）
 */
ERRNO_T Compare_Uint8(const UINT8 *expected,
                      const UINT8 *actual,
                      UINT64 count,
                      CompareResultStru *result);
```

---

## 7. 比较算法

### 7.1 浮点比较策略

```c
/*
 * 匹配条件（满足任一即可）：
 * 1. 绝对误差: |expected - actual| < absTol
 * 2. 相对误差: |expected - actual| / max(|expected|, epsilon) < relTol
 *
 * 特殊值处理：
 * - NaN: 两者都是NaN时视为匹配
 * - Inf: 符号相同的Inf视为匹配
 */

static inline BOOL IsFloatMatch(FLOAT64 expected, FLOAT64 actual,
                                FLOAT64 absTol, FLOAT64 relTol)
{
    /* NaN处理 */
    if (isnan(expected) && isnan(actual)) {
        return TRUE;
    }
    if (isnan(expected) || isnan(actual)) {
        return FALSE;
    }

    /* Inf处理 */
    if (isinf(expected) && isinf(actual)) {
        return (expected > 0) == (actual > 0);
    }

    /* 绝对误差 */
    FLOAT64 absDiff = fabs(expected - actual);
    if (absDiff < absTol) {
        return TRUE;
    }

    /* 相对误差 */
    FLOAT64 base = fabs(expected);
    if (base < 1e-10) {
        base = 1e-10;
    }
    if (absDiff / base < relTol) {
        return TRUE;
    }

    return FALSE;
}
```

### 7.2 余弦相似度

```c
/*
 * cos(theta) = (A · B) / (||A|| × ||B||)
 *
 * 其中:
 *   A · B = sum(a[i] × b[i])
 *   ||A|| = sqrt(sum(a[i]^2))
 */

static FLOAT64 ComputeCosineSimilarity(const FLOAT32 *a,
                                       const FLOAT32 *b,
                                       UINT64 count)
{
    FLOAT64 dotProduct = 0.0;
    FLOAT64 normA = 0.0;
    FLOAT64 normB = 0.0;

    for (UINT64 i = 0; i < count; i++) {
        dotProduct += (FLOAT64)a[i] * b[i];
        normA += (FLOAT64)a[i] * a[i];
        normB += (FLOAT64)b[i] * b[i];
    }

    FLOAT64 denom = sqrt(normA) * sqrt(normB);
    if (denom < 1e-10) {
        return (normA < 1e-10 && normB < 1e-10) ? 1.0 : 0.0;
    }

    return dotProduct / denom;
}
```

---

## 8. 领域事件

```yaml
GoldenUpdated:
  source: MDL
  payload:
    dataset: STRING      # 数据集名称
    version: STRING      # 版本
    checksum: STRING     # 校验和
```

---

## 9. 使用示例

```c
#include "model/golden.h"
#include "model/compare.h"

TEST_CASE(matmul, basic_2x2)
{
    GoldenHeaderStru header;
    VOID *goldenData = NULL;
    FLOAT32 actualResult[4];
    CompareResultStru cmpResult;
    CompareConfigStru config = COMPARE_CONFIG_DEFAULT;

    /* 加载Golden数据 */
    RET_IF_ERR(Golden_Load("testdata/golden/matmul/case001.bin",
                           &header, &goldenData));

    /* 执行被测操作，结果存入actualResult */
    /* ... */

    /* 比较结果 */
    ERRNO_T err = Compare_Float32(
        (FLOAT32 *)goldenData,
        actualResult,
        Golden_GetElementCount(&header),
        &config,
        &cmpResult
    );

    Golden_Free(goldenData);

    TEST_ASSERT_OK(err);
    TEST_ASSERT_TRUE(cmpResult.match);
    TEST_ASSERT_GE(cmpResult.cosineSim, 0.999);

    return AITF_OK;
}
```

---

## 10. Python Golden生成

```python
# tools/data/generate_golden.py
import numpy as np
import struct

def save_golden(path: str, data: np.ndarray):
    """保存Golden数据文件"""
    dtype_map = {
        np.float32: 0,
        np.float64: 1,
        np.int8: 2,
        np.int32: 4,
        np.uint8: 5,
    }

    header = struct.pack(
        '<IIII8IIII',
        0x444C4F47,           # magic
        1,                     # version
        dtype_map[data.dtype.type],  # dtype
        data.ndim,             # ndim
        *data.shape,           # shape (pad to 8)
        *([0] * (8 - data.ndim)),
        64,                    # dataOffset
        0,                     # checksum
        0, 0                   # reserved
    )

    with open(path, 'wb') as f:
        f.write(header)
        f.write(data.tobytes())
```

---

## 11. 需求追溯

| 需求ID | 需求标题 | 设计章节 |
|--------|----------|----------|
| REQ-MDL-001 | Golden数据管理 | 4, 5 |
| REQ-MDL-002 | 结果比较工具 | 6, 7 |
