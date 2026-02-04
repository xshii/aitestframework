# REQ-MDL 参考模型需求

---
id: REQ-MDL
title: 参考模型需求
priority: P0
status: draft
parent: REQ-SYS
---

## 概述

本模块负责golden数据管理和结果比较工具。

**本模块职责：**
- Golden数据文件格式定义和读写接口
- Golden数据与测试用例的关联管理
- 结果比较工具（精度校验）

**参考模型说明：**
- **接口定义**：本框架在REQ-SYS-005中定义参考模型的标准接口规范
- **Python参考模型**：由`pymodel/`目录提供，用于生成Golden数据
- **C参考模型**：可选，用于LinuxUT/ST环境的快速比较，实现在`src/model/`
- **外部参考模型**：复杂算子可使用外部库（如PyTorch/NumPy），通过Python接口调用

**不在本模块范围：**
- 外部第三方库的参考模型实现

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

/* V1格式（初始版本） */
typedef struct GoldenHeaderV1Stru {
    UINT32 magic;                   /* 0x474F4C44 "GOLD" */
    UINT32 version;                 /* 格式版本 = 1 */
    UINT32 dtype;                   /* 数据类型 */
    UINT32 ndim;                    /* 维度数 */
    UINT32 shape[GOLDEN_MAX_DIMS];  /* 各维度大小 */
    UINT32 dataOffset;              /* 数据偏移 */
} GoldenHeaderV1Stru;

/* V2格式（当前版本，增加checksum） */
typedef struct GoldenHeaderStru {
    UINT32 magic;                   /* 0x474F4C44 "GOLD" */
    UINT32 version;                 /* 格式版本 = 2 */
    UINT32 dtype;                   /* 数据类型 */
    UINT32 ndim;                    /* 维度数 */
    UINT32 shape[GOLDEN_MAX_DIMS];  /* 各维度大小 */
    UINT32 dataOffset;              /* 数据偏移 */
    UINT32 dataSize;                /* 数据字节数 */
    UINT32 checksum;                /* CRC32校验和 */
    UINT32 reserved[2];             /* 保留字段 */
} GoldenHeaderStru;
```

### 版本管理

```c
/* 版本定义 */
#define GOLDEN_VERSION_1  1  /* 初始版本，无checksum */
#define GOLDEN_VERSION_2  2  /* 增加checksum、dataSize字段 */
#define GOLDEN_CURRENT    GOLDEN_VERSION_2

/* 版本兼容性：读取时自动识别版本 */
SEC_DDR_TEXT ERRNO_T GOLDEN_ReadHeader(const CHAR *path, GoldenHeaderStru *header);
SEC_DDR_TEXT ERRNO_T GOLDEN_VerifyChecksum(const CHAR *path);
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

---

## REQ-MDL-003 测试数据版本管理

---
id: REQ-MDL-003
title: 测试数据版本管理
priority: P2
status: draft
parent: REQ-MDL
---

### 描述

管理测试数据（输入、Golden）与被测代码版本的绑定关系。

### 版本绑定机制

```yaml
# testdata/manifest.yaml
version: "1.0"
generated_at: "2026-02-03T10:00:00Z"

# 生成环境
generator:
  pymodel_version: "1.2.0"
  numpy_version: "1.24.0"
  commit: "abc123def"

# 数据集
datasets:
  matmul:
    version: "2.0"
    min_framework_version: "1.0.0"
    files:
      - path: inputs/matmul/*.bin
        count: 50
        checksum: "sha256:abcd..."
      - path: golden/matmul/*.bin
        count: 50
        checksum: "sha256:efgh..."

  conv2d:
    version: "1.5"
    min_framework_version: "1.0.0"
    files:
      - path: inputs/conv2d/*.bin
        count: 30
        checksum: "sha256:ijkl..."
```

### 版本检查

```bash
# 检查数据版本兼容性
python -m tools.data.check_version --manifest testdata/manifest.yaml

# 输出
[OK] matmul v2.0: compatible with framework v1.0.0
[WARN] conv2d v1.5: generated with old pymodel (1.1.0 < 1.2.0)
[ERROR] activation v1.0: requires framework >= 1.1.0 (current: 1.0.0)
```

### 数据更新流程

```
1. 被测代码更新 → 2. 重新生成Golden → 3. 更新manifest → 4. 提交数据
```

### 验收标准

1. manifest.yaml记录数据生成环境
2. 运行时检查数据版本兼容性
3. 不兼容时给出明确提示
4. 支持数据完整性校验（checksum）
