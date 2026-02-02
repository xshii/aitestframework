# REQ-200 参考模型需求

---
id: REQ-200
title: 参考模型需求
priority: P0
status: draft
parent: REQ-000
---

## 概述

参考模型用于生成golden数据，与被测算子输出进行比对验证。提供C和Python两套实现。

---

## REQ-201 双实现策略

---
id: REQ-201
title: C/Python双实现
priority: P0
status: draft
parent: REQ-200
---

### 描述

参考模型提供C和Python两套实现，用于不同场景。

### 使用场景

| 实现 | 用途 | 优势 |
|------|------|------|
| Python | 离线生成golden数据 | 开发快、易调试、numpy生态 |
| C | 嵌入式在线比对 | 可编译到目标平台、实时比对 |

### 验证流程

```
流程A（离线golden）:
  Python模型 → golden文件 → C测试加载 → 与硬件输出比对

流程B（在线比对）:
  C参考模型 → 实时计算 → 与硬件输出比对
```

### 验收标准

1. Python和C实现结果一致（在浮点精度范围内）
2. Python模型只依赖numpy，不依赖torch/tensorflow
3. C模型无外部依赖，纯C99实现

---

## REQ-202 Python参考模型

---
id: REQ-202
title: Python参考模型
priority: P0
status: draft
parent: REQ-200
depends:
  - REQ-201
---

### 描述

基于numpy的Python参考模型实现。

### 模块结构

```
pymodel/
├── ops/              # 基础算子
│   ├── matmul.py     # 矩阵乘法
│   ├── conv2d.py     # 2D卷积
│   ├── activation.py # 激活函数
│   └── pooling.py    # 池化
├── quantize/         # 量化工具
│   ├── int8.py       # INT8量化
│   ├── fp16.py       # FP16转换
│   └── bf16.py       # BF16转换
├── layers/           # 复合层
│   ├── linear.py     # 全连接
│   └── attention.py  # 注意力
└── utils/            # 工具
    ├── compare.py    # 比较工具
    └── export.py     # 导出golden
```

### 验收标准

1. 所有算子有完整的Python实现
2. 支持导出为二进制golden文件
3. 提供精度比较工具（余弦相似度、MSE等）

---

## REQ-203 C参考模型

---
id: REQ-203
title: C参考模型
priority: P1
status: draft
parent: REQ-200
depends:
  - REQ-201
---

### 描述

纯C实现的参考模型，用于嵌入式在线比对。

### 接口设计

```c
/* 矩阵乘法 */
void ref_matmul_f32(const float *A, const float *B, float *C,
                    int M, int N, int K);
void ref_matmul_f16(const uint16_t *A, const uint16_t *B, uint16_t *C,
                    int M, int N, int K);

/* 卷积 */
void ref_conv2d_f32(const float *input, const float *weight,
                    const float *bias, float *output,
                    const conv2d_params_t *params);

/* 激活函数 */
void ref_relu_f32(const float *input, float *output, int count);
void ref_gelu_f32(const float *input, float *output, int count);
void ref_sigmoid_f32(const float *input, float *output, int count);
```

### 验收标准

1. 无外部依赖，纯C99
2. 可编译到嵌入式平台
3. 与Python实现结果一致

---

## REQ-204 Golden数据管理

---
id: REQ-204
title: Golden数据管理
priority: P0
status: draft
parent: REQ-200
---

### 描述

管理测试用的golden数据。

### 目录结构

```
testdata/
├── generators/       # 数据生成脚本
│   ├── gen_matmul.py
│   └── gen_conv.py
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
typedef struct {
    uint32_t magic;       /* 0x474F4C44 "GOLD" */
    uint32_t version;     /* 格式版本 */
    uint32_t dtype;       /* 数据类型 */
    uint32_t ndim;        /* 维度数 */
    uint32_t shape[8];    /* 各维度大小 */
    uint32_t data_offset; /* 数据偏移 */
} golden_header_t;
```

### 验收标准

1. 统一的二进制格式，C/Python可读写
2. 包含元信息（shape、dtype）
3. 生成脚本可批量生成
4. 支持版本管理

---

## REQ-205 比较工具

---
id: REQ-205
title: 结果比较工具
priority: P0
status: draft
parent: REQ-200
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
| top_k_match | Top-K匹配 | k |
| mse | 均方误差 | max_mse |

### C接口

```c
typedef struct compare_result {
    int passed;
    float max_error;
    float avg_error;
    float cosine_sim;
    int mismatch_count;
    int first_mismatch_idx;
} compare_result_t;

int compare_float_near(const float *expected, const float *actual,
                       int count, float epsilon, compare_result_t *result);

int compare_cosine_sim(const float *expected, const float *actual,
                       int count, float threshold, compare_result_t *result);
```

### 验收标准

1. C和Python都提供相同的比较方法
2. 比较结果包含详细统计信息
3. 支持配置不同算子使用不同比较方法
