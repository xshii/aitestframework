# SYS 系统基础模块设计

---
module: SYS
version: 1.0
date: 2026-02-04
status: draft
requirements: REQ-SYS-010
---

## 1. 模块概述

### 1.1 职责

提供全框架共享的基础设施：
- 基础类型定义（确保跨平台一致性）
- 统一错误码
- 通用工具宏

### 1.2 设计原则

- 零外部依赖
- 纯C99标准
- 所有模块必须包含

---

## 2. 文件结构

```
include/common/
├── types.h      # 基础类型定义
├── errno.h      # 错误码定义
└── macros.h     # 通用宏
```

---

## 3. 类型定义 (types.h)

### 3.1 整数类型

```c
typedef signed char        INT8;
typedef unsigned char      UINT8;
typedef signed short       INT16;
typedef unsigned short     UINT16;
typedef signed int         INT32;
typedef unsigned int       UINT32;
typedef signed long long   INT64;
typedef unsigned long long UINT64;
```

### 3.2 浮点类型

```c
typedef float              FLOAT32;
typedef double             FLOAT64;
```

### 3.3 通用类型

```c
typedef char               CHAR;
typedef void               VOID;
typedef INT32              ERRNO_T;
typedef INT32              BOOL;

#define TRUE   1
#define FALSE  0
#define NULL   ((VOID *)0)
```

### 3.4 内存段标记

```c
/* 平台相关，LinuxUT下为空 */
#define SEC_DDR_TEXT
#define SEC_DDR_DATA
#define SEC_DDR_BSS
#define SEC_DDR_RODATA  const
```

---

## 4. 错误码定义 (errno.h)

### 4.1 错误码分段

| 段 | 范围 | 用途 |
|----|------|------|
| 0x0000_0000 | 成功 | AITF_OK |
| 0x0001_xxxx | 通用错误 | 空指针、参数、内存等 |
| 0x0002_xxxx | 框架错误 | 测试失败、断言等 |
| 0x0003_xxxx | 平台错误 | HAL相关 |
| 0x0004_xxxx | 模型错误 | Golden、比较等 |

### 4.2 错误码列表

```c
/* 成功 */
#define AITF_OK                     ((ERRNO_T)0x00000000)

/* 通用错误 0x0001_xxxx */
#define AITF_ERR_NULL_PTR           ((ERRNO_T)0x00010001)
#define AITF_ERR_INVALID_PARAM      ((ERRNO_T)0x00010002)
#define AITF_ERR_OUT_OF_MEMORY      ((ERRNO_T)0x00010003)
#define AITF_ERR_NOT_FOUND          ((ERRNO_T)0x00010004)
#define AITF_ERR_TIMEOUT            ((ERRNO_T)0x00010005)
#define AITF_ERR_NOT_SUPPORTED      ((ERRNO_T)0x00010006)
#define AITF_ERR_ALREADY_EXISTS     ((ERRNO_T)0x00010007)
#define AITF_ERR_IO                 ((ERRNO_T)0x00010008)

/* 框架错误 0x0002_xxxx */
#define AITF_ERR_TEST_FAIL          ((ERRNO_T)0x00020001)
#define AITF_ERR_TEST_SKIP          ((ERRNO_T)0x00020002)
#define AITF_ERR_TEST_TIMEOUT       ((ERRNO_T)0x00020003)
#define AITF_ERR_TEST_CRASH         ((ERRNO_T)0x00020004)
#define AITF_ERR_ASSERT_FAIL        ((ERRNO_T)0x00020005)
#define AITF_ERR_NO_TESTCASE        ((ERRNO_T)0x00020006)

/* 平台错误 0x0003_xxxx */
#define AITF_ERR_HAL_INIT           ((ERRNO_T)0x00030001)
#define AITF_ERR_HAL_REG_ACCESS     ((ERRNO_T)0x00030002)
#define AITF_ERR_HAL_MEM_ALLOC      ((ERRNO_T)0x00030003)
#define AITF_ERR_PLATFORM_MISMATCH  ((ERRNO_T)0x00030004)

/* 模型错误 0x0004_xxxx */
#define AITF_ERR_GOLDEN_NOT_FOUND   ((ERRNO_T)0x00040001)
#define AITF_ERR_GOLDEN_FORMAT      ((ERRNO_T)0x00040002)
#define AITF_ERR_COMPARE_MISMATCH   ((ERRNO_T)0x00040003)
#define AITF_ERR_DATA_OVERFLOW      ((ERRNO_T)0x00040004)
```

### 4.3 判断宏

```c
#define AITF_IS_OK(err)      ((err) == AITF_OK)
#define AITF_IS_ERR(err)     ((err) != AITF_OK)
```

---

## 5. 通用宏 (macros.h)

### 5.1 返回值检查

```c
#define RET_IF_NULL(ptr) do { \
    if ((ptr) == NULL) { \
        return AITF_ERR_NULL_PTR; \
    } \
} while (0)

#define RET_IF_ERR(expr) do { \
    ERRNO_T _ret = (expr); \
    if (AITF_IS_ERR(_ret)) { \
        return _ret; \
    } \
} while (0)

#define RET_VAL_IF_ERR(expr, val) do { \
    ERRNO_T _ret = (expr); \
    if (AITF_IS_ERR(_ret)) { \
        return (val); \
    } \
} while (0)
```

### 5.2 工具宏

```c
#define ARRAY_SIZE(arr)    (sizeof(arr) / sizeof((arr)[0]))
#define MIN(a, b)          (((a) < (b)) ? (a) : (b))
#define MAX(a, b)          (((a) > (b)) ? (a) : (b))
#define ALIGN_UP(x, align)   (((x) + (align) - 1) & ~((align) - 1))
#define ALIGN_DOWN(x, align) ((x) & ~((align) - 1))
#define UNUSED(x)          ((VOID)(x))
```

---

## 6. 使用示例

```c
#include "common/types.h"
#include "common/errno.h"
#include "common/macros.h"

ERRNO_T MyFunction(const VOID *input, UINT32 size)
{
    RET_IF_NULL(input);

    if (size == 0) {
        return AITF_ERR_INVALID_PARAM;
    }

    /* 业务逻辑 */

    return AITF_OK;
}
```

---

## 7. 需求追溯

| 需求ID | 需求标题 | 设计章节 |
|--------|----------|----------|
| REQ-SYS-010 | 基础类型定义 | 3, 4, 5 |
