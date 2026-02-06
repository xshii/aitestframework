# PLT 平台适配模块设计

---
module: PLT
version: 1.0
date: 2026-02-04
status: draft
requirements: REQ-PLT-001, REQ-PLT-002, REQ-PLT-008
domain: Core
aggregate: Platform
---

## 1. 模块概述

### 1.1 职责

提供硬件抽象层(HAL)，隔离平台差异：
- 统一HAL接口规范
- 各平台HAL实现
- 平台选择机制

### 1.2 DDD定位

- **限界上下文**：平台适配上下文
- **聚合根**：Platform
- **发布事件**：PlatformInitialized

### 1.3 支持平台

| 平台 | 用途 | 优先级 |
|------|------|--------|
| LinuxUT | 单元测试Mock | P0 (MVP) |
| LinuxST | 系统测试Stub | P1 |
| Simulator | 功能仿真 | P1 |
| ESL | 性能建模 | P2 |
| FPGA | 原型验证 | P2 |
| Chip | 真实芯片 | P2 |

---

## 2. 文件结构

```
include/hal/
└── hal_ops.h              # HAL接口定义

src/platform/
├── platform.c             # 平台选择
├── linux_ut/
│   └── ut_hal.c           # LinuxUT Mock实现
├── linux_st/
│   └── st_hal.c           # LinuxST Stub实现
├── simulator/
│   └── sim_hal.c          # Simulator适配
├── esl/
│   └── esl_hal.c          # ESL适配
├── fpga/
│   └── fpga_hal.c         # FPGA适配
└── chip/
    └── chip_hal.c         # Chip适配
```

---

## 3. 聚合设计

### 3.1 Platform聚合

```
Platform (聚合根)
├── name: STRING               # 平台名称
├── caps: PlatformCaps         # 平台能力（值对象）
│   ├── hasRealHardware: BOOL
│   ├── supportsDma: BOOL
│   ├── supportsIrq: BOOL
│   └── maxMemoryMb: UINT32
└── hal: HalOps                # HAL操作集（实体）
```

---

## 4. HAL接口设计

### 4.1 HalOpsStru

```c
typedef struct HalOpsStru {
    /* 生命周期 */
    ERRNO_T (*Init)(VOID);
    ERRNO_T (*Deinit)(VOID);

    /* 寄存器操作 */
    ERRNO_T (*RegWrite)(UINT64 addr, UINT32 value);
    ERRNO_T (*RegRead)(UINT64 addr, UINT32 *value);

    /* 内存操作 */
    ERRNO_T (*MemAlloc)(UINT64 size, UINT32 align, VOID **ptr);
    VOID    (*MemFree)(VOID *ptr);
    ERRNO_T (*MemCopy)(VOID *dst, const VOID *src, UINT64 size);

    /* DMA操作（可选） */
    ERRNO_T (*DmaTransfer)(UINT64 src, UINT64 dst,
                           UINT64 size, INT32 direction);
    ERRNO_T (*DmaWait)(INT32 channel, UINT32 timeoutMs);

    /* 中断操作（可选） */
    ERRNO_T (*IrqWait)(UINT32 mask, UINT32 timeoutMs);
    ERRNO_T (*IrqClear)(UINT32 mask);

    /* 时间操作 */
    UINT64  (*GetTimeUs)(VOID);
    VOID    (*DelayUs)(UINT32 us);

    /* 日志 */
    VOID    (*LogPrint)(INT32 level, const CHAR *fmt, ...);
} HalOpsStru;
```

### 4.2 PlatformCapsStru

```c
typedef struct PlatformCapsStru {
    const CHAR *name;
    BOOL        hasRealHardware;
    BOOL        supportsDma;
    BOOL        supportsIrq;
    UINT32      maxMemoryMb;
} PlatformCapsStru;
```

### 4.3 全局实例与宏

```c
extern HalOpsStru *g_hal;

#define HAL_INIT()              g_hal->Init()
#define HAL_DEINIT()            g_hal->Deinit()
#define HAL_REG_WRITE(addr, v)  g_hal->RegWrite((addr), (v))
#define HAL_REG_READ(addr, pv)  g_hal->RegRead((addr), (pv))
#define HAL_MEM_ALLOC(sz, al, pp) g_hal->MemAlloc((sz), (al), (pp))
#define HAL_MEM_FREE(p)         g_hal->MemFree(p)
#define HAL_GET_TIME_US()       g_hal->GetTimeUs()
#define HAL_LOG(lvl, fmt, ...)  g_hal->LogPrint((lvl), (fmt), ##__VA_ARGS__)
```

---

## 5. 平台选择机制

### 5.1 编译时选择

```makefile
# Makefile
PLATFORM ?= linux_ut
CFLAGS += -DPLATFORM_$(shell echo $(PLATFORM) | tr 'a-z' 'A-Z')
```

### 5.2 platform.c实现

```c
#if defined(PLATFORM_LINUX_UT)
    #include "linux_ut/ut_hal.c"
    HalOpsStru *g_hal = &g_utHalOps;
    static PlatformCapsStru g_caps = {
        .name = "linux_ut",
        .hasRealHardware = FALSE,
        .supportsDma = FALSE,
        .supportsIrq = FALSE,
        .maxMemoryMb = 1024
    };

#elif defined(PLATFORM_LINUX_ST)
    #include "linux_st/st_hal.c"
    HalOpsStru *g_hal = &g_stHalOps;

#elif defined(PLATFORM_SIMULATOR)
    #include "simulator/sim_hal.c"
    HalOpsStru *g_hal = &g_simHalOps;

#else
    #error "No platform defined. Use -DPLATFORM_XXX"
#endif

HalOpsStru *HAL_GetOps(VOID)
{
    return g_hal;
}

const PlatformCapsStru *HAL_GetCaps(VOID)
{
    return &g_caps;
}
```

---

## 6. LinuxUT Mock实现

### 6.1 设计目标

- 纯软件模拟，无硬件依赖
- 支持Mock注入（预设返回值、回调）
- 支持错误注入

### 6.2 寄存器Mock

```c
#define MOCK_REG_SPACE_SIZE  (64 * 1024)  /* 64KB */

static UINT32 g_mockRegs[MOCK_REG_SPACE_SIZE / sizeof(UINT32)];

static ERRNO_T UT_RegWrite(UINT64 addr, UINT32 value)
{
    if (addr >= MOCK_REG_SPACE_SIZE) {
        return AITF_ERR_INVALID_PARAM;
    }
    g_mockRegs[addr / sizeof(UINT32)] = value;
    return AITF_OK;
}

static ERRNO_T UT_RegRead(UINT64 addr, UINT32 *value)
{
    RET_IF_NULL(value);
    if (addr >= MOCK_REG_SPACE_SIZE) {
        return AITF_ERR_INVALID_PARAM;
    }
    *value = g_mockRegs[addr / sizeof(UINT32)];
    return AITF_OK;
}
```

### 6.3 内存Mock

```c
static ERRNO_T UT_MemAlloc(UINT64 size, UINT32 align, VOID **ptr)
{
    RET_IF_NULL(ptr);

    VOID *p;
    if (align > 1) {
        if (posix_memalign(&p, align, size) != 0) {
            return AITF_ERR_OUT_OF_MEMORY;
        }
    } else {
        p = malloc(size);
        if (p == NULL) {
            return AITF_ERR_OUT_OF_MEMORY;
        }
    }

    *ptr = p;
    return AITF_OK;
}

static VOID UT_MemFree(VOID *ptr)
{
    free(ptr);
}
```

### 6.4 时间Mock

```c
#include <sys/time.h>

static UINT64 UT_GetTimeUs(VOID)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (UINT64)tv.tv_sec * 1000000 + tv.tv_usec;
}

static VOID UT_DelayUs(UINT32 us)
{
    usleep(us);
}
```

### 6.5 Mock控制API

```c
/* 设置寄存器预设值 */
ERRNO_T Mock_SetRegValue(UINT64 addr, UINT32 value);

/* 获取寄存器当前值 */
ERRNO_T Mock_GetRegValue(UINT64 addr, UINT32 *value);

/* 设置寄存器读写回调 */
typedef ERRNO_T (*MockRegCallback)(UINT64 addr, UINT32 *value, BOOL isWrite);
ERRNO_T Mock_SetRegCallback(UINT64 addr, MockRegCallback callback);

/* 注入错误 */
typedef enum MockErrorType {
    MOCK_ERR_NONE,
    MOCK_ERR_REG_ACCESS,
    MOCK_ERR_MEM_ALLOC,
    MOCK_ERR_TIMEOUT,
} MockErrorType;
VOID Mock_InjectError(MockErrorType errType, UINT32 count);

/* 重置所有Mock状态 */
VOID Mock_Reset(VOID);
```

### 6.6 完整HAL实例

```c
static HalOpsStru g_utHalOps = {
    .Init       = UT_Init,
    .Deinit     = UT_Deinit,
    .RegWrite   = UT_RegWrite,
    .RegRead    = UT_RegRead,
    .MemAlloc   = UT_MemAlloc,
    .MemFree    = UT_MemFree,
    .MemCopy    = UT_MemCopy,
    .DmaTransfer = NULL,  /* LinuxUT不支持DMA */
    .DmaWait    = NULL,
    .IrqWait    = NULL,   /* LinuxUT不支持中断 */
    .IrqClear   = NULL,
    .GetTimeUs  = UT_GetTimeUs,
    .DelayUs    = UT_DelayUs,
    .LogPrint   = UT_LogPrint,
};
```

---

## 7. 其他平台设计要点

### 7.1 LinuxST (Stub)

- 使用内核驱动桩
- 通过ioctl与桩驱动通信
- 支持基本的寄存器/内存操作

### 7.2 Simulator

- 通过Socket/共享内存与仿真器通信
- 支持完整的DMA/中断模拟
- 位精确

### 7.3 ESL

- 集成SystemC模型
- 周期精确时序
- 性能统计

### 7.4 FPGA/Chip

- 通过PCIe/mmap访问硬件
- 真实DMA和中断
- 完整硬件功能

---

## 8. 领域事件

```yaml
PlatformInitialized:
  source: PLT
  payload:
    platform: STRING
    capabilities: PlatformCapsStru
```

---

## 9. 使用示例

```c
#include "hal/hal_ops.h"

ERRNO_T TestRegisterAccess(VOID)
{
    UINT32 value;

    /* 写寄存器 */
    RET_IF_ERR(HAL_REG_WRITE(0x1000, 0xDEADBEEF));

    /* 读寄存器 */
    RET_IF_ERR(HAL_REG_READ(0x1000, &value));

    TEST_ASSERT_EQ(0xDEADBEEF, value);

    return AITF_OK;
}

ERRNO_T TestMemoryAlloc(VOID)
{
    VOID *buffer = NULL;

    /* 分配对齐内存 */
    RET_IF_ERR(HAL_MEM_ALLOC(4096, 64, &buffer));
    TEST_ASSERT_NOT_NULL(buffer);

    /* 使用内存... */

    HAL_MEM_FREE(buffer);

    return AITF_OK;
}
```

---

## 10. 需求追溯

| 需求ID | 需求标题 | 设计章节 |
|--------|----------|----------|
| REQ-PLT-001 | HAL统一接口规范 | 4 |
| REQ-PLT-002 | LinuxUT Mock实现 | 6 |
| REQ-PLT-008 | 平台选择机制 | 5 |
