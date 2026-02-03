# REQ-PLT 平台适配层需求

---
id: REQ-PLT
title: 平台适配层需求
priority: P0
status: draft
parent: REQ-SYS
depends:
  - REQ-FWK
---

## 概述

平台适配层为不同验证平台提供统一的HAL接口抽象和适配桩实现，使同一套测试用例可以在不同平台运行。

**本模块职责：**
- 定义统一的HAL接口规范
- 提供LinuxUT/ST的Mock/Stub实现
- 提供与外部平台（Simulator/ESL/FPGA/Chip）通信的适配桩
- 平台选择和初始化机制

**不在本模块范围：**
- 仿真器/ESL/FPGA工具本身（外部二进制下载）
- 芯片驱动实现（被测对象）

```
┌─────────────────────────────────────────────────────────────┐
│                    Test Framework                            │
├─────────────────────────────────────────────────────────────┤
│                 HAL 统一接口 (hal_ops.h)                     │
├──────────┬──────────┬──────────┬──────────┬─────────────────┤
│ LinuxUT  │ LinuxST  │ Sim适配  │ ESL适配  │ FPGA/Chip适配   │
│ Mock实现 │ Stub实现 │ 通信桩   │ 通信桩   │ 通信桩          │
└──────────┴──────────┴──────────┴──────────┴─────────────────┘
                            ↓ 通信协议/驱动接口
               外部: Simulator / ESL / FPGA / Chip
```

---

## REQ-PLT-001 HAL统一接口

---
id: REQ-PLT-001
title: HAL统一接口规范
priority: P0
status: draft
parent: REQ-PLT
---

### 描述

定义统一的硬件抽象层接口，所有平台适配桩必须实现此接口。

### 核心接口

```c
/* include/hal/hal_ops.h */

typedef struct HalOpsStru {
    /* === 寄存器操作 === */
    ERRNO_T (*RegWrite)(VOID *addr, UINT32 value);
    ERRNO_T (*RegRead)(VOID *addr, UINT32 *value);

    /* === DMA操作 === */
    ERRNO_T (*DmaTransfer)(VOID *src, VOID *dst, UINT64 size, INT32 dir);
    ERRNO_T (*DmaWait)(INT32 channel, UINT32 timeoutMs);

    /* === 中断操作 === */
    ERRNO_T (*IrqWait)(UINT32 irqMask, UINT32 timeoutMs);
    ERRNO_T (*IrqClear)(UINT32 irqMask);

    /* === 内存操作 === */
    ERRNO_T (*MemAlloc)(UINT64 size, UINT32 flags, VOID **ptr);
    VOID    (*MemFree)(VOID *ptr);
    ERRNO_T (*MemVirtToPhys)(VOID *virtAddr, UINT64 *physAddr);

    /* === 平台控制 === */
    ERRNO_T (*PlatformInit)(VOID);
    ERRNO_T (*PlatformDeinit)(VOID);
    VOID    (*DelayUs)(UINT32 us);
    UINT64  (*GetTimeUs)(VOID);

    /* === 日志输出 === */
    VOID (*LogPrint)(INT32 level, const CHAR *fmt, ...);

    /* === 平台信息 === */
    ERRNO_T (*GetPlatformName)(const CHAR **name);
    ERRNO_T (*GetPlatformCaps)(PlatformCapsStru *caps);
} HalOpsStru;

/* 全局HAL实例指针 */
SEC_DDR_BSS HalOpsStru *g_hal;
```

### 便捷宏

```c
#define HAL_REG_WRITE(addr, val)       ((g_hal)->RegWrite((addr), (val)))
#define HAL_REG_READ(addr, pVal)       ((g_hal)->RegRead((addr), (pVal)))
#define HAL_DMA_TRANSFER(s, d, sz, dir) ((g_hal)->DmaTransfer((s), (d), (sz), (dir)))
#define HAL_MEM_ALLOC(size, flags, ptr) ((g_hal)->MemAlloc((size), (flags), (ptr)))
#define HAL_MEM_FREE(ptr)              ((g_hal)->MemFree(ptr))
```

### 常量定义

```c
/* DMA方向枚举 */
typedef enum HalDmaDirEnum {
    HAL_DMA_H2D = 0,  /* Host to Device */
    HAL_DMA_D2H = 1,  /* Device to Host */
    HAL_DMA_D2D = 2,  /* Device to Device */
} HalDmaDirEnum;
typedef UINT8 HAL_DMA_DIR_ENUM_UINT8;

/* 内存标志 */
#define HAL_MEM_CACHED   (1U << 0)
#define HAL_MEM_DMA      (1U << 1)
#define HAL_MEM_DEVICE   (1U << 2)
```

### 验收标准

1. 所有平台适配桩实现完整的hal_ops_t接口
2. 接口语义在各平台保持一致
3. 未实现的可选接口返回-ENOTSUP
4. 提供便捷宏简化调用

---

## REQ-PLT-002 LinuxUT Mock实现

---
id: REQ-PLT-002
title: LinuxUT平台Mock实现
priority: P0
status: draft
parent: REQ-PLT
---

### 描述

Linux单元测试平台，使用纯内存Mock实现，用于快速单元测试。

### 特性

| 特性 | 实现方式 |
|------|----------|
| 寄存器访问 | 内存数组模拟 |
| DMA传输 | memcpy模拟 |
| 中断 | 条件变量/立即返回 |
| 内存分配 | malloc/free |
| 时间 | gettimeofday |

### Mock控制接口

```c
/* 供测试用例控制Mock行为 */

/* 寄存器Mock */
SEC_DDR_TEXT VOID UT_MockRegSet(VOID *addr, UINT32 value);
SEC_DDR_TEXT ERRNO_T UT_MockRegGet(VOID *addr, UINT32 *value);
SEC_DDR_TEXT VOID UT_MockRegReset(VOID);

/* 预设寄存器读取序列（多次读取返回不同值） */
SEC_DDR_TEXT ERRNO_T UT_MockRegSequence(VOID *addr, const UINT32 *values, INT32 count);

/* 错误注入 */
SEC_DDR_TEXT VOID UT_MockInjectError(const CHAR *funcName, ERRNO_T errorCode);
SEC_DDR_TEXT VOID UT_MockClearErrors(VOID);

/* 调用记录（用于验证） */
SEC_DDR_TEXT INT32 UT_MockGetCallCount(const CHAR *funcName);
SEC_DDR_TEXT VOID UT_MockResetCallCounts(VOID);
```

### 使用示例

```c
TEST_CASE(driver, init_check_status)
{
    /* 预设寄存器返回值 */
    UT_MockRegSet(REG_STATUS_ADDR, 0x0001);

    /* 调用被测代码 */
    ERRNO_T ret = NPU_Init();

    /* 验证 */
    TEST_ASSERT_EQ(ret, ERR_OK);
    TEST_ASSERT(UT_MockGetCallCount("RegRead") >= 1);

    return TEST_PASS;
}
```

### 验收标准

1. 纯内存操作，无硬件依赖
2. 支持寄存器/DMA/中断Mock
3. 支持错误注入测试
4. 支持调用记录和验证
5. 执行速度最快（目标<1ms/用例）

---

## REQ-PLT-003 LinuxST Stub实现

---
id: REQ-PLT-003
title: LinuxST平台Stub实现
priority: P0
status: draft
parent: REQ-PLT
---

### 描述

Linux系统测试平台，使用Stub实现基本功能逻辑，用于集成测试。

### 与LinuxUT区别

| 特性 | LinuxUT (Mock) | LinuxST (Stub) |
|------|----------------|----------------|
| **目的** | 单元测试 | 集成测试 |
| **外部依赖** | 全部Mock | 部分Stub + 部分真实 |
| **行为逻辑** | 无，只返回预设值 | 有简化逻辑 |
| **数据规模** | 小（<1KB） | 中等（<1MB） |
| **文件IO** | 无 | 支持 |

### Stub实现要点

```c
/* Stub有简化的行为逻辑，不只是返回预设值 */

/* 例：带状态机的任务提交Stub */
typedef enum NpuStateEnum {
    NPU_STATE_IDLE = 0,
    NPU_STATE_BUSY = 1,
    NPU_STATE_DONE = 2,
} NpuStateEnum;
typedef UINT8 NPU_STATE_ENUM_UINT8;

SEC_DDR_BSS NPU_STATE_ENUM_UINT8 g_npuState = NPU_STATE_IDLE;

SEC_DDR_TEXT ERRNO_T StSubmitTask(const TaskDescStru *task)
{
    RET_IF_PTR_INVALID(task, ERR_PLAT_0001);
    RET_IF(g_npuState != NPU_STATE_IDLE, ERR_PLAT_0002);

    g_npuState = NPU_STATE_BUSY;

    /* 简化执行：直接用参考模型计算 */
    (VOID)RefExecuteTask(task);

    g_npuState = NPU_STATE_DONE;
    return ERR_OK;
}
```

### 数据加载支持

```c
/* LinuxST支持从文件加载测试数据 */
SEC_DDR_TEXT ERRNO_T ST_LoadBinary(const CHAR *path, VOID *buf, UINT64 size);
SEC_DDR_TEXT ERRNO_T ST_SaveBinary(const CHAR *path, const VOID *buf, UINT64 size);
```

### 验收标准

1. Stub实现基本功能逻辑
2. 支持组件间集成测试
3. 支持文件IO（加载测试数据）
4. 执行速度快（目标<100ms/用例）

---

## REQ-PLT-004 Simulator适配桩

---
id: REQ-PLT-004
title: Simulator平台适配桩
priority: P0
status: draft
parent: REQ-PLT
---

### 描述

连接外部NPU功能仿真器的适配桩，通过进程间通信与仿真器交互。

### 通信架构

```
┌─────────────────────────────────────────────────────┐
│              Test Framework (本仓库)                 │
├─────────────────────────────────────────────────────┤
│          Simulator HAL 适配桩 (sim_hal.c)            │
├─────────────────────────────────────────────────────┤
│          通信层 (sim_comm.c)                         │
│          Socket / Shared Memory / RPC               │
└─────────────────────────────────────────────────────┘
                         ↓ IPC
┌─────────────────────────────────────────────────────┐
│          NPU Simulator (外部二进制)                  │
└─────────────────────────────────────────────────────┘
```

### 通信协议

```c
/* 消息格式 */
typedef struct SimMsgStru {
    UINT32 type;       /* 消息类型 */
    UINT32 seq;        /* 序列号 */
    UINT32 length;     /* 数据长度 */
    UINT8  data[];     /* 可变长数据 */
} SimMsgStru;

/* 消息类型 */
typedef enum SimMsgTypeEnum {
    SIM_MSG_REG_READ  = 1,
    SIM_MSG_REG_WRITE = 2,
    SIM_MSG_DMA_START = 3,
    SIM_MSG_DMA_DONE  = 4,
    SIM_MSG_IRQ       = 5,
    SIM_MSG_RESET     = 6,
} SimMsgTypeEnum;
typedef UINT8 SIM_MSG_TYPE_ENUM_UINT8;
```

### 连接管理

```c
/* 连接仿真器 */
SEC_DDR_TEXT ERRNO_T SIM_Connect(const CHAR *host, INT32 port);
SEC_DDR_TEXT ERRNO_T SIM_Disconnect(VOID);
SEC_DDR_TEXT bool SIM_IsConnected(VOID);

/* 仿真器控制 */
SEC_DDR_TEXT ERRNO_T SIM_Reset(VOID);
SEC_DDR_TEXT ERRNO_T SIM_Step(INT32 cycles);  /* 单步执行 */
```

### 验收标准

1. 支持Socket通信连接外部仿真器
2. 支持共享内存通信（可选，高性能场景）
3. 通信超时可配置
4. 连接断开自动重试或报错

---

## REQ-PLT-005 ESL适配桩

---
id: REQ-PLT-005
title: ESL平台适配桩
priority: P1
status: draft
parent: REQ-PLT
---

### 描述

连接外部ESL（Electronic System Level）性能模型的适配桩。

### 与Simulator区别

| 特性 | Simulator | ESL |
|------|-----------|-----|
| **精度** | 功能精确 | 周期精确 |
| **输出** | 功能结果 | 功能结果 + 性能数据 |
| **用途** | 功能验证 | 性能验证 |

### 性能数据获取

```c
/* ESL特有：获取性能统计 */
typedef struct EslPerfStatsStru {
    UINT64   totalCycles;
    UINT64   computeCycles;
    UINT64   stallCycles;
    UINT64   ddrReadBytes;
    UINT64   ddrWriteBytes;
    FLOAT64  bandwidthGbps;
    FLOAT64  utilization;
} EslPerfStatsStru;

SEC_DDR_TEXT ERRNO_T ESL_GetPerfStats(EslPerfStatsStru *stats);
SEC_DDR_TEXT ERRNO_T ESL_ResetPerfStats(VOID);
```

### 验收标准

1. 复用Simulator通信架构
2. 支持获取周期级性能数据
3. 支持性能统计重置
4. 性能数据可导出

---

## REQ-PLT-006 FPGA适配桩

---
id: REQ-PLT-006
title: FPGA平台适配桩
priority: P1
status: draft
parent: REQ-PLT
---

### 描述

连接FPGA原型板的适配桩，通过PCIe/USB与FPGA通信。

### 通信方式

```c
/* FPGA通信接口（调用系统驱动） */

/* 打开设备 */
SEC_DDR_TEXT ERRNO_T FPGA_Open(const CHAR *devicePath);  /* 如 /dev/npu_fpga */
SEC_DDR_TEXT ERRNO_T FPGA_Close(VOID);

/* 寄存器访问（通过mmap或ioctl） */
SEC_DDR_TEXT ERRNO_T FPGA_RegWrite(VOID *addr, UINT32 value);
SEC_DDR_TEXT ERRNO_T FPGA_RegRead(VOID *addr, UINT32 *value);

/* DMA（通过驱动） */
SEC_DDR_TEXT ERRNO_T FPGA_DmaTransfer(VOID *src, VOID *dst, UINT64 size, INT32 dir);
SEC_DDR_TEXT ERRNO_T FPGA_DmaWait(UINT32 timeoutMs);
```

### 内存管理

```c
/* FPGA DMA需要物理连续内存 */
typedef struct FpgaDmaBufStru {
    VOID   *virtAddr;      /* 用户空间虚拟地址 */
    UINT64  physAddr;      /* 物理地址（供FPGA使用） */
    UINT64  size;
} FpgaDmaBufStru;

SEC_DDR_TEXT ERRNO_T FPGA_DmaAlloc(UINT64 size, FpgaDmaBufStru *buf);
SEC_DDR_TEXT ERRNO_T FPGA_DmaFree(FpgaDmaBufStru *buf);
```

### 验收标准

1. 支持通过系统驱动访问FPGA
2. 支持DMA缓冲区分配
3. 支持中断等待
4. 设备打开失败有明确错误信息

---

## REQ-PLT-007 Chip适配桩

---
id: REQ-PLT-007
title: Chip平台适配桩
priority: P1
status: draft
parent: REQ-PLT
---

### 描述

真实芯片平台的适配桩，通过内核驱动或直接内存映射访问硬件。

### 访问方式

```c
/* 方式1：通过专用驱动 */
SEC_DDR_TEXT ERRNO_T CHIP_Open(const CHAR *devicePath);  /* 如 /dev/npu */

/* 方式2：直接内存映射（需root权限） */
SEC_DDR_TEXT ERRNO_T CHIP_MmapInit(UINT64 physBase, UINT64 size);
```

### 中断处理

```c
/* 基于poll/select的中断等待 */
SEC_DDR_TEXT ERRNO_T CHIP_IrqWait(UINT32 irqMask, UINT32 timeoutMs);
SEC_DDR_TEXT ERRNO_T CHIP_IrqEnable(UINT32 irqMask);
SEC_DDR_TEXT ERRNO_T CHIP_IrqDisable(UINT32 irqMask);
```

### 验收标准

1. 支持通过内核驱动访问
2. 支持mmap直接访问（可选）
3. 支持中断等待
4. 与FPGA适配桩接口一致（便于复用测试代码）

---

## REQ-PLT-008 平台选择机制

---
id: REQ-PLT-008
title: 平台选择与初始化
priority: P0
status: draft
parent: REQ-PLT
---

### 描述

编译时和运行时选择目标平台的机制。

### 编译时选择

```makefile
# Makefile
PLATFORM ?= linux_ut

ifeq ($(PLATFORM),linux_ut)
    PLATFORM_SRCS = platform/linux_ut/*.c
    CFLAGS += -DPLATFORM_LINUX_UT
else ifeq ($(PLATFORM),linux_st)
    PLATFORM_SRCS = platform/linux_st/*.c
    CFLAGS += -DPLATFORM_LINUX_ST
else ifeq ($(PLATFORM),simulator)
    PLATFORM_SRCS = platform/simulator/*.c
    CFLAGS += -DPLATFORM_SIMULATOR
# ... 其他平台
endif
```

### 平台注册

```c
/* 每个平台实现注册函数 */
SEC_DDR_TEXT ERRNO_T PLAT_LinuxUtGetHal(HalOpsStru **ops);
SEC_DDR_TEXT ERRNO_T PLAT_LinuxStGetHal(HalOpsStru **ops);
SEC_DDR_TEXT ERRNO_T PLAT_SimulatorGetHal(HalOpsStru **ops);
/* ... */

/* 根据编译宏选择 */
SEC_DDR_TEXT ERRNO_T PLAT_GetHal(HalOpsStru **ops)
{
    RET_IF_PTR_INVALID(ops, ERR_PLAT_0001);
#if defined(PLATFORM_LINUX_UT)
    return PLAT_LinuxUtGetHal(ops);
#elif defined(PLATFORM_LINUX_ST)
    return PLAT_LinuxStGetHal(ops);
#elif defined(PLATFORM_SIMULATOR)
    return PLAT_SimulatorGetHal(ops);
#else
    #error "Unknown platform"
#endif
}
```

### 平台能力查询

```c
typedef struct PlatformCapsStru {
    const CHAR *name;           /* 平台名称 */
    UINT32 flags;               /* 能力标志 */
    UINT32 maxDmaSize;          /* 最大DMA大小 */
} PlatformCapsStru;

#define PLAT_CAP_DMA        (1U << 0)  /* 支持DMA */
#define PLAT_CAP_IRQ        (1U << 1)  /* 支持中断 */
#define PLAT_CAP_PERF       (1U << 2)  /* 支持性能统计 */
#define PLAT_CAP_REAL_HW    (1U << 3)  /* 真实硬件 */

/* 测试用例可根据能力决定是否跳过 */
TEST_CASE(perf, bandwidth)
{
    PlatformCapsStru caps;
    ERRNO_T ret = g_hal->GetPlatformCaps(&caps);
    RET_IF_NOT_OK(ret, TEST_ERROR);

    if ((caps.flags & PLAT_CAP_PERF) == 0) {
        TEST_SKIP("Platform does not support perf stats");
    }
    /* ... */
}
```

### 验收标准

1. 编译时通过PLATFORM变量选择
2. 每个平台编译为独立可执行文件
3. 支持查询平台能力
4. 用例可根据能力自动跳过

---

## REQ-PLT-009 平台配置文件

---
id: REQ-PLT-009
title: 平台配置文件
priority: P1
status: draft
parent: REQ-PLT
---

### 描述

各平台的运行时配置文件规范。

### 配置文件格式

```yaml
# configs/platforms/simulator.yaml
platform:
  name: simulator
  description: "NPU功能仿真器"

# 连接配置
connection:
  type: socket        # socket / shm
  host: localhost
  port: 12345
  timeout_ms: 5000

# 超时配置
timeouts:
  init_ms: 30000
  reg_access_ms: 100
  dma_transfer_ms: 10000
  task_complete_ms: 60000

# 日志配置
logging:
  level: info         # debug / info / warn / error
  file: logs/simulator.log
```

```yaml
# configs/platforms/fpga.yaml
platform:
  name: fpga
  description: "FPGA原型板"

# 设备配置
device:
  path: /dev/npu_fpga
  bar0_size: 0x100000

# DMA配置
dma:
  max_transfer_size: 0x1000000  # 16MB
  buffer_count: 4
```

### 配置加载

```c
/* 加载平台配置 */
int platform_load_config(const char *config_path);

/* 获取配置值 */
const char* platform_config_get_str(const char *key, const char *default_val);
int platform_config_get_int(const char *key, int default_val);

/* 配置路径优先级 */
/* 1. 命令行 --platform-config <path> */
/* 2. 环境变量 NPU_PLATFORM_CONFIG */
/* 3. 默认路径 configs/platforms/<platform>.yaml */
```

### 验收标准

1. 每个平台有独立配置文件
2. 支持命令行/环境变量/默认路径
3. 配置加载失败有明确提示
4. 支持配置值覆盖

---

## REQ-PLT-010 跨平台数据兼容

---
id: REQ-PLT-010
title: 跨平台数据兼容
priority: P1
status: draft
parent: REQ-PLT
---

### 描述

确保测试数据和结果在不同平台间兼容。

### 数据格式规范

```c
/* 统一字节序：小端 */
#define DATA_BYTE_ORDER  __ORDER_LITTLE_ENDIAN__

/* 使用统一类型定义（参见 coding_style.md） */
/* INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64 */
/* FLOAT32, FLOAT64, CHAR, VOID */

/* FP16/BF16存储格式 */
typedef UINT16 FP16;
typedef UINT16 BF16;
```

### 地址空间抽象

```c
/* 不同平台地址空间可能不同，提供转换接口 */
SEC_DDR_TEXT ERRNO_T PLAT_HostToDevice(VOID *hostAddr, UINT64 *deviceAddr);
SEC_DDR_TEXT ERRNO_T PLAT_DeviceToHost(UINT64 deviceAddr, VOID **hostAddr);
```

### 验收标准

1. 数据格式跨平台一致（小端）
2. 使用固定宽度类型
3. Golden数据各平台通用
4. 提供地址转换接口
