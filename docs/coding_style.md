# 代码风格规范

本文档定义AI芯片验证框架的项目特有代码风格规范。

---

## 一、C代码规范

### 1.1 类型命名

#### 通用类型（必须使用，禁止原生类型）

```c
INT8  INT16  INT32  INT64      /* 有符号整数 */
UINT8 UINT16 UINT32 UINT64     /* 无符号整数 */
FLOAT32 FLOAT64                /* 浮点数 */
CHAR                           /* 字符 */
VOID                           /* 空类型 */
VOID *                         /* 地址类型(64位) */
bool                           /* 布尔 */
```

#### 自定义类型（大驼峰 + 后缀，无下划线）

| 后缀 | 类型 | 示例 |
|------|------|------|
| `Stru` | 结构体 | `TestCaseStru` |
| `Enum` | 枚举定义 | `HalDmaDirEnum` |
| `_ENUM_UINTx` | 枚举类型别名 | `HAL_DMA_DIR_ENUM_UINT8` |
| `Union` | 联合体 | `DataValueUnion` |
| `Func` | 函数指针 | `HalRegReadFunc` |

### 1.2 变量命名

```c
INT32 regValue;                              /* 局部变量: 小驼峰 */
SEC_DDR_BSS HalOpsStru *g_hal;               /* 全局变量: g_ + 小驼峰 */
#define MAX_TEST_COUNT  1000                 /* 常量: 全大写 + 下划线 */

/* 注意: 避免使用static变量，优先使用全局变量或通过参数传递 */
```

### 1.3 内存段标记

```c
/* 格式: SEC_<内存层级>_<段类型> */
/* 内存层级: L1, L2, L3, DDR */
/* 段类型: BSS(未初始化), DATA(已初始化), RODATA(只读), TEXT(代码) */

/* 变量标记 */
#define FAST_BUFFER_SIZE  256
SEC_L1_BSS   UINT32 g_fastBuffer[FAST_BUFFER_SIZE]; /* L1 未初始化数据 */
SEC_L2_DATA  UINT32 g_configTable[] = {1, 2};       /* L2 已初始化数据 */
SEC_DDR_RODATA const CHAR *g_version = "1.0"; /* DDR 只读数据 */

/* 函数标记 */
SEC_L1_TEXT ERRNO_T HalFastRead(VOID *addr, UINT32 *value)
{
    RET_IF_PTR_INVALID(value, ERR_HAL_0001);
    UINT64 offset = (UINT64)addr / sizeof(UINT32);
    RET_IF(offset >= FAST_BUFFER_SIZE, ERR_HAL_0002);
    *value = g_fastBuffer[offset];
    return ERR_OK;
}
```

### 1.4 函数命名

```c
/* 对外接口: MODULE_大驼峰 */
SEC_DDR_TEXT ERRNO_T HAL_RegWrite(VOID *addr, UINT32 value);

/* 内部接口: 大驼峰（含模块名） */
SEC_DDR_TEXT ERRNO_T HalParseConfig(const CHAR *path);
```

### 1.5 枚举定义

```c
typedef enum HalDmaDirEnum {
    HAL_DMA_H2D = 0,   /* 枚举值: 全大写 */
    HAL_DMA_D2H = 1,
} HalDmaDirEnum;
typedef UINT8 HAL_DMA_DIR_ENUM_UINT8;  /* 类型别名: 全大写 + _ENUM_UINTx */
```

### 1.6 安全函数

```c
/* 必须使用安全函数，禁止 memset/memcpy/strcpy 等 */

/* destMax == srcLen 时，可不校验返回值，但必须用(void)显式说明 */
(void)memset_s(buf, sizeof(buf), 0, sizeof(buf));

/* destMax != srcLen 时，必须校验返回值 */
ERRNO_T ret = memcpy_s(dest, destMax, src, srcLen);
RET_IF_NOT_OK(ret, ERR_HAL_0002);
```

### 1.7 函数返回值规范

```c
/* 错误：直接返回指针，调用方可能忘记判空 */
SEC_DDR_TEXT HalOpsStru *HAL_GetOps(VOID);

/* 正确：POSIX风格，返回错误码，出参为指针的指针 */
SEC_DDR_TEXT ERRNO_T HAL_GetOps(HalOpsStru **ops);
```

### 1.8 判断语法糖

#### 命名规则

- 普通系列：`<动作>_IF_<条件>`
- BUG系列：`BUG_<动作>_<条件>`

| 前缀 | 说明 |
|------|------|
| `RET_IF` | 条件成立则返回 |
| `CONT_IF` | 条件成立则continue |
| `BREAK_IF` | 条件成立则break |
| `BUG_RET` | 不可能发生，触发则为BUG |
| `BUG_CONT` | 不可能发生，触发则continue |
| `BUG_BREAK` | 不可能发生，触发则break |

```c
/* RET_IF 系列 */
RET_IF(condition, ERR_HAL_0001);           /* 通用条件判断 */
RET_IF_PTR_INVALID(ops, ERR_HAL_0003);
RET_IF_NOT_OK(ret, ERR_HAL_0004);
RET_VOID_IF_PTR_INVALID(ptr);

/* BUG 系列 - 不可能发生的情况 */
BUG_RET_PTR_INVALID(ctx, ERR_HAL_0005);
BUG_CONT_PTR_INVALID(item);
BUG_BREAK_NOT_OK(ret);
```

### 1.9 错误码定义

```c
/* 错误码返回值类型 */
typedef INT32 ERRNO_T;

/* 错误码通过枚举定义，格式: ERR_MODULE_XXXX，XXXX为16进制序号，保证全局唯一 */
typedef enum ErrCodeEnum {
    ERR_OK          = 0,
    ERR_HAL_0001    = 0x010001,    /* hal.c:45 参数无效 */
    ERR_HAL_0002    = 0x010002,    /* hal.c:58 越界访问 */
    ERR_PLAT_0001   = 0x020001,    /* ut_hal.c:86 参数无效 */
    ERR_PLAT_0002   = 0x020002,    /* ut_hal.c:71 参数无效 */
    ERR_PLAT_0003   = 0x020003,    /* ut_hal.c:74 越界访问 */
} ErrCodeEnum;
typedef UINT32 ERR_CODE_ENUM_UINT32;
```

---

## 二、示例代码

### 2.1 C平台适配

```c
/* platform/linux_ut/ut_hal.c */

#define REG_COUNT  (REG_SPACE_SIZE / sizeof(UINT32))

SEC_DDR_BSS UINT32 g_mockRegs[REG_COUNT];

/* 前向声明 */
ERRNO_T UtRegRead(VOID *addr, UINT32 *value);
ERRNO_T UtInit(VOID);

SEC_DDR_DATA HalOpsStru g_utHalOps = {
    .RegRead = UtRegRead,
    .Init    = UtInit,
};

SEC_DDR_TEXT ERRNO_T UtRegRead(VOID *addr, UINT32 *value)
{
    RET_IF_PTR_INVALID(value, ERR_PLAT_0002);
    UINT64 offset = (UINT64)addr / sizeof(UINT32);
    RET_IF(offset >= REG_COUNT, ERR_PLAT_0003);
    *value = g_mockRegs[offset];
    return ERR_OK;
}

SEC_DDR_TEXT ERRNO_T UtInit(VOID)
{
    (void)memset_s(g_mockRegs, sizeof(g_mockRegs), 0, sizeof(g_mockRegs));
    return ERR_OK;
}

SEC_DDR_TEXT ERRNO_T PLAT_LinuxUtGetHal(HalOpsStru **ops)
{
    RET_IF_PTR_INVALID(ops, ERR_PLAT_0001);
    *ops = &g_utHalOps;
    return ERR_OK;
}
```

---

## 三、规范汇总表

| 类别 | 规范 | 示例 |
|------|------|------|
| 通用类型 | 全大写 | `INT32`, `CHAR`, `VOID` |
| 地址类型 | VOID * | 64位地址 |
| 自定义类型 | 大驼峰+后缀 | `TestCaseStru` |
| 对外函数 | SEC_xx_TEXT MODULE_大驼峰 | `HAL_RegWrite()` |
| 内部函数 | SEC_xx_TEXT 大驼峰 | `HalParseConfig()` |
| 全局变量 | g_小驼峰 | `g_hal` |
| static变量 | 避免使用 | 用全局变量或参数传递 |
| 内存段标记 | SEC_层级_段类型 | `SEC_L1_BSS` |
| 局部变量 | 小驼峰 | `regValue` |
| 枚举值 | 全大写 | `HAL_DMA_H2D` |
| 枚举类型别名 | 全大写_ENUM_UINTx | `HAL_DMA_DIR_ENUM_UINT8` |
| 宏常量 | 全大写 | `MAX_TEST_COUNT` |
| 安全函数 | 必须使用 | `memset_s()` |
| 错误码 | ERR_MODULE_XXXX | `ERR_HAL_0001` |
| 错误码返回类型 | ERRNO_T | `typedef INT32 ERRNO_T` |
| 返回指针/异常值 | POSIX风格出参 | `ERRNO_T Func(T **out)` |
| 判断语法糖 | RET_IF/CONT_IF/BREAK_IF | `RET_IF_PTR_INVALID(ptr, err)` |
| BUG检测 | BUG_RET/BUG_CONT/BUG_BREAK | `BUG_RET_PTR_INVALID(ptr, err)` |
| 数组下标 | 校验后使用 | `RET_IF(idx >= SIZE, err)` |
