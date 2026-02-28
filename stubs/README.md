# 桩代码框架 (stubs)

## 架构

```
平台 (func_sim / FPGA / ...)
  │
  │  加载 libstub_runner.so
  │  调用 platform_register(send_msg, write_mem, read_mem, stop_case)
  │  调用 stub_entry(argc, argv)      ← -E 入口符号
  │  调用 stub_exit()                 ← -X 退出符号
  │
  ▼
stub_entry()
  ├── 解析参数 (--models, --list, --weight-manifest, ...)
  ├── 初始化注册表 (CMake 自动生成的模型表)
  ├── memmap_reset()
  ├── weight_parse()                  ← 仅 --weight-manifest 时
  └── 遍历模型:
        ├── model.setup(&cfg)         ← 注册 memmap 区域（weights + output）
        ├── weight_load_all()         ← 写入权重（memmap 越界检查）
        │   或 weight_load_embedded() ← EMBED_WEIGHTS 编译时打包
        └── model.run(&cfg)           ← 执行模型逻辑
              ├── memmap_read()       ← 读取权重
              ├── memmap_write()      ← 写入计算结果
              ├── result_export()     ← 导出输出到文件
              ├── result_compare()    ← 与 golden 文件比对
              ├── platform_send_msg() ┐ 通过注册的钩子调用平台实现
              └── platform_stop_case()┘
```

- 平台钩子通过 `typedef` 函数指针定义，平台调用 `platform_register()` 逐个传入钩子实现
- 模型注册表由 CMake `configure_file()` 自动生成，新增模型只需建目录 + 写代码 + 声明名字
- `-DSTUB_MODELS="tdd;fdd"` 可在编译时指定默认执行的模型（不影响编译，所有模型始终编译）

## 目录结构

```
stubs/
├── CMakeLists.txt              # 顶层构建配置
├── build.sh                    # 一键构建脚本
├── hooks/                      # 平台钩子（仅 typedef + 注册接口）
│   └── func_sim/
│       └── include/
│           ├── platform_api.h  # 钩子函数指针类型 + platform_register()
│           └── msg_types.h     # 消息类型定义
├── common/                     # 公共控制层
│   ├── include/
│   │   ├── stub_registry.h     # 模型注册表接口（含 setup 函数指针）
│   │   ├── stub_config.h       # 命令行参数解析
│   │   └── stub_log.h          # 日志宏
│   ├── src/
│   │   ├── stub_main.c         # stub_entry() / stub_exit() 入口
│   │   ├── stub_platform.c     # platform_register() + 钩子包装函数
│   │   ├── stub_registry.c     # 注册表实现
│   │   └── stub_config.c       # 参数解析实现
│   ├── memmap/                 # 内存区域映射
│   │   ├── stub_memmap.h
│   │   └── stub_memmap.c
│   ├── weight/                 # 权重导入
│   │   ├── stub_weight.h
│   │   └── stub_weight.c
│   └── result/                 # 结果导出 / golden 比对
│       ├── stub_result.h
│       └── stub_result.c
├── models/                     # 模型桩代码
│   ├── CMakeLists.txt          # 自动发现模型 + 生成注册表
│   ├── stub_model_table.c.in   # 注册表模板（CMake configure_file）
│   ├── tdd/
│   │   ├── CMakeLists.txt
│   │   ├── tdd_stub.c
│   │   ├── weights/            # 权重输入（bin 文件 + manifest）
│   │   │   ├── manifest.txt
│   │   │   ├── tdd_conv_w.bin
│   │   │   ├── tdd_conv_b.bin
│   │   │   └── tdd_fc_w.bin
│   │   └── golden/             # 预期输出（golden 比对基准）
│   │       └── tdd_output.bin
│   └── fdd/
│       ├── CMakeLists.txt
│       ├── fdd_stub.c
│       ├── weights/
│       │   ├── manifest.txt
│       │   ├── fdd_filter_w.bin
│       │   ├── fdd_filter_b.bin
│       │   ├── fdd_dfe_w.bin
│       │   └── fdd_dfe_b.bin
│       └── golden/
│           └── fdd_output.bin
├── tests/                      # 单元测试
│   ├── CMakeLists.txt
│   ├── test_common.h           # 轻量测试宏（无外部依赖）
│   ├── test_registry.c
│   ├── test_platform.c
│   ├── test_memmap.c           # memmap 单元测试
│   ├── test_weight.c           # weight 单元测试
│   ├── test_result.c           # result 单元测试
│   ├── test_tdd.c
│   ├── test_fdd.c
│   ├── platform_sim.c          # func_sim 平台模拟（仅测试用）
│   └── func_sim_main.c         # 集成测试入口 main()
└── tools/                      # 构建辅助脚本
    ├── embed_weights.py        # 权重打包：manifest + bin → C 源文件
    ├── swap_endian.py          # 大小端转换
    ├── pad_align.py            # 补零对齐
    └── print_info.py           # 二进制信息打印
```

## 构建

```bash
# 构建全部（.so + UT）
./build.sh

# 仅构建 .so
./build.sh -m app

# 仅构建并运行 UT
./build.sh -m ut

# 指定默认执行模型
./build.sh -m app -s tdd

# 大小端转换 + 8 字节对齐
./build.sh -m app -e -a 8

# 构建时打包权重（EMBED_WEIGHTS）
./build.sh -m app -w weights/manifest.txt -d weights/
```

## 数据流

```
输入（权重 bin 文件）                    输出（结果 bin 文件）
models/<name>/weights/                   --result-dir 指定的目录
├── manifest.txt                         ├── <name>_output.bin
├── conv_w.bin          ──→ setup       └── ...
├── conv_b.bin              注册区域
└── fc_w.bin            ──→ weight_load  ──→  run  ──→  result_export
                            写入 memmap      计算        从 memmap 导出到文件

                                            golden 比对
                                        models/<name>/golden/
                                        └── <name>_output.bin  ← result_compare 对比基准
```

- **输入**：权重 bin 文件放在 `models/<name>/weights/` 目录下，通过 `--weight-manifest` + `--weight-dir` 指定路径，框架自动加载到 memmap 区域
- **输出**：模型在 `run` 中调用 `result_export()` 将计算结果从 memmap 导出到 `--result-dir` 目录
- **比对**：模型在 `run` 中调用 `result_compare()` 与 `--golden-dir` 目录下的基准文件逐字节比对

运行示例：

```bash
# 运行 TDD 模型：加载权重 → 执行 → 导出结果 → golden 比对
./func_sim_main --models tdd \
  --weight-manifest models/tdd/weights/manifest.txt \
  --weight-dir models/tdd/weights/ \
  --result-dir output/ \
  --golden-dir models/tdd/golden/

# 运行 FDD 模型（多组权重 + 大小端转换）
./func_sim_main --models fdd \
  --weight-manifest models/fdd/weights/manifest.txt \
  --weight-dir models/fdd/weights/ \
  --result-dir output/ \
  --golden-dir models/fdd/golden/
```

## 三大机制

### memmap — 内存区域映射

模型通过 `setup` 函数注册自己使用的内存区域，后续所有读写操作经过 memmap 层的越界和权限检查。

```c
// setup 中注册区域
memmap_register("weights", 0x1000, 0x2000, MEM_RW);  // [0x1000, 0x3000)
memmap_register("output",  0x4000, 0x1000, MEM_RW);   // [0x4000, 0x5000)

// run 中读写（自动检查越界 + 权限）
memmap_write(0x1000, data, size);  // OK: 在 "weights" 区域内，且可写
memmap_read(0x4000, buf, size);    // OK: 在 "output" 区域内，且可读
memmap_write(0x9000, data, size);  // 失败: 不在任何已注册区域内
```

错误码：`0`=成功，`-1`=区域表满，`-2`=重叠，`-3`=越界，`-4`=权限不足。

### weight — 权重导入

每个 tensor 对应一个独立的 bin 文件，文件大小自动检测。manifest 支持多组权重，每组指定一个基地址和大小端配置：

```
# 第一组：Conv 层权重，加载到 0x1000，不做端序转换
0x1000  0
conv1_w.bin
conv1_b.bin

# 第二组：FC 层权重，加载到 0x5000，按 4 字节翻转
0x5000  4
fc_w.bin
fc_b.bin
```

解析规则：
- `#` 开头的行和空行被忽略
- **2 个字段** → 组头（`base_addr swap`），开始新的一组
- **1 个字段** → 权重文件名，归属当前组

| 组头字段 | 说明 |
|----------|------|
| `base_addr` | 起始目标地址（十六进制），需在已注册的 memmap 区域内 |
| `swap` | 大小端转换粒度：`0`=不转换，`2`/`4`/`8`=按 N 字节翻转 |

同一组内的 tensor 按顺序紧凑排列：第一个从 `base_addr` 开始，后续紧接前一个结尾。大小端转换在加载时对每个 N 字节的 word 内部做字节序翻转（例如 `swap=4` 时 `DE AD BE EF` → `EF BE AD DE`）。使用 `EMBED_WEIGHTS` 打包时，转换在构建阶段完成。

TDD manifest 示例（单组 3 个 bin）：

```
# TDD model weights
0x1000  0
tdd_conv_w.bin
tdd_conv_b.bin
tdd_fc_w.bin
```

FDD manifest 示例（2 组，第二组 swap=4）：

```
# Group 1: filter weights, no swap
0x2000  0
fdd_filter_w.bin
fdd_filter_b.bin

# Group 2: DFE weights, big-endian to little-endian
0x5000  4
fdd_dfe_w.bin
fdd_dfe_b.bin
```

加载流程：逐组处理 → 组内逐个打开 bin 文件 → 自动获取文件大小 → 分块 4KB 流式读取 → 通过 `memmap_write` 顺序写入。

### result — 结果导出 / golden 比对

模型在 `run` 中计算完成后，通过 result 模块将输出从 memmap 导出到文件，或与预期结果进行比对。

```c
// 导出 output 区域到文件
result_export(0x4000, 1024, "output/tdd_output.bin");

// 与 golden 文件比对
result_mismatch_t mm;
int rc = result_compare(0x4000, 1024, "golden/tdd_output.bin", &mm);
if (rc == 1) {
    // 不匹配：mm.offset=首个差异字节偏移, mm.actual, mm.expected
} else if (rc == 0) {
    // 完全匹配
}
```

模型 stub 中的典型用法（以 TDD 为例）：

```c
int tdd_run(const stub_config_t *cfg)
{
    // 1. 读取权重（框架已通过 weight_load_all 加载）
    uint8_t weights[56];
    memmap_read(0x1000, weights, 56);

    // 2. 计算
    uint8_t output[56];
    for (int i = 0; i < 56; i++)
        output[i] = weights[i] + 1;

    // 3. 写入 output 区域
    memmap_write(0x4000, output, 56);

    // 4. 导出结果到 --result-dir
    if (cfg->result_dir[0] != '\0') {
        char path[512];
        snprintf(path, sizeof(path), "%s/tdd_output.bin", cfg->result_dir);
        result_export(0x4000, 56, path);
    }

    // 5. 与 --golden-dir 下的基准文件比对
    if (cfg->golden_dir[0] != '\0') {
        char golden[512];
        snprintf(golden, sizeof(golden), "%s/tdd_output.bin", cfg->golden_dir);
        result_mismatch_t mm;
        int cmp = result_compare(0x4000, 56, golden, &mm);
        if (cmp == 1) return -1;  // mismatch
    }
    return 0;
}
```

## EMBED_WEIGHTS — 编译时打包权重

当仿真环境不支持文件 I/O 时，可在构建时将权重二进制打包进编译产物，运行时零文件读取。

```bash
# 通过 build.sh
./build.sh -m app -w weights/manifest.txt -d weights/

# 通过 CMake
cmake -B build -S stubs \
  -DEMBED_WEIGHTS=ON \
  -DEMBED_WEIGHT_MANIFEST=weights/manifest.txt \
  -DEMBED_WEIGHT_DIR=weights/
```

构建时 `tools/embed_weights.py` 读取 manifest + 所有 bin 文件，生成一个 C 源文件，其中每个 tensor 的数据以 `static const unsigned char[]` 数组形式内联。运行时 `weight_load_embedded()` 直接从数组写入 memmap，无需 `--weight-manifest` 参数。

默认关闭（`EMBED_WEIGHTS=OFF`），避免二进制过大。

## CLI 参数

| 参数 | 说明 |
|------|------|
| `--models a,b,...` | 指定要运行的模型（逗号分隔） |
| `--list` | 列出所有已注册模型 |
| `--weight-manifest <path>` | 权重 manifest 文件路径 |
| `--weight-dir <dir>` | 权重 bin 文件的基础目录 |
| `--result-dir <dir>` | 结果导出目录 |
| `--golden-dir <dir>` | golden 比对文件目录 |
| `--help` | 显示帮助 |

## 新增模型

1. 创建目录 `models/<name>/`
2. 编写 `<name>_stub.c`，实现 `setup` 和 `run` 函数：
   ```c
   #include "stub_config.h"
   #include "stub_memmap.h"
   #include "stub_result.h"
   #include "platform_api.h"

   int <name>_setup(const stub_config_t *cfg)
   {
       (void)cfg;
       memmap_register("<name>_weights", 0x1000, 0x2000, MEM_RW);
       memmap_register("<name>_output",  0x4000, 0x1000, MEM_RW);
       return 0;
   }

   int <name>_run(const stub_config_t *cfg)
   {
       // 1. memmap_read 读取权重（已由框架加载）
       // 2. 计算
       // 3. memmap_write 写入输出
       // 4. result_export 导出结果（可选）
       // 5. result_compare 比对 golden（可选）
       return 0;
   }
   ```
3. 准备权重文件：
   ```
   models/<name>/weights/
   ├── manifest.txt      # 组头 + 文件列表
   ├── layer1_w.bin
   ├── layer1_b.bin
   └── ...
   ```
4. 准备 golden 文件（可选）：
   ```
   models/<name>/golden/
   └── <name>_output.bin
   ```
5. 编写 `CMakeLists.txt`：
   ```cmake
   set(MODEL_NAME     "<name>" PARENT_SCOPE)
   set(MODEL_RUN_FN   "<name>_run" PARENT_SCOPE)
   set(MODEL_SETUP_FN "<name>_setup" PARENT_SCOPE)  # 可选，不需要 setup 则不设置
   ```
6. 重新 cmake configure，注册表自动更新
