# AI芯片应用侧验证框架

## 一、项目定位

### 核心需求
- **应用侧验证**：验证AI加速器的软件/驱动层
- **验证代码用C**：手写测试框架，纯C实现
- **工具链二进制下载**：仿真器、编译器、OS平台软件通过URL下载
- **辅助工具用Python**：CI/CD、网页监控、报告生成等
- **跨平台复用**：同一套测试代码在不同平台运行

### 工具链获取方式
```
┌─────────────────────────────────────────────────────────────┐
│                    外部工具（二进制下载）                     │
├─────────────────────────────────────────────────────────────┤
│  仿真器 (Simulator)    → curl下载二进制                      │
│  编译器 (Compiler)     → curl下载二进制                      │
│  OS平台软件            → curl下载二进制                      │
│  FPGA工具链            → curl下载二进制                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    本仓库代码                                │
├─────────────────────────────────────────────────────────────┤
│  验证代码 (C语言)      → 测试框架、测试用例、参考模型、HAL    │
│  辅助工具 (Python)     → CI/CD、监控、报告、数据生成          │
│  配置文件              → 工具下载地址、平台配置、测试列表      │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、核心架构

### 2.1 验证流水线阶段

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         验证流水线 Pipeline                              │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────────────┤
│  LinuxUT    │  LinuxST    │  功能验证    │  性能ESL    │   原型/EDA      │
│  单元测试   │  系统测试    │             │             │                 │
├─────────────┴─────────────┼─────────────┼─────────────┼─────────────────┤
│     Host Linux环境        │  Simulator  │  ESL Model  │  FPGA/Chip      │
│     快速迭代开发          │  功能正确性  │  性能建模   │  真实硬件        │
└───────────────────────────┴─────────────┴─────────────┴─────────────────┘
         ↓                        ↓              ↓              ↓
    代码逻辑验证              算子功能验证    性能指标验证    硬件实测验证
```

### 2.2 各阶段定位

| 阶段 | 环境 | 目标 | 速度 | 精度 |
|------|------|------|------|------|
| **LinuxUT** | Host Linux | 驱动/框架单元测试 | 最快 | 逻辑级 |
| **LinuxST** | Host Linux | 系统集成测试 | 快 | 逻辑级 |
| **功能验证** | Simulator | 算子功能正确性 | 中 | 位精确 |
| **性能ESL** | ESL Model | 性能/时序/带宽 | 中 | 周期级 |
| **原型/EDA** | FPGA/Chip | 真实硬件验证 | 慢 | 硬件级 |

### 2.3 分层架构

```
┌─────────────────────────────────────────────────────────────┐
│              Python辅助层 (CI/CD, 监控, 报告)                │
├──────────────────────────┬──────────────────────────────────┤
│   Python参考模型         │        C测试用例                  │
│   (numpy, 非torch)       │   UT/ST/功能/性能测试             │
│   matmul/conv/act/quant  │                                  │
├──────────────────────────┴──────────────────────────────────┤
│                  Test Framework (C测试框架)                  │
│    断言宏 | 测试注册 | 结果统计 | 日志输出 | 超时控制          │
├─────────────────────────────────────────────────────────────┤
│           Reference Model (C参考模型，可选用于嵌入式)         │
│          矩阵运算 | 卷积 | 激活函数 | 量化模型                 │
├─────────────────────────────────────────────────────────────┤
│                  Driver Layer (C驱动层)                      │
│          算子接口 | 内存分配 | 任务提交 | 状态查询              │
├─────────────────────────────────────────────────────────────┤
│                  HAL Layer (C硬件抽象层)                     │
│        reg_read/write | dma_transfer | irq_wait | mem_map   │
├────────────┬────────────┬────────────┬────────────┬─────────┤
│  LinuxUT   │  LinuxST   │ Simulator  │    ESL     │FPGA/Chip│
│  Mock HAL  │  Mock HAL  │  Sim HAL   │  ESL HAL   │ HW HAL  │
└────────────┴────────────┴────────────┴────────────┴─────────┘
                              ↓
              外部二进制工具 (通过URL下载)
         Simulator | Compiler | ESL Model | FPGA Tools
```

### 2.4 参考模型双实现策略

```
参考模型使用场景:

┌─────────────────────────────────────────────────────────────┐
│  Python参考模型 (pymodel/)                                  │
│  ├─ 用于：生成golden数据、离线精度验证、算法开发              │
│  ├─ 依赖：numpy (不依赖torch/tensorflow)                    │
│  └─ 优势：开发快、易调试、易对接算法团队                      │
├─────────────────────────────────────────────────────────────┤
│  C参考模型 (src/model/)                                     │
│  ├─ 用于：嵌入式在线比对、芯片端自测                          │
│  ├─ 依赖：无外部依赖，纯C实现                                │
│  └─ 优势：可编译到目标平台、实时比对                          │
└─────────────────────────────────────────────────────────────┘

验证流程:
  1. Python模型生成golden数据 → 保存到testdata/golden/
  2. C测试加载golden数据 → 与硬件输出比对
  或
  3. C测试调用C参考模型 → 实时计算参考值 → 与硬件比对
```

---

## 三、完整目录结构

```
aitestframework/
│
├── README.md                           # 项目说明
├── Makefile                            # 顶层Makefile
├── setup.py                            # Python包安装
├── requirements.txt                    # Python依赖
│
├── docs/                               # 文档
│   ├── architecture.md                 # 架构设计
│   ├── hal_porting_guide.md            # HAL移植指南
│   ├── test_writing_guide.md           # 测试编写指南
│   ├── toolchain_setup.md              # 工具链配置指南
│   └── coding_style.md                 # C编码规范
│
├── deps/                               # 依赖管理
│   ├── README.md                       # 依赖管理说明
│   ├── manifests/                      # 依赖清单
│   │   ├── simulator.yaml
│   │   ├── compiler.yaml
│   │   ├── esl.yaml
│   │   ├── fpga_tools.yaml
│   │   └── os_platform.yaml
│   ├── compatibility/                  # 配套关系定义
│   │   ├── matrix.yaml
│   │   ├── constraints.yaml
│   │   └── known_issues.yaml
│   ├── profiles/                       # 预定义配套组合
│   │   ├── latest.yaml
│   │   ├── stable.yaml
│   │   ├── v1.0.yaml
│   │   └── v2.0.yaml
│   ├── lock/                           # 锁定文件
│   │   └── deps.lock.yaml
│   └── scripts/                        # 依赖管理脚本
│       ├── dep_manager.py
│       ├── compat_checker.py
│       └── version_resolver.py
│
├── configs/                            # 配置文件
│   ├── platforms/                      # 平台配置
│   │   ├── simulator.yaml
│   │   ├── fpga.yaml
│   │   └── chip.yaml
│   └── testlists/                      # 测试列表
│       ├── sanity.yaml
│       ├── nightly.yaml
│       └── full.yaml
│
├── include/                            # C头文件
│   ├── common/                         # 通用定义
│   │   ├── types.h
│   │   ├── error_code.h
│   │   ├── config.h
│   │   └── utils.h
│   ├── hal/                            # HAL接口
│   │   ├── hal_types.h
│   │   ├── hal_ops.h
│   │   └── hal_platform.h
│   ├── driver/                         # 驱动接口
│   │   ├── npu_driver.h
│   │   ├── npu_ops.h
│   │   └── npu_memory.h
│   ├── model/                          # 参考模型接口(C)
│   │   ├── ref_model.h
│   │   ├── matmul.h
│   │   ├── conv2d.h
│   │   └── activation.h
│   └── framework/                      # 测试框架接口
│       ├── test_framework.h
│       ├── test_assert.h
│       ├── test_runner.h
│       └── test_report.h
│
├── src/                                # C源代码
│   ├── common/                         # 通用实现
│   │   ├── utils.c
│   │   └── log.c
│   ├── hal/                            # HAL核心
│   │   ├── hal_core.c
│   │   └── hal_init.c
│   ├── driver/                         # 驱动实现
│   │   ├── npu_driver.c
│   │   ├── npu_ops.c
│   │   └── npu_memory.c
│   ├── model/                          # 参考模型(C实现)
│   │   ├── ref_matmul.c
│   │   ├── ref_conv2d.c
│   │   ├── ref_activation.c
│   │   ├── ref_pooling.c
│   │   └── quantize/
│   │       ├── quant_int8.c
│   │       ├── quant_fp16.c
│   │       └── quant_bf16.c
│   └── framework/                      # 测试框架
│       ├── test_core.c
│       ├── test_runner.c
│       ├── test_filter.c
│       └── test_report.c
│
├── platform/                           # 平台HAL实现
│   ├── common/
│   │   ├── platform.h
│   │   └── mock_hal.h
│   ├── linux_ut/                       # LinuxUT平台
│   │   ├── ut_hal.c
│   │   ├── ut_hal.h
│   │   ├── ut_mock_reg.c
│   │   ├── ut_mock_mem.c
│   │   └── ut_main.c
│   ├── linux_st/                       # LinuxST平台
│   │   ├── st_hal.c
│   │   ├── st_hal.h
│   │   ├── st_stub_driver.c
│   │   └── st_main.c
│   ├── simulator/                      # Simulator平台
│   │   ├── sim_hal.c
│   │   ├── sim_hal.h
│   │   ├── sim_comm.c
│   │   └── sim_main.c
│   ├── esl/                            # ESL平台
│   │   ├── esl_hal.c
│   │   ├── esl_hal.h
│   │   ├── esl_perf_model.c
│   │   ├── esl_trace.c
│   │   └── esl_main.c
│   ├── fpga/                           # FPGA平台
│   │   ├── fpga_hal.c
│   │   ├── fpga_hal.h
│   │   ├── fpga_pcie.c
│   │   ├── fpga_dma.c
│   │   └── fpga_main.c
│   └── chip/                           # Chip平台
│       ├── chip_hal.c
│       ├── chip_hal.h
│       ├── chip_mmap.c
│       ├── chip_irq.c
│       └── chip_main.c
│
├── tests/                              # C测试用例
│   ├── unit/                           # 单元测试
│   │   ├── driver/
│   │   ├── framework/
│   │   └── model/
│   ├── integration/                    # 集成测试
│   ├── functional/                     # 功能测试
│   │   ├── sanity/
│   │   ├── matmul/
│   │   ├── conv/
│   │   ├── activation/
│   │   └── precision/
│   ├── performance/                    # 性能测试
│   ├── stress/                         # 压力测试
│   ├── e2e/                            # 端到端测试
│   └── testcfg/                        # 测试配置
│       ├── platform_mapping.yaml
│       ├── test_execution.yaml
│       └── *.yaml
│
├── pymodel/                            # Python参考模型
│   ├── __init__.py
│   ├── ops/                            # 算子实现
│   ├── quantize/                       # 量化工具
│   ├── layers/                         # 复合层
│   └── utils/                          # 工具函数
│
├── testdata/                           # 测试数据
│   ├── generators/                     # 数据生成器
│   ├── inputs/
│   ├── golden/
│   └── models/
│
├── tools/                              # Python工具
│   ├── toolchain/                      # 工具链管理
│   ├── runner/                         # 测试运行
│   ├── report/                         # 报告生成
│   ├── testmgmt/                       # 用例管理系统
│   ├── archive/                        # 结果归档
│   └── data/                           # 数据工具
│
├── cicd/                               # CI/CD
│   ├── pipelines/                      # 流水线定义
│   ├── jobs/                           # 作业定义
│   ├── jenkins/                        # Jenkins配置
│   ├── gitlab/                         # GitLab CI
│   └── github/                         # GitHub Actions
│
├── scripts/                            # 脚本
│   ├── setup_env.sh
│   ├── download_toolchain.py
│   ├── build.sh
│   ├── run_tests.py
│   └── clean.sh
│
└── build/                              # 构建输出 (git忽略)
    ├── toolchain/
    ├── bin/
    └── reports/
```

---

## 四、实践步骤

### 第一阶段：基础框架搭建

#### 步骤1.1：创建目录结构和配置文件
- [ ] 创建完整目录结构
- [ ] 创建 `.gitignore`
- [ ] 创建顶层 `Makefile`
- [ ] 创建 `setup.py` 和 `requirements.txt`

#### 步骤1.2：实现C头文件 (include/)
- [ ] `include/common/types.h` - 基础类型定义
- [ ] `include/common/error_code.h` - 错误码定义
- [ ] `include/common/config.h` - 编译配置
- [ ] `include/common/utils.h` - 工具宏
- [ ] `include/hal/hal_types.h` - HAL类型
- [ ] `include/hal/hal_ops.h` - HAL操作接口（函数指针结构体）
- [ ] `include/hal/hal_platform.h` - 平台选择
- [ ] `include/driver/npu_driver.h` - NPU驱动接口
- [ ] `include/driver/npu_ops.h` - 算子接口
- [ ] `include/driver/npu_memory.h` - 内存管理
- [ ] `include/model/ref_model.h` - 参考模型接口
- [ ] `include/model/matmul.h` - 矩阵乘法
- [ ] `include/model/conv2d.h` - 卷积
- [ ] `include/model/activation.h` - 激活函数
- [ ] `include/framework/test_framework.h` - 测试框架核心
- [ ] `include/framework/test_assert.h` - 断言宏
- [ ] `include/framework/test_runner.h` - 运行器
- [ ] `include/framework/test_report.h` - 报告接口

#### 步骤1.3：实现C源文件 (src/)
- [ ] `src/common/utils.c` - 通用工具
- [ ] `src/common/log.c` - 日志系统
- [ ] `src/hal/hal_core.c` - HAL核心实现
- [ ] `src/hal/hal_init.c` - HAL初始化
- [ ] `src/driver/npu_driver.c` - 驱动实现
- [ ] `src/driver/npu_ops.c` - 算子实现
- [ ] `src/driver/npu_memory.c` - 内存管理实现
- [ ] `src/model/ref_matmul.c` - 矩阵乘法参考实现
- [ ] `src/model/ref_conv2d.c` - 卷积参考实现
- [ ] `src/model/ref_activation.c` - 激活函数参考实现
- [ ] `src/model/ref_pooling.c` - 池化参考实现
- [ ] `src/model/quantize/quant_int8.c` - INT8量化
- [ ] `src/model/quantize/quant_fp16.c` - FP16量化
- [ ] `src/model/quantize/quant_bf16.c` - BF16量化
- [ ] `src/framework/test_core.c` - 测试核心
- [ ] `src/framework/test_runner.c` - 测试运行器
- [ ] `src/framework/test_filter.c` - 测试过滤
- [ ] `src/framework/test_report.c` - 报告生成

### 第二阶段：平台HAL实现

#### 步骤2.1：公共平台代码
- [ ] `platform/common/platform.h` - 平台公共接口
- [ ] `platform/common/mock_hal.h` - Mock HAL基础

#### 步骤2.2：LinuxUT平台（单元测试Mock）
- [ ] `platform/linux_ut/ut_hal.h`
- [ ] `platform/linux_ut/ut_hal.c`
- [ ] `platform/linux_ut/ut_mock_reg.c` - 寄存器Mock
- [ ] `platform/linux_ut/ut_mock_mem.c` - 内存Mock
- [ ] `platform/linux_ut/ut_main.c` - 入口

#### 步骤2.3：LinuxST平台（系统测试桩）
- [ ] `platform/linux_st/st_hal.h`
- [ ] `platform/linux_st/st_hal.c`
- [ ] `platform/linux_st/st_stub_driver.c` - 桩驱动
- [ ] `platform/linux_st/st_main.c`

#### 步骤2.4：Simulator平台
- [ ] `platform/simulator/sim_hal.h`
- [ ] `platform/simulator/sim_hal.c`
- [ ] `platform/simulator/sim_comm.c` - 仿真器通信
- [ ] `platform/simulator/sim_main.c`

#### 步骤2.5：ESL平台
- [ ] `platform/esl/esl_hal.h`
- [ ] `platform/esl/esl_hal.c`
- [ ] `platform/esl/esl_perf_model.c` - 性能模型
- [ ] `platform/esl/esl_trace.c` - Trace收集
- [ ] `platform/esl/esl_main.c`

#### 步骤2.6：FPGA平台
- [ ] `platform/fpga/fpga_hal.h`
- [ ] `platform/fpga/fpga_hal.c`
- [ ] `platform/fpga/fpga_pcie.c` - PCIe通信
- [ ] `platform/fpga/fpga_dma.c` - DMA
- [ ] `platform/fpga/fpga_main.c`

#### 步骤2.7：Chip平台
- [ ] `platform/chip/chip_hal.h`
- [ ] `platform/chip/chip_hal.c`
- [ ] `platform/chip/chip_mmap.c` - 内存映射
- [ ] `platform/chip/chip_irq.c` - 中断处理
- [ ] `platform/chip/chip_main.c`

### 第三阶段：测试用例

#### 步骤3.1：单元测试
- [ ] `tests/unit/driver/test_npu_init.c`
- [ ] `tests/unit/driver/test_mem_mgmt.c`
- [ ] `tests/unit/driver/test_task_queue.c`
- [ ] `tests/unit/framework/test_hal_interface.c`
- [ ] `tests/unit/framework/test_error_handling.c`
- [ ] `tests/unit/model/test_ref_matmul.c`
- [ ] `tests/unit/model/test_ref_conv.c`

#### 步骤3.2：集成测试
- [ ] `tests/integration/test_driver_stack.c`
- [ ] `tests/integration/test_multi_task.c`
- [ ] `tests/integration/test_api.c`

#### 步骤3.3：功能测试
- [ ] `tests/functional/sanity/test_reg_rw.c`
- [ ] `tests/functional/sanity/test_mem_alloc.c`
- [ ] `tests/functional/sanity/test_basic_op.c`
- [ ] `tests/functional/matmul/test_matmul_basic.c`
- [ ] `tests/functional/matmul/test_matmul_shapes.c`
- [ ] `tests/functional/conv/test_conv2d_basic.c`
- [ ] `tests/functional/activation/test_relu.c`
- [ ] `tests/functional/activation/test_gelu.c`
- [ ] `tests/functional/precision/test_fp16_precision.c`
- [ ] `tests/functional/precision/test_int8_quant.c`

#### 步骤3.4：性能/压力/E2E测试
- [ ] `tests/performance/test_matmul_perf.c`
- [ ] `tests/performance/test_latency.c`
- [ ] `tests/stress/test_long_run.c`
- [ ] `tests/e2e/test_resnet_layer.c`

#### 步骤3.5：测试配置文件
- [ ] `tests/testcfg/platform_mapping.yaml`
- [ ] `tests/testcfg/test_execution.yaml`
- [ ] `tests/testcfg/linux_ut.yaml`
- [ ] `tests/testcfg/linux_st.yaml`
- [ ] `tests/testcfg/functional.yaml`

### 第四阶段：Python模块

#### 步骤4.1：Python参考模型 (pymodel/)
- [ ] `pymodel/__init__.py`
- [ ] `pymodel/ops/__init__.py`
- [ ] `pymodel/ops/matmul.py`
- [ ] `pymodel/ops/conv2d.py`
- [ ] `pymodel/ops/activation.py`
- [ ] `pymodel/ops/pooling.py`
- [ ] `pymodel/quantize/__init__.py`
- [ ] `pymodel/quantize/int8_quant.py`
- [ ] `pymodel/quantize/fp16_utils.py`
- [ ] `pymodel/quantize/bf16_utils.py`
- [ ] `pymodel/layers/__init__.py`
- [ ] `pymodel/layers/linear.py`
- [ ] `pymodel/layers/conv_layer.py`
- [ ] `pymodel/layers/attention.py`
- [ ] `pymodel/utils/__init__.py`
- [ ] `pymodel/utils/tensor_utils.py`
- [ ] `pymodel/utils/compare.py`

#### 步骤4.2：依赖管理 (deps/)
- [ ] `deps/README.md`
- [ ] `deps/manifests/simulator.yaml`
- [ ] `deps/manifests/compiler.yaml`
- [ ] `deps/manifests/esl.yaml`
- [ ] `deps/manifests/fpga_tools.yaml`
- [ ] `deps/manifests/os_platform.yaml`
- [ ] `deps/compatibility/matrix.yaml`
- [ ] `deps/compatibility/constraints.yaml`
- [ ] `deps/profiles/stable.yaml`
- [ ] `deps/profiles/latest.yaml`
- [ ] `deps/scripts/dep_manager.py`
- [ ] `deps/scripts/compat_checker.py`
- [ ] `deps/scripts/version_resolver.py`

#### 步骤4.3：Python工具 (tools/)
- [ ] `tools/__init__.py`
- [ ] `tools/toolchain/downloader.py`
- [ ] `tools/toolchain/installer.py`
- [ ] `tools/runner/test_runner.py`
- [ ] `tools/runner/parallel_runner.py`
- [ ] `tools/runner/result_parser.py`
- [ ] `tools/report/html_report.py`
- [ ] `tools/report/json_report.py`
- [ ] `tools/report/templates/report.html`
- [ ] `tools/testmgmt/server.py`
- [ ] `tools/testmgmt/database.py`
- [ ] `tools/testmgmt/models.py`
- [ ] `tools/testmgmt/templates/*.html`
- [ ] `tools/archive/archiver.py`
- [ ] `tools/archive/storage.py`

#### 步骤4.4：测试数据生成器
- [ ] `testdata/generators/gen_matmul_data.py`
- [ ] `testdata/generators/gen_conv_data.py`
- [ ] `testdata/generators/gen_random_tensor.py`

### 第五阶段：CI/CD

#### 步骤5.1：流水线定义
- [ ] `cicd/pipelines/sanity_pipeline.py`
- [ ] `cicd/pipelines/nightly_pipeline.py`
- [ ] `cicd/pipelines/release_pipeline.py`

#### 步骤5.2：作业定义
- [ ] `cicd/jobs/build_job.py`
- [ ] `cicd/jobs/test_job.py`
- [ ] `cicd/jobs/report_job.py`

#### 步骤5.3：CI配置
- [ ] `cicd/jenkins/Jenkinsfile`
- [ ] `cicd/gitlab/.gitlab-ci.yml`
- [ ] `cicd/github/workflows/sanity.yml`
- [ ] `cicd/github/workflows/nightly.yml`

### 第六阶段：脚本和文档

#### 步骤6.1：脚本
- [ ] `scripts/setup_env.sh`
- [ ] `scripts/download_toolchain.py`
- [ ] `scripts/build.sh`
- [ ] `scripts/run_tests.py`
- [ ] `scripts/clean.sh`

#### 步骤6.2：配置文件
- [ ] `configs/platforms/simulator.yaml`
- [ ] `configs/platforms/fpga.yaml`
- [ ] `configs/platforms/chip.yaml`
- [ ] `configs/testlists/sanity.yaml`
- [ ] `configs/testlists/nightly.yaml`
- [ ] `configs/testlists/full.yaml`

#### 步骤6.3：文档
- [ ] `docs/architecture.md`
- [ ] `docs/pipeline_stages.md`
- [ ] `docs/hal_porting_guide.md`
- [ ] `docs/test_writing_guide.md`
- [ ] `docs/dependency_management.md`
- [ ] `docs/testmgmt_guide.md`
- [ ] `docs/coding_style.md`

---

## 五、关键设计细节

### 5.1 HAL层设计（函数指针多态）

```c
/* include/hal/hal_ops.h */

typedef struct hal_ops {
    /* 寄存器操作 */
    int (*reg_write)(uint32_t addr, uint32_t value);
    uint32_t (*reg_read)(uint32_t addr);

    /* DMA操作 */
    int (*dma_transfer)(uint64_t src, uint64_t dst, size_t size, int dir);
    int (*dma_wait)(int channel, uint32_t timeout_ms);

    /* 中断操作 */
    int (*irq_wait)(uint32_t irq_mask, uint32_t timeout_ms);
    int (*irq_clear)(uint32_t irq_mask);

    /* 内存操作 */
    void* (*mem_alloc)(size_t size, uint32_t flags);
    void  (*mem_free)(void *ptr);
    uint64_t (*mem_virt_to_phys)(void *virt_addr);

    /* 平台控制 */
    int (*platform_init)(void);
    int (*platform_deinit)(void);
    void (*delay_us)(uint32_t us);
    uint64_t (*get_time_us)(void);

    /* 日志 */
    void (*log_print)(int level, const char *fmt, ...);
} hal_ops_t;

/* 全局HAL实例 */
extern hal_ops_t *g_hal;

/* 便捷宏 */
#define HAL_REG_WRITE(addr, val)   g_hal->reg_write(addr, val)
#define HAL_REG_READ(addr)         g_hal->reg_read(addr)
```

### 5.2 测试框架核心

```c
/* include/framework/test_framework.h */

typedef enum {
    TEST_PASS = 0,
    TEST_FAIL = 1,
    TEST_SKIP = 2,
    TEST_TIMEOUT = 3
} test_result_t;

typedef struct test_case {
    const char *name;
    const char *suite;
    test_result_t (*func)(void);
    uint32_t timeout_ms;
    const char *tags;
} test_case_t;

/* 测试注册宏 - 使用linker section */
#define TEST_CASE(suite, name) \
    static test_result_t test_##suite##_##name(void); \
    static test_case_t __test_##suite##_##name \
        __attribute__((used, section("test_cases"))) = { \
        .name = #name, \
        .suite = #suite, \
        .func = test_##suite##_##name, \
        .timeout_ms = 5000 \
    }; \
    static test_result_t test_##suite##_##name(void)

/* 断言宏 */
#define TEST_ASSERT(cond) \
    do { if (!(cond)) { test_fail(__FILE__, __LINE__, #cond); return TEST_FAIL; } } while(0)

#define TEST_ASSERT_EQ(a, b)           TEST_ASSERT((a) == (b))
#define TEST_ASSERT_NEAR(a, b, eps)    TEST_ASSERT(fabs((a)-(b)) < (eps))
```

### 5.3 测试用例跨平台复用

```yaml
# tests/testcfg/platform_mapping.yaml

default_platforms: [linux_ut, linux_st, simulator, esl, fpga, chip]

testcases:
  "performance/*":
    platforms: [esl, fpga, chip]
    skip_reason: "性能测试需要真实时序"

  "performance/test_power.c":
    platforms: [chip]
    skip_reason: "功耗测试只能在实际芯片"

stages:
  linux_ut:
    platforms: [linux_ut]
    includes:
      - "unit/*"
      - "functional/sanity/*"

  functional:
    platforms: [simulator]
    includes:
      - "functional/*"
      - "e2e/*"
```

---

## 六、快速开始

### 环境准备

```bash
# 克隆仓库
git clone <repo_url>
cd aitestframework

# 安装Python依赖
pip install -r requirements.txt

# 安装工具链（稳定版）
python deps/scripts/dep_manager.py install --profile stable
```

### 构建

```bash
# 构建LinuxUT版本
make PLATFORM=linux_ut

# 构建Simulator版本
make PLATFORM=simulator

# 构建所有平台
make all
```

### 运行测试

```bash
# 运行LinuxUT单元测试
./build/bin/linux_ut/test_runner

# 运行指定用例
./build/bin/linux_ut/test_runner --filter "matmul*"

# 使用Python运行器
python scripts/run_tests.py --platform linux_ut --list sanity
```

### 生成报告

```bash
python -m tools.report.html_report --input build/reports/latest.json --output report.html
```

---

## 七、验证要点

1. **验证流水线四阶段**：LinuxUT/ST → 功能验证 → 性能ESL → 原型/EDA
2. **测试用例跨平台复用**：用例按功能分类，通过配置映射到平台
3. **测试执行可配置**：不同用例可配置不同的桩(stubs)、启动流程(setup)、结果检查器(checker)
4. **依赖管理独立**：多张manifest + 配套矩阵 + lock锁定
5. **参考模型双实现**：Python(numpy)离线golden + C嵌入式在线比对
6. **HAL多平台隔离**：linux_ut/linux_st/simulator/esl/fpga/chip
7. **用例管理系统**：SQLite存储 + Flask简洁Web界面
8. **结果归档**：按日期归档，支持保留策略

---

## 八、当前进度

### 已完成
- [x] README.md（本文档）

### 待实现（按优先级）
1. 目录结构和配置文件
2. C头文件 (include/)
3. C源文件 (src/)
4. 平台HAL实现 (platform/)
5. C测试用例 (tests/)
6. Python参考模型 (pymodel/)
7. 依赖管理系统 (deps/)
8. Python工具 (tools/)
9. CI/CD (cicd/)
10. 脚本和文档

---

## 九、联系方式

如有问题，请联系项目维护者。
