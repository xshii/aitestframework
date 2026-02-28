# REQ-1 桩代码管理

---
version: 2.1
date: 2026-02-27
status: draft
---

## 1. 概述

管理 CPU 端发送给仿真器/FPGA 的 C 语言激励代码（桩代码）。CPU 作为主控方，通过调用平台提供的钩子函数（hook）与被测硬件交互。

**桩代码结构：**
- **平台 hooks** — 各平台提供的对接打桩函数接口（头文件 + 桩实现库），按平台区分，当前仅有功能仿真平台
- **公共控制层** — 统一入口、模型注册表、运行配置解析、调度逻辑
- **模型桩代码** — 各模型的业务激励逻辑（C 语言），注册到公共控制层，由其调度执行

**运行时流程：**

```
stub_main 启动
    │
    ├─ 读取运行配置（选择了哪些模型）
    ├─ 从 registry 查找选中模型的注册项
    ├─ 依次调度执行选中模型的桩逻辑
    │       │
    │       │ 调用 platform_api.h 中的 hook 函数
    │       ▼
    │   平台 hooks (func_sim)
    │
    └─ 收集结果，退出
```

**架构原则：** 所有桩代码管理逻辑实现在 Core API 层（`aitestframework.stubs`），CLI 和 Web 均为薄壳调用层。

## 2. 需求详情

### REQ-1.1 目录组织 [P0]

**描述：** 平台 hooks 按平台分目录，公共控制层和模型桩代码分开组织。

**目录结构：**

```
stubs/
├── hooks/                              # 平台提供的钩子函数接口
│   └── func_sim/                       # 功能仿真平台（当前唯一平台）
│       ├── include/
│       │   ├── platform_api.h          # 平台对外接口头文件
│       │   ├── msg_types.h             # 消息类型定义
│       │   └── ...
│       └── lib/                        # 平台提供的桩实现库（预编译）
│           └── libplatform_sim.a
├── common/                             # 公共控制层
│   ├── CMakeLists.txt
│   ├── stub_main.c                     # 统一入口
│   ├── stub_registry.h                 # 模型注册宏/接口
│   ├── stub_registry.c                 # 注册表实现
│   └── stub_config.c                   # 运行配置解析（选哪些模型跑）
├── models/                             # 模型级业务桩代码
│   ├── resnet/
│   │   ├── CMakeLists.txt
│   │   └── resnet_stub.c              # 实现 resnet_run()，CMakeLists.txt 声明注册信息
│   ├── bert/
│   │   ├── CMakeLists.txt
│   │   └── bert_stub.c               # 实现 bert_run()，CMakeLists.txt 声明注册信息
│   └── yolo/
│       └── ...
└── CMakeLists.txt                      # 顶层CMake
```

**模型注册机制（CMake 自动生成注册表）：**

每个模型子目录的 `CMakeLists.txt` 声明 `MODEL_NAME` 和 `MODEL_RUN_FN`，父级 `models/CMakeLists.txt` 收集后通过 `configure_file()` 自动生成 `stub_model_table.c`：

```c
// 自动生成的 stub_model_table.c
#include "stub_registry.h"
extern int resnet_run(const stub_config_t *cfg);
extern int bert_run(const stub_config_t *cfg);

const model_entry_t g_model_table[] = {
    {"resnet", resnet_run},
    {"bert",   bert_run},
};
const int g_model_table_count = 2;
```

新增模型只需：建目录 + 写桩代码 + 写 `CMakeLists.txt` 声明名字和函数名 + 在 `models/CMakeLists.txt` 加 `add_subdirectory`。

**运行配置示例：**

```json
// 通过命令行参数或配置文件指定
{
  "models": ["resnet", "bert"],
  "platform": "func_sim"
}
```

**编译关系：**

```
模型桩代码 (resnet_stub.c) ──┐
                              ├── 链接 ──→ common (stub_main + registry)
模型桩代码 (bert_stub.c)  ──┘                    │
                                                  │ 链接 hooks
                                                  ▼
                                    平台库 (libplatform_sim.a)
                                                  │
                                                  ▼
                                    最终产物: stub_runner（单一可执行文件）
```

**验收标准：**
- hooks 目录按平台区分，模型桩代码按模型组织
- 公共控制层提供注册、配置解析、调度能力
- 新增模型只需新建目录、实现 `model_run_fn`、在 `CMakeLists.txt` 声明 `MODEL_NAME`/`MODEL_RUN_FN`
- 通过运行配置选择一个或多个模型一起跑，无需重新编译
- 新增平台只需在 `hooks/` 下新建目录
- 目录命名规范：小写字母+下划线

### REQ-1.2 版本管理 [P0]

**描述：** 桩代码随仓库整体版本管理，关键里程碑打 git tag。

**规则：**
- 版本号遵循 SemVer：`vMAJOR.MINOR.PATCH`
- 通过 `git tag` 标记发布版本
- 支持 `git checkout <tag>` 检出指定版本的桩代码

**验收标准：**
- `git log --oneline -- stubs/` 可查看桩代码变更历史
- 任意历史 tag 可检出并编译

### REQ-1.3 编译构建 [P0]

**描述：** 提供 CMake 构建脚本，编译时选择对接的平台 hooks。支持两种构建目标：

- **app** — 编译为 `stub_runner` 可执行文件（公共控制层 + 全部模型桩代码 + hooks），运行时通过配置选择模型
- **ut** — 编译桩代码单元测试

**技术方案：**

```cmake
# stubs/CMakeLists.txt
cmake_minimum_required(VERSION 3.16)
project(aitestframework_stubs C)        # 纯 C 项目

# 平台选择（决定链接哪套 hooks）
set(HOOK_PLATFORM "func_sim" CACHE STRING "Hook platform: func_sim")

# 构建目标：app / ut / all
set(STUB_BUILD_TARGET "all" CACHE STRING "Build target: app|ut|all")

# hooks 头文件和库路径
include_directories(hooks/${HOOK_PLATFORM}/include)
link_directories(hooks/${HOOK_PLATFORM}/lib)

add_subdirectory(common)
add_subdirectory(models)
```

**构建命令示例：**

```bash
# 构建 stub_runner（对接功能仿真平台）
cmake -B build/func_sim -S stubs -DHOOK_PLATFORM=func_sim -DSTUB_BUILD_TARGET=app
cmake --build build/func_sim

# 运行（选择模型）
./build/func_sim/stub_runner --models resnet,bert
./build/func_sim/stub_runner --config run_config.json

# 构建并运行桩UT
cmake -B build/func_sim -S stubs -DHOOK_PLATFORM=func_sim -DSTUB_BUILD_TARGET=ut
cmake --build build/func_sim
ctest --test-dir build/func_sim

# 交叉编译（指定工具链）
cmake -B build/func_sim -S stubs \
  -DCMAKE_TOOLCHAIN_FILE=build/toolchain/cross.cmake \
  -DHOOK_PLATFORM=func_sim
cmake --build build/func_sim
```

**验收标准：**
- app 目标编译出单一 `stub_runner` 可执行文件，包含全部已注册模型
- `stub_runner --models X,Y` 通过命令行参数选择模型子集
- ut 目标产出可执行文件，可通过 `ctest` 运行
- 编译产物输出到 `build/<platform>/` 目录
- 纯 C 编译，不引入 C++
- 编译错误有清晰的错误信息

### REQ-1.4 变更触发 [P1]

**描述：** 桩代码变更时，自动标记关联用例为待重新执行。

**机制：**
- 框架维护 `桩代码路径 → 用例ID` 的映射关系（在用例注册时建立）
- 检测到桩代码文件变更（通过 git diff 或文件 mtime）时，将关联用例状态置为 PENDING
- `common/` 变更影响全部用例，`models/<name>/` 变更只影响对应模型的用例
- Jenkins 构建时自动执行受影响的用例

**验收标准：**
- 修改 `stubs/models/resnet/` 后，所有关联 resnet 的用例状态变为 PENDING
- 修改 `stubs/common/` 后，所有桩代码相关用例状态变为 PENDING
- 未修改的模型关联的用例状态不受影响

### REQ-1.5 桩代码模板 [P2]

**描述：** 提供模板工具，快速创建新模型的业务桩代码骨架。

**命令：**

```bash
aitf stub new --name yolo
```

**生成内容：**
- `stubs/models/yolo/CMakeLists.txt`
- `stubs/models/yolo/yolo_stub.c` （含 `#include "platform_api.h"` 和 `yolo_run()` 函数骨架）

**验收标准：**
- 生成的代码可直接编译通过（空实现）
- 生成的代码已包含注册宏和 hooks 头文件引用
- 模板内容可通过配置文件自定义

## 3. 技术选型

| 决策 | 选型 | 备选 | 理由 |
|------|------|------|------|
| 语言 | C | C++ | 团队熟悉，桩代码逻辑简单，C 足够 |
| 构建系统 | CMake 3.16+ | Make, Meson | 跨平台交叉编译支持好，IDE集成好 |
| 工具链管理 | CMake toolchain file | 环境变量 | CMake原生方案，声明式清晰 |
| 模型注册 | CMake `configure_file()` 自动生成注册表 | 手动注册列表 | 静态库环境下可控，新增模型只需声明 CMake 变量 |
| 模板引擎 | Jinja2 (Python) | cookiecutter | 项目已依赖Jinja2，无需额外引入 |

## 4. 数据模型

```python
@dataclass
class StubInfo:
    """模型桩代码元信息"""
    name: str               # 模型名（resnet, bert, ...）
    path: str               # 相对于 stubs/models/ 的路径
    sources: list[str]      # .c 源文件列表
    headers: list[str]      # .h 头文件列表
    last_modified: datetime  # 最后修改时间
    git_hash: str           # 最后修改的 commit hash

@dataclass
class HookPlatform:
    """平台 hooks 信息"""
    name: str               # func_sim
    include_dir: str        # hooks/func_sim/include/
    lib_dir: str            # hooks/func_sim/lib/
    libraries: list[str]    # ["platform_sim"]
```

## 5. 对外接口

```python
class StubManager:
    def list_stubs(self) -> list[StubInfo]: ...
    def get_stub(self, name: str) -> StubInfo: ...
    def list_platforms(self) -> list[HookPlatform]: ...
    def build(self, platform: str = "func_sim", toolchain: str | None = None,
              target: str = "all") -> BuildResult:
        """target: app / ut / all"""
        ...
    def detect_changes(self, since: str = "HEAD~1") -> list[StubInfo]: ...
    def create_from_template(self, name: str) -> Path: ...
```

## 6. 依赖

- REQ-3（构建管理）：工具链下载和版本管理
- REQ-4（用例管理）：桩代码与用例的关联关系
