# 需求文档索引

## 项目边界

```
┌─────────────────────────────────────────────────────────────┐
│                    本仓库：验证框架                          │
├─────────────────────────────────────────────────────────────┤
│  测试框架 (C)      - 断言宏、测试注册、运行器、报告输出       │
│  参考模型          - Golden数据管理、结果比较工具            │
│  平台适配层        - HAL接口、各平台适配桩/Mock              │
│  依赖管理          - 工具链版本管理、配套关系检查            │
│  用例管理          - 用例组织、筛选、配置、标签、平台映射     │
│  结果管理          - 收集、归档、报告、趋势分析              │
│  CI/CD            - 流水线定义、触发策略、通知              │
│  辅助工具          - Python工具、Web界面、数据生成          │
└─────────────────────────────────────────────────────────────┘
                              ↓ 测试
┌─────────────────────────────────────────────────────────────┐
│                    外部：被测对象                            │
├─────────────────────────────────────────────────────────────┤
│  HAL层             - 硬件抽象层（被测）                      │
│  Driver层          - NPU驱动（被测）                        │
│  算子实现          - 硬件算子（被测）                        │
└─────────────────────────────────────────────────────────────┘
                              ↓ 依赖
┌─────────────────────────────────────────────────────────────┐
│                    外部：工具链（二进制下载）                 │
├─────────────────────────────────────────────────────────────┤
│  仿真器 / 编译器 / ESL模型 / FPGA工具                        │
└─────────────────────────────────────────────────────────────┘
```

## 文档结构

```
docs/req/
├── README.md                 # 本文件：需求索引
├── REQ-000-system.md         # REQ-SYS 系统级需求
├── REQ-100-framework.md      # REQ-FWK 测试框架需求
├── REQ-200-model.md          # REQ-MDL 参考模型需求
├── REQ-300-deps.md           # REQ-DEP 依赖管理需求
├── REQ-400-testmgmt.md       # REQ-TMT 用例管理需求
├── REQ-500-result.md         # REQ-RST 结果管理需求
├── REQ-600-cicd.md           # REQ-CIC CI/CD需求
├── REQ-700-platform.md       # REQ-PLT 平台适配层需求
├── REQ-800-tools.md          # REQ-TLS 辅助工具需求
├── REQ-900-quality.md        # REQ-QCK 代码质量需求
└── REQ-A00-efficiency.md     # REQ-EFF 效率提升需求
```

## 需求编号规则

```
REQ-<模块缩写>-<序号>
     │          │
     └──────────┴── 模块缩写(3字符) + 序号(3位)

模块分配:
  SYS - 系统级需求
  FWK - 测试框架
  MDL - 参考模型
  DEP - 依赖管理
  TMT - 用例管理
  RST - 结果管理
  CIC - CI/CD
  PLT - 平台适配层
  TLS - 辅助工具
  QCK - 代码质量
  EFF - 效率提升
```

## 模块概览

| 缩写 | 模块名 | 描述 | 优先级 | 文件 |
|------|--------|------|--------|------|
| SYS | System | 系统级需求、总体约束 | P0 | REQ-000-system.md |
| FWK | Framework | C测试框架核心 | P0 | REQ-100-framework.md |
| MDL | Model | Golden数据与比较工具 | P0 | REQ-200-model.md |
| DEP | Deps | 依赖管理、版本配套 | P1 | REQ-300-deps.md |
| TMT | TestMgmt | 用例管理、配置、筛选 | P0 | REQ-400-testmgmt.md |
| RST | Result | 结果收集、归档、报告 | P1 | REQ-500-result.md |
| CIC | CICD | 流水线、自动化 | P1 | REQ-600-cicd.md |
| PLT | Platform | 平台适配层、HAL接口 | P0 | REQ-700-platform.md |
| TLS | Tools | Python辅助工具 | P1 | REQ-800-tools.md |
| QCK | Quality | 代码质量静态检查 | P0 | REQ-900-quality.md |
| EFF | Efficiency | 开发效率提升工具 | P1 | REQ-A00-efficiency.md |

## 状态与优先级

**状态**:
| 状态 | 含义 |
|------|------|
| draft | 草稿 |
| approved | 已批准 |
| implemented | 已实现 |
| verified | 已验证 |

**优先级**:
| 级别 | 含义 |
|------|------|
| P0 | 必须，阻塞性 |
| P1 | 重要，应实现 |
| P2 | 可延后 |

## 依赖关系

```
REQ-SYS (System)
    │
    ├──→ REQ-FWK (Framework 测试框架)
    │        │
    │        └──→ REQ-TMT (TestMgmt 用例管理)
    │                  │
    │                  └──→ REQ-RST (Result 结果管理)
    │
    ├──→ REQ-MDL (Model 参考模型)
    │
    ├──→ REQ-DEP (Deps 依赖管理)
    │
    ├──→ REQ-PLT (Platform 平台适配层)
    │        │
    │        └──→ 依赖 REQ-FWK
    │
    ├──→ REQ-TLS (Tools 辅助工具)
    │        │
    │        └──→ 依赖 REQ-MDL, REQ-RST
    │
    ├──→ REQ-CIC (CICD)
    │        │
    │        └──→ 依赖以上所有模块
    │
    ├──→ REQ-QCK (Quality 代码质量)
    │        │
    │        └──→ 依赖 REQ-CIC (CI集成)
    │
    └──→ REQ-EFF (Efficiency 效率提升)
              │
              └──→ 依赖 REQ-QCK, REQ-TLS
```

---

## 需求汇总表

### REQ-SYS 系统级需求
| ID | 标题 | 优先级 | 状态 |
|----|------|--------|------|
| REQ-SYS-001 | 验证框架定位 | P0 | draft |
| REQ-SYS-002 | 编程语言约束 | P0 | draft |
| REQ-SYS-003 | 跨平台验证支持 | P0 | draft |
| REQ-SYS-004 | 验证流水线阶段 | P0 | draft |
| REQ-SYS-005 | 框架可扩展性 | P1 | draft |
| REQ-SYS-006 | 文档要求 | P1 | draft |
| REQ-SYS-007 | 代码质量要求 | P0 | draft |
| REQ-SYS-008 | 测试覆盖率要求 | P0 | draft |

### REQ-FWK 测试框架需求
| ID | 标题 | 优先级 | 状态 |
|----|------|--------|------|
| REQ-FWK-001 | 测试用例自动注册 | P0 | draft |
| REQ-FWK-002 | 断言宏集合 | P0 | draft |
| REQ-FWK-003 | 测试结果类型 | P0 | draft |
| REQ-FWK-004 | 测试运行器 | P0 | draft |
| REQ-FWK-005 | 测试输出格式 | P0 | draft |
| REQ-FWK-006 | Setup/Teardown | P1 | draft |
| REQ-FWK-007 | 测试日志接口 | P1 | draft |
| REQ-FWK-008 | 参数化测试 | P0 | draft |
| REQ-FWK-009 | 数据驱动测试 | P0 | draft |
| REQ-FWK-010 | Mock框架 | P1 | draft |
| REQ-FWK-011 | 约束随机测试 | P2 | draft |
| REQ-FWK-012 | 性能基准测试 | P1 | draft |
| REQ-FWK-013 | 功能覆盖率 | P2 | draft |
| REQ-FWK-014 | Fixture环境管理 | P1 | draft |
| REQ-FWK-015 | 失败调试支持 | P2 | draft |
| REQ-FWK-016 | 测试分片执行 | P2 | draft |

### REQ-MDL 参考模型需求
| ID | 标题 | 优先级 | 状态 |
|----|------|--------|------|
| REQ-MDL-001 | Golden数据管理 | P0 | draft |
| REQ-MDL-002 | 结果比较工具 | P0 | draft |

### REQ-DEP 依赖管理需求
| ID | 标题 | 优先级 | 状态 |
|----|------|--------|------|
| REQ-DEP-001 | 依赖配置 | P0 | draft |
| REQ-DEP-002 | 依赖关系刷新 | P0 | draft |
| REQ-DEP-003 | 命令行接口 | P0 | draft |

### REQ-TMT 用例管理需求
| ID | 标题 | 优先级 | 状态 |
|----|------|--------|------|
| REQ-TMT-001 | 用例目录结构 | P0 | draft |
| REQ-TMT-002 | 用例平台映射 | P0 | draft |
| REQ-TMT-003 | 用例标签系统 | P0 | draft |
| REQ-TMT-004 | 执行配置 | P1 | draft |
| REQ-TMT-005 | 测试列表定义 | P0 | draft |
| REQ-TMT-006 | 用例元数据管理 | P2 | draft |
| REQ-TMT-007 | 自动用例发现 | P0 | draft |

### REQ-RST 结果管理需求
| ID | 标题 | 优先级 | 状态 |
|----|------|--------|------|
| REQ-RST-001 | 测试结果收集 | P0 | draft |
| REQ-RST-002 | 结果持久化存储 | P0 | draft |
| REQ-RST-003 | 历史结果归档 | P1 | draft |
| REQ-RST-004 | 测试报告生成 | P0 | draft |
| REQ-RST-005 | 结果趋势分析 | P1 | draft |
| REQ-RST-006 | 结果差异对比 | P1 | draft |
| REQ-RST-007 | 结果通知 | P2 | draft |

### REQ-CIC CI/CD需求
| ID | 标题 | 优先级 | 状态 |
|----|------|--------|------|
| REQ-CIC-001 | 验证流水线阶段 | P0 | draft |
| REQ-CIC-002 | 流水线触发策略 | P0 | draft |
| REQ-CIC-003 | CI作业定义 | P0 | draft |
| REQ-CIC-004 | 并行测试执行 | P1 | draft |
| REQ-CIC-005 | 执行环境管理 | P1 | draft |
| REQ-CIC-006 | CI平台集成 | P0 | draft |
| REQ-CIC-007 | 失败处理策略 | P1 | draft |
| REQ-CIC-008 | CI报告集成 | P1 | draft |

### REQ-PLT 平台适配层需求
| ID | 标题 | 优先级 | 状态 |
|----|------|--------|------|
| REQ-PLT-001 | HAL统一接口规范 | P0 | draft |
| REQ-PLT-002 | LinuxUT Mock实现 | P0 | draft |
| REQ-PLT-003 | LinuxST Stub实现 | P0 | draft |
| REQ-PLT-004 | Simulator适配桩 | P0 | draft |
| REQ-PLT-005 | ESL适配桩 | P1 | draft |
| REQ-PLT-006 | FPGA适配桩 | P1 | draft |
| REQ-PLT-007 | Chip适配桩 | P1 | draft |
| REQ-PLT-008 | 平台选择机制 | P0 | draft |
| REQ-PLT-009 | 平台配置文件 | P1 | draft |
| REQ-PLT-010 | 跨平台数据兼容 | P1 | draft |

### REQ-TLS 辅助工具需求
| ID | 标题 | 优先级 | 状态 |
|----|------|--------|------|
| REQ-TLS-001 | Python测试运行器 | P0 | draft |
| REQ-TLS-002 | 数据生成工具 | P0 | draft |
| REQ-TLS-003 | 报告生成工具 | P0 | draft |
| REQ-TLS-004 | 用例管理Web界面 | P2 | draft |
| REQ-TLS-005 | 结果归档工具 | P1 | draft |
| REQ-TLS-006 | 数据格式转换工具 | P1 | draft |
| REQ-TLS-007 | 环境检查工具 | P1 | draft |

### REQ-QCK 代码质量需求
| ID | 标题 | 优先级 | 状态 |
|----|------|--------|------|
| REQ-QCK-001 | 类型命名检查 | P0 | draft |
| REQ-QCK-002 | 命名规范检查 | P0 | draft |
| REQ-QCK-003 | 内存段标记检查 | P0 | draft |
| REQ-QCK-004 | 安全函数检查 | P0 | draft |
| REQ-QCK-005 | 错误码检查 | P0 | draft |
| REQ-QCK-006 | 判断宏使用检查 | P1 | draft |
| REQ-QCK-007 | 静态分析工具集成 | P0 | draft |
| REQ-QCK-008 | 增量检查 | P1 | draft |
| REQ-QCK-009 | 规则配置 | P1 | draft |

### REQ-EFF 效率提升需求
| ID | 标题 | 优先级 | 状态 |
|----|------|--------|------|
| REQ-EFF-001 | VSCode插件集成 | P0 | draft |
| REQ-EFF-002 | 右键菜单扩展 | P0 | draft |
| REQ-EFF-003 | 代码片段模板 | P1 | draft |
| REQ-EFF-004 | 快捷命令面板 | P1 | draft |
| REQ-EFF-005 | 一键测试执行 | P0 | draft |
| REQ-EFF-006 | 问题导航跳转 | P1 | draft |

---

## 统计

| 模块 | P0 | P1 | P2 | 总计 |
|------|-----|-----|-----|------|
| SYS | 6 | 2 | 0 | 8 |
| FWK | 6 | 6 | 4 | 16 |
| MDL | 2 | 0 | 0 | 2 |
| DEP | 3 | 0 | 0 | 3 |
| TMT | 5 | 1 | 1 | 7 |
| RST | 3 | 3 | 1 | 7 |
| CIC | 4 | 4 | 0 | 8 |
| PLT | 5 | 5 | 0 | 10 |
| TLS | 3 | 3 | 1 | 7 |
| QCK | 6 | 3 | 0 | 9 |
| EFF | 3 | 3 | 0 | 6 |
| **总计** | **46** | **30** | **7** | **83** |
