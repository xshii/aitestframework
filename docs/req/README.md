# 需求文档索引

---
version: 1.2
date: 2026-02-27
status: draft
---

## 环境约束

- **内网环境**，无法访问外网
- **禁止使用 Docker / 虚拟机**
- 工具链获取方式：本地上传、内网Bash脚本下载、打包成套分发
- 前端静态资源（Bootstrap、ECharts）本地文件引入，不依赖CDN

## 文档清单

| 编号 | 文档 | 模块 | 需求数 |
|------|------|------|--------|
| REQ-1 | [REQ-1-stubs.md](REQ-1-stubs.md) | 桩代码管理 | 5 |
| REQ-2 | [REQ-2-golden.md](REQ-2-golden.md) | Golden数据维护 | 6 |
| REQ-3 | [REQ-3-build.md](REQ-3-build.md) | 版本构建管理 | 9 |
| REQ-4 | [REQ-4-cases.md](REQ-4-cases.md) | 用例管理 | 7 |
| REQ-5 | [REQ-5-logparser.md](REQ-5-logparser.md) | 日志分析工具 | 5 |
| REQ-6 | [REQ-6-runner.md](REQ-6-runner.md) | 自动化执行框架 | 7 |
| REQ-7 | [REQ-7-web.md](REQ-7-web.md) | Flask CI Dashboard | 9 |
| NFR | [REQ-NFR.md](REQ-NFR.md) | 非功能需求 | 6 |

**合计：54 条功能需求 + 6 条非功能需求**

## 架构原则

**Core API 唯一逻辑，CLI 和 Web 都是薄壳调用层。**

```
用例代码 / Jenkins / 终端用户  ──→  CLI (argparse薄壳)  ──┐
                                                          ├──→  Core API (唯一业务逻辑)  ──→  SQLite
Web Dashboard 用户             ──→  Flask路由 (薄壳)     ──┘
```

- 所有业务逻辑只在 `aitestframework.*` Core API 中实现一次
- CLI（`aitf` 命令）和 Web（Flask 路由）只做参数解析和结果渲染
- 避免 CLI / Web / 脚本三套触发源维护多套逻辑的问题

## 技术选型总览

| 决策项 | 选型 | 理由 |
|--------|------|------|
| 用例框架 | unittest (标准库) | 类似nosetest，类继承方式，标准库无额外依赖 |
| 桩代码构建 | CMake 3.16+ | 跨平台交叉编译支持好，IDE集成好 |
| Python版本 | 3.10+ | match语句、更好的类型注解 |
| Web框架 | Flask + Jinja2 | 服务端渲染为主，简洁够用 |
| 前端CSS | Bootstrap 5（本地静态文件） | 组件丰富，只用内置class，零自定义CSS |
| 前端图表 | ECharts 5（本地静态文件） | 功能强大，中文支持好 |
| 数据存储 | SQLite + SQLAlchemy | 零部署成本，ORM简化数据操作 |
| SSH远程执行 | paramiko | Python原生SSH库，成熟稳定 |
| 异步任务 | subprocess + threading | 无额外依赖，适合轻量调度场景 |
| CI | Jenkins（从零搭建，提供Jenkinsfile） | 项目提供Jenkinsfile模板 |
| 依赖管理 | 本地上传 + Bash脚本(scp) | 适配内网，只有SSH服务器 |
| 配置集管理 | 自研 Bundle YAML + CLI | 版本组合打包分发 |
| 代码质量 | ruff + mypy | ruff检查风格，mypy检查类型 |
| 测试框架 | pytest | Python标准测试框架 |

## 模块依赖关系

```
REQ-7 Web Dashboard
  ├── 读取 ─→ REQ-4 用例管理
  ├── 读取 ─→ REQ-5 日志分析
  ├── 读取 ─→ REQ-3 构建管理
  └── 接收 ─→ Jenkins webhook

REQ-6 执行框架
  ├── 依赖 ─→ REQ-3 构建管理（编译桩代码）
  ├── 依赖 ─→ REQ-4 用例管理（获取用例列表）
  ├── 输入 ─→ REQ-2 Golden数据（比较基准）
  └── 输出 ─→ REQ-5 日志分析（产生日志）

REQ-4 用例管理
  ├── 关联 ─→ REQ-1 桩代码（用例对应的激励代码）
  └── 关联 ─→ REQ-2 Golden数据（用例期望结果）

REQ-3 构建管理
  ├── 编译 ─→ REQ-1 桩代码
  └── 管理 ─→ 工具链 + 第三方库（Bundle配置集）
```

## 术语表

| 术语 | 说明 |
|------|------|
| 桩代码 (Stub) | 模拟外部给仿真器/FPGA发送激励消息的C/C++源码 |
| Golden数据 | 已确认正确的基准参考数据，用于比较验证测试输出 |
| 算子 (Operator) | AI计算的基本单元，如conv2d、matmul、relu等 |
| 模型 (Model) | 由多个算子组成的完整神经网络，如ResNet、BERT等 |
| 工具链 (Toolchain) | 编译器、链接器、仿真器等构建/运行所需的工具集合 |
| 配置集 (Bundle) | 一组经过验证的工具链+库的版本组合，整套管理和分发 |
| Shard | 测试分片，将用例集拆分到多个执行器并行运行 |
