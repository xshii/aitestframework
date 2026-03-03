# AITF 功能接口文档

当前已实现的功能，按 Web API、Web UI、CLI 三层列出。

## 1. Golden 数据管理

每个 **模型/版本** 对应一个文件，支持 `.pth / .bin / .zip / .tar / .tar.gz` 格式。
重复上传同 模型/版本 会覆盖旧文件。

存储层公共逻辑：`aitf/ds/store.py` (`GoldenStore`)

### Web API

| 方法 | 路由 | 说明 | 代码位置 |
|------|------|------|----------|
| GET | `/api/golden` | 列出所有 golden 数据 | `aitf/web/api/ds_routes.py` |
| POST | `/api/golden/upload` | 上传文件（multipart: model, version, file） | 同上 |
| GET | `/api/golden/<model>/<version>/download` | 下载文件 | 同上 |
| DELETE | `/api/golden/<model>/<version>` | 删除 model/version | 同上 |

### Web UI（Data 标签页）

| 功能 | 说明 |
|------|------|
| 上传文件 | 填写 模型 + 版本 + 选择文件，点击上传 |
| 文件列表 | 表格展示所有 golden 数据（模型、版本、文件名、大小） |
| 下载 | 点击文件名直接下载 |
| 删除 | 每行有删除按钮 |

### CLI

| 命令 | 说明 |
|------|------|
| `aitf data list [model]` | 列出所有 golden 数据，可按模型过滤 |
| `aitf data upload <model> <version> <file>` | 上传本地文件 |
| `aitf data download <model> <version> [-o dir]` | 下载到本地目录（默认当前目录） |
| `aitf data delete <model> <version>` | 删除 model/version |

---

## 2. 依赖管理 (Deps)

通过 `deps.yaml` 管理工具链、库、仓库依赖。

### Web API

| 方法 | 路由 | 说明 | 代码位置 |
|------|------|------|----------|
| GET | `/api/deps` | 列出所有依赖及安装状态 | `aitf/web/api/deps_routes.py` |
| POST | `/api/deps/install` | 安装依赖（异步，返回 task_id） | 同上 |
| POST | `/api/deps/clean` | 清理缓存 | 同上 |
| GET | `/api/deps/doctor` | 诊断检查 | 同上 |
| POST | `/api/deps/upload` | 上传 .tar.gz 依赖包 | 同上 |
| GET | `/api/deps/uploads` | 列出已上传的依赖包 | 同上 |
| GET | `/api/deps/uploads/<filename>/download` | 下载已上传的依赖包 | 同上 |
| DELETE | `/api/deps/uploads/<filename>` | 删除已上传的依赖包 | 同上 |
| GET | `/api/tasks/<task_id>` | 查询异步任务状态 | 同上 |

### Web UI（Deps 标签页）

| 功能 | 说明 |
|------|------|
| 依赖列表 | 表格展示所有 toolchain / library / repo |
| 上传依赖包 | 选择 .tar.gz 文件上传 |
| 已上传包列表 | 展示已上传的 Archives，支持下载和删除 |

### CLI

| 命令 | 说明 |
|------|------|
| `aitf deps list` | 列出所有依赖及安装状态 |
| `aitf deps install [name]` | 安装全部或指定依赖 |
| `aitf deps lock` | 生成 deps.lock.yaml |
| `aitf deps clean` | 清理缓存 |
| `aitf deps doctor` | 运行依赖诊断 |

---

## 3. Bundle 管理

Bundle = 一组依赖版本的快照组合，用于切换不同测试环境配置。

### Web API

| 方法 | 路由 | 说明 | 代码位置 |
|------|------|------|----------|
| GET | `/api/bundles` | 列出所有 bundles | `aitf/web/api/deps_routes.py` |
| GET | `/api/bundles/<name>` | 查看 bundle 详情 | 同上 |
| POST | `/api/bundles` | 创建 bundle | 同上 |
| DELETE | `/api/bundles/<name>` | 删除 bundle | 同上 |
| GET | `/api/bundles/<name>/export` | 导出 bundle 为 YAML | 同上 |
| POST | `/api/bundles/import` | 导入 bundle（.tar.gz） | 同上 |
| POST | `/api/bundles/<name>/install` | 安装 bundle 依赖（异步） | 同上 |
| POST | `/api/bundles/<name>/use` | 切换到某个 bundle（异步） | 同上 |

### Web UI（Bundles 标签页）

| 功能 | 说明 |
|------|------|
| Bundle 列表 | 表格展示名称、描述、状态、依赖项，高亮 active bundle |
| 导出 | Export 按钮下载 YAML |
| 删除 | Delete 按钮 |
| 创建 | 填写名称、描述、状态，勾选依赖项 |
| 导入 | 上传 .tar.gz 文件导入 |

### CLI

| 命令 | 说明 |
|------|------|
| `aitf bundle list` | 列出所有 bundles（* 标记 active） |
| `aitf bundle show <name>` | 查看详情 |
| `aitf bundle use <name> [--force]` | 切换 bundle |
| `aitf bundle install <name>` | 安装 bundle 依赖 |
| `aitf bundle export <name> -o <file>` | 导出为离线归档 |
| `aitf bundle import <file>` | 导入归档 |

---

## 4. 日志查看 (Logs)

浏览 `data/logs/` 目录下的日志文件。

### Web UI（Logs 标签页）

| 功能 | 说明 |
|------|------|
| 目录浏览 | 递归展示日志目录和文件 |
| 查看日志 | 点击文件名查看内容 |
| 下载日志 | Download 按钮 |

### Web 页面路由

| 路由 | 说明 | 代码位置 |
|------|------|----------|
| `/logs/` | 日志目录列表 | `aitf/web/views/logs.py` |
| `/logs/<subpath>` | 查看日志内容 | 同上 |
| `/logs/<subpath>/download` | 下载日志文件 | 同上 |

> Logs 暂无 REST API 和 CLI 支持。

---

## 5. Web 服务

### 启动方式

| 方式 | 命令 |
|------|------|
| CLI | `aitf web [--host 127.0.0.1] [--port 5000] [--debug]` |
| 脚本 | `bash scripts/start.sh [--no-reload]` |

### 首页 Tab 布局

| 标签页 | 对应功能 |
|--------|----------|
| Data | Golden 数据管理 |
| Deps | 依赖管理 |
| Bundles | Bundle 管理 |
| Logs | 日志查看 |

---

## 功能覆盖对照表

| 功能 | Web API | Web UI | CLI |
|------|---------|--------|-----|
| **Golden 数据** | | | |
| 列表 | GET /api/golden | Data tab | `aitf data list` |
| 上传 | POST /api/golden/upload | 上传表单 | `aitf data upload` |
| 下载 | GET /api/golden/.../download | 点击文件名 | `aitf data download` |
| 删除 | DELETE /api/golden/... | 删除按钮 | `aitf data delete` |
| **依赖管理** | | | |
| 列表 | GET /api/deps | Deps tab | `aitf deps list` |
| 安装 | POST /api/deps/install | - | `aitf deps install` |
| 清理缓存 | POST /api/deps/clean | - | `aitf deps clean` |
| 诊断 | GET /api/deps/doctor | - | `aitf deps doctor` |
| 生成 lock | - | - | `aitf deps lock` |
| 上传依赖包 | POST /api/deps/upload | Upload 表单 | - |
| 下载依赖包 | GET /api/deps/uploads/.../download | Download 按钮 | - |
| 删除依赖包 | DELETE /api/deps/uploads/... | Delete 按钮 | - |
| **Bundle** | | | |
| 列表 | GET /api/bundles | Bundles tab | `aitf bundle list` |
| 详情 | GET /api/bundles/\<name\> | - | `aitf bundle show` |
| 创建 | POST /api/bundles | Create 表单 | - |
| 删除 | DELETE /api/bundles/\<name\> | Delete 按钮 | - |
| 导出 | GET /api/bundles/.../export | Export 按钮 | `aitf bundle export` |
| 导入 | POST /api/bundles/import | Import 表单 | `aitf bundle import` |
| 安装 | POST /api/bundles/.../install | - | `aitf bundle install` |
| 切换 | POST /api/bundles/.../use | - | `aitf bundle use` |
| **日志** | | | |
| 浏览/查看/下载 | - | Logs tab | - |
