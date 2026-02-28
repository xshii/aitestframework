# 部署指南

## 前置要求

- Python 3.10+
- pip

## 一键启动

```bash
./scripts/start.sh            # 正常启动
./scripts/start.sh --debug    # 调试模式（代码修改自动重载）
```

自动完成：创建 `.venv` → 安装依赖 → 启动 Web 服务。

默认监听 `http://0.0.0.0:5000`。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `AITF_HOST` | `0.0.0.0` | 监听地址 |
| `AITF_PORT` | `5000` | 监听端口 |

示例：

```bash
AITF_PORT=8080 ./scripts/start.sh
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/cases` | 列表（?platform=&model=） |
| GET | `/api/cases/<case_id>` | 详情 |
| POST | `/api/cases` | 注册 `{case_id, local_path}` |
| DELETE | `/api/cases/<case_id>` | 删除 |
| POST | `/api/cases/<case_id>/verify` | 校验 |
| POST | `/api/cases/<case_id>/pull` | 拉取 `{remote}` |
| POST | `/api/cases/<case_id>/push` | 推送 `{remote}` |
| POST | `/api/cases/<case_id>/push-artifacts` | 归档 `{remote, artifacts_dir}` |
| GET | `/api/cases/<case_id>/versions` | 版本列表 |
| POST | `/api/verify` | 全量校验 |
| POST | `/api/rebuild-cache` | 重建缓存 |

## CLI 命令

```bash
# 数据管理
aitf data register <case_id> <local_path>
aitf data list [--platform X] [--model Y]
aitf data get <case_id>
aitf data delete <case_id>
aitf data verify [--case <id>]
aitf data pull --remote <name> [--case <id>]
aitf data push --remote <name> --case <id>
aitf data push-artifacts --remote <name> --case <id> --dir <path>
aitf data rebuild-cache

# Web 服务器
aitf web [--host 0.0.0.0] [--port 5000] [--debug]
```
