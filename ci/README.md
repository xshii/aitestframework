# CI 集成说明

## 目录结构

```
ci/
├── Jenkinsfile                  # 桩代码构建流水线
├── Jenkinsfile.smoke            # 冒烟测试流水线（每日定时 + 手动触发）
├── deps_config/
│   ├── latest.yaml              # 每日冒烟的"最新"依赖配置
│   └── example.yaml             # 手动冒烟的示例配置
├── deploy/
│   ├── jenkins_deploy_macos.sh  # macOS 一键部署
│   ├── jenkins_deploy_linux.sh  # Linux 一键部署
│   └── jenkins_deploy_windows.ps1  # Windows 一键部署
├── scripts/
│   └── archive_artifacts.sh     # 产物打包 + 版本标记
└── README.md                    # 本文件

tests/
├── conftest.py                  # pytest fixtures
├── pytest.ini                   # pytest 配置
├── requirements.txt             # Python 依赖
├── test_build_sanity.py         # .so 存在性、可加载、符号导出
├── test_ut_pass.py              # ctest 全部通过验证
└── test_stub_runner.py          # func_sim_main 集成测试
```

## 快速开始（macOS）

```bash
# 一键部署 Jenkins + 所有依赖 + 构建 + 冒烟测试
ci/deploy/jenkins_deploy_macos.sh --repo-url <你的仓库地址>
```

脚本会自动完成：
1. 通过 Homebrew 安装 Java 17、Jenkins LTS、cmake、python3、git
2. 启动 Jenkins 服务
3. 构建项目（`stubs/build.sh -m all`）
4. 安装 Python 依赖并运行冒烟测试

## 仅本地运行冒烟测试（不需要 Jenkins）

```bash
# 1. 构建
stubs/build.sh -m all

# 2. 安装 Python 依赖
python3 -m pip install -r tests/requirements.txt

# 3. 运行冒烟测试
python3 -m pytest tests/ -v --platform=func_sim
```

## 流水线说明

### 构建流水线（Jenkinsfile）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| PLATFORM | func_sim | Hook 平台 |
| BUILD_MODE | all | app / ut / all |
| MODELS | (空=全部) | 逗号分隔的模型名 |
| SWAP_ENDIAN | false | 字节序翻转 |
| PAD_ALIGN | 0 | 对齐填充字节数 |

触发方式：SCM webhook / pollSCM 每 5 分钟 / 手动

### 冒烟流水线（Jenkinsfile.smoke）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| DEPS_CONFIG | latest | deps_config/ 下的配置文件名 |
| PLATFORM | func_sim | Hook 平台 |
| MODELS | (空=全部) | 逗号分隔的模型名 |

触发方式：每日 2:00 AM 定时 / 手动 Build with Parameters

### 在 Jenkins Web UI 创建 Job

1. New Item → Pipeline → 输入名称 `stubs-build`
2. Pipeline → Definition: **Pipeline script from SCM**
3. SCM: Git → Repository URL: 填仓库地址
4. Script Path: `ci/Jenkinsfile`
5. 保存

冒烟 Job 同理，Script Path 改为 `ci/Jenkinsfile.smoke`。
