# AI Test Framework 安装指南

## 依赖分层设计

```
┌─────────────────────────────────────────────────┐
│                    full                          │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌────────┐ │
│  │  torch  │ │  xlsx   │ │ report  │ │ cpp    │ │
│  │ PyTorch │ │openpyxl │ │jinja2   │ │pybind11│ │
│  │ 算子执行 │ │Excel工作流│ │报告生成  │ │C++编译 │ │
│  └─────────┘ └─────────┘ └─────────┘ └────────┘ │
├─────────────────────────────────────────────────┤
│                    base                          │
│              numpy, pyyaml                       │
│              （核心依赖）                          │
└─────────────────────────────────────────────────┘
```

## 安装方式

### 方式1: pip安装（推荐）

```bash
# 最小安装（仅核心功能）
pip install -e .

# 安装特定功能
pip install -e ".[torch]"      # PyTorch算子支持
pip install -e ".[xlsx]"       # Excel工作流
pip install -e ".[report]"     # 报告生成
pip install -e ".[cpp]"        # C++ Golden编译

# 完整安装
pip install -e ".[all]"

# 开发环境
pip install -e ".[all,dev]"
```

### 方式2: requirements文件

```bash
# 最小安装
pip install -r requirements/base.txt

# PyTorch支持（CPU版本，体积小）
pip install -r requirements/torch.txt

# 完整安装
pip install -r requirements/full.txt

# 开发环境
pip install -r requirements/dev.txt
```

### 方式3: 按需手动安装

```bash
# 核心（必须）
pip install numpy pyyaml

# 按需添加
pip install torch --index-url https://download.pytorch.org/whl/cpu  # PyTorch CPU
pip install openpyxl      # Excel工作流
pip install jinja2        # 报告生成
pip install pybind11      # C++ Golden
```

## libs/外部库依赖

各外部库的依赖说明：

| 库 | 必需依赖 | 可选依赖 | 说明 |
|---|---------|---------|------|
| aidevtools | numpy | torch, openpyxl, pybind11 | 算子验证工具 |
| prettycli | - | - | CLI美化 |

### aidevtools功能与依赖对应

| 功能 | 依赖 | 安装命令 |
|-----|------|---------|
| 四状态比对 | numpy | `pip install numpy` |
| 算子Golden生成 | torch | `pip install torch` |
| Excel工作流 | openpyxl | `pip install openpyxl` |
| C++ Golden | pybind11 + cmake | `pip install pybind11` + 系统cmake |

## C++ Golden编译

部分demos需要编译C++ Golden：

```bash
cd libs/aidevtools/golden/cpp
./build.sh
```

**系统依赖**：
- cmake >= 3.20
- g++ >= 9.0 或 clang >= 10.0

## 验证安装

```bash
# 验证核心功能
python -c "from aitestframework import CompareEngine; print('OK')"

# 验证PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# 验证aidevtools
PYTHONPATH=libs python -c "from aidevtools.compare import CompareEngine; print('OK')"

# 运行测试
pytest tests/ -v
```

## 常见问题

### Q: ModuleNotFoundError: No module named 'torch'
A: 安装PyTorch：`pip install torch --index-url https://download.pytorch.org/whl/cpu`

### Q: 需要安装 openpyxl
A: 安装openpyxl：`pip install openpyxl`

### Q: CPU Golden 可执行文件未找到
A: 编译C++ Golden：`cd libs/aidevtools/golden/cpp && ./build.sh`

### Q: 导入aitestframework报错
A: 确保PYTHONPATH包含项目根目录：`export PYTHONPATH=/path/to/aitestframework:$PYTHONPATH`
