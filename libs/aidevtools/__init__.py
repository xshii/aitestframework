"""AI Dev Tools

推荐用法 - 通过 PyTorch 劫持:
    import aidevtools.golden  # 导入即启用劫持

    import torch.nn.functional as F
    y = F.linear(x, w)  # 自动走 golden

工具函数:
    from aidevtools import ops

    ops.seed(42)
    ops.clear()
    # ... 执行算子 ...
    ops.dump("./workspace")

比对模块:
    from aidevtools.compare import compare_full, CompareStatus

    result = compare_full(dut, golden_pure, golden_qnt)
    print(f"Status: {result.status.value}")

前端模块:
    from aidevtools.frontend import DataGenerator, Tensor

    gen = DataGenerator(seed=42)
    x = gen.gen_input(shape=(2, 64), dtype="bfp16")
"""
__version__ = "0.1.0"

# 延迟导入模块（避免强制依赖torch）
# 注意: seed/clear/dump 通过 __getattr__ 动态导出，不在 __all__ 中声明
__all__ = ["compare", "frontend", "ops"]

_submodules = {}


def __getattr__(name):
    """延迟导入模块"""
    import importlib

    if name in _submodules:
        return _submodules[name]

    if name in ("compare", "frontend", "ops"):
        module = importlib.import_module(f".{name}", __name__)
        _submodules[name] = module
        return module

    if name in ("seed", "clear", "dump"):
        if "ops" not in _submodules:
            _submodules["ops"] = importlib.import_module(".ops", __name__)
        return getattr(_submodules["ops"], name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
