"""
后端抽象层

支持多种计算后端的 DUT 验证：
- CPU: x86/ARM CPU 实现验证
- NPU: 自研 NPU 芯片实现验证

使用示例:
    from backends import get_backend

    # 获取 CPU 后端
    cpu_backend = get_backend("cpu")
    cpu_output = cpu_backend.run(op_name, inputs)

    # 获取 NPU 后端
    npu_backend = get_backend("npu")
    npu_output = npu_backend.run(op_name, inputs)
"""

from typing import Dict, Type

# 后端注册表
_backends: Dict[str, Type["BaseBackend"]] = {}


class BaseBackend:
    """后端基类"""

    name: str = "base"

    def run(self, op_name: str, inputs: dict, **kwargs):
        """执行算子"""
        raise NotImplementedError

    def load_model(self, model_path: str):
        """加载模型"""
        raise NotImplementedError


def register_backend(name: str):
    """注册后端装饰器"""
    def decorator(cls):
        _backends[name] = cls
        return cls
    return decorator


def get_backend(name: str) -> BaseBackend:
    """获取后端实例"""
    if name not in _backends:
        raise ValueError(f"Unknown backend: {name}, available: {list(_backends.keys())}")
    return _backends[name]()


def list_backends() -> list:
    """列出所有可用后端"""
    return list(_backends.keys())
