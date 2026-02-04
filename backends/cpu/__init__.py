"""
CPU 后端

用于 x86/ARM CPU 实现的验证。

主要用途：
- CPU Golden 生成（基于 aidevtools）
- CPU 实现的正确性验证
- CPU vs NPU 一致性对比的基准
"""

from backends import BaseBackend, register_backend


@register_backend("cpu")
class CPUBackend(BaseBackend):
    """CPU 后端实现"""

    name = "cpu"

    def __init__(self):
        self.device = "cpu"

    def run(self, op_name: str, inputs: dict, **kwargs):
        """
        执行 CPU 算子

        Args:
            op_name: 算子名称 (matmul, softmax, layernorm, etc.)
            inputs: 输入数据字典
            **kwargs: 算子参数

        Returns:
            算子输出
        """
        # TODO: 集成 aidevtools.ops 或 PyTorch CPU 实现
        raise NotImplementedError("CPU backend not implemented yet")

    def load_model(self, model_path: str):
        """加载 CPU 模型"""
        raise NotImplementedError("CPU model loading not implemented yet")
