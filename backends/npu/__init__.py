"""
NPU 后端

用于自研 NPU 芯片实现的验证。

主要用途：
- NPU DUT 输出获取
- NPU 实现的正确性验证（vs CPU Golden）
- NPU 性能测试
"""

from backends import BaseBackend, register_backend


@register_backend("npu")
class NPUBackend(BaseBackend):
    """NPU 后端实现"""

    name = "npu"

    def __init__(self):
        self.device = "npu"
        self._initialized = False

    def initialize(self, device_id: int = 0):
        """
        初始化 NPU 设备

        Args:
            device_id: NPU 设备 ID
        """
        # TODO: NPU 设备初始化
        self._initialized = True

    def run(self, op_name: str, inputs: dict, **kwargs):
        """
        执行 NPU 算子

        Args:
            op_name: 算子名称
            inputs: 输入数据字典
            **kwargs: 算子参数

        Returns:
            NPU 算子输出 (DUT output)
        """
        if not self._initialized:
            self.initialize()
        # TODO: 调用 NPU 驱动执行算子
        raise NotImplementedError("NPU backend not implemented yet")

    def load_model(self, model_path: str):
        """加载 NPU 模型"""
        # TODO: 加载编译后的 NPU 模型
        raise NotImplementedError("NPU model loading not implemented yet")

    def compile(self, model_path: str, output_path: str, **kwargs):
        """
        编译模型到 NPU 格式

        Args:
            model_path: 原始模型路径
            output_path: 编译后模型输出路径
            **kwargs: 编译选项
        """
        # TODO: 调用 NPU 编译器
        raise NotImplementedError("NPU compilation not implemented yet")
