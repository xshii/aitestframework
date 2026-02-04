"""pytest 配置 - aidevtools"""
import sys
from pathlib import Path

# 添加 libs 到 Python 路径
_libs_path = Path(__file__).parent.parent.parent
if str(_libs_path) not in sys.path:
    sys.path.insert(0, str(_libs_path))

import pytest
import numpy as np


@pytest.fixture
def tmp_workspace(tmp_path):
    """临时工作目录"""
    return tmp_path


@pytest.fixture
def sample_data():
    """样例数据"""
    np.random.seed(42)
    return np.random.randn(1, 3, 8, 8).astype(np.float32)


@pytest.fixture
def golden_result(sample_data):
    """模拟 golden 结果"""
    return sample_data * 2


@pytest.fixture
def sim_result(golden_result):
    """模拟仿真器结果 (带少量误差)"""
    noise = np.random.randn(*golden_result.shape).astype(np.float32) * 1e-6
    return golden_result + noise
