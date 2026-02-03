# 扩展性模块详细设计 (Extensibility)

## 模块概述

| 属性 | 值 |
|------|-----|
| **模块ID** | EXT |
| **模块名称** | 扩展性 |
| **职责** | 插件系统、自定义扩展、API接口 |
| **需求覆盖** | EXT-001 ~ EXT-009 |

---

## 1. 逻辑视图

### 1.1 插件系统类图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Extensibility Classes                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        Plugin System                                  │  │
│  │                                                                       │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │  │
│  │  │                      PluginManager                               │ │  │
│  │  ├─────────────────────────────────────────────────────────────────┤ │  │
│  │  │ - plugins: Dict[str, IPlugin]                                   │ │  │
│  │  │ - hooks: HookRegistry                                           │ │  │
│  │  │ - enabled: Set[str]                                             │ │  │
│  │  ├─────────────────────────────────────────────────────────────────┤ │  │
│  │  │ + discover_plugins() -> List[IPlugin]                           │ │  │
│  │  │ + register(plugin: IPlugin)                                     │ │  │
│  │  │ + enable(plugin_name: str)                                      │ │  │
│  │  │ + disable(plugin_name: str)                                     │ │  │
│  │  │ + get_plugin(name: str) -> IPlugin                              │ │  │
│  │  └─────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                       │  │
│  │              ┌─────────────────────────────┐                          │  │
│  │              │     <<interface>>           │                          │  │
│  │              │        IPlugin              │                          │  │
│  │              ├─────────────────────────────┤                          │  │
│  │              │ + name: str                 │                          │  │
│  │              │ + version: str              │                          │  │
│  │              │ + initialize(config)        │                          │  │
│  │              │ + shutdown()                │                          │  │
│  │              └──────────────┬──────────────┘                          │  │
│  │                             │                                         │  │
│  │       ┌─────────────┬───────┼───────┬─────────────┬─────────────┐     │  │
│  │       ▼             ▼       ▼       ▼             ▼             ▼     │  │
│  │  ┌─────────┐  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  ┌─────────┐
│  │  │ Model   │  │  Data   │ │Reporter │ │Assertion│ │ Metric  │  │  Hook   │
│  │  │ Loader  │  │ Loader  │ │ Plugin  │ │ Plugin  │ │ Plugin  │  │ Plugin  │
│  │  │ Plugin  │  │ Plugin  │ │         │ │         │ │         │  │         │
│  │  └─────────┘  └─────────┘ └─────────┘ └─────────┘ └─────────┘  └─────────┘
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                          Hook System                                  │  │
│  │                                                                       │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │  │
│  │  │                      HookRegistry                                │ │  │
│  │  ├─────────────────────────────────────────────────────────────────┤ │  │
│  │  │ - hooks: Dict[str, List[Callable]]                              │ │  │
│  │  ├─────────────────────────────────────────────────────────────────┤ │  │
│  │  │ + register(event: str, callback: Callable, priority: int)       │ │  │
│  │  │ + unregister(event: str, callback: Callable)                    │ │  │
│  │  │ + trigger(event: str, context: Dict) -> List[Any]               │ │  │
│  │  │ + get_hooks(event: str) -> List[Callable]                       │ │  │
│  │  └─────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                       │  │
│  │  Hook Events:                                                         │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │  │
│  │  │ before_collection  │ after_collection  │ before_test            │ │  │
│  │  │ after_test         │ on_test_success   │ on_test_failure        │ │  │
│  │  │ before_report      │ after_report      │ on_exception           │ │  │
│  │  └─────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                          API Interfaces                               │  │
│  │                                                                       │  │
│  │  ┌──────────────────────┐    ┌──────────────────────┐                 │  │
│  │  │    Python API        │    │     REST API         │                 │  │
│  │  ├──────────────────────┤    ├──────────────────────┤                 │  │
│  │  │ + TestEngine         │    │ POST /api/run        │                 │  │
│  │  │ + configure()        │    │ GET  /api/status     │                 │  │
│  │  │ + run()              │    │ GET  /api/results    │                 │  │
│  │  │ + discover()         │    │ POST /api/cancel     │                 │  │
│  │  └──────────────────────┘    └──────────────────────┘                 │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 插件接口定义

```python
# plugin/interfaces.py

from typing import Protocol, Dict, Any, Optional, List
from abc import ABC, abstractmethod


class IPlugin(Protocol):
    """插件接口"""

    @property
    def name(self) -> str:
        """插件名称"""
        ...

    @property
    def version(self) -> str:
        """插件版本"""
        ...

    def initialize(self, config: Dict[str, Any]) -> None:
        """初始化插件"""
        ...

    def shutdown(self) -> None:
        """关闭插件"""
        ...


class BasePlugin(ABC):
    """插件基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        pass

    def initialize(self, config: Dict[str, Any]) -> None:
        """初始化，子类可重写"""
        pass

    def shutdown(self) -> None:
        """关闭，子类可重写"""
        pass


class IModelLoaderPlugin(IPlugin):
    """模型加载器插件接口"""

    def load(self, path: str, **kwargs) -> Any:
        """加载模型"""
        ...

    def supports(self, format: str) -> bool:
        """检查支持的格式"""
        ...


class IDataLoaderPlugin(IPlugin):
    """数据加载器插件接口"""

    def load(self, source: str, **kwargs) -> Any:
        """加载数据"""
        ...

    def supports(self, format: str) -> bool:
        """检查支持的格式"""
        ...


class IReporterPlugin(IPlugin):
    """报告器插件接口"""

    def generate(self, results: Any) -> Any:
        """生成报告"""
        ...

    def export(self, report: Any, path: str) -> None:
        """导出报告"""
        ...


class IAssertionPlugin(IPlugin):
    """断言插件接口"""

    def register_assertions(self, engine: Any) -> None:
        """注册自定义断言"""
        ...


class IMetricPlugin(IPlugin):
    """指标插件接口"""

    def compute(self, predictions: Any, targets: Any) -> float:
        """计算指标"""
        ...

    @property
    def metric_name(self) -> str:
        """指标名称"""
        ...
```

---

## 2. 开发视图

### 2.1 包结构

```
aitest/plugin/
├── __init__.py
├── manager.py               # PluginManager
├── discovery.py             # 插件发现
├── registry.py              # 插件注册
├── hooks.py                 # HookRegistry
├── base.py                  # BasePlugin
├── interfaces.py            # 插件接口定义
├── decorators.py            # @plugin, @hook装饰器
└── builtin/                 # 内置插件
    ├── __init__.py
    └── ...

aitest/api/
├── __init__.py
├── public.py                # Python公共API
├── rest/
│   ├── __init__.py
│   ├── server.py            # REST API服务器
│   ├── routes.py            # 路由定义
│   └── schemas.py           # 请求/响应Schema
└── client.py                # API客户端
```

### 2.2 实现示例

```python
# plugin/manager.py

from typing import Dict, List, Set, Optional, Any
from importlib.metadata import entry_points
import logging

from .interfaces import IPlugin
from .hooks import HookRegistry

logger = logging.getLogger(__name__)


class PluginManager:
    """插件管理器"""

    ENTRY_POINT_GROUP = "aitest.plugins"

    def __init__(self):
        self._plugins: Dict[str, IPlugin] = {}
        self._enabled: Set[str] = set()
        self.hooks = HookRegistry()

    def discover_plugins(self) -> List[IPlugin]:
        """发现已安装的插件"""
        discovered = []

        # 从entry_points发现插件
        eps = entry_points(group=self.ENTRY_POINT_GROUP)
        for ep in eps:
            try:
                plugin_class = ep.load()
                plugin = plugin_class()
                discovered.append(plugin)
                logger.info(f"Discovered plugin: {plugin.name} v{plugin.version}")
            except Exception as e:
                logger.error(f"Failed to load plugin {ep.name}: {e}")

        return discovered

    def register(self, plugin: IPlugin) -> None:
        """注册插件"""
        if plugin.name in self._plugins:
            logger.warning(f"Plugin {plugin.name} already registered, replacing")

        self._plugins[plugin.name] = plugin
        logger.info(f"Registered plugin: {plugin.name}")

    def enable(self, plugin_name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """启用插件"""
        if plugin_name not in self._plugins:
            raise ValueError(f"Plugin {plugin_name} not found")

        plugin = self._plugins[plugin_name]
        plugin.initialize(config or {})
        self._enabled.add(plugin_name)

        # 注册插件的钩子
        self._register_plugin_hooks(plugin)

        logger.info(f"Enabled plugin: {plugin_name}")

    def disable(self, plugin_name: str) -> None:
        """禁用插件"""
        if plugin_name not in self._enabled:
            return

        plugin = self._plugins[plugin_name]
        plugin.shutdown()
        self._enabled.remove(plugin_name)

        logger.info(f"Disabled plugin: {plugin_name}")

    def get_plugin(self, name: str) -> Optional[IPlugin]:
        """获取插件"""
        return self._plugins.get(name)

    def list_plugins(self) -> List[Dict[str, Any]]:
        """列出所有插件"""
        return [
            {
                "name": p.name,
                "version": p.version,
                "enabled": p.name in self._enabled
            }
            for p in self._plugins.values()
        ]

    def _register_plugin_hooks(self, plugin: IPlugin) -> None:
        """注册插件的钩子"""
        for attr_name in dir(plugin):
            attr = getattr(plugin, attr_name)
            if hasattr(attr, '_hook_event'):
                event = attr._hook_event
                priority = getattr(attr, '_hook_priority', 0)
                self.hooks.register(event, attr, priority)


# plugin/hooks.py

from typing import Dict, List, Callable, Any
from collections import defaultdict


class HookRegistry:
    """钩子注册表"""

    def __init__(self):
        self._hooks: Dict[str, List[tuple]] = defaultdict(list)

    def register(self, event: str, callback: Callable, priority: int = 0) -> None:
        """注册钩子"""
        self._hooks[event].append((priority, callback))
        # 按优先级排序
        self._hooks[event].sort(key=lambda x: x[0], reverse=True)

    def unregister(self, event: str, callback: Callable) -> None:
        """注销钩子"""
        self._hooks[event] = [
            (p, c) for p, c in self._hooks[event] if c != callback
        ]

    def trigger(self, event: str, context: Dict[str, Any]) -> List[Any]:
        """触发钩子"""
        results = []
        for _, callback in self._hooks.get(event, []):
            try:
                result = callback(context)
                results.append(result)
            except Exception as e:
                # 钩子执行失败不应影响主流程
                logger.error(f"Hook {callback.__name__} failed: {e}")
        return results

    def get_hooks(self, event: str) -> List[Callable]:
        """获取事件的所有钩子"""
        return [c for _, c in self._hooks.get(event, [])]


# plugin/decorators.py

from typing import Callable, Optional


def plugin(name: str, version: str):
    """插件装饰器"""
    def decorator(cls):
        cls._plugin_name = name
        cls._plugin_version = version

        @property
        def plugin_name(self) -> str:
            return self._plugin_name

        @property
        def plugin_version(self) -> str:
            return self._plugin_version

        cls.name = plugin_name
        cls.version = plugin_version

        return cls
    return decorator


def hook(event: str, priority: int = 0):
    """钩子装饰器"""
    def decorator(func: Callable) -> Callable:
        func._hook_event = event
        func._hook_priority = priority
        return func
    return decorator


# 示例插件实现
# my_plugin/__init__.py

from aitest.plugin import BasePlugin, plugin, hook


@plugin(name="my-custom-plugin", version="1.0.0")
class MyCustomPlugin(BasePlugin):
    """自定义插件示例"""

    def initialize(self, config):
        self.config = config
        print(f"Initializing {self.name}")

    @hook("before_test", priority=10)
    def on_before_test(self, context):
        """测试前钩子"""
        print(f"Before test: {context.get('test_name')}")

    @hook("after_test")
    def on_after_test(self, context):
        """测试后钩子"""
        result = context.get('result')
        if result.status == 'failed':
            self.send_notification(result)

    def send_notification(self, result):
        """发送通知"""
        # 自定义通知逻辑
        pass

    def shutdown(self):
        print(f"Shutting down {self.name}")
```

### 2.3 REST API实现

```python
# api/rest/server.py

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uuid

from aitest import TestEngine

app = FastAPI(title="AI Test Framework API", version="1.0.0")

# 存储运行中的测试
_running_tests: Dict[str, Dict] = {}


class RunTestRequest(BaseModel):
    test_paths: List[str]
    config: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None


class TestStatus(BaseModel):
    run_id: str
    status: str
    progress: float
    passed: int
    failed: int
    total: int


@app.post("/api/run")
async def run_tests(request: RunTestRequest, background_tasks: BackgroundTasks):
    """启动测试运行"""
    run_id = str(uuid.uuid4())

    _running_tests[run_id] = {
        "status": "starting",
        "progress": 0,
        "passed": 0,
        "failed": 0,
        "total": 0
    }

    background_tasks.add_task(execute_tests, run_id, request)

    return {"run_id": run_id, "status": "started"}


@app.get("/api/status/{run_id}")
async def get_status(run_id: str) -> TestStatus:
    """获取测试状态"""
    if run_id not in _running_tests:
        raise HTTPException(status_code=404, detail="Run not found")

    data = _running_tests[run_id]
    return TestStatus(run_id=run_id, **data)


@app.get("/api/results/{run_id}")
async def get_results(run_id: str):
    """获取测试结果"""
    if run_id not in _running_tests:
        raise HTTPException(status_code=404, detail="Run not found")

    data = _running_tests[run_id]
    if data["status"] != "completed":
        raise HTTPException(status_code=400, detail="Test not completed")

    return data.get("results", {})


@app.post("/api/cancel/{run_id}")
async def cancel_test(run_id: str):
    """取消测试"""
    if run_id not in _running_tests:
        raise HTTPException(status_code=404, detail="Run not found")

    _running_tests[run_id]["status"] = "cancelling"
    return {"status": "cancelling"}


async def execute_tests(run_id: str, request: RunTestRequest):
    """后台执行测试"""
    engine = TestEngine()
    engine.initialize()

    tests = engine.discover_tests(request.test_paths)
    _running_tests[run_id]["total"] = len(tests)
    _running_tests[run_id]["status"] = "running"

    results = []
    for i, test in enumerate(tests):
        if _running_tests[run_id]["status"] == "cancelling":
            _running_tests[run_id]["status"] = "cancelled"
            break

        result = engine.run_single(test)
        results.append(result)

        if result.status == "passed":
            _running_tests[run_id]["passed"] += 1
        else:
            _running_tests[run_id]["failed"] += 1

        _running_tests[run_id]["progress"] = (i + 1) / len(tests)

    _running_tests[run_id]["status"] = "completed"
    _running_tests[run_id]["results"] = results
```

---

## 3. 场景视图

### 3.1 插件开发示例

```python
# 创建自定义模型加载器插件

from aitest.plugin import BasePlugin, plugin
from aitest.plugin.interfaces import IModelLoaderPlugin


@plugin(name="custom-model-loader", version="1.0.0")
class CustomModelLoaderPlugin(BasePlugin, IModelLoaderPlugin):
    """自定义模型加载器"""

    def load(self, path: str, **kwargs):
        # 自定义加载逻辑
        import my_framework
        return my_framework.load_model(path)

    def supports(self, format: str) -> bool:
        return format in ['myformat', 'custom']


# 在 setup.py 中注册
# entry_points={
#     'aitest.plugins': [
#         'custom-model-loader = my_plugin:CustomModelLoaderPlugin',
#     ],
# }
```

### 3.2 需求追溯

| 需求ID | 实现类/方法 | 测试用例 |
|--------|-------------|----------|
| EXT-001 | `PluginManager` | test_plugin_management |
| EXT-002 | `IModelLoaderPlugin`, `IReporterPlugin` | test_plugin_types |
| EXT-003 | `HookRegistry` | test_hooks |
| EXT-004 | `BasePlugin` | test_custom_tests |
| EXT-005 | `IDataLoaderPlugin` | test_custom_data |
| EXT-006 | `api/public.py` | test_python_api |
| EXT-007 | `api/rest/` | test_rest_api |
| EXT-008 | `ConfigExtension` | test_config_extension |
| EXT-009 | `docs/plugins/` | N/A (Documentation) |

---

*本文档为扩展性模块的详细设计，基于4+1视图方法。*
