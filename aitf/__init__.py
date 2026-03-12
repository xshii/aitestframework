"""AI Test Framework — test data management, stub orchestration, and CI tooling."""

__version__ = "0.1.0"

# Lazy re-exports for convenience: from aitf import ctx, load_golden, ...
_LAZY_IMPORTS = {
    "ctx": "aitf.tc.api",
    "load_golden": "aitf.tc.api",
    "load_golden_inputs": "aitf.tc.api",
    "load_golden_outputs": "aitf.tc.api",
    "get_golden_file": "aitf.tc.api",
    "ensure_dep": "aitf.tc.api",
    "ensure_bundle": "aitf.tc.api",
    "get_dep_path": "aitf.tc.api",
    "get_dep_env": "aitf.tc.api",
    "get_bundle_env": "aitf.tc.api",
    "run_script": "aitf.tc.api",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib
        mod = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'aitf' has no attribute {name!r}")
