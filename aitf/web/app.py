"""Flask application factory."""

from __future__ import annotations

import importlib
import logging
import os
from datetime import datetime
from typing import Any

import pkgutil

from flask import Flask
from flask.json.provider import DefaultJSONProvider

import aitf
from aitf.web.views.index import index_bp
from aitf.web.views.logs import logs_bp


class _JSONProvider(DefaultJSONProvider):
    """Auto-serialise datetime to ISO-8601 strings."""

    def default(self, o: Any) -> Any:
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)


_log = logging.getLogger(__name__)


def _discover_blueprints() -> list:
    """Scan ``aitf.*`` packages for ``routes.py`` exposing a ``bp`` attribute."""
    blueprints = []
    for info in pkgutil.iter_modules(aitf.__path__, aitf.__name__ + "."):
        if not info.ispkg:
            continue
        mod_name = info.name + ".routes"
        spec = importlib.util.find_spec(mod_name)
        if spec is None:
            continue
        try:
            mod = importlib.import_module(mod_name)
            bp = getattr(mod, "bp", None)
            if bp is not None:
                blueprints.append(bp)
                _log.debug("discovered blueprint: %s", mod_name)
        except Exception:
            _log.warning("failed to import plugin routes: %s", mod_name, exc_info=True)
    return blueprints


def create_app(config: dict | None = None, aitf_config=None) -> Flask:
    """Create and configure the Flask application."""
    from aitf.config import AitfConfig

    if aitf_config is None:
        aitf_config = AitfConfig()

    tmpl_dir = os.path.join(os.path.dirname(__file__), "templates")
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    app = Flask(__name__, template_folder=tmpl_dir, static_folder=static_dir)
    app.json_provider_class = _JSONProvider
    app.json = _JSONProvider(app)

    app.config['MAX_CONTENT_LENGTH'] = 512 * 1024 * 1024  # 512 MB
    app.config["AITF_CONFIG"] = aitf_config
    app.config.setdefault("DATASTORE_BASE_DIR", str(aitf_config.datastore_dir))

    if config:
        app.config.update(config)

    @app.context_processor
    def inject_aitf_globals():
        return {"aitf_mode": aitf_config.mode.value, "aitf_server": aitf_config.server}

    # Auto-discover blueprints from aitf/*/routes.py
    for bp in _discover_blueprints():
        app.register_blueprint(bp)

    # Explicitly registered blueprints (web-layer only)
    app.register_blueprint(logs_bp)
    app.register_blueprint(index_bp)

    return app
