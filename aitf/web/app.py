"""Flask application factory."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from flask import Flask
from flask.json.provider import DefaultJSONProvider

from aitf.ds.manager import DataStoreManager
from aitf.web.api.ds_routes import ds_bp


class _JSONProvider(DefaultJSONProvider):
    """Auto-serialise datetime to ISO-8601 strings."""

    def default(self, o: Any) -> Any:
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)


def create_app(config: dict | None = None) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.json_provider_class = _JSONProvider
    app.json = _JSONProvider(app)

    if config:
        app.config.update(config)

    base_dir = app.config.get("DATASTORE_BASE_DIR", "datastore")
    db_path = app.config.get("DATASTORE_DB_PATH", "data/aitf.db")

    manager = DataStoreManager(base_dir=base_dir, db_path=db_path)
    app.config["datastore_manager"] = manager

    app.register_blueprint(ds_bp)

    return app
