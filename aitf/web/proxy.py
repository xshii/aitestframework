"""Server proxy helpers — forward requests to remote server in client mode."""

from __future__ import annotations

import json as _json
import urllib.error
from urllib.request import Request, urlopen

from flask import current_app, jsonify, request


def get_server_url() -> str | None:
    """Return the server URL if in client/debug mode, else None."""
    cfg = current_app.config.get("AITF_CONFIG")
    return cfg.server_url if cfg else None


def forward_to_server(path: str, body: dict | None = None):
    """Forward a POST request to the remote server. Returns Flask response."""
    server_url = get_server_url()
    if not server_url:
        return None

    url = f"{server_url}{path}"
    try:
        data = _json.dumps(body).encode() if body is not None else request.get_data()
        req = Request(url, data=data,
                      headers={"Content-Type": "application/json"},
                      method="POST")
        resp = urlopen(req, timeout=30)
        return current_app.response_class(
            resp.read(), status=resp.status, mimetype="application/json")
    except urllib.error.HTTPError as exc:
        return jsonify({"error": f"服务器返回 HTTP {exc.code}"}), 502
    except Exception as exc:
        return jsonify({"error": f"无法连接服务器: {exc}"}), 502


def proxy_if_client(path: str):
    """Proxy GET or POST to server if in client mode. Returns response or None."""
    server_url = get_server_url()
    if not server_url:
        return None

    url = f"{server_url}{path}"
    try:
        if request.method == "POST":
            req = Request(url, data=request.get_data(),
                          headers={"Content-Type": "application/json"},
                          method="POST")
        else:
            req = Request(url)
        resp = urlopen(req, timeout=30)
        return current_app.response_class(
            resp.read(), status=resp.status, mimetype="application/json")
    except urllib.error.HTTPError as exc:
        return jsonify({"error": f"服务器返回 HTTP {exc.code}"}), 502
    except Exception as exc:
        return jsonify({"error": f"无法连接服务器: {exc}"}), 502


def fetch_server_json(path: str) -> dict | None:
    """GET JSON from remote server, return None on failure."""
    server_url = get_server_url()
    if not server_url:
        return None
    try:
        resp = urlopen(f"{server_url}{path}", timeout=10)
        return _json.loads(resp.read())
    except Exception:
        return None
