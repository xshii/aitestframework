"""Unified CLI entry-point for the AI Test Framework.

Sub-commands are auto-discovered from ``aitf/*/commands.py`` modules.
Each command module exposes: ``register(subparsers)``,
``dispatch(args) -> bool``, and either ``COMMAND`` (str) or
``COMMANDS`` (list[str]).
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys

import pkgutil

import aitf


# ---------------------------------------------------------------------------
# Auto-discovery
# ---------------------------------------------------------------------------

_log = logging.getLogger(__name__)


def _discover_command_modules() -> list:
    """Scan ``aitf.*`` packages for ``commands.py`` with register/dispatch."""
    modules = []
    for info in pkgutil.iter_modules(aitf.__path__, aitf.__name__ + "."):
        if not info.ispkg:
            continue
        mod_name = info.name + ".commands"
        spec = importlib.util.find_spec(mod_name)
        if spec is None:
            continue
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "register") and hasattr(mod, "dispatch"):
                modules.append(mod)
                _log.debug("discovered command module: %s", mod_name)
        except Exception:
            _log.warning("failed to import plugin commands: %s", mod_name, exc_info=True)
    return modules


# ---------------------------------------------------------------------------
# Web command
# ---------------------------------------------------------------------------

def _cmd_web(args: argparse.Namespace) -> None:
    from aitf.config import Mode, load_config
    from aitf.web.app import create_app

    cfg = load_config()
    if cfg.mode == Mode.CLIENT:
        print(
            f"Error: config.yaml server is a remote IP ({cfg.server}), "
            "cannot start web server in client mode.",
            file=sys.stderr,
        )
        sys.exit(1)
    host = args.host if args.host is not None else cfg.bind_host
    port = args.port if args.port is not None else cfg.port
    app = create_app(aitf_config=cfg)
    app.run(host=host, port=port, debug=args.debug)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _build_parser() -> tuple[argparse.ArgumentParser, list]:
    parser = argparse.ArgumentParser(prog="aitf", description="AI Test Framework CLI")
    sub = parser.add_subparsers(dest="command")

    # Auto-discover command modules (e.g. aitf.ds.commands, aitf.deps.commands)
    cmd_modules = _discover_command_modules()
    for mod in cmd_modules:
        mod.register(sub)

    # -- web (built-in) -----------------------------------------------------
    p = sub.add_parser("web", help="Start the web server")
    p.add_argument("--host", default=None, help="Bind host (default: from config.yaml)")
    p.add_argument("--port", type=int, default=None, help="Bind port (default: from config.yaml)")
    p.add_argument("--debug", action="store_true")

    return parser, cmd_modules


def main() -> None:
    """CLI entry-point."""
    parser, cmd_modules = _build_parser()
    args = parser.parse_args()

    if args.command == "web":
        _cmd_web(args)
        return

    # Try auto-discovered modules
    for mod in cmd_modules:
        commands = getattr(mod, "COMMANDS", [getattr(mod, "COMMAND", None)])
        if args.command in commands:
            if mod.dispatch(args):
                return
            # sub_cmd missing — print help
            parser.parse_args([args.command, "-h"])

    parser.print_help()
    sys.exit(1)
