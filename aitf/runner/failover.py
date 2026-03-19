"""Failover logic — try alternate targets of the same platform/pool.

Rules:
- Connection failure → try next target (infrastructure issue)
- Test logic failure  → do NOT retry (test is broken)
"""

from __future__ import annotations

import logging
import socket

from .config import TargetConfig

logger = logging.getLogger(__name__)


def check_target_reachable(target: TargetConfig, timeout: float = 5) -> bool:
    """Quick TCP check on the first port with a host."""
    for p in target.ports.values():
        if p.host and p.port:
            try:
                s = socket.create_connection((p.host, p.port), timeout=timeout)
                s.close()
                return True
            except OSError:
                return False
    # No remote ports → local target, always reachable
    return True


def pick_reachable(targets: list[TargetConfig],
                   timeout: float = 5) -> TargetConfig | None:
    """Return the first reachable target from the list, or None."""
    for t in targets:
        if check_target_reachable(t, timeout):
            logger.info("Target %s is reachable", t.name)
            return t
        logger.warning("Target %s unreachable, trying next", t.name)
    return None
