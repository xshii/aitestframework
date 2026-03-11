"""SQLite database initialisation and session helpers."""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from aitf.tc.models import Base

logger = logging.getLogger(__name__)

_engine = None
_SessionFactory: sessionmaker[Session] | None = None


def init_db(db_path: str | Path) -> None:
    """Create engine, enable WAL mode, and create tables if needed."""
    global _engine, _SessionFactory

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    url = f"sqlite:///{db_path}"
    _engine = create_engine(url, echo=False)

    # Enable WAL for better concurrent read/write
    @event.listens_for(_engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _rec):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    Base.metadata.create_all(_engine)
    _SessionFactory = sessionmaker(bind=_engine)
    logger.info("tc database ready: %s", db_path)


def get_session() -> Session:
    """Return a new SQLAlchemy session. Caller must close it."""
    if _SessionFactory is None:
        raise RuntimeError("Database not initialised — call init_db() first")
    return _SessionFactory()
