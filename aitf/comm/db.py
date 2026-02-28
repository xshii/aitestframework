"""Database engine and session helpers.

Uses SQLite with WAL journal mode for concurrent read access.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from aitf.comm.models import Base

_engine: Engine | None = None
_SessionFactory: sessionmaker[Session] | None = None


def _set_wal_mode(dbapi_conn, connection_record):  # noqa: ARG001
    """Enable WAL journal mode for SQLite."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


def init_db(path: str = "data/aitf.db") -> Engine:
    """Create the database engine, register WAL pragma, and create tables.

    Args:
        path: Filesystem path for the SQLite database file.

    Returns:
        The SQLAlchemy :class:`Engine`.
    """
    global _engine, _SessionFactory  # noqa: PLW0603

    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    url = f"sqlite:///{db_path}"
    _engine = create_engine(url, echo=False)
    event.listen(_engine, "connect", _set_wal_mode)

    Base.metadata.create_all(_engine)

    _SessionFactory = sessionmaker(bind=_engine)
    return _engine


def get_session() -> Session:
    """Return a new :class:`Session` bound to the current engine."""
    if _SessionFactory is None:
        init_db()
    assert _SessionFactory is not None
    return _SessionFactory()


def reset() -> None:
    """Reset global state (useful in tests)."""
    global _engine, _SessionFactory  # noqa: PLW0603
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionFactory = None
