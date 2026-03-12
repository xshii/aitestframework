"""SQLite database initialisation and session helpers."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from aitf.tc.models import Base

logger = logging.getLogger(__name__)

_engine = None
_SessionFactory: sessionmaker[Session] | None = None
_init_path: str | None = None
_init_lock = threading.Lock()


def init_db(db_path: str | Path) -> None:
    """Create engine, enable WAL mode, and create tables if needed.

    Safe to call multiple times — skips if already initialised for the same path.
    Thread-safe via _init_lock.
    """
    global _engine, _SessionFactory, _init_path

    db_path = Path(db_path).resolve()
    db_key = str(db_path)

    if _init_path == db_key:
        return

    with _init_lock:
        # Double-check after acquiring lock
        if _init_path == db_key:
            return

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
        _migrate(_engine)
        _SessionFactory = sessionmaker(bind=_engine)
        _init_path = db_key
        logger.info("tc database ready: %s", db_path)


def _migrate(engine) -> None:
    """Lightweight schema migrations for existing databases.

    Adds missing columns that were introduced after the initial schema.
    Safe to run repeatedly — checks before altering.
    """
    insp = inspect(engine)

    # Collect existing column names per table
    tables = {}
    for table_name in ("executions", "case_results", "suite_info"):
        try:
            tables[table_name] = {c["name"] for c in insp.get_columns(table_name)}
        except Exception:
            tables[table_name] = set()

    migrations = [
        # (table, column, SQL type, default)
        ("executions", "plan_name", "VARCHAR", None),
        ("executions", "platform", "VARCHAR", None),
        ("executions", "git_commit", "VARCHAR", None),
    ]

    with engine.begin() as conn:
        for table, col, sql_type, default in migrations:
            if table in tables and col not in tables[table]:
                ddl = f"ALTER TABLE {table} ADD COLUMN {col} {sql_type}"
                if default is not None:
                    ddl += f" DEFAULT {default}"
                conn.execute(text(ddl))
                logger.info("migration: added %s.%s", table, col)


def get_session() -> Session:
    """Return a new SQLAlchemy session. Caller must close it."""
    if _SessionFactory is None:
        raise RuntimeError("Database not initialised — call init_db() first")
    return _SessionFactory()
