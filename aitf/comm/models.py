"""ORM models for the datastore cache database.

Tables use a ``ds_`` prefix to avoid collisions with models from other
framework modules that will share the same database.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class CaseDataRow(Base):
    """A registered test-data case (mirrors :class:`CaseData`)."""

    __tablename__ = "ds_cases"

    case_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    name: Mapped[str] = mapped_column(String(256), default="")
    platform: Mapped[str] = mapped_column(String(64), default="")
    model: Mapped[str] = mapped_column(String(64), default="")
    variant: Mapped[str] = mapped_column(String(128), default="")
    version: Mapped[str] = mapped_column(String(32), default="v1")
    source: Mapped[str] = mapped_column(String(32), default="local")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    files: Mapped[list[FileEntryRow]] = relationship(
        "FileEntryRow", back_populates="case", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_ds_cases_platform", "platform"),
        Index("ix_ds_cases_model", "model"),
    )


class FileEntryRow(Base):
    """A single file belonging to a case."""

    __tablename__ = "ds_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_id: Mapped[str] = mapped_column(
        String(256), ForeignKey("ds_cases.case_id", ondelete="CASCADE")
    )
    data_type: Mapped[str] = mapped_column(String(32))  # weights/inputs/golden/artifacts
    path: Mapped[str] = mapped_column(Text)
    size: Mapped[int] = mapped_column(Integer, default=0)
    checksum: Mapped[str] = mapped_column(String(80), default="")

    case: Mapped[CaseDataRow] = relationship("CaseDataRow", back_populates="files")
