"""ORM model for a PPE violation event recorded by the edge inference service."""

from __future__ import annotations

import datetime as dt

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class Violation(Base):
    __tablename__ = "violations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(String(64), index=True)
    zone: Mapped[str] = mapped_column(String(128), index=True)
    violation_type: Mapped[str] = mapped_column(String(64), index=True)  # e.g. "no_helmet", "no_vest"
    confidence: Mapped[float] = mapped_column(Float)
    image_path: Mapped[str] = mapped_column(String(512))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), index=True)
