"""Query/aggregation logic for violations — kept separate from route handlers so the
chatbot module (LLM tool-use) can call the same functions instead of hitting HTTP."""

from __future__ import annotations

import datetime as dt

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.violation import Violation
from app.schemas.violation import ViolationCreate


async def create_violation(db: AsyncSession, data: ViolationCreate) -> Violation:
    violation = Violation(**data.model_dump())
    db.add(violation)
    await db.commit()
    await db.refresh(violation)
    return violation


async def list_violations(
    db: AsyncSession,
    zone: str | None = None,
    violation_type: str | None = None,
    since: dt.datetime | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[Violation]:
    stmt = select(Violation).order_by(Violation.created_at.desc()).limit(limit).offset(offset)
    if zone:
        stmt = stmt.where(Violation.zone == zone)
    if violation_type:
        stmt = stmt.where(Violation.violation_type == violation_type)
    if since:
        stmt = stmt.where(Violation.created_at >= since)
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def count_by_type(db: AsyncSession, since: dt.datetime | None = None) -> list[tuple[str, int]]:
    stmt = select(Violation.violation_type, func.count(Violation.id)).group_by(Violation.violation_type)
    if since:
        stmt = stmt.where(Violation.created_at >= since)
    result = await db.execute(stmt)
    return list(result.all())
