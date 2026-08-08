"""Endpoints for recording and querying PPE violation events."""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.violation import ViolationCreate, ViolationOut, ViolationStats
from app.services import violation_service

router = APIRouter(prefix="/violations", tags=["violations"])


@router.post("", response_model=ViolationOut, status_code=201)
async def report_violation(data: ViolationCreate, db: AsyncSession = Depends(get_db)) -> ViolationOut:
    """Called by the edge inference service each time a violation is detected."""
    violation = await violation_service.create_violation(db, data)
    return ViolationOut.model_validate(violation)


@router.get("", response_model=list[ViolationOut])
async def get_violations(
    zone: str | None = None,
    violation_type: str | None = None,
    since: dt.datetime | None = None,
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
) -> list[ViolationOut]:
    violations = await violation_service.list_violations(db, zone, violation_type, since, limit, offset)
    return [ViolationOut.model_validate(v) for v in violations]


@router.get("/stats/by-type", response_model=list[ViolationStats])
async def get_stats_by_type(since: dt.datetime | None = None, db: AsyncSession = Depends(get_db)) -> list[ViolationStats]:
    rows = await violation_service.count_by_type(db, since)
    return [ViolationStats(violation_type=t, count=c) for t, c in rows]
