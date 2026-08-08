"""Pydantic schemas for the violations API."""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, ConfigDict


class ViolationBase(BaseModel):
    camera_id: str
    zone: str
    violation_type: str
    confidence: float
    image_path: str
    created_at: dt.datetime


class ViolationCreate(ViolationBase):
    pass


class ViolationOut(ViolationBase):
    model_config = ConfigDict(from_attributes=True)

    id: int


class ViolationStats(BaseModel):
    violation_type: str
    count: int
