"""Aggregates all route modules under the versioned API prefix."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import health, violations

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(violations.router)
