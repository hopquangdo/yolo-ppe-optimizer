"""App-wide settings, loaded from environment variables (.env)."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    app_name: str = "PPE-YOLO26-Edge Dashboard API"
    api_v1_prefix: str = "/api/v1"

    database_url: str = "postgresql+asyncpg://ppe:ppe@localhost:5432/ppe_violations"

    cors_origins: list[str] = ["http://localhost:5173"]

    inference_engine_path: str = "weights/yolo26_int8.engine"


settings = Settings()
