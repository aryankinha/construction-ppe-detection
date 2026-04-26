from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    APP_ENV: Literal["dev", "staging", "prod"] = "dev"
    LOG_LEVEL: str = "INFO"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    MODEL_PATH: str = "Model/ppe.pt"
    DETECTION_CONFIDENCE: float = 0.5

    # Comma-separated list of allowed CORS origins; "*" allows all
    CORS_ORIGINS: str = "*"


settings = Settings()
