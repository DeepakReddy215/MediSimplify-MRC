from functools import lru_cache
from pathlib import Path
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    mongodb_url: str = Field(
        default="mongodb://localhost:27017",
        validation_alias=AliasChoices("MONGODB_URL", "MONGODB_URI")
    )
    database_name: str = "medical_report_simplifier"
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440
    upload_dir: str = "./uploads"
    tesseract_cmd: str = ""
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    simplifier_base_model: str = "google/flan-t5-small"
    simplifier_model_path: str = ""
    simplifier_prompt_max_chars: int = 2500
    simplifier_max_new_tokens: int = 180
    simplifier_min_new_tokens: int = 24
    simplifier_num_beams: int = 2

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
