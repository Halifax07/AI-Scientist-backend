from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_PROJECT_ROOT = Path(__file__).resolve().parents[2]
MONOREPO_ROOT = PACKAGE_PROJECT_ROOT.parent
PROJECT_ROOT = (
    MONOREPO_ROOT
    if (MONOREPO_ROOT / "frontend").is_dir()
    and (MONOREPO_ROOT / "pyproject.toml").is_file()
    else PACKAGE_PROJECT_ROOT
)
ENV_FILES = tuple(dict.fromkeys((PACKAGE_PROJECT_ROOT / ".env", PROJECT_ROOT / ".env")))


class Settings(BaseSettings):
    """Runtime settings. All filesystem paths are resolved from the project root."""

    model_config = SettingsConfigDict(
        env_prefix="AISCIENTIST_",
        env_file=ENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    app_name: str = "FSAD Scientist"
    environment: str = "development"
    runtime: str = "mock"
    storage_root: str = "storage/projects"
    artifact_root: str = "artifacts"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    live_evidence: bool = True
    evidence_mailto: str | None = None
    dinov2_profile_model: str = "facebook/dinov2-small"

    dashscope_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "DASHSCOPE_API_KEY",
            "AISCIENTIST_DASHSCOPE_API_KEY",
        ),
    )
    bailian_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    reasoning_model: str = "qwen3.7-plus"
    vision_model: str = "qwen3-vl-plus"
    embedding_model: str = "text-embedding-v4"
    rerank_model: str = "qwen3-rerank"

    database_url: str = "postgresql+psycopg://fsad:fsad@localhost:5432/fsad_scientist"
    redis_url: str = "redis://localhost:6379/0"
    mlflow_tracking_uri: str = "http://localhost:5000"
    minio_endpoint: str = "http://localhost:9000"

    @property
    def storage_path(self) -> Path:
        return _resolve_project_path(self.storage_root)

    @property
    def artifact_path(self) -> Path:
        return _resolve_project_path(self.artifact_root)

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def dashscope_api_key_value(self) -> str | None:
        return (
            self.dashscope_api_key.get_secret_value()
            if self.dashscope_api_key is not None
            else None
        )


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache
def get_settings() -> Settings:
    return Settings()
