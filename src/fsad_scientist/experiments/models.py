from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from fsad_scientist.domain.models import utc_now


class SupportSetManifest(BaseModel):
    """Frozen support-set selection, created before a detector sees test labels."""

    dataset: str
    category: str
    protocol: str
    strategy: str
    shots: int
    seed: int
    candidate_pool_files: list[str]
    selected_files: list[str]
    feature_extractor: str
    selection_metadata: dict[str, float | int | str] = Field(default_factory=dict)
    digest: str


class CommandSpec(BaseModel):
    method: str
    executable: str
    args: list[str]
    cwd: Path
    environment: dict[str, str] = Field(default_factory=dict)
    expected_outputs: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def display_command(self) -> str:
        parts = [self.executable, *self.args]
        return " ".join(_quote(item) for item in parts)


class NormalizedExperimentResult(BaseModel):
    parser: str
    metrics: dict[str, float]
    source_files: list[str]


class ExecutionRecord(BaseModel):
    schema_version: str = "1.0"
    run_id: str
    method: str
    status: Literal["running", "succeeded", "failed", "timed_out"]
    command: list[str]
    cwd: str
    output_dir: str
    environment_overrides: dict[str, str]
    dataset_view_digest: str
    support_manifest_digest: str
    code_revision: str | None = None
    environment_digest: str
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None
    duration_seconds: float | None = None
    exit_code: int | None = None
    stdout_path: str
    stderr_path: str
    discovered_artifacts: list[str] = Field(default_factory=list)
    normalized_result: NormalizedExperimentResult | None = None
    error: str | None = None


class PreparedRunArtifacts(BaseModel):
    schema_version: str = "1.0"
    project_id: str
    run_id: str
    dataset_manifest_path: str
    support_manifest_path: str
    dataset_view_manifest_path: str
    dataset_view_root: str
    support_manifest_digest: str
    dataset_view_digest: str


def _quote(value: str) -> str:
    if not value or any(character.isspace() for character in value):
        return f'"{value}"'
    return value
