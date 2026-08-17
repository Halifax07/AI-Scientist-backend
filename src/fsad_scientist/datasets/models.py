from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from fsad_scientist.domain.models import utc_now


class DatasetFileRecord(BaseModel):
    relative_path: str
    category: str
    split: Literal["train", "validation", "test", "ground_truth"]
    anomaly_type: str
    kind: Literal["image", "mask"]
    byte_size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DatasetAuditIssue(BaseModel):
    severity: Literal["info", "warning", "error"]
    code: str
    message: str
    paths: list[str] = Field(default_factory=list)


class DatasetManifest(BaseModel):
    schema_version: str = "1.0"
    dataset: str
    format: Literal["mvtec_ad"] = "mvtec_ad"
    root: str
    categories: list[str]
    files: list[DatasetFileRecord]
    counts: dict[str, int]
    issues: list[DatasetAuditIssue] = Field(default_factory=list)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime = Field(default_factory=utc_now)

    @property
    def is_valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def support_candidates(self, category: str) -> list[str]:
        """Return only defect-free training images; test labels are never consulted."""

        return sorted(
            item.relative_path
            for item in self.files
            if item.category == category
            and item.split == "train"
            and item.anomaly_type == "good"
            and item.kind == "image"
        )

    def file_by_path(self) -> dict[str, DatasetFileRecord]:
        return {item.relative_path: item for item in self.files}


class DatasetViewManifest(BaseModel):
    schema_version: str = "1.0"
    dataset_digest: str
    support_manifest_digest: str
    category: str
    source_root: str
    view_root: str
    materialization: Literal["hardlink", "copy", "mixed"]
    files: list[str]
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime = Field(default_factory=utc_now)
