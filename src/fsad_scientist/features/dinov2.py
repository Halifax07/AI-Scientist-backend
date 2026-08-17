from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from fsad_scientist.datasets.models import DatasetManifest
from fsad_scientist.domain.models import utc_now


class DinoEmbeddingManifest(BaseModel):
    schema_version: str = "1.0"
    dataset_digest: str
    category: str
    model_id: str
    manifest_path: str | None = None
    requested_revision: str | None = None
    resolved_revision: str | None = None
    pooling: str = "normalized_cls_token"
    preprocessing: str = "hf_auto_image_processor_use_fast_false"
    image_files: list[str]
    embeddings: dict[str, list[float]]
    device: str
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime = Field(default_factory=utc_now)


class DinoV2Embedder:
    """Extract deterministic, cached DINOv2 image embeddings from train/good only."""

    def __init__(
        self,
        artifact_root: Path,
        *,
        model_id: str = "facebook/dinov2-small",
        revision: str | None = None,
        device: str = "auto",
        batch_size: int = 8,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.artifact_root = artifact_root.expanduser().resolve()
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.batch_size = batch_size

    def extract(
        self,
        dataset: DatasetManifest,
        *,
        category: str,
        image_files: list[str] | None = None,
        force: bool = False,
    ) -> DinoEmbeddingManifest:
        candidates = dataset.support_candidates(category)
        selected = sorted(image_files or candidates)
        if not selected:
            raise ValueError(f"no train/good images found for category {category}")
        invalid = sorted(set(selected) - set(candidates))
        if invalid:
            raise ValueError(
                "DINO profiling accepts only train/good images: " + ", ".join(invalid)
            )

        cache_path = self._cache_path(dataset, category, selected)
        if cache_path.is_file() and not force:
            return DinoEmbeddingManifest.model_validate_json(
                cache_path.read_text(encoding="utf-8")
            )

        torch, image_class, processor_class, model_class = _load_dependencies()
        execution_device = self._resolve_device(torch)
        processor = processor_class.from_pretrained(
            self.model_id,
            revision=self.revision,
            use_fast=False,
        )
        model = model_class.from_pretrained(
            self.model_id,
            revision=self.revision,
        ).to(execution_device)
        model.eval()

        root = Path(dataset.root).resolve()
        embeddings: dict[str, list[float]] = {}
        with torch.inference_mode():
            for offset in range(0, len(selected), self.batch_size):
                batch_files = selected[offset : offset + self.batch_size]
                images = []
                for relative_path in batch_files:
                    path = (root / Path(relative_path)).resolve()
                    _ensure_within(path, root)
                    with image_class.open(path) as image:
                        images.append(image.convert("RGB").copy())
                inputs = processor(images=images, return_tensors="pt")
                inputs = {name: value.to(execution_device) for name, value in inputs.items()}
                outputs = model(**inputs)
                vectors = outputs.last_hidden_state[:, 0, :].float()
                vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
                for relative_path, vector in zip(batch_files, vectors.cpu(), strict=True):
                    embeddings[relative_path] = [round(float(value), 8) for value in vector]

        resolved_revision = getattr(model.config, "_commit_hash", None)
        digest_payload: dict[str, Any] = {
            "dataset_digest": dataset.digest,
            "category": category,
            "model_id": self.model_id,
            "requested_revision": self.revision,
            "resolved_revision": resolved_revision,
            "pooling": "normalized_cls_token",
            "preprocessing": "hf_auto_image_processor_use_fast_false",
            "image_files": selected,
            "embeddings": embeddings,
        }
        digest = hashlib.sha256(
            json.dumps(digest_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        manifest = DinoEmbeddingManifest(
            **digest_payload,
            device=execution_device,
            digest=digest,
        )
        manifest.manifest_path = str(cache_path.resolve())
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".json.tmp")
        temporary.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(cache_path)
        return manifest

    def _cache_path(
        self,
        dataset: DatasetManifest,
        category: str,
        image_files: list[str],
    ) -> Path:
        request_digest = hashlib.sha256(
            json.dumps(
                {
                    "model_id": self.model_id,
                    "revision": self.revision,
                    "preprocessing": "hf_auto_image_processor_use_fast_false",
                    "image_files": image_files,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()[:16]
        model_slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", self.model_id)
        return (
            self.artifact_root
            / "features"
            / dataset.digest[:16]
            / model_slug
            / category
            / f"{request_digest}.json"
        )

    def _resolve_device(self, torch: Any) -> str:
        if self.device == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device")
        return self.device


def _load_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "DINOv2 extraction requires the 'vision' optional dependencies; "
            "run `uv sync --extra vision --extra dev`"
        ) from exc
    return torch, Image, AutoImageProcessor, AutoModel


def _ensure_within(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"image path escapes dataset root: {path}") from exc
