from __future__ import annotations

from pathlib import Path

from fsad_scientist.datasets.models import DatasetManifest
from fsad_scientist.datasets.view import DatasetViewBuilder
from fsad_scientist.domain.models import ExperimentRun
from fsad_scientist.experiments.models import PreparedRunArtifacts
from fsad_scientist.experiments.support_selection import plan_support_set
from fsad_scientist.features.dinov2 import DinoEmbeddingManifest


class ExperimentPreparationService:
    """Derive and freeze every detector-visible input before execution."""

    def __init__(self, artifact_root: Path) -> None:
        self.artifact_root = artifact_root.expanduser().resolve()

    def prepare(
        self,
        *,
        project_id: str,
        run: ExperimentRun,
        dataset: DatasetManifest,
        dataset_manifest_path: Path,
        embeddings: DinoEmbeddingManifest | None = None,
        candidate_pool_size: int = 30,
    ) -> PreparedRunArtifacts:
        if dataset.dataset.casefold() != run.dataset.casefold():
            raise ValueError("dataset manifest does not match the queued run")
        embedding_values = None
        feature_extractor = "none"
        if embeddings is not None:
            if embeddings.dataset_digest != dataset.digest or embeddings.category != run.category:
                raise ValueError("embedding manifest does not match run dataset/category")
            embedding_values = embeddings.embeddings
            revision = embeddings.resolved_revision or embeddings.requested_revision or "floating"
            feature_extractor = f"{embeddings.model_id}@{revision}"

        support = plan_support_set(
            dataset,
            category=run.category,
            protocol=run.protocol,
            strategy=run.selection_strategy,
            shots=run.shots,
            seed=run.seed,
            candidate_pool_size=candidate_pool_size,
            embeddings=embedding_values,
            feature_extractor=feature_extractor,
        )
        support_path = self.artifact_root / "support_sets" / f"{support.digest}.json"
        _write_json(support_path, support.model_dump_json(indent=2))
        view = DatasetViewBuilder(self.artifact_root).build(dataset, support)
        view_manifest_path = Path(view.view_root) / "fsad_view_manifest.json"
        prepared = PreparedRunArtifacts(
            project_id=project_id,
            run_id=run.id,
            dataset_manifest_path=str(dataset_manifest_path.resolve()),
            support_manifest_path=str(support_path.resolve()),
            dataset_view_manifest_path=str(view_manifest_path.resolve()),
            dataset_view_root=view.view_root,
            support_manifest_digest=support.digest,
            dataset_view_digest=view.digest,
        )
        output = self.artifact_root / "prepared_runs" / project_id / f"{run.id}.json"
        _write_json(output, prepared.model_dump_json(indent=2))
        return prepared


def _write_json(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
