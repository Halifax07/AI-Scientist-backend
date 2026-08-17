from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from fsad_scientist.config import PROJECT_ROOT
from fsad_scientist.datasets.scanner import MvtecDatasetScanner
from fsad_scientist.datasets.view import DatasetViewBuilder
from fsad_scientist.experiments.support_selection import plan_support_set
from fsad_scientist.features.dinov2 import DinoV2Embedder


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    smoke_root = PROJECT_ROOT / "artifacts" / "smoke" / "dinov2"
    dataset_root = smoke_root / "synthetic_mvtec"
    _build_synthetic_dataset(dataset_root)

    scanner = MvtecDatasetScanner()
    dataset = scanner.scan(dataset_root, dataset_name="Synthetic MVTec Smoke")
    dataset_path = smoke_root / "dataset_manifest.json"
    scanner.save(dataset, dataset_path)
    embeddings = DinoV2Embedder(
        PROJECT_ROOT / "artifacts",
        model_id="facebook/dinov2-small",
        device="cuda:0",
        batch_size=4,
    ).extract(dataset, category="bottle")
    support = plan_support_set(
        dataset,
        category="bottle",
        protocol="pool_compression_m4",
        strategy="k_center",
        shots=2,
        seed=11,
        candidate_pool_size=4,
        embeddings=embeddings.embeddings,
        feature_extractor=(
            f"{embeddings.model_id}@{embeddings.resolved_revision or 'floating'}"
        ),
    )
    view = DatasetViewBuilder(PROJECT_ROOT / "artifacts").build(dataset, support)
    print(
        json.dumps(
            {
                "purpose": "synthetic_pipeline_smoke_only",
                "scientific_result": False,
                "dataset_manifest": str(dataset_path),
                "dataset_digest": dataset.digest,
                "embedding_manifest": embeddings.manifest_path,
                "embedding_digest": embeddings.digest,
                "embedding_dimension": len(next(iter(embeddings.embeddings.values()))),
                "device": embeddings.device,
                "support_digest": support.digest,
                "selected_files": support.selected_files,
                "selection_geometry": support.selection_metadata,
                "dataset_view": view.view_root,
                "dataset_view_digest": view.digest,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _build_synthetic_dataset(root: Path) -> None:
    training = root / "bottle" / "train" / "good"
    test_good = root / "bottle" / "test" / "good"
    test_bad = root / "bottle" / "test" / "broken"
    masks = root / "bottle" / "ground_truth" / "broken"
    for directory in (training, test_good, test_bad, masks):
        directory.mkdir(parents=True, exist_ok=True)

    for index, color in enumerate(((40, 100, 180), (60, 120, 190), (80, 90, 170), (45, 135, 165))):
        path = training / f"{index:03}.png"
        if not path.exists():
            image = Image.new("RGB", (224, 224), color)
            draw = ImageDraw.Draw(image)
            draw.ellipse((62 + index, 28, 162 + index, 202), outline="white", width=8)
            image.save(path)

    good_path = test_good / "100.png"
    if not good_path.exists():
        Image.new("RGB", (224, 224), (55, 110, 180)).save(good_path)
    bad_path = test_bad / "101.png"
    mask_path = masks / "101_mask.png"
    if not bad_path.exists():
        image = Image.new("RGB", (224, 224), (55, 110, 180))
        ImageDraw.Draw(image).rectangle((90, 90, 135, 135), fill=(230, 30, 30))
        image.save(bad_path)
    if not mask_path.exists():
        mask = Image.new("L", (224, 224), 0)
        ImageDraw.Draw(mask).rectangle((90, 90, 135, 135), fill=255)
        mask.save(mask_path)


if __name__ == "__main__":
    main()
