from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from fsad_scientist.config import PROJECT_ROOT
from fsad_scientist.datasets.scanner import MvtecDatasetScanner
from fsad_scientist.datasets.view import DatasetViewBuilder
from fsad_scientist.domain.models import ExperimentRun
from fsad_scientist.experiments.adapters import MethodRegistry
from fsad_scientist.experiments.runner import ExperimentRunner
from fsad_scientist.experiments.support_selection import plan_support_set


async def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    smoke_root = PROJECT_ROOT / "artifacts" / "smoke" / "anomalydino"
    dataset_root = smoke_root / "synthetic_mvtec_v2"
    _build_dataset(dataset_root)
    dataset = MvtecDatasetScanner().scan(
        dataset_root,
        dataset_name="Synthetic MVTec Smoke",
    )
    support = plan_support_set(
        dataset,
        category="bottle",
        protocol="strict_k_shot",
        strategy="random",
        shots=2,
        seed=11,
    )
    view = DatasetViewBuilder(PROJECT_ROOT / "artifacts").build(dataset, support)
    experiment = ExperimentRun(
        plan_id="synthetic_smoke_plan",
        hypothesis_id="synthetic_smoke_hypothesis",
        protocol=support.protocol,
        dataset=support.dataset,
        category=support.category,
        detector="anomalydino",
        selection_strategy=support.strategy,
        shots=support.shots,
        seed=support.seed,
    )
    output_dir = smoke_root / "runs" / experiment.id
    command = MethodRegistry(PROJECT_ROOT).get(experiment.detector).build_command(
        experiment,
        dataset_view=Path(view.view_root),
        output_dir=output_dir,
        device="cuda:0",
    )
    execution = await ExperimentRunner(
        project_root=PROJECT_ROOT,
        artifact_root=PROJECT_ROOT / "artifacts",
    ).execute(
        experiment,
        command,
        output_dir=output_dir,
        dataset_view=view,
        support_manifest=support,
        timeout_seconds=1800,
    )
    summary = {
        "purpose": "synthetic_detector_integration_smoke_only",
        "scientific_result": False,
        "status": execution.status,
        "run_id": execution.run_id,
        "output_dir": execution.output_dir,
        "metrics_are_synthetic_smoke_only": (
            execution.normalized_result.metrics if execution.normalized_result else None
        ),
        "error": execution.error,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if execution.status != "succeeded":
        stderr = Path(execution.stderr_path)
        if stderr.is_file():
            print(stderr.read_text(encoding="utf-8", errors="replace")[-4000:])
        raise SystemExit(1)


def _build_dataset(root: Path) -> None:
    paths = {
        "train": root / "bottle" / "train" / "good",
        "good": root / "bottle" / "test" / "good",
        "bad": root / "bottle" / "test" / "broken_large",
        "mask": root / "bottle" / "ground_truth" / "broken_large",
    }
    for directory in paths.values():
        directory.mkdir(parents=True, exist_ok=True)
    for index, color in enumerate(((40, 100, 180), (75, 125, 175), (45, 135, 165))):
        target = paths["train"] / f"{index:03}.png"
        if not target.exists():
            image = Image.new("RGB", (224, 224), color)
            ImageDraw.Draw(image).ellipse((62, 28, 162, 202), outline="white", width=8)
            image.save(target)
    good = paths["good"] / "100.png"
    if not good.exists():
        Image.new("RGB", (224, 224), (55, 110, 180)).save(good)
    bad = paths["bad"] / "101.png"
    if not bad.exists():
        image = Image.new("RGB", (224, 224), (55, 110, 180))
        ImageDraw.Draw(image).rectangle((90, 90, 135, 135), fill=(230, 30, 30))
        image.save(bad)
    mask = paths["mask"] / "101_mask.png"
    if not mask.exists():
        image = Image.new("L", (224, 224), 0)
        ImageDraw.Draw(image).rectangle((90, 90, 135, 135), fill=255)
        image.save(mask)


if __name__ == "__main__":
    asyncio.run(main())
