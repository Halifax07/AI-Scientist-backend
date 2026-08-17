import asyncio
from pathlib import Path

from fsad_scientist.datasets.models import DatasetViewManifest
from fsad_scientist.domain.models import ExperimentRun
from fsad_scientist.experiments.models import CommandSpec, SupportSetManifest
from fsad_scientist.experiments.runner import ExperimentRunner


def run(coro):
    return asyncio.run(coro)


def test_runner_executes_argv_only_and_normalizes_real_output(tmp_path):
    project_root = tmp_path / "project"
    artifact_root = project_root / "artifacts"
    output_dir = artifact_root / "runs" / "run_1"
    project_root.mkdir()
    script = project_root / "fake_detector.py"
    script.write_text(
        "from pathlib import Path\n"
        "Path('results.csv').write_text("
        "'Row Names,Instance AUROC,Full Pixel AUROC,Full PRO,Anomaly Pixel AUROC,Anomaly PRO\\n'"
        "+'bottle,0.91,0.82,0.73,0.80,0.70\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )
    experiment = ExperimentRun(
        plan_id="plan_1",
        hypothesis_id="h_1",
        protocol="strict_k_shot",
        dataset="MVTec AD",
        category="bottle",
        detector="patchcore",
        selection_strategy="random",
        shots=1,
        seed=0,
    )
    support = SupportSetManifest(
        dataset="MVTec AD",
        category="bottle",
        protocol="strict_k_shot",
        strategy="random",
        shots=1,
        seed=0,
        candidate_pool_files=["bottle/train/good/000.png"],
        selected_files=["bottle/train/good/000.png"],
        feature_extractor="none",
        digest="a" * 64,
    )
    view = DatasetViewManifest(
        dataset_digest="b" * 64,
        support_manifest_digest=support.digest,
        category="bottle",
        source_root=str(tmp_path / "dataset"),
        view_root=str(artifact_root / "dataset_views" / "view"),
        materialization="copy",
        files=["bottle/train/good/000.png"],
        digest="c" * 64,
    )
    command = CommandSpec(
        method="patchcore",
        executable="python",
        args=[str(script)],
        cwd=output_dir,
    )

    result = run(
        ExperimentRunner(
            project_root=project_root,
            artifact_root=artifact_root,
        ).execute(
            experiment,
            command,
            output_dir=output_dir,
            dataset_view=view,
            support_manifest=support,
            timeout_seconds=30,
        )
    )

    assert result.status == "succeeded"
    assert result.normalized_result is not None
    assert result.normalized_result.metrics["image_auroc"] == 0.91
    assert Path(result.stdout_path).is_file()
    assert (output_dir / "execution.json").is_file()
