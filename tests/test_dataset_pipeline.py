from pathlib import Path

from fsad_scientist.datasets.scanner import MvtecDatasetScanner
from fsad_scientist.datasets.view import DatasetViewBuilder
from fsad_scientist.domain.models import ExperimentRun
from fsad_scientist.experiments.preparation import ExperimentPreparationService
from fsad_scientist.experiments.support_selection import plan_support_set


def _write(root: Path, relative: str, payload: bytes) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _fake_mvtec(root: Path) -> None:
    _write(root, "bottle/train/good/000.png", b"train-0")
    _write(root, "bottle/train/good/001.png", b"train-1")
    _write(root, "bottle/train/good/002.png", b"train-2")
    _write(root, "bottle/test/good/100.png", b"test-good")
    _write(root, "bottle/test/broken/101.png", b"test-broken")
    _write(root, "bottle/ground_truth/broken/101_mask.png", b"mask")


def test_scan_plan_and_view_never_expose_unselected_train_images(tmp_path):
    dataset_root = tmp_path / "mvtec"
    _fake_mvtec(dataset_root)
    dataset = MvtecDatasetScanner().scan(dataset_root)

    assert dataset.is_valid
    assert len(dataset.support_candidates("bottle")) == 3

    support = plan_support_set(
        dataset,
        category="bottle",
        protocol="strict_k_shot",
        strategy="random",
        shots=1,
        seed=7,
    )
    view = DatasetViewBuilder(tmp_path / "artifacts").build(dataset, support)
    view_root = Path(view.view_root)

    visible_train = list((view_root / "bottle" / "train" / "good").glob("*.png"))
    assert len(visible_train) == 1
    assert visible_train[0].relative_to(view_root).as_posix() in support.selected_files
    assert (view_root / "bottle" / "test" / "broken" / "101.png").is_file()
    assert (view_root / "bottle" / "ground_truth" / "broken" / "101_mask.png").is_file()


def test_scanner_detects_train_test_content_leakage(tmp_path):
    dataset_root = tmp_path / "mvtec"
    _fake_mvtec(dataset_root)
    (dataset_root / "bottle/test/good/100.png").write_bytes(b"train-0")

    dataset = MvtecDatasetScanner().scan(dataset_root)

    assert not dataset.is_valid
    assert any(issue.code == "TRAIN_TEST_DUPLICATE" for issue in dataset.issues)


def test_run_preparation_freezes_support_and_view_artifacts(tmp_path):
    dataset_root = tmp_path / "mvtec"
    _fake_mvtec(dataset_root)
    dataset = MvtecDatasetScanner().scan(dataset_root)
    manifest_path = tmp_path / "artifacts" / "datasets" / "dataset.json"
    MvtecDatasetScanner.save(dataset, manifest_path)
    experiment = ExperimentRun(
        plan_id="plan",
        hypothesis_id="hypothesis",
        protocol="strict_k_shot",
        dataset="MVTec AD",
        category="bottle",
        detector="anomalydino",
        selection_strategy="random",
        shots=2,
        seed=4,
    )

    prepared = ExperimentPreparationService(tmp_path / "artifacts").prepare(
        project_id="project",
        run=experiment,
        dataset=dataset,
        dataset_manifest_path=manifest_path,
    )

    assert Path(prepared.support_manifest_path).is_file()
    assert Path(prepared.dataset_view_manifest_path).is_file()
    assert len(list((Path(prepared.dataset_view_root) / "bottle/train/good").glob("*.png"))) == 2
