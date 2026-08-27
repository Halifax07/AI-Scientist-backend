import shutil
import stat
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


def test_view_build_reuses_view_published_by_concurrent_builder(tmp_path, monkeypatch):
    dataset_root = tmp_path / "mvtec"
    _fake_mvtec(dataset_root)
    dataset = MvtecDatasetScanner().scan(dataset_root)
    support = plan_support_set(
        dataset,
        category="bottle",
        protocol="strict_k_shot",
        strategy="random",
        shots=1,
        seed=7,
    )

    original_replace = Path.replace
    published = False

    def publish_then_report_windows_race(source, destination):
        nonlocal published
        if not published:
            published = True
            shutil.copytree(source, destination)
            raise PermissionError(13, "access denied", str(destination))
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", publish_then_report_windows_race)

    view = DatasetViewBuilder(tmp_path / "artifacts").build(dataset, support)

    assert Path(view.view_root, "fsad_view_manifest.json").is_file()
    assert not list((tmp_path / "artifacts" / "dataset_views").rglob("*.building-*"))


def test_view_build_retries_transient_windows_publish_error(tmp_path, monkeypatch):
    dataset_root = tmp_path / "mvtec"
    _fake_mvtec(dataset_root)
    dataset = MvtecDatasetScanner().scan(dataset_root)
    support = plan_support_set(
        dataset,
        category="bottle",
        protocol="strict_k_shot",
        strategy="random",
        shots=1,
        seed=7,
    )

    original_replace = Path.replace
    attempts = 0

    def fail_then_publish(source, destination):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(13, "transient access denied", str(destination))
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", fail_then_publish)
    monkeypatch.setattr("fsad_scientist.datasets.view.time.sleep", lambda _delay: None)

    view = DatasetViewBuilder(tmp_path / "artifacts").build(dataset, support)

    assert attempts == 3
    assert Path(view.view_root, "fsad_view_manifest.json").is_file()
    assert not list((tmp_path / "artifacts" / "dataset_views").rglob("*.building-*"))


def test_view_build_cleanup_does_not_hide_publish_error(tmp_path, monkeypatch):
    dataset_root = tmp_path / "mvtec"
    _fake_mvtec(dataset_root)
    dataset = MvtecDatasetScanner().scan(dataset_root)
    support = plan_support_set(
        dataset,
        category="bottle",
        protocol="strict_k_shot",
        strategy="random",
        shots=1,
        seed=7,
    )
    publish_error = PermissionError(13, "access denied")

    def fail_publish(_source, _destination):
        raise publish_error

    def fail_cleanup(*_args, **_kwargs):
        raise PermissionError(13, "read-only cleanup")

    monkeypatch.setattr(Path, "replace", fail_publish)
    monkeypatch.setattr("fsad_scientist.datasets.view.shutil.rmtree", fail_cleanup)

    try:
        DatasetViewBuilder(tmp_path / "artifacts").build(dataset, support)
    except PermissionError as exc:
        assert exc is publish_error
    else:
        raise AssertionError("publish error was unexpectedly swallowed")


def test_readonly_source_is_copied_without_mutating_dataset(tmp_path, monkeypatch):
    dataset_root = tmp_path / "mvtec"
    _fake_mvtec(dataset_root)
    dataset = MvtecDatasetScanner().scan(dataset_root)
    support = plan_support_set(
        dataset,
        category="bottle",
        protocol="strict_k_shot",
        strategy="random",
        shots=1,
        seed=7,
    )
    source = dataset_root / Path(support.selected_files[0])
    source.chmod(stat.S_IREAD)
    original_content = source.read_bytes()
    original_mode = source.stat().st_mode

    original_replace = Path.replace
    published = False

    def publish_then_report_windows_race(source_path, destination):
        nonlocal published
        if not published:
            published = True
            shutil.copytree(source_path, destination)
            raise PermissionError(13, "access denied", str(destination))
        return original_replace(source_path, destination)

    monkeypatch.setattr(Path, "replace", publish_then_report_windows_race)

    view = DatasetViewBuilder(tmp_path / "artifacts").build(dataset, support)

    try:
        selected_view_file = Path(view.view_root) / Path(support.selected_files[0])
        assert Path(view.view_root, "fsad_view_manifest.json").is_file()
        assert view.materialization == "mixed"
        assert not source.samefile(selected_view_file)
        assert source.read_bytes() == original_content
        assert source.stat().st_mode == original_mode
        assert not list(
            (tmp_path / "artifacts" / "dataset_views").rglob("*.building-*")
        )
    finally:
        source.chmod(original_mode | stat.S_IWRITE)


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
