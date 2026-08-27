from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from fsad_scientist.experiments.results import ResultNormalizer
from fsad_scientist.integrations.anomalydino_pixel_metrics import (
    PIXEL_METRIC_CONFIG,
    PIXEL_METRIC_PROTOCOL,
    PIXEL_METRIC_PROTOCOL_VERSION,
    _label_components,
    _read_ground_truth,
    evaluate_segmentation,
    inject_metric_protocol,
    install_post_eval_overrides,
)


def _write_fixture(
    tmp_path: Path,
) -> tuple[list[Path | None], list[Path], list[np.ndarray], list[np.ndarray]]:
    predictions = [
        np.array([[0.05, 0.10, 0.20, 0.30], [0.15, 0.25, 0.35, 0.40]], dtype=np.float32),
        np.array([[0.80, 0.90, 0.85, 0.20], [0.75, 0.95, 0.10, 0.15]], dtype=np.float32),
    ]
    masks = [
        np.zeros((2, 4), dtype=np.uint8),
        np.array([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=np.uint8),
    ]
    gt_paths: list[Path | None] = [None, tmp_path / "mask.png"]
    prediction_bases = [tmp_path / "good", tmp_path / "defect"]
    for base, prediction in zip(prediction_bases, predictions, strict=True):
        Image.fromarray(prediction, mode="F").save(str(base) + ".tiff", format="TIFF")
        np.save(str(base) + ".npy", prediction)
    Image.fromarray(masks[1] * 255).save(gt_paths[1])
    return gt_paths, prediction_bases, predictions, masks


def _exact_pixel_metrics(
    predictions: list[np.ndarray], masks: list[np.ndarray]
) -> tuple[float, float]:
    scores = np.concatenate([prediction.reshape(-1) for prediction in predictions])
    labels = np.concatenate([mask.reshape(-1) for mask in masks])
    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    ordered_labels = labels[order]
    positive_total = int(np.count_nonzero(labels))
    negative_total = int(labels.size - positive_total)
    true_positive = 0
    false_positive = 0
    auroc = 0.0
    f1 = 0.0
    start = 0
    while start < ordered_scores.size:
        end = start + 1
        while end < ordered_scores.size and ordered_scores[end] == ordered_scores[start]:
            end += 1
        group_positive = int(np.count_nonzero(ordered_labels[start:end]))
        group_negative = end - start - group_positive
        next_true_positive = true_positive + group_positive
        next_false_positive = false_positive + group_negative
        auroc += ((true_positive + next_true_positive) / (2 * positive_total)) * (
            (next_false_positive - false_positive) / negative_total
        )
        true_positive = next_true_positive
        false_positive = next_false_positive
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / positive_total
        f1 = max(f1, 2 * precision * recall / (precision + recall))
        start = end
    return float(auroc), float(f1)


def _exact_aupro(
    predictions: list[np.ndarray], masks: list[np.ndarray], integration_limit: float
) -> float:
    structure = np.ones((3, 3), dtype=np.uint8)
    normal_count = sum(int(np.count_nonzero(mask == 0)) for mask in masks)
    regions: list[np.ndarray] = []
    for mask in masks:
        labeled, region_count = _label_components(mask, structure=structure)
        regions.extend([labeled == region_id for region_id in range(1, int(region_count) + 1)])
    scores = np.unique(np.concatenate([prediction.reshape(-1) for prediction in predictions]))
    scores = scores[::-1]
    fprs = [0.0]
    pros = [0.0]
    for threshold in scores:
        false_positive = 0
        for prediction, mask in zip(predictions, masks, strict=True):
            false_positive += int(np.count_nonzero((prediction >= threshold) & (mask == 0)))
        fprs.append(false_positive / normal_count)
        pros.append(
            sum(
                np.count_nonzero((prediction >= threshold) & region) / np.count_nonzero(region)
                for prediction, mask in zip(predictions, masks, strict=True)
                for region in _regions_for_mask(mask, structure)
            )
            / len(regions)
        )
    fprs.append(1.0)
    pros.append(1.0)
    area = 0.0
    for left_x, right_x, left_y, right_y in zip(fprs, fprs[1:], pros, pros[1:], strict=True):
        if left_x >= integration_limit:
            break
        right = min(right_x, integration_limit)
        if right > left_x:
            right_y = (
                right_y
                if right == right_x
                else left_y + (right_y - left_y) * (right - left_x) / (right_x - left_x)
            )
            area += 0.5 * (left_y + right_y) * (right - left_x)
        if right_x >= integration_limit:
            break
    return area / integration_limit


def _regions_for_mask(mask: np.ndarray, structure: np.ndarray) -> list[np.ndarray]:
    labeled, region_count = _label_components(mask, structure=structure)
    return [labeled == region_id for region_id in range(1, int(region_count) + 1)]


def test_streaming_metrics_match_exact_metrics_and_do_not_sort(monkeypatch, tmp_path: Path) -> None:
    gt_paths, prediction_bases, predictions, masks = _write_fixture(tmp_path)
    exact_auroc, exact_f1 = _exact_pixel_metrics(predictions, masks)
    exact_aupro = _exact_aupro(predictions, masks, 0.3)

    def fail_argsort(*args, **kwargs):
        raise AssertionError("global argsort is forbidden in streaming evaluation")

    monkeypatch.setattr(np, "argsort", fail_argsort)
    actual_aupro, actual_auroc, actual_f1 = evaluate_segmentation(
        gt_paths,
        prediction_bases,
        pro_integration_limit=0.3,
        histogram_bins=32,
        pro_threshold_count=64,
        delete_tiff_files=False,
    )

    assert abs(actual_auroc - exact_auroc) <= 0.03
    assert abs(actual_f1 - exact_f1) <= 0.03
    assert abs(actual_aupro - exact_aupro) <= 0.03


def test_default_binning_matches_exact_metrics_on_small_fixture(tmp_path: Path) -> None:
    gt_paths, prediction_bases, predictions, masks = _write_fixture(tmp_path)
    exact_auroc, exact_f1 = _exact_pixel_metrics(predictions, masks)
    exact_aupro = _exact_aupro(predictions, masks, 0.3)

    actual = evaluate_segmentation(gt_paths, prediction_bases, delete_tiff_files=False)

    assert actual == (exact_aupro, exact_auroc, exact_f1)


def test_evaluator_has_no_full_map_container_or_global_sort() -> None:
    source = inspect.getsource(evaluate_segmentation)
    assert "np.argsort" not in source
    assert "np.array(" not in source
    assert "predictions.append" not in source
    assert "ground_truth.append" not in source


def test_success_deletes_tiffs_but_keeps_npy_and_normal_gt_is_uint8(tmp_path: Path) -> None:
    gt_paths, prediction_bases, _, _ = _write_fixture(tmp_path)
    evaluate_segmentation(gt_paths, prediction_bases, delete_tiff_files=True)

    assert not (tmp_path / "good.tiff").exists()
    assert not (tmp_path / "defect.tiff").exists()
    assert (tmp_path / "good.npy").exists()
    assert (tmp_path / "defect.npy").exists()
    assert _read_ground_truth(None, (2, 4)).dtype == np.uint8


def test_protocol_metadata_is_non_numeric_and_result_normalizer_ignores_it(tmp_path: Path) -> None:
    metric_path = tmp_path / "metrics_seed=7.json"
    metric_path.write_text(
        json.dumps({"bottle": {"classification_AUROC": 0.9, "seg_AUROC": 0.8}}),
        encoding="utf-8",
    )
    inject_metric_protocol(tmp_path, 7)
    payload = json.loads(metric_path.read_text(encoding="utf-8"))

    assert payload["metric_protocol"] == PIXEL_METRIC_PROTOCOL
    assert payload["metric_protocol_version"] == PIXEL_METRIC_PROTOCOL_VERSION
    assert payload["pixel_metric_config"] == PIXEL_METRIC_CONFIG
    assert isinstance(payload["metric_protocol"], str)
    assert isinstance(payload["metric_protocol_version"], str)
    normalized = ResultNormalizer().parse("anomalydino", tmp_path, category="bottle")
    assert normalized.metrics == {"image_auroc": 0.9, "pixel_auroc": 0.8}


def test_post_eval_override_keeps_upstream_output_shape_and_adds_protocol(tmp_path: Path) -> None:
    def original_eval_finished_run(*args, **kwargs):
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics_seed=3.json").write_text(
            json.dumps({"bottle": {"seg_AUROC": 0.7}}), encoding="utf-8"
        )
        return "upstream-result"

    module = SimpleNamespace(
        eval_segmentation=lambda *args, **kwargs: (0.0, 0.0, 0.0),
        eval_finished_run=original_eval_finished_run,
    )
    install_post_eval_overrides(module)
    assert module.eval_segmentation is evaluate_segmentation
    assert module.eval_finished_run(output_dir=tmp_path, seed=3) == "upstream-result"
    payload = json.loads((tmp_path / "metrics_seed=3.json").read_text(encoding="utf-8"))
    assert payload["metric_protocol"] == PIXEL_METRIC_PROTOCOL
