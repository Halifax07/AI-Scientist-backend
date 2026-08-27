from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PIXEL_METRIC_PROTOCOL = "streaming-binned-pixel-v1"
PIXEL_METRIC_PROTOCOL_VERSION = "1"
PIXEL_HISTOGRAM_BINS = 65536
PRO_THRESHOLD_COUNT = 4096
PIXEL_METRIC_CONFIG = {
    "histogram_bins": PIXEL_HISTOGRAM_BINS,
    "pro_threshold_count": PRO_THRESHOLD_COUNT,
    "prediction_dtype": "float32",
}


def evaluate_segmentation(
    gt_filenames: Sequence[str | os.PathLike[str] | None],
    prediction_filenames: Sequence[str | os.PathLike[str]],
    *,
    pro_integration_limit: float = 0.3,
    delete_tiff_files: bool = True,
    histogram_bins: int = PIXEL_HISTOGRAM_BINS,
    pro_threshold_count: int = PRO_THRESHOLD_COUNT,
) -> tuple[float, float, float]:
    """Evaluate AnomalyDINO maps without retaining the complete pixel corpus.

    The two passes are over filenames only. Each pass loads one prediction and
    one uint8 mask, then releases both before moving to the next pair.
    """
    gt_paths = tuple(gt_filenames)
    prediction_paths = tuple(prediction_filenames)
    if len(gt_paths) != len(prediction_paths) or not prediction_paths:
        raise ValueError("ground truth and prediction filenames must be non-empty and aligned")
    if not 0 < pro_integration_limit <= 1:
        raise ValueError("pro_integration_limit must be in (0, 1]")
    if histogram_bins < 2 or pro_threshold_count < 2:
        raise ValueError("histogram_bins and pro_threshold_count must be at least 2")

    score_min, score_max = _scan_score_range(prediction_paths)
    pixel_positive = np.zeros(histogram_bins, dtype=np.uint64)
    pixel_negative = np.zeros(histogram_bins, dtype=np.uint64)
    pro_false_positive = np.zeros(pro_threshold_count, dtype=np.float64)
    pro_overlap = np.zeros(pro_threshold_count, dtype=np.float64)
    normal_pixel_count = 0
    ground_truth_region_count = 0
    threshold_edges = np.linspace(
        score_min,
        score_max,
        num=pro_threshold_count,
        dtype=np.float64,
    )

    print("Stream pixel maps and accumulate binned metrics...")
    for gt_name, prediction_name in zip(gt_paths, prediction_paths, strict=True):
        prediction = _read_prediction(prediction_name)
        ground_truth = _read_ground_truth(gt_name, prediction.shape)
        flat_prediction = prediction.reshape(-1)
        flat_ground_truth = ground_truth.reshape(-1)
        bin_indices = _score_bin_indices(
            flat_prediction,
            score_min=score_min,
            score_max=score_max,
            bin_count=histogram_bins,
        )
        positive = flat_ground_truth != 0
        pixel_positive += np.bincount(bin_indices[positive], minlength=histogram_bins).astype(
            np.uint64, copy=False
        )
        pixel_negative += np.bincount(bin_indices[~positive], minlength=histogram_bins).astype(
            np.uint64, copy=False
        )

        labeled_ground_truth, region_count = _label_components(
            ground_truth,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        ground_truth_region_count += int(region_count)
        normal_mask = labeled_ground_truth == 0
        normal_pixel_count += int(np.count_nonzero(normal_mask))
        pro_false_positive += _threshold_counts(
            flat_prediction[normal_mask.reshape(-1)], threshold_edges
        )
        for region_id in range(1, int(region_count) + 1):
            region_mask = labeled_ground_truth == region_id
            region_scores = prediction[region_mask]
            pro_overlap += _threshold_counts(region_scores, threshold_edges) / region_scores.size

    auroc, pixel_f1 = _histogram_roc_and_f1(pixel_positive, pixel_negative)
    aupro = _integrate_pro(
        pro_false_positive,
        pro_overlap,
        normal_pixel_count=normal_pixel_count,
        ground_truth_region_count=ground_truth_region_count,
        score_min=score_min,
        score_max=score_max,
        integration_limit=pro_integration_limit,
    )

    if delete_tiff_files:
        for prediction_name in prediction_paths:
            _resolve_tiff_path(prediction_name).unlink()

    print(f"AU-PRO (FPR limit: {pro_integration_limit}): {aupro}", end=" -- ")
    print(f"AUROC (pixel-level): {auroc}", end=" -- ")
    print(f"F1 (pixel-level): {pixel_f1}")
    return aupro, auroc, pixel_f1


def inject_metric_protocol(output_dir: str | os.PathLike[str], seed: int | None) -> Path:
    """Annotate the upstream metrics file without changing numeric fields."""
    output_path = Path(output_dir)
    metric_name = f"metrics_seed={seed}.json" if seed is not None else "metrics.json"
    metric_path = output_path / metric_name
    payload = json.loads(metric_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("AnomalyDINO metrics output must be a JSON object")
    payload["metric_protocol"] = PIXEL_METRIC_PROTOCOL
    payload["metric_protocol_version"] = PIXEL_METRIC_PROTOCOL_VERSION
    payload["pixel_metric_config"] = PIXEL_METRIC_CONFIG
    metric_path.write_text(json.dumps(payload, indent=4), encoding="utf-8")
    return metric_path


def install_post_eval_overrides(post_eval_module: Any) -> None:
    """Install the project evaluator while preserving upstream call signatures."""
    if getattr(post_eval_module, "_fsad_streaming_pixel_metrics", False):
        return
    original_eval_finished_run = post_eval_module.eval_finished_run

    def eval_finished_run(*args: Any, **kwargs: Any) -> Any:
        result = original_eval_finished_run(*args, **kwargs)
        output_dir = kwargs.get("output_dir")
        if output_dir is None and len(args) >= 4:
            output_dir = args[3]
        if output_dir is not None:
            seed = kwargs.get("seed")
            if seed is None and len(args) >= 5:
                seed = args[4]
            inject_metric_protocol(output_dir, seed)
        return result

    post_eval_module.eval_segmentation = evaluate_segmentation
    post_eval_module.eval_finished_run = eval_finished_run
    post_eval_module._fsad_streaming_pixel_metrics = True


def _scan_score_range(prediction_paths: Iterable[str | os.PathLike[str]]) -> tuple[float, float]:
    score_min = np.inf
    score_max = -np.inf
    for prediction_name in prediction_paths:
        prediction = _read_prediction(prediction_name)
        if not np.isfinite(prediction).all():
            raise ValueError(f"Prediction contains non-finite values: {prediction_name}")
        score_min = min(score_min, float(np.min(prediction)))
        score_max = max(score_max, float(np.max(prediction)))
    if not np.isfinite(score_min) or not np.isfinite(score_max):
        raise ValueError("Prediction maps contain no finite values")
    return score_min, score_max


def _read_prediction(prediction_name: str | os.PathLike[str]) -> np.ndarray:
    prediction_path = _resolve_tiff_path(prediction_name)
    try:
        import tifffile
    except ModuleNotFoundError:
        with Image.open(prediction_path) as image:
            prediction = np.asarray(image, dtype=np.float32)
    else:
        prediction = np.asarray(tifffile.imread(prediction_path), dtype=np.float32)
    if prediction.ndim != 2:
        raise ValueError(f"Prediction must be a 2D map: {prediction_name}")
    return prediction


def _read_ground_truth(
    gt_name: str | os.PathLike[str] | None,
    shape: tuple[int, ...],
) -> np.ndarray:
    if gt_name is None:
        return np.zeros(shape, dtype=np.uint8)
    with Image.open(gt_name) as image:
        ground_truth = (np.asarray(image) > 0).astype(np.uint8, copy=False)
    if ground_truth.shape != shape:
        raise ValueError(
            f"Ground truth shape {ground_truth.shape} does not match prediction {shape}"
        )
    return ground_truth


def _label_components(mask: np.ndarray, structure: np.ndarray) -> tuple[np.ndarray, int]:
    try:
        from scipy.ndimage import label as scipy_label
    except ModuleNotFoundError:
        return _label_components_fallback(mask)
    return scipy_label(mask, structure=structure)


def _label_components_fallback(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels = np.zeros(mask.shape, dtype=np.int32)
    height, width = mask.shape
    region_count = 0
    for row in range(height):
        for column in range(width):
            if not mask[row, column] or labels[row, column] != 0:
                continue
            region_count += 1
            labels[row, column] = region_count
            pending = [(row, column)]
            while pending:
                current_row, current_column = pending.pop()
                for row_offset in (-1, 0, 1):
                    for column_offset in (-1, 0, 1):
                        neighbor_row = current_row + row_offset
                        neighbor_column = current_column + column_offset
                        if (
                            0 <= neighbor_row < height
                            and 0 <= neighbor_column < width
                            and mask[neighbor_row, neighbor_column]
                            and labels[neighbor_row, neighbor_column] == 0
                        ):
                            labels[neighbor_row, neighbor_column] = region_count
                            pending.append((neighbor_row, neighbor_column))
    return labels, region_count


def _resolve_tiff_path(prediction_name: str | os.PathLike[str]) -> Path:
    path = Path(prediction_name)
    if path.suffix.casefold() in {".tif", ".tiff"}:
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = [path.with_suffix(extension) for extension in (".tiff", ".tif")]
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        raise OSError(f"Found multiple TIFF files for {path}")
    raise FileNotFoundError(f"Could not find TIFF prediction for {path}")


def _score_bin_indices(
    scores: np.ndarray,
    *,
    score_min: float,
    score_max: float,
    bin_count: int,
) -> np.ndarray:
    if score_max == score_min:
        return np.zeros(scores.shape, dtype=np.int32)
    scaled = (scores - score_min) * (bin_count / (score_max - score_min))
    indices = np.floor(scaled).astype(np.int32, copy=False)
    np.clip(indices, 0, bin_count - 1, out=indices)
    return indices


def _threshold_counts(scores: np.ndarray, ascending_thresholds: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return np.zeros(ascending_thresholds.size, dtype=np.float64)
    threshold_indices = np.searchsorted(ascending_thresholds, scores, side="right")
    counts = np.bincount(threshold_indices, minlength=ascending_thresholds.size + 1)
    return np.cumsum(counts[:0:-1], dtype=np.float64)[::-1]


def _histogram_roc_and_f1(
    positive_counts: np.ndarray,
    negative_counts: np.ndarray,
) -> tuple[float, float]:
    positive_total = int(positive_counts.sum())
    negative_total = int(negative_counts.sum())
    if positive_total == 0 or negative_total == 0:
        raise ValueError("pixel metrics require both normal and anomalous pixels")

    true_positive = 0
    false_positive = 0
    auroc = 0.0
    best_f1 = 0.0
    for index in range(positive_counts.size - 1, -1, -1):
        next_true_positive = true_positive + int(positive_counts[index])
        next_false_positive = false_positive + int(negative_counts[index])
        auroc += ((true_positive + next_true_positive) / (2.0 * positive_total)) * (
            (next_false_positive - false_positive) / negative_total
        )
        true_positive = next_true_positive
        false_positive = next_false_positive
        predicted_positive = true_positive + false_positive
        if predicted_positive:
            precision = true_positive / predicted_positive
            recall = true_positive / positive_total
            denominator = precision + recall
            if denominator:
                best_f1 = max(best_f1, 2.0 * precision * recall / denominator)
    return float(auroc), float(best_f1)


def _integrate_pro(
    false_positive_counts: np.ndarray,
    overlap_sums: np.ndarray,
    *,
    normal_pixel_count: int,
    ground_truth_region_count: int,
    score_min: float,
    score_max: float,
    integration_limit: float,
) -> float:
    if normal_pixel_count == 0 or ground_truth_region_count == 0:
        return 0.0
    false_positive_rates = false_positive_counts[::-1] / normal_pixel_count
    pros = np.clip(overlap_sums[::-1] / ground_truth_region_count, 0.0, 1.0)
    area = 0.0
    previous_fpr = 0.0
    previous_pro = 0.0
    for false_positive_rate, pro in zip(false_positive_rates, pros, strict=True):
        current_fpr = min(1.0, max(previous_fpr, float(false_positive_rate)))
        current_pro = min(1.0, max(0.0, float(pro)))
        if current_fpr <= integration_limit:
            area += 0.5 * (previous_pro + current_pro) * (current_fpr - previous_fpr)
            previous_fpr = current_fpr
            previous_pro = current_pro
            continue
        if current_fpr > previous_fpr:
            interpolated_pro = previous_pro + (
                (current_pro - previous_pro)
                * (integration_limit - previous_fpr)
                / (current_fpr - previous_fpr)
            )
            area += 0.5 * (previous_pro + interpolated_pro) * (integration_limit - previous_fpr)
        return float(area / integration_limit)

    if previous_fpr < integration_limit and score_max != score_min:
        area += 0.5 * (previous_pro + 1.0) * (integration_limit - previous_fpr)
    return float(area / integration_limit)
