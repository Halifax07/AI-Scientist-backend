from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from fsad_scientist.experiments.models import NormalizedExperimentResult


class ResultParseError(RuntimeError):
    pass


class ResultNormalizer:
    """Normalize pinned detector outputs to one metric vocabulary."""

    def parse(self, method: str, output_dir: Path, *, category: str) -> NormalizedExperimentResult:
        parsers = {
            "patchcore": self._parse_patchcore,
            "anomalydino": self._parse_anomalydino,
            "subspacead": self._parse_subspacead,
        }
        try:
            parser = parsers[method.casefold()]
        except KeyError as exc:
            raise ResultParseError(f"No result parser for method {method}") from exc
        result = parser(output_dir, category=category)
        _validate_metrics(result.metrics)
        return result

    @staticmethod
    def _parse_patchcore(output_dir: Path, *, category: str) -> NormalizedExperimentResult:
        source = _one_result_file(output_dir, "results.csv")
        rows = _csv_rows(source)
        row = _category_row(rows, category, aliases=("Row Names", "Category"))
        mapping = {
            "Instance AUROC": "image_auroc",
            "Full Pixel AUROC": "pixel_auroc",
            "Full PRO": "aupro",
            "Anomaly Pixel AUROC": "anomaly_pixel_auroc",
            "Anomaly PRO": "anomaly_aupro",
        }
        return NormalizedExperimentResult(
            parser="patchcore-results-csv-v1",
            metrics=_mapped_metrics(row, mapping),
            source_files=[str(source.resolve())],
        )

    @staticmethod
    def _parse_anomalydino(output_dir: Path, *, category: str) -> NormalizedExperimentResult:
        candidates = sorted(output_dir.rglob("metrics_seed=*.json"))
        if len(candidates) != 1:
            raise ResultParseError(
                f"Expected one AnomalyDINO metrics JSON, found {len(candidates)}"
            )
        source = candidates[0]
        payload = json.loads(source.read_text(encoding="utf-8"))
        row = payload.get(category)
        if not isinstance(row, dict):
            raise ResultParseError(f"AnomalyDINO output has no category {category!r}")
        mapping = {
            "classification_AUROC": "image_auroc",
            "classification_AP": "image_ap",
            "classification_F1": "image_f1",
            "seg_AUROC": "pixel_auroc",
            "seg_AUPRO": "aupro",
            "seg_F1": "pixel_f1",
        }
        return NormalizedExperimentResult(
            parser="anomalydino-metrics-json-v1",
            metrics=_mapped_metrics(row, mapping),
            source_files=[str(source.resolve())],
        )

    @staticmethod
    def _parse_subspacead(output_dir: Path, *, category: str) -> NormalizedExperimentResult:
        source = _one_result_file(output_dir, "benchmark_results.csv")
        rows = _csv_rows(source)
        row = _category_row(rows, category, aliases=("Category",))
        mapping = {
            "Image AUROC": "image_auroc",
            "Image AUPR": "image_ap",
            "Pixel AUROC": "pixel_auroc",
            "AU-PRO": "aupro",
            "Image F1": "image_f1",
            "Pixel F1": "pixel_f1",
        }
        return NormalizedExperimentResult(
            parser="subspacead-benchmark-csv-v1",
            metrics=_mapped_metrics(row, mapping),
            source_files=[str(source.resolve())],
        )


def _one_result_file(output_dir: Path, filename: str) -> Path:
    candidates = sorted(output_dir.rglob(filename))
    if len(candidates) != 1:
        raise ResultParseError(f"Expected one {filename}, found {len(candidates)}")
    return candidates[0]


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _category_row(
    rows: list[dict[str, str]],
    category: str,
    *,
    aliases: tuple[str, ...],
) -> dict[str, str]:
    for row in rows:
        value = next((row.get(alias) for alias in aliases if row.get(alias)), None)
        if value and value.casefold() in {category.casefold(), f"mvtec_{category}".casefold()}:
            return row
    if len(rows) == 1:
        return rows[0]
    raise ResultParseError(f"No result row matched category {category!r}")


def _mapped_metrics(row: dict[str, object], mapping: dict[str, str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for source_name, target_name in mapping.items():
        value = row.get(source_name)
        if value not in (None, "", "nan", "NaN", "N/A"):
            try:
                metrics[target_name] = float(value)
            except (TypeError, ValueError) as exc:
                raise ResultParseError(f"Metric {source_name} is not numeric: {value!r}") from exc
    if "image_auroc" not in metrics:
        raise ResultParseError("Normalized output is missing required image_auroc")
    return metrics


def _validate_metrics(metrics: dict[str, float]) -> None:
    for name, value in metrics.items():
        if not math.isfinite(value):
            raise ResultParseError(f"Metric {name} is not finite")
        if not 0.0 <= value <= 1.0:
            raise ResultParseError(f"Metric {name}={value} is outside [0, 1]")
