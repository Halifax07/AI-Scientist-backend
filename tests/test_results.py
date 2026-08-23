import json
from pathlib import Path

import pytest

from fsad_scientist.experiments.results import ResultNormalizer, ResultParseError
from fsad_scientist.experiments.runner import ExperimentRunner


class TestSafeCommandEnvironment:
    def test_accepts_offline_keys(self) -> None:
        environment = ExperimentRunner._build_environment(
            {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        )
        assert environment["HF_HUB_OFFLINE"] == "1"
        assert environment["TRANSFORMERS_OFFLINE"] == "1"

    def test_rejects_unknown_keys(self) -> None:
        with pytest.raises(ValueError, match="unsupported command environment"):
            ExperimentRunner._build_environment({"BOGUS_KEY": "x"})


class TestGeneratedDetectorParser:
    def test_parses_metrics_json_and_drops_non_numeric_fields(self, tmp_path: Path) -> None:
        (tmp_path / "metrics.json").write_text(
            json.dumps(
                {
                    "image_auroc": 0.91,
                    "image_ap": 0.88,
                    "category": "bottle",
                    "flag": True,
                    "test_images": ["a.png"],
                }
            ),
            encoding="utf-8",
        )
        result = ResultNormalizer().parse(
            "generated_det_nearest_ab12cd34", tmp_path, category="bottle"
        )
        assert result.metrics == {"image_auroc": 0.91, "image_ap": 0.88}
        assert result.parser == "generated-detector-metrics-json-v1"

    def test_rejects_missing_image_auroc(self, tmp_path: Path) -> None:
        (tmp_path / "metrics.json").write_text('{"image_ap": 0.88}', encoding="utf-8")
        with pytest.raises(ResultParseError, match="image_auroc"):
            ResultNormalizer().parse(
                "generated_det_nearest_ab12cd34", tmp_path, category="bottle"
            )

    def test_rejects_multiple_metrics_files(self, tmp_path: Path) -> None:
        (tmp_path / "metrics.json").write_text('{"image_auroc": 0.91}', encoding="utf-8")
        (tmp_path / "nested").mkdir()
        (tmp_path / "nested" / "metrics.json").write_text(
            '{"image_auroc": 0.92}', encoding="utf-8"
        )
        with pytest.raises(ResultParseError, match="Expected one"):
            ResultNormalizer().parse(
                "generated_det_nearest_ab12cd34", tmp_path, category="bottle"
            )

    def test_rejects_out_of_range_metric(self, tmp_path: Path) -> None:
        (tmp_path / "metrics.json").write_text('{"image_auroc": 1.7}', encoding="utf-8")
        with pytest.raises(ResultParseError, match="outside"):
            ResultNormalizer().parse(
                "generated_det_nearest_ab12cd34", tmp_path, category="bottle"
            )
