import asyncio
import hashlib
from pathlib import Path

import pytest

from fsad_scientist.agents.mock_runtime import MOCK_DETECTOR_SOURCE
from fsad_scientist.domain.models import MethodImplementation
from fsad_scientist.experiments.detector_runner import (
    DETECTOR_TEMPLATE_FOOTER,
    DETECTOR_TEMPLATE_HEADER,
    _write_detector_once,
    assemble_detector_file,
    run_detector_smoke,
)

DETECTOR_SOURCE = (
    "import numpy as np\n"
    "\n"
    "\n"
    "def anomaly_score(image, support_images, seed):\n"
    "    differences = [image.astype(np.float32) - support.astype(np.float32)\n"
    "                   for support in support_images]\n"
    "    distances = [float(np.mean(difference ** 2)) for difference in differences]\n"
    "    return 1.0 / (1.0 + min(distances))\n"
)


def _implementation(source: str = DETECTOR_SOURCE, **overrides) -> MethodImplementation:
    assembled = assemble_detector_file(source)
    digest = hashlib.sha256(assembled.encode("utf-8")).hexdigest()
    fields = {
        "name": "generated_det_nearest_ab12cd34",
        "kind": "detector",
        "hypothesis_id": "hypothesis_x",
        "source_code": source,
        "code_digest": digest,
    }
    fields.update(overrides)
    return MethodImplementation(**fields)


class TestAssembleDetectorFile:
    def test_wraps_source_with_offline_header_and_footer(self) -> None:
        assembled = assemble_detector_file(DETECTOR_SOURCE)
        assert assembled.startswith(DETECTOR_TEMPLATE_HEADER)
        assert "HF_HUB_OFFLINE" in assembled
        assert "TRANSFORMERS_OFFLINE" in assembled
        assert assembled.endswith(DETECTOR_TEMPLATE_FOOTER)
        assert "def anomaly_score(image, support_images, seed):" in assembled
        assert "metrics.json" in assembled
        assert 'if __name__ == "__main__":' in assembled

    def test_digest_is_deterministic(self) -> None:
        first = hashlib.sha256(assemble_detector_file(DETECTOR_SOURCE).encode()).hexdigest()
        second = hashlib.sha256(assemble_detector_file(DETECTOR_SOURCE).encode()).hexdigest()
        assert first == second


class TestWriteDetectorOnce:
    def test_writes_detector_file_once(self, tmp_path: Path) -> None:
        implementation = _implementation()
        path = _write_detector_once(tmp_path / "artifacts", implementation)
        assert path.is_file()
        assert "def anomaly_score(" in path.read_text(encoding="utf-8")
        again = _write_detector_once(tmp_path / "artifacts", implementation)
        assert again == path

    def test_rejects_tampered_source(self, tmp_path: Path) -> None:
        implementation = _implementation()
        tampered = implementation.model_copy(
            update={"source_code": DETECTOR_SOURCE.replace("min(", "max(")}
        )
        with pytest.raises(ValueError, match="摘要不一致"):
            _write_detector_once(tmp_path / "artifacts", tampered)


NONDETERMINISTIC_DETECTOR_SOURCE = (
    "import random\n"
    "\n"
    "\n"
    "def anomaly_score(image, support_images, seed):\n"
    "    return random.random()\n"
)

BROKEN_DETECTOR_SOURCE = (
    "import numpy as np\n"
    "\n"
    "\n"
    "def anomaly_score(image, support_images, seed):\n"
    "    return missing_name\n"
)


class TestRunDetectorSmoke:
    def test_smoke_passes_for_valid_detector(self, tmp_path: Path) -> None:
        implementation = _implementation(MOCK_DETECTOR_SOURCE)
        result = asyncio.run(run_detector_smoke(implementation, tmp_path / "artifacts"))
        assert result.passed is True, result.summary
        assert result.deterministic is True
        assert "冒烟通过" in result.summary
        assert (tmp_path / "artifacts" / "generated_methods").is_dir()

    def test_smoke_detects_nondeterministic_detector(self, tmp_path: Path) -> None:
        implementation = _implementation(NONDETERMINISTIC_DETECTOR_SOURCE)
        result = asyncio.run(run_detector_smoke(implementation, tmp_path / "artifacts"))
        assert result.passed is False

    def test_smoke_reports_failure_instead_of_raising(self, tmp_path: Path) -> None:
        implementation = _implementation(BROKEN_DETECTOR_SOURCE)
        result = asyncio.run(run_detector_smoke(implementation, tmp_path / "artifacts"))
        assert result.passed is False
        assert "失败" in result.summary

    def test_smoke_rejects_tampered_digest(self, tmp_path: Path) -> None:
        implementation = _implementation(MOCK_DETECTOR_SOURCE, code_digest="e" * 64)
        result = asyncio.run(run_detector_smoke(implementation, tmp_path / "artifacts"))
        assert result.passed is False
        assert "摘要不一致" in result.summary
