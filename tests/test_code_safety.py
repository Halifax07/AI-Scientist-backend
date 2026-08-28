import pytest

from fsad_scientist.experiments.code_safety import (
    MAX_DETECTOR_SOURCE_CHARS,
    MAX_SOURCE_CHARS,
    extract_detector_source,
    extract_select_function,
    implementation_detector_name,
    sanitize_detector_name,
    sanitize_strategy_name,
    validate_detector_source,
    validate_strategy_source,
)

VALID_SOURCE = (
    "def select(candidate_ids, embeddings, k, seed):\n"
    "    norms = {\n"
    "        file_id: sum(v * v for v in embeddings[file_id])\n"
    "        for file_id in candidate_ids\n"
    "    }\n"
    "    ranked = sorted(candidate_ids, key=lambda file_id: norms[file_id], reverse=True)\n"
    "    return ranked[:k]\n"
)


def _validate_fails(source: str) -> str:
    result = validate_strategy_source(source)
    assert result.passed is False
    assert result.issues
    return " ".join(result.issues)


class TestValidateStrategySource:
    def test_valid_function_passes(self) -> None:
        result = validate_strategy_source(VALID_SOURCE)
        assert result.passed is True, result.issues
        assert result.issues == []

    def test_import_inside_function_rejected(self) -> None:
        source = (
            "def select(candidate_ids, embeddings, k, seed):\n"
            "    import os\n"
            "    return candidate_ids[:k]\n"
        )
        message = _validate_fails(source)
        assert "import" in message

    def test_forbidden_modules_rejected(self) -> None:
        for module in ("os", "subprocess", "socket", "sys", "pathlib"):
            source = (
                f"def select(candidate_ids, embeddings, k, seed):\n"
                f"    import {module}\n    return candidate_ids[:k]\n"
            )
            assert _validate_fails(source)

    def test_forbidden_calls_rejected(self) -> None:
        for call in ("eval", "exec", "__import__", "open", "hash"):
            source = (
                f"def select(candidate_ids, embeddings, k, seed):\n"
                f"    {call}('x')\n    return candidate_ids[:k]\n"
            )
            message = _validate_fails(source)
            assert call in message

    def test_undefined_helper_call_rejected_before_smoke(self) -> None:
        source = (
            "def select(candidate_ids, embeddings, k, seed):\n"
            "    score = _euclidean_dist_sq(embeddings[candidate_ids[0]], [0.0])\n"
            "    return candidate_ids[:k] if score >= 0 else []\n"
        )
        message = _validate_fails(source)
        assert "_euclidean_dist_sq" in message
        assert "直接写入 select" in message

    def test_builtins_access_rejected(self) -> None:
        source = (
            "def select(candidate_ids, embeddings, k, seed):\n"
            "    x = __builtins__\n"
            "    return candidate_ids[:k]\n"
        )
        message = _validate_fails(source)
        assert "__builtins__" in message

    def test_class_definition_rejected(self) -> None:
        source = (
            "def select(candidate_ids, embeddings, k, seed):\n"
            "    class Helper:\n        pass\n    return candidate_ids[:k]\n"
        )
        message = _validate_fails(source)
        assert "类" in message

    def test_async_function_rejected(self) -> None:
        source = (
            "async def select(candidate_ids, embeddings, k, seed):\n"
            "    return candidate_ids[:k]\n"
        )
        message = _validate_fails(source)
        assert "async" in message or "def select" in message

    def test_wrong_parameter_names_rejected(self) -> None:
        source = "def select(a, embeddings, k, seed):\n    return a[:k]\n"
        message = _validate_fails(source)
        assert "candidate_ids" in message

    def test_module_level_statements_rejected(self) -> None:
        source = (
            "print('hello')\n"
            "def select(candidate_ids, embeddings, k, seed):\n"
            "    return candidate_ids[:k]\n"
        )
        message = _validate_fails(source)
        assert "恰好" in message or "模块级" in message

    def test_oversized_source_rejected(self) -> None:
        filler = "    value = value + 1\n"
        source = (
            "def select(candidate_ids, embeddings, k, seed):\n"
            f"{filler * (MAX_SOURCE_CHARS // len(filler) + 1)}"
            "    return candidate_ids[:k]\n"
        )
        message = _validate_fails(source)
        assert "上限" in message


class TestExtractSelectFunction:
    def test_keeps_only_function_block(self) -> None:
        text = (
            "```python\n"
            "print('preamble to drop')\n"
            "import os\n"
            "def select(candidate_ids, embeddings, k, seed):\n"
            "    return candidate_ids[:k]\n"
            "print('trailing to drop')\n"
            "```"
        )
        segment = extract_select_function(text)
        assert segment.startswith("def select(")
        assert "preamble" not in segment
        assert "import os" not in segment
        assert "trailing" not in segment
        assert validate_strategy_source(segment).passed is True

    def test_missing_select_raises(self) -> None:
        with pytest.raises(ValueError, match="select"):
            extract_select_function("def other(x):\n    return x\n")

    def test_duplicate_select_raises(self) -> None:
        text = (
            "def select(a, b, c, d):\n    return a[:c]\n"
            "def select(candidate_ids, embeddings, k, seed):\n    return candidate_ids[:k]\n"
        )
        with pytest.raises(ValueError, match="多个"):
            extract_select_function(text)

    def test_invalid_syntax_raises(self) -> None:
        with pytest.raises(ValueError, match="无法解析"):
            extract_select_function("def select(:\n")


VALID_DETECTOR_SOURCE = (
    "import numpy as np\n"
    "from sklearn.metrics import roc_auc_score\n"
    "\n"
    "\n"
    "def mean_squared_distance(a, b):\n"
    "    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))\n"
    "\n"
    "\n"
    "def anomaly_score(image, support_images, seed):\n"
    "    distances = [\n"
    "        mean_squared_distance(image, support)\n"
    "        for support in support_images\n"
    "    ]\n"
    "    return 1.0 / (1.0 + min(distances))\n"
)


def _validate_detector_fails(source: str) -> str:
    result = validate_detector_source(source)
    assert result.passed is False
    assert result.issues
    return " ".join(result.issues)


class TestValidateDetectorSource:
    def test_valid_detector_source_passes(self) -> None:
        result = validate_detector_source(VALID_DETECTOR_SOURCE)
        assert result.passed is True, result.issues
        assert result.issues == []

    def test_missing_anomaly_score_rejected(self) -> None:
        source = "import numpy as np\n\ndef helper(a):\n    return a\n"
        message = _validate_detector_fails(source)
        assert "anomaly_score" in message

    def test_duplicate_anomaly_score_rejected(self) -> None:
        source = (
            "import numpy as np\n"
            "def anomaly_score(image, support_images, seed):\n    return 1.0\n"
            "def anomaly_score(image, support_images, seed):\n    return 2.0\n"
        )
        message = _validate_detector_fails(source)
        assert "恰好" in message

    def test_wrong_signature_rejected(self) -> None:
        source = (
            "import numpy as np\n"
            "def anomaly_score(a, b):\n    return 1.0\n"
        )
        message = _validate_detector_fails(source)
        assert "image" in message

    def test_import_inside_function_rejected(self) -> None:
        source = (
            "def anomaly_score(image, support_images, seed):\n"
            "    import os\n"
            "    return 1.0\n"
        )
        message = _validate_detector_fails(source)
        assert "import" in message

    def test_forbidden_modules_rejected(self) -> None:
        for module in ("os", "subprocess", "socket", "sys", "pathlib", "shutil", "huggingface_hub"):
            source = (
                f"import {module}\n"
                "def anomaly_score(image, support_images, seed):\n"
                "    return 1.0\n"
            )
            message = _validate_detector_fails(source)
            assert module in message

    def test_torch_hub_load_rejected(self) -> None:
        for call in ("torch.hub.load('repo', 'model')", "hub.load('repo')"):
            source = (
                "import torch\n"
                "def anomaly_score(image, support_images, seed):\n"
                f"    model = {call}\n"
                "    return 1.0\n"
            )
            message = _validate_detector_fails(source)
            assert "hub.load" in message

    def test_from_pretrained_allowed(self) -> None:
        source = (
            "from transformers import AutoModel\n"
            "def anomaly_score(image, support_images, seed):\n"
            "    model = AutoModel.from_pretrained('facebook/dinov2-small')\n"
            "    return 1.0\n"
        )
        result = validate_detector_source(source)
        assert result.passed is True, result.issues

    def test_class_definition_rejected(self) -> None:
        source = (
            "import numpy as np\n"
            "class Detector:\n    pass\n"
            "def anomaly_score(image, support_images, seed):\n    return 1.0\n"
        )
        message = _validate_detector_fails(source)
        assert "模块级" in message

    def test_async_function_rejected(self) -> None:
        source = (
            "import numpy as np\n"
            "async def anomaly_score(image, support_images, seed):\n    return 1.0\n"
        )
        message = _validate_detector_fails(source)
        assert "async" in message

    def test_module_level_assignment_rejected(self) -> None:
        source = (
            "import numpy as np\n"
            "THRESHOLD = 0.5\n"
            "def anomaly_score(image, support_images, seed):\n    return 1.0\n"
        )
        message = _validate_detector_fails(source)
        assert "模块级" in message

    def test_underscore_helper_rejected(self) -> None:
        source = (
            "import numpy as np\n"
            "def _main():\n    pass\n"
            "def anomaly_score(image, support_images, seed):\n    return 1.0\n"
        )
        message = _validate_detector_fails(source)
        assert "下划线" in message

    def test_oversized_source_rejected(self) -> None:
        filler = "    value = value + 1\n"
        source = (
            "import numpy as np\n"
            "def anomaly_score(image, support_images, seed):\n"
            f"{filler * (MAX_DETECTOR_SOURCE_CHARS // len(filler) + 1)}"
            "    return 1.0\n"
        )
        message = _validate_detector_fails(source)
        assert "上限" in message


class TestExtractDetectorSource:
    def test_strips_fences(self) -> None:
        text = (
            "```python\n"
            "import numpy as np\n"
            "def anomaly_score(image, support_images, seed):\n"
            "    return 1.0\n"
            "```"
        )
        extracted = extract_detector_source(text)
        assert extracted.startswith("import numpy as np")
        assert "```" not in extracted


class TestSanitizeDetectorName:
    def test_builtin_name_rejected(self) -> None:
        assert sanitize_detector_name("AnomalyDINO") == "anomalydino_custom"

    def test_plain_stem_passes_through(self) -> None:
        assert sanitize_detector_name("nearest_prototype") == "nearest_prototype"

    def test_implementation_name_composition(self) -> None:
        name = implementation_detector_name("Nearest-Proto", "ab" * 32)
        assert name.startswith("generated_det_nearest_proto_")
        assert name.endswith("_abababab")


class TestSanitizeStrategyName:
    def test_plain_stem_passes_through(self) -> None:
        assert sanitize_strategy_name("query_adaptive") == "query_adaptive"

    def test_uppercase_and_special_characters(self) -> None:
        assert sanitize_strategy_name("Query-Adaptive 选样!") == "query_adaptive"
        assert sanitize_strategy_name("  ") == "strategy"

    def test_digit_prefix(self) -> None:
        name = sanitize_strategy_name("2stage")
        assert name.startswith("s_")

    def test_builtin_name_rejected(self) -> None:
        assert sanitize_strategy_name("random") == "random_custom"

    def test_bounded_length(self) -> None:
        assert len(sanitize_strategy_name("x" * 200)) <= 55
