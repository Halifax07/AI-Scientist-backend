import asyncio

from fsad_scientist.agents.mock_runtime import MockScientistRuntime
from fsad_scientist.agents.qwen_runtime import (
    QwenScientistRuntime,
    _normalize_hypothesis_payload,
)
from fsad_scientist.domain.models import Hypothesis, ProjectSpec, ResearchProject
from fsad_scientist.experiments.code_safety import (
    BUILTIN_STRATEGIES,
    GENERATED_DETECTOR_PREFIX,
    validate_detector_source,
    validate_strategy_source,
)


def test_hypothesis_payload_normalizes_model_container_drift() -> None:
    normalized = _normalize_hypothesis_payload(
        {
            "id": "model-owned-id",
            "gap_id": "gap_1",
            "independent_variables": "support selection",
            "dependent_variables": ["image_auroc"],
            "falsification_conditions": "paired effect is not positive",
            "evidence_ids": "evidence_1",
            "closest_prior_work": "PatchCore",
            "analysis_contract": "use a paired comparison",
            "status": "supported",
        }
    )

    assert normalized["independent_variables"] == ["support selection"]
    assert normalized["falsification_conditions"] == [
        "paired effect is not positive"
    ]
    assert normalized["evidence_ids"] == ["evidence_1"]
    assert normalized["closest_prior_work"] == ["PatchCore"]
    assert normalized["analysis_contract"] is None
    assert "id" not in normalized
    assert "status" not in normalized


def _build_project_and_hypothesis() -> tuple[ResearchProject, Hypothesis]:
    project = ResearchProject(spec=ProjectSpec())
    hypothesis = Hypothesis(
        gap_id="gap_x",
        title="查询自适应选样",
        claim="查询自适应选样提升检测稳定性。",
        null_hypothesis="与随机选样无差异。",
        rationale="测试机制。",
        independent_variables=["选样策略"],
        dependent_variables=["image_auroc"],
        predicted_direction="正效应",
        falsification_conditions=["效应置信区间包含零"],
    )
    return project, hypothesis


def test_mock_implement_selection_strategy_returns_valid_draft() -> None:
    project, hypothesis = _build_project_and_hypothesis()
    runtime = MockScientistRuntime()
    implementation = asyncio.run(
        runtime.implement_selection_strategy(
            project,
            hypothesis=hypothesis,
            strategy_name="query_adaptive",
            control_name="random",
        )
    )
    assert validate_strategy_source(implementation.source_code).passed is True
    assert implementation.status == "draft"
    assert implementation.name == "query_adaptive"
    assert implementation.name not in BUILTIN_STRATEGIES
    assert "mock-scientist-runtime" in implementation.provenance


class _FakeClient:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error
        self.calls = []

    async def complete(self, *, role_name, system_prompt, payload):
        self.calls.append({"role_name": role_name, "payload": payload})
        if self._error is not None:
            raise self._error
        return self._response


class _SequenceClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def complete(self, *, role_name, system_prompt, payload):
        self.calls.append(
            {
                "role_name": role_name,
                "system_prompt": system_prompt,
                "payload": payload,
            }
        )
        if not self._responses:
            raise AssertionError("unexpected extra client call")
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def test_qwen_implement_selection_strategy_uses_client() -> None:
    project, hypothesis = _build_project_and_hypothesis()
    runtime = QwenScientistRuntime()
    runtime.client = _FakeClient(  # type: ignore[assignment]
        response={
            "name": "Query-Adaptive 选样",
            "source_code": (
                "```python\n"
                "def select(candidate_ids, embeddings, k, seed):\n"
                "    return sorted(candidate_ids)[:k]\n"
                "```"
            ),
            "explanation": "确定性前 k 个。",
        }
    )
    implementation = asyncio.run(
        runtime.implement_selection_strategy(
            project,
            hypothesis=hypothesis,
            strategy_name="query_adaptive",
            control_name="no_adaptation",
        )
    )
    assert validate_strategy_source(implementation.source_code).passed is True
    assert implementation.name == "query_adaptive"
    assert implementation.name not in BUILTIN_STRATEGIES
    assert "qwen-agentscope-runtime" in implementation.provenance
    assert runtime.client.calls[0]["role_name"] == "MethodImplementerAgent"


def test_qwen_implement_selection_strategy_falls_back_to_mock() -> None:
    project, hypothesis = _build_project_and_hypothesis()
    runtime = QwenScientistRuntime()
    runtime.client = _FakeClient(error=RuntimeError("boom"))  # type: ignore[assignment]
    implementation = asyncio.run(
        runtime.implement_selection_strategy(
            project,
            hypothesis=hypothesis,
            strategy_name="query_adaptive",
            control_name="random",
        )
    )
    assert any("deterministic-fallback" in item for item in implementation.provenance)
    assert validate_strategy_source(implementation.source_code).passed is True


def test_qwen_implement_selection_strategy_repairs_invalid_source() -> None:
    project, hypothesis = _build_project_and_hypothesis()
    runtime = QwenScientistRuntime()
    runtime.client = _SequenceClient(
        [
            {
                "source_code": (
                    "def select(candidate_ids, embeddings, k, seed):\n"
                    "    import random\n"
                    "    def helper():\n"
                    "        return 1\n"
                    "    return candidate_ids[:k]\n"
                )
            },
            {
                "source_code": (
                    "def select(candidate_ids, embeddings, k, seed):\n"
                    "    return sorted(candidate_ids)[:k]\n"
                )
            },
        ]
    )  # type: ignore[assignment]

    implementation = asyncio.run(
        runtime.implement_selection_strategy(
            project,
            hypothesis=hypothesis,
            strategy_name="query_adaptive",
            control_name="random",
        )
    )

    assert len(runtime.client.calls) == 2
    repair_payload = runtime.client.calls[1]["payload"]
    assert repair_payload["previous_source_code"].startswith("def select")
    assert repair_payload["validation_issues"]
    assert validate_strategy_source(implementation.source_code).passed is True
    assert any("validation-repair" in item for item in implementation.provenance)
    assert not any("fallback" in item for item in implementation.provenance)


def test_qwen_implement_selection_strategy_retries_malformed_response() -> None:
    project, hypothesis = _build_project_and_hypothesis()
    runtime = QwenScientistRuntime()
    runtime.client = _SequenceClient(
        [
            ValueError("Agent output must contain one valid JSON object"),
            {
                "source_code": (
                    "def select(candidate_ids, embeddings, k, seed):\n"
                    "    return sorted(candidate_ids)[:k]\n"
                )
            },
        ]
    )  # type: ignore[assignment]

    implementation = asyncio.run(
        runtime.implement_selection_strategy(
            project,
            hypothesis=hypothesis,
            strategy_name="query_adaptive",
            control_name="random",
        )
    )

    assert len(runtime.client.calls) == 2
    assert validate_strategy_source(implementation.source_code).passed is True
    assert any("validation-repair" in item for item in implementation.provenance)
    assert not any("fallback" in item for item in implementation.provenance)


def test_qwen_implement_selection_strategy_falls_back_after_invalid_repair() -> None:
    project, hypothesis = _build_project_and_hypothesis()
    runtime = QwenScientistRuntime()
    invalid_source = (
        "def select(candidate_ids, embeddings, k, seed):\n"
        "    import random\n"
        "    def helper():\n"
        "        return 1\n"
        "    return candidate_ids[:k]\n"
    )
    runtime.client = _SequenceClient(
        [{"source_code": invalid_source}] * 4
    )  # type: ignore[assignment]

    implementation = asyncio.run(
        runtime.implement_selection_strategy(
            project,
            hypothesis=hypothesis,
            strategy_name="query_adaptive",
            control_name="random",
        )
    )

    assert len(runtime.client.calls) == 4
    assert validate_strategy_source(implementation.source_code).passed is True
    assert any("validation-fallback" in item for item in implementation.provenance)
    assert any("deterministic-fallback" in item for item in implementation.provenance)


def test_mock_implement_detector_returns_valid_draft() -> None:
    project, hypothesis = _build_project_and_hypothesis()
    runtime = MockScientistRuntime()
    implementation = asyncio.run(
        runtime.implement_detector(
            project,
            hypothesis=hypothesis,
            name_stem="nearest_prototype",
            reference_description="PatchCore 式最近邻记忆库",
        )
    )
    assert implementation.kind == "detector"
    assert implementation.status == "draft"
    assert implementation.name.startswith(GENERATED_DETECTOR_PREFIX + "nearest_prototype_")
    assert validate_detector_source(implementation.source_code).passed is True
    assert "mock-scientist-runtime" in implementation.provenance


def test_qwen_implement_detector_uses_client() -> None:
    project, hypothesis = _build_project_and_hypothesis()
    runtime = QwenScientistRuntime()
    runtime.client = _FakeClient(  # type: ignore[assignment]
        response={
            "source_code": (
                "```python\n"
                "import numpy as np\n"
                "def anomaly_score(image, support_images, seed):\n"
                "    return 1.0\n"
                "```"
            ),
            "explanation": "恒定分数占位。",
        }
    )
    implementation = asyncio.run(
        runtime.implement_detector(
            project,
            hypothesis=hypothesis,
            name_stem="nearest_prototype",
            reference_description=None,
        )
    )
    assert implementation.name.startswith(GENERATED_DETECTOR_PREFIX + "nearest_prototype_")
    assert validate_detector_source(implementation.source_code).passed is True
    assert "qwen-agentscope-runtime" in implementation.provenance
    assert runtime.client.calls[0]["role_name"] == "DetectorImplementerAgent"


def test_qwen_implement_detector_falls_back_to_mock() -> None:
    project, hypothesis = _build_project_and_hypothesis()
    runtime = QwenScientistRuntime()
    runtime.client = _FakeClient(error=RuntimeError("boom"))  # type: ignore[assignment]
    implementation = asyncio.run(
        runtime.implement_detector(
            project,
            hypothesis=hypothesis,
            name_stem="nearest_prototype",
            reference_description=None,
        )
    )
    assert any("deterministic-fallback" in item for item in implementation.provenance)
    assert validate_detector_source(implementation.source_code).passed is True


def test_qwen_implement_detector_repairs_invalid_source() -> None:
    project, hypothesis = _build_project_and_hypothesis()
    runtime = QwenScientistRuntime()
    runtime.client = _SequenceClient(
        [
            {
                "source_code": (
                    "import torch.nn.functional as F\n"
                    "def _helper(image):\n"
                    "    return 1\n"
                    "def anomaly_score(image, support_images, seed):\n"
                    "    return 1.0\n"
                )
            },
            {
                "source_code": (
                    "import numpy as np\n"
                    "def anomaly_score(image, support_images, seed):\n"
                    "    return float(np.mean(image))\n"
                )
            },
        ]
    )  # type: ignore[assignment]

    implementation = asyncio.run(
        runtime.implement_detector(
            project,
            hypothesis=hypothesis,
            name_stem="nearest_prototype",
            reference_description=None,
        )
    )

    assert len(runtime.client.calls) == 2
    repair_payload = runtime.client.calls[1]["payload"]
    assert repair_payload["previous_source_code"].startswith("import")
    assert repair_payload["validation_issues"]
    assert validate_detector_source(implementation.source_code).passed is True
    assert any("validation-repair" in item for item in implementation.provenance)
    assert not any("fallback" in item for item in implementation.provenance)


def test_qwen_implement_detector_falls_back_after_invalid_repair() -> None:
    project, hypothesis = _build_project_and_hypothesis()
    runtime = QwenScientistRuntime()
    invalid_source = (
        "import torch.nn.functional as F\n"
        "def _helper(image):\n"
        "    return 1\n"
        "def anomaly_score(image, support_images, seed):\n"
        "    return 1.0\n"
    )
    runtime.client = _SequenceClient(
        [{"source_code": invalid_source}] * 4
    )  # type: ignore[assignment]

    implementation = asyncio.run(
        runtime.implement_detector(
            project,
            hypothesis=hypothesis,
            name_stem="nearest_prototype",
            reference_description=None,
        )
    )

    assert len(runtime.client.calls) == 4
    assert validate_detector_source(implementation.source_code).passed is True
    assert any("validation-fallback" in item for item in implementation.provenance)
    assert any("deterministic-fallback" in item for item in implementation.provenance)
