import asyncio

from fsad_scientist.agents.agentscope_client import AgentOutputValidationError
from fsad_scientist.agents.mock_runtime import MockScientistRuntime
from fsad_scientist.agents.qwen_runtime import (
    QwenScientistRuntime,
    _normalize_hypothesis_payload,
)
from fsad_scientist.domain.enums import HypothesisStatus, ResearchStage
from fsad_scientist.domain.models import (
    AnalysisContract,
    ExperimentAnalysisSpec,
    ExperimentConditionSpec,
    ExperimentDesignSpec,
    ExperimentFactorSpec,
    ExperimentPlan,
    Hypothesis,
    ProjectSpec,
    ResearchProject,
)
from fsad_scientist.experiments.code_safety import (
    BUILTIN_STRATEGIES,
    GENERATED_DETECTOR_PREFIX,
    validate_detector_source,
    validate_strategy_source,
)
from fsad_scientist.repository import JsonProjectRepository
from fsad_scientist.workflow import ResearchWorkflow


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


def test_qwen_formalize_scope_falls_back_after_invalid_json_output() -> None:
    project = ResearchProject(spec=ProjectSpec())
    runtime = QwenScientistRuntime()
    runtime.client = _FakeClient(  # type: ignore[assignment]
        error=AgentOutputValidationError("Qwen returned malformed JSON")
    )

    artifact = asyncio.run(runtime.formalize_scope(project))

    assert artifact.kind == "research_scope"
    assert "invalid_json_fallback" in artifact.provenance


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


def test_qwen_design_runtime_accepts_three_condition_design() -> None:
    project = ResearchProject(spec=ProjectSpec())
    hypothesis = Hypothesis(
        id="h_design",
        gap_id="gap_x",
        title="多条件检测器实验",
        claim="因素组合改变检测性能。",
        null_hypothesis="因素组合不改变检测性能。",
        rationale="验证交互效应。",
        independent_variables=["检测器", "选样策略"],
        dependent_variables=["image_auroc"],
        predicted_direction="increase",
        falsification_conditions=["效应为零"],
        analysis_contract=AnalysisContract(
            kind="detector_interaction",
            metric="image_auroc",
            treatment="k_center",
            control="random",
        ),
        status=HypothesisStatus.SHORTLISTED,
    )
    project.hypotheses = [hypothesis]
    runtime = QwenScientistRuntime()
    runtime.client = _SequenceClient(
        [
            {
                "designs": [
                    {
                        "id": "three_arm",
                        "hypothesis_id": "h_design",
                        "factors": [
                            {
                                "name": "strategy",
                                "field": "selection_strategy",
                                "levels": ["random", "k_center"],
                            },
                            {
                                "name": "detector",
                                "field": "detector",
                                "levels": ["anomalydino", "patchcore"],
                            },
                        ],
                        "conditions": [
                            {
                                "id": "r_a",
                                "factor_values": {
                                    "strategy": "random",
                                    "detector": "anomalydino",
                                },
                            },
                            {
                                "id": "k_a",
                                "factor_values": {
                                    "strategy": "k_center",
                                    "detector": "anomalydino",
                                },
                            },
                            {
                                "id": "r_p",
                                "factor_values": {
                                    "strategy": "random",
                                    "detector": "patchcore",
                                },
                            },
                        ],
                        "analysis": {
                            "mode": "group_comparison",
                            "primary_metric": "image_auroc",
                            "minimum_pairs": 2,
                        },
                        "presentation_spec": {
                            "schema_version": 2,
                            "layout": "grid",
                            "density": "compact",
                                "blocks": [
                                    {
                                        "id": "factors",
                                        "kind": "chart",
                                        "source": "factor_effects",
                                        "chart_mark": "bar",
                                        "span": "half",
                                    },
                                    {"kind": "narrative", "source": "design"},
                                    {"kind": "progress", "source": "progress"},
                                    {"kind": "evidence", "source": "evidence"},
                                ],
                        },
                    }
                ]
            }
        ]
    )  # type: ignore[assignment]

    plan = asyncio.run(runtime.design_experiments(project))

    assert len(plan.designs) == 1
    assert len(plan.designs[0].conditions) == 3
    assert "不要把设计限制为 k_center/random" in runtime.client.calls[0]["system_prompt"]


def test_qwen_design_repairs_required_presentation_blocks_without_paired_fallback(
    tmp_path,
) -> None:
    project, hypothesis = _build_project_and_hypothesis()
    hypothesis.status = HypothesisStatus.SHORTLISTED
    hypothesis.analysis_contract = AnalysisContract(
        kind="detector_interaction",
        metric="image_auroc",
        design_mode="custom_design",
        treatment=None,
        control=None,
    )
    project.hypotheses = [hypothesis]
    runtime = QwenScientistRuntime()
    runtime.client = _SequenceClient(
        [
            {
                "designs": [
                    {
                        "id": "custom_without_required_blocks",
                        "name": "detector_sweep",
                        "hypothesis_id": hypothesis.id,
                        "design_type": "custom_design",
                        "design_mode": "custom_design",
                        "support_selection_strategy": "random",
                        "factors": [
                            {
                                "name": "detector",
                                "field": "detector",
                                "levels": ["anomalydino", "patchcore"],
                            }
                        ],
                        "conditions": [
                            {
                                "id": "dino",
                                "label": "DINOv2",
                                "factor_values": {"detector": "anomalydino"},
                            },
                            {
                                "id": "patchcore",
                                "label": "PatchCore",
                                "factor_values": {"detector": "patchcore"},
                            },
                        ],
                        "analysis": {
                            "mode": "factor_effects",
                            "primary_metric": "image_auroc",
                            "minimum_pairs": 2,
                        },
                        "presentation_spec": {
                            "schema_version": 2,
                            "layout": "grid",
                            "density": "compact",
                            "blocks": [
                                {
                                    "id": "factor-chart",
                                    "kind": "chart",
                                    "source": "factor_effects",
                                    "chart_mark": "bar",
                                    "title": "检测器因素效应",
                                    "content": "保留 AI 提供的结果说明。",
                                    "config": {"showLegend": True, "unit": "AUROC"},
                                }
                            ],
                        },
                    }
                ]
            }
        ]
    )  # type: ignore[assignment]

    plan = asyncio.run(runtime.design_experiments(project))

    assert plan.design_generation_status == "ai_selected"
    assert plan.design_generation_errors == []
    assert len(plan.designs) == 1
    design = plan.designs[0]
    assert design.design_mode == "custom_design"
    assert design.design_type == "custom_design"
    assert design.hypothesis_id == hypothesis.id
    assert design.conditions[0].factor_values == {"detector": "anomalydino"}
    assert {(block.kind, block.source) for block in design.presentation_spec.blocks} >= {
        ("narrative", "design"),
        ("progress", "progress"),
        ("evidence", "evidence"),
    }
    factor_chart = next(
        block
        for block in design.presentation_spec.blocks
        if block.id == "factor-chart"
    )
    assert factor_chart.config == {"showLegend": True, "unit": "AUROC"}
    assert factor_chart.title == "检测器因素效应"
    assert not any(
        block.source in {"condition_effects", "condition_statistics"}
        and block.kind == "chart"
        for block in design.presentation_spec.blocks
    )

    project.experiment_plan = plan
    project.stage = ResearchStage.AWAITING_EXPERIMENT_APPROVAL
    workflow = ResearchWorkflow(
        repository=JsonProjectRepository(tmp_path / "ledger"),
        runtime=runtime,
    )
    workflow.repository.save(project)
    approved = workflow.approve_experiment_plan(
        project.id,
        approved_by="regression-test",
    )
    assert approved.stage == ResearchStage.EXPERIMENTS_QUEUED
    assert any(run.condition_id == "dino" for run in approved.runs)


def test_qwen_design_failure_keeps_executable_deterministic_fallback() -> None:
    project, hypothesis = _build_project_and_hypothesis()
    hypothesis.status = HypothesisStatus.SHORTLISTED
    hypothesis.analysis_contract = AnalysisContract(
        kind="detector_interaction",
        metric="image_auroc",
        design_mode="custom_design",
        treatment=None,
        control=None,
    )
    project.hypotheses = [hypothesis]
    runtime = QwenScientistRuntime()
    runtime.client = _FakeClient(error=RuntimeError("DashScope HTTP 400: invalid input"))  # type: ignore[assignment]

    plan = asyncio.run(runtime.design_experiments(project))

    assert plan.design_generation_status == "fallback"
    assert plan.design_generation_fallback_reason is not None
    assert "deterministic fallback" in plan.design_generation_fallback_reason
    assert plan.design_generation_errors == ["DashScope HTTP 400: invalid input"]
    assert plan.designs[0].design_mode == "custom_design"


def test_qwen_design_accepts_fixed_controls_and_interaction_mode_alias() -> None:
    project, hypothesis = _build_project_and_hypothesis()
    hypothesis.status = HypothesisStatus.SHORTLISTED
    hypothesis.analysis_contract = AnalysisContract(
        kind="detector_interaction",
        metric="image_auroc",
        design_mode="custom_design",
        treatment=None,
        control=None,
    )
    project.hypotheses = [hypothesis]
    runtime = QwenScientistRuntime()
    runtime.client = _SequenceClient(
        [
            {
                "designs": [
                    {
                        "id": "fixed_controls_interaction",
                        "hypothesis_id": hypothesis.id,
                        "design_type": "custom_design",
                        "design_mode": "custom_design",
                        "support_selection_strategy": "random",
                        "factors": [
                            {
                                "name": "protocol",
                                "field": "protocol",
                                "levels": ["pool_compression_m30"],
                            },
                            {
                                "name": "detector",
                                "field": "detector",
                                "levels": ["patchcore"],
                            },
                            {"name": "shots", "field": "shots", "levels": [4]},
                            {
                                "name": "category",
                                "field": "category",
                                "levels": ["bottle", "carpet"],
                            },
                        ],
                        "conditions": [
                            {
                                "id": "bottle",
                                "factor_values": {
                                    "protocol": "pool_compression_m30",
                                    "detector": "patchcore",
                                    "shots": 4,
                                    "category": "bottle",
                                },
                            },
                            {
                                "id": "bottle",
                                "factor_values": {
                                    "protocol": "pool_compression_m30",
                                    "detector": "patchcore",
                                    "shots": 4,
                                    "category": "bottle",
                                },
                            },
                            {
                                "id": "carpet",
                                "factor_values": {
                                    "protocol": "pool_compression_m30",
                                    "detector": "patchcore",
                                    "shots": 4,
                                    "category": "carpet",
                                },
                            },
                        ],
                        "analysis": {
                            "mode": "interaction_summary",
                            "primary_metric": "image_auroc",
                            "minimum_pairs": 2,
                        },
                        "presentation_spec": {
                            "schema_version": 2,
                            "layout": "grid",
                            "blocks": [
                                {"kind": "narrative", "source": "design"},
                                {"kind": "progress", "source": "progress"},
                                {
                                    "kind": "chart",
                                    "source": "interaction_summary",
                                    "chart_mark": "heatmap",
                                },
                                {
                                    "kind": "chart",
                                    "source": "group_comparison",
                                    "chart_mark": "bar",
                                },
                                {"kind": "evidence", "source": "evidence"},
                            ],
                        },
                    }
                ]
            }
        ]
    )  # type: ignore[assignment]

    plan = asyncio.run(runtime.design_experiments(project))

    assert plan.design_generation_status == "ai_selected"
    assert len(plan.designs) == 1
    design = plan.designs[0]
    assert design.design_mode == "custom_design"
    assert design.analysis.mode == "factor_effects"
    assert len(design.conditions) == 2
    assert {factor.name for factor in design.factors} == {
        "protocol",
        "detector",
        "shots",
        "category",
    }
    assert any(
        block.kind == "chart"
        and block.source == "interaction_summary"
        and block.chart_mark == "heatmap"
        for block in design.presentation_spec.blocks
    )
    assert any(
        block.kind == "chart"
        and block.source == "condition_statistics"
        and block.chart_mark == "bar"
        for block in design.presentation_spec.blocks
    )


def test_qwen_explicit_feedback_preserves_valid_result_aware_spec() -> None:
    project, hypothesis = _build_project_and_hypothesis()
    hypothesis.analysis_contract = AnalysisContract(
        kind="detector_interaction",
        metric="image_auroc",
        treatment="k_center",
        control="random",
    )
    project.hypotheses = [hypothesis]
    design = ExperimentDesignSpec(
        id="feedback_design",
        hypothesis_id=hypothesis.id,
        factors=[
            ExperimentFactorSpec(
                name="strategy", field="selection_strategy", levels=["random", "k_center"]
            ),
            ExperimentFactorSpec(name="K", field="shots", levels=[1, 2]),
        ],
        conditions=[
            ExperimentConditionSpec(
                id=f"{strategy}_{shots}",
                factor_values={"strategy": strategy, "K": shots},
            )
            for strategy in ["random", "k_center"]
            for shots in [1, 2]
        ],
        analysis=ExperimentAnalysisSpec(primary_metric="image_auroc", mode="factor_effects"),
    )
    project.experiment_plan = ExperimentPlan(
        hypothesis_ids=[hypothesis.id],
        hypothesis_contracts={hypothesis.id: hypothesis.analysis_contract.model_dump(mode="json")},
        protocols=["pool_compression_m30"],
        detectors=["anomalydino"],
        selection_strategies=["random", "k_center"],
        datasets=["MVTec AD"],
        categories=["bottle"],
        shots=[1, 2],
        seeds=[0, 1],
        metrics=["image_auroc"],
        analysis_methods=[],
        stages=[],
        stopping_conditions=[],
        estimated_gpu_hours=1,
        preregistration_digest="digest",
        approved=True,
        designs=[design],
    )
    runtime = QwenScientistRuntime()
    runtime.client = _SequenceClient(
        [
            {
                "decision": "expand",
                "rationale": "继续覆盖已批准条件。",
                "presentation_spec": {
                    "schema_version": 2,
                    "layout": "grid",
                    "blocks": [
                        {"kind": "narrative", "source": "design"},
                        {"kind": "progress", "source": "progress"},
                        {"kind": "table", "source": "condition_statistics"},
                        {"kind": "evidence", "source": "evidence"},
                    ],
                },
            }
        ]
    )

    proposal = asyncio.run(
        runtime.recommend_next_experiments(
            project,
            round_summary={
                "design_id": design.id,
                "analysis_mode": "factor_effects",
                "sample_size": 2,
                "evidence_status": "below_threshold",
                "failed_run_ids": [],
            },
            allowed_cells=[],
        )
    )

    assert proposal.presentation_spec is not None
    assert proposal.presentation_spec.schema_version == 2
    assert "配对" not in runtime.client.calls[0]["system_prompt"]
    assert "p 值" not in runtime.client.calls[0]["system_prompt"]

    runtime.client = _SequenceClient(
        [
            {
                "decision": "expand",
                "rationale": "继续观察。",
                "presentation_spec": {
                    "schema_version": 2,
                    "blocks": [
                        {"kind": "narrative", "source": "design"},
                        {"kind": "progress", "source": "progress"},
                        {
                            "kind": "chart",
                            "source": "interaction_summary",
                            "chart_mark": "heatmap",
                        },
                        {"kind": "evidence", "source": "evidence"},
                    ],
                },
            }
        ]
    )
    replacement = asyncio.run(
        runtime.recommend_next_experiments(
            project,
            round_summary={
                "design_id": design.id,
                "analysis_mode": "factor_effects",
                "sample_size": 1,
                "evidence_status": "below_threshold",
                "failed_run_ids": [],
                "interaction_summary": [{}],
            },
            allowed_cells=[],
        )
    )
    assert replacement.presentation_spec is not None
    assert not any(
        block.source == "interaction_summary" and block.chart_mark == "heatmap"
        for block in replacement.presentation_spec.blocks
    )
    assert any(
        block.kind == "table" and block.source in {"condition_statistics", "factor_effects"}
        for block in replacement.presentation_spec.blocks
    )
