import asyncio

import pytest

from fsad_scientist.agents.mock_runtime import MockScientistRuntime
from fsad_scientist.agents.qwen_runtime import QwenScientistRuntime
from fsad_scientist.datasets.models import DatasetManifest
from fsad_scientist.domain.enums import HypothesisStatus, RunStatus
from fsad_scientist.domain.models import (
    AnalysisContract,
    ComputeBudget,
    DatasetAuditRecord,
    ExperimentAnalysisSpec,
    ExperimentCardBlockSpec,
    ExperimentCardPresentationSpec,
    ExperimentConditionSpec,
    ExperimentDesignSpec,
    ExperimentFactorSpec,
    ExperimentPlan,
    Hypothesis,
    ProjectSpec,
    ResearchProject,
)
from fsad_scientist.experiments.design import (
    compile_design,
    default_presentation_spec,
    is_result_compatible_presentation_spec,
    normalize_design_conditions_payload,
    normalize_design,
    normalize_presentation_spec_payload,
    result_aware_presentation_spec,
    validate_design,
)
from fsad_scientist.experiments.loop import AdaptiveExperimentPlanner


def _plan(*, designs=None, metrics=None, categories=None, protocols=None) -> ExperimentPlan:
    return ExperimentPlan(
        hypothesis_ids=["h1"],
        protocols=protocols or ["pool_compression_m30"],
        detectors=["anomalydino", "patchcore"],
        selection_strategies=["random", "k_center"],
        datasets=["MVTec AD"],
        categories=categories or ["bottle"],
        shots=[1, 2, 4],
        seeds=[0, 1, 2],
        metrics=metrics or ["image_auroc"],
        analysis_methods=[],
        stages=[],
        stopping_conditions=[],
        estimated_gpu_hours=1,
        preregistration_digest="digest",
        approved=True,
        designs=designs or [],
    )


def test_compile_full_factorial_and_replication_designs() -> None:
    design = ExperimentDesignSpec(
        id="factorial",
        design_type="full_factorial",
        factors=[
            ExperimentFactorSpec(
                name="strategy", field="selection_strategy", levels=["random", "k_center"]
            ),
            ExperimentFactorSpec(name="K", field="shots", levels=[1, 2]),
        ],
        conditions=[],
        analysis=ExperimentAnalysisSpec(primary_metric="image_auroc", mode="factor_effects"),
    )

    compiled = compile_design(design, plan=_plan())

    assert len(compiled) == 4
    assert compiled[0].factor_values == {"strategy": "random", "K": 1}

    replication = design.model_copy(update={"id": "replication", "design_type": "replication"})
    assert len(compile_design(replication, plan=_plan())) == 4
    exploration = design.model_copy(update={"id": "exploration", "design_type": "exploration"})
    reproduction = design.model_copy(
        update={"id": "reproduction", "design_type": "reproduction", "purpose": "reproduction"}
    )
    assert len(compile_design(exploration, plan=_plan())) == 4
    assert len(compile_design(reproduction, plan=_plan())) == 4


def test_normalize_legacy_contract_produces_two_runtime_conditions() -> None:
    plan = _plan()
    contract = AnalysisContract(
        kind="selection_main_effect",
        metric="image_auroc",
        treatment="k_center",
        control="random",
    )

    design = normalize_design(plan, hypothesis_id="h1", contract=contract)

    assert [item.id for item in design.conditions] == ["control", "treatment"]
    assert design.conditions[0].factor_values["selection_strategy"] == "random"


def test_invalid_design_rejects_duplicate_conditions_and_out_of_plan_values() -> None:
    design = ExperimentDesignSpec(
        id="invalid",
        factors=[
            ExperimentFactorSpec(
                name="strategy", field="selection_strategy", levels=["random", "k_center"]
            )
        ],
        conditions=[
            ExperimentConditionSpec(id="same", factor_values={"strategy": "random"}),
            ExperimentConditionSpec(id="same", factor_values={"strategy": "k_center"}),
        ],
        analysis=ExperimentAnalysisSpec(primary_metric="image_auroc"),
    )

    with pytest.raises(ValueError, match="condition"):
        validate_design(design, _plan())

    invalid_level = design.model_copy(
        update={
            "conditions": [
                ExperimentConditionSpec(id="one", factor_values={"strategy": "query_adaptive"})
            ]
        }
    )
    with pytest.raises(ValueError, match="outside"):
        validate_design(invalid_level, _plan())


def test_fixed_factors_are_allowed_when_conditions_still_vary() -> None:
    design = ExperimentDesignSpec(
        id="fixed_controls",
        design_mode="custom_design",
        factors=[
            ExperimentFactorSpec(
                name="protocol", field="protocol", levels=["pool_compression_m30"]
            ),
            ExperimentFactorSpec(name="detector", field="detector", levels=["patchcore"]),
            ExperimentFactorSpec(name="shots", field="shots", levels=[4]),
            ExperimentFactorSpec(
                name="category", field="category", levels=["bottle", "carpet"]
            ),
        ],
        conditions=[
            ExperimentConditionSpec(
                id="bottle",
                factor_values={
                    "protocol": "pool_compression_m30",
                    "detector": "patchcore",
                    "shots": 4,
                    "category": "bottle",
                },
            ),
            ExperimentConditionSpec(
                id="carpet",
                factor_values={
                    "protocol": "pool_compression_m30",
                    "detector": "patchcore",
                    "shots": 4,
                    "category": "carpet",
                },
            ),
        ],
        analysis=ExperimentAnalysisSpec(
            primary_metric="image_auroc", mode="interaction_summary"
        ),
    )

    validate_design(design, _plan(categories=["bottle", "carpet"]))
    assert design.analysis.mode == "factor_effects"
    assert len(compile_design(design, plan=_plan(categories=["bottle", "carpet"]))) == 2


def test_design_rejects_all_fixed_factors_without_executable_variation() -> None:
    design = ExperimentDesignSpec(
        id="no_variation",
        design_mode="custom_design",
        factors=[ExperimentFactorSpec(name="shots", field="shots", levels=[4])],
        conditions=[
            ExperimentConditionSpec(id="only", factor_values={"shots": 4}),
        ],
        analysis=ExperimentAnalysisSpec(primary_metric="image_auroc"),
    )

    with pytest.raises(ValueError, match="at least two executable condition assignments"):
        validate_design(design, _plan())


def test_design_rejects_duplicate_run_fields_and_unsupported_protocol_strategy() -> None:
    duplicate_field = ExperimentDesignSpec(
        id="duplicate_field",
        factors=[
            ExperimentFactorSpec(name="K", field="shots", levels=[1, 2]),
            ExperimentFactorSpec(name="shot_alias", field="shots", levels=[1, 2]),
        ],
        conditions=[],
        analysis=ExperimentAnalysisSpec(primary_metric="image_auroc"),
    )
    with pytest.raises(ValueError, match="Run field shots cannot be bound"):
        validate_design(duplicate_field, _plan())

    unsupported_combination = ExperimentDesignSpec(
        id="unsupported_combination",
        factors=[
            ExperimentFactorSpec(
                name="protocol",
                field="protocol",
                levels=["strict_k_shot", "pool_compression_m30"],
            ),
            ExperimentFactorSpec(
                name="strategy",
                field="selection_strategy",
                levels=["random", "k_center"],
            ),
        ],
        conditions=[],
        analysis=ExperimentAnalysisSpec(primary_metric="image_auroc"),
    )
    with pytest.raises(ValueError, match="strict_k_shot requires random, got k_center"):
        validate_design(
            unsupported_combination,
            _plan(protocols=["strict_k_shot", "pool_compression_m30"]),
        )


def test_strict_k_shot_rejects_incompatible_contract_default_strategy() -> None:
    protocol_only = ExperimentDesignSpec(
        id="protocol_only",
        hypothesis_id="h1",
        factors=[
            ExperimentFactorSpec(
                name="protocol",
                field="protocol",
                levels=["strict_k_shot", "pool_compression_m30"],
            )
        ],
        conditions=[],
        analysis=ExperimentAnalysisSpec(primary_metric="image_auroc"),
    )
    plan = _plan(protocols=["strict_k_shot", "pool_compression_m30"]).model_copy(
        update={
            "hypothesis_contracts": {
                "h1": AnalysisContract(
                    kind="selection_main_effect",
                    metric="image_auroc",
                    treatment="k_center",
                    control="random",
                )
            }
        }
    )

    with pytest.raises(
        ValueError,
        match=r"strict_k_shot requires AnalysisContract\.treatment=random.*h1=k_center",
    ):
        validate_design(protocol_only, plan)


def test_generic_strict_k_shot_checks_every_applicable_contract_default() -> None:
    generic_design = ExperimentDesignSpec(
        id="generic_protocol",
        factors=[
            ExperimentFactorSpec(
                name="protocol",
                field="protocol",
                levels=["strict_k_shot", "pool_compression_m30"],
            )
        ],
        conditions=[],
        analysis=ExperimentAnalysisSpec(primary_metric="image_auroc"),
    )
    plan = _plan(protocols=["strict_k_shot", "pool_compression_m30"]).model_copy(
        update={
            "hypothesis_ids": ["h1", "h2"],
            "hypothesis_contracts": {
                "h1": AnalysisContract(
                    kind="selection_main_effect",
                    metric="image_auroc",
                    treatment="random",
                    control="k_center",
                ),
                "h2": AnalysisContract(
                    kind="selection_main_effect",
                    metric="image_auroc",
                    treatment="k_center",
                    control="random",
                ),
            },
        }
    )

    with pytest.raises(ValueError, match=r"h2=k_center"):
        validate_design(generic_design, plan)

    plan.hypothesis_contracts["h2"].treatment = "random"
    validate_design(generic_design, plan)


def _campaign_project(
    design: ExperimentDesignSpec,
    max_runs: int,
    *,
    contract_kind: str = "selection_main_effect",
    contract_metric: str = "image_auroc",
    metrics: list[str] | None = None,
    categories: list[str] | None = None,
) -> ResearchProject:
    contract = AnalysisContract(
        kind=contract_kind,
        metric=contract_metric,
        treatment="k_center",
        control="random",
        minimum_pairs=2,
    )
    hypothesis = Hypothesis(
        id="h1",
        gap_id="gap",
        title="design",
        claim="claim",
        null_hypothesis="null",
        rationale="rationale",
        independent_variables=["design"],
        dependent_variables=["image_auroc"],
        predicted_direction="increase",
        falsification_conditions=["zero"],
        analysis_contract=contract,
        status=HypothesisStatus.SHORTLISTED,
    )
    project = ResearchProject(
        spec=ProjectSpec(budget=ComputeBudget(max_experiments=max_runs)),
        hypotheses=[hypothesis],
        experiment_plan=_plan(designs=[design], metrics=metrics, categories=categories),
    )
    manifest = DatasetManifest(
        dataset="MVTec AD",
        root="C:/fixture/mvtec",
        categories=categories or ["bottle"],
        files=[],
        counts={"files": 0},
        digest="a" * 64,
    )
    project.dataset_audits.append(
        DatasetAuditRecord(
            dataset="MVTec AD",
            root=manifest.root,
            manifest_path="C:/fixture/manifest.json",
            digest=manifest.digest,
            categories=manifest.categories,
            verified=True,
        )
    )
    planner = AdaptiveExperimentPlanner()
    campaign, runs = planner.initialize(
        project,
        audit=project.dataset_audits[0],
        dataset=manifest,
        hypothesis_id="h1",
        device="cpu",
        max_rounds=1,
        max_runs=max_runs,
    )
    project.experiment_campaign = campaign
    project.runs = runs
    return project


def test_explicit_design_makes_detector_interaction_executable() -> None:
    design = ExperimentDesignSpec(
        id="detector_interaction",
        hypothesis_id="h1",
        factors=[
            ExperimentFactorSpec(
                name="detector", field="detector", levels=["anomalydino", "patchcore"]
            ),
            ExperimentFactorSpec(
                name="strategy", field="selection_strategy", levels=["random", "k_center"]
            ),
        ],
        conditions=[
            ExperimentConditionSpec(
                id="a", factor_values={"detector": "anomalydino", "strategy": "random"}
            ),
            ExperimentConditionSpec(
                id="b", factor_values={"detector": "patchcore", "strategy": "k_center"}
            ),
        ],
        analysis=ExperimentAnalysisSpec(primary_metric="image_auroc"),
    )
    project = _campaign_project(design, 6, contract_kind="detector_interaction")

    assert AdaptiveExperimentPlanner._contract_is_executable(
        project, project.hypotheses[0].analysis_contract, hypothesis_id="h1"
    )


def test_design_factors_do_not_supply_outer_iteration_cells() -> None:
    design = ExperimentDesignSpec(
        id="k_seed",
        hypothesis_id="h1",
        factors=[
            ExperimentFactorSpec(name="K", field="shots", levels=[1, 2]),
            ExperimentFactorSpec(name="seed", field="seed", levels=[0, 1]),
        ],
        conditions=[
            ExperimentConditionSpec(
                id=f"k{k}_s{seed}", factor_values={"K": k, "seed": seed}
            )
            for k in [1, 2]
            for seed in [0, 1]
        ],
        analysis=ExperimentAnalysisSpec(primary_metric="image_auroc"),
    )
    with pytest.raises(ValueError, match="外层迭代轴"):
        _campaign_project(design, 12)

    project = _campaign_project(
        design,
        12,
        categories=["bottle", "carpet", "capsule"],
    )
    planner = AdaptiveExperimentPlanner()
    assert len(project.runs) == 4
    assert {cell.category for cell in planner.allowed_next_cells(project)} == {
        "carpet",
        "capsule",
    }
    added = planner.fill_current_round(project, target_cells=3)
    assert len(added) == 8
    all_runs = [*project.runs, *added]
    assert len(all_runs) == 12
    assert {
        run.category for run in all_runs
    } == {"bottle", "carpet", "capsule"}


def test_design_baseline_must_reference_a_real_condition() -> None:
    design = ExperimentDesignSpec(
        id="bad_baseline",
        hypothesis_id="h1",
        factors=[
            ExperimentFactorSpec(
                name="strategy", field="selection_strategy", levels=["random", "k_center"]
            )
        ],
        conditions=[
            ExperimentConditionSpec(id="random", factor_values={"strategy": "random"}),
            ExperimentConditionSpec(id="k_center", factor_values={"strategy": "k_center"}),
        ],
        analysis=ExperimentAnalysisSpec(
            primary_metric="image_auroc", baseline_condition_id="missing"
        ),
    )
    with pytest.raises(ValueError, match="baseline_condition_id"):
        validate_design(design, _plan())


def test_design_metric_overrides_legacy_contract_metric() -> None:
    design = ExperimentDesignSpec(
        id="metric_override",
        hypothesis_id="h1",
        factors=[
            ExperimentFactorSpec(
                name="strategy", field="selection_strategy", levels=["random", "k_center"]
            )
        ],
        conditions=[
            ExperimentConditionSpec(id="r", factor_values={"strategy": "random"}),
            ExperimentConditionSpec(id="k", factor_values={"strategy": "k_center"}),
        ],
        analysis=ExperimentAnalysisSpec(primary_metric="image_ap"),
    )
    project = _campaign_project(
        design,
        6,
        contract_metric="image_auroc",
        metrics=["image_auroc", "image_ap"],
    )

    assert project.experiment_campaign is not None
    assert project.experiment_campaign.metric == "image_ap"
    assert project.experiment_campaign.rounds[0].metric == "image_ap"
    user_text = " ".join(
        [
            project.experiment_campaign.rounds[0].objective,
            project.experiment_campaign.rounds[0].rationale,
            *(node.objective for node in project.experiment_campaign.nodes),
        ]
    )
    assert all(term not in user_text for term in ("成对", "treatment", "control"))

    for run, value in zip(project.runs, [0.6, 0.8], strict=True):
        run.status = RunStatus.SUCCEEDED
        run.verified = True
        run.metrics = {"image_ap": value, "image_auroc": 0.1}
    summary = AdaptiveExperimentPlanner().summarize_current_round(project)

    assert summary["metric"] == "image_ap"
    assert summary["condition_statistics"][0]["mean"] in {0.6, 0.8}


def test_three_condition_budget_fill_never_exceeds_campaign_limit() -> None:
    design = ExperimentDesignSpec(
        id="three_conditions",
        hypothesis_id="h1",
        factors=[
            ExperimentFactorSpec(
                name="strategy", field="selection_strategy", levels=["random", "k_center"]
            ),
            ExperimentFactorSpec(
                name="detector", field="detector", levels=["anomalydino", "patchcore"]
            ),
        ],
        conditions=[
            ExperimentConditionSpec(
                id="r_a", factor_values={"strategy": "random", "detector": "anomalydino"}
            ),
            ExperimentConditionSpec(
                id="k_a", factor_values={"strategy": "k_center", "detector": "anomalydino"}
            ),
            ExperimentConditionSpec(
                id="r_p", factor_values={"strategy": "random", "detector": "patchcore"}
            ),
        ],
        analysis=ExperimentAnalysisSpec(primary_metric="image_auroc"),
    )
    project = _campaign_project(design, 9)
    planner = AdaptiveExperimentPlanner()
    added = planner.fill_current_round(project, target_cells=3)

    assert len(project.runs) + len(added) == 9
    assert len(added) == 6
    assert len(project.runs) + len(added) <= project.experiment_campaign.max_runs


def test_multi_arm_design_summary_keeps_condition_provenance() -> None:
    design = ExperimentDesignSpec(
        id="multi_arm",
        hypothesis_id="h1",
        factors=[
            ExperimentFactorSpec(
                name="strategy", field="selection_strategy", levels=["random", "k_center"]
            ),
            ExperimentFactorSpec(
                name="detector", field="detector", levels=["anomalydino", "patchcore"]
            ),
        ],
        conditions=[
            ExperimentConditionSpec(
                id="control",
                factor_values={"strategy": "random", "detector": "anomalydino"},
            ),
            ExperimentConditionSpec(
                id="treatment",
                factor_values={"strategy": "k_center", "detector": "anomalydino"},
            ),
            ExperimentConditionSpec(
                id="alternate",
                factor_values={"strategy": "random", "detector": "patchcore"},
            ),
        ],
        analysis=ExperimentAnalysisSpec(primary_metric="image_auroc", mode="group_comparison"),
    )
    project = _campaign_project(design, 9)
    for run, value in zip(project.runs, [0.80, 0.85, 0.82], strict=True):
        run.status = RunStatus.SUCCEEDED
        run.verified = True
        run.metrics = {"image_auroc": value}

    summary = AdaptiveExperimentPlanner().summarize_current_round(project)

    assert summary["analysis_mode"] == "group_comparison"
    assert {item["condition_id"] for item in summary["condition_statistics"]} == {
        "control",
        "treatment",
        "alternate",
    }
    assert summary["source_run_ids"] == [run.id for run in project.runs]


def test_two_factor_and_ordered_trend_summaries() -> None:
    trend_design = ExperimentDesignSpec(
        id="k_trend",
        hypothesis_id="h1",
        factors=[ExperimentFactorSpec(name="K", field="shots", levels=[1, 2, 4])],
        conditions=[
            ExperimentConditionSpec(id="k1", factor_values={"K": 1}),
            ExperimentConditionSpec(id="k2", factor_values={"K": 2}),
            ExperimentConditionSpec(id="k4", factor_values={"K": 4}),
        ],
        analysis=ExperimentAnalysisSpec(
            primary_metric="image_auroc", mode="ordered_trend", ordered_factor="K"
        ),
    )
    project = _campaign_project(trend_design, 9)
    for run, value in zip(project.runs, [0.70, 0.80, 0.86], strict=True):
        run.status = RunStatus.SUCCEEDED
        run.verified = True
        run.metrics = {"image_auroc": value}
    summary = AdaptiveExperimentPlanner().summarize_current_round(project)
    assert [point["level"] for point in summary["ordered_trend"]] == [1, 2, 4]

    factorial_design = ExperimentDesignSpec(
        id="two_factor",
        hypothesis_id="h1",
        factors=[
            ExperimentFactorSpec(
                name="strategy", field="selection_strategy", levels=["random", "k_center"]
            ),
            ExperimentFactorSpec(name="K", field="shots", levels=[1, 2]),
        ],
        conditions=[
            ExperimentConditionSpec(
                id=condition_id,
                factor_values={"strategy": strategy, "K": shots},
            )
            for condition_id, strategy, shots in (
                ("r1", "random", 1),
                ("k1", "k_center", 1),
                ("r2", "random", 2),
                ("k2", "k_center", 2),
            )
        ],
        analysis=ExperimentAnalysisSpec(primary_metric="image_auroc", mode="factor_effects"),
    )
    factorial_project = _campaign_project(factorial_design, 12)
    for run, value in zip(factorial_project.runs, [0.70, 0.80, 0.75, 0.90], strict=True):
        run.status = RunStatus.SUCCEEDED
        run.verified = True
        run.metrics = {"image_auroc": value}
    factorial_summary = AdaptiveExperimentPlanner().summarize_current_round(factorial_project)
    assert len(factorial_summary["factor_effects"]) == 2
    assert factorial_summary["sample_size"] == 4
    interaction = factorial_summary["interaction_summary"][0]
    assert interaction["difference_in_differences"] == 0.05
    assert interaction["sample_size"] == 4
    assert factorial_project.experiment_campaign.summary is not None
    assert (
        factorial_project.experiment_campaign.summary.interaction_summary[
            0
        ].difference_in_differences
        == 0.05
    )

    distribution_design = factorial_design.model_copy(
        update={
            "id": "distribution",
            "analysis": ExperimentAnalysisSpec(
                primary_metric="image_auroc", mode="distribution_summary"
            ),
        }
    )
    distribution_project = _campaign_project(distribution_design, 12)
    for run, value in zip(distribution_project.runs, [0.70, 0.80, 0.75, 0.90], strict=True):
        run.status = RunStatus.SUCCEEDED
        run.verified = True
        run.metrics = {"image_auroc": value}
    distribution_summary = AdaptiveExperimentPlanner().summarize_current_round(
        distribution_project
    )
    assert distribution_summary["distribution_summary"]["sample_size"] == 4


@pytest.mark.parametrize("runtime_class", [MockScientistRuntime, QwenScientistRuntime])
def test_explicit_ordered_trend_analysis_uses_generic_round_summary(runtime_class) -> None:
    design = ExperimentDesignSpec(
        id="finding_trend",
        hypothesis_id="h1",
        factors=[ExperimentFactorSpec(name="K", field="shots", levels=[1, 2, 4])],
        conditions=[
            ExperimentConditionSpec(id=f"k{level}", factor_values={"K": level})
            for level in [1, 2, 4]
        ],
        analysis=ExperimentAnalysisSpec(
            primary_metric="image_auroc", mode="ordered_trend", ordered_factor="K"
        ),
    )
    project = _campaign_project(design, 9)
    for run, value in zip(project.runs, [0.70, 0.80, 0.86], strict=True):
        run.status = RunStatus.SUCCEEDED
        run.verified = True
        run.metrics = {"image_auroc": value}
    summary = AdaptiveExperimentPlanner().summarize_current_round(project)
    campaign = project.experiment_campaign
    assert campaign is not None
    campaign.rounds[-1].summary = None
    campaign.rounds[-1].result_summary = {"summary": summary["summary"]}
    project.experiment_campaign_history.append(campaign)
    project.experiment_campaign = None

    findings = asyncio.run(runtime_class().analyze_results(project))
    finding = next(item for item in findings if item.hypothesis_id == "h1")

    assert finding.sample_size == 3
    assert finding.supporting_run_ids == [run.id for run in project.runs]
    assert finding.analysis_method == "descriptive_ordered_trend"
    assert finding.claim_verdict == "inconclusive"
    assert finding.verified is False
    assert "首尾呈上升趋势" in finding.statement
    assert "未执行推断统计" in finding.statement
    assert "成对结果不足" not in finding.statement


def test_factor_effects_cover_every_factor_and_pairwise_interaction() -> None:
    design = ExperimentDesignSpec(
        id="three_factor",
        hypothesis_id="h1",
        factors=[
            ExperimentFactorSpec(
                name="strategy", field="selection_strategy", levels=["random", "k_center"]
            ),
            ExperimentFactorSpec(name="K", field="shots", levels=[1, 2]),
            ExperimentFactorSpec(
                name="detector", field="detector", levels=["anomalydino", "patchcore"]
            ),
        ],
        conditions=[],
        analysis=ExperimentAnalysisSpec(primary_metric="image_auroc", mode="factor_effects"),
    )
    project = _campaign_project(design, 24)
    for index, run in enumerate(project.runs):
        run.status = RunStatus.SUCCEEDED
        run.verified = True
        run.metrics = {"image_auroc": 0.6 + index * 0.02}

    summary = AdaptiveExperimentPlanner().summarize_current_round(project)

    assert {item["factor"] for item in summary["factor_effects"]} == {
        "strategy",
        "K",
        "detector",
    }
    assert {
        frozenset((item["factor_a"], item["factor_b"]))
        for item in summary["interaction_summary"]
    } == {
        frozenset(("strategy", "K")),
        frozenset(("strategy", "detector")),
        frozenset(("K", "detector")),
    }


def test_three_by_three_interaction_requires_complete_cells_and_has_no_did() -> None:
    categories = ["bottle", "cable", "capsule"]
    design = ExperimentDesignSpec(
        id="three_by_three",
        hypothesis_id="h1",
        factors=[
            ExperimentFactorSpec(name="category", field="category", levels=categories),
            ExperimentFactorSpec(name="K", field="shots", levels=[1, 2, 4]),
        ],
        conditions=[],
        analysis=ExperimentAnalysisSpec(primary_metric="image_auroc", mode="factor_effects"),
    )
    project = _campaign_project(design, 27, categories=categories)
    for index, run in enumerate(project.runs):
        run.status = RunStatus.SUCCEEDED
        run.verified = True
        run.metrics = {"image_auroc": 0.6 + index * 0.01}

    complete_summary = AdaptiveExperimentPlanner().summarize_current_round(project)
    interaction = complete_summary["interaction_summary"][0]
    assert len(interaction["cell_means"]) == 9
    assert interaction["difference_in_differences"] is None

    heatmap_spec = ExperimentCardPresentationSpec(
        blocks=[
            ExperimentCardBlockSpec(kind="narrative", source="design"),
            ExperimentCardBlockSpec(kind="progress", source="progress"),
            ExperimentCardBlockSpec(
                kind="chart", source="interaction_summary", chart_mark="heatmap"
            ),
            ExperimentCardBlockSpec(kind="evidence", source="evidence"),
        ]
    )
    assert is_result_compatible_presentation_spec(heatmap_spec, design, complete_summary)

    project.runs[-1].verified = False
    incomplete_summary = AdaptiveExperimentPlanner().summarize_current_round(project)
    incomplete_interaction = incomplete_summary["interaction_summary"][0]
    assert len(incomplete_interaction["cell_means"]) == 8
    assert incomplete_interaction["difference_in_differences"] is None
    assert not is_result_compatible_presentation_spec(
        heatmap_spec, design, incomplete_summary
    )


def test_exhaustive_count_uses_unbound_outer_axes_and_design_conditions() -> None:
    design = ExperimentDesignSpec(
        id="bound_k",
        hypothesis_id="h1",
        factors=[ExperimentFactorSpec(name="K", field="shots", levels=[1, 2, 4])],
        conditions=[
            ExperimentConditionSpec(id=f"k_{level}", factor_values={"K": level})
            for level in [1, 2, 4]
        ],
        analysis=ExperimentAnalysisSpec(
            primary_metric="image_auroc", mode="ordered_trend", ordered_factor="K"
        ),
    )
    project = _campaign_project(design, 9)

    assert project.experiment_campaign is not None
    assert project.experiment_campaign.exhaustive_run_count == 9


def test_mock_plan_preserves_each_shortlisted_hypothesis_comparison() -> None:
    seed_design = ExperimentDesignSpec(
        id="seed",
        hypothesis_id="h1",
        factors=[
            ExperimentFactorSpec(
                name="strategy", field="selection_strategy", levels=["random", "k_center"]
            )
        ],
        conditions=[
            ExperimentConditionSpec(id="r", factor_values={"strategy": "random"}),
            ExperimentConditionSpec(id="k", factor_values={"strategy": "k_center"}),
        ],
        analysis=ExperimentAnalysisSpec(primary_metric="image_auroc"),
    )
    project = _campaign_project(seed_design, 6)
    project.hypotheses.append(
        project.hypotheses[0].model_copy(update={"id": "h2"}, deep=True)
    )

    plan = asyncio.run(MockScientistRuntime().design_experiments(project))

    assert len(plan.designs) == 2
    assert all(design.analysis.mode == "group_comparison" for design in plan.designs)
    assert all(
        design.factors[0].levels == ["random", "k_center"] for design in plan.designs
    )


def test_mock_default_flow_selects_paired_and_custom_designs() -> None:
    runtime = MockScientistRuntime()
    project = ResearchProject(spec=ProjectSpec())

    project.gaps = asyncio.run(runtime.discover_gaps(project))
    project.hypotheses = asyncio.run(runtime.propose_hypotheses(project))
    proposed_modes = {
        hypothesis.analysis_contract.design_mode
        for hypothesis in project.hypotheses
        if hypothesis.analysis_contract is not None
    }
    assert proposed_modes == {"paired_comparison", "custom_design"}

    project.hypotheses = asyncio.run(runtime.review_hypotheses(project))
    plan = asyncio.run(runtime.design_experiments(project))

    assert {design.design_mode for design in plan.designs} == {
        "paired_comparison",
        "custom_design",
    }
    custom_design = next(
        design for design in plan.designs if design.design_mode == "custom_design"
    )
    assert custom_design.factors[0].field == "shots"
    assert custom_design.conditions
    assert project.hypotheses[0].analysis_contract is not None
    assert project.hypotheses[0].analysis_contract.design_mode == "paired_comparison"
    assert project.hypotheses[1].analysis_contract is not None
    assert project.hypotheses[1].analysis_contract.design_mode == "paired_comparison"
    assert project.hypotheses[2].analysis_contract is not None
    assert project.hypotheses[2].analysis_contract.design_mode == "custom_design"


def test_mock_plan_skips_same_condition_contract_and_uses_detector_factor() -> None:
    project = ResearchProject(spec=ProjectSpec())
    invalid = Hypothesis(
        id="invalid",
        gap_id="gap",
        title="无效比较",
        claim="同一检测器提升效果。",
        null_hypothesis="无差异。",
        rationale="该对照不成立。",
        independent_variables=["检测器"],
        dependent_variables=["image_auroc"],
        predicted_direction="无",
        falsification_conditions=["不可比较"],
        status=HypothesisStatus.SHORTLISTED,
        analysis_contract=AnalysisContract(
            kind="selection_main_effect",
            metric="image_auroc",
            treatment="patchcore",
            control="patchcore",
        ),
    )
    valid = invalid.model_copy(
        update={
            "id": "valid",
            "title": "检测器比较",
            "status": HypothesisStatus.SHORTLISTED,
            "analysis_contract": AnalysisContract(
                kind="detector_interaction",
                metric="image_auroc",
                treatment="anomalydino",
                control="patchcore",
            ),
        },
        deep=True,
    )
    project.hypotheses = [invalid, valid]

    plan = asyncio.run(MockScientistRuntime().design_experiments(project))

    assert [design.hypothesis_id for design in plan.designs] == ["valid"]
    assert plan.designs[0].factors[0].field == "detector"
    assert plan.designs[0].factors[0].levels == ["patchcore", "anomalydino"]


def test_mock_detector_interaction_with_strategy_arms_uses_strategy_factor() -> None:
    project = ResearchProject(spec=ProjectSpec())
    hypothesis = Hypothesis(
        id="strategy_interaction",
        gap_id="gap",
        title="策略交互",
        claim="策略与检测器存在交互。",
        null_hypothesis="不存在交互。",
        rationale="比较两个选样策略。",
        independent_variables=["选样策略"],
        dependent_variables=["image_auroc"],
        predicted_direction="正向",
        falsification_conditions=["无差异"],
        status=HypothesisStatus.SHORTLISTED,
        analysis_contract=AnalysisContract(
            kind="detector_interaction",
            metric="image_auroc",
            treatment="k_center",
            control="random",
        ),
    )
    project.hypotheses = [hypothesis]

    plan = asyncio.run(MockScientistRuntime().design_experiments(project))

    assert plan.designs[0].factors[0].field == "selection_strategy"
    assert plan.designs[0].factors[0].levels == ["random", "k_center"]


def test_custom_design_without_treatment_control_binds_detector_and_default_strategy() -> None:
    design = ExperimentDesignSpec(
        id="custom_detector",
        hypothesis_id="h1",
        design_mode="custom_design",
        factors=[
            ExperimentFactorSpec(
                name="detector", field="detector", levels=["anomalydino", "patchcore"]
            )
        ],
        conditions=[
            ExperimentConditionSpec(id="dino", factor_values={"detector": "anomalydino"}),
            ExperimentConditionSpec(id="patch", factor_values={"detector": "patchcore"}),
        ],
        support_selection_strategy="random",
        analysis=ExperimentAnalysisSpec(
            primary_metric="image_auroc", mode="group_comparison"
        ),
    )
    project = _campaign_project(design, 6)
    project.experiment_campaign = None
    project.runs = []
    custom_contract = AnalysisContract(
        kind="detector_interaction",
        metric="image_auroc",
        design_mode="custom_design",
        treatment=None,
        control=None,
        minimum_pairs=2,
    )
    project.hypotheses[0].analysis_contract = custom_contract
    assert project.experiment_plan is not None
    project.experiment_plan.hypothesis_contracts = {"h1": custom_contract}

    planner = AdaptiveExperimentPlanner()
    campaign, runs = planner.initialize(
        project,
        audit=project.dataset_audits[0],
        dataset=DatasetManifest(
            dataset="MVTec AD",
            root="C:/fixture/mvtec",
            categories=["bottle"],
            files=[],
            counts={"files": 0},
            digest="a" * 64,
        ),
        hypothesis_id="h1",
        device="cpu",
        max_rounds=1,
        max_runs=6,
    )

    assert campaign.treatment == ""
    assert {run.detector for run in runs} == {"anomalydino", "patchcore"}
    assert {run.selection_strategy for run in runs} == {"random"}
    assert {run.condition_id for run in runs} == {"dino", "patch"}

    for run, value in zip(runs, [0.72, 0.81], strict=True):
        run.status = RunStatus.SUCCEEDED
        run.verified = True
        run.metrics = {"image_auroc": value}
    project.experiment_campaign = campaign
    project.runs = runs
    summary = planner.summarize_current_round(project)
    assert {item["condition_id"] for item in summary["condition_statistics"]} == {
        "dino",
        "patch",
    }
    assert "pair_differences" not in summary
    assert "treatment_run_id" not in summary
    assert "control_run_id" not in summary
    assert "mean_difference" not in summary
    assert "pair_count" not in summary


def test_default_presentation_specs_follow_analysis_mode() -> None:
    designs = [
        ExperimentDesignSpec(
            id="arms",
            factors=[
                ExperimentFactorSpec(
                    name="strategy",
                    field="selection_strategy",
                    levels=["random", "k_center"],
                )
            ],
            conditions=[
                ExperimentConditionSpec(id="r", factor_values={"strategy": "random"}),
                ExperimentConditionSpec(id="k", factor_values={"strategy": "k_center"}),
                ExperimentConditionSpec(id="a", factor_values={"strategy": "random"}),
            ],
            analysis=ExperimentAnalysisSpec(primary_metric="image_auroc", mode="group_comparison"),
        ),
        ExperimentDesignSpec(
            id="trend",
            factors=[ExperimentFactorSpec(name="K", field="shots", levels=[1, 2, 4])],
            conditions=[
                ExperimentConditionSpec(id=f"k{level}", factor_values={"K": level})
                for level in [1, 2, 4]
            ],
            analysis=ExperimentAnalysisSpec(
                primary_metric="image_auroc", mode="ordered_trend", ordered_factor="K"
            ),
        ),
        ExperimentDesignSpec(
            id="factorial",
            factors=[
                ExperimentFactorSpec(
                    name="strategy", field="selection_strategy", levels=["random", "k_center"]
                ),
                ExperimentFactorSpec(name="K", field="shots", levels=[1, 2]),
            ],
            conditions=[],
            analysis=ExperimentAnalysisSpec(primary_metric="image_auroc", mode="factor_effects"),
        ),
    ]


    specs = [default_presentation_spec(design) for design in designs]

    assert [spec.layout for spec in specs] == ["stack", "sequence", "grid"]
    assert [
        [(block.kind, block.source, block.chart_mark) for block in spec.blocks]
        for spec in specs
    ] == [
        [
            ("narrative", "design", None),
            ("progress", "progress", None),
            ("table", "condition_statistics", None),
            ("chart", "condition_effects", "bar"),
            ("evidence", "evidence", None),
        ],
        [
            ("narrative", "design", None),
            ("progress", "progress", None),
            ("chart", "ordered_trend", "line"),
            ("table", "condition_statistics", None),
            ("evidence", "evidence", None),
        ],
        [
            ("narrative", "design", None),
            ("progress", "progress", None),
            ("chart", "factor_effects", "bar"),
            ("chart", "interaction_summary", "heatmap"),
            ("evidence", "evidence", None),
        ],
    ]


def test_result_aware_presentation_changes_blocks_for_observed_results() -> None:
    trend_design = ExperimentDesignSpec(
        id="result_trend",
        factors=[ExperimentFactorSpec(name="K", field="shots", levels=[1, 2, 4])],
        conditions=[
            ExperimentConditionSpec(id=f"k{level}", factor_values={"K": level})
            for level in [1, 2, 4]
        ],
        analysis=ExperimentAnalysisSpec(
            primary_metric="image_auroc", mode="ordered_trend", ordered_factor="K"
        ),
    )
    complete_spec = result_aware_presentation_spec(
        trend_design,
        {
            "ordered_trend": [
                {"level": 1, "mean": 0.6},
                {"level": 2, "mean": 0.7},
            ]
        },
    )
    assert ("chart", "ordered_trend", "line") in {
        (block.kind, block.source, block.chart_mark) for block in complete_spec.blocks
    }

    incomplete_spec = result_aware_presentation_spec(
        trend_design,
        {"ordered_trend": [{"level": 1, "mean": 0.6}, {"level": 2, "mean": None}]},
    )
    assert ("table", "ordered_trend", None) in {
        (block.kind, block.source, block.chart_mark) for block in incomplete_spec.blocks
    }

    factorial_design = ExperimentDesignSpec(
        id="result_factorial",
        factors=[
            ExperimentFactorSpec(
                name="strategy", field="selection_strategy", levels=["random", "k_center"]
            ),
            ExperimentFactorSpec(name="K", field="shots", levels=[1, 2]),
        ],
        conditions=[],
        analysis=ExperimentAnalysisSpec(primary_metric="image_auroc", mode="factor_effects"),
    )
    failed_spec = result_aware_presentation_spec(
        factorial_design,
        {"failed_run_ids": ["run_failed"], "factor_effects": [], "interaction_summary": []},
    )
    block_kinds = {(block.kind, block.source) for block in failed_spec.blocks}
    assert ("diagnostics", "diagnostics") in block_kinds
    assert ("runs", "runs") in block_kinds
    assert ("chart", "interaction_summary") not in {
        (block.kind, block.source) for block in failed_spec.blocks
    }

    distribution_design = factorial_design.model_copy(
        update={
            "id": "result_distribution",
            "analysis": ExperimentAnalysisSpec(
                primary_metric="image_auroc", mode="distribution_summary"
            ),
        }
    )
    distribution_failed_spec = result_aware_presentation_spec(
        distribution_design,
        {
            "failed_run_ids": ["run_failed"],
            "distribution_summary": {"minimum": 0.4, "mean": 0.6, "maximum": 0.8},
        },
    )
    assert sum(block.kind == "runs" for block in distribution_failed_spec.blocks) == 1


def test_result_presentation_compatibility_rejects_empty_result_charts() -> None:
    trend_design = ExperimentDesignSpec(
        id="compatibility_trend",
        factors=[ExperimentFactorSpec(name="K", field="shots", levels=[1, 2])],
        conditions=[
            ExperimentConditionSpec(id="k1", factor_values={"K": 1}),
            ExperimentConditionSpec(id="k2", factor_values={"K": 2}),
        ],
        analysis=ExperimentAnalysisSpec(
            primary_metric="image_auroc", mode="ordered_trend", ordered_factor="K"
        ),
    )
    invalid_trend = ExperimentCardPresentationSpec(
        blocks=[
            ExperimentCardBlockSpec(kind="narrative", source="design"),
            ExperimentCardBlockSpec(kind="progress", source="progress"),
            ExperimentCardBlockSpec(
                kind="chart", source="ordered_trend", chart_mark="line"
            ),
            ExperimentCardBlockSpec(kind="evidence", source="evidence"),
        ]
    )
    assert not is_result_compatible_presentation_spec(
        invalid_trend,
        trend_design,
        {"ordered_trend": [{"level": 1, "mean": 0.5}]},
    )

    factorial_design = ExperimentDesignSpec(
        id="compatibility_factorial",
        factors=[
            ExperimentFactorSpec(
                name="strategy", field="selection_strategy", levels=["random", "k_center"]
            ),
            ExperimentFactorSpec(name="K", field="shots", levels=[1, 2]),
        ],
        conditions=[],
        analysis=ExperimentAnalysisSpec(primary_metric="image_auroc", mode="factor_effects"),
    )
    invalid_interaction = ExperimentCardPresentationSpec(
        blocks=[
            ExperimentCardBlockSpec(kind="narrative", source="design"),
            ExperimentCardBlockSpec(kind="progress", source="progress"),
            ExperimentCardBlockSpec(
                kind="chart", source="interaction_summary", chart_mark="heatmap"
            ),
            ExperimentCardBlockSpec(kind="evidence", source="evidence"),
        ]
    )
    partial_interaction = {
        "factor_a": "strategy",
        "factor_b": "K",
        "difference_in_differences": None,
        "cell_means": {"strategy=random|K=1": 0.7},
    }
    assert not is_result_compatible_presentation_spec(
        invalid_interaction,
        factorial_design,
        {"interaction_summary": [partial_interaction]},
    )
    fallback = result_aware_presentation_spec(
        factorial_design,
        {"factor_effects": [], "interaction_summary": [partial_interaction]},
    )
    assert any(
        block.kind == "table" and block.source == "interaction_summary"
        for block in fallback.blocks
    )

    complete_interaction = {
        **partial_interaction,
        "cell_means": {
            f"strategy={strategy}|K={level}": 0.7
            for strategy in ["random", "k_center"]
            for level in [1, 2]
        },
    }
    assert is_result_compatible_presentation_spec(
        invalid_interaction,
        factorial_design,
        {"interaction_summary": [complete_interaction]},
    )

    invalid_failure = ExperimentCardPresentationSpec(
        blocks=[
            ExperimentCardBlockSpec(kind="narrative", source="design"),
            ExperimentCardBlockSpec(kind="progress", source="progress"),
            ExperimentCardBlockSpec(kind="table", source="condition_statistics"),
            ExperimentCardBlockSpec(kind="evidence", source="evidence"),
        ]
    )
    assert not is_result_compatible_presentation_spec(
        invalid_failure,
        factorial_design,
        {"failed_run_ids": ["run-1"]},
    )


def test_interaction_compatibility_checks_every_pair_independent_of_order() -> None:
    design = ExperimentDesignSpec(
        id="three_factor_interactions",
        factors=[
            ExperimentFactorSpec(
                name="strategy", field="selection_strategy", levels=["random", "k_center"]
            ),
            ExperimentFactorSpec(name="K", field="shots", levels=[1, 2]),
            ExperimentFactorSpec(
                name="detector", field="detector", levels=["anomalydino", "patchcore"]
            ),
        ],
        conditions=[],
        analysis=ExperimentAnalysisSpec(primary_metric="image_auroc", mode="factor_effects"),
    )
    partial = {
        "factor_a": "strategy",
        "factor_b": "K",
        "cell_means": {"strategy=random|K=1": 0.70},
    }
    complete = {
        "factor_a": "strategy",
        "factor_b": "detector",
        "cell_means": {
            f"strategy={strategy}|detector={detector}": 0.75
            for strategy in ["random", "k_center"]
            for detector in ["anomalydino", "patchcore"]
        },
    }
    chart_spec = ExperimentCardPresentationSpec(
        blocks=[
            ExperimentCardBlockSpec(kind="narrative", source="design"),
            ExperimentCardBlockSpec(kind="progress", source="progress"),
            ExperimentCardBlockSpec(
                kind="chart", source="interaction_summary", chart_mark="heatmap"
            ),
            ExperimentCardBlockSpec(kind="evidence", source="evidence"),
        ]
    )

    for items in ([partial, complete], [complete, partial]):
        summary = {"factor_effects": [], "interaction_summary": items}
        generated = result_aware_presentation_spec(design, summary)
        assert any(
            block.kind == "chart" and block.source == "interaction_summary"
            for block in generated.blocks
        )
        assert is_result_compatible_presentation_spec(chart_spec, design, summary)

    second_partial = {
        "factor_a": "K",
        "factor_b": "detector",
        "cell_means": {"K=1|detector=anomalydino": 0.72},
    }
    all_partial_summary = {
        "factor_effects": [],
        "interaction_summary": [partial, second_partial],
    }
    generated = result_aware_presentation_spec(design, all_partial_summary)
    assert any(
        block.kind == "table" and block.source == "interaction_summary"
        for block in generated.blocks
    )
    assert not is_result_compatible_presentation_spec(
        chart_spec, design, all_partial_summary
    )
    assert is_result_compatible_presentation_spec(
        generated, design, all_partial_summary
    )

@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "unknown", "source": "design"},
        {"kind": "table", "source": "unknown"},
        {"kind": "chart", "source": "condition_effects", "chart_mark": "unknown"},
        {"kind": "table", "source": "condition_statistics", "options": {}},
    ],
)
def test_presentation_dsl_rejects_invalid_blocks(payload: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ExperimentCardBlockSpec.model_validate(payload)


@pytest.mark.parametrize(
    "blocks",
    [
        [
            {"kind": "progress", "source": "progress"},
            {"kind": "evidence", "source": "evidence"},
        ],
        [
            {"kind": "narrative", "source": "progress"},
            {"kind": "progress", "source": "progress"},
            {"kind": "evidence", "source": "evidence"},
        ],
        [
            {"kind": "narrative", "source": "design"},
            {"kind": "progress", "source": "progress"},
            {"kind": "evidence", "source": "evidence"},
            {"kind": "chart", "source": "ordered_trend", "chart_mark": "bar"},
        ],
        [
            {"kind": "narrative", "source": "design"},
            {"kind": "progress", "source": "progress"},
            {"kind": "evidence", "source": "evidence"},
            {"kind": "chart", "source": "interaction_summary", "chart_mark": "bar"},
        ],
        [
            {"kind": "narrative", "source": "design"},
            {"kind": "progress", "source": "progress"},
            {"kind": "evidence", "source": "evidence"},
            {"kind": "chart", "source": "distribution_summary", "chart_mark": "point"},
        ],
        [
            {"kind": "narrative", "source": "design"},
            {"kind": "progress", "source": "progress"},
            {"kind": "evidence", "source": "evidence"},
            {"kind": "decision", "source": "evidence"},
        ],
    ],
)
def test_presentation_dsl_rejects_invalid_semantics(
    blocks: list[dict[str, object]],
) -> None:
    with pytest.raises(ValueError):
        ExperimentCardPresentationSpec.model_validate(
            {"schema_version": 2, "layout": "stack", "density": "comfortable", "blocks": blocks}
        )


def test_presentation_dsl_requires_a_result_block() -> None:
    with pytest.raises(ValueError, match="result block"):
        ExperimentCardPresentationSpec(
            blocks=[
                ExperimentCardBlockSpec(kind="narrative", source="design"),
                ExperimentCardBlockSpec(kind="progress", source="progress"),
                ExperimentCardBlockSpec(kind="evidence", source="evidence"),
            ]
        )


def test_presentation_normalizer_repairs_distribution_point_to_supported_bar() -> None:
    normalized = normalize_presentation_spec_payload(
        {
            "schema_version": 2,
            "layout": "split",
            "blocks": [
                {"kind": "narrative", "source": "design"},
                {"kind": "progress", "source": "progress"},
                {
                    "kind": "chart",
                    "source": "distribution_summary",
                    "chart_mark": "point",
                },
                {"kind": "evidence", "source": "evidence"},
            ],
        },
        analysis_mode="distribution_summary",
    )

    spec = ExperimentCardPresentationSpec.model_validate(normalized)
    distribution_chart = next(
        block for block in spec.blocks if block.source == "distribution_summary"
    )
    assert distribution_chart.kind == "chart"
    assert distribution_chart.chart_mark == "bar"


def test_presentation_normalizer_maps_comparison_source_to_condition_statistics() -> None:
    normalized = normalize_presentation_spec_payload(
        {
            "schema_version": 2,
            "blocks": [
                {"kind": "narrative", "source": "design"},
                {"kind": "progress", "source": "progress"},
                {
                    "kind": "chart",
                    "source": "group_comparison",
                    "chart_mark": "bar",
                },
                {"kind": "evidence", "source": "evidence"},
            ],
        },
        analysis_mode="group_comparison",
    )

    spec = ExperimentCardPresentationSpec.model_validate(normalized)
    comparison_chart = next(block for block in spec.blocks if block.kind == "chart")
    assert comparison_chart.source == "condition_statistics"
    assert comparison_chart.chart_mark == "bar"


def test_condition_normalizer_deduplicates_same_factor_assignments() -> None:
    conditions = normalize_design_conditions_payload(
        [
            {
                "id": "baseline",
                "label": "基线",
                "factor_values": {"category": "bottle"},
            },
            {
                "id": "alternate_id",
                "label": "另一个标签",
                "factor_values": {"category": "bottle"},
            },
            {
                "id": "baseline",
                "label": "另一条件",
                "factor_values": {"category": "carpet"},
            },
        ]
    )

    assert len(conditions) == 2
    design = ExperimentDesignSpec(
        id="conflicting_condition_id",
        factors=[
            ExperimentFactorSpec(
                name="category", field="category", levels=["bottle", "carpet"]
            )
        ],
        conditions=conditions,
        analysis=ExperimentAnalysisSpec(primary_metric="image_auroc"),
    )
    with pytest.raises(ValueError, match="Duplicate experiment condition"):
        validate_design(design, _plan(categories=["bottle", "carpet"]))


def test_round_deep_copies_design_presentation_and_old_plan_defaults() -> None:
    design = ExperimentDesignSpec(
        id="copy_me",
        hypothesis_id="h1",
        question="复制这个布局",
        rationale="保持注册展示不变",
        factors=[
            ExperimentFactorSpec(
                name="strategy", field="selection_strategy", levels=["random", "k_center"]
            )
        ],
        conditions=[
            ExperimentConditionSpec(id="r", factor_values={"strategy": "random"}),
            ExperimentConditionSpec(id="k", factor_values={"strategy": "k_center"}),
        ],
        analysis=ExperimentAnalysisSpec(primary_metric="image_auroc"),
        presentation_spec=ExperimentCardPresentationSpec(
            layout="split",
            density="compact",
            blocks=[
                ExperimentCardBlockSpec(kind="narrative", source="design"),
                ExperimentCardBlockSpec(kind="progress", source="progress"),
                ExperimentCardBlockSpec(kind="runs", source="runs"),
                ExperimentCardBlockSpec(kind="evidence", source="evidence"),
            ],
        ),
    )
    project = _campaign_project(design, 6)
    round_spec = project.experiment_campaign.rounds[0].presentation_spec
    assert round_spec is not None
    design.presentation_spec.blocks[0].title = "changed after build"
    assert round_spec.blocks[0].title is None

    old_payload = _plan().model_dump(mode="json")
    old_payload.pop("designs")
    parsed = ExperimentPlan.model_validate(old_payload)
    assert parsed.designs == []
