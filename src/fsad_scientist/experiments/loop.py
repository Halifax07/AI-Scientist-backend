from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from statistics import fmean, median, pstdev
from typing import Any, Literal

from fsad_scientist.datasets.models import DatasetManifest
from fsad_scientist.domain.enums import HypothesisStatus, RunStatus
from fsad_scientist.domain.models import (
    DatasetAuditRecord,
    ExperimentCampaign,
    ExperimentCell,
    ExperimentConditionEffectSummary,
    ExperimentConditionSpec,
    ExperimentConditionSummary,
    ExperimentDesignSpec,
    ExperimentDistributionSummary,
    ExperimentFactorEffectSummary,
    ExperimentFeedbackProposal,
    ExperimentInteractionSummary,
    ExperimentNodeRecord,
    ExperimentRound,
    ExperimentRun,
    ExperimentSummary,
    ExperimentTrendPoint,
    Hypothesis,
    ResearchProject,
    new_id,
    utc_now,
)
from fsad_scientist.experiments.code_safety import BUILTIN_DETECTORS, BUILTIN_STRATEGIES
from fsad_scientist.experiments.design import (
    compile_design,
    default_presentation_spec,
    normalize_design,
    validate_design,
    validate_plan_designs,
)
from fsad_scientist.science.experiment_tree import ExperimentNode, ExperimentPhase

ExperimentPhaseName = Literal[
    "feasibility",
    "sensitivity",
    "main_study",
    "replication",
    "ablation",
    "cross_dataset",
]

SUPPORTED_PRIMARY_METRICS = frozenset(
    {"image_auroc", "pixel_auroc", "image_ap", "aupro"}
)
_PRIMARY_METRIC_ALIASES = {
    "imageauroc": "image_auroc",
    "pixelauroc": "pixel_auroc",
    "imageap": "image_ap",
    "aupro": "aupro",
}


def normalize_primary_metric(metric: str) -> str:
    """Normalize display-style metric labels to the executor vocabulary."""

    key = "".join(character for character in metric.strip().casefold() if character.isalnum())
    return _PRIMARY_METRIC_ALIASES.get(key, metric.strip())


def is_supported_primary_metric(metric: str) -> bool:
    return normalize_primary_metric(metric) in SUPPORTED_PRIMARY_METRICS


def _mean(observations: list[tuple[float, str]]) -> float | None:
    return fmean(value for value, _ in observations) if observations else None


def _ordered_value(value: Any) -> tuple[int, Any]:
    return (0 if isinstance(value, (int, float)) and not isinstance(value, bool) else 1, value)


class AdaptiveExperimentPlanner:
    """Build and validate a budgeted result-to-next-experiment loop.

    The language model may advise which *registered* cells are most informative,
    but this class owns the admissible search space, pairing, budgets, lifecycle,
    and run construction. It therefore cannot turn an LLM response into an
    arbitrary command or leak test labels into support-set selection.
    """

    supported_detectors = BUILTIN_DETECTORS
    supported_strategies = BUILTIN_STRATEGIES

    def initialize(
        self,
        project: ResearchProject,
        *,
        audit: DatasetAuditRecord,
        dataset: DatasetManifest,
        hypothesis_id: str,
        device: str,
        detector: str = "anomalydino",
        max_rounds: int = 3,
        max_runs: int = 24,
        execution_mode: Literal["sequential", "parallel"] = "sequential",
        parallelism: int | None = None,
        selected_hypothesis_ids: list[str] | None = None,
    ) -> tuple[ExperimentCampaign, list[ExperimentRun]]:
        plan = project.experiment_plan
        if plan is None or not plan.approved:
            raise ValueError("The preregistered experiment plan must be approved first")
        if not audit.verified or audit.digest != dataset.digest:
            raise ValueError("A verified dataset audit matching the manifest is required")
        if detector not in self._approved_detectors(project) or detector not in plan.detectors:
            raise ValueError(
                f"Detector is not approved and executable: {detector}"
                "；自定义检测器需先生成实现并获批准"
            )

        hypothesis = self._select_hypothesis(project, hypothesis_id=hypothesis_id)
        eligible_hypothesis_ids = [
            hypothesis.id,
            *[
                item.id
                for item in self._eligible_hypotheses(project)
                if item.id != hypothesis.id
            ],
        ]
        requested_ids = list(dict.fromkeys(selected_hypothesis_ids or eligible_hypothesis_ids))
        if hypothesis.id not in requested_ids:
            requested_ids.insert(0, hypothesis.id)
        unknown_ids = [item for item in requested_ids if item not in eligible_hypothesis_ids]
        if unknown_ids:
            raise ValueError(
                "Selected innovations are not approved or executable: "
                + ", ".join(unknown_ids)
            )
        # Preserve the user's priority order for parallel campaigns.  The
        # legacy sequential mode keeps the explicitly selected innovation first
        # and then follows the preregistered order.
        hypothesis_ids = requested_ids if execution_mode == "parallel" else [
            hypothesis.id,
            *[item for item in eligible_hypothesis_ids if item != hypothesis.id],
        ]
        contract = hypothesis.analysis_contract
        if contract is None:
            raise ValueError("The selected hypothesis has no analysis contract")
        approved_strategies = self._approved_strategies(project)
        explicit_design = self._explicit_design(plan, hypothesis.id)
        if explicit_design is not None and explicit_design.design_mode != "custom_design":
            explicit_design = None
        if plan.design_generation_status == "needs_correction":
            raise ValueError(
                "Experiment design generation needs correction before execution: "
                f"{plan.design_generation_fallback_reason or 'unknown reason'}"
            )
        if explicit_design is None and contract.design_mode == "custom_design":
            raise ValueError(
                "custom_design requires an explicit validated ExperimentDesignSpec"
            )
        if explicit_design is None and contract.treatment not in approved_strategies:
            raise ValueError(
                f"Unsupported treatment strategy: {contract.treatment}"
                "；自定义策略需先生成实现并获批准"
            )
        if explicit_design is None and contract.control not in approved_strategies:
            raise ValueError(
                f"Unsupported control strategy: {contract.control}"
                "；自定义策略需先生成实现并获批准"
            )

        categories = self._approved_categories(project, dataset)
        if not categories:
            raise ValueError("No approved experiment category exists in the dataset")
        shots = sorted(set(plan.shots) & set(project.spec.constraints.shots))
        seeds = sorted(set(plan.seeds))
        if not shots or not seeds:
            raise ValueError("The experiment plan must contain at least one K and one seed")
        historical_campaign_runs = sum(
            run.round_id is not None and run.plan_id == plan.id for run in project.runs
        )
        remaining_run_budget = project.spec.budget.max_experiments - historical_campaign_runs
        effective_max_runs = min(max_runs, remaining_run_budget)
        validate_plan_designs(
            plan,
            budget=effective_max_runs,
            allowed_categories=set(categories),
            allowed_detectors=self._approved_detectors(project),
            allowed_strategies=approved_strategies,
        )
        design_conditions = self._design_conditions(
            project,
            hypothesis_id=hypothesis.id,
            categories=set(categories),
            budget=effective_max_runs,
        )
        outer_cells_by_hypothesis = {
            item.id: self._outer_cells(
                plan,
                hypothesis_id=item.id,
                categories=categories,
                shots=shots,
                seeds=seeds,
            )
            for item in [
                hypothesis,
                *[
                    candidate
                    for candidate in self._eligible_hypotheses(project)
                    if candidate.id != hypothesis.id
                ],
            ]
        }
        for candidate_id, outer_cells in outer_cells_by_hypothesis.items():
            if len(outer_cells) < 3:
                design = self._explicit_design(plan, candidate_id)
                detail = (
                    f"设计 {design.id} 的因素已占用 category/shots/seed 外层迭代轴；"
                    if design is not None
                    else "legacy 实验需要至少三个 category/K/seed 外层单元；"
                )
                raise ValueError(
                    f"{detail}当前仅有 {len(outer_cells)} 个独立外层单元，"
                    "无法支撑三次内部迭代"
                )
        condition_counts_by_hypothesis = {
            candidate_id: len(
                self._design_conditions(
                    project,
                    hypothesis_id=candidate_id,
                    categories=set(categories),
                    budget=effective_max_runs,
                )
            ) or 2
            for candidate_id in hypothesis_ids
        }
        condition_count = condition_counts_by_hypothesis[hypothesis.id]
        minimum_round_runs = condition_count * 3
        if effective_max_runs < minimum_round_runs:
            raise ValueError(
                "The remaining project budget cannot fund one innovation Round"
            )
        # A complete Round consumes exactly three paired iterations.  Never
        # promise more innovation rounds than the frozen project budget can fund.
        round_capacity = min(
            len(hypothesis_ids),
            max_rounds,
            effective_max_runs // minimum_round_runs,
        )
        hypothesis_ids = hypothesis_ids[: max(1, round_capacity)]
        if not hypothesis_ids:
            raise ValueError("At least one selected innovation is required")
        # Sequential campaigns retain the original one-pair-at-a-time protocol;
        # explicit designs may define the outer iteration cells.
        initial_cells = outer_cells_by_hypothesis[hypothesis.id][:1]
        # Parallel campaigns pre-register all three iterations for every
        # selected innovation, allowing independent rounds to run concurrently
        # without serially enumerating the full factorial space.
        parallel_cells = [
            ExperimentCell(category=category, shots=shot, seed=seed)
            for category in categories
            for shot in shots
            for seed in seeds
        ][:3]
        # A Round is one innovation.  Its first experimental iteration is
        # executed before the single human midpoint-guidance gate.
        exhaustive_run_count = sum(
            len(outer_cells_by_hypothesis[candidate_id])
            * condition_counts_by_hypothesis[candidate_id]
            for candidate_id in hypothesis_ids
        )
        primary_metric = normalize_primary_metric(
            explicit_design.analysis.primary_metric
            if explicit_design is not None
            else contract.metric
        )
        if primary_metric not in SUPPORTED_PRIMARY_METRICS:
            raise ValueError(f"Unsupported primary metric: {primary_metric}")
        campaign = ExperimentCampaign(
            hypothesis_id=hypothesis.id,
            hypothesis_ids=hypothesis_ids,
            dataset_audit_id=audit.id,
            dataset_manifest_path=audit.manifest_path,
            dataset_digest=audit.digest,
            protocol=f"pool_compression_m{project.spec.constraints.candidate_pool_size}",
            candidate_pool_size=project.spec.constraints.candidate_pool_size,
            detector=detector,
            treatment=contract.treatment or "",
            control=contract.control or "",
            metric=primary_metric,
            device=device,
            max_rounds=len(hypothesis_ids),
            max_runs=effective_max_runs,
            exhaustive_run_count=exhaustive_run_count,
            execution_mode=execution_mode,
            parallelism=max(
                1,
                min(
                    parallelism or project.spec.budget.max_parallel_runs,
                    project.spec.budget.max_parallel_runs,
                    32,
                ),
            ),
            selected_hypothesis_ids=hypothesis_ids,
            design_id=explicit_design.id if explicit_design is not None else None,
        )
        if execution_mode == "parallel":
            all_runs: list[ExperimentRun] = []
            for index, selected_id in enumerate(hypothesis_ids, start=1):
                selected_hypothesis = self._hypothesis(project, selected_id)
                selected_contract = selected_hypothesis.analysis_contract
                if selected_contract is None:
                    raise ValueError(f"Selected hypothesis has no analysis contract: {selected_id}")
                experiment_round, nodes, round_runs = self._build_round(
                    project,
                    campaign=campaign,
                    index=index,
                    phase="feasibility",
                    objective=f"验证创新点 H{index}：{selected_hypothesis.title}",
                    rationale=(
                        "并行预注册该创新点的三个独立迭代，比较 "
                        f"{selected_contract.control} 与 {selected_contract.treatment}，"
                        "并通过真实成对结果评估可证伪性。"
                    ),
                    cells=parallel_cells,
                    information_gain=0.90,
                    falsification_value=0.85,
                    parent_id=None,
                    hypothesis_id=selected_id,
                )
                campaign.rounds.append(experiment_round)
                campaign.nodes.extend(nodes)
                all_runs.extend(round_runs)
            # _build_round keeps these fields useful for legacy clients.  Set
            # them back to the first selected innovation for a stable summary.
            campaign.hypothesis_id = hypothesis_ids[0]
            first_contract = self._hypothesis(project, hypothesis_ids[0]).analysis_contract
            if first_contract is not None:
                # custom_design contracts have no arms: 顶层字段必须留空串,
                # 否则持久化后 ResearchProject 重新校验会因 None 失败 (整库 500)。
                campaign.treatment = first_contract.treatment or ""
                campaign.control = first_contract.control or ""
                campaign.metric = normalize_primary_metric(first_contract.metric)
            campaign.next_action = "execute_parallel_batch"
            self._refresh_efficiency(campaign)
            return campaign, all_runs

        first_round, nodes, runs = self._build_round(
            project,
            campaign=campaign,
            index=1,
            phase="feasibility",
            objective=(
                self._design_objective(explicit_design, len(design_conditions))
                if explicit_design is not None
                else "验证真实数据、特征、支持集选择和检测器链路，并获得首批成对效应。"
            ),
            rationale=(
                self._design_rationale(explicit_design, len(design_conditions), detector)
                if explicit_design is not None
                else (
                    f"先在 bottle、K=2 和一个随机种子上比较 {contract.control} 与 "
                    f"{contract.treatment}，并使用 {detector} 执行检测；"
                    "用一组成对真实运行换取端到端可行性和初始效应信息。"
                )
            ),
            cells=initial_cells,
            information_gain=0.90,
            falsification_value=0.80,
            parent_id=None,
            hypothesis_id=hypothesis.id,
        )
        campaign.rounds.append(first_round)
        campaign.nodes.extend(nodes)
        self._refresh_efficiency(campaign)
        return campaign, runs

    def summarize_current_round(self, project: ResearchProject) -> dict[str, Any]:
        campaign = self._campaign(project)
        return self.summarize_round(project, round_id=campaign.rounds[-1].id)

    def summarize_round(
        self, project: ResearchProject, *, round_id: str
    ) -> dict[str, Any]:
        campaign = self._campaign(project)
        current = next(
            (item for item in campaign.rounds if item.id == round_id),
            None,
        )
        if current is None:
            raise ValueError(f"Unknown experiment round: {round_id}")
        runs_by_id = {run.id: run for run in project.runs}
        runs = [runs_by_id[run_id] for run_id in current.run_ids if run_id in runs_by_id]
        explicit_design = self._explicit_design(project.experiment_plan, current.hypothesis_id)
        if explicit_design is not None and explicit_design.design_mode != "custom_design":
            explicit_design = None
        metric = normalize_primary_metric(
            explicit_design.analysis.primary_metric
            if explicit_design is not None
            else current.metric
        )
        if explicit_design is not None:
            current.metric = metric
            return self._summarize_design_round(
                project,
                current=current,
                runs=runs,
                design=explicit_design,
                metric=metric,
            )
        grouped: dict[tuple[str, int, int], dict[str, ExperimentRun]] = defaultdict(dict)
        failed_run_ids: list[str] = []
        duration_seconds = 0.0
        for run in runs:
            duration_seconds += run.duration_seconds or 0.0
            if run.status == RunStatus.FAILED:
                failed_run_ids.append(run.id)
            if run.status == RunStatus.SUCCEEDED and run.verified:
                grouped[(run.category, run.shots, run.seed)][run.selection_strategy] = run

        differences: list[dict[str, Any]] = []
        by_category: dict[str, list[float]] = defaultdict(list)
        paired_metrics: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for (category, shots, seed), strategies in sorted(grouped.items()):
            treatment = strategies.get(current.treatment)
            control = strategies.get(current.control)
            if treatment is None or control is None:
                continue
            common_metrics = sorted(set(treatment.metrics) & set(control.metrics))
            for metric_name in common_metrics:
                paired_metrics[metric_name].append(
                    (treatment.metrics[metric_name], control.metrics[metric_name])
                )
            if metric not in treatment.metrics or metric not in control.metrics:
                continue
            difference = treatment.metrics[metric] - control.metrics[metric]
            by_category[category].append(difference)
            differences.append(
                {
                    "category": category,
                    "shots": shots,
                    "seed": seed,
                    "treatment_run_id": treatment.id,
                    "control_run_id": control.id,
                    "difference": round(difference, 8),
                }
            )

        hypothesis = self._hypothesis(project, current.hypothesis_id)
        minimum_pairs = (
            hypothesis.analysis_contract.minimum_pairs
            if hypothesis.analysis_contract is not None
            else 6
        )
        terminal_count = sum(
            run.status in {RunStatus.SUCCEEDED, RunStatus.FAILED} for run in runs
        )
        campaign_pair_runs: dict[
            tuple[str, int, int], dict[str, ExperimentRun]
        ] = defaultdict(dict)
        for run in project.runs:
            if (
                run.round_id is not None
                and run.hypothesis_id == current.hypothesis_id
                and run.status == RunStatus.SUCCEEDED
                and run.verified
                and metric in run.metrics
            ):
                campaign_pair_runs[(run.category, run.shots, run.seed)][
                    run.selection_strategy
                ] = run
        cumulative_differences = [
            strategies[current.treatment].metrics[metric]
            - strategies[current.control].metrics[metric]
            for strategies in campaign_pair_runs.values()
            if current.treatment in strategies and current.control in strategies
        ]
        cumulative_pair_count = len(cumulative_differences)
        mean_difference = fmean(item["difference"] for item in differences) if differences else None
        paired_metric_summaries = {
            metric_name: {
                "pair_count": len(values),
                "treatment_mean": round(fmean(item[0] for item in values), 8),
                "control_mean": round(fmean(item[1] for item in values), 8),
                "mean_difference": round(fmean(item[0] - item[1] for item in values), 8),
                "positive_pair_fraction": round(
                    sum(item[0] > item[1] for item in values) / len(values), 4
                ),
            }
            for metric_name, values in sorted(paired_metrics.items())
            if values
        }
        primary_values = [value for pair in paired_metrics.get(metric, []) for value in pair]
        return {
            "round_id": current.id,
            "round_index": current.index,
            "hypothesis_id": current.hypothesis_id,
            "completed_iterations": current.completed_iterations,
            "phase": current.phase,
            "metric": metric,
            "planned_runs": len(runs),
            "terminal_runs": terminal_count,
            "successful_verified_runs": sum(
                run.status == RunStatus.SUCCEEDED and run.verified for run in runs
            ),
            "failed_run_ids": failed_run_ids,
            "duration_seconds": round(duration_seconds, 3),
            "round_pair_count": len(differences),
            "pair_count": cumulative_pair_count,
            "cumulative_pair_count": cumulative_pair_count,
            "minimum_pairs": minimum_pairs,
            "pair_differences": differences,
            "mean_difference": round(mean_difference, 8) if mean_difference is not None else None,
            "positive_pair_fraction": (
                round(sum(item["difference"] > 0 for item in differences) / len(differences), 4)
                if differences
                else None
            ),
            "category_mean_differences": {
                category: round(fmean(values), 8)
                for category, values in sorted(by_category.items())
            },
            "paired_metric_summaries": paired_metric_summaries,
            "cumulative_primary_summary": {
                "pair_count": cumulative_pair_count,
                "mean_difference": (
                    round(fmean(cumulative_differences), 8)
                    if cumulative_differences
                    else None
                ),
                "positive_pair_fraction": (
                    round(
                        sum(value > 0 for value in cumulative_differences)
                        / len(cumulative_differences),
                        4,
                    )
                    if cumulative_differences
                    else None
                ),
            },
            "primary_metric_saturated": bool(primary_values)
            and all(value >= 0.995 for value in primary_values),
            "cumulative_terminal_runs": sum(
                run.status in {RunStatus.SUCCEEDED, RunStatus.FAILED}
                for run in project.runs
                if run.round_id is not None
            ),
            "run_budget": campaign.max_runs,
            "exhaustive_run_count": campaign.exhaustive_run_count,
        }

    def _summarize_design_round(
        self,
        project: ResearchProject,
        *,
        current: ExperimentRound,
        runs: list[ExperimentRun],
        design: ExperimentDesignSpec,
        metric: str,
    ) -> dict[str, Any]:
        conditions = self._design_conditions(
            project,
            hypothesis_id=current.hypothesis_id,
            categories=self._campaign_categories(project),
        )
        values_by_condition: dict[str, list[tuple[float, str]]] = defaultdict(list)
        failed_run_ids = [run.id for run in runs if run.status == RunStatus.FAILED]
        duration_seconds = sum(run.duration_seconds or 0.0 for run in runs)
        for run in runs:
            if (
                run.status != RunStatus.SUCCEEDED
                or not run.verified
                or metric not in run.metrics
            ):
                continue
            condition_id = run.condition_id or run.selection_strategy
            values_by_condition[condition_id].append((run.metrics[metric], run.id))

        condition_statistics: list[ExperimentConditionSummary] = []
        for condition in conditions:
            observations = values_by_condition.get(condition.id, [])
            numbers = [value for value, _ in observations]
            condition_statistics.append(
                ExperimentConditionSummary(
                    condition_id=condition.id,
                    label=condition.label,
                    factor_values=condition.factor_values,
                    sample_size=len(numbers),
                    mean=round(fmean(numbers), 8) if numbers else None,
                    standard_deviation=round(pstdev(numbers), 8) if len(numbers) > 1 else 0.0
                    if numbers
                    else None,
                    minimum=round(min(numbers), 8) if numbers else None,
                    maximum=round(max(numbers), 8) if numbers else None,
                    median=round(median(numbers), 8) if numbers else None,
                    source_run_ids=[run_id for _, run_id in observations],
                )
            )

        effects: list[ExperimentConditionEffectSummary] = []
        baseline_id = design.analysis.baseline_condition_id or (
            conditions[0].id if conditions else None
        )
        baseline_mean = _mean(values_by_condition.get(baseline_id or "", []))
        for condition in conditions:
            if condition.id == baseline_id:
                continue
            condition_mean = _mean(values_by_condition.get(condition.id, []))
            if condition_mean is None or baseline_mean is None:
                continue
            effects.append(
                ExperimentConditionEffectSummary(
                    condition_id=condition.id,
                    baseline_condition_id=baseline_id,
                    effect=round(condition_mean - baseline_mean, 8),
                    sample_size=len(values_by_condition.get(condition.id, [])),
                    source_run_ids=[
                        run_id
                        for _, run_id in values_by_condition.get(condition.id, [])
                    ],
                )
            )

        factor_effects: list[ExperimentFactorEffectSummary] = []
        for factor in design.factors:
            by_level: dict[str, list[tuple[float, str]]] = defaultdict(list)
            for condition in conditions:
                level = condition.factor_values.get(factor.name)
                key = str(level)
                by_level[key].extend(values_by_condition.get(condition.id, []))
            level_means = {
                level: round(fmean(value for value, _ in observations), 8)
                for level, observations in sorted(by_level.items())
                if observations
            }
            level_values = list(level_means.values())
            factor_effects.append(
                ExperimentFactorEffectSummary(
                    factor=factor.name,
                    level_means=level_means,
                    effect=round(max(level_values) - min(level_values), 8)
                    if len(level_values) > 1
                    else None,
                    sample_size=sum(len(item) for item in by_level.values()),
                    source_run_ids=[
                        run_id for observations in by_level.values() for _, run_id in observations
                    ],
                )
            )

        interaction_summaries: list[ExperimentInteractionSummary] = []
        for factor_a, factor_b in combinations(design.factors, 2):
            observations_by_cell: dict[tuple[Any, Any], list[tuple[float, str]]] = defaultdict(list)
            for condition in conditions:
                level_a = condition.factor_values.get(factor_a.name)
                level_b = condition.factor_values.get(factor_b.name)
                observations_by_cell[(level_a, level_b)].extend(
                    values_by_condition.get(condition.id, [])
                )
            cell_means: dict[str, float] = {}
            for (level_a, level_b), observations in observations_by_cell.items():
                if observations:
                    cell_means[
                        f"{factor_a.name}={level_a}|{factor_b.name}={level_b}"
                    ] = round(fmean(value for value, _ in observations), 8)

            simple_effects: dict[str, float] = {}
            for level_a in factor_a.levels:
                means = [
                    fmean(value for value, _ in observations_by_cell[(level_a, level_b)])
                    for level_b in factor_b.levels
                    if observations_by_cell[(level_a, level_b)]
                ]
                if len(means) > 1:
                    simple_effects[str(level_a)] = round(max(means) - min(means), 8)

            difference_in_differences: float | None = None
            if len(factor_a.levels) == 2 and len(factor_b.levels) == 2:
                first_a, second_a = factor_a.levels
                first_b, second_b = factor_b.levels
                cells = [
                    observations_by_cell[(first_a, first_b)],
                    observations_by_cell[(first_a, second_b)],
                    observations_by_cell[(second_a, first_b)],
                    observations_by_cell[(second_a, second_b)],
                ]
                if all(cells):
                    means = [fmean(value for value, _ in cell) for cell in cells]
                    difference_in_differences = round(
                        (means[3] - means[2]) - (means[1] - means[0]), 8
                    )

            interaction_observations = [
                item
                for observations in observations_by_cell.values()
                for item in observations
            ]
            interaction_summaries.append(
                ExperimentInteractionSummary(
                    factor_a=factor_a.name,
                    factor_b=factor_b.name,
                    levels={
                        factor_a.name: list(factor_a.levels),
                        factor_b.name: list(factor_b.levels),
                    },
                    cell_means=cell_means,
                    simple_effects=simple_effects,
                    difference_in_differences=difference_in_differences,
                    sample_size=len(interaction_observations),
                    source_run_ids=[
                        run_id for _, run_id in interaction_observations
                    ],
                )
            )

        trend: list[ExperimentTrendPoint] = []
        if design.analysis.mode == "ordered_trend":
            ordered_factor = design.analysis.ordered_factor or design.factors[0].name
            factor = next(item for item in design.factors if item.name == ordered_factor)
            for level in sorted(factor.levels, key=_ordered_value):
                observations = [
                    item
                    for condition in conditions
                    if condition.factor_values.get(ordered_factor) == level
                    for item in values_by_condition.get(condition.id, [])
                ]
                trend.append(
                    ExperimentTrendPoint(
                        level=level,
                        mean=round(fmean(value for value, _ in observations), 8)
                        if observations
                        else None,
                        sample_size=len(observations),
                        source_run_ids=[run_id for _, run_id in observations],
                    )
                )

        all_observations = [
            item for observations in values_by_condition.values() for item in observations
        ]
        all_values = [value for value, _ in all_observations]
        distribution = ExperimentDistributionSummary(
            sample_size=len(all_values),
            mean=round(fmean(all_values), 8) if all_values else None,
            standard_deviation=round(pstdev(all_values), 8) if len(all_values) > 1 else 0.0
            if all_values
            else None,
            minimum=round(min(all_values), 8) if all_values else None,
            maximum=round(max(all_values), 8) if all_values else None,
            median=round(median(all_values), 8) if all_values else None,
        )
        minimum_pairs = design.analysis.minimum_pairs
        minimum_group_size = min(
            (item.sample_size for item in condition_statistics), default=0
        )
        evidence_status = (
            "sample_threshold_met"
            if minimum_group_size >= minimum_pairs
            else "below_threshold"
            if all_values
            else "not_ready"
        )
        summary = ExperimentSummary(
            analysis_mode=design.analysis.mode,
            primary_metric=metric,
            sample_size=len(all_values),
            evidence_status=evidence_status,
            source_run_ids=[run_id for _, run_id in all_observations],
            condition_statistics=condition_statistics,
            condition_effects=effects,
            factor_effects=factor_effects if design.analysis.mode == "factor_effects" else [],
            interaction_summary=(
                interaction_summaries if design.analysis.mode == "factor_effects" else []
            ),
            ordered_trend=trend,
            distribution_summary=distribution,
        )
        current.summary = summary
        self._campaign(project).summary = summary

        pair_payload: dict[str, Any] = {}
        if design.design_mode == "paired_comparison":
            pair_differences: list[dict[str, Any]] = []
            grouped: dict[
                tuple[str, str, str, int, int], dict[str, ExperimentRun]
            ] = defaultdict(dict)
            for run in runs:
                if (
                    run.status == RunStatus.SUCCEEDED
                    and run.verified
                    and metric in run.metrics
                ):
                    key = (run.protocol, run.dataset, run.category, run.shots, run.seed)
                    grouped[key][run.condition_id or run.selection_strategy] = run
            first, second = conditions
            for cell, pair in sorted(grouped.items()):
                left, right = pair.get(first.id), pair.get(second.id)
                if left is None or right is None:
                    continue
                pair_differences.append(
                    {
                        "category": cell[2],
                        "shots": cell[3],
                        "seed": cell[4],
                        "treatment_run_id": right.id,
                        "control_run_id": left.id,
                        "difference": round(right.metrics[metric] - left.metrics[metric], 8),
                    }
                )
            pair_payload = {
                "pair_count": len(pair_differences),
                "round_pair_count": len(pair_differences),
                "pair_differences": pair_differences,
                "mean_difference": (
                    round(fmean(item["difference"] for item in pair_differences), 8)
                    if pair_differences
                    else None
                ),
            }
        return {
            "round_id": current.id,
            "round_index": current.index,
            "hypothesis_id": current.hypothesis_id,
            "completed_iterations": current.completed_iterations,
            "phase": current.phase,
            "metric": metric,
            "analysis_mode": design.analysis.mode,
            "design_id": design.id,
            "planned_runs": len(runs),
            "terminal_runs": sum(
                run.status in {RunStatus.SUCCEEDED, RunStatus.FAILED} for run in runs
            ),
            "successful_verified_runs": len(all_values),
            "failed_run_ids": failed_run_ids,
            "duration_seconds": round(duration_seconds, 3),
            "minimum_pairs": minimum_pairs,
            "condition_statistics": [item.model_dump(mode="json") for item in condition_statistics],
            "condition_effects": [item.model_dump(mode="json") for item in effects],
            "factor_effects": [item.model_dump(mode="json") for item in factor_effects],
            "interaction_summary": [
                item.model_dump(mode="json") for item in interaction_summaries
            ],
            "ordered_trend": [item.model_dump(mode="json") for item in trend],
            "distribution_summary": distribution.model_dump(mode="json"),
            "sample_size": len(all_values),
            "evidence_status": evidence_status,
            "inference_status": "not_performed",
            "source_run_ids": [run_id for _, run_id in all_observations],
            "summary": summary.model_dump(mode="json"),
            "primary_metric_saturated": bool(all_values)
            and all(value >= 0.995 for value in all_values),
            "run_budget": self._campaign(project).max_runs,
            "exhaustive_run_count": self._campaign(project).exhaustive_run_count,
            **pair_payload,
        }

    def allowed_next_cells(
        self,
        project: ResearchProject,
        *,
        hypothesis_id: str | None = None,
    ) -> list[ExperimentCell]:
        campaign = self._campaign(project)
        current = campaign.rounds[-1]
        target_hypothesis_id = hypothesis_id or current.hypothesis_id
        plan = project.experiment_plan
        if plan is None:
            return []
        audit = next(
            (item for item in project.dataset_audits if item.id == campaign.dataset_audit_id),
            None,
        )
        if audit is None:
            return []
        categories = [item for item in plan.categories if item in audit.categories]
        shots = sorted(set(plan.shots) & set(project.spec.constraints.shots))
        seeds = sorted(set(plan.seeds))
        candidates = self._outer_cells(
            plan,
            hypothesis_id=target_hypothesis_id,
            categories=categories,
            shots=shots,
            seeds=seeds,
        )
        target_round_ids = {
            item.id
            for item in campaign.rounds
            if item.hypothesis_id == target_hypothesis_id
        }
        target_node_ids = {
            item.id for item in campaign.nodes if item.round_id in target_round_ids
        }
        used = {
            (node.config.get("category"), node.config.get("shots"), node.config.get("seed"))
            for node in campaign.nodes
            if node.id in target_node_ids
        }
        candidates = [
            cell
            for cell in candidates
            if (cell.category, cell.shots, cell.seed) not in used
        ]
        # Prefer replication breadth, then K sensitivity, before accumulating seeds.
        current_categories = {
            node.config.get("category")
            for node in campaign.nodes
            if node.id in target_node_ids
        }
        current_shots = {
            node.config.get("shots")
            for node in campaign.nodes
            if node.id in target_node_ids
        }
        return sorted(
            candidates,
            key=lambda cell: (
                cell.category in current_categories,
                cell.shots in current_shots,
                cell.seed,
                cell.category,
                cell.shots,
            ),
        )

    def remaining_round_cells(
        self,
        project: ResearchProject,
        *,
        round_id: str,
    ) -> list[ExperimentCell]:
        """Return queued, pre-registered cells for iterations 2–3 of a Round.

        Parallel campaigns register all three iterations before execution.  A
        midpoint guidance decision may reorder those frozen cells, but it must
        not create new cells or silently expand the search space.
        """

        campaign = self._campaign(project)
        current = next(
            (item for item in campaign.rounds if item.id == round_id),
            None,
        )
        if current is None:
            raise ValueError(f"Unknown experiment round: {round_id}")
        runs_by_id = {run.id: run for run in project.runs}
        nodes_by_id = {node.id: node for node in campaign.nodes}
        cells: list[ExperimentCell] = []
        for node_id in current.node_ids:
            node = nodes_by_id.get(node_id)
            node_runs = [
                runs_by_id[run_id]
                for run_id in (node.run_ids if node else [])
                if run_id in runs_by_id
            ]
            if node is None or node.iteration < 2 or not node_runs:
                continue
            if any(run.status != RunStatus.QUEUED for run in node_runs):
                continue
            reference = node_runs[0]
            cells.append(
                ExperimentCell(
                    category=reference.category,
                    shots=reference.shots,
                    seed=reference.seed,
                )
            )
        return cells

    def apply_midpoint_guidance(
        self,
        project: ResearchProject,
        *,
        proposal: ExperimentFeedbackProposal,
        summary: dict[str, Any],
        round_id: str | None = None,
    ) -> list[ExperimentRun]:
        """Schedule iterations 2 and 3 inside the same innovation Round."""

        campaign = self._campaign(project)
        current = next(
            (item for item in campaign.rounds if item.id == round_id),
            campaign.rounds[-1],
        )
        if campaign.status != "awaiting_guidance" or current.status != "awaiting_guidance":
            raise ValueError("The current round is not waiting for midpoint guidance")

        if campaign.execution_mode == "parallel":
            # All three iterations already exist in the frozen parallel queue.
            # Guidance can choose their order, but never adds a new run or
            # changes the registered treatment/control/metric contract.
            runs_by_id = {run.id: run for run in project.runs}
            nodes_by_id = {node.id: node for node in campaign.nodes}
            remaining_nodes = []
            for node_id in current.node_ids:
                node = nodes_by_id.get(node_id)
                node_runs = [
                    runs_by_id[run_id]
                    for run_id in (node.run_ids if node else [])
                    if run_id in runs_by_id
                ]
                if node is None or node.iteration < 2 or not node_runs:
                    continue
                if any(run.status != RunStatus.QUEUED for run in node_runs):
                    continue
                remaining_nodes.append((node, node_runs))
            if len(remaining_nodes) != 2:
                raise ValueError(
                    "并行 Round 的预注册队列必须保留第 2、3 次迭代"
                )
            node_by_cell = {
                (
                    node_runs[0].category,
                    node_runs[0].shots,
                    node_runs[0].seed,
                ): (node, node_runs)
                for node, node_runs in remaining_nodes
            }
            ordered: list[tuple[ExperimentNodeRecord, list[ExperimentRun]]] = []
            used_keys: set[tuple[str, int, int]] = set()
            for cell in proposal.recommended_cells:
                key = (cell.category, cell.shots, cell.seed)
                item = node_by_cell.get(key)
                if item is not None and key not in used_keys:
                    ordered.append(item)
                    used_keys.add(key)
            for item in remaining_nodes:
                node, node_runs = item
                key = (node_runs[0].category, node_runs[0].shots, node_runs[0].seed)
                if key not in used_keys:
                    ordered.append(item)
                    used_keys.add(key)
            for iteration, (node, node_runs) in enumerate(ordered, start=2):
                node.iteration = iteration
                for run in node_runs:
                    run.iteration = iteration
            current.node_ids = [
                node_id
                for node, _ in ordered
                for node_id in [node.id]
            ] + [
                node_id
                for node_id in current.node_ids
                if node_id not in {node.id for node, _ in ordered}
            ]
            current.run_ids = [
                run.id
                for node, node_runs in ordered
                for run in node_runs
            ] + [
                run_id
                for run_id in current.run_ids
                if run_id not in {
                    run.id for _, node_runs in ordered for run in node_runs
                }
            ]
            current.completed_iterations = 1
            current.guidance_received = True
            current.feedback = proposal
            current.result_summary = summary
            current.status = "planned"
            current.efficiency["planned_runs"] = len(current.run_ids)
            # This Round now has queued work (iterations 2–3).  Mark the
            # campaign active so the caller can immediately dispatch only its
            # guided queue; other Rounds remain individually paused at their
            # own guidance gates and are not selected by ``queued_runs``.
            campaign.status = "active"
            campaign.next_action = "execute_parallel_batch"
            campaign.current_round = current.index
            self._refresh_efficiency(campaign, project=project)
            return [run for _, node_runs in ordered for run in node_runs]

        allowed = self.allowed_next_cells(project)
        allowed_by_key = {
            (item.category, item.shots, item.seed): item for item in allowed
        }
        remaining_iterations = max(current.iteration_target - len(current.node_ids), 0)
        condition_count = self._condition_count(project, current.hypothesis_id)
        remaining_runs = max(campaign.max_runs - self._campaign_run_count(project), 0)
        affordable_iterations = remaining_runs // condition_count
        iterations_to_schedule = min(remaining_iterations, affordable_iterations)
        if iterations_to_schedule != remaining_iterations:
            raise ValueError("The remaining run budget cannot fund the rest of this round")
        selected: list[ExperimentCell] = []
        for item in proposal.recommended_cells:
            key = (item.category, item.shots, item.seed)
            if key in allowed_by_key and allowed_by_key[key] not in selected:
                selected.append(allowed_by_key[key])
            if len(selected) >= iterations_to_schedule:
                break
        for item in allowed:
            if len(selected) >= iterations_to_schedule:
                break
            if item not in selected:
                selected.append(item)
        if len(selected) != iterations_to_schedule:
            raise ValueError("The preregistered search space cannot fund three iterations")

        _, nodes, runs = self._build_round(
            project,
            campaign=campaign,
            index=current.index,
            phase=current.phase,
            objective=current.objective,
            rationale=current.rationale,
            cells=selected,
            information_gain=proposal.expected_information_gain,
            falsification_value=0.85,
            parent_id=current.node_ids[-1] if current.node_ids else None,
            hypothesis_id=current.hypothesis_id,
            iteration_start=2,
        )
        for node in nodes:
            node.round_id = current.id
        for run in runs:
            run.round_id = current.id
        current.node_ids.extend(node.id for node in nodes)
        current.run_ids.extend(run.id for run in runs)
        current.guidance_received = True
        current.feedback = proposal
        current.result_summary = summary
        current.status = "planned"
        current.efficiency["planned_runs"] = len(current.run_ids)
        campaign.nodes.extend(nodes)
        campaign.status = "active"
        campaign.next_action = "execute_remaining_round_iterations"
        self._refresh_efficiency(campaign, project=project, additional_runs=len(runs))
        return runs

    def apply_feedback(
        self,
        project: ResearchProject,
        *,
        proposal: ExperimentFeedbackProposal,
        summary: dict[str, Any],
    ) -> list[ExperimentRun]:
        campaign = self._campaign(project)
        current = campaign.rounds[-1]
        if current.status != "ready_for_feedback":
            raise ValueError("The current round is not ready for feedback")
        current.feedback = proposal
        current.result_summary = summary
        current.status = "completed"
        current.completed_at = utc_now()
        self._update_nodes_for_round(campaign, current, project, summary)

        next_index = current.index + 1
        if next_index > len(campaign.hypothesis_ids):
            campaign.status = "completed"
            campaign.next_action = "analyze_verified_results"
            campaign.termination_reason = "all_innovation_rounds_completed"
            campaign.completed_at = utc_now()
            self._refresh_efficiency(campaign, project=project)
            return []

        next_hypothesis_id = campaign.hypothesis_ids[next_index - 1]
        next_condition_count = self._condition_count(project, next_hypothesis_id)
        remaining_runs = campaign.max_runs - self._campaign_run_count(project)
        if remaining_runs < next_condition_count * 3:
            campaign.status = "completed"
            campaign.next_action = "analyze_verified_results"
            campaign.termination_reason = "run_budget_exhausted_before_next_innovation"
            campaign.completed_at = utc_now()
            self._refresh_efficiency(campaign, project=project)
            return []
        next_hypothesis = self._hypothesis(project, next_hypothesis_id)
        next_contract = next_hypothesis.analysis_contract
        if next_contract is None:
            raise ValueError("The next innovation has no analysis contract")
        next_design = self._explicit_design(project.experiment_plan, next_hypothesis_id)
        allowed = self.allowed_next_cells(project, hypothesis_id=next_hypothesis_id)
        if not allowed:
            raise ValueError("No experiment cell is available for the next innovation")
        parent_id = current.node_ids[0] if current.node_ids else None
        next_round, nodes, runs = self._build_round(
            project,
            campaign=campaign,
            index=next_index,
            phase="feasibility",
            objective=(
                self._design_objective(next_design, next_condition_count)
                if next_design is not None
                else f"验证创新点 H{next_index}：{next_hypothesis.title}"
            ),
            rationale=(
                self._design_rationale(
                    next_design,
                    next_condition_count,
                    campaign.detector,
                )
                if next_design is not None
                else (
                    "上一创新点已完成三次自动迭代。现在切换到下一创新点，并先执行"
                    f"第 1 次迭代；比较 {next_contract.treatment} 与 {next_contract.control}。"
                )
            ),
            cells=[allowed[0]],
            information_gain=0.9,
            falsification_value=0.85,
            parent_id=parent_id,
            hypothesis_id=next_hypothesis_id,
        )
        campaign.current_round = next_index
        campaign.rounds.append(next_round)
        campaign.nodes.extend(nodes)
        campaign.status = "active"
        campaign.next_action = "execute_next_experiment"
        self._refresh_efficiency(campaign, project=project, additional_runs=len(runs))
        return runs

    def refresh_after_run(self, project: ResearchProject) -> None:
        campaign = project.experiment_campaign
        if campaign is None or campaign.status == "completed":
            return

        if campaign.execution_mode == "parallel":
            # Parallel campaigns have one pre-registered Round per selected
            # innovation.  The first iteration of every Round runs in parallel;
            # after those results arrive each Round pauses once for human
            # guidance before its queued iterations 2–3 are dispatched.
            by_id = {run.id: run for run in project.runs}

            def completed_iteration_count(round_runs: list[ExperimentRun]) -> int:
                completed = 0
                for iteration in (1, 2, 3):
                    iteration_runs = [
                        run for run in round_runs if run.iteration == iteration
                    ]
                    if not iteration_runs or not all(
                        run.status in {RunStatus.SUCCEEDED, RunStatus.FAILED}
                        for run in iteration_runs
                    ):
                        break
                    completed = iteration
                return completed

            for experiment_round in campaign.rounds:
                round_runs = [
                    by_id[run_id]
                    for run_id in experiment_round.run_ids
                    if run_id in by_id
                ]
                for node in campaign.nodes:
                    if node.id not in experiment_round.node_ids:
                        continue
                    node_runs = [
                        by_id[run_id] for run_id in node.run_ids if run_id in by_id
                    ]
                    if any(run.status == RunStatus.RUNNING for run in node_runs):
                        node.status = "running"
                    elif node_runs and all(
                        run.status in {RunStatus.SUCCEEDED, RunStatus.FAILED}
                        for run in node_runs
                    ):
                        node.status = (
                            "succeeded"
                            if all(
                                run.status == RunStatus.SUCCEEDED and run.verified
                                for run in node_runs
                            )
                            else "failed"
                        )
                    else:
                        node.status = "pending"

                if not round_runs:
                    continue
                first_iteration_runs = [
                    run for run in round_runs if run.iteration == 1
                ]
                first_iteration_terminal = bool(first_iteration_runs) and all(
                    run.status in {RunStatus.SUCCEEDED, RunStatus.FAILED}
                    for run in first_iteration_runs
                )
                terminal = all(
                    run.status in {RunStatus.SUCCEEDED, RunStatus.FAILED}
                    for run in round_runs
                )
                running = any(run.status == RunStatus.RUNNING for run in round_runs)
                experiment_round.completed_iterations = completed_iteration_count(
                    round_runs
                )
                if not experiment_round.guidance_received and first_iteration_terminal:
                    # Persist the first-iteration evidence immediately so the
                    # UI can explain what was observed while waiting for the
                    # user's one midpoint instruction for this Round.
                    experiment_round.result_summary = self.summarize_round(
                        project,
                        round_id=experiment_round.id,
                    )
                    experiment_round.status = "awaiting_guidance"
                elif terminal and experiment_round.guidance_received:
                    experiment_round.completed_at = experiment_round.completed_at or utc_now()
                    # Persist the deterministic metric summary as soon as a
                    # Round becomes terminal.  The streaming UI can therefore
                    # show paired effects immediately, even while other
                    # innovation Rounds are still running and before the
                    # advisor performs the final innovation review.
                    experiment_round.result_summary = self.summarize_round(
                        project,
                        round_id=experiment_round.id,
                    )
                    if experiment_round.status != "completed":
                        experiment_round.status = "ready_for_feedback"
                elif running:
                    experiment_round.status = "running"
                    experiment_round.started_at = experiment_round.started_at or utc_now()
                else:
                    experiment_round.status = "planned"

            pending_guidance = [
                item for item in campaign.rounds
                if item.status == "awaiting_guidance"
            ]
            unfinished_rounds = [
                item
                for item in campaign.rounds
                if item.status in {"planned", "running"}
            ]
            ready_rounds = [
                item for item in campaign.rounds
                if item.status == "ready_for_feedback"
            ]
            if pending_guidance:
                campaign.status = "awaiting_guidance"
                campaign.next_action = "collect_midpoint_guidance"
                campaign.current_round = min(item.index for item in pending_guidance)
            elif unfinished_rounds:
                campaign.status = "active"
                campaign.next_action = "execute_parallel_batch"
                campaign.current_round = min(item.index for item in unfinished_rounds)
            elif ready_rounds:
                campaign.status = "awaiting_feedback"
                campaign.next_action = "review_parallel_rounds"
                campaign.current_round = max(item.index for item in ready_rounds)
            else:
                campaign.status = "completed"
                campaign.next_action = "analyze_verified_results"
                campaign.current_round = max(item.index for item in campaign.rounds)
            self._refresh_efficiency(campaign, project=project)
            return

        current = campaign.rounds[-1]
        runs = [run for run in project.runs if run.id in current.run_ids]
        for node in campaign.nodes:
            if node.id not in current.node_ids:
                continue
            node_runs = [run for run in runs if run.node_id == node.id]
            if any(run.status == RunStatus.RUNNING for run in node_runs):
                node.status = "running"
            elif node_runs and all(
                run.status in {RunStatus.SUCCEEDED, RunStatus.FAILED} for run in node_runs
            ):
                node.status = (
                    "succeeded"
                    if all(
                        run.status == RunStatus.SUCCEEDED and run.verified
                        for run in node_runs
                    )
                    else "failed"
                )
            else:
                node.status = "pending"

        if runs and all(
            run.status in {RunStatus.SUCCEEDED, RunStatus.FAILED} for run in runs
        ):
            current.completed_iterations = len(current.node_ids)
            current.completed_at = utc_now()
            if current.completed_iterations == 1 and not current.guidance_received:
                current.status = "awaiting_guidance"
                campaign.status = "awaiting_guidance"
                campaign.next_action = "collect_midpoint_guidance"
            else:
                current.status = "ready_for_feedback"
                campaign.status = "awaiting_feedback"
                campaign.next_action = "analyze_round_and_advance_innovation"
        elif any(run.status == RunStatus.RUNNING for run in runs):
            current.status = "running"
            current.started_at = current.started_at or utc_now()
        self._refresh_efficiency(campaign, project=project)

    def fill_current_round(
        self,
        project: ResearchProject,
        *,
        target_cells: int = 2,
    ) -> list[ExperimentRun]:
        """Fill a not-yet-started round when an advisor proposed invalid cells."""

        if target_cells < 1:
            raise ValueError("target_cells must be positive")
        campaign = self._campaign(project)
        current = campaign.rounds[-1]
        if campaign.status != "active" or current.status != "planned":
            raise ValueError("Only an active, planned round can be filled")
        current_runs = [run for run in project.runs if run.id in current.run_ids]
        if any(run.status != RunStatus.QUEUED for run in current_runs):
            raise ValueError("The current round has already started")

        existing_cells = len(current.node_ids)
        condition_count = self._condition_count(project, current.hypothesis_id)
        remaining_cells = max(
            (campaign.max_runs - self._campaign_run_count(project)) // condition_count,
            0,
        )
        needed = min(max(target_cells - existing_cells, 0), remaining_cells)
        cells = self.allowed_next_cells(project)[:needed]
        if not cells:
            return []

        previous = campaign.rounds[-2] if len(campaign.rounds) > 1 else None
        parent_id = previous.node_ids[0] if previous and previous.node_ids else None
        _, nodes, runs = self._build_round(
            project,
            campaign=campaign,
            index=current.index,
            phase=current.phase,
            objective=current.objective,
            rationale=current.rationale,
            cells=cells,
            information_gain=0.70,
            falsification_value=0.80,
            parent_id=parent_id,
        )
        for node in nodes:
            node.round_id = current.id
        for run in runs:
            run.round_id = current.id
        current.node_ids.extend(node.id for node in nodes)
        current.run_ids.extend(run.id for run in runs)
        current.efficiency["planned_runs"] = len(current.run_ids)
        current.rationale += (
            " 系统动作校验器剔除重复或越界建议后，从允许空间补齐了本轮单元。"
        )
        campaign.nodes.extend(nodes)
        self._refresh_efficiency(campaign, project=project, additional_runs=len(runs))
        return runs

    @staticmethod
    def queued_runs(project: ResearchProject) -> list[ExperimentRun]:
        campaign = project.experiment_campaign
        if campaign is None or campaign.status != "active":
            return []
        by_id = {run.id: run for run in project.runs}
        rounds = campaign.rounds if campaign.execution_mode == "parallel" else campaign.rounds[-1:]
        round_index = {
            experiment_round.id: experiment_round.index
            for experiment_round in campaign.rounds
        }
        candidates = [
            by_id[run_id]
            for experiment_round in rounds
            for run_id in experiment_round.run_ids
            if run_id in by_id and by_id[run_id].status == RunStatus.QUEUED
        ]
        node_priority = {node.id: node.priority for node in campaign.nodes}
        return sorted(
            candidates,
            key=lambda run: (
                round_index.get(run.round_id or "", 0),
                -node_priority.get(run.node_id or "", 0.0),
                run.id,
            ),
        )

    @classmethod
    def next_queued_run(cls, project: ResearchProject) -> ExperimentRun | None:
        return next(iter(cls.queued_runs(project)), None)

    def _build_round(
        self,
        project: ResearchProject,
        *,
        campaign: ExperimentCampaign,
        index: int,
        phase: ExperimentPhaseName,
        objective: str,
        rationale: str,
        cells: list[ExperimentCell],
        information_gain: float,
        falsification_value: float,
        parent_id: str | None,
        hypothesis_id: str | None = None,
        iteration_start: int = 1,
    ) -> tuple[ExperimentRound, list[ExperimentNodeRecord], list[ExperimentRun]]:
        plan = project.experiment_plan
        if plan is None:
            raise ValueError("Experiment plan is missing")
        round_hypothesis_id = hypothesis_id or campaign.hypothesis_id
        hypothesis = self._hypothesis(project, round_hypothesis_id)
        contract = hypothesis.analysis_contract
        if contract is None or not self._contract_is_executable(
            project, contract, hypothesis_id=round_hypothesis_id
        ):
            raise ValueError(f"Hypothesis is not executable: {round_hypothesis_id}")
        design = self._explicit_design(plan, round_hypothesis_id)
        if design is not None and design.design_mode != "custom_design":
            design = None
        primary_metric = normalize_primary_metric(
            design.analysis.primary_metric if design is not None else contract.metric
        )
        if primary_metric not in SUPPORTED_PRIMARY_METRICS:
            raise ValueError(f"Unsupported primary metric: {primary_metric}")
        campaign.hypothesis_id = round_hypothesis_id
        campaign.treatment = contract.treatment or ""
        campaign.control = contract.control or ""
        campaign.metric = primary_metric
        runtime_design = design
        presentation_spec = (
            runtime_design.presentation_spec or default_presentation_spec(runtime_design)
        ).model_copy(deep=True) if runtime_design is not None else None
        condition_specs = (
            self._design_conditions(
                project,
                hypothesis_id=round_hypothesis_id,
                categories=self._campaign_categories(project),
            )
            if runtime_design is not None
            else []
        )
        if runtime_design is not None and not condition_specs:
            condition_specs = compile_design(runtime_design)
        if design is not None:
            campaign.design_id = design.id
        round_id = new_id("round")
        nodes: list[ExperimentNodeRecord] = []
        runs: list[ExperimentRun] = []
        specs = condition_specs or [
            ExperimentConditionSpec(
                id="control", factor_values={"selection_strategy": campaign.control}
            ),
            ExperimentConditionSpec(
                id="treatment", factor_values={"selection_strategy": campaign.treatment}
            ),
        ]
        cost = float(len(specs))
        for cell_offset, cell in enumerate(cells):
            iteration = min(iteration_start + cell_offset, 3)
            node_id = new_id("experiment_node")
            priority_node = ExperimentNode(
                id=node_id,
                hypothesis_id=round_hypothesis_id,
                phase=ExperimentPhase(phase),
                parent_id=parent_id,
                information_gain=max(min(information_gain, 1.0), 0.0),
                falsification_value=max(min(falsification_value, 1.0), 0.0),
                estimated_cost=cost,
                novelty=0.2 if phase == "feasibility" else 0.5,
            )
            node_runs: list[ExperimentRun] = []
            for condition in specs:
                values = dict(condition.factor_values)
                factor_fields = {
                    factor.name: factor.field or factor.run_field
                    for factor in runtime_design.factors
                } if runtime_design is not None else {}
                bound_values = {
                    factor_fields.get(name, name): value for name, value in values.items()
                }
                default_selection_strategy = (
                    (
                        runtime_design.support_selection_strategy
                        or runtime_design.default_selection_strategy
                        if runtime_design is not None
                        else values.get("selection_strategy")
                    )
                    or campaign.treatment
                    or "random"
                )
                run_values = {
                    "protocol": campaign.protocol,
                    "dataset": "MVTec AD",
                    "category": cell.category,
                    "detector": campaign.detector,
                    "selection_strategy": default_selection_strategy,
                    "shots": cell.shots,
                    "seed": cell.seed,
                }
                run_values.update(
                    {field: value for field, value in bound_values.items() if field != "dataset"}
                )
                node_runs.append(
                    ExperimentRun(
                        plan_id=plan.id,
                        hypothesis_id=round_hypothesis_id,
                        protocol=run_values["protocol"],
                        dataset=run_values["dataset"],
                        category=run_values["category"],
                        detector=run_values["detector"],
                        selection_strategy=run_values["selection_strategy"],
                        shots=run_values["shots"],
                        seed=run_values["seed"],
                        iteration=iteration,
                        round_id=round_id,
                        node_id=node_id,
                        condition_id=condition.id,
                        factor_values=values,
                        phase=phase,
                        status=RunStatus.QUEUED,
                    )
                )
            nodes.append(
                ExperimentNodeRecord(
                    id=node_id,
                    round_id=round_id,
                    iteration=iteration,
                    parent_id=parent_id,
                    phase=phase,
                    objective=(
                        self._design_node_objective(
                            runtime_design,
                            cell=cell,
                            condition_count=len(specs),
                        )
                        if design is not None
                        else (
                            f"{cell.category} / K={cell.shots} / seed={cell.seed}："
                            f"成对比较 {campaign.treatment} 与 {campaign.control}。"
                        )
                    ),
                    information_gain=priority_node.information_gain,
                    falsification_value=priority_node.falsification_value,
                    estimated_cost=cost,
                    novelty=priority_node.novelty,
                    priority=round(priority_node.priority, 6),
                    config=cell.model_dump(mode="json"),
                    run_ids=[run.id for run in node_runs],
                )
            )
            runs.extend(node_runs)
        experiment_round = ExperimentRound(
            id=round_id,
            index=index,
            phase=phase,
            objective=objective,
            rationale=rationale,
            hypothesis_id=round_hypothesis_id,
            design_id=design.id if design is not None else None,
            presentation_spec=presentation_spec,
            treatment=contract.treatment or "",
            control=contract.control or "",
            metric=primary_metric,
            node_ids=[node.id for node in nodes],
            run_ids=[run.id for run in runs],
            efficiency={"planned_runs": len(runs)},
        )
        return experiment_round, nodes, runs

    @staticmethod
    def _explicit_design(
        plan: Any, hypothesis_id: str
    ) -> ExperimentDesignSpec | None:
        if plan is None:
            return None
        exact = next(
            (design for design in plan.designs if design.hypothesis_id == hypothesis_id),
            None,
        )
        return exact or next(
            (design for design in plan.designs if design.hypothesis_id is None),
            None,
        )

    @staticmethod
    def _design_objective(
        design: ExperimentDesignSpec | None, condition_count: int
    ) -> str:
        if design is None:
            return ""
        factors = "、".join(factor.name for factor in design.factors)
        return (
            f"{design.question or '验证通用实验设计'}：因素为 {factors}，"
            f"展开 {condition_count} 个条件，"
            f"使用 {design.analysis.mode} 分析，主指标为 {design.analysis.primary_metric}。"
        )

    @staticmethod
    def _design_rationale(
        design: ExperimentDesignSpec | None,
        condition_count: int,
        detector: str,
    ) -> str:
        if design is None:
            return ""
        factors = "、".join(
            f"{factor.name}={factor.levels}" for factor in design.factors
        )
        return (
            f"{design.rationale or f'使用 {detector} 执行已注册设计'}，"
            f"按 {factors} 展开 {condition_count} 个条件；"
            f"结果采用 {design.analysis.mode} 汇总，"
            f"以 {design.analysis.primary_metric} 作为主指标。"
        )

    @staticmethod
    def _design_node_objective(
        design: ExperimentDesignSpec,
        *,
        cell: ExperimentCell,
        condition_count: int,
    ) -> str:
        factors = "、".join(factor.name for factor in design.factors)
        return (
            f"基础单元 {cell.category} / K={cell.shots} / seed={cell.seed}："
            f"按因素 {factors} 展开 {condition_count} 个条件，分析模式 {design.analysis.mode}。"
        )

    def _condition_count(self, project: ResearchProject, hypothesis_id: str) -> int:
        design = self._explicit_design(project.experiment_plan, hypothesis_id)
        if design is None:
            return 2
        return len(
            self._design_conditions(
                project,
                hypothesis_id=hypothesis_id,
                categories=self._campaign_categories(project),
            )
        )

    @staticmethod
    def _outer_cells(
        plan,
        *,
        hypothesis_id: str,
        categories: list[str],
        shots: list[int],
        seeds: list[int],
    ) -> list[ExperimentCell]:
        design = AdaptiveExperimentPlanner._explicit_design(plan, hypothesis_id)
        bound_fields = {
            factor.field or factor.run_field
            for factor in design.factors
        } if design is not None else set()
        axes = {
            "category": ["__design_category__"] if "category" in bound_fields else categories,
            "shots": [1] if "shots" in bound_fields else shots,
            "seed": [0] if "seed" in bound_fields else seeds,
        }
        cells: list[ExperimentCell] = []
        seen: set[tuple[str, int, int]] = set()
        for category in axes["category"]:
            for shot in axes["shots"]:
                for seed in axes["seed"]:
                    key = (category, shot, seed)
                    if key not in seen:
                        seen.add(key)
                        cells.append(ExperimentCell(category=category, shots=shot, seed=seed))
        return cells

    def _design_conditions(
        self,
        project: ResearchProject,
        *,
        hypothesis_id: str,
        categories: set[str],
        budget: int | None = None,
    ) -> list[ExperimentConditionSpec]:
        plan = project.experiment_plan
        if plan is None:
            return []
        design = self._explicit_design(plan, hypothesis_id)
        if design is None:
            return []
        validate_design(
            design,
            plan,
            budget=budget,
            allowed_categories=categories,
            allowed_detectors=self._approved_detectors(project),
            allowed_strategies=self._approved_strategies(project),
        )
        return compile_design(
            design,
            plan=plan,
            budget=budget,
            allowed_categories=categories,
            allowed_detectors=self._approved_detectors(project),
            allowed_strategies=self._approved_strategies(project),
        )

    @staticmethod
    def _campaign_categories(project: ResearchProject) -> set[str]:
        campaign = project.experiment_campaign
        plan = project.experiment_plan
        if campaign is None or plan is None:
            return set(plan.categories if plan is not None else [])
        audit = next(
            (item for item in project.dataset_audits if item.id == campaign.dataset_audit_id),
            None,
        )
        return {
            category
            for category in plan.categories
            if audit is None or category in audit.categories
        }

    @staticmethod
    def _approved_strategies(project: ResearchProject) -> set[str]:
        return BUILTIN_STRATEGIES | {
            item.name
            for item in project.method_implementations
            if item.kind == "selection_strategy" and item.status == "approved"
        }

    @staticmethod
    def _approved_detectors(project: ResearchProject) -> set[str]:
        return BUILTIN_DETECTORS | {
            item.name
            for item in project.method_implementations
            if item.kind == "detector" and item.status == "approved"
        }

    @classmethod
    def _contract_is_executable(
        cls, project: ResearchProject, contract, *, hypothesis_id: str | None = None
    ) -> bool:
        if hypothesis_id and project.experiment_plan is not None:
            design = cls._explicit_design(project.experiment_plan, hypothesis_id)
            if design is not None:
                try:
                    validate_design(
                        design,
                        project.experiment_plan,
                        allowed_categories=set(project.experiment_plan.categories),
                        allowed_detectors=cls._approved_detectors(project),
                        allowed_strategies=cls._approved_strategies(project),
                    )
                except ValueError:
                    return False
                return is_supported_primary_metric(design.analysis.primary_metric or "")
        if contract.kind not in {
            "selection_main_effect",
            "detector_interaction",
            "query_adaptation",
        }:
            return False
        if contract.kind == "detector_interaction":
            approved = cls._approved_detectors(project)
            return contract.treatment in approved and contract.control in approved
        approved = cls._approved_strategies(project)
        return contract.treatment in approved and contract.control in approved

    @classmethod
    def _select_hypothesis(
        cls,
        project: ResearchProject,
        *,
        hypothesis_id: str,
    ) -> Hypothesis:
        approved_hypothesis_ids = (
            project.experiment_plan.hypothesis_ids if project.experiment_plan else []
        )
        eligible = [
            hypothesis
            for hypothesis in project.hypotheses
            if hypothesis.analysis_contract is not None
            and cls._contract_is_executable(
                project, hypothesis.analysis_contract, hypothesis_id=hypothesis.id
            )
            and hypothesis.id in approved_hypothesis_ids
            and hypothesis.id == hypothesis_id
        ]
        eligible.sort(
            key=lambda item: (
                item.status not in {HypothesisStatus.APPROVED, HypothesisStatus.SHORTLISTED},
                -(item.score.elo if item.score else 0),
            )
        )
        if not eligible:
            plan = project.experiment_plan
            approved_ids = plan.hypothesis_ids if plan is not None else []
            diagnostics = []
            if plan is not None and plan.design_generation_status == "needs_correction":
                diagnostics.append(
                    "实验设计仍待修正："
                    f"{plan.design_generation_fallback_reason or '未提供原因'}"
                )
            if plan is not None and hypothesis_id not in approved_ids:
                diagnostics.append(
                    "该创新点不在已批准的预注册计划内（计划保留："
                    + (", ".join(approved_ids) if approved_ids else "无")
                    + "）。可能是主指标或自定义方法未被当前执行器支持，"
                    "请返回计划审批页重新生成实验计划"
                )
            if not diagnostics:
                diagnostics.append(
                    "该创新点的分析契约引用了尚未注册实现的自定义检测器或选样策略，"
                    "请先调用对应的方法生成端点并获批后再执行"
                )
            raise ValueError(
                "所选创新点无法进入实验：" + "；".join(diagnostics)
            )
        return eligible[0]

    @classmethod
    def _eligible_hypotheses(cls, project: ResearchProject) -> list[Hypothesis]:
        approved_ids = project.experiment_plan.hypothesis_ids if project.experiment_plan else []
        by_id = {item.id: item for item in project.hypotheses}
        return [
            by_id[hypothesis_id]
            for hypothesis_id in approved_ids
            if hypothesis_id in by_id
            and by_id[hypothesis_id].analysis_contract is not None
            and cls._contract_is_executable(
                project,
                by_id[hypothesis_id].analysis_contract,
                hypothesis_id=hypothesis_id,
            )
        ]

    @staticmethod
    def _approved_categories(
        project: ResearchProject, dataset: DatasetManifest
    ) -> list[str]:
        plan = project.experiment_plan
        if plan is None:
            return []
        return [category for category in plan.categories if category in dataset.categories]

    @staticmethod
    def _campaign(project: ResearchProject) -> ExperimentCampaign:
        if project.experiment_campaign is None:
            raise ValueError("The project has no active experiment campaign")
        campaign = project.experiment_campaign
        # Backfill fields introduced by the innovation-per-round protocol for
        # projects created by the previous single-hypothesis version.
        if not campaign.hypothesis_ids and campaign.hypothesis_id:
            campaign.hypothesis_ids = [campaign.hypothesis_id]
        for experiment_round in campaign.rounds:
            if not experiment_round.hypothesis_id:
                experiment_round.hypothesis_id = campaign.hypothesis_id
            if not experiment_round.treatment:
                experiment_round.treatment = campaign.treatment
            if not experiment_round.control:
                experiment_round.control = campaign.control
            if not experiment_round.metric:
                experiment_round.metric = campaign.metric
            if experiment_round.iteration_target < 3:
                experiment_round.iteration_target = 3
        return campaign

    @staticmethod
    def _hypothesis(project: ResearchProject, hypothesis_id: str) -> Hypothesis:
        hypothesis = next(
            (item for item in project.hypotheses if item.id == hypothesis_id), None
        )
        if hypothesis is None:
            raise ValueError(f"Unknown campaign hypothesis: {hypothesis_id}")
        return hypothesis

    @staticmethod
    def _campaign_run_count(project: ResearchProject) -> int:
        return sum(run.round_id is not None for run in project.runs)

    @staticmethod
    def _validated_next_phase(value: str) -> ExperimentPhaseName:
        if value == "complete":
            return "replication"
        allowed: tuple[ExperimentPhaseName, ...] = (
            "sensitivity",
            "main_study",
            "replication",
            "ablation",
            "cross_dataset",
        )
        return value if value in allowed else "sensitivity"  # type: ignore[return-value]

    @staticmethod
    def _objective_for(decision: str, metric: str) -> str:
        objectives = {
            "expand": f"扩展类别覆盖，检验 {metric} 效应是否跨类别复现。",
            "replicate": f"增加独立随机种子，收紧 {metric} 成对效应的不确定性。",
            "diagnose": "针对失败、反向效应或边界条件执行最小诊断实验。",
            "stop": "补足停止前所需的最小证据。",
        }
        return objectives.get(decision, objectives["diagnose"])

    @staticmethod
    def _update_nodes_for_round(
        campaign: ExperimentCampaign,
        experiment_round: ExperimentRound,
        project: ResearchProject,
        summary: dict[str, Any],
    ) -> None:
        run_by_id = {run.id: run for run in project.runs}
        for node in campaign.nodes:
            if node.id not in experiment_round.node_ids:
                continue
            node_runs = [run_by_id[run_id] for run_id in node.run_ids if run_id in run_by_id]
            node.status = (
                "succeeded"
                if node_runs
                and all(
                    run.status == RunStatus.SUCCEEDED and run.verified
                    for run in node_runs
                )
                else "failed"
            )
            node.result_summary = {
                "metric": experiment_round.metric,
                "runs": [
                    {
                        "run_id": run.id,
                        "strategy": run.selection_strategy,
                        "status": run.status,
                        "value": run.metrics.get(experiment_round.metric),
                    }
                    for run in node_runs
                ],
                "round_mean_difference": summary.get("mean_difference"),
            }
            node.error_history = [run.error for run in node_runs if run.error]

    @staticmethod
    def _refresh_efficiency(
        campaign: ExperimentCampaign,
        *,
        project: ResearchProject | None = None,
        additional_runs: int = 0,
    ) -> None:
        if project is None:
            selected = sum(len(item.run_ids) for item in campaign.rounds)
            terminal = 0
        else:
            selected = sum(run.round_id is not None for run in project.runs) + additional_runs
            terminal = sum(
                run.round_id is not None
                and run.status in {RunStatus.SUCCEEDED, RunStatus.FAILED}
                for run in project.runs
            )
        exhaustive = campaign.exhaustive_run_count
        runs_avoided = max(exhaustive - selected, 0)
        savings = runs_avoided / exhaustive if exhaustive else 0.0
        for experiment_round in campaign.rounds:
            experiment_round.efficiency.update(
                {
                    "campaign_selected_runs": selected,
                    "campaign_terminal_runs": terminal,
                    "exhaustive_run_count": exhaustive,
                    "runs_avoided": runs_avoided,
                    "estimated_compute_savings_ratio": round(savings, 4),
                }
            )
