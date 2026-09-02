from __future__ import annotations

from collections import defaultdict
from statistics import fmean
from typing import Any, Literal

from fsad_scientist.datasets.models import DatasetManifest
from fsad_scientist.domain.enums import HypothesisStatus, RunStatus
from fsad_scientist.domain.models import (
    DatasetAuditRecord,
    ExperimentCampaign,
    ExperimentCell,
    ExperimentFeedbackProposal,
    ExperimentNodeRecord,
    ExperimentRound,
    ExperimentRun,
    Hypothesis,
    ResearchProject,
    new_id,
    utc_now,
)
from fsad_scientist.experiments.code_safety import BUILTIN_DETECTORS, BUILTIN_STRATEGIES
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
        if contract.treatment not in approved_strategies:
            raise ValueError(
                f"Unsupported treatment strategy: {contract.treatment}"
                "；自定义策略需先生成实现并获批准"
            )
        if contract.control not in approved_strategies:
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
        if len(categories) * len(shots) * len(seeds) < 3:
            raise ValueError(
                "Each innovation Round needs at least three distinct registered cells "
                "for its three internal iterations"
            )

        historical_campaign_runs = sum(
            run.round_id is not None and run.plan_id == plan.id for run in project.runs
        )
        remaining_run_budget = project.spec.budget.max_experiments - historical_campaign_runs
        effective_max_runs = min(max_runs, remaining_run_budget)
        if effective_max_runs < 6:
            raise ValueError(
                "The remaining project budget cannot fund one innovation Round (6 runs)"
            )
        # A complete Round consumes exactly three paired iterations.  Never
        # promise more innovation rounds than the frozen project budget can fund.
        round_capacity = min(
            len(hypothesis_ids),
            max_rounds,
            effective_max_runs // 6,
        )
        hypothesis_ids = hypothesis_ids[: max(1, round_capacity)]
        if not hypothesis_ids:
            raise ValueError("At least one selected innovation is required")
        initial_k = 2 if 2 in shots else shots[0]
        # Sequential campaigns retain the original one-pair-at-a-time protocol.
        # Parallel campaigns pre-register all three iterations for every
        # selected innovation, allowing independent rounds to run concurrently
        # without serially enumerating the full factorial space.
        initial_seeds = seeds[:1]
        initial_cells = [
            ExperimentCell(category=categories[0], shots=initial_k, seed=seed)
            for seed in initial_seeds
        ]
        parallel_cells = [
            ExperimentCell(category=category, shots=shot, seed=seed)
            for category in categories
            for shot in shots
            for seed in seeds
        ][:3]
        exhaustive_run_count = (
            len(hypothesis_ids) * len(categories) * len(shots) * len(seeds) * 2
        )
        primary_metric = normalize_primary_metric(contract.metric)
        if primary_metric not in SUPPORTED_PRIMARY_METRICS:
            raise ValueError(f"Unsupported primary metric: {contract.metric}")
        campaign = ExperimentCampaign(
            hypothesis_id=hypothesis.id,
            hypothesis_ids=hypothesis_ids,
            dataset_audit_id=audit.id,
            dataset_manifest_path=audit.manifest_path,
            dataset_digest=audit.digest,
            protocol=f"pool_compression_m{project.spec.constraints.candidate_pool_size}",
            candidate_pool_size=project.spec.constraints.candidate_pool_size,
            detector=detector,
            treatment=contract.treatment,
            control=contract.control,
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
                campaign.treatment = first_contract.treatment
                campaign.control = first_contract.control
                campaign.metric = normalize_primary_metric(first_contract.metric)
            campaign.next_action = "execute_parallel_batch"
            self._refresh_efficiency(campaign)
            return campaign, all_runs

        first_round, nodes, runs = self._build_round(
            project,
            campaign=campaign,
            index=1,
            phase="feasibility",
            objective="验证真实数据、特征、支持集选择和检测器链路，并获得首批成对效应。",
            rationale=(
                f"先在 bottle、K=2 和一个随机种子上比较 {contract.control} 与 "
                f"{contract.treatment}，并使用 {detector} 执行检测；"
                "用一组成对真实运行换取端到端可行性和初始效应信息。"
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
        metric = normalize_primary_metric(current.metric)
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
        used = {
            (run.category, run.shots, run.seed)
            for run in project.runs
            if run.round_id is not None and run.hypothesis_id == target_hypothesis_id
        }
        candidates = [
            ExperimentCell(category=category, shots=shots, seed=seed)
            for category in categories
            for shots in sorted(plan.shots)
            for seed in sorted(plan.seeds)
            if (category, shots, seed) not in used
        ]
        # Prefer replication breadth, then K sensitivity, before accumulating seeds.
        current_categories = {
            run.category
            for run in project.runs
            if run.round_id is not None and run.hypothesis_id == target_hypothesis_id
        }
        current_shots = {
            run.shots
            for run in project.runs
            if run.round_id is not None and run.hypothesis_id == target_hypothesis_id
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
        selected: list[ExperimentCell] = []
        for item in proposal.recommended_cells:
            key = (item.category, item.shots, item.seed)
            if key in allowed_by_key and allowed_by_key[key] not in selected:
                selected.append(allowed_by_key[key])
            if len(selected) == 2:
                break
        for item in allowed:
            if len(selected) >= 2:
                break
            if item not in selected:
                selected.append(item)
        if len(selected) != 2:
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

        remaining_runs = campaign.max_runs - self._campaign_run_count(project)
        if remaining_runs < 6:
            campaign.status = "completed"
            campaign.next_action = "analyze_verified_results"
            campaign.termination_reason = "run_budget_exhausted_before_next_innovation"
            campaign.completed_at = utc_now()
            self._refresh_efficiency(campaign, project=project)
            return []
        next_hypothesis_id = campaign.hypothesis_ids[next_index - 1]
        next_hypothesis = self._hypothesis(project, next_hypothesis_id)
        next_contract = next_hypothesis.analysis_contract
        if next_contract is None:
            raise ValueError("The next innovation has no analysis contract")
        allowed = self.allowed_next_cells(project, hypothesis_id=next_hypothesis_id)
        if not allowed:
            raise ValueError("No experiment cell is available for the next innovation")
        parent_id = current.node_ids[0] if current.node_ids else None
        next_round, nodes, runs = self._build_round(
            project,
            campaign=campaign,
            index=next_index,
            phase="feasibility",
            objective=f"验证创新点 H{next_index}：{next_hypothesis.title}",
            rationale=(
                "上一创新点已完成三次自动迭代。现在切换到下一创新点，并先执行"
                f"第 1 次迭代；比较 {next_contract.treatment} 与 {next_contract.control}。"
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
        remaining_pairs = max(
            (campaign.max_runs - self._campaign_run_count(project)) // 2,
            0,
        )
        needed = min(max(target_cells - existing_cells, 0), remaining_pairs)
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
        if contract is None or not self._contract_is_executable(project, contract):
            raise ValueError(f"Hypothesis is not executable: {round_hypothesis_id}")
        primary_metric = normalize_primary_metric(contract.metric)
        if primary_metric not in SUPPORTED_PRIMARY_METRICS:
            raise ValueError(f"Unsupported primary metric: {contract.metric}")
        campaign.hypothesis_id = round_hypothesis_id
        campaign.treatment = contract.treatment
        campaign.control = contract.control
        campaign.metric = primary_metric
        round_id = new_id("round")
        nodes: list[ExperimentNodeRecord] = []
        runs: list[ExperimentRun] = []
        for cell_offset, cell in enumerate(cells):
            iteration = min(iteration_start + cell_offset, 3)
            node_id = new_id("experiment_node")
            cost = 2.0
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
            node_runs = [
                ExperimentRun(
                    plan_id=plan.id,
                    hypothesis_id=round_hypothesis_id,
                    protocol=campaign.protocol,
                    dataset="MVTec AD",
                    category=cell.category,
                    detector=campaign.detector,
                    selection_strategy=strategy,
                    shots=cell.shots,
                    seed=cell.seed,
                    iteration=iteration,
                    round_id=round_id,
                    node_id=node_id,
                    phase=phase,
                    status=RunStatus.QUEUED,
                )
                for strategy in (campaign.control, campaign.treatment)
            ]
            nodes.append(
                ExperimentNodeRecord(
                    id=node_id,
                    round_id=round_id,
                    iteration=iteration,
                    parent_id=parent_id,
                    phase=phase,
                    objective=(
                        f"{cell.category} / K={cell.shots} / seed={cell.seed}："
                        f"成对比较 {campaign.treatment} 与 {campaign.control}。"
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
            treatment=contract.treatment,
            control=contract.control,
            metric=primary_metric,
            node_ids=[node.id for node in nodes],
            run_ids=[run.id for run in runs],
            efficiency={"planned_runs": len(runs)},
        )
        return experiment_round, nodes, runs

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
    def _contract_is_executable(cls, project: ResearchProject, contract) -> bool:
        if contract.kind not in {"selection_main_effect", "query_adaptation"}:
            return False
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
            and cls._contract_is_executable(project, hypothesis.analysis_contract)
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
            raise ValueError(
                "The selected innovation is not approved or executable by the current toolchain "
                "(random/k_center or an approved custom strategy)"
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
            and cls._contract_is_executable(project, by_id[hypothesis_id].analysis_contract)
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
