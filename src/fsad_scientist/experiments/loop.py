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
from fsad_scientist.science.experiment_tree import ExperimentNode, ExperimentPhase

ExperimentPhaseName = Literal[
    "feasibility",
    "sensitivity",
    "main_study",
    "replication",
    "ablation",
    "cross_dataset",
]


class AdaptiveExperimentPlanner:
    """Build and validate a budgeted result-to-next-experiment loop.

    The language model may advise which *registered* cells are most informative,
    but this class owns the admissible search space, pairing, budgets, lifecycle,
    and run construction. It therefore cannot turn an LLM response into an
    arbitrary command or leak test labels into support-set selection.
    """

    supported_detectors = {"anomalydino", "patchcore", "subspacead"}
    supported_strategies = {"random", "k_center"}

    def initialize(
        self,
        project: ResearchProject,
        *,
        audit: DatasetAuditRecord,
        dataset: DatasetManifest,
        device: str,
        detector: str = "anomalydino",
        max_rounds: int = 3,
        max_runs: int = 24,
    ) -> tuple[ExperimentCampaign, list[ExperimentRun]]:
        plan = project.experiment_plan
        if plan is None or not plan.approved:
            raise ValueError("The preregistered experiment plan must be approved first")
        if not audit.verified or audit.digest != dataset.digest:
            raise ValueError("A verified dataset audit matching the manifest is required")
        if detector not in self.supported_detectors or detector not in plan.detectors:
            raise ValueError(f"Detector is not approved and executable: {detector}")

        hypothesis = self._select_hypothesis(project)
        contract = hypothesis.analysis_contract
        if contract is None:
            raise ValueError("The selected hypothesis has no analysis contract")
        if contract.treatment not in self.supported_strategies:
            raise ValueError(f"Unsupported treatment strategy: {contract.treatment}")
        if contract.control not in self.supported_strategies:
            raise ValueError(f"Unsupported control strategy: {contract.control}")

        categories = self._approved_categories(project, dataset)
        if not categories:
            raise ValueError("No approved experiment category exists in the dataset")
        shots = sorted(set(plan.shots) & set(project.spec.constraints.shots))
        seeds = sorted(set(plan.seeds))
        if not shots or not seeds:
            raise ValueError("The experiment plan must contain at least one K and one seed")

        effective_max_runs = min(max_runs, project.spec.budget.max_experiments)
        if effective_max_runs < 2:
            raise ValueError("The adaptive campaign requires budget for one paired experiment")
        initial_k = 2 if 2 in shots else shots[0]
        initial_seeds = seeds[: min(2, effective_max_runs // 2)]
        initial_cells = [
            ExperimentCell(category=categories[0], shots=initial_k, seed=seed)
            for seed in initial_seeds
        ]
        exhaustive_run_count = len(categories) * len(shots) * len(seeds) * 2
        campaign = ExperimentCampaign(
            hypothesis_id=hypothesis.id,
            dataset_audit_id=audit.id,
            dataset_manifest_path=audit.manifest_path,
            dataset_digest=audit.digest,
            protocol=f"pool_compression_m{project.spec.constraints.candidate_pool_size}",
            candidate_pool_size=project.spec.constraints.candidate_pool_size,
            detector=detector,
            treatment=contract.treatment,
            control=contract.control,
            metric=contract.metric,
            device=device,
            max_rounds=max_rounds,
            max_runs=effective_max_runs,
            exhaustive_run_count=exhaustive_run_count,
        )
        first_round, nodes, runs = self._build_round(
            project,
            campaign=campaign,
            index=1,
            phase="feasibility",
            objective="验证真实数据、特征、支持集选择和检测器链路，并获得首批成对效应。",
            rationale=(
                "先在 bottle、K=2 和两个随机种子上比较 random 与 k-center；"
                "用四次真实运行换取端到端可行性和初始效应信息。"
            ),
            cells=initial_cells,
            information_gain=0.90,
            falsification_value=0.80,
            parent_id=None,
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
        metric = campaign.metric
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
            treatment = strategies.get(campaign.treatment)
            control = strategies.get(campaign.control)
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

        hypothesis = self._hypothesis(project, campaign.hypothesis_id)
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
                and run.hypothesis_id == campaign.hypothesis_id
                and run.status == RunStatus.SUCCEEDED
                and run.verified
                and metric in run.metrics
            ):
                campaign_pair_runs[(run.category, run.shots, run.seed)][
                    run.selection_strategy
                ] = run
        cumulative_differences = [
            strategies[campaign.treatment].metrics[metric]
            - strategies[campaign.control].metrics[metric]
            for strategies in campaign_pair_runs.values()
            if campaign.treatment in strategies and campaign.control in strategies
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

    def allowed_next_cells(self, project: ResearchProject) -> list[ExperimentCell]:
        campaign = self._campaign(project)
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
            if run.round_id is not None and run.hypothesis_id == campaign.hypothesis_id
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
            run.category for run in project.runs if run.round_id is not None
        }
        current_shots = {run.shots for run in project.runs if run.round_id is not None}
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

        allowed = self.allowed_next_cells(project)
        allowed_keys = {(cell.category, cell.shots, cell.seed) for cell in allowed}
        remaining_pairs = max((campaign.max_runs - self._campaign_run_count(project)) // 2, 0)
        selected: list[ExperimentCell] = []
        for cell in proposal.recommended_cells:
            key = (cell.category, cell.shots, cell.seed)
            if key in allowed_keys and key not in {
                (item.category, item.shots, item.seed) for item in selected
            }:
                selected.append(cell)
            if len(selected) >= min(2, remaining_pairs):
                break

        minimum_pairs = int(summary.get("minimum_pairs", 6))
        enough_evidence = int(summary.get("pair_count", 0)) >= minimum_pairs
        if proposal.stop and not enough_evidence:
            proposal = proposal.model_copy(
                deep=True,
                update={
                    "decision": "expand",
                    "next_phase": "replication",
                    "stop": False,
                    "rationale": (
                        proposal.rationale
                        + " 系统否决了提前停止：尚未达到预注册最小成对样本数。"
                    ),
                },
            )
        exhausted = (
            campaign.current_round >= campaign.max_rounds
            or remaining_pairs == 0
            or not allowed
        )
        should_stop = proposal.stop and enough_evidence
        target_cells = min(2, remaining_pairs)
        if not should_stop and not exhausted and len(selected) < target_cells:
            selected_keys = {
                (item.category, item.shots, item.seed) for item in selected
            }
            for cell in allowed:
                key = (cell.category, cell.shots, cell.seed)
                if key not in selected_keys:
                    selected.append(cell)
                    selected_keys.add(key)
                if len(selected) >= target_cells:
                    break
        if not should_stop:
            proposal = proposal.model_copy(
                deep=True,
                update={"recommended_cells": selected},
            )

        current.feedback = proposal
        current.result_summary = summary
        current.status = "completed"
        current.completed_at = utc_now()
        self._update_nodes_for_round(campaign, current, project, summary)

        if should_stop or exhausted:
            campaign.status = "completed"
            campaign.next_action = "analyze_verified_results"
            if should_stop:
                campaign.termination_reason = "advisor_stop_after_minimum_pairs"
            elif campaign.current_round >= campaign.max_rounds:
                campaign.termination_reason = "maximum_rounds_reached"
            elif remaining_pairs == 0:
                campaign.termination_reason = "run_budget_exhausted"
            else:
                campaign.termination_reason = "allowed_search_space_exhausted"
            campaign.completed_at = utc_now()
            self._refresh_efficiency(campaign, project=project)
            return []

        if not selected:
            campaign.status = "completed"
            campaign.next_action = "analyze_verified_results"
            campaign.termination_reason = "no_valid_next_experiment"
            campaign.completed_at = utc_now()
            self._refresh_efficiency(campaign, project=project)
            return []

        phase = self._validated_next_phase(proposal.next_phase)
        parent_id = current.node_ids[0] if current.node_ids else None
        next_index = campaign.current_round + 1
        next_round, nodes, runs = self._build_round(
            project,
            campaign=campaign,
            index=next_index,
            phase=phase,
            objective=self._objective_for(proposal.decision, campaign.metric),
            rationale=proposal.rationale,
            cells=selected,
            information_gain=proposal.expected_information_gain,
            falsification_value=0.90 if proposal.decision == "diagnose" else 0.75,
            parent_id=parent_id,
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
            current.status = "ready_for_feedback"
            current.completed_at = utc_now()
            campaign.status = "awaiting_feedback"
            campaign.next_action = "analyze_round_and_plan_next"
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
        current = campaign.rounds[-1]
        by_id = {run.id: run for run in project.runs}
        candidates = [
            by_id[run_id]
            for run_id in current.run_ids
            if run_id in by_id and by_id[run_id].status == RunStatus.QUEUED
        ]
        node_priority = {node.id: node.priority for node in campaign.nodes}
        return sorted(
            candidates,
            key=lambda run: (-node_priority.get(run.node_id or "", 0.0), run.id),
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
    ) -> tuple[ExperimentRound, list[ExperimentNodeRecord], list[ExperimentRun]]:
        plan = project.experiment_plan
        if plan is None:
            raise ValueError("Experiment plan is missing")
        round_id = new_id("round")
        nodes: list[ExperimentNodeRecord] = []
        runs: list[ExperimentRun] = []
        for cell in cells:
            node_id = new_id("experiment_node")
            cost = 2.0
            priority_node = ExperimentNode(
                id=node_id,
                hypothesis_id=campaign.hypothesis_id,
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
                    hypothesis_id=campaign.hypothesis_id,
                    protocol=campaign.protocol,
                    dataset="MVTec AD",
                    category=cell.category,
                    detector=campaign.detector,
                    selection_strategy=strategy,
                    shots=cell.shots,
                    seed=cell.seed,
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
            node_ids=[node.id for node in nodes],
            run_ids=[run.id for run in runs],
            efficiency={"planned_runs": len(runs)},
        )
        return experiment_round, nodes, runs

    @staticmethod
    def _select_hypothesis(project: ResearchProject) -> Hypothesis:
        approved_hypothesis_ids = (
            project.experiment_plan.hypothesis_ids if project.experiment_plan else []
        )
        eligible = [
            hypothesis
            for hypothesis in project.hypotheses
            if hypothesis.analysis_contract is not None
            and hypothesis.analysis_contract.kind == "selection_main_effect"
            and hypothesis.id in approved_hypothesis_ids
        ]
        eligible.sort(
            key=lambda item: (
                item.status not in {HypothesisStatus.APPROVED, HypothesisStatus.SHORTLISTED},
                -(item.score.elo if item.score else 0),
            )
        )
        if not eligible:
            raise ValueError(
                "The approved plan has no selection_main_effect hypothesis for paired execution"
            )
        return eligible[0]

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
        return project.experiment_campaign

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
                "metric": campaign.metric,
                "runs": [
                    {
                        "run_id": run.id,
                        "strategy": run.selection_strategy,
                        "status": run.status,
                        "value": run.metrics.get(campaign.metric),
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
