from __future__ import annotations

import argparse
import sys

from fsad_scientist.config import get_settings
from fsad_scientist.experiments.loop import AdaptiveExperimentPlanner
from fsad_scientist.repository import JsonProjectRepository


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Rebuild derived round summaries from immutable experiment runs."
    )
    parser.add_argument("project_id")
    arguments = parser.parse_args()

    settings = get_settings()
    repository = JsonProjectRepository(settings.storage_path)
    project = repository.get(arguments.project_id)
    campaign = project.experiment_campaign
    if campaign is None:
        raise SystemExit("project has no experiment campaign")

    planner = AdaptiveExperimentPlanner()
    refreshed: list[int] = []
    for experiment_round in campaign.rounds:
        if experiment_round.status != "completed":
            continue
        experiment_round.result_summary = planner.summarize_round(
            project,
            round_id=experiment_round.id,
        )
        refreshed.append(experiment_round.index)
    if campaign.status == "completed" and campaign.termination_reason is None:
        if campaign.current_round >= campaign.max_rounds:
            campaign.termination_reason = "maximum_rounds_reached"
        elif len(project.runs) >= campaign.max_runs:
            campaign.termination_reason = "run_budget_exhausted"
        else:
            campaign.termination_reason = "legacy_completed_campaign"
    project.record_event(
        actor="derived_summary_rebuilder",
        action="refresh_campaign_summaries",
        summary=f"已从不可变 Run 重新计算 {len(refreshed)} 个完成轮次的多指标摘要。",
        payload={"round_indices": refreshed},
    )
    repository.save(project)
    print(f"refreshed project={project.id}, rounds={refreshed}")


if __name__ == "__main__":
    main()
