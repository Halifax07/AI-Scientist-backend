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
        description="Fill missing cells in an unstarted adaptive experiment round."
    )
    parser.add_argument("project_id")
    parser.add_argument("--target-cells", type=int, default=2)
    arguments = parser.parse_args()

    settings = get_settings()
    repository = JsonProjectRepository(settings.storage_path)
    project = repository.get(arguments.project_id)
    new_runs = AdaptiveExperimentPlanner().fill_current_round(
        project,
        target_cells=arguments.target_cells,
    )
    project.runs.extend(new_runs)
    project.record_event(
        actor="experiment_action_validator",
        action="fill_current_experiment_round",
        summary=(
            f"动作校验后为尚未启动的当前轮补齐 {len(new_runs) // 2} 个成对实验单元。"
        ),
        payload={"new_run_ids": [run.id for run in new_runs]},
    )
    repository.save(project)
    print(f"project={project.id}, added_runs={len(new_runs)}")


if __name__ == "__main__":
    main()
