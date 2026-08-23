from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from fastapi.testclient import TestClient

from fsad_scientist.agents.mock_runtime import MockScientistRuntime
from fsad_scientist.api.app import create_app
from fsad_scientist.config import get_settings


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Continue a real adaptive campaign until the preregistered verdict."
    )
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--max-runs", type=int, default=0, help="0 = until campaign completes")
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument(
        "--start-cycle2",
        action="store_true",
        help="revise hypotheses, redesign, approve and initialize the next cycle first",
    )
    arguments = parser.parse_args()

    settings = get_settings()
    app = create_app(settings=settings, runtime=MockScientistRuntime())
    completed: list[dict[str, Any]] = []
    with TestClient(app) as client:
        if arguments.start_cycle2:
            project = checked(client.get(f"/api/v1/projects/{arguments.project_id}"))
            print(f"Starting cycle 2 from stage {project['stage']}", flush=True)
            for _ in range(3):
                project = checked(
                    client.post(f"/api/v1/projects/{arguments.project_id}/advance")
                )
                print(
                    f"advanced -> {project['stage']} "
                    f"(cycle {project['research_cycle']}, "
                    f"{len(project['hypotheses'])} hypotheses)",
                    flush=True,
                )
            if project["stage"] != "awaiting_experiment_approval":
                raise SystemExit(f"Unexpected stage before approval: {project['stage']}")
            project = checked(
                client.post(
                    f"/api/v1/projects/{arguments.project_id}/approve",
                    json={"approved_by": "user-claude-session"},
                )
            )
            print(f"plan approved -> {project['stage']}", flush=True)
            audit = project["dataset_audits"][-1]
            project = checked(
                client.post(
                    f"/api/v1/projects/{arguments.project_id}/experiment-campaign/initialize",
                    json={
                        "dataset_manifest_path": audit["manifest_path"],
                        "detector": "anomalydino",
                        "device": "cuda:0",
                        "max_rounds": 3,
                        "max_runs": 24,
                    },
                )
            )
            print(
                f"cycle-2 campaign initialized: {project['experiment_campaign']['id']}",
                flush=True,
            )

        project = checked(client.get(f"/api/v1/projects/{arguments.project_id}"))
        if project["experiment_campaign"] is None:
            raise RuntimeError("The resumed project has no experiment campaign")
        print(
            f"Resuming campaign {project['experiment_campaign']['id']} "
            f"(round {project['experiment_campaign']['current_round']}/"
            f"{project['experiment_campaign']['max_rounds']}, "
            f"status {project['experiment_campaign']['status']})",
            flush=True,
        )

        while True:
            campaign = project["experiment_campaign"]
            if campaign["status"] == "awaiting_feedback":
                project = checked(
                    client.post(
                        f"/api/v1/projects/{arguments.project_id}/experiment-campaign/review"
                    )
                )
                campaign = project["experiment_campaign"]
                rounds = campaign.get("rounds") or []
                feedback = rounds[-2].get("feedback") if len(rounds) >= 2 else None
                print(
                    json.dumps(
                        {
                            "event": "round_reviewed",
                            "round": campaign["current_round"],
                            "decision": feedback.get("decision") if feedback else None,
                            "next_phase": feedback.get("next_phase") if feedback else None,
                            "rationale": (feedback.get("rationale") or "")[:200]
                            if feedback
                            else None,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if campaign["status"] != "active":
                print(f"Campaign finished: {campaign['status']} "
                      f"({campaign.get('termination_reason')})", flush=True)
                break
            if arguments.max_runs and len(completed) >= arguments.max_runs:
                print(f"Stopped by --max-runs={arguments.max_runs}", flush=True)
                break

            print(f"Executing real run {len(completed) + 1}...", flush=True)
            response = checked(
                client.post(
                    f"/api/v1/projects/{arguments.project_id}/experiment-campaign/execute-next",
                    json={
                        "candidate_pool_size": campaign["candidate_pool_size"],
                        "timeout_seconds": arguments.timeout,
                        "force_embeddings": False,
                    },
                )
            )
            execution = response["execution"]
            project = response["project"]
            completed.append(
                {
                    "run_id": response["run_id"],
                    "status": execution["status"],
                    "duration_seconds": execution["duration_seconds"],
                    "metrics": (
                        execution["normalized_result"]["metrics"]
                        if execution["normalized_result"]
                        else None
                    ),
                    "error": execution["error"],
                }
            )
            print(json.dumps(completed[-1], ensure_ascii=False), flush=True)
            failures = [item for item in completed if item["status"] != "succeeded"]
            if len(failures) >= 2 and all(
                item["status"] != "succeeded" for item in completed[-2:]
            ):
                print(
                    f"Stopping: {len(failures)} failed runs, last two consecutive "
                    "(environment issue suspected)",
                    flush=True,
                )
                break

        if project["experiment_campaign"]["status"] != "completed":
            raise SystemExit(
                f"Campaign is {project['experiment_campaign']['status']}; "
                "no verdict can be computed yet"
            )

        project = checked(
            client.post(f"/api/v1/projects/{arguments.project_id}/results/finalize")
        )
        project = checked(client.post(f"/api/v1/projects/{arguments.project_id}/advance"))
        hypothesis_id = project["experiment_campaign"]["hypothesis_id"]
        hypothesis = next(
            item for item in project["hypotheses"] if item["id"] == hypothesis_id
        )
        findings = [
            item for item in project["findings"] if item["hypothesis_id"] == hypothesis["id"]
        ]
        verdict = {
            "project_id": project["id"],
            "stage": project["stage"],
            "research_cycle": project["research_cycle"],
            "hypothesis_id": hypothesis["id"],
            "hypothesis_title": hypothesis["title"],
            "hypothesis_status": hypothesis["status"],
            "finding": findings[0] if findings else None,
            "completed_runs": len(completed),
        }
        print(json.dumps(verdict, ensure_ascii=False, indent=2), flush=True)


def checked(response) -> dict[str, Any]:
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
    return response.json()


if __name__ == "__main__":
    main()
