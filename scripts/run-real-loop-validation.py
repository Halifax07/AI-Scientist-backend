from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from fsad_scientist.agents.mock_runtime import MockScientistRuntime
from fsad_scientist.api.app import create_app
from fsad_scientist.config import get_settings


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Exercise the complete adaptive loop on real MVTec images and local GPU."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--project-id")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--review-only", action="store_true")
    parser.add_argument("--qwen-feedback", action="store_true")
    arguments = parser.parse_args()
    if arguments.runs < 1:
        raise SystemExit("--runs must be positive")

    settings = get_settings()
    app = (
        create_app(settings=settings)
        if arguments.qwen_feedback
        else create_app(settings=settings, runtime=MockScientistRuntime())
    )
    with TestClient(app) as client:
        if arguments.project_id:
            project = checked(
                client.get(f"/api/v1/projects/{arguments.project_id}")
            )
            if project["experiment_campaign"] is None:
                raise RuntimeError("The resumed project has no experiment campaign")
        else:
            project = checked(client.post("/api/v1/projects/demo"))
            while project["stage"] != "awaiting_experiment_approval":
                project = checked(
                    client.post(f"/api/v1/projects/{project['id']}/advance")
                )
            project = checked(
                client.post(
                    f"/api/v1/projects/{project['id']}/approve",
                    json={"approved_by": "real-loop-validation"},
                )
            )
            project = checked(
                client.post(
                    f"/api/v1/projects/{project['id']}/dataset/audit",
                    json={
                        "root": str(arguments.dataset.resolve()),
                        "dataset_name": "MVTec AD",
                    },
                )
            )
            manifest_path = project["dataset_audits"][-1]["manifest_path"]
            project = checked(
                client.post(
                    f"/api/v1/projects/{project['id']}/experiment-campaign/initialize",
                    json={
                        "dataset_manifest_path": manifest_path,
                        "detector": "anomalydino",
                        "device": "cuda:0",
                        "max_rounds": 3,
                        "max_runs": 24,
                    },
                )
            )

        if arguments.review_only:
            if project["experiment_campaign"]["status"] != "awaiting_feedback":
                raise RuntimeError("The current round is not ready for feedback")
            round_count_before = len(project["experiment_campaign"]["rounds"])
            project = checked(
                client.post(
                    f"/api/v1/projects/{project['id']}/experiment-campaign/review"
                )
            )
            rounds = project["experiment_campaign"]["rounds"]
            added_round = len(rounds) > round_count_before
            reviewed_round = rounds[-2] if added_round else rounds[-1]
            print(
                json.dumps(
                    {
                        "project_id": project["id"],
                        "reviewed_round": reviewed_round["index"],
                        "result_summary": reviewed_round["result_summary"],
                        "feedback": reviewed_round["feedback"],
                        "campaign_status": project["experiment_campaign"]["status"],
                        "termination_reason": project["experiment_campaign"].get(
                            "termination_reason"
                        ),
                        "next_round": rounds[-1] if added_round else None,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return

        completed: list[dict[str, Any]] = []
        while len(completed) < arguments.runs:
            campaign = project["experiment_campaign"]
            if campaign["status"] == "awaiting_feedback":
                project = checked(
                    client.post(
                        f"/api/v1/projects/{project['id']}/experiment-campaign/review"
                    )
                )
                campaign = project["experiment_campaign"]
            if campaign["status"] != "active":
                break
            print(
                f"Executing real run {len(completed) + 1}/{arguments.runs} "
                f"for project {project['id']}...",
                flush=True,
            )
            response = checked(
                client.post(
                    f"/api/v1/projects/{project['id']}/experiment-campaign/execute-next",
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
            print(json.dumps(completed[-1], ensure_ascii=False, indent=2), flush=True)
            if execution["status"] != "succeeded":
                break

    print(
        json.dumps(
            {
                "project_id": project["id"],
                "dataset_digest": project["experiment_campaign"]["dataset_digest"],
                "campaign_id": project["experiment_campaign"]["id"],
                "real_runs": completed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not completed or completed[-1]["status"] != "succeeded":
        raise SystemExit(1)


def checked(response) -> dict[str, Any]:
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
    return response.json()


if __name__ == "__main__":
    main()
