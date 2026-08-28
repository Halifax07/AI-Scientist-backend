"""End-to-end functional test: one hypothesis that REQUIRES an AI-generated
selection strategy AND an AI-generated detector, driven through the complete
API workflow.

Flow: synthetic MVTec dataset -> create project -> advance to approval gate ->
generate strategy implementation + detector implementation (static validation
and behavioral smoke included) -> shortlist the hypothesis and register both
methods in the experiment plan -> approve -> dataset audit -> initialize the
adaptive campaign (generated detector as detector, generated strategy as
treatment, random as control) -> real execute-next loop -> round reviews ->
campaign verdict.

Deterministic mock runtime, CPU device, no API keys and no GPU required.
Purpose: functional validation only, not a scientific result.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from fsad_scientist.agents.mock_runtime import MockScientistRuntime
from fsad_scientist.api.app import create_app
from fsad_scientist.config import PROJECT_ROOT, Settings
from fsad_scientist.datasets.scanner import MvtecDatasetScanner
from fsad_scientist.domain.enums import HypothesisStatus
from fsad_scientist.domain.models import ComputeBudget, ProjectSpec
from fsad_scientist.repository import JsonProjectRepository


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _expect(condition: bool, message: str) -> None:
    if not condition:
        _emit({"event": "assertion_failed", "message": message})
        raise SystemExit(1)


def _checked(response) -> dict[str, Any]:
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
    return response.json()


def _build_synthetic_dataset(root: Path) -> None:
    """Deterministic bottle category: 40 train/good, 6 test/good, 4 test/broken."""
    if root.exists():
        shutil.rmtree(root)
    training = root / "bottle" / "train" / "good"
    test_good = root / "bottle" / "test" / "good"
    test_bad = root / "bottle" / "test" / "broken"
    masks = root / "bottle" / "ground_truth" / "broken"
    for directory in (training, test_good, test_bad, masks):
        directory.mkdir(parents=True, exist_ok=True)

    for index in range(40):
        base = (40 + (index % 8) * 6, 90 + (index % 11) * 4, 150 + (index % 20) * 3)
        image = Image.new("RGB", (224, 224), base)
        draw = ImageDraw.Draw(image)
        offset = index % 5
        draw.ellipse((50 + offset, 22, 174 + offset, 200), outline=(255, 255, 255), width=7)
        draw.ellipse((86, 58, 138, 112), fill=(255, 255, 255))
        image.save(training / f"{index:03}.png")

    for index in range(6):
        base = (55 + index * 3, 105 + index * 4, 170 - index * 4)
        image = Image.new("RGB", (224, 224), base)
        draw = ImageDraw.Draw(image)
        draw.ellipse((52, 24, 172, 198), outline=(250, 250, 250), width=7)
        image.save(test_good / f"{100 + index}.png")

    for index in range(4):
        base = (55 + index * 3, 105 + index * 4, 170 - index * 4)
        image = Image.new("RGB", (224, 224), base)
        draw = ImageDraw.Draw(image)
        draw.ellipse((52, 24, 172, 198), outline=(250, 250, 250), width=7)
        left = 88 + index * 4
        draw.rectangle((left, 88, left + 48, 136), fill=(230, 30, 30))
        image.save(test_bad / f"{200 + index}.png")
        mask = Image.new("L", (224, 224), 0)
        ImageDraw.Draw(mask).rectangle((left, 88, left + 48, 136), fill=255)
        mask.save(masks / f"{200 + index}_mask.png")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-executions", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=1800.0)
    arguments = parser.parse_args()

    smoke_root = PROJECT_ROOT / "artifacts" / "smoke" / "e2e_generated"
    dataset_root = smoke_root / "synthetic_mvtec"
    _build_synthetic_dataset(dataset_root)
    dataset = MvtecDatasetScanner().scan(dataset_root, dataset_name="MVTec AD")
    dataset_path = smoke_root / "dataset_manifest.json"
    MvtecDatasetScanner().save(dataset, dataset_path)

    storage_path = Path(tempfile.mkdtemp(prefix="fsad-e2e-ledger-"))
    app = create_app(
        settings=Settings(),
        storage_path=storage_path,
        runtime=MockScientistRuntime(),
    )
    repository = JsonProjectRepository(storage_path)

    with TestClient(app) as client:
        project = _checked(
            client.post(
                "/api/v1/projects",
                json={
                    "spec": ProjectSpec(
                        budget=ComputeBudget(max_experiments=12)
                    ).model_dump()
                },
            )
        )
        _emit(
            {
                "event": "project_created",
                "project_id": project["id"],
                "storage": str(storage_path),
            }
        )

        while project["stage"] != "awaiting_experiment_approval":
            project = _checked(client.post(f"/api/v1/projects/{project['id']}/advance"))
        _emit({"event": "reached_approval_gate", "stage": project["stage"]})

        hypothesis = next(
            item
            for item in project["hypotheses"]
            if item.get("analysis_contract") is not None
            and item["analysis_contract"]["kind"] == "query_adaptation"
        )
        _emit(
            {
                "event": "hypothesis_selected",
                "hypothesis_id": hypothesis["id"],
                "title": hypothesis["title"],
                "contract": hypothesis["analysis_contract"],
            }
        )

        # 1) AI-generated selection strategy (treatment of the hypothesis contract).
        project = _checked(
            client.post(
                f"/api/v1/projects/{project['id']}/experiment-methods/generate",
                json={"hypothesis_id": hypothesis["id"]},
            )
        )
        strategy = next(
            item
            for item in project["method_implementations"]
            if item["kind"] == "selection_strategy"
        )
        _expect(strategy["status"] == "validated", f"strategy status: {strategy['status']}")
        _expect(
            strategy["smoke_result"] is not None and strategy["smoke_result"]["passed"],
            "strategy smoke did not pass",
        )
        _emit(
            {
                "event": "strategy_generated",
                "name": strategy["name"],
                "status": strategy["status"],
                "code_digest": strategy["code_digest"],
            }
        )

        # 2) AI-generated detector for the same hypothesis.
        project = _checked(
            client.post(
                f"/api/v1/projects/{project['id']}/experiment-methods/generate-detector",
                json={
                    "hypothesis_id": hypothesis["id"],
                    "name_stem": "nearest_support",
                    "reference_description": "最近支持样本距离检测器（功能测试桩）",
                },
            )
        )
        detector = next(
            item for item in project["method_implementations"] if item["kind"] == "detector"
        )
        _expect(detector["status"] == "validated", f"detector status: {detector['status']}")
        _expect(
            detector["smoke_result"] is not None and detector["smoke_result"]["passed"],
            "detector smoke did not pass",
        )
        _emit(
            {
                "event": "detector_generated",
                "name": detector["name"],
                "status": detector["status"],
                "code_digest": detector["code_digest"],
            }
        )

        # 3) Register the hypothesis and both generated methods in the plan.
        draft = repository.get(project["id"])
        target = next(
            item for item in draft.hypotheses if item.id == hypothesis["id"]
        )
        target.status = HypothesisStatus.SHORTLISTED
        plan = draft.experiment_plan
        _expect(plan is not None, "experiment plan is missing")
        plan.hypothesis_ids = [hypothesis["id"]]
        if detector["name"] not in plan.detectors:
            plan.detectors.append(detector["name"])
        repository.save(draft)
        _emit(
            {
                "event": "plan_registered",
                "hypothesis_ids": plan.hypothesis_ids,
                "detectors": plan.detectors,
            }
        )

        # 4) Human approval gate; implementations move to approved here.
        project = _checked(
            client.post(
                f"/api/v1/projects/{project['id']}/approve",
                json={"approved_by": "e2e-functional-test"},
            )
        )
        project = _checked(client.get(f"/api/v1/projects/{project['id']}"))
        approved_status = {
            item["name"]: item["status"] for item in project["method_implementations"]
        }
        _expect(
            approved_status.get(strategy["name"]) == "approved"
            and approved_status.get(detector["name"]) == "approved",
            f"implementations not approved: {approved_status}",
        )
        _emit({"event": "plan_approved", "implementations": approved_status})

        # 5) Verified dataset audit.
        project = _checked(
            client.post(
                f"/api/v1/projects/{project['id']}/dataset/audit",
                json={"root": str(dataset_root.resolve()), "dataset_name": "MVTec AD"},
            )
        )
        manifest_path = project["dataset_audits"][-1]["manifest_path"]
        _emit({"event": "dataset_audited", "manifest_path": manifest_path})

        # 6) Adaptive campaign with the generated detector + generated strategy.
        project = _checked(
            client.post(
                f"/api/v1/projects/{project['id']}/experiment-campaign/initialize",
                json={
                    "dataset_manifest_path": manifest_path,
                    "hypothesis_id": hypothesis["id"],
                    "detector": detector["name"],
                    "device": "cpu",
                    "max_rounds": 3,
                    "max_runs": 8,
                },
            )
        )
        campaign = project["experiment_campaign"]
        _expect(campaign["detector"] == detector["name"], "campaign detector mismatch")
        _expect(campaign["treatment"] == strategy["name"], "campaign treatment mismatch")
        _expect(campaign["control"] == "random", "campaign control mismatch")
        queued = [
            run
            for run in project["runs"]
            if run["round_id"] is not None and run["status"] == "queued"
        ]
        _expect(
            all(run["detector"] == detector["name"] for run in queued),
            "queued run uses a non-generated detector",
        )
        _expect(
            {run["selection_strategy"] for run in queued} == {strategy["name"], "random"},
            f"queued strategies: {[run['selection_strategy'] for run in queued]}",
        )
        _emit(
            {
                "event": "campaign_initialized",
                "campaign_id": campaign["id"],
                "detector": campaign["detector"],
                "treatment": campaign["treatment"],
                "control": campaign["control"],
                "queued_runs": len(queued),
            }
        )

        # 7) Real experiment loop.
        completed: list[dict[str, Any]] = []
        reviewed = 0
        while True:
            campaign = project["experiment_campaign"]
            if campaign["status"] == "awaiting_feedback":
                project = _checked(
                    client.post(
                        f"/api/v1/projects/{project['id']}/experiment-campaign/review"
                    )
                )
                reviewed += 1
                rounds = campaign["rounds"] or []
                feedback = rounds[-2].get("feedback") if len(rounds) >= 2 else None
                _emit(
                    {
                        "event": "round_reviewed",
                        "review_count": reviewed,
                        "campaign_status": project["experiment_campaign"]["status"],
                        "decision": feedback.get("decision") if feedback else None,
                    }
                )
                continue
            if campaign["status"] != "active":
                _emit(
                    {
                        "event": "campaign_finished",
                        "status": campaign["status"],
                        "termination_reason": campaign.get("termination_reason"),
                    }
                )
                break
            if len(completed) >= arguments.max_executions:
                _emit({"event": "execution_cap_reached", "cap": arguments.max_executions})
                break

            response = _checked(
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
            run_id = response["run_id"]
            run = next(item for item in project["runs"] if item["id"] == run_id)
            metrics = (
                execution["normalized_result"]["metrics"]
                if execution["normalized_result"]
                else None
            )
            completed.append(
                {
                    "run_id": run_id,
                    "detector": run["detector"],
                    "selection_strategy": run["selection_strategy"],
                    "shots": run["shots"],
                    "seed": run["seed"],
                    "status": execution["status"],
                    "duration_seconds": execution["duration_seconds"],
                    "metrics": metrics,
                    "error": execution["error"],
                }
            )
            _emit(
                {
                    "event": "run_executed",
                    **{k: v for k, v in completed[-1].items() if k != "metrics"},
                }
            )
            if len(completed) >= 2 and all(
                item["status"] != "succeeded" for item in completed[-2:]
            ):
                _emit({"event": "stopped_after_consecutive_failures"})
                break

        succeeded = [item for item in completed if item["status"] == "succeeded"]
        failed = [item for item in completed if item["status"] != "succeeded"]
        _expect(len(succeeded) >= 1, "no experiment run succeeded")
        _expect(
            all(
                item["metrics"] is not None and 0.5 <= item["metrics"]["image_auroc"] <= 1.0
                for item in succeeded
            ),
            "a succeeded run has missing or out-of-range image_auroc",
        )
        strategies_used = {item["selection_strategy"] for item in completed}
        _expect(
            strategy["name"] in strategies_used and "random" in strategies_used,
            f"executed strategies {strategies_used} do not cover treatment and control",
        )
        _expect(
            all(item["detector"] == detector["name"] for item in completed),
            "an executed run did not use the generated detector",
        )

        # 8) Verdict.
        final = project["experiment_campaign"]
        verdict: dict[str, Any] = {"campaign_status": final["status"]}
        if final["status"] == "completed":
            project = _checked(
                client.post(f"/api/v1/projects/{project['id']}/results/finalize")
            )
            project = _checked(client.post(f"/api/v1/projects/{project['id']}/advance"))
            campaign_hypothesis = next(
                item
                for item in project["hypotheses"]
                if item["id"] == final["hypothesis_id"]
            )
            findings = [
                item
                for item in project.get("findings", [])
                if item["hypothesis_id"] == final["hypothesis_id"]
            ]
            verdict.update(
                {
                    "stage": project["stage"],
                    "hypothesis_status": campaign_hypothesis["status"],
                    "finding": findings[0] if findings else None,
                }
            )
        else:
            verdict["note"] = "campaign did not reach completed; no verdict computed"

        strategy_call_evidence = _experiment_strategy_calls(
            PROJECT_ROOT / "artifacts", strategy["code_digest"]
        )
        _emit(
            {
                "event": "summary",
                "purpose": "e2e_generated_methods_functional_test",
                "project_id": project["id"],
                "storage": str(storage_path),
                "hypothesis_id": hypothesis["id"],
                "generated_strategy": strategy["name"],
                "generated_detector": detector["name"],
                "executed_runs": completed,
                "succeeded": len(succeeded),
                "failed": len(failed),
                "verdict": verdict,
                "strategy_experiment_calls": strategy_call_evidence,
            }
        )
        _expect(final["status"] == "completed", f"campaign status: {final['status']}")
        _expect(verdict.get("finding") is not None, "no analysis finding was produced")


def _experiment_strategy_calls(artifact_root: Path, code_digest: str) -> list[dict[str, Any]]:
    """Calls with a full-size (>=30) candidate pool are experiment-time calls,
    distinct from the small verification calls of the behavioral smoke."""
    evidence: list[dict[str, Any]] = []
    call_root = artifact_root / "generated_methods" / code_digest
    if not call_root.is_dir():
        return evidence
    for directory in sorted(call_root.glob("call_*")):
        input_path = directory / "input.json"
        if not input_path.is_file():
            continue
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if len(payload.get("candidate_ids", [])) >= 30:
            evidence.append(
                {
                    "call": directory.name,
                    "candidate_pool_size": len(payload["candidate_ids"]),
                    "k": payload.get("k"),
                }
            )
    return evidence


if __name__ == "__main__":
    main()
