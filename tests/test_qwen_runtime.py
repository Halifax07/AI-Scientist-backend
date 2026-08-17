from fsad_scientist.agents.qwen_runtime import _normalize_hypothesis_payload


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
