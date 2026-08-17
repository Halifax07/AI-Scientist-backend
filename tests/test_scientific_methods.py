from fsad_scientist.experiments.support_selection import (
    build_support_manifest,
    k_center_select,
    plan_support_set,
)
from fsad_scientist.science.experiment_tree import (
    ExperimentNode,
    ExperimentPhase,
    NodeStatus,
    ProgressiveExperimentTree,
)
from fsad_scientist.science.hypothesis_tournament import EloTournament, PairwiseJudgment


def test_elo_tournament_orders_pairwise_winner_first():
    tournament = EloTournament(["h1", "h2", "h3"])
    tournament.apply(PairwiseJudgment("h2", "h1", 1.0, confidence=0.9))
    tournament.apply(PairwiseJudgment("h2", "h3", 1.0, confidence=0.8))

    assert tournament.ranked()[0][0] == "h2"
    assert len(tournament.history) == 2


def test_experiment_tree_selects_information_per_cost_and_limits_debugging():
    tree = ProgressiveExperimentTree(max_nodes=4, max_debug_depth=1)
    tree.add(
        ExperimentNode(
            id="cheap_falsification",
            hypothesis_id="h1",
            phase=ExperimentPhase.FEASIBILITY,
            parent_id=None,
            information_gain=0.8,
            falsification_value=0.9,
            estimated_cost=0.2,
        )
    )
    tree.add(
        ExperimentNode(
            id="expensive_accuracy",
            hypothesis_id="h1",
            phase=ExperimentPhase.MAIN_STUDY,
            parent_id=None,
            information_gain=0.9,
            falsification_value=0.4,
            estimated_cost=1.0,
        )
    )

    assert tree.select_next().id == "cheap_falsification"
    assert tree.record_failure("cheap_falsification", "temporary error") == NodeStatus.PENDING
    assert tree.record_failure("cheap_falsification", "same error") == NodeStatus.FAILED


def test_k_center_manifest_is_deterministic_and_auditable():
    embeddings = {
        "a.png": [0.0, 0.0],
        "b.png": [0.1, 0.0],
        "c.png": [1.0, 1.0],
        "d.png": [0.9, 1.0],
    }
    selected = k_center_select(embeddings, shots=2)
    manifest = build_support_manifest(
        dataset="MVTec AD",
        category="bottle",
        protocol="pool_compression_m30",
        strategy="k_center",
        shots=2,
        seed=0,
        candidate_pool_files=list(embeddings),
        selected_files=selected,
        feature_extractor="dinov2_vitg14",
    )

    assert len(selected) == 2
    assert len(manifest.digest) == 64
    assert set(manifest.selected_files) <= set(manifest.candidate_pool_files)


def test_pool_compression_records_geometry_without_test_data(tmp_path):
    from fsad_scientist.datasets.scanner import MvtecDatasetScanner

    root = tmp_path / "dataset"
    for index in range(4):
        path = root / "bottle" / "train" / "good" / f"{index:03}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"train-{index}".encode())
    for relative, payload in [
        ("bottle/test/good/100.png", b"good"),
        ("bottle/test/broken/101.png", b"broken"),
        ("bottle/ground_truth/broken/101_mask.png", b"mask"),
    ]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    dataset = MvtecDatasetScanner().scan(root)
    candidates = dataset.support_candidates("bottle")
    embeddings = {
        file_id: [float(index), float(index % 2)]
        for index, file_id in enumerate(candidates)
    }

    support = plan_support_set(
        dataset,
        category="bottle",
        protocol="pool_compression_m30",
        strategy="k_center",
        shots=2,
        seed=0,
        candidate_pool_size=4,
        embeddings=embeddings,
        feature_extractor="test-dino",
    )

    assert support.selection_metadata["coverage_radius"] >= 0
    assert support.selection_metadata["effective_rank"] >= 1
