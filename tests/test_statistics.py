from fsad_scientist.domain.enums import RunStatus
from fsad_scientist.domain.models import ExperimentRun
from fsad_scientist.science.statistics import compare_paired_runs


def _run(strategy: str, seed: int, score: float) -> ExperimentRun:
    return ExperimentRun(
        plan_id="plan",
        hypothesis_id="hypothesis",
        protocol="pool_compression_m30",
        dataset="MVTec AD",
        category="bottle",
        detector="anomalydino",
        selection_strategy=strategy,
        shots=2,
        seed=seed,
        status=RunStatus.SUCCEEDED,
        metrics={"image_auroc": score},
        verified=True,
        result_source="real_executor",
    )


def test_paired_bootstrap_and_exact_permutation_are_deterministic():
    runs = []
    for seed, baseline in enumerate([0.70, 0.72, 0.74, 0.71, 0.73, 0.75]):
        runs.extend(
            [
                _run("random", seed, baseline),
                _run("k_center", seed, baseline + 0.04),
            ]
        )

    result = compare_paired_runs(
        runs,
        hypothesis_id="hypothesis",
        metric="image_auroc",
        treatment="k_center",
        control="random",
        bootstrap_samples=1000,
        permutation_samples=1000,
        random_seed=3,
    )

    assert result.pair_count == 6
    assert abs(result.mean_difference - 0.04) < 1e-12
    assert result.confidence_interval[0] > 0
    assert result.permutation_p_value == 2 / 64
