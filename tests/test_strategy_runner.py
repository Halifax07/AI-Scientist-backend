import hashlib
import json
from pathlib import Path

import pytest

from fsad_scientist.domain.models import MethodImplementation
from fsad_scientist.experiments.runner import _ensure_within
from fsad_scientist.experiments.strategy_runner import (
    GeneratedStrategyRunner,
    assemble_strategy_file,
    build_smoke_fixture,
    run_strategy_smoke,
)

DETERMINISTIC_SOURCE = (
    "def select(candidate_ids, embeddings, k, seed):\n"
    "    generator = random.Random(seed)\n"
    "    return sorted(generator.sample(candidate_ids, k))\n"
)

NONDETERMINISTIC_SOURCE = (
    "def select(candidate_ids, embeddings, k, seed):\n"
    "    shuffled = sorted(candidate_ids, key=lambda file_id: random.random())\n"
    "    return shuffled[:k]\n"
)

EMPTY_OUTPUT_SOURCE = (
    "def select(candidate_ids, embeddings, k, seed):\n"
    "    return []\n"
)

OUTSIDE_POOL_SOURCE = (
    "def select(candidate_ids, embeddings, k, seed):\n"
    "    return ['outside_' + str(i) for i in range(k)]\n"
)


def _implementation(source: str) -> MethodImplementation:
    assembled = assemble_strategy_file(source)
    digest = hashlib.sha256(assembled.encode("utf-8")).hexdigest()
    return MethodImplementation(
        name="test_strategy_abc12345",
        hypothesis_id="hypothesis_x",
        source_code=source,
        code_digest=digest,
    )


class TestGeneratedStrategyRunner:
    def test_runner_executes_generated_strategy_out_of_process(self, tmp_path: Path) -> None:
        artifact_root = tmp_path / "artifacts"
        runner = GeneratedStrategyRunner(artifact_root)
        ids, vectors = build_smoke_fixture(count=6)
        implementation = _implementation(DETERMINISTIC_SOURCE)
        selected = runner.run(implementation, pool=ids, embeddings=vectors, k=3, seed=1)
        assert set(selected) <= set(ids)
        assert len(selected) == 3
        assert len(set(selected)) == 3

        digest_dir = artifact_root / "generated_methods" / implementation.code_digest
        assert (digest_dir / "strategy.py").is_file()
        call_dirs = [path for path in digest_dir.iterdir() if path.name.startswith("call_")]
        assert len(call_dirs) == 1
        payload = json.loads((call_dirs[0] / "input.json").read_text(encoding="utf-8"))
        assert payload["candidate_ids"] == ids
        assert payload["k"] == 3

    def test_runner_rejects_empty_output(self, tmp_path: Path) -> None:
        runner = GeneratedStrategyRunner(tmp_path / "artifacts")
        ids, vectors = build_smoke_fixture()
        with pytest.raises(ValueError, match="恰好 3 个"):
            runner.run(
                _implementation(EMPTY_OUTPUT_SOURCE), pool=ids, embeddings=vectors, k=3, seed=1
            )

    def test_runner_rejects_outside_pool_output(self, tmp_path: Path) -> None:
        runner = GeneratedStrategyRunner(tmp_path / "artifacts")
        ids, vectors = build_smoke_fixture()
        with pytest.raises(ValueError, match="候选池之外"):
            runner.run(
                _implementation(OUTSIDE_POOL_SOURCE), pool=ids, embeddings=vectors, k=3, seed=1
            )

    def test_runner_rejects_missing_embeddings(self, tmp_path: Path) -> None:
        runner = GeneratedStrategyRunner(tmp_path / "artifacts")
        ids, vectors = build_smoke_fixture(count=6)
        with pytest.raises(ValueError, match="embeddings are missing"):
            runner.run(
                _implementation(DETERMINISTIC_SOURCE),
                pool=ids,
                embeddings=dict(list(vectors.items())[1:]),
                k=2,
                seed=1,
            )

    def test_runner_rejects_tampered_source(self, tmp_path: Path) -> None:
        runner = GeneratedStrategyRunner(tmp_path / "artifacts")
        ids, vectors = build_smoke_fixture(count=6)
        implementation = _implementation(DETERMINISTIC_SOURCE)
        tampered = implementation.model_copy(
            update={"source_code": DETERMINISTIC_SOURCE.replace("sample", "choice")}
        )
        with pytest.raises(ValueError, match="摘要不一致"):
            runner.run(tampered, pool=ids, embeddings=vectors, k=2, seed=1)

    def test_bad_digest_rejected(self, tmp_path: Path) -> None:
        artifact_root = tmp_path / "artifacts"
        with pytest.raises(ValueError, match="hexadecimal"):
            GeneratedStrategyRunner._call_directory(artifact_root, "../escape")

    def test_path_escape_is_contained(self, tmp_path: Path) -> None:
        artifact_root = tmp_path / "artifacts"
        with pytest.raises(ValueError):
            _ensure_within(Path("C:/tmp/escape"), artifact_root)


class TestRunStrategySmoke:
    def test_smoke_passes_for_valid_strategy(self, tmp_path: Path) -> None:
        runner = GeneratedStrategyRunner(tmp_path / "artifacts")
        result = run_strategy_smoke(_implementation(DETERMINISTIC_SOURCE), runner)
        assert result.passed is True
        assert result.deterministic is True
        assert len(result.selected or []) == 3

    def test_smoke_detects_nondeterministic_strategy(self, tmp_path: Path) -> None:
        runner = GeneratedStrategyRunner(tmp_path / "artifacts")
        result = run_strategy_smoke(_implementation(NONDETERMINISTIC_SOURCE), runner)
        assert result.passed is False
        assert result.deterministic is False

    def test_smoke_reports_failure_instead_of_raising(self, tmp_path: Path) -> None:
        runner = GeneratedStrategyRunner(tmp_path / "artifacts")
        result = run_strategy_smoke(_implementation(EMPTY_OUTPUT_SOURCE), runner)
        assert result.passed is False
        assert "失败" in result.summary
