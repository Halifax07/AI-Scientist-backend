"""Out-of-process execution for AI-generated support-set selection strategies.

Generated code is NEVER imported into this process. Each strategy runs as an
argv-only Python subprocess (no shell), inherits only the same environment
allowlist as detector runs, reads one JSON input file and writes one JSON
output file, all inside a digest-addressed directory under the artifact root.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import uuid
from pathlib import Path

from fsad_scientist.domain.models import MethodImplementation, MethodSmokeResult
from fsad_scientist.experiments.runner import ExperimentRunner, _ensure_within

STRATEGY_TEMPLATE_HEADER = '''"""Generated support-set selection strategy.
Digest-audited; do not edit.
"""
import collections
import functools
import hashlib
import itertools
import json
import math
import random
import statistics
import sys

import numpy as np
'''

STRATEGY_TEMPLATE_FOOTER = '''


def _main() -> None:
    if len(sys.argv) != 5 or sys.argv[1] != "--input" or sys.argv[3] != "--output":
        raise SystemExit("usage: strategy.py --input input.json --output output.json")
    with open(sys.argv[2], encoding="utf-8") as handle:
        payload = json.load(handle)
    selected = select(
        payload["candidate_ids"],
        payload["embeddings"],
        payload["k"],
        payload["seed"],
    )
    with open(sys.argv[4], "w", encoding="utf-8") as handle:
        json.dump({"selected": selected}, handle, ensure_ascii=False)


if __name__ == "__main__":
    _main()
'''


def assemble_strategy_file(source_code: str) -> str:
    """Wrap the validated select function in the trusted import/IO template."""

    return STRATEGY_TEMPLATE_HEADER + "\n" + source_code.rstrip() + STRATEGY_TEMPLATE_FOOTER


def build_smoke_fixture(*, count: int = 10, dimension: int = 8, seed: int = 7):
    """Deterministic in-memory pool + embeddings for smoke testing (pure stdlib)."""

    generator = random.Random(seed)
    ids = [f"fixture_{index:03d}" for index in range(count)]
    vectors = {
        file_id: [round(generator.random(), 6) for _ in range(dimension)] for file_id in ids
    }
    return ids, vectors


class GeneratedStrategyRunner:
    """Execute one registered strategy implementation in an isolated subprocess."""

    def __init__(self, artifact_root: Path, *, timeout_seconds: float = 60.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.artifact_root = artifact_root.expanduser().resolve()
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _call_directory(artifact_root: Path, code_digest: str) -> Path:
        if len(code_digest) != 64 or any(char not in "0123456789abcdef" for char in code_digest):
            raise ValueError("code_digest must be a 64-character hexadecimal string")
        call_dir = (
            artifact_root
            / "generated_methods"
            / code_digest
            / f"call_{uuid.uuid4().hex[:8]}"
        )
        _ensure_within(call_dir, artifact_root)
        return call_dir

    def run(
        self,
        implementation: MethodImplementation,
        *,
        pool: list[str],
        embeddings: dict[str, list[float]],
        k: int,
        seed: int,
    ) -> list[str]:
        if k < 1 or k > len(pool):
            raise ValueError("k must be within the candidate pool size")
        missing = set(pool) - set(embeddings)
        if missing:
            raise ValueError(f"embeddings are missing {len(missing)} candidate files")

        call_dir = self._call_directory(self.artifact_root, implementation.code_digest)
        call_dir.mkdir(parents=True, exist_ok=True)
        self._write_strategy_once(implementation)
        input_path = call_dir / "input.json"
        output_path = call_dir / "output.json"
        _write_json(
            input_path,
            json.dumps(
                {
                    "candidate_ids": list(pool),
                    "embeddings": embeddings,
                    "k": k,
                    "seed": seed,
                },
                ensure_ascii=False,
            ),
        )
        environment = ExperimentRunner._build_environment({})
        command = [
            ExperimentRunner._resolve_executable("python"),
            str(call_dir.parent / "strategy.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=str(call_dir),
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError(f"策略执行超过 {self.timeout_seconds:.1f} 秒被终止") from exc
        if completed.returncode != 0:
            tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-8:]
            raise ValueError(
                f"策略进程退出码 {completed.returncode}；输出末尾：{chr(10).join(tail)}"
            )
        return self._parse_selection(output_path, pool=pool, k=k)

    def _write_strategy_once(self, implementation: MethodImplementation) -> None:
        assembled = assemble_strategy_file(implementation.source_code)
        digest = hashlib.sha256(assembled.encode("utf-8")).hexdigest()
        if digest != implementation.code_digest:
            raise ValueError(
                "策略源码与注册摘要不一致；请先重新生成并校验 implementation"
            )
        strategy_path = (
            self.artifact_root / "generated_methods" / implementation.code_digest / "strategy.py"
        )
        if strategy_path.exists():
            return
        strategy_path.parent.mkdir(parents=True, exist_ok=True)
        _write_text_atomic(strategy_path, assembled)

    @staticmethod
    def _parse_selection(output_path: Path, *, pool: list[str], k: int) -> list[str]:
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"策略输出无法解析为 JSON：{exc}") from exc
        selected = payload.get("selected")
        if not isinstance(selected, list) or not all(
            isinstance(item, str) for item in selected
        ):
            raise ValueError("策略输出必须包含字符串列表字段 selected")
        outside = sorted(set(selected) - set(pool))
        if outside:
            raise ValueError(f"策略返回了候选池之外的文件：{outside[:5]}")
        if len(selected) != k:
            raise ValueError(f"策略应返回恰好 {k} 个文件，实际返回 {len(selected)} 个")
        if len(set(selected)) != len(selected):
            raise ValueError("策略返回的文件存在重复")
        return selected


def run_strategy_smoke(
    implementation: MethodImplementation,
    runner: GeneratedStrategyRunner,
) -> MethodSmokeResult:
    """Behavioral smoke test; never raises, only reports."""

    ids, vectors = build_smoke_fixture()
    try:
        first = runner.run(implementation, pool=ids, embeddings=vectors, k=3, seed=42)
        second = runner.run(implementation, pool=ids, embeddings=vectors, k=3, seed=42)
        third = runner.run(implementation, pool=ids, embeddings=vectors, k=1, seed=42)
    except Exception as exc:  # smoke must always return a result, not crash the pipeline
        return MethodSmokeResult(passed=False, summary=f"冒烟执行失败：{exc}")
    deterministic = first == second
    passed = deterministic and len(third) == 1
    summary = (
        "冒烟通过：候选池约束、长度与确定性检查全部满足。"
        if passed
        else "冒烟未通过：确定性或输出契约检查失败。"
    )
    return MethodSmokeResult(
        passed=passed,
        summary=summary,
        selected=first,
        deterministic=deterministic,
    )


def _write_json(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(path, content)


def _write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
