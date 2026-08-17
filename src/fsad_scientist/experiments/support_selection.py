from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence

from fsad_scientist.datasets.models import DatasetManifest
from fsad_scientist.experiments.models import SupportSetManifest


def random_select(file_ids: Sequence[str], *, shots: int, seed: int) -> list[str]:
    _validate_shots(file_ids, shots)
    generator = random.Random(seed)
    return sorted(generator.sample(list(file_ids), shots))


def k_center_select(
    embeddings: Mapping[str, Sequence[float]],
    *,
    shots: int,
) -> list[str]:
    """Deterministic farthest-first traversal over candidate normal embeddings."""

    file_ids = sorted(embeddings)
    _validate_shots(file_ids, shots)
    _validate_embeddings(embeddings)

    centroid = [
        sum(embeddings[file_id][dimension] for file_id in file_ids) / len(file_ids)
        for dimension in range(len(embeddings[file_ids[0]]))
    ]
    first = max(file_ids, key=lambda file_id: (_distance(embeddings[file_id], centroid), file_id))
    selected = [first]

    while len(selected) < shots:
        remaining = [file_id for file_id in file_ids if file_id not in selected]
        next_id = max(
            remaining,
            key=lambda file_id: (
                min(
                    _distance(embeddings[file_id], embeddings[selected_id])
                    for selected_id in selected
                ),
                file_id,
            ),
        )
        selected.append(next_id)
    return selected


def build_support_manifest(
    *,
    dataset: str,
    category: str,
    protocol: str,
    strategy: str,
    shots: int,
    seed: int,
    candidate_pool_files: list[str],
    selected_files: list[str],
    feature_extractor: str,
    selection_metadata: dict[str, float | int | str] | None = None,
) -> SupportSetManifest:
    if not set(selected_files) <= set(candidate_pool_files):
        raise ValueError("selected files must be members of the candidate normal pool")
    if len(selected_files) != shots or len(set(selected_files)) != shots:
        raise ValueError("selected files must contain exactly K unique items")

    payload = {
        "dataset": dataset,
        "category": category,
        "protocol": protocol,
        "strategy": strategy,
        "shots": shots,
        "seed": seed,
        "candidate_pool_files": sorted(candidate_pool_files),
        "selected_files": sorted(selected_files),
        "feature_extractor": feature_extractor,
        "selection_metadata": selection_metadata or {},
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return SupportSetManifest(**payload, digest=digest)


def plan_support_set(
    dataset: DatasetManifest,
    *,
    category: str,
    protocol: str,
    strategy: str,
    shots: int,
    seed: int,
    candidate_pool_size: int = 30,
    embeddings: Mapping[str, Sequence[float]] | None = None,
    feature_extractor: str = "none",
) -> SupportSetManifest:
    """Freeze a support set while preserving strict-K and pool-compression semantics."""

    all_normal = dataset.support_candidates(category)
    _validate_shots(all_normal, shots)
    if protocol == "strict_k_shot":
        if strategy != "random":
            raise ValueError("strict_k_shot does not permit selection from a larger pool")
        selected = random_select(all_normal, shots=shots, seed=seed)
        pool = selected
        extractor = "none"
    elif protocol.startswith("pool_compression"):
        if candidate_pool_size < shots:
            raise ValueError("candidate_pool_size cannot be smaller than shots")
        pool_size = min(candidate_pool_size, len(all_normal))
        pool = random_select(all_normal, shots=pool_size, seed=seed)
        if strategy == "random":
            selected = random_select(pool, shots=shots, seed=_selection_seed(seed))
        elif strategy == "k_center":
            if embeddings is None:
                raise ValueError("k_center requires precomputed normal-image embeddings")
            missing = sorted(set(pool) - set(embeddings))
            if missing:
                raise ValueError(f"embeddings are missing {len(missing)} candidate files")
            selected = k_center_select(
                {file_id: embeddings[file_id] for file_id in pool},
                shots=shots,
            )
        else:
            raise ValueError(f"unsupported selection strategy: {strategy}")
        extractor = feature_extractor
    else:
        raise ValueError(f"unsupported few-shot protocol: {protocol}")

    selection_metadata: dict[str, float | int | str] = {
        "dataset_manifest_digest": dataset.digest,
        "candidate_pool_size": len(pool),
    }
    if protocol.startswith("pool_compression") and embeddings is not None:
        selection_metadata.update(
            _selection_geometry(
                {file_id: embeddings[file_id] for file_id in pool},
                selected,
            )
        )
    return build_support_manifest(
        dataset=dataset.dataset,
        category=category,
        protocol=protocol,
        strategy=strategy,
        shots=shots,
        seed=seed,
        candidate_pool_files=pool,
        selected_files=selected,
        feature_extractor=extractor,
        selection_metadata=selection_metadata,
    )


def _validate_shots(file_ids: Sequence[str], shots: int) -> None:
    if shots < 1:
        raise ValueError("shots must be positive")
    if shots > len(file_ids):
        raise ValueError("shots cannot exceed candidate pool size")
    if len(set(file_ids)) != len(file_ids):
        raise ValueError("candidate file ids must be unique")


def _validate_embeddings(embeddings: Mapping[str, Sequence[float]]) -> None:
    if not embeddings:
        raise ValueError("embeddings cannot be empty")
    dimensions = {len(vector) for vector in embeddings.values()}
    if len(dimensions) != 1 or next(iter(dimensions)) == 0:
        raise ValueError("all embeddings must have one shared non-zero dimension")
    if any(not math.isfinite(value) for vector in embeddings.values() for value in vector):
        raise ValueError("embeddings must be finite")


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(a, b, strict=True)))


def _selection_seed(seed: int) -> int:
    payload = f"support-selection:{seed}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _selection_geometry(
    embeddings: Mapping[str, Sequence[float]],
    selected: list[str],
) -> dict[str, float]:
    nearest_distances = [
        min(_distance(vector, embeddings[selected_id]) for selected_id in selected)
        for vector in embeddings.values()
    ]
    pairwise = [
        _distance(embeddings[left], embeddings[right])
        for index, left in enumerate(selected)
        for right in selected[index + 1 :]
    ]
    metadata = {
        "coverage_radius": max(nearest_distances),
        "mean_coverage_distance": sum(nearest_distances) / len(nearest_distances),
        "selected_pairwise_distance": sum(pairwise) / len(pairwise) if pairwise else 0.0,
    }
    try:
        import numpy as np
    except ImportError:
        return metadata
    matrix = np.asarray([embeddings[file_id] for file_id in selected], dtype=float)
    matrix -= matrix.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    energy = singular_values**2
    total = float(energy.sum())
    if total <= 0:
        metadata["effective_rank"] = 0.0
    else:
        probabilities = energy[energy > 0] / total
        metadata["effective_rank"] = float(np.exp(-(probabilities * np.log(probabilities)).sum()))
    return metadata
