"""Template, assembly and smoke execution for AI-generated anomaly detectors.

Generated code is NEVER imported into this process. Each detector runs as an
argv-only Python subprocess through the existing hardened ``ExperimentRunner``:
the trusted header forces offline model loading before any LLM import, the
LLM-authored section contributes only imports plus the ``anomaly_score`` core
function, and the trusted footer handles data loading, label derivation from
ground-truth masks, AUROC computation and the metrics.json output contract.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

from fsad_scientist.datasets.scanner import MvtecDatasetScanner
from fsad_scientist.datasets.synthetic import build_synthetic_mvtec_smoke_dataset
from fsad_scientist.datasets.view import DatasetViewBuilder
from fsad_scientist.domain.models import ExperimentRun, MethodImplementation, MethodSmokeResult
from fsad_scientist.experiments.runner import ExperimentRunner
from fsad_scientist.experiments.support_selection import plan_support_set

DETECTOR_TEMPLATE_HEADER = '''"""Generated anomaly detector; digest-audited, do not edit."""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
'''

DETECTOR_TEMPLATE_FOOTER = '''


def _load_support_images(data_root, category):
    import json
    from pathlib import Path

    from PIL import Image
    import numpy as np

    manifest_path = Path(data_root) / "fsad_support_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"support manifest missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    images = []
    for relative_path in payload.get("selected_files", []):
        path = Path(data_root) / Path(relative_path)
        if path.is_file() and "train" in path.parts:
            with Image.open(path) as opened:
                images.append(np.asarray(opened.convert("RGB")))
    return images


def _iter_test_images(data_root, category):
    from pathlib import Path

    from PIL import Image
    import numpy as np

    test_root = Path(data_root) / category / "test"
    mask_root = Path(data_root) / category / "ground_truth"
    for image_path in sorted(test_root.rglob("*.png")):
        with Image.open(image_path) as opened:
            image = np.asarray(opened.convert("RGB"))
        defect = image_path.parent.name
        mask_path = mask_root / defect / f"{image_path.stem}_mask.png"
        label = 0
        if mask_path.is_file():
            with Image.open(mask_path) as opened:
                mask = np.asarray(opened)
            label = 1 if float(mask.sum()) > 0 else 0
        yield image_path.name, image, label


def _main():
    import json
    import sys
    from pathlib import Path

    from sklearn.metrics import roc_auc_score

    arguments = {}
    for index in range(1, len(sys.argv), 2):
        if not sys.argv[index].startswith("--"):
            raise SystemExit(f"unexpected argument: {sys.argv[index]}")
        arguments[sys.argv[index][2:]] = sys.argv[index + 1]
    data_root = arguments.get("data_root")
    category = arguments.get("category")
    output = arguments.get("output")
    if not data_root or not category or not output:
        raise SystemExit(
            "usage: detector.py --data_root DIR --category NAME --output DIR"
        )
    seed = int(arguments.get("seed", "0"))
    support_images = _load_support_images(data_root, category)
    scores = []
    labels = []
    names = []
    for name, image, label in _iter_test_images(data_root, category):
        names.append(name)
        scores.append(anomaly_score(image, support_images, seed))
        labels.append(label)
    if len(set(labels)) < 2:
        image_auroc = 0.5
        image_ap = 0.5
    else:
        image_auroc = float(roc_auc_score(labels, scores))
        try:
            from sklearn.metrics import average_precision_score

            image_ap = float(average_precision_score(labels, scores))
        except ImportError:
            image_ap = image_auroc
    payload = {
        "image_auroc": round(image_auroc, 9),
        "image_ap": round(image_ap, 9),
        "scores": [round(float(score), 9) for score in scores],
        "test_images": names,
    }
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    _main()
'''


def assemble_detector_file(source_code: str) -> str:
    """Wrap the validated LLM section in the trusted offline/IO template."""

    return DETECTOR_TEMPLATE_HEADER + "\n" + source_code.rstrip() + DETECTOR_TEMPLATE_FOOTER


def _write_detector_once(artifact_root: Path, implementation: MethodImplementation) -> Path:
    assembled = assemble_detector_file(implementation.source_code)
    digest = hashlib.sha256(assembled.encode("utf-8")).hexdigest()
    if digest != implementation.code_digest:
        raise ValueError("检测器源码与注册摘要不一致；请先重新生成并校验 implementation")
    detector_path = artifact_root / "generated_methods" / digest / "detector.py"
    if detector_path.exists():
        return detector_path
    detector_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = detector_path.with_suffix(".py.tmp")
    temporary.write_text(assembled, encoding="utf-8")
    temporary.replace(detector_path)
    return detector_path


async def run_detector_smoke(
    implementation: MethodImplementation,
    artifact_root: Path,
    *,
    timeout_seconds: float = 1800.0,
) -> MethodSmokeResult:
    """Behavioral smoke through the real execution path; never raises.

    Two fresh executions on a tiny synthetic MVTec view must both succeed,
    report image_auroc in [0.5, 1] (upper bound enforced by the normalizer),
    and agree within 1e-9 (determinism gate).
    """

    try:
        artifact_root = artifact_root.expanduser().resolve()
        detector_path = _write_detector_once(artifact_root, implementation)
        if implementation.artifact_path is None:
            implementation.artifact_path = str(detector_path.resolve())
        dataset_root = artifact_root / "smoke" / "generated_detector" / "synthetic_mvtec"
        build_synthetic_mvtec_smoke_dataset(dataset_root)
        dataset = MvtecDatasetScanner().scan(dataset_root, dataset_name="MVTec AD")
        support = plan_support_set(
            dataset,
            category="bottle",
            protocol="strict_k_shot",
            strategy="random",
            shots=2,
            seed=11,
        )
        view = DatasetViewBuilder(artifact_root).build(dataset, support)

        try:
            import torch  # type: ignore[import-not-found]

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        from fsad_scientist.experiments.adapters import build_generated_detector_command

        values: list[float] = []
        score_vectors: list[list[float]] = []
        runner = ExperimentRunner(
            project_root=artifact_root,
            artifact_root=artifact_root,
        )
        for _ in range(2):
            run = ExperimentRun(
                plan_id="smoke_plan",
                hypothesis_id="smoke_hypothesis",
                protocol="strict_k_shot",
                dataset="MVTec AD",
                category="bottle",
                detector=implementation.name,
                selection_strategy="random",
                shots=2,
                seed=11,
            )
            output_dir = (
                artifact_root
                / "generated_methods"
                / implementation.code_digest
                / f"smoke_{uuid.uuid4().hex[:8]}"
            )
            command = build_generated_detector_command(
                run,
                implementation,
                dataset_view=Path(view.view_root),
                output_dir=output_dir,
                device=device,
            )
            record = await runner.execute(
                run,
                command,
                output_dir=output_dir,
                dataset_view=view,
                support_manifest=support,
                timeout_seconds=timeout_seconds,
            )
            normalized = record.normalized_result
            if record.status != "succeeded" or normalized is None:
                raise ValueError(f"冒烟执行未成功：{record.status} {record.error or ''}")
            values.append(normalized.metrics["image_auroc"])
            metrics_payload = json.loads(
                (output_dir / "metrics.json").read_text(encoding="utf-8")
            )
            score_vectors.append(
                [float(item) for item in metrics_payload.get("scores", [])]
            )
    except Exception as exc:  # smoke must always return a result, not crash the pipeline
        return MethodSmokeResult(passed=False, summary=f"冒烟执行失败：{exc}")

    deterministic = _vectors_agree(score_vectors[0], score_vectors[1])
    passed = deterministic and 0.5 <= values[0] <= 1.0
    summary = (
        "冒烟通过：两次真实执行成功，AUROC 在合理区间且结果确定。"
        if passed
        else "冒烟未通过：AUROC 合理性或确定性检查失败。"
    )
    return MethodSmokeResult(
        passed=passed,
        summary=summary,
        selected=None,
        deterministic=deterministic,
    )


def _vectors_agree(first: list[float], second: list[float]) -> bool:
    if not first or len(first) != len(second):
        return False
    return all(abs(left - right) <= 1e-9 for left, right in zip(first, second, strict=True))
