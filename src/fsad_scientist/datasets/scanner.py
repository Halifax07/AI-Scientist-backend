from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from fsad_scientist.datasets.models import (
    DatasetAuditIssue,
    DatasetFileRecord,
    DatasetManifest,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class MvtecDatasetScanner:
    """Build a content-addressed MVTec AD manifest and detect label leakage risks."""

    def scan(self, root: Path, *, dataset_name: str = "MVTec AD") -> DatasetManifest:
        resolved_root = root.expanduser().resolve()
        if not resolved_root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {resolved_root}")

        categories = sorted(
            path.name
            for path in resolved_root.iterdir()
            if path.is_dir() and (path / "train").is_dir() and (path / "test").is_dir()
        )
        if not categories:
            raise ValueError(
                "No MVTec categories found. Expected <root>/<category>/train and test"
            )

        files: list[DatasetFileRecord] = []
        issues: list[DatasetAuditIssue] = []
        for category in categories:
            self._scan_category(resolved_root, category, files, issues)

        self._audit_content_duplicates(files, issues)
        files.sort(key=lambda item: item.relative_path)
        counts = Counter(f"{item.split}_{item.kind}" for item in files)
        counts["files"] = len(files)
        for category in categories:
            counts[f"category:{category}"] = sum(item.category == category for item in files)

        digest_payload = {
            "dataset": dataset_name,
            "format": "mvtec_ad",
            "files": [item.model_dump(mode="json") for item in files],
        }
        digest = hashlib.sha256(
            json.dumps(digest_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return DatasetManifest(
            dataset=dataset_name,
            root=str(resolved_root),
            categories=categories,
            files=files,
            counts=dict(sorted(counts.items())),
            issues=issues,
            digest=digest,
        )

    @staticmethod
    def save(manifest: DatasetManifest, destination: Path) -> Path:
        destination = destination.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(f"{destination.suffix}.tmp")
        temporary.write_text(
            manifest.model_dump_json(indent=2),
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination

    def _scan_category(
        self,
        root: Path,
        category: str,
        files: list[DatasetFileRecord],
        issues: list[DatasetAuditIssue],
    ) -> None:
        category_root = root / category
        train_root = category_root / "train"
        test_root = category_root / "test"
        ground_truth_root = category_root / "ground_truth"

        train_groups = self._image_groups(train_root)
        non_good_train = {
            group: paths for group, paths in train_groups.items() if group != "good" and paths
        }
        if non_good_train:
            issues.append(
                DatasetAuditIssue(
                    severity="error",
                    code="TRAIN_CONTAINS_NON_GOOD",
                    message=f"{category}: training split contains non-good folders",
                    paths=[
                        self._relative(root, path)
                        for paths in non_good_train.values()
                        for path in paths
                    ],
                )
            )
        if not train_groups.get("good"):
            issues.append(
                DatasetAuditIssue(
                    severity="error",
                    code="EMPTY_NORMAL_TRAIN",
                    message=f"{category}: train/good has no supported images",
                    paths=[self._relative(root, train_root / "good")],
                )
            )
        for anomaly_type, paths in train_groups.items():
            files.extend(
                self._records(root, category, "train", anomaly_type, "image", paths)
            )

        test_groups = self._image_groups(test_root)
        if not test_groups.get("good"):
            issues.append(
                DatasetAuditIssue(
                    severity="warning",
                    code="MISSING_GOOD_TEST",
                    message=f"{category}: test/good is absent or empty; image AUROC is not valid",
                    paths=[self._relative(root, test_root / "good")],
                )
            )
        defect_test_paths: list[Path] = []
        for anomaly_type, paths in test_groups.items():
            files.extend(self._records(root, category, "test", anomaly_type, "image", paths))
            if anomaly_type != "good":
                defect_test_paths.extend(paths)

        mask_groups = self._image_groups(ground_truth_root)
        for anomaly_type, paths in mask_groups.items():
            files.extend(
                self._records(root, category, "ground_truth", anomaly_type, "mask", paths)
            )
        self._audit_masks(root, category, defect_test_paths, mask_groups, issues)

    @staticmethod
    def _image_groups(split_root: Path) -> dict[str, list[Path]]:
        if not split_root.is_dir():
            return {}
        return {
            directory.name: sorted(
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
            )
            for directory in sorted(split_root.iterdir())
            if directory.is_dir()
        }

    def _records(
        self,
        root: Path,
        category: str,
        split: str,
        anomaly_type: str,
        kind: str,
        paths: list[Path],
    ) -> list[DatasetFileRecord]:
        return [
            DatasetFileRecord(
                relative_path=self._relative(root, path),
                category=category,
                split=split,
                anomaly_type=anomaly_type,
                kind=kind,
                byte_size=path.stat().st_size,
                sha256=_sha256_file(path),
            )
            for path in paths
        ]

    def _audit_masks(
        self,
        root: Path,
        category: str,
        defect_test_paths: list[Path],
        mask_groups: dict[str, list[Path]],
        issues: list[DatasetAuditIssue],
    ) -> None:
        mask_keys = {
            (anomaly_type, path.stem.removesuffix("_mask"))
            for anomaly_type, paths in mask_groups.items()
            for path in paths
        }
        missing = [
            path
            for path in defect_test_paths
            if (path.parent.name, path.stem) not in mask_keys
        ]
        if missing:
            issues.append(
                DatasetAuditIssue(
                    severity="error",
                    code="MISSING_GROUND_TRUTH_MASK",
                    message=f"{category}: {len(missing)} anomalous test images lack masks",
                    paths=[self._relative(root, path) for path in missing],
                )
            )

    @staticmethod
    def _audit_content_duplicates(
        files: list[DatasetFileRecord],
        issues: list[DatasetAuditIssue],
    ) -> None:
        by_digest: dict[str, list[DatasetFileRecord]] = defaultdict(list)
        for item in files:
            if item.kind == "image":
                by_digest[item.sha256].append(item)
        for duplicates in by_digest.values():
            splits = {item.split for item in duplicates}
            if "train" in splits and "test" in splits:
                issues.append(
                    DatasetAuditIssue(
                        severity="error",
                        code="TRAIN_TEST_DUPLICATE",
                        message="Byte-identical image appears in train and test",
                        paths=[item.relative_path for item in duplicates],
                    )
                )

    @staticmethod
    def _relative(root: Path, path: Path) -> str:
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            return path.as_posix()


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
