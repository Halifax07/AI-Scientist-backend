from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import time
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

from fsad_scientist.datasets.models import DatasetManifest, DatasetViewManifest
from fsad_scientist.experiments.models import SupportSetManifest

_PUBLISH_RETRY_ATTEMPTS = 3
_PUBLISH_RETRY_DELAY_SECONDS = 0.05


class DatasetViewBuilder:
    """Materialize a detector-visible view containing only the frozen support set."""

    def __init__(self, artifact_root: Path) -> None:
        self.artifact_root = artifact_root.expanduser().resolve()

    def build(
        self,
        dataset: DatasetManifest,
        support: SupportSetManifest,
        *,
        include_test: bool = True,
    ) -> DatasetViewManifest:
        if not dataset.is_valid:
            raise ValueError("dataset manifest has audit errors")
        if support.dataset.casefold() != dataset.dataset.casefold():
            raise ValueError("support manifest dataset does not match dataset manifest")
        if support.category not in dataset.categories:
            raise ValueError(f"unknown category in support manifest: {support.category}")

        eligible = set(dataset.support_candidates(support.category))
        selected = set(support.selected_files)
        if not selected <= eligible:
            invalid = sorted(selected - eligible)
            raise ValueError(
                "support manifest contains files outside train/good: " + ", ".join(invalid)
            )

        target = (
            self.artifact_root
            / "dataset_views"
            / dataset.digest[:16]
            / support.digest
        ).resolve()
        _ensure_within(target, self.artifact_root)
        manifest_path = target / "fsad_view_manifest.json"
        if target.exists():
            return self._load_existing(manifest_path, dataset, support)

        temporary = target.with_name(f".{target.name}.building-{uuid4().hex[:8]}")
        _ensure_within(temporary, self.artifact_root)
        temporary.mkdir(parents=True, exist_ok=False)
        source_root = Path(dataset.root).resolve()
        link_modes: set[str] = set()
        materialized: list[str] = []
        try:
            records = [
                item
                for item in dataset.files
                if item.category == support.category
                and (
                    item.relative_path in selected
                    or (include_test and item.split in {"test", "ground_truth"})
                )
            ]
            for record in records:
                source = (source_root / Path(record.relative_path)).resolve()
                _ensure_within(source, source_root)
                if not source.is_file():
                    raise FileNotFoundError(f"manifest source file is missing: {source}")
                destination = temporary / Path(record.relative_path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                link_modes.add(_materialize_file(source, destination))
                materialized.append(record.relative_path)

            payload = {
                "dataset_digest": dataset.digest,
                "support_manifest_digest": support.digest,
                "category": support.category,
                "files": sorted(materialized),
            }
            digest = hashlib.sha256(
                json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            mode = next(iter(link_modes)) if len(link_modes) == 1 else "mixed"
            view = DatasetViewManifest(
                dataset_digest=dataset.digest,
                support_manifest_digest=support.digest,
                category=support.category,
                source_root=str(source_root),
                view_root=str(target),
                materialization=mode,
                files=sorted(materialized),
                digest=digest,
            )
            (temporary / "fsad_support_manifest.json").write_text(
                support.model_dump_json(indent=2),
                encoding="utf-8",
            )
            (temporary / "fsad_view_manifest.json").write_text(
                view.model_dump_json(indent=2),
                encoding="utf-8",
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            for attempt in range(_PUBLISH_RETRY_ATTEMPTS):
                try:
                    temporary.replace(target)
                except PermissionError:
                    # Another request may have published the same immutable
                    # view between the initial existence check and this
                    # rename. Windows reports that directory replacement as
                    # PermissionError.
                    if target.is_dir():
                        try:
                            existing = self._load_existing(
                                manifest_path, dataset, support
                            )
                        except (FileExistsError, ValueError):
                            pass
                        else:
                            _cleanup_temporary(temporary)
                            return existing
                    if target.exists() or attempt == _PUBLISH_RETRY_ATTEMPTS - 1:
                        raise
                    time.sleep(_PUBLISH_RETRY_DELAY_SECONDS * (2**attempt))
                else:
                    return view
            raise AssertionError("unreachable publish retry state")
        except BaseException:
            _cleanup_temporary(temporary)
            raise

    @staticmethod
    def _load_existing(
        manifest_path: Path,
        dataset: DatasetManifest,
        support: SupportSetManifest,
    ) -> DatasetViewManifest:
        if not manifest_path.is_file():
            raise FileExistsError(
                f"dataset view exists without an audit manifest: {manifest_path.parent}"
            )
        view = DatasetViewManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        if (
            view.dataset_digest != dataset.digest
            or view.support_manifest_digest != support.digest
        ):
            raise ValueError("existing dataset view does not match requested manifests")
        return view


def _materialize_file(source: Path, destination: Path) -> str:
    # A hardlink shares the source inode. On Windows, the cleanup retry may
    # clear a read-only bit on that inode and unexpectedly mutate the dataset.
    try:
        if not (source.stat().st_mode & stat.S_IWRITE):
            shutil.copyfile(source, destination)
            return "copy"
    except OSError:
        pass
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _cleanup_temporary(path: Path) -> None:
    """Best-effort cleanup that cannot hide the operation's original error."""

    if not path.exists():
        return
    try:
        shutil.rmtree(path, onerror=_retry_readonly_removal)
    except OSError:
        # A process holding a file handle can outlive the failed build on
        # Windows. The hidden orphan is harmless; never mask the root error.
        return


def _retry_readonly_removal(function, path: str, _exc_info) -> None:
    """Retry rmtree operations after clearing Windows read-only attributes."""

    with suppress(OSError):
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    with suppress(OSError):
        function(path)


def _ensure_within(path: Path, parent: Path) -> None:
    try:
        path.relative_to(parent)
    except ValueError as exc:
        raise ValueError(f"path escapes the allowed root: {path}") from exc
