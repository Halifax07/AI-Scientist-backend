from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the pinned AnomalyDINO implementation for one audited category"
    )
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--category", required=True)
    arguments, forwarded = parser.parse_known_args()
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    repository = arguments.repository.expanduser().resolve()
    script = repository / "run_anomalydino.py"
    if not script.is_file():
        raise FileNotFoundError(f"Pinned AnomalyDINO entry point not found: {script}")

    sys.path.insert(0, str(repository))
    import cv2  # type: ignore[import-not-found]
    import numpy as np

    original_imread = cv2.imread

    def unicode_safe_imread(filename: str, flags: int = cv2.IMREAD_COLOR):
        if os.name == "nt" and isinstance(filename, (str, os.PathLike)):
            encoded = np.fromfile(filename, dtype=np.uint8)
            return cv2.imdecode(encoded, flags)
        return original_imread(filename, flags)

    cv2.imread = unicode_safe_imread
    from src import detection, post_eval, utils  # type: ignore[import-not-found]

    original_read_tiff = post_eval.read_tiff

    def case_insensitive_safe_read_tiff(
        file_path_no_ext: str,
        exts: tuple[str, ...] = (".tif", ".tiff", ".TIF", ".TIFF"),
    ):
        # On Windows, `x.tiff` also exists when queried as `x.TIFF`. The
        # upstream reader counts those aliases as two files and aborts.
        unique_exts = tuple(dict.fromkeys(ext.casefold() for ext in exts))
        return original_read_tiff(file_path_no_ext, exts=unique_exts)

    post_eval.read_tiff = case_insensitive_safe_read_tiff

    original_dataset_info = utils.get_dataset_info
    audited_data_root: Path | None = None

    def single_category_info(
        dataset: str,
        preprocess: str,
        data_path: str | None = None,
    ) -> tuple[list[str], dict[str, Any], dict[str, Any], dict[str, Any]]:
        nonlocal audited_data_root
        objects, _anomalies, masking, rotation = original_dataset_info(
            dataset,
            preprocess,
            data_path=data_path,
        )
        if arguments.category not in objects:
            raise ValueError(
                f"Category {arguments.category!r} is not valid for dataset {dataset!r}"
            )
        if data_path is not None:
            requested_root = Path(data_path).expanduser().resolve()
            if audited_data_root is not None and requested_root != audited_data_root:
                raise ValueError("Dataset root changed during one detector execution")
            audited_data_root = requested_root
        if audited_data_root is None:
            raise ValueError("An audited dataset view path is required")
        test_root = audited_data_root / arguments.category / "test"
        observed_anomalies = sorted(
            child.name
            for child in test_root.iterdir()
            if child.is_dir() and child.name != "good"
        )
        if not observed_anomalies:
            raise ValueError(
                f"No anomaly directories were found in audited test view: {test_root}"
            )
        return (
            [arguments.category],
            {arguments.category: observed_anomalies},
            {arguments.category: masking[arguments.category]},
            {arguments.category: rotation[arguments.category]},
        )

    utils.get_dataset_info = single_category_info
    post_eval.get_objects_from_dataset = lambda _dataset: [arguments.category]
    original_detection = detection.run_anomaly_detection

    def frozen_support_detection(*args: Any, **kwargs: Any):
        # The audited view already contains exactly K images. The upstream seed
        # implementation slices K-sized blocks and would return an empty support
        # set for seed > 0, so use its full-shot branch over this K-only view.
        if "n_ref_samples" in kwargs:
            kwargs["n_ref_samples"] = -1
        elif len(args) >= 3:
            args = (*args[:2], -1, *args[3:])
        return original_detection(*args, **kwargs)

    detection.run_anomaly_detection = frozen_support_detection
    sys.argv = [str(script), *forwarded]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
