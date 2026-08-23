"""Tiny synthetic MVTec-shaped datasets for pipeline smoke tests.

The bottle category has three blue training images with a white ellipse, one
good test image, one test image with a red defect rectangle, and the matching
ground-truth mask. The shape is chosen so a trivial nearest-pixel-distance
detector scores the defective image far above the good one (AUROC ≈ 1.0).
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def build_synthetic_mvtec_smoke_dataset(root: Path, *, defect: str = "broken") -> None:
    """Idempotently create a minimal MVTec-shaped dataset under ``root``."""

    paths = {
        "train": root / "bottle" / "train" / "good",
        "good": root / "bottle" / "test" / "good",
        "bad": root / "bottle" / "test" / defect,
        "mask": root / "bottle" / "ground_truth" / defect,
    }
    for directory in paths.values():
        directory.mkdir(parents=True, exist_ok=True)

    for index, color in enumerate(((40, 100, 180), (75, 125, 175), (45, 135, 165))):
        target = paths["train"] / f"{index:03}.png"
        if not target.exists():
            image = Image.new("RGB", (224, 224), color)
            ImageDraw.Draw(image).ellipse((62, 28, 162, 202), outline="white", width=8)
            image.save(target)

    good = paths["good"] / "100.png"
    if not good.exists():
        Image.new("RGB", (224, 224), (55, 110, 180)).save(good)

    bad = paths["bad"] / "101.png"
    if not bad.exists():
        image = Image.new("RGB", (224, 224), (55, 110, 180))
        ImageDraw.Draw(image).rectangle((90, 90, 135, 135), fill=(230, 30, 30))
        image.save(bad)

    mask = paths["mask"] / "101_mask.png"
    if not mask.exists():
        image = Image.new("L", (224, 224), 0)
        ImageDraw.Draw(image).rectangle((90, 90, 135, 135), fill=255)
        image.save(mask)
