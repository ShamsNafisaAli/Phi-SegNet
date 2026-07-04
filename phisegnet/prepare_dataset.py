"""Prepare raw image/mask folders into Train/Val/Test folders."""

from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _list_images(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])


def _make_dirs(output_dir: Path, image_folder_name: str = "Main", mask_folder_name: str = "Mask") -> None:
    for split in ["Train", "Val", "Test"]:
        (output_dir / split / image_folder_name).mkdir(parents=True, exist_ok=True)
        (output_dir / split / mask_folder_name).mkdir(parents=True, exist_ok=True)


def _find_mask(image_path: Path, mask_paths_by_stem: Dict[str, Path]) -> Path:
    if image_path.stem not in mask_paths_by_stem:
        raise FileNotFoundError(f"No matching mask found for {image_path.name}")
    return mask_paths_by_stem[image_path.stem]


def split_files_random(files: Sequence[Path], ratios: Sequence[float], seed: int = 42) -> Dict[str, List[Path]]:
    if len(ratios) != 3:
        raise ValueError("ratios must have three values: train val test")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("split ratios must sum to 1.0")
    files = list(files)
    random.Random(seed).shuffle(files)
    n = len(files)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return {
        "Train": files[:n_train],
        "Val": files[n_train : n_train + n_val],
        "Test": files[n_train + n_val :],
    }


def split_files_from_csv(csv_path: str | Path, raw_image_dir: str | Path) -> Dict[str, List[Path]]:
    raw_image_dir = Path(raw_image_dir)
    split_map = {"train": "Train", "val": "Val", "valid": "Val", "validation": "Val", "test": "Test"}
    output = {"Train": [], "Val": [], "Test": []}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "filename" not in reader.fieldnames or "split" not in reader.fieldnames:
            raise ValueError("split CSV must contain columns: filename, split")
        for row in reader:
            split = split_map[row["split"].strip().lower()]
            output[split].append(raw_image_dir / row["filename"].strip())
    return output


def prepare_dataset(
    raw_image_dir: str | Path,
    raw_mask_dir: str | Path,
    output_dir: str | Path,
    split_ratio: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    split_csv: str | Path | None = None,
    image_folder_name: str = "Main",
    mask_folder_name: str = "Mask",
    copy: bool = True,
) -> Dict[str, int]:
    raw_image_dir = Path(raw_image_dir)
    raw_mask_dir = Path(raw_mask_dir)
    output_dir = Path(output_dir)
    _make_dirs(output_dir, image_folder_name=image_folder_name, mask_folder_name=mask_folder_name)

    image_paths = _list_images(raw_image_dir)
    mask_paths_by_stem = {p.stem: p for p in _list_images(raw_mask_dir)}
    splits = split_files_from_csv(split_csv, raw_image_dir) if split_csv else split_files_random(image_paths, split_ratio, seed=seed)

    action = shutil.copy2 if copy else shutil.move
    counts: Dict[str, int] = {}
    for split, files in splits.items():
        counts[split] = 0
        for image_path in files:
            mask_path = _find_mask(image_path, mask_paths_by_stem)
            dst_img = output_dir / split / image_folder_name / image_path.name
            dst_mask = output_dir / split / mask_folder_name / mask_path.name
            action(image_path, dst_img)
            action(mask_path, dst_mask)
            counts[split] += 1
    return counts
