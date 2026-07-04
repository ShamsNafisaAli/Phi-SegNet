"""Dataset and paired transforms for binary medical image segmentation."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_images(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    files = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files)


def resize_and_pad_image(image: np.ndarray, mask: np.ndarray, scale: float = 0.75) -> Tuple[np.ndarray, np.ndarray]:
    """Notebook-style random scale augmentation followed by zero padding."""
    import cv2

    h, w = image.shape[:2]
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros_like(image)
    canvas_mask = np.zeros_like(mask)
    top = np.random.randint(0, h - new_h + 1)
    left = np.random.randint(0, w - new_w + 1)
    canvas[top : top + new_h, left : left + new_w] = resized_image
    canvas_mask[top : top + new_h, left : left + new_w] = resized_mask
    return canvas, canvas_mask


def paired_random_flip(image: np.ndarray, mask: np.ndarray, p: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    if random.random() < p:
        image = np.ascontiguousarray(np.fliplr(image))
        mask = np.ascontiguousarray(np.fliplr(mask))
    if random.random() < p:
        image = np.ascontiguousarray(np.flipud(image))
        mask = np.ascontiguousarray(np.flipud(mask))
    return image, mask


def to_tensor(image: np.ndarray, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    image = image.astype(np.float32) / 255.0
    if image.ndim == 2:
        image = image[..., None]
    image = torch.from_numpy(image.transpose(2, 0, 1)).float()

    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = (mask.astype(np.float32) / 255.0 > 0.5).astype(np.float32)
    mask = torch.from_numpy(mask[None, ...]).float()
    return image, mask


class SegmentationDataset(Dataset):
    """Paired image-mask dataset.

    Expected folder examples:
        root/Train/Main, root/Train/Mask
        root/Train/image, root/Train/mask

    Or pass image_dir and mask_dir directly.
    """

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        image_size: int = 256,
        mode: str = "train",
        augment: Optional[bool] = None,
        return_filename: bool = False,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = int(image_size)
        self.mode = mode.lower()
        self.augment = self.mode == "train" if augment is None else augment
        self.return_filename = return_filename

        self.image_paths = list_images(self.image_dir)
        self.mask_paths = list_images(self.mask_dir)
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {self.image_dir}")
        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError(f"Image/mask count mismatch: {len(self.image_paths)} images vs {len(self.mask_paths)} masks")

        self.mask_by_stem = {p.stem: p for p in self.mask_paths}

    def __len__(self) -> int:
        return len(self.image_paths)

    def _find_mask(self, image_path: Path) -> Path:
        if image_path.stem in self.mask_by_stem:
            return self.mask_by_stem[image_path.stem]
        # fallback to sorted pairing for datasets where extensions differ or names are not identical
        idx = self.image_paths.index(image_path)
        return self.mask_paths[idx]

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        mask_path = self._find_mask(image_path)

        image = Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize((self.image_size, self.image_size), Image.NEAREST)
        image_np = np.array(image)
        mask_np = np.array(mask)

        if self.augment:
            # Matches the notebook's random scale/pad idea while adding safe paired flips.
            scale = min(random.uniform(0.5, 1.25), 1.0)
            image_np, mask_np = resize_and_pad_image(image_np, mask_np, scale=scale)
            image_np, mask_np = paired_random_flip(image_np, mask_np, p=0.5)

        image_t, mask_t = to_tensor(image_np, mask_np)
        if self.return_filename:
            return image_t, mask_t, image_path.name
        return image_t, mask_t


def build_dataset_from_root(root: str | Path, split: str, image_size: int = 256, return_filename: bool = False) -> SegmentationDataset:
    """Convenience builder for root/split/Main+Mask or root/split/image+mask."""
    split_dir = Path(root) / split
    candidates = [("Main", "Mask"), ("image", "mask"), ("images", "masks")]
    for image_name, mask_name in candidates:
        image_dir, mask_dir = split_dir / image_name, split_dir / mask_name
        if image_dir.exists() and mask_dir.exists():
            return SegmentationDataset(image_dir, mask_dir, image_size=image_size, mode=split, return_filename=return_filename)
    raise FileNotFoundError(f"Could not find image/mask folders under {split_dir}. Tried {candidates}.")
