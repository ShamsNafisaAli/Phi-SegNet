"""Evaluation metrics and test loop for binary segmentation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def _to_numpy_binary(x, threshold: float = 0.5, from_logits: bool = True) -> np.ndarray:
    if torch.is_tensor(x):
        with torch.no_grad():
            if from_logits:
                x = torch.sigmoid(x)
            x = x.detach().cpu().numpy()
    x = np.asarray(x)
    return (x > threshold).astype(np.uint8)


def accuracy(pred, target, smooth: float = 1e-7) -> float:
    pred = _to_numpy_binary(pred, from_logits=False)
    target = _to_numpy_binary(target, from_logits=False)
    return float((pred == target).sum() / (target.size + smooth))


def iou_score(pred, target, smooth: float = 1e-7) -> float:
    pred = _to_numpy_binary(pred, from_logits=False)
    target = _to_numpy_binary(target, from_logits=False)
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return float((intersection + smooth) / (union + smooth))


def dice_coefficient(pred, target, smooth: float = 1e-7) -> float:
    pred = _to_numpy_binary(pred, from_logits=False)
    target = _to_numpy_binary(target, from_logits=False)
    intersection = np.logical_and(pred, target).sum()
    return float((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))


def precision(pred, target, smooth: float = 1e-7) -> float:
    pred = _to_numpy_binary(pred, from_logits=False)
    target = _to_numpy_binary(target, from_logits=False)
    tp = np.logical_and(pred == 1, target == 1).sum()
    fp = np.logical_and(pred == 1, target == 0).sum()
    return float((tp + smooth) / (tp + fp + smooth))


def recall(pred, target, smooth: float = 1e-7) -> float:
    pred = _to_numpy_binary(pred, from_logits=False)
    target = _to_numpy_binary(target, from_logits=False)
    tp = np.logical_and(pred == 1, target == 1).sum()
    fn = np.logical_and(pred == 0, target == 1).sum()
    return float((tp + smooth) / (tp + fn + smooth))


def f1_score(pred, target, smooth: float = 1e-7) -> float:
    p = precision(pred, target, smooth=smooth)
    r = recall(pred, target, smooth=smooth)
    return float((2 * p * r + smooth) / (p + r + smooth))


def assd(pred, target) -> float:
    """Average symmetric surface distance in pixels."""
    from scipy.ndimage import binary_erosion, distance_transform_edt

    pred = _to_numpy_binary(pred, from_logits=False).astype(bool).squeeze()
    target = _to_numpy_binary(target, from_logits=False).astype(bool).squeeze()
    if pred.sum() == 0 and target.sum() == 0:
        return 0.0
    if pred.sum() == 0 or target.sum() == 0:
        return float("inf")

    pred_surface = np.logical_xor(pred, binary_erosion(pred))
    target_surface = np.logical_xor(target, binary_erosion(target))
    dt_pred = distance_transform_edt(~pred_surface)
    dt_target = distance_transform_edt(~target_surface)
    d1 = dt_target[pred_surface]
    d2 = dt_pred[target_surface]
    return float((d1.mean() + d2.mean()) / 2.0)


def save_prediction(pred_prob: torch.Tensor, save_path: str | Path, threshold: float = 0.5) -> None:
    pred = (pred_prob.detach().cpu().numpy().squeeze() > threshold).astype(np.uint8) * 255
    Image.fromarray(pred).save(save_path)


def evaluate_model(model, dataloader, device: str | torch.device = "cuda", threshold: float = 0.5, save_dir: str | Path | None = None) -> Dict[str, float]:
    device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    model = model.to(device).eval()
    rows: List[Dict[str, float]] = []

    pred_dir = None
    if save_dir is not None:
        save_dir = Path(save_dir)
        pred_dir = save_dir / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if len(batch) == 3:
                images, masks, filenames = batch
            else:
                images, masks = batch
                filenames = [f"sample_{len(rows):05d}.png"]
            images = images.to(device)
            masks = masks.to(device)
            logits, _, _ = model(images)
            probs = torch.sigmoid(logits)

            for i in range(images.shape[0]):
                pred_i = (probs[i] > threshold).float().cpu()
                mask_i = masks[i].cpu()
                fname = filenames[i] if isinstance(filenames, (list, tuple)) else str(filenames)
                row = {
                    "filename": fname,
                    "iou": iou_score(pred_i, mask_i),
                    "dice": dice_coefficient(pred_i, mask_i),
                    "accuracy": accuracy(pred_i, mask_i),
                    "precision": precision(pred_i, mask_i),
                    "recall": recall(pred_i, mask_i),
                    "f1": f1_score(pred_i, mask_i),
                    "assd": assd(pred_i, mask_i),
                }
                rows.append(row)
                if pred_dir is not None:
                    save_prediction(probs[i], pred_dir / Path(fname).with_suffix(".png").name, threshold=threshold)

    numeric_keys = ["iou", "dice", "accuracy", "precision", "recall", "f1", "assd"]
    summary = {k: float(np.nanmean([r[k] for r in rows])) for k in numeric_keys}

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename"] + numeric_keys)
            writer.writeheader()
            writer.writerows(rows)
        with open(save_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return summary
