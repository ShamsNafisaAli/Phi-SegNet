"""Loss functions for Phi-SegNet.

This file is deliberately self-contained and does not require
segmentation_models_pytorch, so it can be copy-pasted into another project.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F


class BinaryJaccardLoss(nn.Module):
    """Jaccard/IoU loss for binary segmentation logits."""

    def __init__(self, smooth: float = 1.0, from_logits: bool = True):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            pred = torch.sigmoid(pred)
        target = target.float()
        pred = pred.float()
        dims = tuple(range(1, pred.ndim))
        intersection = torch.sum(pred * target, dim=dims)
        union = torch.sum(pred + target, dim=dims) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1.0 - iou.mean()


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation logits."""

    def __init__(self, smooth: float = 1.0, from_logits: bool = True):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            pred = torch.sigmoid(pred)
        target = target.float()
        pred = pred.float()
        dims = tuple(range(1, pred.ndim))
        intersection = torch.sum(pred * target, dim=dims)
        dice = (2.0 * intersection + self.smooth) / (torch.sum(pred, dim=dims) + torch.sum(target, dim=dims) + self.smooth)
        return 1.0 - dice.mean()


def jaccard_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0, from_logits: bool = True) -> torch.Tensor:
    return BinaryJaccardLoss(smooth=smooth, from_logits=from_logits)(pred, target)


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0, from_logits: bool = True) -> torch.Tensor:
    return DiceLoss(smooth=smooth, from_logits=from_logits)(pred, target)


def manual_unwrap_torch(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """Torch-only phase unwrap along one axis."""
    dx = torch.diff(x, dim=axis)
    two_pi = 2 * torch.pi
    delta_phase = torch.remainder(dx + torch.pi, two_pi) - torch.pi
    cum_sum = torch.cumsum(delta_phase, dim=axis)
    initial_value = x.select(dim=axis, index=0)
    return torch.cat((initial_value.unsqueeze(axis), cum_sum), dim=axis)


def phase_extract(tensor: torch.Tensor, fft_size: Tuple[int, int] | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract unwrapped FFT phase along the two spatial axes."""
    if fft_size is None:
        fft_size = tuple(tensor.shape[-2:])
    fft_result = torch.fft.fft2(tensor, s=fft_size, dim=(-2, -1))
    phase = torch.angle(fft_result)
    unwrapped_h = manual_unwrap_torch(phase, axis=-2)
    unwrapped_w = manual_unwrap_torch(phase, axis=-1)
    return unwrapped_h, unwrapped_w


def phase_loss(decoder_layers: Sequence[torch.Tensor], gt_mask: torch.Tensor, fft_size: Tuple[int, int] | None = None) -> torch.Tensor:
    """Phase-supervision loss between decoder phase predictions and the ground-truth mask."""
    gt_mask = gt_mask.float()
    gt_phase_h, gt_phase_w = phase_extract(gt_mask, fft_size=fft_size)
    loss = gt_mask.new_tensor(0.0)

    for layer in decoder_layers:
        layer = torch.sigmoid(layer)
        pred_phase_h, pred_phase_w = phase_extract(layer, fft_size=fft_size)
        loss = loss + F.mse_loss(pred_phase_h, gt_phase_h) + F.mse_loss(pred_phase_w, gt_phase_w)
    return loss


# Backward-compatible function names from the original notebook.
overall_phase_loss = phase_loss


def Jaccardloss_list(predicted: Sequence[torch.Tensor], ground: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    criterion = BinaryJaccardLoss(from_logits=True)
    loss = ground.new_tensor(0.0)
    for pred in predicted:
        loss = loss + criterion(pred, ground)
    loss = loss / max(len(predicted), 1)
    return loss, [loss * 0, loss]


def total_loss(
    predicted: torch.Tensor,
    decoder_layers: Sequence[torch.Tensor],
    ground: torch.Tensor,
    coupling_factor: Sequence[float] = (0.01, 1.0),
    fft_size: Tuple[int, int] | None = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Original-style total loss: alpha * phase loss + beta * Jaccard loss."""
    alpha, beta = float(coupling_factor[0]), float(coupling_factor[1])
    p_loss = phase_loss(decoder_layers, ground, fft_size=fft_size) if alpha > 0 else predicted.new_tensor(0.0)
    iou_loss = BinaryJaccardLoss(from_logits=True)(predicted, ground) if beta > 0 else predicted.new_tensor(0.0)
    weighted_phase = alpha * p_loss
    weighted_iou = beta * iou_loss
    return weighted_phase + weighted_iou, [weighted_phase, weighted_iou]


class PhiSegLoss(nn.Module):
    """Reusable combined loss for Phi-SegNet."""

    def __init__(self, phase_weight: float = 0.01, jaccard_weight: float = 1.0, fft_size: Tuple[int, int] | None = None):
        super().__init__()
        self.phase_weight = phase_weight
        self.jaccard_weight = jaccard_weight
        self.fft_size = fft_size
        self.jaccard = BinaryJaccardLoss(from_logits=True)

    def forward(
        self,
        pred_logits: torch.Tensor,
        decoder_layers: Sequence[torch.Tensor],
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        p_loss = phase_loss(decoder_layers, target, fft_size=self.fft_size) if self.phase_weight > 0 else pred_logits.new_tensor(0.0)
        j_loss = self.jaccard(pred_logits, target) if self.jaccard_weight > 0 else pred_logits.new_tensor(0.0)
        weighted_phase = self.phase_weight * p_loss
        weighted_jaccard = self.jaccard_weight * j_loss
        total = weighted_phase + weighted_jaccard
        return total, {
            "total": total.detach(),
            "phase": weighted_phase.detach(),
            "jaccard": weighted_jaccard.detach(),
        }
