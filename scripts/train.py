#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from phisegnet.dataset import SegmentationDataset
from phisegnet.loss import PhiSegLoss
from phisegnet.model import PhiSegNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running = {"total": 0.0, "phase": 0.0, "jaccard": 0.0}
    n = 0
    for images, masks in tqdm(loader, desc="Train", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, decoder_layers, _ = model(images)
        loss, loss_dict = criterion(logits, decoder_layers, masks)
        loss.backward()
        optimizer.step()
        bs = images.size(0)
        n += bs
        running["total"] += float(loss.item()) * bs
        running["phase"] += float(loss_dict["phase"].item()) * bs
        running["jaccard"] += float(loss_dict["jaccard"].item()) * bs
    return {k: v / max(n, 1) for k, v in running.items()}


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running = {"total": 0.0, "phase": 0.0, "jaccard": 0.0}
    n = 0
    for images, masks in tqdm(loader, desc="Val", leave=False):
        images, masks = images.to(device), masks.to(device)
        logits, decoder_layers, _ = model(images)
        loss, loss_dict = criterion(logits, decoder_layers, masks)
        bs = images.size(0)
        n += bs
        running["total"] += float(loss.item()) * bs
        running["phase"] += float(loss_dict["phase"].item()) * bs
        running["jaccard"] += float(loss_dict["jaccard"].item()) * bs
    return {k: v / max(n, 1) for k, v in running.items()}


def main():
    parser = argparse.ArgumentParser(description="Train Phi-SegNet.")
    parser.add_argument("--config", default="configs/phisegnet.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["train"].get("device", "cuda") if torch.cuda.is_available() else "cpu")
    save_dir = Path(cfg["train"].get("save_dir", "outputs/run"))
    save_dir.mkdir(parents=True, exist_ok=True)

    train_ds = SegmentationDataset(
        cfg["data"]["train_image_dir"], cfg["data"]["train_mask_dir"],
        image_size=cfg["data"].get("image_size", 256), mode="train"
    )
    val_ds = SegmentationDataset(
        cfg["data"]["val_image_dir"], cfg["data"]["val_mask_dir"],
        image_size=cfg["data"].get("image_size", 256), mode="val"
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["train"].get("batch_size", 4), shuffle=True, num_workers=cfg["train"].get("num_workers", 2))
    val_loader = DataLoader(val_ds, batch_size=cfg["train"].get("val_batch_size", 1), shuffle=False, num_workers=cfg["train"].get("num_workers", 2))

    model = PhiSegNet(
        input_channels=cfg["model"].get("input_channels", 3),
        num_classes=cfg["model"].get("num_classes", 1),
        pretrained=cfg["model"].get("pretrained", True),
    ).to(device)

    criterion = PhiSegLoss(
        phase_weight=cfg["loss"].get("phase_weight", 0.01),
        jaccard_weight=cfg["loss"].get("jaccard_weight", 1.0),
        fft_size=tuple(cfg["loss"].get("fft_size", [cfg["data"].get("image_size", 256), cfg["data"].get("image_size", 256)])),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"].get("lr", 1e-5), weight_decay=cfg["train"].get("weight_decay", 0.0))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"].get("scheduler_tmax", 25), eta_min=cfg["train"].get("min_lr", 1e-7)
    )

    best_val = float("inf")
    history = []
    for epoch in range(1, cfg["train"].get("epochs", 150) + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step()
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "lr": optimizer.param_groups[0]["lr"]}
        history.append(record)
        print(f"Epoch {epoch:03d}: train={train_metrics['total']:.4f}, val={val_metrics['total']:.4f}, lr={record['lr']:.2e}")

        torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch}, save_dir / "last_model.pth")
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch}, save_dir / "best_model.pth")

        with open(save_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
