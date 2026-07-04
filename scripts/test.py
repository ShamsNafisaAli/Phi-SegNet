#!/usr/bin/env python
from __future__ import annotations

import argparse
import torch
from torch.utils.data import DataLoader

from phisegnet.dataset import SegmentationDataset
from phisegnet.evaluation import evaluate_model
from phisegnet.model import PhiSegNet


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phi-SegNet on a test set.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--save-dir", default="outputs/test_results")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pretrained", action="store_true", help="Only affects model construction before loading checkpoint.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dataset = SegmentationDataset(args.image_dir, args.mask_dir, image_size=args.image_size, mode="test", return_filename=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = PhiSegNet(input_channels=3, num_classes=1, pretrained=args.pretrained)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)

    summary = evaluate_model(model, loader, device=device, threshold=args.threshold, save_dir=args.save_dir)
    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
