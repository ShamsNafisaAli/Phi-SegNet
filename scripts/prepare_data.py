#!/usr/bin/env python
from __future__ import annotations

import argparse
from phisegnet.prepare_dataset import prepare_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare raw segmentation data into Train/Val/Test folders.")
    parser.add_argument("--raw-image-dir", required=True)
    parser.add_argument("--raw-mask-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-ratio", nargs=3, type=float, default=[0.8, 0.1, 0.1])
    parser.add_argument("--split-csv", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-folder-name", default="Main")
    parser.add_argument("--mask-folder-name", default="Mask")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying them.")
    args = parser.parse_args()

    counts = prepare_dataset(
        raw_image_dir=args.raw_image_dir,
        raw_mask_dir=args.raw_mask_dir,
        output_dir=args.output_dir,
        split_ratio=args.split_ratio,
        split_csv=args.split_csv,
        seed=args.seed,
        image_folder_name=args.image_folder_name,
        mask_folder_name=args.mask_folder_name,
        copy=not args.move,
    )
    print("Prepared dataset:", counts)


if __name__ == "__main__":
    main()
