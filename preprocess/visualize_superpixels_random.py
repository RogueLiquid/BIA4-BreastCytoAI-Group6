#!/usr/bin/env python3
"""
visualize_superpixels_random.py

Randomly visualize SLIC–KMeans–BIRCH superpixel segmentations.

Expected structure (relative to main_folder = repo root):

    BreakHis_400X_full_augmented/
        train/<subtype>/*.png
        validation/<subtype>/*.png
        test/<subtype>/*.png   # (optional)

    BreakHis_400X_augmented_superpixels/
        train/<subtype>/*_labels.npz
        validation/<subtype>/*_labels.npz
        test/<subtype>/*_labels.npz

Usage (from main_folder):

    # show 4 random ORIGINAL (non-augmented) images from train+val+test
    python preprocess/visualize_superpixels_random.py --n 4

    # include augmented images as well
    python preprocess/visualize_superpixels_random.py --n 4 --include-augmented
"""

import argparse
import os
import random
from pathlib import Path
from typing import List

import numpy as np
from skimage.io import imread
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt


# ==========================
# CONFIGURATION
# ==========================

REPO_ROOT = Path(__file__).resolve().parent.parent

# Must match your preprocessing script
IMAGE_ROOT = REPO_ROOT / "BreakHis_400X_full_augmented"        # has train/, validation/, test/
SUPERPIXEL_ROOT = REPO_ROOT / "BreakHis_400X_augmented_superpixels"

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
SPLITS = ["train", "validation", "test"]  # change if you only want some splits


# ==========================
# UTILITIES
# ==========================

def iter_image_files(root: Path) -> List[Path]:
    """Collect all image paths under root (recursively)."""
    out: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if any(fname.lower().endswith(ext) for ext in IMAGE_EXTS):
                out.append(Path(dirpath) / fname)
    return sorted(out)


def load_superpixel_labels_for_image(img_path: Path) -> np.ndarray:
    """
    Given an image path under IMAGE_ROOT, load corresponding superpixel labels
    from SUPERPIXEL_ROOT.

    Example:
        BreakHis_400X_full_augmented/train/adenosis/img1.png
    ->  BreakHis_400X_augmented_superpixels/train/adenosis/img1_labels.npz
    """
    rel = img_path.relative_to(IMAGE_ROOT)
    labels_path = SUPERPIXEL_ROOT / rel.parent / (img_path.stem + "_labels.npz")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found for image {img_path} at {labels_path}")
    data = np.load(labels_path)
    return data["labels"]


# ==========================
# MAIN VISUALIZATION
# ==========================

def main(n_samples: int, include_augmented: bool) -> None:
    print(f"IMAGE_ROOT        = {IMAGE_ROOT}")
    print(f"SUPERPIXEL_ROOT   = {SUPERPIXEL_ROOT}")
    print(f"n_samples         = {n_samples}")
    print(f"include_augmented = {include_augmented}")

    # Collect all images from specified splits
    all_img_paths: List[Path] = []
    for split in SPLITS:
        split_root = IMAGE_ROOT / split
        if not split_root.exists():
            print(f"[WARN] Split folder '{split_root}' not found, skipping.")
            continue
        paths = iter_image_files(split_root)
        all_img_paths.extend(paths)

    if not all_img_paths:
        raise RuntimeError(f"No images found under {IMAGE_ROOT}")

    # Optionally filter out augmented images (filenames containing '_augment')
    if not include_augmented:
        before = len(all_img_paths)
        all_img_paths = [p for p in all_img_paths if "_augment" not in p.stem]
        print(f"Filtered augmented images: {before} -> {len(all_img_paths)} originals")

    if not all_img_paths:
        raise RuntimeError("No images to visualize after filtering.")

    # If n_samples is larger than available, cap it
    n_samples = min(n_samples, len(all_img_paths))
    print(f"Total images available: {len(all_img_paths)}")
    print(f"Randomly sampling {n_samples} images.\n")

    sampled_paths = random.sample(all_img_paths, n_samples)

    for idx, img_path in enumerate(sampled_paths, start=1):
        try:
            labels = load_superpixel_labels_for_image(img_path)
        except FileNotFoundError as e:
            print(f"[{idx}/{n_samples}] {e}")
            continue

        img = imread(img_path)
        if img.ndim == 2:  # grayscale -> fake RGB
            img = np.stack([img, img, img], axis=-1)

        overlay = mark_boundaries(img, labels, color=(1, 0, 0))  # red boundaries

        plt.figure(figsize=(5, 5))
        plt.imshow(overlay)
        plt.axis("off")
        plt.title(str(img_path.relative_to(IMAGE_ROOT)))
        plt.tight_layout()
        print(f"[{idx}/{n_samples}] Showing {img_path}")
        plt.show()  # blocks until you close the window


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Randomly visualize SLIC-BIRCH superpixels.")
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of random images to visualize.",
    )
    parser.add_argument(
        "--include-augmented",
        action="store_true",
        help="If set, also sample augmented images (filenames with '_augment').",
    )

    # parse_known_args will ignore Jupyter/VS Code extra args like --f=...
    args, _ = parser.parse_known_args()

    if args.n <= 0:
        raise ValueError("--n must be a positive integer")

    main(n_samples=args.n, include_augmented=args.include_augmented)
