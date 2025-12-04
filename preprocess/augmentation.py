import random
import shutil
from pathlib import Path

from PIL import Image
import torch
from torchvision import transforms

# Set deterministic seeds for reproducible augmentation ordering.
random.seed(42)
torch.manual_seed(42)

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = REPO_ROOT / "BreakHis_400X_full"
OUT_DIR = REPO_ROOT / "BreakHis_400X_full_augmented"

# Augmentations applied only to training images.
aug = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=45),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        ),
    ]
)

N_AUG_PER_IMAGE = 4  # how many extra images per original


def copy_with_augmentation(src_dir: Path, dst_dir: Path, apply_aug: bool) -> None:
    """Copy PNGs and optionally write augmented variants."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for img_path in src_dir.glob("*.png"):
        dst_path = dst_dir / img_path.name
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, dst_path)

        if not apply_aug:
            continue

        img = Image.open(img_path).convert("RGB")
        stem = img_path.stem
        for i in range(1, N_AUG_PER_IMAGE + 1):
            aug_img = aug(img)
            aug_filename = f"{stem}_augmented_{i}.png"
            aug_img.save(dst_dir / aug_filename)


def augment_breakhis(source_root: Path = SOURCE_DIR, output_root: Path = OUT_DIR) -> None:
    """Copy BreakHis_400X_full and augment only train images."""
    if not source_root.exists():
        raise FileNotFoundError(f"Source directory not found: {source_root}")

    for split in ["train", "validation"]:
        split_src = source_root / split
        split_dst = output_root / split
        if not split_src.exists():
            print(f"[WARN] Split folder missing, skipping: {split_src}")
            continue

        for subtype_dir in split_src.iterdir():
            if not subtype_dir.is_dir():
                continue
            subtype_dst = split_dst / subtype_dir.name
            copy_with_augmentation(subtype_dir, subtype_dst, apply_aug=(split == "train"))

    print(f"Finished writing augmented dataset to {output_root}")


# Example use inside VS Code / notebook:
# from preprocess.augmentation import augment_breakhis
augment_breakhis()
