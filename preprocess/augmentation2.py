import random
import shutil
from pathlib import Path

from PIL import Image
import torch
from torchvision import transforms

# Deterministic ordering for reproducible oversampling/augmentation.
random.seed(42)
torch.manual_seed(42)

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = REPO_ROOT / "BreakHis_400X_full"
OUT_DIR = REPO_ROOT / "BreakHis_400X_full_augmented2"

# Map BreakHis subtypes to binary classes.
SUBTYPE_TO_CLASS = {
    # benign
    "adenosis": "benign",
    "fibroadenoma": "benign",
    "phyllodes_tumor": "benign",
    "tubular_adenoma": "benign",
    # malignant
    "ductal_carcinoma": "malignant",
    "lobular_carcinoma": "malignant",
    "mucinous_carcinoma": "malignant",
    "papillary_carcinoma": "malignant",
}
CLASS_NAMES = ["benign", "malignant"]

# Augmentations applied only to training images after balancing.
aug = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),  # optional; keep if you believe orientation is irrelevant
    transforms.RandomRotation(degrees=45),  # small rotation, not 90
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05,
    )
])

N_AUG_PER_IMAGE = 1  # how many extra images per (original or upsampled)


def copy_with_augmentation(src_dir: Path, dst_dir: Path, apply_aug: bool) -> None:
    """Copy PNGs, balance benign to malignant (train only), then write augmented variants."""
    dst_dir.mkdir(parents=True, exist_ok=True)

    records: list[tuple[Path, str, str]] = []  # (img_path, subtype, class_name)
    for subtype_dir in src_dir.iterdir():
        if not subtype_dir.is_dir():
            continue
        if subtype_dir.name not in SUBTYPE_TO_CLASS:
            raise ValueError(f"Unknown subtype folder '{subtype_dir.name}' in {src_dir}")
        class_name = SUBTYPE_TO_CLASS[subtype_dir.name]
        for img_path in subtype_dir.glob("*.png"):
            records.append((img_path, subtype_dir.name, class_name))

    # Copy originals (used later for augmentation).
    original_paths: list[Path] = []
    for img_path, subtype, _class_name in records:
        dst_path = dst_dir / subtype / img_path.name
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, dst_path)
        original_paths.append(dst_path)

    # Upsample benign to match malignant in the training split.
    upsampled_augmented_paths: list[Path] = []
    if apply_aug:
        benign_pool = [rec for rec in records if rec[2] == "benign"]
        malignant_count = sum(1 for rec in records if rec[2] == "malignant")
        benign_count = len(benign_pool)
        if benign_count == 0:
            raise ValueError("No benign images found; cannot upsample to balance classes.")
        if benign_count < malignant_count:
            needed = malignant_count - benign_count
            for idx, (img_path, subtype, _) in enumerate(random.choices(benign_pool, k=needed), start=1):
                dst_path = dst_dir / subtype / f"{img_path.stem}_upsampled_augmented_{idx}{img_path.suffix}"
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                img = Image.open(img_path).convert("RGB")
                aug(img).save(dst_path)
                upsampled_augmented_paths.append(dst_path)

    # Apply augmentations after balancing (train split only).
    if apply_aug:
        augmented_paths: list[Path] = []
        for img_path in original_paths:
            img = Image.open(img_path).convert("RGB")
            for i in range(1, N_AUG_PER_IMAGE + 1):
                aug_img = aug(img)
                aug_filename = f"{img_path.stem}_augmented_{i}{img_path.suffix}"
                aug_path = img_path.parent / aug_filename
                aug_img.save(aug_path)
                augmented_paths.append(aug_path)

        # Ensure even total count after augmentation (train split only).
        total_count = len(original_paths) + len(upsampled_augmented_paths) + len(augmented_paths)
        if total_count % 2 == 1 and original_paths:
            img_path = original_paths[0]
            img = Image.open(img_path).convert("RGB")
            extra_aug = aug(img)
            extra_path = img_path.parent / f"{img_path.stem}_augmented_extra{img_path.suffix}"
            extra_aug.save(extra_path)


def augment_breakhis_balanced(source_root: Path = SOURCE_DIR, output_root: Path = OUT_DIR) -> None:
    """Copy BreakHis_400X_full to BreakHis_400X_full_augmented2 with balancing + augmentation."""
    if not source_root.exists():
        raise FileNotFoundError(f"Source directory not found: {source_root}")

    for split in ["train", "validation"]:
        split_src = source_root / split
        split_dst = output_root / split
        if not split_src.exists():
            print(f"[WARN] Split folder missing, skipping: {split_src}")
            continue

        copy_with_augmentation(split_src, split_dst, apply_aug=(split == "train"))

    print(f"Finished writing balanced augmented dataset to {output_root}")


# Example use inside VS Code / notebook:
# from preprocess.augmentation2 import augment_breakhis_balanced
augment_breakhis_balanced()
