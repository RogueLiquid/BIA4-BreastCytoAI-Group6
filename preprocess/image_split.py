#!/usr/bin/env python3
"""Patient-level train/validation split for BreakHis 400X images.

This script provides a callable helper (split_breakhis) to copy 400X tiles into
BreakHis_400X_full/train and BreakHis_400X_full/validation with an approximately
70/30 patient-level split per subtype (4 benign, 4 malignant). Designed to be
run from VS Code without CLI parsing.
"""
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

# Mapping of high-level label to subtype folder names
SUBTYPES: Dict[str, Sequence[str]] = {
    "benign": [
        "adenosis",
        "fibroadenoma",
        "phyllodes_tumor",
        "tubular_adenoma",
    ],
    "malignant": [
        "ductal_carcinoma",
        "lobular_carcinoma",
        "mucinous_carcinoma",
        "papillary_carcinoma",
    ],
}

# Defaults used when no arguments are provided to split_breakhis.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = REPO_ROOT / "BreakHis" / "BreaKHis_v1" / "BreaKHis_v1" / "histology_slides" / "breast"
DEFAULT_OUTPUT = REPO_ROOT / "BreakHis_400X_full"
DEFAULT_VAL_RATIO = 0.30
DEFAULT_SEED = 42


def discover_patients(subtype_dir: Path) -> List[Path]:
    """Return patient directories inside a subtype directory (SOB prefix)."""
    return sorted([p for p in subtype_dir.iterdir() if p.is_dir()])


def split_patients(patients: Sequence[Path], val_ratio: float) -> Tuple[List[Path], List[Path]]:
    patients = list(patients)
    random.shuffle(patients)
    n = len(patients)
    if n == 0:
        return [], []
    n_val = max(1, int(round(n * val_ratio)))
    if n_val >= n and n > 1:
        n_val = n - 1
    val_patients = patients[:n_val]
    train_patients = patients[n_val:]
    return train_patients, val_patients


def copy_patient_images(patients: Iterable[Path], dest_root: Path) -> int:
    """Copy all PNGs from 400X folder of each patient into dest_root/subtype."""
    copied = 0
    for patient_dir in patients:
        mag_dir = patient_dir / "400X"
        if not mag_dir.exists():
            print(f"[WARN] Missing 400X folder: {mag_dir}")
            continue
        for img_path in mag_dir.glob("*.png"):
            target_path = dest_root / img_path.name
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, target_path)
            copied += 1
    return copied


def process_subtype(label: str, subtype: str, source_root: Path, output_root: Path, val_ratio: float) -> Tuple[int, int]:
    subtype_dir = source_root / label / "SOB" / subtype
    patients = discover_patients(subtype_dir)
    train_patients, val_patients = split_patients(patients, val_ratio)

    train_target = output_root / "train" / subtype
    val_target = output_root / "validation" / subtype

    copied_train = copy_patient_images(train_patients, train_target)
    copied_val = copy_patient_images(val_patients, val_target)

    print(
        f"Subtype {subtype}: {len(train_patients)} train patients -> {copied_train} imgs, "
        f"{len(val_patients)} val patients -> {copied_val} imgs"
    )
    return copied_train, copied_val


def split_breakhis(
    source_root: Path | None = None,
    output_root: Path | None = None,
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = DEFAULT_SEED,
) -> None:
    """Run the patient-level split with reproducible shuffling.

    Args:
        source_root: Path to BreakHis breast root (defaults to repo BreakHis_v1 path).
        output_root: Destination directory for BreakHis_400X_full (defaults to repo root path).
        val_ratio: Fraction of patients per subtype to place in validation.
        seed: Random seed to keep splits reproducible.
    """
    random.seed(seed)

    source_root = source_root or DEFAULT_SOURCE
    output_root = output_root or DEFAULT_OUTPUT

    if not source_root.exists():
        raise FileNotFoundError(f"Source path not found: {source_root}")

    (output_root / "train").mkdir(parents=True, exist_ok=True)
    (output_root / "validation").mkdir(parents=True, exist_ok=True)

    total_train = total_val = 0
    for label, subtypes in SUBTYPES.items():
        for subtype in subtypes:
            copied_train, copied_val = process_subtype(label, subtype, source_root, output_root, val_ratio)
            total_train += copied_train
            total_val += copied_val

    print(f"Done. Copied {total_train} train images and {total_val} validation images into {output_root}")


# Example call when running inside VS Code or a notebook:
split_breakhis()
