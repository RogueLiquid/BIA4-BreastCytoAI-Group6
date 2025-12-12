#!/usr/bin/env python3
"""
Patient-level split of the full BreakHis dataset (all magnifications) into train/validation.

- Source layout (original BreakHis):
  BreakHis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/{benign,malignant}/SOB/<subtype>/<patient_id>/<mag>/*.png
  where <mag> is one of {40X,100X,200X,400X}, and patient_id folders look like "SOB_B_A_14-22549AB".

- Output layout:
  BreakHis_full/
    train/<subtype>/*.png
    validation/<subtype>/*.png
  (images from all magnifications are copied into the subtype folder; filenames retain mag info)

- Split policy:
  * Patient-level split: no patient leaks across splits.
  * Target: 70% train, 30% val. If total patients == 82, this becomes exactly 56 train / 26 val.
  * Stratified by subtype; we also ensure every magnification appears in both splits (by swapping patients if needed).
  * Deterministic with SEED.
"""

from __future__ import annotations

import random
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

SEED = 42
random.seed(SEED)

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO_ROOT / "BreakHis" / "BreaKHis_v1" / "BreaKHis_v1" / "histology_slides" / "breast"
OUT_ROOT = REPO_ROOT / "BreakHis_full"

SUBTYPE_ROOTS = {
    "adenosis": SOURCE_ROOT / "benign" / "SOB" / "adenosis",
    "fibroadenoma": SOURCE_ROOT / "benign" / "SOB" / "fibroadenoma",
    "phyllodes_tumor": SOURCE_ROOT / "benign" / "SOB" / "phyllodes_tumor",
    "tubular_adenoma": SOURCE_ROOT / "benign" / "SOB" / "tubular_adenoma",
    "ductal_carcinoma": SOURCE_ROOT / "malignant" / "SOB" / "ductal_carcinoma",
    "lobular_carcinoma": SOURCE_ROOT / "malignant" / "SOB" / "lobular_carcinoma",
    "mucinous_carcinoma": SOURCE_ROOT / "malignant" / "SOB" / "mucinous_carcinoma",
    "papillary_carcinoma": SOURCE_ROOT / "malignant" / "SOB" / "papillary_carcinoma",
}

MAGNIFICATIONS = {"40X", "100X", "200X", "400X"}
TRAIN_RATIO = 0.7


@dataclass
class PatientRecord:
    patient_id: str
    subtype: str
    mags: set[str]
    images: List[Path]


def extract_patient_id(patient_dir: Path) -> str:
    """
    Extract patient id from directory name. Examples:
    - SOB_B_A_14-22549AB -> 22549AB
    - SOB_M_DC-14-12345 -> 12345
    """
    m = re.search(r"-([0-9A-Za-z]+)$", patient_dir.name)
    if not m:
        raise ValueError(f"Cannot parse patient id from folder: {patient_dir}")
    return m.group(1)


def collect_patients() -> List[PatientRecord]:
    patients: List[PatientRecord] = []
    for subtype, root in SUBTYPE_ROOTS.items():
        if not root.exists():
            raise FileNotFoundError(f"Subtype folder missing: {root}")
        for patient_dir in sorted(root.iterdir()):
            if not patient_dir.is_dir():
                continue
            pid = extract_patient_id(patient_dir)
            mags_found = set()
            imgs: List[Path] = []
            for mag_dir in patient_dir.iterdir():
                if not mag_dir.is_dir():
                    continue
                if mag_dir.name not in MAGNIFICATIONS:
                    continue
                mags_found.add(mag_dir.name)
                imgs.extend(sorted(mag_dir.glob("*.png")))
            if not imgs:
                continue
            patients.append(PatientRecord(patient_id=pid, subtype=subtype, mags=mags_found, images=imgs))
    return patients


def stratified_patient_split(patients: List[PatientRecord]) -> tuple[List[PatientRecord], List[PatientRecord]]:
    total = len(patients)
    target_train = int(round(total * TRAIN_RATIO))
    # if expected 82 patients, force 56/26
    if total == 82:
        target_train = 56
    target_val = total - target_train

    by_subtype: Dict[str, List[PatientRecord]] = defaultdict(list)
    for rec in patients:
        by_subtype[rec.subtype].append(rec)

    train: List[PatientRecord] = []
    val: List[PatientRecord] = []

    # initial stratified split by subtype
    for subtype, recs in by_subtype.items():
        recs_shuffled = recs[:]
        random.shuffle(recs_shuffled)
        n_train = max(1, min(len(recs_shuffled) - 1, int(round(len(recs_shuffled) * TRAIN_RATIO))))
        train.extend(recs_shuffled[:n_train])
        val.extend(recs_shuffled[n_train:])

    # adjust to hit exact target counts
    def move_one(src: List[PatientRecord], dst: List[PatientRecord]) -> bool:
        if not src:
            return False
        rec = src.pop()
        dst.append(rec)
        return True

    while len(train) > target_train:
        if not move_one(train, val):
            break
    while len(train) < target_train and len(val) > target_val:
        if not move_one(val, train):
            break

    # ensure all magnifications appear in both splits by swapping patients if needed
    def mags_in_split(split: Sequence[PatientRecord]) -> set[str]:
        mags = set()
        for r in split:
            mags.update(r.mags)
        return mags

    def ensure_mags(train_split: List[PatientRecord], val_split: List[PatientRecord]) -> None:
        missing_train = MAGNIFICATIONS - mags_in_split(train_split)
        missing_val = MAGNIFICATIONS - mags_in_split(val_split)

        # move from val -> train if train is missing
        for mag in list(missing_train):
            donor = next((r for r in val_split if mag in r.mags), None)
            if donor:
                val_split.remove(donor)
                train_split.append(donor)
        # move from train -> val if val is missing
        for mag in list(missing_val):
            donor = next((r for r in train_split if mag in r.mags), None)
            if donor:
                train_split.remove(donor)
                val_split.append(donor)

    ensure_mags(train, val)

    # final sanity checks
    assert len(train) + len(val) == total, "Patient counts mismatch"
    assert len(train) == target_train, f"Train patients {len(train)} != target {target_train}"
    assert len(val) == target_val, f"Val patients {len(val)} != target {target_val}"
    assert mags_in_split(train) == MAGNIFICATIONS, f"Train missing magnification: {MAGNIFICATIONS - mags_in_split(train)}"
    assert mags_in_split(val) == MAGNIFICATIONS, f"Val missing magnification: {MAGNIFICATIONS - mags_in_split(val)}"
    return train, val


def copy_split(patients: List[PatientRecord], split_name: str) -> None:
    for rec in patients:
        for img_path in rec.images:
            dst_dir = OUT_ROOT / split_name / rec.subtype
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dst_dir / img_path.name)


def main() -> None:
    patients = collect_patients()
    print(f"[Info] Found {len(patients)} patients.")
    train, val = stratified_patient_split(patients)
    print(f"[Info] Train patients: {len(train)}, Val patients: {len(val)}")
    print(f"[Info] Train mags: {sorted({m for r in train for m in r.mags})}")
    print(f"[Info] Val mags: {sorted({m for r in val for m in r.mags})}")

    # Clear/create output root
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    (OUT_ROOT / "train").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "validation").mkdir(parents=True, exist_ok=True)

    copy_split(train, "train")
    copy_split(val, "validation")
    print(f"[Done] Wrote split to {OUT_ROOT}")


if __name__ == "__main__":
    main()
