"""
Utilities for creating and reusing consistent train/val/test splits
from the original BreakHis dataset (400X).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit


@dataclass
class SplitRecord:
    path: str
    label: int
    patient_id: str


def collect_records(image_root: Path, magnification: str = "400X") -> Tuple[List[SplitRecord], List[str]]:
    """
    Walk BreakHis folder and collect all images for the given magnification.
    Returns (records, class_names).
    """
    class_names = ["benign", "malignant"]
    records: List[SplitRecord] = []

    for class_idx, class_name in enumerate(class_names):
        class_root = image_root / class_name / "SOB"
        if not class_root.exists():
            continue

        for subtype_dir in class_root.iterdir():
            if not subtype_dir.is_dir():
                continue
            for patient_dir in subtype_dir.iterdir():
                if not patient_dir.is_dir():
                    continue
                mag_dir = patient_dir / magnification
                if not mag_dir.exists():
                    continue
                for img_path in mag_dir.glob("*.png"):
                    records.append(
                        SplitRecord(
                            path=str(img_path),
                            label=class_idx,
                            patient_id=patient_dir.name,
                        )
                    )

    if not records:
        raise RuntimeError(f"No images found under {image_root} for {magnification}")

    return records, class_names


def _generate_single_split(
    labels: np.ndarray,
    groups: np.ndarray | None,
    split_mode: str,
    test_ratio: float,
    val_ratio: float,
    seed: int,
    split_idx: int,
) -> Tuple[List[int], List[int], List[int]]:
    """Generate one train/val/test split and return index lists."""
    if split_mode == "patient":
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_ratio,
            random_state=seed + split_idx,
        )
        train_val_idx, test_idx = next(splitter.split(labels, labels, groups=groups))

        # validation from remaining train/val pool
        val_splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=val_ratio / (1 - test_ratio),
            random_state=seed + 1000 + split_idx,
        )
        tr_rel_idx, val_rel_idx = next(
            val_splitter.split(
                labels[train_val_idx],
                labels[train_val_idx],
                groups=groups[train_val_idx],
            )
        )
        train_idx = np.array(train_val_idx)[tr_rel_idx].tolist()
        val_idx = np.array(train_val_idx)[val_rel_idx].tolist()
        test_idx = np.array(test_idx).tolist()
    else:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_ratio,
            random_state=seed + split_idx,
        )
        train_val_idx, test_idx = next(splitter.split(np.zeros_like(labels), labels))

        val_splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_ratio / (1 - test_ratio),
            random_state=seed + 1000 + split_idx,
        )
        tr_rel_idx, val_rel_idx = next(
            val_splitter.split(
                np.zeros_like(labels[train_val_idx]),
                labels[train_val_idx],
            )
        )
        train_idx = np.array(train_val_idx)[tr_rel_idx].tolist()
        val_idx = np.array(train_val_idx)[val_rel_idx].tolist()
        test_idx = np.array(test_idx).tolist()

    return train_idx, val_idx, test_idx


def generate_splits(
    records: List[SplitRecord],
    split_mode: str,
    *,
    n_splits: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict:
    """Generate multiple consistent splits and return a serializable payload."""
    if split_mode not in {"patient", "image"}:
        raise ValueError("split_mode must be 'patient' or 'image'")
    if val_ratio <= 0 or test_ratio <= 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be >0 and sum to <1")

    labels = np.array([r.label for r in records], dtype=np.int64)
    groups = np.array([r.patient_id for r in records]) if split_mode == "patient" else None

    splits: List[Dict[str, List[int]]] = []
    for split_idx in range(n_splits):
        train_idx, val_idx, test_idx = _generate_single_split(
            labels=labels,
            groups=groups,
            split_mode=split_mode,
            test_ratio=test_ratio,
            val_ratio=val_ratio,
            seed=seed,
            split_idx=split_idx,
        )
        splits.append(
            {
                "train": train_idx,
                "val": val_idx,
                "test": test_idx,
            }
        )

    payload = {
        "meta": {
            "created": datetime.now(timezone.utc).isoformat(),
            "split_mode": split_mode,
            "n_splits": n_splits,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "seed": seed,
        },
        "records": [asdict(r) for r in records],
        "splits": splits,
    }
    return payload


def default_splits_path(
    final_root: Path,
    *,
    split_mode: str,
    n_splits: int,
    seed: int,
    magnification: str,
) -> Path:
    """Compute default path under final/ for storing splits."""
    fname = f"splits_{split_mode}_{magnification}_{n_splits}fold_seed{seed}.json"
    return final_root / "splits" / fname


def save_splits(payload: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_splits(path: Path, repo_root: Path) -> Tuple[List[SplitRecord], List[Dict[str, List[int]]], Dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    records = [
        SplitRecord(
            path=str(repo_root / rec["path"]) if not os.path.isabs(rec["path"]) else rec["path"],
            label=rec["label"],
            patient_id=rec["patient_id"],
        )
        for rec in payload["records"]
    ]
    return records, payload["splits"], payload["meta"]


def ensure_splits_on_disk(
    *,
    final_root: Path,
    repo_root: Path,
    image_root: Path,
    split_mode: str,
    n_splits: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    magnification: str = "400X",
    splits_path: Path | None = None,
) -> Path:
    """
    Ensure a splits file exists; generate and persist if missing.
    Returns the path to the splits file.
    """
    target_path = splits_path or default_splits_path(
        final_root,
        split_mode=split_mode,
        n_splits=n_splits,
        seed=seed,
        magnification=magnification,
    )

    if target_path.exists():
        return target_path

    records, class_names = collect_records(image_root, magnification=magnification)
    payload = generate_splits(
        records,
        split_mode=split_mode,
        n_splits=n_splits,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    # store paths relative to repo root for portability
    for rec in payload["records"]:
        rec["path"] = str(Path(rec["path"]).resolve().relative_to(repo_root))
    payload["meta"]["class_names"] = class_names
    payload["meta"]["magnification"] = magnification
    payload["meta"]["image_root"] = str(image_root)

    save_splits(payload, target_path)
    return target_path


def _default_paths() -> Tuple[Path, Path, Path]:
    """Convenience helper to locate repo_root, final_root, and image_root."""
    final_root = Path(__file__).resolve().parent
    repo_root = final_root.parent
    image_root = (
        repo_root
        / "BreakHis"
        / "BreaKHis_v1"
        / "BreaKHis_v1"
        / "histology_slides"
        / "breast"
    )
    return repo_root, final_root, image_root


if __name__ == "__main__":
    """
    Quick script entrypoint to generate and persist 5-fold splits for both
    patient-level and image-level at 400X using 8:1:1 splits (val/test = 0.1).

    Run (from repo root):
        python -m final.split_utils
    """
    repo_root, final_root, image_root = _default_paths()
    n_splits = 5
    val_ratio = 0.1
    test_ratio = 0.1
    seed = 124
    magnification = "400X"

    for split_mode in ("patient", "image"):
        path = ensure_splits_on_disk(
            final_root=final_root,
            repo_root=repo_root,
            image_root=image_root,
            split_mode=split_mode,
            n_splits=n_splits,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            magnification=magnification,
            splits_path=None,
        )
        print(f"Saved {split_mode} splits to: {path}")
