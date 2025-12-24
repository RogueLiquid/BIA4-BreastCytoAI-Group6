#!/usr/bin/env python3
"""
pipeline_superpixels_sift_bof_fcm.py

MERGED script (minimum-change merge) of:
  - superpixels_slic_birch.py
  - sift_bof_features_fcm.py

It will:
  A) Compute SLIC-KMeans-BIRCH superpixels for all images (train/validation/test)
     and save label maps under BreakHis_400X_augmented_superpixels/

  B) Compute SIFT on reconstructed "segmented" image (mean color per region),
     build an FCM codebook on TRAIN, and save ONLY FCM outputs under BoF_cache_fcm/.

IMPORTANT: Keeps your current hyperparameters, including BIRCH_THRESHOLD = 0.1.
No argparse (safe for ipykernel).
"""

import os
from pathlib import Path
from typing import Iterable, List

import numpy as np

from skimage.io import imread
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.color import rgb2lab, rgb2gray
from skimage.feature import local_binary_pattern
from skimage import img_as_ubyte

from sklearn.cluster import KMeans, Birch

import cv2
import skfuzzy as fuzz  # Fuzzy C-Means


# ==========================
# CONFIGURATION (kept as-is)
# ==========================

REPO_ROOT = Path(__file__).resolve().parent  # <- merged file at repo root OR adjust if placed elsewhere

# Dataset root
DATA_ROOT = REPO_ROOT / "BreakHis_400X_full_augmented"  # train/, validation/, test/

# Superpixel output root
SUPERPIXEL_ROOT = REPO_ROOT / "BreakHis_400X_augmented_superpixels"

# FCM BoF output root (ONLY outputs you want)
CACHE_ROOT = REPO_ROOT / "BoF_cache_fcm"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "validation", "test"]
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

# ---- Superpixel hyperparams (unchanged) ----
N_SLIC_SEGMENTS = 400
SLIC_COMPACTNESS = 10.0
SLIC_SIGMA = 0.0

N_KMEANS_CLUSTERS = 400

N_BIRCH_CLUSTERS = 200
BIRCH_THRESHOLD = 0.1  # KEEP as requested

LBP_P = 8
LBP_R = 1
LBP_METHOD = "uniform"

# ---- FCM/BoF hyperparams (unchanged) ----
N_WORDS = 256
MAX_DESCRIPTORS = 100_000
FCM_M = 2.0
FCM_MAXITER = 300
FCM_ERROR = 1e-5

# ---- subtype -> class (unchanged) ----
SUBTYPE_TO_CLASS = {
    "adenosis": "benign",
    "fibroadenoma": "benign",
    "phyllodes_tumor": "benign",
    "tubular_adenoma": "benign",
    "ductal_carcinoma": "malignant",
    "lobular_carcinoma": "malignant",
    "mucinous_carcinoma": "malignant",
    "papillary_carcinoma": "malignant",
}
CLASS_NAMES = ["benign", "malignant"]


# ==========================
# IO UTILITIES
# ==========================

def read_image_rgb_u8(img_path: Path, size=(224, 224)) -> np.ndarray:
    img = imread(img_path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    img = img.astype(np.uint8)
    if size is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def iter_image_files(root: Path) -> Iterable[Path]:
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if any(fname.lower().endswith(ext) for ext in IMAGE_EXTS):
                yield Path(dirpath) / fname


def load_superpixel_labels_for_image(img_path: Path) -> np.ndarray:
    """
    IMAGE path is under DATA_ROOT. We mirror relative structure in SUPERPIXEL_ROOT.
    """
    rel = img_path.relative_to(DATA_ROOT)
    labels_path = SUPERPIXEL_ROOT / rel.parent / (img_path.stem + "_labels.npz")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found for image {img_path} at {labels_path}")
    return np.load(labels_path)["labels"]


# ==========================
# SUPERPIXEL PIPELINE (unchanged logic)
# ==========================

def slic_kmeans_birch(
    image: np.ndarray,
    n_slic_segments: int = N_SLIC_SEGMENTS,
    slic_compactness: float = SLIC_COMPACTNESS,
    n_kmeans_clusters: int = N_KMEANS_CLUSTERS,
    n_birch_clusters: int = N_BIRCH_CLUSTERS,
) -> np.ndarray:
    # Ensure RGB float in [0,1]
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    img_f = img_as_float(image)

    # Stage 1: SLIC in Lab
    lab = rgb2lab(img_f)
    labels_slic = slic(
        lab,
        n_segments=n_slic_segments,
        compactness=slic_compactness,
        sigma=SLIC_SIGMA,
        start_label=0,
    ).astype(np.int32)

    superpixel_ids = np.unique(labels_slic)

    # Stage 2: KMeans on mean Lab + centroid
    feats_kmeans = []
    for sid in superpixel_ids:
        mask = labels_slic == sid
        coords = np.column_stack(np.nonzero(mask))
        mean_rc = coords.mean(axis=0)
        mean_lab = lab[mask].mean(axis=0)
        feats_kmeans.append(np.concatenate([mean_lab, mean_rc]))
    feats_kmeans = np.vstack(feats_kmeans).astype(np.float32)

    effective_k = min(n_kmeans_clusters, len(superpixel_ids))
    kmeans = KMeans(n_clusters=effective_k, n_init="auto", random_state=0)
    kmeans_cluster_ids = kmeans.fit_predict(feats_kmeans)

    sp_to_kmeans = {int(sp): int(cid) for sp, cid in zip(superpixel_ids, kmeans_cluster_ids)}

    # Stage 3: BIRCH on LBP hist + KMeans id scalar
    gray = rgb2gray(img_f)
    gray_u8 = img_as_ubyte(gray)
    lbp = local_binary_pattern(gray_u8, P=LBP_P, R=LBP_R, method=LBP_METHOD)

    n_lbp_bins = (LBP_P + 2) if LBP_METHOD == "uniform" else (2 ** LBP_P)

    feats_birch = []
    for sid in superpixel_ids:
        mask = labels_slic == sid
        lbp_vals = lbp[mask].ravel()
        hist, _ = np.histogram(
            lbp_vals,
            bins=np.arange(n_lbp_bins + 1),
            density=True,
        )
        hist = hist.astype(np.float32)

        km_id = sp_to_kmeans[int(sid)]
        km_scalar = km_id / max(1, (effective_k - 1))
        km_feat = np.array([km_scalar], dtype=np.float32)

        feats_birch.append(np.concatenate([hist, km_feat]))

    feats_birch = np.vstack(feats_birch)

    # KEEP your behavior: n_clusters=None + threshold=0.1, and cap if too many clusters
    birch = Birch(n_clusters=None, threshold=BIRCH_THRESHOLD)
    birch_ids = birch.fit_predict(feats_birch)

    unique_ids = np.unique(birch_ids)
    if len(unique_ids) > n_birch_clusters:
        centers = birch.subcluster_centers_
        km2 = KMeans(n_clusters=n_birch_clusters, n_init="auto", random_state=0)
        big_labels = km2.fit_predict(centers)
        birch_ids = big_labels[birch_ids]

    sp_to_birch = {int(sp): int(cid) for sp, cid in zip(superpixel_ids, birch_ids)}
    merged_labels = np.vectorize(sp_to_birch.get, otypes=[np.int32])(labels_slic)
    return merged_labels


def compute_and_save_superpixels_for_split(split: str) -> None:
    split_root = DATA_ROOT / split
    if not split_root.exists():
        print(f"[WARN] Split folder '{split_root}' not found, skipping.")
        return

    img_paths = list(iter_image_files(split_root))
    print(f"[{split}] Found {len(img_paths)} images under {split_root}")

    for idx, img_path in enumerate(img_paths, start=1):
        rel = img_path.relative_to(DATA_ROOT)
        out_dir = SUPERPIXEL_ROOT / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / (img_path.stem + "_labels.npz")
        if out_path.exists():
            # keep minimal output noise
            if idx % 200 == 0:
                print(f"[{split} {idx}/{len(img_paths)}] exists, skipping ...")
            continue

        img = read_image_rgb_u8(img_path)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)

        labels = slic_kmeans_birch(img)
        np.savez_compressed(out_path, labels=labels)
        if idx % 50 == 0 or idx == 1:
            print(f"[{split} {idx}/{len(img_paths)}] Saved {out_path.name}")


# ==========================
# SIFT on segmented image + FCM + BoF (unchanged intent)
# ==========================

def iter_image_files_list(root: Path) -> List[Path]:
    return sorted(list(iter_image_files(root)))


def reconstruct_segmented_rgb(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)

    img = image.astype(np.float32)
    seg = np.zeros_like(img, dtype=np.float32)

    for sid in np.unique(labels):
        mask = labels == sid
        if not np.any(mask):
            continue
        mean_rgb = img[mask].mean(axis=0)
        seg[mask] = mean_rgb

    return np.clip(seg, 0.0, 255.0).astype(np.uint8)


def extract_sift_descriptors_segmented(image: np.ndarray, labels: np.ndarray) -> np.ndarray | None:
    """
    Paper wording: SIFT on *segmented images* (Step 1).
    Here: we use your existing segmented reconstruction (mean RGB per superpixel),
    then run global SIFT on that segmented image.
    """
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)

    # <-- IMPORTANT CHANGE: use segmented reconstruction
    seg_rgb = reconstruct_segmented_rgb(image, labels)
    gray = cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, mask=None)

    if des is None or len(des) == 0:
        return None
    return des.astype(np.float32)


def build_fcm_codebook_from_train(train_paths: List[Path]) -> np.ndarray:
    descriptor_list: list[np.ndarray] = []

    for idx, img_path in enumerate(train_paths, start=1):
        img = read_image_rgb_u8(img_path)
        labels = load_superpixel_labels_for_image(img_path)
        des = extract_sift_descriptors_segmented(img, labels)
        if des is not None and len(des) > 0:
            descriptor_list.append(des)
        if idx % 50 == 0 or idx == 1:
            print(f"[FCM] train {idx}/{len(train_paths)} processed")

    if not descriptor_list:
        raise RuntimeError("No SIFT descriptors found in training set.")

    all_desc = np.vstack(descriptor_list).astype(np.float32)
    print(f"[FCM] Total descriptors before subsampling: {all_desc.shape[0]}")

    if all_desc.shape[0] > MAX_DESCRIPTORS:
        sel = np.random.choice(all_desc.shape[0], MAX_DESCRIPTORS, replace=False)
        all_desc = all_desc[sel]
        print(f"[FCM] Subsampled to {all_desc.shape[0]} descriptors")

    data = all_desc.T  # [128, N]
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data=data,
        c=N_WORDS,
        m=FCM_M,
        error=FCM_ERROR,
        maxiter=FCM_MAXITER,
        init=None,
        seed=0,
    )
    print(f"[FCM] centers: {cntr.shape}, FPC={fpc:.4f}")
    return cntr.astype(np.float32)


def assign_descriptors_to_codebook(descriptors: np.ndarray, centers: np.ndarray) -> np.ndarray:
    diff = descriptors[:, None, :] - centers[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    return np.argmin(d2, axis=1).astype(np.int64)


def bof_histogram_for_image(img_path: Path, centers: np.ndarray) -> np.ndarray:
    img = read_image_rgb_u8(img_path)
    labels = load_superpixel_labels_for_image(img_path)
    des = extract_sift_descriptors_segmented(img, labels)

    n_words = centers.shape[0]
    if des is None or len(des) == 0:
        return np.zeros(n_words, dtype=np.float32)

    word_ids = assign_descriptors_to_codebook(des, centers)
    hist, _ = np.histogram(word_ids, bins=np.arange(n_words + 1), density=False)
    hist = hist.astype(np.float32)

    return hist




def build_bof_for_split(split: str, img_paths: List[Path], centers: np.ndarray) -> None:
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    path_list: list[str] = []

    for idx, img_path in enumerate(img_paths, start=1):
        hist = bof_histogram_for_image(img_path, centers)
        X_list.append(hist)
        path_list.append(str(img_path))

        subtype = img_path.parent.name
        class_name = SUBTYPE_TO_CLASS.get(subtype, None)
        if class_name is None:
            raise ValueError(f"Unknown subtype folder: {subtype}")
        y_list.append(CLASS_NAMES.index(class_name))

        if idx % 100 == 0 or idx == 1:
            print(f"[BoF {split}] {idx}/{len(img_paths)}")

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    paths_arr = np.array(path_list)

    out_path = CACHE_ROOT / f"bof_features_{split}.npz"
    np.savez_compressed(out_path, X=X, y=y, paths=paths_arr)
    print(f"[BoF {split}] saved -> {out_path} | X={X.shape}, y={y.shape}")


# ==========================
# MAIN
# ==========================

if __name__ == "__main__":
    print(f"REPO_ROOT       = {REPO_ROOT}")
    print(f"DATA_ROOT       = {DATA_ROOT}")
    print(f"SUPERPIXEL_ROOT = {SUPERPIXEL_ROOT}")
    print(f"CACHE_ROOT      = {CACHE_ROOT}")
    print(f"BIRCH_THRESHOLD = {BIRCH_THRESHOLD} (kept)")

    # A) superpixels
    for split in SPLITS:
        compute_and_save_superpixels_for_split(split)

    # B) FCM BoF (ONLY outputs requested)
    train_root = DATA_ROOT / "train"
    val_root = DATA_ROOT / "validation"
    test_root = DATA_ROOT / "test"

    train_paths = iter_image_files_list(train_root) if train_root.exists() else []
    val_paths = iter_image_files_list(val_root) if val_root.exists() else []
    test_paths = iter_image_files_list(test_root) if test_root.exists() else []

    print(f"[Paths] train={len(train_paths)} val={len(val_paths)} test={len(test_paths)}")
    if len(train_paths) == 0:
        raise RuntimeError("No training images found under BreakHis_400X_full_augmented/train")

    centers = build_fcm_codebook_from_train(train_paths)
    codebook_path = CACHE_ROOT / "codebook_fcm_centers.npz"
    np.savez_compressed(codebook_path, centers=centers)
    print(f"[FCM] codebook saved -> {codebook_path}")

    build_bof_for_split("train", train_paths, centers)
    if len(val_paths) > 0:
        build_bof_for_split("validation", val_paths, centers)
    if len(test_paths) > 0:
        build_bof_for_split("test", test_paths, centers)

    print("DONE: superpixels + SIFT-BoF-FCM")
