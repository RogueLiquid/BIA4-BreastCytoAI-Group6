#!/usr/bin/env python3
"""
sift_bof_features_fcm.py

Strict implementation (as close as practicable) of Section 3.5 in:

  Manivannan et al., "An integrated breast cancer detection approach
  using modified SLIC-K-Means-BIRCH superpixel segmentation, SIFT with
  BoF features, and deep learning", Int. J. Comput. Intell. Syst., 2025.

Pipeline here:

  1) Use saved SLIC-KMeans-BIRCH superpixels + original BreakHis images
     (non-augmented + augmented) to compute SIFT descriptors on segmented
     regions.

  2) Build a visual codebook using **Fuzzy C-Means (FCM)** on TRAIN
     descriptors only (as described in Step 2 of Section 3.5).

  3) For each image (train/val/test), build a Bag-of-Features (BoF)
     histogram by assigning each SIFT descriptor to the nearest FCM
     cluster center (Step 3–4 in Section 3.5).

Outputs:

    BoF_cache_fcm/
        codebook_fcm_centers.npz   # "centers" of shape [g, 128]
        bof_features_train.npz     # X, y, paths
        bof_features_validation.npz
        bof_features_test.npz      # if test exists

Hyperparameters **NOT specified** in the paper (chosen by us):

    - N_WORDS (g, codebook size): default 256.
    - FCM_M (fuzziness exponent m): default 2.0.
    - FCM_MAXITER, FCM_ERROR.
    - MAX_DESCRIPTORS (max descriptors used to train codebook).
    - SIFT detector parameters (OpenCV defaults).
    - Using TRAIN-only descriptors to learn codebook (standard practice).
"""

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from skimage.io import imread

import cv2
import joblib
import skfuzzy as fuzz  # Fuzzy C-Means

# ==========================
# CONFIGURATION
# ==========================

REPO_ROOT = Path(__file__).resolve().parent.parent

IMAGE_ROOT = REPO_ROOT / "BreakHis_400X_full_augmented"       # has train/, validation/, test/
SUPERPIXEL_ROOT = REPO_ROOT / "BreakHis_400X_augmented_superpixels"
CACHE_ROOT = REPO_ROOT / "BoF_cache_fcm"

CACHE_ROOT.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "validation", "test"]  # will skip if a split folder doesn't exist

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

# --------- BoF / codebook hyperparameters (NOT given by the paper) ---------
N_WORDS = 256             # codebook size g (number of visual words C_q)
MAX_DESCRIPTORS = 100_000 # cap for descriptors used to train FCM
# Fuzzy C-Means parameters (standard defaults)
FCM_M = 2.0               # fuzziness exponent m
FCM_MAXITER = 300
FCM_ERROR = 1e-5

# Optional: mapping from BreakHis subtypes to binary classes
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


# ==========================
# UTILITIES
# ==========================

def iter_image_files(root: Path) -> List[Path]:
    """Collect all image paths under root (recursively)."""
    out = []
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


def reconstruct_segmented_rgb(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Build a 'segmented image' by replacing each superpixel with its mean RGB color.

    This corresponds to the 'segmented images' described in the paper:
    SLIC-K-Means-BIRCH partitions the image, then SIFT is applied on the
    segmented image rather than on raw noisy pixels.

    Args
    ----
    image : [H, W, 3] uint8 or float
    labels: [H, W] int (superpixel id per pixel)

    Returns
    -------
    seg_rgb : [H, W, 3] uint8 segmented image.
    """
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)

    # work in float for averaging
    img = image.astype(np.float32)
    seg = np.zeros_like(img, dtype=np.float32)

    for sid in np.unique(labels):
        mask = labels == sid
        if not np.any(mask):
            continue
        mean_rgb = img[mask].mean(axis=0)  # [3]
        seg[mask] = mean_rgb

    seg = np.clip(seg, 0.0, 255.0).astype(np.uint8)
    return seg


def extract_sift_descriptors_segmented(
    image: np.ndarray,
    superpixel_labels: np.ndarray,
) -> np.ndarray | None:
    """
    Extract SIFT descriptors from the *segmented* image, as described in the paper.

    - First, reconstruct the segmented RGB image by replacing each
      superpixel with its mean RGB color.
    - Then run SIFT once on this segmented image (no per-superpixel mask).
    - Each keypoint yields a 128-D descriptor.

    image: RGB [H,W,3] uint8
    superpixel_labels: [H,W] int

    Returns:
        descriptors: [N,128] float32 or None if no keypoints found.
    """
    if image.ndim == 2:  # grayscale -> fake RGB
        image = np.stack([image, image, image], axis=-1)

    seg_rgb = reconstruct_segmented_rgb(image, superpixel_labels)
    gray = cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2GRAY)

    # SIFT parameters are NOT specified in the paper; we use OpenCV defaults.
    sift = cv2.SIFT_create()
    keypoints, des = sift.detectAndCompute(gray, mask=None)

    if des is None or len(des) == 0:
        return None

    return des.astype(np.float32)  # [N,128]


# ==========================
# FCM CODEBOOK (Step 2 in paper)
# ==========================

def build_fcm_codebook_from_train(
    train_image_paths: List[Path],
    max_descriptors: int = MAX_DESCRIPTORS,
    n_words: int = N_WORDS,
) -> np.ndarray:
    """
    Collect SIFT descriptors from TRAIN images and train an FCM codebook
    (visual vocabulary) as described in Section 3.5 Step 2.

    The paper states:
      "Utilize Fuzzy C-Means (FCM) clustering to group the SIFT descriptors
       that were extracted for creating a visual codebook (vocabulary)
       collection of the clusters C_q, where g indicates the size of the
       codebook."

    Here:
      - g = n_words (hyperparameter, not specified in the paper).
      - FCM parameters (m, maxiter, error) are also not specified in the paper
        and are set to standard defaults.
    """
    descriptor_list: list[np.ndarray] = []

    for idx, img_path in enumerate(train_image_paths, start=1):
        img = imread(img_path)
        labels = load_superpixel_labels_for_image(img_path)
        des = extract_sift_descriptors_segmented(img, labels)
        if des is not None and len(des) > 0:
            descriptor_list.append(des)
        print(f"[Codebook-FCM] Processed train image {idx}/{len(train_image_paths)}: {img_path.name}")

    if not descriptor_list:
        raise RuntimeError("No SIFT descriptors found in training set; check images / SIFT installation.")

    all_desc = np.vstack(descriptor_list).astype(np.float32)
    print(f"[Codebook-FCM] Total descriptors before subsampling: {all_desc.shape[0]}")

    # For computational reasons, subsample descriptors if too many
    if all_desc.shape[0] > max_descriptors:
        idx = np.random.choice(all_desc.shape[0], max_descriptors, replace=False)
        all_desc = all_desc[idx]
        print(f"[Codebook-FCM] Subsampled to {all_desc.shape[0]} descriptors for FCM.")

    # FCM expects data shape (features, samples)
    data = all_desc.T  # [128, N_samples]

    print("[Codebook-FCM] Running Fuzzy C-Means clustering ...")
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data=data,
        c=n_words,
        m=FCM_M,
        error=FCM_ERROR,
        maxiter=FCM_MAXITER,
        init=None,
        seed=0,  # reproducibility
    )
    # cntr: [n_words, 128] = cluster centers C_q
    print("[Codebook-FCM] FCM completed.")
    print(f"[Codebook-FCM] Final codebook size (g) = {cntr.shape[0]} visual words.")
    print(f"[Codebook-FCM] Fuzzy partition coefficient (FPC) = {fpc:.4f}")

    return cntr.astype(np.float32)  # [g, 128]


def assign_descriptors_to_codebook(
    descriptors: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    """
    Step 3 in the paper:
      "Assign individual SIFT descriptor Dep to the nearest cluster C_q
       in the codebook. Here continuous descriptors are converted into
       discrete visual codes, using Euclidean distance."

    This is a hard assignment equivalent to:
      q* = argmin_q ||Dep - C_q||^2

    Args:
        descriptors: [N, 128]
        centers:     [g, 128]

    Returns:
        word_ids: [N] int, indices in [0, g-1]
    """
    # Compute squared Euclidean distance to each center
    # descriptors: [N, 128], centers: [g, 128]
    # -> diff: [N, g, 128]
    diff = descriptors[:, None, :] - centers[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)  # [N, g]
    word_ids = np.argmin(d2, axis=1).astype(np.int64)
    return word_ids


# ==========================
# BoF HISTOGRAMS (Step 4)
# ==========================

def bof_histogram_for_image(
    img_path: Path,
    centers: np.ndarray,
) -> np.ndarray:
    """
    Compute a Bag-of-Features histogram for a single image using the FCM
    codebook centers.

    Step 4 in the paper:
      "When creating histograms for individual breast cancer segmented images,
       bin h_q counts the descriptors assigned to the q-th visual word."

    Here we implement:
        h_q = number of descriptors in this image whose nearest center
              is C_q (hard assignment), followed by L1 normalization.
    """
    img = imread(img_path)
    labels = load_superpixel_labels_for_image(img_path)
    des = extract_sift_descriptors_segmented(img, labels)

    n_words = centers.shape[0]
    if des is None or len(des) == 0:
        # No keypoints: return all-zero histogram
        return np.zeros(n_words, dtype=np.float32)

    word_ids = assign_descriptors_to_codebook(des, centers)  # [N_descriptors]
    hist, _ = np.histogram(word_ids, bins=np.arange(n_words + 1), density=False)
    hist = hist.astype(np.float32)

    # L1 normalization (not explicitly specified, but standard in BoF)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


def build_bof_for_split(
    split: str,
    img_paths: List[Path],
    centers: np.ndarray,
) -> None:
    """
    Build and save BoF features for all images in a given split.

    Saves:
        BoF_cache_fcm/bof_features_<split>.npz

    Contents:
        X: [N_images, g] float32 BoF histograms
        y: [N_images] int64 labels (0: benign, 1: malignant)
        paths: [N_images] str, relative image paths
    """
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    path_list: list[str] = []

    for idx, img_path in enumerate(img_paths, start=1):
        hist = bof_histogram_for_image(img_path, centers)
        X_list.append(hist)
        path_list.append(str(img_path))

        # determine subtype and binary class
        subtype = img_path.parent.name
        class_name = SUBTYPE_TO_CLASS.get(subtype, None)
        if class_name is None:
            raise ValueError(f"Unknown subtype folder: {subtype}")
        label_idx = CLASS_NAMES.index(class_name)
        y_list.append(label_idx)

        print(f"[BoF-FCM {split}] {idx}/{len(img_paths)}: hist shape = {hist.shape}")

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    paths_arr = np.array(path_list)

    out_path = CACHE_ROOT / f"bof_features_{split}.npz"
    np.savez_compressed(out_path, X=X, y=y, paths=paths_arr)
    print(f"[BoF-FCM {split}] Saved features to {out_path}")
    print(f"[BoF-FCM {split}] X shape = {X.shape}, y shape = {y.shape}")


# ==========================
# MAIN
# ==========================

if __name__ == "__main__":
    print(f"IMAGE_ROOT      = {IMAGE_ROOT}")
    print(f"SUPERPIXEL_ROOT = {SUPERPIXEL_ROOT}")
    print(f"CACHE_ROOT      = {CACHE_ROOT}")
    print(f"N_WORDS (g)     = {N_WORDS}")
    print(f"FCM_M           = {FCM_M}, FCM_MAXITER = {FCM_MAXITER}, FCM_ERROR = {FCM_ERROR}")
    print(f"MAX_DESCRIPTORS = {MAX_DESCRIPTORS}")

    # 1) Collect train / val / test paths
    train_root = IMAGE_ROOT / "train"
    val_root = IMAGE_ROOT / "validation"
    test_root = IMAGE_ROOT / "test"

    train_paths = iter_image_files(train_root) if train_root.exists() else []
    val_paths = iter_image_files(val_root) if val_root.exists() else []
    test_paths = iter_image_files(test_root) if test_root.exists() else []

    print(f"[Paths] Train images: {len(train_paths)}")
    print(f"[Paths] Val images:   {len(val_paths)}")
    print(f"[Paths] Test images:  {len(test_paths)}")

    if len(train_paths) == 0:
        raise RuntimeError("No training images found under BreakHis_400X_full_augmented/train")

    # 2) Build FCM codebook from TRAIN only (per typical BoF practice)
    centers = build_fcm_codebook_from_train(train_paths)
    codebook_path = CACHE_ROOT / "codebook_fcm_centers.npz"
    np.savez_compressed(codebook_path, centers=centers)
    print(f"[BoF-FCM] Saved codebook centers to {codebook_path}")

    # 3) Build BoF features for each split
    build_bof_for_split("train", train_paths, centers)
    if len(val_paths) > 0:
        build_bof_for_split("validation", val_paths, centers)
    if len(test_paths) > 0:
        build_bof_for_split("test", test_paths, centers)

    print("SIFT + BoF (FCM) feature extraction finished.")
