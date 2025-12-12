#!/usr/bin/env python3
"""
superpixels_slic_birch.py

Three-stage SLIC-K-Means-BIRCH superpixel segmentation for BreakHis 400X images,
following the structure described in Manivannan et al. (2025), but WITHOUT the
Adaptive Fuzzy Filter.

Pipeline:
    Stage 1: SLIC superpixels in CIELAB space.
    Stage 2: K-Means clustering of superpixel-level color+position features.
    Stage 3: BIRCH clustering of superpixel texture traits ("binary arrangements")
             approximated via LBP histograms + K-Means cluster indicator.

Optional:
    If ground-truth superpixel/segmentation labels are available, compute
    superpixel metrics:
        - ASA  (Achievable Segmentation Accuracy)
        - USE  (Undersegmentation Error)
        - SSE  (Sum of Squared Error in Lab space)
        - C    (Compactness)
        - RI   (Rand Index)

Dataset layout (relative to repo root):

    BreakHis_400X_full_augmented/
        train/<subtype>/*.png
        validation/<subtype>/*.png
        test/<subtype>/*.png   # typically no augmentation

Outputs:

    BreakHis_400X_augmented_superpixels/
        train/<subtype>/*_labels.npz
        validation/<subtype>/*_labels.npz
        test/<subtype>/*_labels.npz

NOTE on augmentation:
    We process BOTH original and augmented images.
    For metrics, we usually only aggregate over ORIGINAL images
    (filenames NOT containing '_augment').
"""

import os
from pathlib import Path
from typing import Iterable

import numpy as np
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.io import imread
from skimage.color import rgb2lab, rgb2gray
from skimage.feature import local_binary_pattern
from skimage.measure import regionprops
from sklearn.cluster import KMeans, Birch
from skimage import img_as_ubyte


# ==========================
# CONFIGURATION
# ==========================

REPO_ROOT = Path(__file__).resolve().parent.parent

# Where the original BreakHis images live (400X augmented only).
DATA_ROOT = REPO_ROOT / "BreakHis_400X_full_augmented"   # has train/, validation/, test/

# Where to save superpixel label maps
SUPERPIXEL_ROOT = REPO_ROOT / "BreakHis_400X_augmented_superpixels"

# We include test; code will skip missing splits automatically.
SPLITS = ["train", "validation", "test"]

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

# ========== Hyperparameters NOT explicitly specified in the paper ==========

# SLIC parameters (paper: SLIC in CIELAB, balance color/spatial distance).
# Numeric values below are chosen by us.
N_SLIC_SEGMENTS = 400       # initial SLIC superpixels
SLIC_COMPACTNESS = 10.0     # m in distance formula
SLIC_SIGMA = 0.0            # no extra Gaussian smoothing

# K-Means parameters (stage 2).
# Number of clusters over superpixels; must be <= number of SLIC regions.
N_KMEANS_CLUSTERS = 400

# BIRCH parameters (stage 3).
# Final number of merged regions.
N_BIRCH_CLUSTERS = 200
BIRCH_THRESHOLD = 0.1

# LBP (texture / "binary arrangement") parameters for BIRCH features.
# LBP is used to encode per-superpixel binary texture patterns.
LBP_P = 8                   # number of circular neighbors
LBP_R = 1                   # radius
LBP_METHOD = "uniform"      # gives P+2 bins

# Ground-truth root for optional metric computation.
# If you later create GT masks, put them here with the same relative paths as
# the images and file names like <basename>_gt_labels.npz (array "labels").
GT_ROOT = REPO_ROOT / "BreakHis_400X_superpixels_gt"

# Enable/disable metric computation globally.
COMPUTE_METRICS = False     # set to True only if you have GT label maps


# ==========================
# CORE SLIC-KMEANS-BIRCH
# ==========================

def slic_kmeans_birch(
    image: np.ndarray,
    n_slic_segments: int = N_SLIC_SEGMENTS,
    slic_compactness: float = SLIC_COMPACTNESS,
    n_kmeans_clusters: int = N_KMEANS_CLUSTERS,
    n_birch_clusters: int = N_BIRCH_CLUSTERS,
) -> np.ndarray:
    """
    Three-stage SLIC-K-Means-BIRCH superpixel segmentation.

    Stage 1: SLIC superpixels in CIELAB space.
    Stage 2: K-Means clustering of superpixel-level color+position features.
    Stage 3: BIRCH clustering of superpixel texture traits ("binary arrangements")
             approximated via LBP histograms + K-Means cluster indicator.

    Args
    ----
    image : np.ndarray
        RGB image [H, W, 3], uint8 or float. Assumed already resized
        (e.g. 224x224) BEFORE calling this function if you want strict size
        control (resize is not saved to disk).
    n_slic_segments : int
        Desired number of initial SLIC superpixels (NOT specified by paper).
    slic_compactness : float
        SLIC compactness parameter (NOT specified by paper).
    n_kmeans_clusters : int
        Number of K-Means clusters over superpixels (NOT specified by paper).
    n_birch_clusters : int
        Number of BIRCH clusters over texture traits (NOT specified by paper).

    Returns
    -------
    labels : np.ndarray
        [H, W] int32 array; final region ID (0..K-1) after BIRCH merging.
    """
    # Ensure RGB float in [0,1]
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    img_f = img_as_float(image)

    # -------------------------
    # Stage 1: SLIC in CIELAB
    # -------------------------
    lab = rgb2lab(img_f)  # [H, W, 3]

    labels_slic = slic(
        lab,
        n_segments=n_slic_segments,
        compactness=slic_compactness,
        sigma=SLIC_SIGMA,
        start_label=0,
    ).astype(np.int32)            # [H, W], labels 0..K-1

    superpixel_ids = np.unique(labels_slic)

    # ------------------------------------------------
    # Stage 2: K-Means on superpixel "characteristics"
    #         (mean Lab color + centroid coordinates)
    # ------------------------------------------------
    feats_kmeans = []  # [num_superpixels, 5]
    for sid in superpixel_ids:
        mask = labels_slic == sid
        coords = np.column_stack(np.nonzero(mask))   # (row, col)
        mean_rc = coords.mean(axis=0)               # [2]
        mean_lab = lab[mask].mean(axis=0)           # [3]
        feats_kmeans.append(np.concatenate([mean_lab, mean_rc]))

    feats_kmeans = np.vstack(feats_kmeans).astype(np.float32)
    effective_k = min(n_kmeans_clusters, len(superpixel_ids))

    kmeans = KMeans(
        n_clusters=effective_k,
        n_init="auto",      # CHOSEN by us, not specified in paper
        random_state=0,     # CHOSEN by us for reproducibility
    )
    kmeans_cluster_ids = kmeans.fit_predict(feats_kmeans)  # [num_superpixels]

    # Map superpixel ID -> K-Means cluster index
    sp_to_kmeans = {
        int(sp): int(cid)
        for sp, cid in zip(superpixel_ids, kmeans_cluster_ids)
    }

    # -----------------------------------------------------------
    # Stage 3: BIRCH on textured traits / "binary arrangements"
    #          (LBP histogram + normalized K-Means cluster index)
    # -----------------------------------------------------------
    gray = rgb2gray(img_f)
    gray_u8 = img_as_ubyte(gray)  # convert to uint8 [0, 255]
    lbp = local_binary_pattern(gray_u8, P=LBP_P, R=LBP_R, method=LBP_METHOD)

    if LBP_METHOD == "uniform":
        n_lbp_bins = LBP_P + 2
    else:
        n_lbp_bins = 2 ** LBP_P

    feats_birch = []  # per-superpixel texture+cluster feature
    for sid in superpixel_ids:
        mask = labels_slic == sid

        # LBP histogram (normalized) for texture traits
        lbp_vals = lbp[mask].ravel()
        hist, _ = np.histogram(
            lbp_vals,
            bins=np.arange(n_lbp_bins + 1),
            density=True,  # probability distribution
        )
        hist = hist.astype(np.float32)

        # Encode which K-Means cluster the superpixel belongs to
        km_id = sp_to_kmeans[int(sid)]
        km_scalar = km_id / max(1, (effective_k - 1))  # normalize to [0,1]
        km_feat = np.array([km_scalar], dtype=np.float32)

        feats_birch.append(np.concatenate([hist, km_feat]))

    feats_birch = np.vstack(feats_birch)  # [num_superpixels, n_lbp_bins+1]

    # First, let BIRCH find its own number of clusters (leaf subclusters)
    birch = Birch(n_clusters=None, threshold=BIRCH_THRESHOLD)
    birch_ids = birch.fit_predict(feats_birch)  # [num_superpixels]

    # Optionally cap the number of clusters to N_BIRCH_CLUSTERS
    unique_ids = np.unique(birch_ids)
    if len(unique_ids) > n_birch_clusters:
        # Compress subclusters to n_birch_clusters using K-Means on subcluster centers
        centers = birch.subcluster_centers_   # [n_subclusters, d]
        # map each superpixel's birch_id -> its center index
        # (they are aligned: label k corresponds to centers[k])

        km2 = KMeans(
            n_clusters=n_birch_clusters,
            n_init="auto",
            random_state=0,
        )
        big_labels = km2.fit_predict(centers)  # label per subcluster

        # remap per-superpixel birch_ids -> compressed labels
        birch_ids = big_labels[birch_ids]

    # Final mapping: SLIC superpixel -> BIRCH region label
    sp_to_birch = {
        int(sp): int(cid)
        for sp, cid in zip(superpixel_ids, birch_ids)
    }
    merged_labels = np.vectorize(sp_to_birch.get, otypes=[np.int32])(labels_slic)

    # Final mapping: SLIC superpixel -> BIRCH region label
    sp_to_birch = {
        int(sp): int(cid)
        for sp, cid in zip(superpixel_ids, birch_ids)
    }

    merged_labels = np.vectorize(sp_to_birch.get, otypes=[np.int32])(labels_slic)
    return merged_labels


# ==========================
# METRIC UTILITIES
# ==========================

def _contingency(seg_labels: np.ndarray, gt_labels: np.ndarray) -> np.ndarray:
    """Build contingency table between segmentation and GT partitions."""
    seg_flat = seg_labels.ravel()
    gt_flat = gt_labels.ravel()

    seg_ids, seg_inv = np.unique(seg_flat, return_inverse=True)
    gt_ids, gt_inv = np.unique(gt_flat, return_inverse=True)

    contingency = np.zeros((len(seg_ids), len(gt_ids)), dtype=np.int64)
    np.add.at(contingency, (seg_inv, gt_inv), 1)
    return contingency


def compute_asa(seg_labels: np.ndarray, gt_labels: np.ndarray) -> float:
    """Achievable Segmentation Accuracy (ASA)."""
    contingency = _contingency(seg_labels, gt_labels)
    n_pixels = seg_labels.size
    max_overlaps = contingency.max(axis=1)  # largest overlap per superpixel
    asa = max_overlaps.sum() / float(n_pixels)
    return float(asa)


def compute_use(seg_labels: np.ndarray, gt_labels: np.ndarray) -> float:
    """
    Undersegmentation Error (USE) in Achanta-style definition:

        USE = (1/N) * sum_k [ sum_{i: S_i ∩ G_k ≠ ∅} |S_i| - |G_k| ]

    where {G_k} are GT regions, {S_i} are superpixels, N is total #pixels.
    """
    contingency = _contingency(seg_labels, gt_labels)
    n_pixels = seg_labels.size

    row_sums = contingency.sum(axis=1)  # |S_i|
    col_sums = contingency.sum(axis=0)  # |G_k|

    use_sum = 0.0
    for j in range(contingency.shape[1]):
        overlaps = contingency[:, j] > 0
        sum_s_areas = row_sums[overlaps].sum()
        g_area = col_sums[j]
        use_sum += (sum_s_areas - g_area)

    use = use_sum / float(n_pixels)
    return float(use)


def compute_sse(seg_labels: np.ndarray, image: np.ndarray) -> float:
    """
    Sum of Squared Error in Lab space per pixel.

    SSE = (1/N) * sum_p || Lab(p) - mean_Lab(region(p)) ||^2
    """
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    img_f = img_as_float(image)
    lab = rgb2lab(img_f)

    n_pixels = lab.shape[0] * lab.shape[1]
    sse = 0.0

    for region in np.unique(seg_labels):
        mask = (seg_labels == region)
        region_pixels = lab[mask]  # [M, 3]
        mean_lab = region_pixels.mean(axis=0)
        diffs = region_pixels - mean_lab
        sse += np.sum(diffs ** 2)

    sse /= float(n_pixels)
    return float(sse)


def compute_compactness(seg_labels: np.ndarray) -> float:
    """
    Compactness measure C, defined here as the average of:

        C_i = perimeter_i^2 / (4π * area_i)

    over all regions i. A perfect circle has C_i = 1; more irregular shapes > 1.

    NOTE: The exact formula used in the paper is not explicitly given; this is a
    standard choice for shape compactness and may not match their exact numbers.
    """
    props = regionprops(seg_labels + 1)  # +1 to ensure labels >= 1
    ratios = []
    for p in props:
        area = float(p.area)
        if area <= 0:
            continue
        perimeter = float(p.perimeter)
        ratio = (perimeter ** 2) / (4.0 * np.pi * area)
        ratios.append(ratio)
    if not ratios:
        return float("nan")
    return float(np.mean(ratios))


def compute_rand_index(seg_labels: np.ndarray, gt_labels: np.ndarray) -> float:
    """
    Rand Index (RI) between segmentation and ground truth partitions.

    Implemented using the contingency table to avoid O(N^2) loops.
    """
    contingency = _contingency(seg_labels, gt_labels)
    n = seg_labels.size
    if n <= 1:
        return 1.0

    # Pairs in same cluster in both partitions
    a = (contingency * (contingency - 1) // 2).sum()

    # Pairs in same cluster in segmentation
    row_sums = contingency.sum(axis=1)
    t1 = (row_sums * (row_sums - 1) // 2).sum()

    # Pairs in same cluster in GT
    col_sums = contingency.sum(axis=0)
    t2 = (col_sums * (col_sums - 1) // 2).sum()

    total_pairs = n * (n - 1) // 2
    # Pairs in different clusters in both partitions
    d = total_pairs - t1 - t2 + a

    ri = (a + d) / float(total_pairs)
    return float(ri)


def compute_all_metrics(
    seg_labels: np.ndarray,
    gt_labels: np.ndarray,
    image: np.ndarray,
) -> dict:
    """Compute ASA, USE, SSE, C, RI for one image."""
    metrics = {}
    metrics["ASA"] = compute_asa(seg_labels, gt_labels)
    metrics["USE"] = compute_use(seg_labels, gt_labels)
    metrics["SSE"] = compute_sse(seg_labels, image)
    metrics["C"] = compute_compactness(seg_labels)
    metrics["RI"] = compute_rand_index(seg_labels, gt_labels)
    return metrics


# ==========================
# I/O UTILITIES
# ==========================

def iter_image_files(root: Path) -> Iterable[Path]:
    """Yield all image files (png/jpg/...) under root."""
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if any(fname.lower().endswith(ext) for ext in IMAGE_EXTS):
                yield Path(dirpath) / fname


def process_split(split: str) -> None:
    """
    Process one split (train, validation, or test) under BreakHis_400X_full_augmented.

    - Superpixels are computed for ALL images (original + augmented).
    - If COMPUTE_METRICS is True and GT exists, metrics are computed and
      aggregated ONLY for ORIGINAL images (filenames without '_augment').
    """
    split_root = DATA_ROOT / split
    if not split_root.exists():
        print(f"[WARN] Split folder '{split_root}' not found, skipping.")
        return

    img_paths = list(iter_image_files(split_root))
    print(f"[{split}] Found {len(img_paths)} images under {split_root}")

    asa_list, use_list, sse_list, c_list, ri_list = [], [], [], [], []

    for idx, img_path in enumerate(img_paths, start=1):
        rel = img_path.relative_to(DATA_ROOT)
        out_dir = SUPERPIXEL_ROOT / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        out_name = img_path.stem + "_labels.npz"
        out_path = out_dir / out_name

        img = imread(img_path)
        if img.ndim == 2:  # grayscale -> fake RGB
            img = np.stack([img, img, img], axis=-1)

        if out_path.exists():
            print(f"[{split} {idx}/{len(img_paths)}] Skipping existing {out_path}")
            seg_labels = np.load(out_path)["labels"]
        else:
            # (Optional) resize here to 224x224 if you want strict matching to the paper
            # e.g. using skimage.transform.resize(..., preserve_range=True)
            seg_labels = slic_kmeans_birch(img)
            np.savez_compressed(out_path, labels=seg_labels)
            print(f"[{split} {idx}/{len(img_paths)}] Saved labels to {out_path}")

        # Decide whether this is an augmented image
        is_augmented = "_augment" in img_path.stem

        # Optional metrics: usually we only care about original images
        if COMPUTE_METRICS and (not is_augmented):
            gt_rel = rel.parent / (img_path.stem + "_gt_labels.npz")
            gt_path = GT_ROOT / gt_rel
            if gt_path.exists():
                gt_data = np.load(gt_path)
                gt_labels = gt_data["labels"]

                if gt_labels.shape != seg_labels.shape:
                    print(
                        f"[{split} {idx}/{len(img_paths)}] "
                        f"GT shape {gt_labels.shape} != seg shape {seg_labels.shape}, skipping metrics."
                    )
                else:
                    m = compute_all_metrics(seg_labels, gt_labels, img)
                    asa_list.append(m["ASA"])
                    use_list.append(m["USE"])
                    sse_list.append(m["SSE"])
                    c_list.append(m["C"])
                    ri_list.append(m["RI"])

                    print(
                        f"[{split} {idx}/{len(img_paths)}] "
                        f"ASA={m['ASA']:.4f}, USE={m['USE']:.4f}, "
                        f"SSE={m['SSE']:.4f}, C={m['C']:.4f}, RI={m['RI']:.4f}"
                    )

    # Aggregate metrics for this split (if any were computed)
    if COMPUTE_METRICS and asa_list:
        print(f"\n[{split}] Aggregated metrics over {len(asa_list)} ORIGINAL images with GT:")
        print(f"  ASA (mean ± std): {np.mean(asa_list):.4f} ± {np.std(asa_list):.4f}")
        print(f"  USE (mean ± std): {np.mean(use_list):.4f} ± {np.std(use_list):.4f}")
        print(f"  SSE (mean ± std): {np.mean(sse_list):.4f} ± {np.std(sse_list):.4f}")
        print(f"  C   (mean ± std): {np.mean(c_list):.4f} ± {np.std(c_list):.4f}")
        print(f"  RI  (mean ± std): {np.mean(ri_list):.4f} ± {np.std(ri_list):.4f}")
        print("")


# ==========================
# MAIN
# ==========================

if __name__ == "__main__":
    print(f"DATA_ROOT       = {DATA_ROOT}")
    print(f"SUPERPIXEL_ROOT = {SUPERPIXEL_ROOT}")
    print(f"GT_ROOT         = {GT_ROOT}")
    print(f"COMPUTE_METRICS = {COMPUTE_METRICS}")

    for split in SPLITS:
        process_split(split)

    print("Superpixel computation finished.")
