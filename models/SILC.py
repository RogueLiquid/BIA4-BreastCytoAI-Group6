import numpy as np
from skimage.segmentation import slic
from skimage.util import img_as_float
from sklearn.cluster import Birch

def slic_kmeans_birch(
    image,
    n_segments: int = 400,
    compactness: float = 10.0,
    n_birch_clusters: int | None = None,
):
    """
    image: RGB image as uint8 [H,W,3] or float.
    Returns: superpixel label map [H,W] (int64).

    Steps:
      1) SLIC superpixels (SLIC is a k-means-like algorithm).
      2) Optionally, merge/refine superpixels using BIRCH on region features.
    """
    # Ensure float in [0,1] for SLIC
    img_f = img_as_float(image)

    # Step 1: SLIC superpixels
    labels_slic = slic(
        img_f,
        n_segments=n_segments,
        compactness=compactness,
        start_label=0,
    )  # [H,W], labels 0..K-1

    if n_birch_clusters is None:
        return labels_slic

    # Step 2: Build a simple feature vector per superpixel
    superpixel_ids = np.unique(labels_slic)
    feats = []
    for sid in superpixel_ids:
        mask = labels_slic == sid
        coords = np.column_stack(np.nonzero(mask))  # (row, col)
        mean_rc = coords.mean(axis=0)              # [2]
        mean_color = img_f[mask].mean(axis=0)      # [3]
        feats.append(np.concatenate([mean_color, mean_rc]))
    feats = np.vstack(feats)  # [num_superpixels, 5]

    # Step 3: BIRCH clustering of superpixel features
    birch = Birch(n_clusters=n_birch_clusters)
    cluster_ids = birch.fit_predict(feats)  # [num_superpixels]

    # Map old superpixel IDs -> merged cluster IDs
    sp_to_cluster = {int(sp): int(c) for sp, c in zip(superpixel_ids, cluster_ids)}

    merged_labels = np.vectorize(sp_to_cluster.get)(labels_slic)
    return merged_labels

from skimage.io import imread

img = imread("some_breakhis_patch.png")  # [H,W,3], uint8
labels = slic_kmeans_birch(
    img,
    n_segments=400,
    compactness=10.0,
    n_birch_clusters=200,  # or None if you just want plain SLIC
)

import cv2

def extract_sift_descriptors_on_superpixels(
    image,
    superpixel_labels: np.ndarray,
):
    """
    image: RGB image [H,W,3] uint8
    superpixel_labels: [H,W] int, from slic_kmeans_birch

    Returns:
        descriptors: np.ndarray [N_descriptors, 128] (or None if no keypoints)
    """
    # Convert to grayscale for SIFT
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create()
    all_desc = []

    for sid in np.unique(superpixel_labels):
        # Build mask for this superpixel
        mask = (superpixel_labels == sid).astype("uint8") * 255  # [H,W]

        kp, des = sift.detectAndCompute(gray, mask)
        if des is not None and len(des) > 0:
            all_desc.append(des)

    if not all_desc:
        return None

    return np.vstack(all_desc)  # [N,128]

from sklearn.cluster import MiniBatchKMeans

def build_codebook(
    all_descriptor_arrays: list[np.ndarray],
    n_words: int = 256,
    max_descriptors: int = 100_000,
    batch_size: int = 1000,
) -> MiniBatchKMeans:
    """
    all_descriptor_arrays: list of [N_i, 128] arrays from many training images.
    Returns: fitted MiniBatchKMeans model (the visual vocabulary).
    """
    # Concatenate all descriptors
    all_desc = np.vstack(
        [d for d in all_descriptor_arrays if d is not None and len(d) > 0]
    )
    print(f"[BoF] Total descriptors before subsampling: {all_desc.shape[0]}")

    # Optionally subsample for speed
    if all_desc.shape[0] > max_descriptors:
        idx = np.random.choice(all_desc.shape[0], max_descriptors, replace=False)
        all_desc = all_desc[idx]

    print(f"[BoF] Using {all_desc.shape[0]} descriptors to train codebook.")

    kmeans = MiniBatchKMeans(
        n_clusters=n_words,
        batch_size=batch_size,
        verbose=1,
    )
    kmeans.fit(all_desc)
    return kmeans

from glob import glob
from skimage.io import imread

image_paths = glob("path/to/train_patches/*.png")

descriptor_list = []
for p in image_paths:
    img = imread(p)
    labels = slic_kmeans_birch(img, n_segments=400, compactness=10.0, n_birch_clusters=200)
    des = extract_sift_descriptors_on_superpixels(img, labels)
    if des is not None:
        descriptor_list.append(des)

codebook = build_codebook(descriptor_list, n_words=256)

def bof_histogram_for_image(
    image,
    codebook: MiniBatchKMeans,
    n_segments: int = 400,
    compactness: float = 10.0,
    n_birch_clusters: int | None = 200,
) -> np.ndarray:
    """
    image: RGB [H,W,3] uint8
    codebook: fitted MiniBatchKMeans

    Returns:
        hist: [n_words] L1-normalized BoF histogram.
    """
    labels = slic_kmeans_birch(
        image,
        n_segments=n_segments,
        compactness=compactness,
        n_birch_clusters=n_birch_clusters,
    )

    desc = extract_sift_descriptors_on_superpixels(image, labels)
    if desc is None or len(desc) == 0:
        # no descriptors: return all zeros
        return np.zeros(codebook.n_clusters, dtype=np.float32)

    # Assign descriptors to nearest visual words
    word_ids = codebook.predict(desc)  # [N_descriptors]

    # Build histogram
    hist, _ = np.histogram(
        word_ids,
        bins=np.arange(codebook.n_clusters + 1),
        density=False,
    )
    hist = hist.astype(np.float32)

    # L1-normalize
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist

img = imread("some_breakhis_patch.png")
hist = bof_histogram_for_image(img, codebook, n_segments=400, compactness=10.0, n_birch_clusters=200)
