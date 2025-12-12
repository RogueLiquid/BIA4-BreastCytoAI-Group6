import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
import glob
import numpy as np
from tqdm import tqdm

def show_image(img, ax, title=""):
    """Helper to show a PIL image on a given axis."""
    ax.imshow(img)
    ax.set_title(title, fontsize=10)
    ax.axis("off")

def demo_augmentations(image_path):
    # Load image as RGB
    img = Image.open(image_path).convert("RGB")

    # No randomness now – everything is deterministic
    # (you can drop manual_seed entirely)
    # torch.manual_seed(0)

    # --- Geometric transforms (deterministic) ---
    # Always flip horizontally
    img_flip = F.hflip(img)

    # Rotate by a fixed angle (e.g. +90°) instead of random in [-10, 10]
    img_rot = F.rotate(img, angle=90)  # change to 10 if you want ±10° example

    # --- Color transforms with MAX changes ---

    # 1) Brightness
    # ColorJitter(brightness=0.8) means factor ∈ [0.2, 1.8].
    # Here we explicitly use the max factor 1.8 (much brighter).
    brightness_amount = -0.4
    img_bright = F.adjust_brightness(img, 1.0 + brightness_amount)  # factor=1.8

    # 2) Contrast
    # contrast=0.4 -> factor ∈ [0.6, 1.4]; use 1.4 as “max contrast”.
    contrast_amount = -0.6
    img_contrast = F.adjust_contrast(img, 1.0 + contrast_amount)    # factor=1.4

    # 3) Saturation
    # saturation=0.8 -> factor ∈ [0.2, 1.8]; use 1.8 as “max saturation”.
    saturation_amount = -0.8
    img_sat = F.adjust_saturation(img, 1.0 + saturation_amount)     # factor=1.8

    # 4) Hue
    # hue=0.15 -> random hue_factor ∈ [-0.15, 0.15]; here use +0.15 (max shift).
    # hue_factor must be in [-0.5, 0.5].
    hue_amount = -0.15
    img_hue = F.adjust_hue(img, hue_factor=hue_amount)              # +0.15

    # 5) A combined “max-ish” jitter configuration
    img_full = img
    img_full = F.adjust_brightness(img_full, 1.0 + 0.2)
    img_full = F.adjust_contrast(img_full,   1.0 + 0.2)
    img_full = F.adjust_saturation(img_full, 1.0 + 0.2)
    img_full = F.adjust_hue(img_full,        hue_factor=0.05)

    # Plot grid
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.ravel()

    show_image(img,        axes[0], "Original")
    show_image(img_flip,   axes[1], "Horizontal Flip")
    show_image(img_rot,    axes[2], "Rotate(90°)")
    show_image(img_bright, axes[3], "Brightness factor = 1.8")

    show_image(img_contrast, axes[4], "Contrast factor = 1.4")
    show_image(img_sat,      axes[5], "Saturation factor = 1.8")
    show_image(img_hue,      axes[6], "Hue shift = +0.15")
    show_image(img_full,     axes[7], "Combined jitter (max-ish)")

    plt.tight_layout()
    plt.show()

def estimate_color_distributions(
    root_dir,
    pattern="*.png",
    max_images=None,
    percentiles=(5, 95),
):
    """
    Scan images under root_dir and estimate distributions of:
    - brightness (mean luminance)
    - contrast (std of luminance)
    - saturation (mean saturation in HSV)

    Returns:
        stats: dict with raw distributions and suggested ColorJitter params.
    """
    paths = glob.glob(os.path.join(root_dir, "**", pattern), recursive=True)
    if max_images is not None:
        paths = paths[:max_images]

    print(f"Found {len(paths)} images (using up to {max_images} for stats).")

    brightness_vals = []
    contrast_vals = []
    saturation_vals = []

    for p in tqdm(paths, desc="Scanning images"):
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # float in [0,1]
        img_f = img.astype(np.float32) / 255.0

        # ----- luminance (Y) proxy -----
        # standard ITU-R BT.601 luma transform
        lum = 0.299 * img_f[..., 0] + 0.587 * img_f[..., 1] + 0.114 * img_f[..., 2]

        brightness_vals.append(lum.mean())
        contrast_vals.append(lum.std())

        # ----- saturation from HSV -----
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # OpenCV HSV: H in [0,179], S,V in [0,255]
        sat = hsv[..., 1].astype(np.float32) / 255.0
        saturation_vals.append(sat.mean())

    brightness_vals = np.array(brightness_vals)
    contrast_vals = np.array(contrast_vals)
    saturation_vals = np.array(saturation_vals)

    print("\n=== Raw distributions ===")
    def summarize(name, arr):
        p_low, p_high = np.percentile(arr, percentiles)
        print(
            f"{name:>10} | mean={arr.mean():.4f}, std={arr.std():.4f}, "
            f"p{percentiles[0]}={p_low:.4f}, p{percentiles[1]}={p_high:.4f}"
        )

    summarize("brightness", brightness_vals)
    summarize("contrast", contrast_vals)
    summarize("saturation", saturation_vals)

    # ---- Suggest ColorJitter ranges ----
    # Heuristic:
    #  - Take median value m
    #  - Convert each value to factor = val / m
    #  - Look at 5th and 95th percentile of factors -> [lo, hi]
    #  - We want factor in [1 - x, 1 + x] to cover that range
    #    so x ≈ max(1 - lo, hi - 1)

    def suggest_jitter_param(vals, name):
        m = np.median(vals)
        if m <= 0:
            return 0.0, (1.0, 1.0)
        factors = vals / m
        lo, hi = np.percentile(factors, percentiles)
        x = max(1.0 - lo, hi - 1.0)
        x = float(max(0.0, x))  # ensure non-negative

        print(
            f"\nSuggested ColorJitter('{name}') parameter ≈ {x:.3f} "
            f"  (covers ~p{percentiles[0]}–p{percentiles[1]} range)"
        )
        print(f"  -> approximate factor range: [{1.0 - x:.3f}, {1.0 + x:.3f}]")

        return x, (1.0 - x, 1.0 + x)

    b_param, b_range = suggest_jitter_param(brightness_vals, "brightness")
    c_param, c_range = suggest_jitter_param(contrast_vals, "contrast")
    s_param, s_range = suggest_jitter_param(saturation_vals, "saturation")

    stats = {
        "brightness_vals": brightness_vals,
        "contrast_vals": contrast_vals,
        "saturation_vals": saturation_vals,
        "suggested": {
            "brightness": {"param": b_param, "factor_range": b_range},
            "contrast": {"param": c_param, "factor_range": c_range},
            "saturation": {"param": s_param, "factor_range": s_range},
        },
    }

    print("\n=== Example ColorJitter config (heuristic) ===")
    print(
        "ColorJitter(" 
        f"brightness={b_param:.3f}, "
        f"contrast={c_param:.3f}, "
        f"saturation={s_param:.3f}, "
        "hue=0.05  # hue chosen manually (small)"
        ")"
    )

    return stats

image_path = "./BreaKHis_400X_patient_split/train/malignant/SOB_M_DC-14-13993-400-033.png"
demo_augmentations(image_path)
