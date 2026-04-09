#!/usr/bin/env python3
"""
Vesuvius Scroll Volume Visualizer
=================================
Standalone script extracted from the Vesuvius EDA notebook.
Scans a directory of .tif scroll volumes, runs preprocessing and
visualization, and saves per-volume output images + a 3D Plotly HTML
to the specified output directory.

Usage:
    python vesuvius_visualize.py --input_dir /path/to/tif_volumes \
                                 --output_dir /path/to/output \
                                 [--labels_dir /path/to/label_tifs] \
                                 [--denoise] \
                                 [--sub_volume_size 80]

Arguments:
    --input_dir         Directory containing .tif volume files
    --output_dir        Directory where output images/HTML will be saved
    --labels_dir        (Optional) Directory containing matching label .tif files
    --denoise           (Optional) Enable NL-means denoising (slow)
    --sub_volume_size   (Optional) Cube side length for 3D render (default: 80)

Outputs (per volume):
    <vol_id>_volume_eda.png          – slices, histograms, projections, stats
    <vol_id>_texture_analysis.png    – Sobel, Laplacian, Canny, Otsu, etc.
    <vol_id>_texture_profile.png     – intensity profile along center row
    <vol_id>_preprocessing.png       – raw vs denoised vs normalized comparison
    <vol_id>_orthogonal_slices.png   – axial / sagittal / coronal mid-slices
    <vol_id>_3d_volume.html          – interactive Plotly 3D sub-volume render
    <vol_id>_mask_analysis.png       – (only if labels_dir is provided)

Global outputs:
    volume_comparison.png            – side-by-side comparison (up to 3 volumes)
    summary_statistics.csv           – per-volume stats table
"""

import os
import sys
import json
import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle
import seaborn as sns

sns.set_context("poster")

import tifffile
import cv2

from skimage.restoration import denoise_nl_means, estimate_sigma

import plotly.graph_objects as go

try:
    import kmeans1d

    HAS_KMEANS1D = True
except ImportError:
    HAS_KMEANS1D = False
    print("⚠️  kmeans1d not installed — contrast normalization will use percentile fallback.")

# ─── Constants ────────────────────────────────────────────────────────────────
BACKGROUND_LEVEL = 0.25
FOREGROUND_LEVEL = 0.50
CLASS_NAMES = {0: "Background", 1: "Foreground", 2: "Unlabeled"}


# =============================================================================
#  DATA LOADING & PREPROCESSING
# =============================================================================


def load_tiff_volume(path):
    """Load a multi-frame or single-frame TIFF as a (Z, H, W) numpy array."""
    if not os.path.exists(path):
        return None
    try:
        data = tifffile.imread(path)
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        return data
    except Exception as e:
        print(f"  ⚠️  Failed to load {path}: {e}")
        return None


def remove_zeropadding(data, labels=None):
    """Crop away fully-zero border slices along each axis."""
    offsets = [0, 0, 0]

    def crop_axis(arr, axis):
        other = tuple(i for i in range(arr.ndim) if i != axis)
        sums = np.sum(arr, axis=other)
        idxs = np.where(sums > 0)[0]
        if len(idxs) == 0:
            return slice(0, arr.shape[axis]), 0
        return slice(idxs[0], idxs[-1] + 1), int(idxs[0])

    zslice, offsets[0] = crop_axis(data, 0)
    yslice, offsets[1] = crop_axis(data, 1)
    xslice, offsets[2] = crop_axis(data, 2)

    cropped = data[zslice, yslice, xslice]
    if labels is not None:
        labels = labels[zslice, yslice, xslice]
    return cropped, labels, tuple(offsets)


def to_float_rescaled(data):
    """Convert any integer or float volume to float32 in [0, 1].

    Unlike skimage's img_as_float which divides by the dtype's theoretical
    max (e.g. 65535 for uint16), this rescales by the *actual* data range.
    This is critical for Vesuvius CT data where intensities rarely span the
    full uint16 range — img_as_float would squash everything near zero.
    """
    data_f = data.astype(np.float32)
    dmin = data_f.min()
    dmax = data_f.max()
    if dmax - dmin < 1e-8:
        # Constant volume — return zeros
        return np.zeros_like(data_f)
    return (data_f - dmin) / (dmax - dmin)


def denoise_volume(data_f):
    """Apply NL-means denoising to a float32 [0,1] volume.

    Expects already-rescaled float input (from to_float_rescaled).
    """
    sigma = estimate_sigma(data_f)
    if sigma is None or (np.isscalar(sigma) and sigma < 1e-8):
        print("    ℹ️  Estimated sigma ≈ 0 — skipping denoise (data may be very clean).")
        return data_f.copy()
    denoised = denoise_nl_means(
        data_f,
        patch_size=7,
        patch_distance=1,
        sigma=sigma,
        fast_mode=True,           # ~5× faster, still good quality
    )
    return denoised.astype(np.float32, copy=False)


def renorm_image_contrast(data):
    """Align background/foreground levels using k-means (or percentile fallback)."""
    flat = data.ravel()
    total = flat.size

    if total == 0:
        return np.clip(data, 0, 1).astype(np.float32, copy=False)

    sample_size = min(max(total // 1000, 50000), 200000, total)
    if sample_size < total:
        # Sampling with replacement avoids large temporary allocations.
        sampled_indices = np.random.randint(0, total, size=sample_size)
        sample = flat[sampled_indices]
    else:
        sample = flat

    if HAS_KMEANS1D:
        _, (bg, fg) = kmeans1d.cluster(sample, 2)
    else:
        bg = float(np.percentile(sample, 30))
        fg = float(np.percentile(sample, 70))

    if abs(fg - bg) < 1e-8:
        return np.clip(data, 0, 1).astype(np.float32, copy=False)

    scaled = (data - bg) * (FOREGROUND_LEVEL - BACKGROUND_LEVEL) / (fg - bg) + BACKGROUND_LEVEL
    return np.clip(scaled, 0, 1).astype(np.float32, copy=False)


def load_and_preprocess(vol_id, img_path, label_path=None, run_denoise=True):
    """Load volume + optional mask, strip padding, denoise, contrast-normalize."""
    raw_vol = load_tiff_volume(img_path)
    raw_mask = load_tiff_volume(label_path) if label_path else None

    if raw_vol is None:
        print(f"  ⚠️  Volume not found: {vol_id}")
        return None

    vol, mask, offsets = remove_zeropadding(raw_vol, raw_mask)
    print(f"  [{vol_id}] raw={raw_vol.shape} → cropped={vol.shape}, dtype={vol.dtype}, "
          f"range=[{vol.min()}, {vol.max()}], offsets={offsets}")

    # Rescale to [0, 1] using the ACTUAL data range (not dtype max)
    vol_f = to_float_rescaled(vol)

    if run_denoise:
        print(f"  [{vol_id}] Denoising (NL-means) …")
        denoised = denoise_volume(vol_f)  # pass float data, not raw int
    else:
        denoised = vol_f.copy()

    normalized = renorm_image_contrast(denoised)

    return {
        "vol_id": vol_id,
        "raw": vol_f,
        "denoised": denoised,
        "normalized": normalized,
        "mask": mask,
        "offsets": offsets,
    }


# =============================================================================
#  VISUALIZATION FUNCTIONS
# =============================================================================


def analyze_volume(vol_result, output_dir):
    """Comprehensive per-volume EDA: slices, histogram, projections, stats."""
    vol_id = vol_result["vol_id"]
    volume = vol_result["normalized"]
    depth, height, width = volume.shape

    fig = plt.figure(figsize=(22, 13))
    gs = plt.GridSpec(3, 5, figure=fig, hspace=0.35, wspace=0.3)

    # Row 0 – 5 representative slices
    slice_indices = [0, depth // 4, depth // 2, 3 * depth // 4, depth - 1]
    for i, idx in enumerate(slice_indices):
        ax = fig.add_subplot(gs[0, i])
        sl = volume[idx]
        im = ax.imshow(sl, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Z={idx}\n[{sl.min():.2f}, {sl.max():.2f}]", fontsize=8)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 1 – intensity histogram + stats box
    ax_hist = fig.add_subplot(gs[1, :3])
    ax_hist.hist(vol_result["raw"].flatten(), bins=128, alpha=0.5,
                 color="steelblue", label="Raw", histtype="step", linewidth=1.5)
    ax_hist.hist(vol_result["denoised"].flatten(), bins=128, alpha=0.5,
                 color="darkorange", label="Denoised", histtype="step", linewidth=1.5)
    ax_hist.hist(vol_result["normalized"].flatten(), bins=128, alpha=0.5,
                 color="green", label="Normalized", histtype="step", linewidth=1.5)
    ax_hist.axvline(BACKGROUND_LEVEL, color="k", linestyle="--", label=f"BG target ({BACKGROUND_LEVEL})")
    ax_hist.axvline(FOREGROUND_LEVEL, color="r", linestyle="--", label=f"FG target ({FOREGROUND_LEVEL})")
    ax_hist.set_title("Intensity Distribution: Raw vs Denoised vs Normalized", fontsize=11)
    ax_hist.set_xlabel("Intensity")
    ax_hist.set_ylabel("Voxel Count")
    ax_hist.legend(fontsize=9)
    ax_hist.grid(alpha=0.3)

    ax_stat = fig.add_subplot(gs[1, 3:])
    ax_stat.axis("off")
    stats_txt = (
        f"Volume: {vol_id}\n"
        f"Shape  : {volume.shape}\n"
        f"DType  : {volume.dtype}\n"
        f"Min    : {volume.min():.4f}\n"
        f"Max    : {volume.max():.4f}\n"
        f"Mean   : {volume.mean():.4f}\n"
        f"Std    : {volume.std():.4f}\n"
        f"Size   : {volume.nbytes / 1024**3:.3f} GB"
    )
    ax_stat.text(0.05, 0.5, stats_txt, fontsize=11, family="monospace",
                 verticalalignment="center",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))

    # Row 2 – max/mean projections
    proj_specs = [
        ("XY max-proj", np.max(volume, axis=0)),
        ("XZ max-proj", np.max(volume, axis=1)),
        ("YZ max-proj", np.max(volume, axis=2)),
        ("XY mean-proj", np.mean(volume, axis=0)),
        ("YZ std-proj", np.std(volume, axis=2)),
    ]
    cmaps = ["viridis", "viridis", "viridis", "plasma", "inferno"]
    for i, ((title, proj), cmap) in enumerate(zip(proj_specs, cmaps)):
        ax = fig.add_subplot(gs[2, i])
        im = ax.imshow(proj, cmap=cmap, aspect="auto")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f"Volume EDA: {vol_id}", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{vol_id}_volume_eda.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✅ Saved {out_path}")


def analyze_mask(vol_result, output_dir):
    """Class distribution, mid-slice overlay, and foreground projections."""
    vol_id = vol_result["vol_id"]
    volume = vol_result["normalized"]
    mask = vol_result["mask"]

    if mask is None:
        print(f"  [{vol_id}] No mask available — skipping mask analysis.")
        return

    unique_vals = np.unique(mask)
    total = mask.size
    mid = mask.shape[0] // 2

    mask_cmap = ListedColormap(["#B0BEC5", "#EF5350", "#FFA726"])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    mask_norm = BoundaryNorm(bounds, mask_cmap.N)
    legend_patches = [
        mpatches.Patch(color="#B0BEC5", label="0: Background"),
        mpatches.Patch(color="#EF5350", label="1: Foreground"),
        mpatches.Patch(color="#FFA726", label="2: Unlabeled"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))

    axes[0, 0].imshow(volume[mid], cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Volume mid-slice", fontsize=11)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(volume[mid], cmap="gray", vmin=0, vmax=1)
    axes[0, 1].imshow(mask[mid], cmap=mask_cmap, norm=mask_norm, alpha=0.45, interpolation="nearest")
    axes[0, 1].legend(handles=legend_patches, loc="lower right", fontsize=8)
    axes[0, 1].set_title("Mask overlay (mid-slice)", fontsize=11)
    axes[0, 1].axis("off")

    colors_bar = ["#B0BEC5", "#EF5350", "#FFA726"]
    counts = [int(np.sum(mask == cls)) for cls in [0, 1, 2] if cls in unique_vals]
    names = [CLASS_NAMES[cls] for cls in [0, 1, 2] if cls in unique_vals]
    c_colors = [colors_bar[cls] for cls in [0, 1, 2] if cls in unique_vals]

    axes[0, 2].bar(names, counts, color=c_colors, edgecolor="white")
    axes[0, 2].set_title("Absolute Voxel Counts", fontsize=11)
    axes[0, 2].set_ylabel("Count")
    axes[0, 2].grid(axis="y", alpha=0.3)
    axes[0, 2].tick_params(axis="x", rotation=15)

    pcts = [c / total * 100 for c in counts]
    axes[0, 3].bar(names, pcts, color=c_colors, edgecolor="white")
    axes[0, 3].set_title("Class Distribution (%)", fontsize=11)
    axes[0, 3].set_ylabel("%")
    axes[0, 3].grid(axis="y", alpha=0.3)
    axes[0, 3].tick_params(axis="x", rotation=15)
    for i, (n, p) in enumerate(zip(names, pcts)):
        axes[0, 3].text(i, p + 0.3, f"{p:.1f}%", ha="center", fontsize=9)

    fg = (mask == 1).astype(np.uint8)
    proj_specs = [
        ("XY foreground proj", np.max(fg, axis=0)),
        ("XZ foreground proj", np.max(fg, axis=1)),
        ("YZ foreground proj", np.max(fg, axis=2)),
    ]
    for i, (title, proj) in enumerate(proj_specs):
        axes[1, i].imshow(proj, cmap="hot", aspect="auto")
        axes[1, i].set_title(title, fontsize=10)
        axes[1, i].axis("off")

    fg_vox = volume[fg == 1]
    if len(fg_vox) > 0:
        axes[1, 3].hist(fg_vox.flatten(), bins=64, color="#EF5350", alpha=0.8, edgecolor="darkred")
        axes[1, 3].set_title("Foreground Voxel Intensity", fontsize=10)
        axes[1, 3].set_xlabel("Intensity")
        axes[1, 3].set_ylabel("Count")
        axes[1, 3].grid(alpha=0.3)
    else:
        axes[1, 3].text(0.5, 0.5, "No foreground\nvoxels found",
                        ha="center", va="center", transform=axes[1, 3].transAxes)
        axes[1, 3].axis("off")

    plt.suptitle(f"Mask Analysis: {vol_id}", fontsize=15, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{vol_id}_mask_analysis.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✅ Saved {out_path}")


def analyze_texture(vol_result, output_dir):
    """Gradient-based and morphological texture features on the mid-slice."""
    vol_id = vol_result["vol_id"]
    volume = vol_result["normalized"]
    mid = volume.shape[0] // 2

    sl_norm = cv2.normalize(volume[mid], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    sobel_x = cv2.Sobel(sl_norm, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(sl_norm, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    laplacian = cv2.Laplacian(sl_norm, cv2.CV_64F)
    edges = cv2.Canny(sl_norm, 50, 150)
    _, otsu = cv2.threshold(sl_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    eq = cv2.equalizeHist(sl_norm)
    grad_diag = np.sqrt(cv2.Sobel(sl_norm, cv2.CV_64F, 1, 1, ksize=3) ** 2)

    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    panels = [
        ("Original (normalized)", volume[mid], "gray"),
        ("Sobel Gradient Mag", sobel_mag, "hot"),
        ("|Laplacian|", np.abs(laplacian), "coolwarm"),
        ("Canny Edges", edges, "gray"),
        ("Otsu Binary", otsu, "gray"),
        ("Diagonal Gradient", grad_diag, "plasma"),
        ("Histogram Equalized", eq, "gray"),
        ("Sobel X", np.abs(sobel_x), "RdBu"),
        ("Sobel Y", np.abs(sobel_y), "RdBu"),
    ]
    for ax, (title, data, cmap) in zip(axes.flat, panels):
        ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.suptitle(f"Texture Analysis – mid-slice: {vol_id}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{vol_id}_texture_analysis.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✅ Saved {out_path}")

    # Intensity profile
    fig, ax = plt.subplots(figsize=(14, 3))
    row = volume[mid, volume.shape[1] // 2, :]
    ax.plot(row, color="steelblue", linewidth=1.5)
    ax.set_title(f"Intensity Profile – center row of mid-slice: {vol_id}", fontsize=11)
    ax.set_xlabel("X position")
    ax.set_ylabel("Intensity")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{vol_id}_texture_profile.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✅ Saved {out_path}")


def plot_preprocessing_comparison(vol_result, output_dir):
    """Side-by-side comparison of raw / denoised / normalized."""
    vol_id = vol_result["vol_id"]
    mid = vol_result["raw"].shape[0] // 2

    stages = [
        ("Raw", vol_result["raw"]),
        ("Denoised", vol_result["denoised"]),
        ("Normalized", vol_result["normalized"]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for i, (name, vol) in enumerate(stages):
        axes[0, i].imshow(vol[mid], cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"{name}  (z={mid})", fontsize=12)
        axes[0, i].axis("off")

        axes[1, i].hist(vol.flatten(), bins=128,
                        color=["steelblue", "darkorange", "green"][i],
                        alpha=0.8, histtype="step", linewidth=2)
        axes[1, i].axvline(BACKGROUND_LEVEL, color="k", linestyle="--", label="BG target")
        axes[1, i].axvline(FOREGROUND_LEVEL, color="r", linestyle="--", label="FG target")
        axes[1, i].set_title(f"{name} – histogram", fontsize=11)
        axes[1, i].set_xlabel("Intensity")
        axes[1, i].set_ylabel("Count")
        axes[1, i].legend(fontsize=8)
        axes[1, i].grid(alpha=0.3)
        axes[1, i].set_xlim(0, 1)

    plt.suptitle(f"Preprocessing Pipeline Comparison: {vol_id}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{vol_id}_preprocessing.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✅ Saved {out_path}")


def plot_orthogonal_slices(vol_result, output_dir):
    """Static orthogonal mid-slice view (axial, sagittal, coronal)."""
    vol_id = vol_result["vol_id"]
    vol = vol_result["normalized"]
    mask = vol_result["mask"]
    v_z, v_y, v_x = vol.shape
    z, y, x = v_z // 2, v_y // 2, v_x // 2

    mask_cmap = ListedColormap(["#B0BEC5", "#EF5350", "#FFA726"])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    mask_norm_obj = BoundaryNorm(bounds, mask_cmap.N)
    legend_patches = [
        mpatches.Patch(color="#B0BEC5", label="0: Background"),
        mpatches.Patch(color="#EF5350", label="1: Foreground"),
        mpatches.Patch(color="#FFA726", label="2: Unlabeled"),
    ]
    _rec = dict(linewidth=4, facecolor="none")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(vol[z], cmap="gray", vmin=0, vmax=1)
    if mask is not None:
        axes[0].imshow(mask[z], cmap=mask_cmap, norm=mask_norm_obj, alpha=0.4, interpolation="nearest")
    axes[0].add_patch(Rectangle((-0.5, -0.5), v_x, v_y, edgecolor="red", **_rec))
    axes[0].set_title(f"Axial (XY) – Z = {z}", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(vol[:, :, x].T, cmap="gray", vmin=0, vmax=1)
    if mask is not None:
        axes[1].imshow(mask[:, :, x].T, cmap=mask_cmap, norm=mask_norm_obj, alpha=0.4, interpolation="nearest")
    axes[1].add_patch(Rectangle((-0.5, -0.5), v_z, v_y, edgecolor="green", **_rec))
    axes[1].set_title(f"Sagittal (YZ) – X = {x}", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(vol[:, y, :], cmap="gray", vmin=0, vmax=1)
    if mask is not None:
        axes[2].imshow(mask[:, y, :], cmap=mask_cmap, norm=mask_norm_obj, alpha=0.4, interpolation="nearest")
    axes[2].add_patch(Rectangle((-0.5, -0.5), v_x, v_z, edgecolor="blue", **_rec))
    axes[2].set_title(f"Coronal (XZ) – Y = {y}", fontsize=12)
    axes[2].axis("off")

    if mask is not None:
        axes[2].legend(handles=legend_patches, loc="lower right", fontsize=9)

    plt.suptitle(f"Orthogonal Slice Viewer: {vol_id}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{vol_id}_orthogonal_slices.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✅ Saved {out_path}")


def generate_3d_volume_html(vol_result, output_dir, sub_size=48):
    """Interactive Plotly 3D volume rendering saved as standalone HTML."""
    vol_id = vol_result["vol_id"]
    volume = vol_result["normalized"]
    d, h, w = volume.shape

    # Extract a sub-volume from the center to keep rendering manageable
    sz, sy, sx = min(sub_size, d), min(sub_size, h), min(sub_size, w)
    z0 = max(0, d // 2 - sz // 2)
    y0 = max(0, h // 2 - sy // 2)
    x0 = max(0, w // 2 - sx // 2)
    sub = volume[z0 : z0 + sz, y0 : y0 + sy, x0 : x0 + sx]

    vol = np.asarray(sub, dtype=np.float32)

    # If still too large, stride down further to keep browser memory stable.
    max_voxels = 120_000
    total_voxels = int(np.prod(vol.shape))
    if total_voxels > max_voxels:
        stride = int(np.ceil((total_voxels / max_voxels) ** (1 / 3)))
        vol = vol[::stride, ::stride, ::stride]
    nx, ny, nz = vol.shape
    X, Y, Z = np.mgrid[0:nx, 0:ny, 0:nz]

    fig = go.Figure(
        data=go.Volume(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=vol.flatten(),
            colorscale="Greys",
            cmin=float(vol.min()), cmax=float(vol.max()),
            isomin=float(vol.min()), isomax=float(vol.max()),
            surface_count=12,
            opacity=0.25,
            opacityscale=[[0.0, 1.0], [0.3, 1.0], [1.0, 1.0]],
            caps=dict(x_show=False, y_show=False, z_show=False),
        )
    )
    fig.update_layout(
        title=f"3D Sub-volume: {vol_id} (shape {vol.shape})",
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode="data",
            camera=dict(eye=dict(x=2, y=2, z=1.5)),
        ),
        width=800, height=700,
    )

    out_path = os.path.join(output_dir, f"{vol_id}_3d_volume.html")
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"    ✅ Saved {out_path}")


def compare_volumes(vol_data_dict, output_dir, n=3):
    """Compare up to n volumes side by side."""
    items = list(vol_data_dict.values())[:n]
    n = len(items)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 5, figsize=(22, 4.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    mask_cmap = ListedColormap(["#B0BEC5", "#EF5350", "#FFA726"])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    mask_norm_obj = BoundaryNorm(bounds, mask_cmap.N)

    for i, d in enumerate(items):
        vol = d["normalized"]
        mask = d["mask"]
        vol_id = d["vol_id"]
        mid = vol.shape[0] // 2

        axes[i, 0].imshow(vol[mid], cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title(f"{vol_id}\nNormalized mid-slice", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].hist(d["raw"].flatten(), bins=64, histtype="step",
                        color="steelblue", label="Raw", linewidth=1.5)
        axes[i, 1].hist(d["normalized"].flatten(), bins=64, histtype="step",
                        color="green", label="Normalized", linewidth=1.5)
        axes[i, 1].axvline(BACKGROUND_LEVEL, color="k", linestyle="--")
        axes[i, 1].axvline(FOREGROUND_LEVEL, color="r", linestyle="--")
        axes[i, 1].set_title("Intensity Hist", fontsize=9)
        axes[i, 1].legend(fontsize=7)
        axes[i, 1].grid(alpha=0.3)
        axes[i, 1].set_xlim(0, 1)

        if mask is not None:
            axes[i, 2].imshow(vol[mid], cmap="gray", vmin=0, vmax=1)
            axes[i, 2].imshow(mask[mid], cmap=mask_cmap, norm=mask_norm_obj, alpha=0.45)
            axes[i, 2].set_title("Mask overlay", fontsize=9)
        else:
            axes[i, 2].text(0.5, 0.5, "No mask", ha="center", va="center",
                            transform=axes[i, 2].transAxes)
        axes[i, 2].axis("off")

        if mask is not None:
            uniq = np.unique(mask)
            bar_colors = ["#B0BEC5", "#EF5350", "#FFA726"]
            bar_vals = [int(np.sum(mask == c)) for c in uniq]
            bar_labels = [CLASS_NAMES.get(int(c), str(c)) for c in uniq]
            axes[i, 3].bar(bar_labels, bar_vals,
                           color=[bar_colors[int(c)] for c in uniq])
            axes[i, 3].set_title("Class Counts", fontsize=9)
            axes[i, 3].tick_params(axis="x", rotation=15, labelsize=8)
            axes[i, 3].grid(axis="y", alpha=0.3)
        else:
            axes[i, 3].axis("off")

        if mask is not None:
            fg_vox = vol[mask == 1]
            if len(fg_vox) > 0:
                axes[i, 4].hist(fg_vox, bins=64, color="#EF5350", alpha=0.8)
                axes[i, 4].set_title("FG Intensity", fontsize=9)
                axes[i, 4].grid(alpha=0.3)
            else:
                axes[i, 4].text(0.5, 0.5, "No FG voxels",
                                ha="center", va="center", transform=axes[i, 4].transAxes)
                axes[i, 4].axis("off")
        else:
            axes[i, 4].axis("off")

    plt.suptitle("Cross-Volume Comparison", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "volume_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved {out_path}")


def save_summary_csv(vol_data_dict, output_dir):
    """Save per-volume summary statistics to CSV."""
    import pandas as pd

    rows = []
    for d in vol_data_dict.values():
        vol = d["normalized"]
        mask = d["mask"]
        row = {
            "vol_id": d["vol_id"],
            "shape": str(vol.shape),
            "dtype": str(vol.dtype),
            "vol_min": round(float(vol.min()), 4),
            "vol_max": round(float(vol.max()), 4),
            "vol_mean": round(float(vol.mean()), 4),
            "vol_std": round(float(vol.std()), 4),
            "size_GB": round(vol.nbytes / 1024 ** 3, 4),
        }
        if mask is not None:
            total = mask.size
            row["bg_pct"] = round(float(np.sum(mask == 0)) / total * 100, 2)
            row["fg_pct"] = round(float(np.sum(mask == 1)) / total * 100, 2)
            row["unk_pct"] = round(float(np.sum(mask == 2)) / total * 100, 2)
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, "summary_statistics.csv")
    df.to_csv(out_path, index=False)
    print(f"  ✅ Saved {out_path}")


# =============================================================================
#  MANIFEST (for web integration)
# =============================================================================


def save_manifest(vol_data_dict, output_dir, sub_size):
    """Save a JSON manifest listing all generated outputs per volume.
    Useful for a web frontend to discover and display the results.
    """
    manifest = {"volumes": []}
    for d in vol_data_dict.values():
        vid = d["vol_id"]
        vol = d["normalized"]
        entry = {
            "vol_id": vid,
            "shape": list(vol.shape),
            "outputs": {
                "volume_eda": f"{vid}_volume_eda.png",
                "texture_analysis": f"{vid}_texture_analysis.png",
                "texture_profile": f"{vid}_texture_profile.png",
                "preprocessing": f"{vid}_preprocessing.png",
                "orthogonal_slices": f"{vid}_orthogonal_slices.png",
                "3d_volume": f"{vid}_3d_volume.html",
            },
        }
        if d["mask"] is not None:
            entry["outputs"]["mask_analysis"] = f"{vid}_mask_analysis.png"
        manifest["volumes"].append(entry)

    manifest["global_outputs"] = {
        "volume_comparison": "volume_comparison.png",
        "summary_statistics": "summary_statistics.csv",
    }

    out_path = os.path.join(output_dir, "manifest.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  ✅ Saved {out_path}")


# =============================================================================
#  MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Vesuvius Scroll Volume Visualizer – generates per-volume EDA outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing .tif volume files")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save output images and HTML")
    parser.add_argument("--labels_dir", default=None,
                        help="(Optional) Directory containing matching label .tif files")
    parser.add_argument("--denoise", action="store_true",
                        help="Enable NL-means denoising (significantly slower)")
    parser.add_argument("--sub_volume_size", type=int, default=48,
                        help="Cube side length for 3D Plotly render (default: 48)")

    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"❌ Input directory not found: {args.input_dir}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Discover .tif files
    tif_files = sorted([f for f in os.listdir(args.input_dir) if f.lower().endswith((".tif", ".tiff"))])
    if not tif_files:
        print(f"❌ No .tif files found in {args.input_dir}")
        sys.exit(1)

    print("═" * 60)
    print("  VESUVIUS VOLUME VISUALIZER")
    print("═" * 60)
    print(f"  Input dir    : {args.input_dir}")
    print(f"  Output dir   : {args.output_dir}")
    print(f"  Labels dir   : {args.labels_dir or '(none)'}")
    print(f"  Denoise      : {args.denoise}")
    print(f"  3D sub-volume: {args.sub_volume_size}³")
    print(f"  Volumes found: {len(tif_files)}")
    print("═" * 60)

    # ── Load & preprocess all volumes ─────────────────────────────────────
    vol_data = {}
    for tif_name in tif_files:
        vol_id = os.path.splitext(tif_name)[0]
        img_path = os.path.join(args.input_dir, tif_name)
        label_path = None
        if args.labels_dir:
            candidate = os.path.join(args.labels_dir, tif_name)
            if os.path.exists(candidate):
                label_path = candidate

        print(f"\n📂 Loading: {vol_id}")
        result = load_and_preprocess(vol_id, img_path, label_path, run_denoise=args.denoise)
        if result is not None:
            vol_data[vol_id] = result

    if not vol_data:
        print("❌ No volumes could be loaded. Exiting.")
        sys.exit(1)

    print(f"\n✅ Successfully loaded {len(vol_data)} volume(s).\n")

    # ── Per-volume visualizations ─────────────────────────────────────────
    for vid, d in vol_data.items():
        print(f"\n🔬 Analyzing: {vid}")
        analyze_volume(d, args.output_dir)
        analyze_mask(d, args.output_dir)
        analyze_texture(d, args.output_dir)
        plot_preprocessing_comparison(d, args.output_dir)
        plot_orthogonal_slices(d, args.output_dir)
        generate_3d_volume_html(d, args.output_dir, sub_size=args.sub_volume_size)

    # ── Global outputs ────────────────────────────────────────────────────
    print("\n📊 Generating cross-volume comparison …")
    compare_volumes(vol_data, args.output_dir, n=min(3, len(vol_data)))
    save_summary_csv(vol_data, args.output_dir)
    save_manifest(vol_data, args.output_dir, args.sub_volume_size)

    print("\n" + "═" * 60)
    print("  ✅ ALL DONE")
    print(f"  Outputs saved to: {args.output_dir}")
    print("═" * 60)


if __name__ == "__main__":
    main()
