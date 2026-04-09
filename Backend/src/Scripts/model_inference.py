#!/usr/bin/env python3
"""
Vesuvius Surface Detection — Model Inference Script
====================================================
Standalone script extracted from the inference notebook.
Runs 3D U-Net and/or nnU-Net ensemble inference on .tif scroll volumes,
applies post-processing, and saves segmentation masks + visualizations.

Usage:
    python model_inference.py --input-dir /path/to/tif_volumes \
                              --output-dir /path/to/output \
                              --project-dir /path/to/cv_project \
                              [--checkpoints-dir /path/to/checkpoints] \
                              [--gt-dir /path/to/ground_truth_labels] \
                              [--skip-unet] [--skip-nnunet] \
                              [--t-low 0.2] [--t-high 0.83] [--min-size 2000] \
                              [--patch-size 128] [--overlap 0.5] \
                              [--device cuda:0]

Outputs (per volume):
    <vol_id>.tif                 — 3-class segmentation mask (0=air, 1=surface, 2=papyrus)
    <vol_id>_comparison.png      — axial slice comparison across models (+ GT if provided)
    <vol_id>_surface_prob.png    — surface probability heatmap at mid-slice

Global outputs:
    manifest.json                — JSON listing all volumes and their output files
"""

import os
import sys
import json
import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
from glob import glob

import tifffile
import scipy.ndimage as ndi
from scipy.ndimage import binary_closing, binary_fill_holes
from skimage.morphology import remove_small_objects, ball

import torch


# =============================================================================
#  INFERENCE FUNCTIONS
# =============================================================================


def unet_inference(model, volume_np, device, gaussian_fn, patch_size=128, overlap=0.5):
    """3D U-Net sliding-window inference with Gaussian weighting."""
    volume = (
        torch.from_numpy(volume_np.astype(np.float32) / 255.0)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )
    step = int(patch_size * (1 - overlap))
    D, H, W = volume.shape[2:]
    gaussian = gaussian_fn(patch_size).to(device)
    output_sum = torch.zeros(3, D, H, W, device=device)
    weight_sum = torch.zeros(1, D, H, W, device=device)

    def starts(sz):
        return sorted(
            set(list(range(0, max(sz - patch_size, 0) + 1, step)) + [max(sz - patch_size, 0)])
        )

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        for d in starts(D):
            for h in starts(H):
                for w in starts(W):
                    patch = volume[:, :, d : d + patch_size, h : h + patch_size, w : w + patch_size]
                    probs = torch.softmax(model(patch)["logits"].float(), dim=1)[0]
                    output_sum[:, d : d + patch_size, h : h + patch_size, w : w + patch_size] += (
                        probs * gaussian
                    )
                    weight_sum[:, d : d + patch_size, h : h + patch_size, w : w + patch_size] += gaussian

    return (output_sum / weight_sum.clamp(min=1e-8)).cpu().numpy()


def nnunet_inference(predictor, volume_np):
    """nnU-Net single-model inference."""
    img = volume_np.astype(np.float32)[np.newaxis]
    _, probs = predictor.predict_single_npy_array(
        img, {"spacing": [1.0, 1.0, 1.0]}, None, None, True
    )
    return probs  # (C, D, H, W)


def ensemble_nnunet(predictors_dict, volume_np):
    """Weighted ensemble of multiple nnU-Net models."""
    probs_sum = None
    for name, (predictor, weight) in predictors_dict.items():
        print(f"    Running nnU-Net [{name}] (weight={weight}) ...")
        probs = nnunet_inference(predictor, volume_np)
        if probs_sum is None:
            probs_sum = weight * probs
        else:
            probs_sum += weight * probs
        torch.cuda.empty_cache()
    return probs_sum


# =============================================================================
#  POST-PROCESSING
# =============================================================================


def postprocess(surface_probs, t_low=0.2, t_high=0.83, min_size=2000):
    """Hysteresis thresholding + morphological cleanup."""
    strong = surface_probs >= t_high
    weak = surface_probs >= t_low
    if not strong.any():
        return np.zeros_like(surface_probs, dtype=np.uint8)

    struct = ndi.generate_binary_structure(3, 3)

    # This step can allocate additional full-volume buffers on large inputs.
    # Fall back to a conservative strong-threshold mask if memory is tight.
    try:
        mask = ndi.binary_propagation(strong, mask=weak, structure=struct)
    except Exception as exc:
        if isinstance(exc, MemoryError) or "ArrayMemoryError" in type(exc).__name__:
            print("  [WARN] Low-memory fallback in postprocess: using strong-threshold mask")
            mask = strong.copy()
        else:
            raise

    # Closing
    try:
        mask = ndi.binary_closing(mask, structure=ndi.generate_binary_structure(3, 1), iterations=2)
    except Exception as exc:
        if isinstance(exc, MemoryError) or "ArrayMemoryError" in type(exc).__name__:
            print("  [WARN] Skipping binary_closing due to low memory")
        else:
            raise

    # Dust removal
    mask = remove_small_objects(mask, min_size=min_size)

    # Zero faces (3 voxels deep)
    for t in range(3):
        mask[t] = mask[-t - 1] = False
        mask[:, t] = mask[:, -t - 1] = False
        mask[:, :, t] = mask[:, :, -t - 1] = False
    mask = remove_small_objects(mask, min_size=1000)

    # Close + fill
    try:
        mask = binary_closing(mask, structure=ball(3))
    except Exception as exc:
        if isinstance(exc, MemoryError) or "ArrayMemoryError" in type(exc).__name__:
            print("  [WARN] Skipping ball-based closing due to low memory")
        else:
            raise
    for z in range(mask.shape[0]):
        if mask[z].any():
            mask[z] = binary_fill_holes(mask[z])
    mask = binary_fill_holes(mask)

    return mask.astype(np.uint8)


def to_3class(surface_mask, air_prob, papyrus_prob):
    """Convert binary surface mask to 3-class (0=air, 1=surface, 2=papyrus)."""
    result = np.zeros_like(surface_mask, dtype=np.uint8)
    result[surface_mask > 0] = 1
    non_surf = surface_mask == 0
    result[non_surf & (papyrus_prob > air_prob)] = 2
    return result


# =============================================================================
#  VISUALIZATION
# =============================================================================

SEG_CMAP = ListedColormap(["black", "red", "dodgerblue"])


def visualize_volume(image, predictions, sample_id, output_dir, gt=None):
    """Save axial slice comparison at 25%, 50%, 75% depth."""
    slices = [0.25, 0.5, 0.75]
    n_cols = 1 + (1 if gt is not None else 0) + len(predictions)
    fig, axes = plt.subplots(len(slices), n_cols, figsize=(4 * n_cols, 4 * len(slices)))
    if len(slices) == 1:
        axes = axes[np.newaxis, :]

    titles = ["CT Image"]
    if gt is not None:
        titles.append("Ground Truth")
    titles += list(predictions.keys())

    for row, frac in enumerate(slices):
        idx = int(frac * image.shape[0])
        img_s = image[idx]

        col = 0
        axes[row, col].imshow(img_s, cmap="gray")
        axes[row, col].set_ylabel(f"Slice {idx}", fontsize=11)
        col += 1

        if gt is not None:
            axes[row, col].imshow(img_s, cmap="gray", alpha=0.5)
            axes[row, col].imshow(gt[idx], cmap=SEG_CMAP, alpha=0.5, vmin=0, vmax=2)
            col += 1

        for name, pred in predictions.items():
            axes[row, col].imshow(img_s, cmap="gray", alpha=0.5)
            axes[row, col].imshow(pred[idx], cmap=SEG_CMAP, alpha=0.5, vmin=0, vmax=2)
            col += 1

        for c in range(n_cols):
            axes[row, c].set_xticks([])
            axes[row, c].set_yticks([])

    for c, t in enumerate(titles):
        axes[0, c].set_title(t, fontsize=11, fontweight="bold")

    plt.suptitle(f"Volume: {sample_id}", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{sample_id}_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {out_path}")


def visualize_surface_prob(image, prob_maps, sample_id, output_dir, gt=None):
    """Surface probability heatmap at mid-slice."""
    idx = image.shape[0] // 2
    n_cols = (1 if gt is not None else 0) + len(prob_maps)
    if n_cols == 0:
        return

    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    col = 0
    if gt is not None:
        axes[col].imshow(image[idx], cmap="gray", alpha=0.5)
        axes[col].imshow(gt[idx] == 1, cmap="Reds", alpha=0.6)
        axes[col].set_title("GT Surface", fontweight="bold")
        axes[col].set_xticks([])
        axes[col].set_yticks([])
        col += 1

    for name, prob in prob_maps.items():
        im = axes[col].imshow(prob[idx], cmap="hot", vmin=0, vmax=1)
        axes[col].set_title(name, fontweight="bold")
        axes[col].set_xticks([])
        axes[col].set_yticks([])
        plt.colorbar(im, ax=axes[col], fraction=0.046)
        col += 1

    plt.suptitle(f"Surface Probability - {sample_id} (slice {idx})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{sample_id}_surface_prob.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {out_path}")


# =============================================================================
#  MODEL LOADING
# =============================================================================


def load_custom_unet(project_dir, checkpoints_dir, device):
    """Load the custom 3D U-Net model from checkpoint."""
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)

    try:
        from src.models.unet3d import UNet3D
        from src.utils.utils import get_gaussian_3d
    except ImportError as e:
        print(f"  Could not import UNet3D from project: {e}")
        return None, None

    ckpt_path = os.path.join(checkpoints_dir, "checkpoint_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"  U-Net checkpoint not found: {ckpt_path}")
        return None, None

    model = UNet3D(1, 3, 32, [1, 2, 4, 8, 10], 2, True).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    epoch = ckpt.get("epoch", "?")
    sd = ckpt.get("best_surface_dice", 0)
    print(f"  3D U-Net loaded (epoch {epoch}, surface_dice={sd:.4f})")

    return model, get_gaussian_3d


def load_nnunet_ensemble(project_dir, device):
    """Load nnU-Net ensemble predictors."""
    os.environ["nnUNet_raw"] = os.path.join(project_dir, "nnUNet_data", "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = os.path.join(project_dir, "nnUNet_data", "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = os.path.join(project_dir, "nnUNet_data", "nnUNet_results")

    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    except ImportError as e:
        print(f"  Cannot import nnUNetPredictor: {e}")
        return {}

    results_base = os.path.join(
        project_dir, "nnUNet_data", "nnUNet_results", "Dataset200_VesuviusSurface"
    )

    model_configs = {
        "Default": ("nnUNetTrainer__nnUNetPlans__3d_fullres", 0.40),
        "MPlans": ("nnUNetTrainer_4000epochs__nnUNetResEncUNetMPlans__3d_fullres", 0.35),
        "LPlans": ("nnUNetTrainer_4000epochs__nnUNetResEncUNetLPlans__3d_fullres", 0.25),
    }

    predictors = {}
    for name, (subdir, weight) in model_configs.items():
        model_dir = os.path.join(results_base, subdir)
        ckpt_file = os.path.join(model_dir, "fold_all", "checkpoint_best.pth")
        if not os.path.exists(ckpt_file):
            print(f"  nnU-Net [{name}] checkpoint not found, skipping")
            continue

        pred = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            device=device,
            verbose=False,
        )
        pred.initialize_from_trained_model_folder(
            model_dir, use_folds=("all",), checkpoint_name="checkpoint_best.pth"
        )
        predictors[name] = (pred, weight)
        print(f"  nnU-Net [{name}] loaded (weight={weight})")

    return predictors


# =============================================================================
#  METRICS (optional, when GT is provided)
# =============================================================================


def compute_metrics(predictions, gt, project_dir):
    """Compute segmentation metrics if GT is available."""
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)

    try:
        from src.training.metrics import SegmentationMetrics
    except ImportError:
        print("  Cannot import SegmentationMetrics - skipping metrics.")
        return {}

    results = {}
    for name, pred in predictions.items():
        m = SegmentationMetrics(3, ["air", "surface", "papyrus"])
        m.update(torch.from_numpy(pred), torch.from_numpy(gt))
        r = m.compute()
        results[name] = {
            "surface_dice": float(r.get("surface_dice", 0)),
            "mean_dice": float(r.get("mean_dice", 0)),
        }
    return results


# =============================================================================
#  MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Vesuvius Surface Detection - run inference on .tif scroll volumes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory containing .tif volume files",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to save masks, visualizations, and manifest",
    )
    parser.add_argument(
        "--project-dir", default=None,
        help="Root of the CV project (contains src/, nnUNet_data/, etc.). "
             "Defaults to two levels above this script.",
    )
    parser.add_argument(
        "--checkpoints-dir", default=None,
        help="Directory containing checkpoint_best.pth for custom U-Net. "
             "Defaults to <project-dir>/checkpoints",
    )
    parser.add_argument(
        "--gt-dir", default=None,
        help="(Optional) Ground-truth labels directory for metric computation",
    )
    parser.add_argument("--skip-unet", action="store_true", help="Skip custom 3D U-Net")
    parser.add_argument("--skip-nnunet", action="store_true", help="Skip nnU-Net ensemble")
    parser.add_argument("--t-low", type=float, default=0.2, help="Hysteresis low threshold (default: 0.2)")
    parser.add_argument("--t-high", type=float, default=0.83, help="Hysteresis high threshold (default: 0.83)")
    parser.add_argument("--min-size", type=int, default=2000, help="Min component size for dust removal (default: 2000)")
    parser.add_argument("--patch-size", type=int, default=128, help="U-Net sliding window patch size (default: 128)")
    parser.add_argument("--overlap", type=float, default=0.5, help="U-Net sliding window overlap (default: 0.5)")
    parser.add_argument("--device", default=None, help="Torch device (default: auto-detect cuda/cpu)")

    args = parser.parse_args()

    # ── Resolve paths ─────────────────────────────────────────────────────
    if args.project_dir is None:
        backend_dir = Path(__file__).resolve().parent.parent.parent
        sibling_cv_project = backend_dir.parent / "CV_project"
        if sibling_cv_project.is_dir():
            args.project_dir = str(sibling_cv_project)
        else:
            args.project_dir = str(backend_dir)

    if args.checkpoints_dir is None:
        args.checkpoints_dir = os.path.join(args.project_dir, "checkpoints")

    if not os.path.isdir(args.input_dir):
        print(f"Input directory not found: {args.input_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ── Discover volumes ──────────────────────────────────────────────────
    tif_files = sorted(glob(os.path.join(args.input_dir, "*.tif")))
    if not tif_files:
        print(f"No .tif files found in {args.input_dir}")
        sys.exit(1)

    print("=" * 60)
    print("  VESUVIUS MODEL INFERENCE")
    print("=" * 60)
    print(f"  Input dir      : {args.input_dir}")
    print(f"  Output dir     : {args.output_dir}")
    print(f"  Project dir    : {args.project_dir}")
    print(f"  Checkpoints    : {args.checkpoints_dir}")
    print(f"  GT dir         : {args.gt_dir or '(none)'}")
    print(f"  Device         : {device}")
    print(f"  Custom U-Net   : {'skip' if args.skip_unet else 'enabled'}")
    print(f"  nnU-Net        : {'skip' if args.skip_nnunet else 'enabled'}")
    print(f"  Post-proc      : t_low={args.t_low}, t_high={args.t_high}, min_size={args.min_size}")
    print(f"  Patch/overlap  : {args.patch_size} / {args.overlap}")
    print(f"  Volumes found  : {len(tif_files)}")
    print("=" * 60)

    # ── Load models ───────────────────────────────────────────────────────
    model_unet = None
    gaussian_fn = None
    nnunet_predictors = {}

    if not args.skip_unet:
        print("\nLoading custom 3D U-Net ...")
        model_unet, gaussian_fn = load_custom_unet(args.project_dir, args.checkpoints_dir, device)

    if not args.skip_nnunet:
        print("\nLoading nnU-Net ensemble ...")
        nnunet_predictors = load_nnunet_ensemble(args.project_dir, device)

    if model_unet is None and not nnunet_predictors:
        print("No models could be loaded. Exiting.")
        sys.exit(1)

    print(
        f"\nModels ready: custom_unet={model_unet is not None}, "
        f"nnunet_models={list(nnunet_predictors.keys())}"
    )

    # ── Inference loop ────────────────────────────────────────────────────
    manifest_volumes = []

    for tif_path in tif_files:
        sample_id = Path(tif_path).stem
        print(f"\n{'=' * 60}")
        print(f"  Processing: {sample_id}")
        print(f"{'=' * 60}")

        image = tifffile.imread(tif_path)
        print(f"  Volume: {image.shape} {image.dtype}")

        # Load GT if available
        gt = None
        if args.gt_dir:
            gt_path = os.path.join(args.gt_dir, f"{sample_id}.tif")
            if os.path.exists(gt_path):
                gt = tifffile.imread(gt_path)
                print(f"  Ground truth loaded: {gt.shape}")

        predictions = {}
        prob_maps = {}
        metrics_all = {}

        # ── Custom 3D U-Net ──
        if model_unet is not None:
            print("  Running 3D U-Net ...")
            probs = unet_inference(
                model_unet, image, device, gaussian_fn,
                patch_size=args.patch_size, overlap=args.overlap,
            )
            surface_pp = postprocess(probs[1], args.t_low, args.t_high, args.min_size)
            pred_3c = to_3class(surface_pp, probs[0], probs[2])
            predictions["3D U-Net"] = pred_3c
            prob_maps["3D U-Net"] = probs[1]
            print(f"    Surface voxels: {(pred_3c == 1).sum():,}")

        # ── nnU-Net Ensemble ──
        if nnunet_predictors:
            print("  Running nnU-Net ensemble ...")
            probs_ens = ensemble_nnunet(nnunet_predictors, image)
            surface_pp = postprocess(probs_ens[1], args.t_low, args.t_high, args.min_size)
            pred_3c = to_3class(surface_pp, probs_ens[0], probs_ens[2])
            predictions["nnU-Net Ensemble"] = pred_3c
            prob_maps["nnU-Net Ensemble"] = probs_ens[1]
            print(f"    Surface voxels: {(pred_3c == 1).sum():,}")

        if not predictions:
            print(f"  No predictions for {sample_id}, skipping.")
            continue

        # ── Metrics ──
        if gt is not None:
            metrics_all = compute_metrics(predictions, gt, args.project_dir)
            if metrics_all:
                print(f"\n  {'Model':<25} {'Surface Dice':>13} {'Mean Dice':>10}")
                print(f"  {'-' * 50}")
                for name, m in metrics_all.items():
                    print(f"  {name:<25} {m['surface_dice']:>13.4f} {m['mean_dice']:>10.4f}")

        # ── Save best mask ──
        best_name = (
            "nnU-Net Ensemble" if "nnU-Net Ensemble" in predictions
            else list(predictions.keys())[0]
        )
        best_pred = predictions[best_name]
        mask_path = os.path.join(args.output_dir, f"{sample_id}.tif")
        tifffile.imwrite(mask_path, best_pred)
        print(f"\n  Saved mask: {mask_path}  (model: {best_name})")

        # ── Visualize ──
        visualize_volume(image, predictions, sample_id, args.output_dir, gt=gt)
        visualize_surface_prob(image, prob_maps, sample_id, args.output_dir, gt=gt)

        torch.cuda.empty_cache()

        # ── Manifest entry ──
        entry = {
            "vol_id": sample_id,
            "shape": list(image.shape),
            "best_model": best_name,
            "outputs": {
                "mask": f"{sample_id}.tif",
                "comparison": f"{sample_id}_comparison.png",
                "surface_prob": f"{sample_id}_surface_prob.png",
            },
        }
        if metrics_all:
            entry["metrics"] = metrics_all
        manifest_volumes.append(entry)

    # ── Save manifest ─────────────────────────────────────────────────────
    manifest = {
        "volumes": manifest_volumes,
        "config": {
            "t_low": args.t_low,
            "t_high": args.t_high,
            "min_size": args.min_size,
            "patch_size": args.patch_size,
            "overlap": args.overlap,
            "device": str(device),
            "models": {
                "custom_unet": model_unet is not None,
                "nnunet_models": list(nnunet_predictors.keys()),
            },
        },
    }
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Saved {manifest_path}")

    print("\n" + "=" * 60)
    print("  ALL DONE")
    print(f"  Outputs saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
