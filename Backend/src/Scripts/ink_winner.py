#!/usr/bin/env python3
"""
Ink-detection driver for the TimeSformer Winner checkpoints.

Expects input segments in the Kaggle ink-detection format:
  <source_dir>/<segment_id>/
    surface_volume/00.tif, 01.tif, ..., 64.tif
    mask.png

Stages a read-only symlink tree matching what `inference_timesformer.py`
wants:
  <staging_dir>/<segment_id>/
    layers/00.tif, 01.tif, ..., 64.tif    (symlinks into surface_volume)
    <segment_id>_mask.png                 (symlink to mask.png)

Then invokes the Winner's inference_timesformer.py directly (no format
conversion, no re-tiling) and copies prediction PNGs into --output-dir.

Environment knobs:
  INK_REPO_DIR          path to Winner repo (has inference_timesformer.py)
  INK_BATCH_SIZE        batch size passed to Winner (default 2 — 4 GB safe)
  INK_WORKERS           dataloader workers passed to Winner (default 2)
  INK_DEVICE            "cuda:0" (unused by Winner; inherited via env)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def resolve_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    return Path(path_str).expanduser().resolve()


def stage_segments(source_dir: Path, segment_ids: list[str], staging_dir: Path) -> list[str]:
    """
    Build Winner-compatible symlink tree under staging_dir. Returns the
    list of segment ids that were successfully staged (silently drops
    segments missing required files).
    """
    ok: list[str] = []
    for sid in segment_ids:
        src_seg = source_dir / sid
        src_layers = src_seg / "surface_volume"
        src_mask = src_seg / "mask.png"

        if not src_layers.is_dir():
            print(f"[ink-winner] Skipping {sid}: no surface_volume/ directory", flush=True)
            continue
        if not src_mask.is_file():
            print(f"[ink-winner] Skipping {sid}: no mask.png", flush=True)
            continue

        dst_seg = staging_dir / sid
        dst_layers = dst_seg / "layers"
        dst_mask = dst_seg / f"{sid}_mask.png"
        dst_seg.mkdir(parents=True, exist_ok=True)

        if not dst_layers.exists():
            os.symlink(src_layers.resolve(), dst_layers)
        if not dst_mask.exists():
            os.symlink(src_mask.resolve(), dst_mask)

        ok.append(sid)
    return ok


def enhance_prediction(image_path: Path, output_dir: Path) -> None:
    """
    Percentile-stretch + Otsu-ish binarize. Produces two sibling images:
    <stem>_enhanced.png and <stem>_binary.png.
    """
    img = Image.open(image_path)
    gray = ImageOps.grayscale(img)
    arr = np.asarray(gray, dtype=np.float32)

    if arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    else:
        arr = np.zeros_like(arr)

    p_low = float(np.percentile(arr, 5))
    p_high = float(np.percentile(arr, 99))
    if p_high <= p_low:
        p_low, p_high = 0.0, 1.0
    stretched = np.clip((arr - p_low) / (p_high - p_low + 1e-6), 0, 1)

    threshold = float(np.percentile(stretched, 85))
    enhanced = (stretched * 255).astype(np.uint8)
    binary = (stretched >= threshold).astype(np.uint8) * 255

    Image.fromarray(enhanced).save(output_dir / f"{image_path.stem}_enhanced.png")
    Image.fromarray(binary).save(output_dir / f"{image_path.stem}_binary.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TimeSformer Winner ink detection on Kaggle-format segments")
    parser.add_argument("--source-dir", required=True, help="Root of Kaggle-format segments (has <id>/surface_volume, <id>/mask.png)")
    parser.add_argument("--segment-id", action="append", default=[], help="Repeatable; at least one required")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-dir", default=os.environ.get("INK_REPO_DIR"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--start-layer", type=int, default=int(os.environ.get("INK_START_LAYER", "17")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("INK_BATCH_SIZE", "2")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("INK_WORKERS", "2")))

    args = parser.parse_args()

    source_dir = Path(args.source_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = resolve_path(args.repo_dir)
    checkpoint = resolve_path(args.checkpoint)

    if repo_dir is None or not repo_dir.is_dir():
        raise SystemExit("INK_REPO_DIR missing or invalid")
    if checkpoint is None or not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if not args.segment_id:
        raise SystemExit("Pass at least one --segment-id")
    if not source_dir.is_dir():
        raise SystemExit(f"--source-dir not found: {source_dir}")

    script = repo_dir / "inference_timesformer.py"
    if not script.is_file():
        raise SystemExit(f"inference_timesformer.py not found in {repo_dir}")

    staging_dir = Path(tempfile.mkdtemp(prefix="ink_winner_"))
    try:
        staged = stage_segments(source_dir, args.segment_id, staging_dir)
        if not staged:
            raise SystemExit("No segments successfully staged. Check source-dir layout.")

        command = [
            sys.executable,
            str(script),
            "--segment_path", str(staging_dir),
            "--model_path", str(checkpoint),
            "--out_path", str(output_dir),
            "--batch_size", str(args.batch_size),
            "--workers", str(args.workers),
            "--gpus", "1",
            "--start_idx", str(args.start_layer),
            "--segment_id", *staged,
        ]

        env = os.environ.copy()
        env.setdefault("WANDB_MODE", "disabled")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        print(f"[ink-winner] Staged: {staged}", flush=True)
        print(f"[ink-winner] Running: {' '.join(command)}", flush=True)

        rc = subprocess.run(command, cwd=str(repo_dir), env=env, check=False).returncode
        if rc != 0:
            raise SystemExit(rc)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    for pred in sorted(output_dir.glob("*_prediction_rotated_*.png")):
        try:
            enhance_prediction(pred, output_dir)
        except Exception as exc:
            print(f"[ink-winner] Enhancement skipped for {pred.name}: {exc}", flush=True)

    print("[ink-winner] Completed successfully", flush=True)


if __name__ == "__main__":
    main()
