#!/usr/bin/env python3
"""
Bridge runner for TimeSformer ink detection inference.

Supports explicit command templates for external repositories like:
- jaredlandau/Vesuvius-Grandprize-Winner-Plus
- younader/Vesuvius-Grandprize-Winner

Template placeholders for --command:
  {repo_dir} {input_dir} {segmentation_dir} {output_dir} {checkpoint} {device}

After inference, the script creates enhanced text visualizations from model outputs.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import tifffile


def resolve_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    return Path(path_str).expanduser().resolve()


def run_external(command: list[str], cwd: Path, env: dict[str, str]) -> int:
    print(f"[ink-detect] Running: {' '.join(command)}")
    env.setdefault("WANDB_MODE", "disabled")
    completed = subprocess.run(command, cwd=str(cwd), env=env, check=False)
    return completed.returncode


def build_from_template(
    template: str,
    repo_dir: Path,
    input_dir: Path,
    segmentation_dir: Path,
    output_dir: Path,
    checkpoint: Path | None,
    device: str,
) -> list[str]:
    cmd = template.format(
        repo_dir=str(repo_dir),
        input_dir=str(input_dir),
        segmentation_dir=str(segmentation_dir),
        output_dir=str(output_dir),
        checkpoint=str(checkpoint) if checkpoint else "",
        device=device,
    ).strip()
    return shlex.split(cmd)


def autodetect_command(
    repo_dir: Path,
    segment_path: Path,
    segment_ids: list[str],
    output_dir: Path,
    checkpoint: Path | None,
    device: str,
    start_layer: int,
    num_layers: int,
) -> list[str] | None:
    candidates = [
        "inference_timesformer.py",
        "inference.py",
        "run_inference.py",
        "predict.py",
        "scripts/inference.py",
        "scripts/predict.py",
    ]

    for rel in candidates:
        script = repo_dir / rel
        if script.is_file():
            if script.name == "inference_timesformer.py":
                script_text = script.read_text(encoding="utf-8", errors="ignore")
                uses_start_idx = "start_idx" in script_text
                command = [
                    sys.executable,
                    str(script),
                    "--segment_path",
                    str(segment_path),
                    "--model_path",
                    str(checkpoint) if checkpoint else "",
                    "--out_path",
                    str(output_dir),
                ]
                if uses_start_idx:
                    command.extend(["--start_idx", str(start_layer)])
                else:
                    command.extend(["--start", str(start_layer), "--num_layers", str(num_layers)])
                if segment_ids:
                    command.append("--segment_id")
                    command.extend(segment_ids)
                return [piece for piece in command if piece != ""]

            command = [
                sys.executable,
                str(script),
                "--input-dir",
                str(segment_path),
                "--segmentation-dir",
                str(segment_path),
                "--output-dir",
                str(output_dir),
            ]
            if checkpoint:
                command.extend(["--checkpoint", str(checkpoint)])
            if device:
                command.extend(["--device", device])
            return command

    return None


def enhance_prob_map(image_path: Path, output_dir: Path) -> Path:
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
    binary = (stretched >= threshold).astype(np.uint8) * 255

    enhanced = (stretched * 255).astype(np.uint8)

    stem = image_path.stem
    enhanced_path = output_dir / f"{stem}_enhanced.png"
    binary_path = output_dir / f"{stem}_binary.png"

    Image.fromarray(enhanced).save(enhanced_path)
    Image.fromarray(binary).save(binary_path)

    return enhanced_path


def collect_candidate_images(output_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    scores = []

    for path in output_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue

        name = path.name.lower()
        score = 0
        for token in ("ink", "prob", "pred", "mask", "timesformer"):
            if token in name:
                score += 1
        scores.append((score, path))

    scores.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in scores[:20]]


def to_uint8_mask(mask_array: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask_array)
    if arr.ndim == 3:
        # If segmentation is volumetric, collapse depth to 2D mask.
        arr = np.any(arr > 0, axis=0).astype(np.uint8)
    else:
        arr = (arr > 0).astype(np.uint8)
    return arr * 255


def prepare_segment_layout(input_dir: Path, segmentation_dir: Path, work_root: Path) -> tuple[Path, list[str]]:
    """
    Build/normalize structure expected by winner inference scripts:
      <segment_path>/<fragment_id>/layers/00.tif, 01.tif, ...
      <segment_path>/<fragment_id>/<fragment_id>_mask.png (optional)
    """
    segment_path = work_root / "segments"
    segment_path.mkdir(parents=True, exist_ok=True)

    segment_ids: list[str] = []

    # Case A: input already has fragment folders with layers.
    for child in sorted(input_dir.iterdir() if input_dir.exists() else []):
        if child.is_dir() and (child / "layers").is_dir():
            segment_ids.append(child.name)

    if segment_ids:
        return input_dir, segment_ids

    # Case B: input contains TIFF volumes/files; convert to layer folders.
    tif_candidates = sorted(
        [
            p
            for p in input_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
        ]
    )

    for tif_file in tif_candidates:
        fragment_id = tif_file.stem
        fragment_dir = segment_path / fragment_id
        layers_dir = fragment_dir / "layers"
        layers_dir.mkdir(parents=True, exist_ok=True)

        volume = tifffile.imread(tif_file)
        if volume.ndim == 2:
            volume = volume[np.newaxis, ...]
        if volume.ndim != 3:
            print(f"[ink-detect] Skipping unsupported input shape for {tif_file.name}: {volume.shape}")
            continue

        for idx in range(volume.shape[0]):
            layer = volume[idx]
            layer_path = layers_dir / f"{idx:02}.tif"
            tifffile.imwrite(layer_path, layer)

        segment_ids.append(fragment_id)

    # Optional mask from segmentation output.
    for segment_id in segment_ids:
        seg_mask_tif = segmentation_dir / f"{segment_id}.tif"
        if not seg_mask_tif.exists():
            continue
        try:
            seg_mask = tifffile.imread(seg_mask_tif)
            if seg_mask.ndim == 3:
                surface = np.any(seg_mask == 1, axis=0).astype(np.uint8)
            else:
                surface = (seg_mask == 1).astype(np.uint8)
            mask_path = segment_path / segment_id / f"{segment_id}_mask.png"
            Image.fromarray((surface * 255).astype(np.uint8)).save(mask_path)
        except Exception as exc:
            print(f"[ink-detect] Could not create mask for {segment_id}: {exc}")

    return segment_path, segment_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Run external TimeSformer ink detection inference")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--segmentation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-dir", default=os.environ.get("INK_REPO_DIR"))
    parser.add_argument("--checkpoint", default=os.environ.get("INK_CHECKPOINT"))
    parser.add_argument("--command", default=os.environ.get("INK_COMMAND"))
    parser.add_argument("--device", default=os.environ.get("INK_DEVICE", "cuda:0"))
    parser.add_argument("--start-layer", type=int, default=int(os.environ.get("INK_START_LAYER", "17")))
    parser.add_argument("--num-layers", type=int, default=int(os.environ.get("INK_NUM_LAYERS", "5")))

    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    segmentation_dir = Path(args.segmentation_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = resolve_path(args.repo_dir)
    checkpoint = resolve_path(args.checkpoint)

    if repo_dir is None or not repo_dir.is_dir():
        raise SystemExit("INK_REPO_DIR is not set or does not exist. Clone a TimeSformer repo and set the path.")
    if checkpoint is None or not checkpoint.is_file():
        raise SystemExit("INK_CHECKPOINT is not set or does not exist. This stage only supports pretrained checkpoint inference.")

    prepared_root = output_dir / "_prepared"
    segment_path, segment_ids = prepare_segment_layout(input_dir, segmentation_dir, prepared_root)
    if not segment_ids:
        raise SystemExit("No valid segments found for ink detection. Provide fragment folders with layers or TIFF volumes.")

    env = os.environ.copy()
    env["INK_INPUT_DIR"] = str(input_dir)
    env["INK_SEGMENTATION_DIR"] = str(segmentation_dir)
    env["INK_OUTPUT_DIR"] = str(output_dir)
    env["INK_SEGMENT_PATH"] = str(segment_path)
    env["INK_REPO_DIR"] = str(repo_dir)
    env["INK_DEVICE"] = args.device
    env["INK_CHECKPOINT"] = str(checkpoint)

    if args.command:
        command = build_from_template(
            args.command,
            repo_dir,
            segment_path,
            segmentation_dir,
            output_dir,
            checkpoint,
            args.device,
        )
    else:
        command = autodetect_command(
            repo_dir,
            segment_path,
            segment_ids,
            output_dir,
            checkpoint,
            args.device,
            args.start_layer,
            args.num_layers,
        )

    if not command:
        raise SystemExit(
            "Could not auto-detect an ink inference entry point. Set INK_COMMAND with placeholders."
        )

    code = run_external(command, repo_dir, env)
    if code != 0:
        raise SystemExit(code)

    candidates = collect_candidate_images(output_dir)
    if not candidates:
        print("[ink-detect] No output images found to enhance")
    else:
        for image in candidates:
            try:
                enhance_prob_map(image, output_dir)
            except Exception as exc:
                print(f"[ink-detect] Enhancement skipped for {image.name}: {exc}")

    print("[ink-detect] Completed successfully")


if __name__ == "__main__":
    main()
