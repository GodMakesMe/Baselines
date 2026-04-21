#!/usr/bin/env python3
"""
Bridge runner for Kaggle 1st-place segmentation inference.

This script does not hardcode one repository layout. Instead, it supports:
1) Explicit command templates for the external repo.
2) A best-effort auto-detection mode for common inference entry points.

Template placeholders for --command:
  {repo_dir} {input_dir} {output_dir} {weights_dir} {device}
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def resolve_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    return Path(path_str).expanduser().resolve()


def run_external(command: list[str], cwd: Path, env: dict[str, str]) -> int:
    print(f"[kaggle-seg] Running: {' '.join(command)}")
    completed = subprocess.run(command, cwd=str(cwd), env=env, check=False)
    return completed.returncode


def build_from_template(template: str, repo_dir: Path, input_dir: Path, output_dir: Path, weights_dir: Path | None, device: str) -> list[str]:
    cmd = template.format(
        repo_dir=str(repo_dir),
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        weights_dir=str(weights_dir) if weights_dir else "",
        device=device,
    ).strip()
    return shlex.split(cmd)


def autodetect_command(repo_dir: Path, input_dir: Path, output_dir: Path, weights_dir: Path | None, device: str) -> list[str] | None:
    candidates = [
        "inference.py",
        "run_inference.py",
        "scripts/inference.py",
        "src/inference.py",
    ]

    for rel in candidates:
        script = repo_dir / rel
        if script.is_file():
            command = [sys.executable, str(script), "--input-dir", str(input_dir), "--output-dir", str(output_dir)]
            if weights_dir:
                command.extend(["--weights-dir", str(weights_dir)])
            if device:
                command.extend(["--device", device])
            return command

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run external Kaggle segmentation inference")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-dir", default=os.environ.get("KAGGLE_SEG_REPO_DIR"))
    parser.add_argument("--weights-dir", default=os.environ.get("KAGGLE_SEG_WEIGHTS_DIR"))
    parser.add_argument("--command", default=os.environ.get("KAGGLE_SEG_COMMAND"))
    parser.add_argument("--device", default=os.environ.get("KAGGLE_SEG_DEVICE", "cuda:0"))

    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = resolve_path(args.repo_dir)
    weights_dir = resolve_path(args.weights_dir)

    if repo_dir is None or not repo_dir.is_dir():
        raise SystemExit("KAGGLE_SEG_REPO_DIR is not set or does not exist. Clone ainatersol/Vesuvius-InkDetection and set the path.")

    env = os.environ.copy()
    env["KAGGLE_SEG_INPUT_DIR"] = str(input_dir)
    env["KAGGLE_SEG_OUTPUT_DIR"] = str(output_dir)
    env["KAGGLE_SEG_REPO_DIR"] = str(repo_dir)
    env["KAGGLE_SEG_DEVICE"] = args.device
    if weights_dir:
        env["KAGGLE_SEG_WEIGHTS_DIR"] = str(weights_dir)

    if args.command:
        command = build_from_template(args.command, repo_dir, input_dir, output_dir, weights_dir, args.device)
    else:
        command = autodetect_command(repo_dir, input_dir, output_dir, weights_dir, args.device)

    if not command:
        raise SystemExit(
            "Could not auto-detect a Kaggle segmentation entry point. Set KAGGLE_SEG_COMMAND with placeholders."
        )

    code = run_external(command, repo_dir, env)
    if code != 0:
        raise SystemExit(code)

    print("[kaggle-seg] Completed successfully")


if __name__ == "__main__":
    main()
