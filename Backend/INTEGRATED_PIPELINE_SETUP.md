# Integrated Segmentation + Ink Detection Setup

This backend can now run optional external stages after the existing model inference:

- Kaggle 1st-place segmentation baseline (ainatersol/Vesuvius-InkDetection)
- TimeSformer ink detection + text enhancement

Important: the ink stage is checkpoint-only inference. No training/fine-tuning path is used.

## 1) Enable stages in environment

Add these keys to Baselines/Backend/.env:

ENABLE_KAGGLE_SEG=true
ENABLE_INK_DETECTION=true

## 2) Clone external repositories

Example locations:

E:/Project/Computer Vision/external/Vesuvius-InkDetection
E:/Project/Computer Vision/external/Vesuvius-Grandprize-Winner-Plus

## 3) Configure paths and commands

Add these keys to Baselines/Backend/.env.

### Kaggle segmentation bridge

KAGGLE_SEG_REPO_DIR=E:/Project/Computer Vision/external/Vesuvius-InkDetection
KAGGLE_SEG_WEIGHTS_DIR=E:/Project/Computer Vision/external/weights/unet3d
KAGGLE_SEG_DEVICE=cuda:0
KAGGLE_SEG_COMMAND=python inference.py --input-dir {input_dir} --output-dir {output_dir} --weights-dir {weights_dir} --device {device}

### TimeSformer bridge

INK_REPO_DIR=E:/Project/Computer Vision/external/Vesuvius-Grandprize-Winner-Plus
INK_CHECKPOINT=E:/Project/Computer Vision/external/weights/timesformer_weights.ckpt
INK_DEVICE=cuda:0
INK_START_LAYER=17
INK_NUM_LAYERS=5
INK_COMMAND=python inference_timesformer.py --segment_path {input_dir} --model_path {checkpoint} --out_path {output_dir} --start 17 --num_layers 5

For the original winner repo, this command style is also supported:

INK_COMMAND=python inference_timesformer.py --model_path {checkpoint} --segment_path {input_dir} --segment_id 20231005123336 --out_path {output_dir}

If INK_COMMAND is empty, the bridge auto-detects inference_timesformer.py and builds args automatically.

## Inputs by stage

### Part 1 input (segmentation)

- Your current uploaded TIFF volume(s).

### Part 2 input (ink detection)

- Segment folders in winner format:
	- <segment_path>/<fragment_id>/layers/00.tif, 01.tif, ...
	- optional <segment_path>/<fragment_id>/<fragment_id>_mask.png
- Pretrained checkpoint file (timesformer_weights.ckpt).

The bridge can convert single multi-page TIFF volumes into this segment format automatically.

### Part 3 input (text recovery)

- The Part 2 output probability maps (PNG/TIFF) from TimeSformer.
- No training input. This stage applies post-processing only (contrast stretch + thresholded binary map).

## 4) Output folders exposed to frontend

When enabled, new files appear under:

- outputs/kaggle_seg/
- outputs/text_recovery/

The frontend groups these into Segmentation and Text Recovery tabs.

## 5) Notes

- If your external repo CLI differs, update KAGGLE_SEG_COMMAND or INK_COMMAND only.
- The ink bridge auto-generates enhanced PNG outputs with contrast stretch and binary threshold for readability.
- The backend continues even if optional stages fail and reports warnings in the UI.
