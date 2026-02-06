# Lite6 Visual Segmentation (YOLOv8-seg)

Goal: train a segmentation model that highlights the xArm Lite6 in camera video (mask + outline).

## 0) Pick a source video
Example (combined side-by-side):

```bash
VIDEO=/home/r91/Videos/robot_monitor/robot-2026-02-06_17-21-45.mp4
```

## 1) Extract frames for labeling

```bash
python3 vision_seg/scripts/extract_frames.py \
  --video "$VIDEO" \
  --out vision_seg/data/raw_frames \
  --count 120
```

This will save `frame_000000.jpg ...`.

## 2) Label frames (Labelme)

Install (recommended inside a venv/conda):

```bash
pip install labelme
```

Run:

```bash
labelme vision_seg/data/raw_frames --output vision_seg/data/labels_labelme
```

Create **one polygon class** named exactly:
- `robot_arm`

Labelme will write `.json` files.

## 3) Convert labelme -> YOLOv8 segmentation format

```bash
python3 vision_seg/scripts/labelme_to_yolo_seg.py \
  --labelme_dir vision_seg/data/labels_labelme \
  --images_dir vision_seg/data/raw_frames \
  --out_dir vision_seg/data/yolo \
  --val_ratio 0.15
```

## 4) Train (GPU)

```bash
pip install ultralytics
python3 vision_seg/scripts/train_yolov8_seg.py \
  --data vision_seg/data/yolo/data.yaml \
  --epochs 80 \
  --imgsz 640 \
  --batch 16
```

## 5) Inference on a video

```bash
python3 vision_seg/scripts/infer_video.py \
  --weights vision_seg/runs/segment/train/weights/best.pt \
  --video "$VIDEO" \
  --out_dir vision_seg/runs/infer
```

Output will include an annotated video with masks.
