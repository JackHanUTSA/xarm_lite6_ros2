import argparse
import json
import random
import shutil
from pathlib import Path

import cv2

# YOLOv8-seg expects: one txt per image.
# Each line: class_id x1 y1 x2 y2 ... normalized (0-1), polygon points.

CLASS_NAME = 'robot_arm'


def poly_to_yolo(points, w, h):
    coords = []
    for x, y in points:
        coords.append(x / w)
        coords.append(y / h)
    return coords


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labelme_dir', required=True)
    ap.add_argument('--images_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--val_ratio', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    lm_dir = Path(args.labelme_dir)
    img_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)

    (out_dir / 'images/train').mkdir(parents=True, exist_ok=True)
    (out_dir / 'images/val').mkdir(parents=True, exist_ok=True)
    (out_dir / 'labels/train').mkdir(parents=True, exist_ok=True)
    (out_dir / 'labels/val').mkdir(parents=True, exist_ok=True)

    jsons = sorted(lm_dir.glob('*.json'))
    if not jsons:
        raise SystemExit(f'No labelme json found in {lm_dir}')

    random.seed(args.seed)
    random.shuffle(jsons)
    n_val = int(len(jsons) * args.val_ratio)
    val_set = set(jsons[:n_val])

    for js in jsons:
        data = json.loads(js.read_text())
        image_filename = data.get('imagePath')
        if not image_filename:
            # labelme may store full path; fall back to json name
            image_filename = js.with_suffix('.jpg').name

        src_img = img_dir / Path(image_filename).name
        if not src_img.exists():
            # common if labelme wrote .png
            alt = img_dir / (Path(image_filename).stem + '.jpg')
            if alt.exists():
                src_img = alt
            else:
                raise SystemExit(f'Missing image for {js}: expected {src_img}')

        img = cv2.imread(str(src_img))
        if img is None:
            raise SystemExit(f'Cannot read image: {src_img}')
        h, w = img.shape[:2]

        lines = []
        for shape in data.get('shapes', []):
            if shape.get('shape_type') != 'polygon':
                continue
            if shape.get('label') != CLASS_NAME:
                continue
            pts = shape.get('points', [])
            if len(pts) < 3:
                continue
            coords = poly_to_yolo(pts, w, h)
            lines.append('0 ' + ' '.join(f'{c:.6f}' for c in coords))

        split = 'val' if js in val_set else 'train'
        dst_img = out_dir / f'images/{split}/{src_img.name}'
        dst_lbl = out_dir / f'labels/{split}/{src_img.stem}.txt'
        shutil.copy2(src_img, dst_img)
        dst_lbl.write_text('\n'.join(lines) + ('\n' if lines else ''))

    # Write data.yaml
    yaml = out_dir / 'data.yaml'
    yaml.write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: robot_arm\n"
    )
    print(f'Wrote dataset to {out_dir}')


if __name__ == '__main__':
    main()
