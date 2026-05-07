import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def contour_to_points(cnt, eps=2.0):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps, True)
    pts = approx.reshape(-1, 2).astype(float).tolist()
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images_dir', required=True)
    ap.add_argument('--out_labelme_dir', required=True)
    ap.add_argument('--label', default='robot_arm')
    ap.add_argument('--min_area', type=int, default=2000)
    args = ap.parse_args()

    img_dir = Path(args.images_dir)
    out_dir = Path(args.out_labelme_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(img_dir.glob('*.jpg'))
    if not imgs:
        raise SystemExit(f'No images in {img_dir}')

    # Use first frame as background reference
    bg = cv2.imread(str(imgs[0]))
    if bg is None:
        raise SystemExit('Cannot read first image')
    bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

    wrote = 0
    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, bg_gray)
        diff = cv2.GaussianBlur(diff, (5,5), 0)
        _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5,5), np.uint8), iterations=1)
        th = cv2.morphologyEx(th, cv2.MORPH_DILATE, np.ones((7,7), np.uint8), iterations=2)

        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) < args.min_area:
            continue

        pts = contour_to_points(cnt, eps=3.0)
        if len(pts) < 3:
            continue

        data = {
            'version': '5.0.0',
            'flags': {},
            'shapes': [
                {
                    'label': args.label,
                    'points': pts,
                    'group_id': None,
                    'shape_type': 'polygon',
                    'flags': {},
                }
            ],
            'imagePath': p.name,
            'imageData': None,
            'imageHeight': img.shape[0],
            'imageWidth': img.shape[1],
        }
        (out_dir / f'{p.stem}.json').write_text(json.dumps(data))
        wrote += 1

    print(f'WROTE_LABELS:{wrote} to {out_dir}')


if __name__ == '__main__':
    main()
