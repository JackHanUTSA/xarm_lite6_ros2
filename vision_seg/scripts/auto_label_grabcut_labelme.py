import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def contour_to_points(cnt, eps=2.0):
    approx = cv2.approxPolyDP(cnt, eps, True)
    pts = approx.reshape(-1, 2).astype(float).tolist()
    return pts


def motion_union_mask(imgs, blur=5, thresh=20):
    # Build union of motion across all frames to seed probable foreground
    bg = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY)
    union = np.zeros(bg.shape, dtype=np.uint8)
    for im in imgs[1:]:
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        d = cv2.absdiff(g, bg)
        d = cv2.GaussianBlur(d, (blur, blur), 0)
        _, t = cv2.threshold(d, thresh, 255, cv2.THRESH_BINARY)
        union = cv2.bitwise_or(union, t)
    union = cv2.morphologyEx(union, cv2.MORPH_DILATE, np.ones((7,7), np.uint8), iterations=2)
    return union


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images_dir', required=True)
    ap.add_argument('--out_labelme_dir', required=True)
    ap.add_argument('--label', default='robot_arm')
    ap.add_argument('--rect', default='40,20,560,700', help='x,y,w,h in pixels')
    ap.add_argument('--min_area', type=int, default=5000)
    args = ap.parse_args()

    img_dir = Path(args.images_dir)
    out_dir = Path(args.out_labelme_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs_p = sorted(img_dir.glob('*.jpg'))
    if not imgs_p:
        raise SystemExit(f'No images in {img_dir}')

    imgs = []
    for p in imgs_p:
        im = cv2.imread(str(p))
        if im is not None:
            imgs.append(im)
    if not imgs:
        raise SystemExit('No readable images')

    x,y,w,h = [int(v) for v in args.rect.split(',')]
    x = max(0, x); y = max(0, y)

    union = motion_union_mask(imgs)

    wrote = 0
    for p in imgs_p:
        img = cv2.imread(str(p))
        if img is None:
            continue
        H,W = img.shape[:2]
        rect = (x, y, min(w, W-x), min(h, H-y))

        # GrabCut mask: 0=bg,1=fg,2=prob bg,3=prob fg
        gc_mask = np.full((H,W), cv2.GC_PR_BGD, dtype=np.uint8)
        # outside rect is sure background
        gc_mask[:] = cv2.GC_BGD
        rx,ry,rw,rh = rect
        gc_mask[ry:ry+rh, rx:rx+rw] = cv2.GC_PR_BGD

        # motion union inside rect is probable foreground
        u = union
        gc_mask[(u>0) & (gc_mask!=cv2.GC_BGD)] = cv2.GC_PR_FGD

        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)

        cv2.grabCut(img, gc_mask, rect, bgdModel, fgdModel, 4, mode=cv2.GC_INIT_WITH_MASK)

        mask = np.where((gc_mask==cv2.GC_FGD) | (gc_mask==cv2.GC_PR_FGD), 255, 0).astype('uint8')
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((5,5), np.uint8), iterations=2)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) < args.min_area:
            continue

        pts = contour_to_points(cnt, eps=2.5)
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
            'imageHeight': H,
            'imageWidth': W,
        }
        (out_dir / f'{p.stem}.json').write_text(json.dumps(data))
        wrote += 1

    print(f'WROTE_LABELS:{wrote} to {out_dir}')


if __name__ == '__main__':
    main()
