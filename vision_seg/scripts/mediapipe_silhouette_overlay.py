import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--bg', default=None, help='optional background image path (same size as frames)')
    ap.add_argument('--thr', type=int, default=25)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f'Cannot open video: {args.video}')

    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    bg = None
    if args.bg:
        bg = cv2.imread(args.bg)
    else:
        # use first frame as background reference
        ok, first = cap.read()
        if not ok:
            raise SystemExit('Empty video')
        bg = first
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, bg_gray)
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        _, mask = cv2.threshold(diff, args.thr, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((7, 7), np.uint8), iterations=2)

        # Find largest contour as robot silhouette candidate
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = frame.copy()
        if cnts:
            cnt = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(cnt) > 2000:
                # filled mask overlay
                overlay = out.copy()
                cv2.drawContours(overlay, [cnt], -1, (0, 255, 0), thickness=cv2.FILLED)
                out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
                # outline
                cv2.drawContours(out, [cnt], -1, (0, 255, 0), thickness=2)

        vw.write(out)

    cap.release()
    vw.release()
    print(f'WROTE:{out_path}')


if __name__ == '__main__':
    main()
