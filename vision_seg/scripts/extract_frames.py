import argparse
import os
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--count', type=int, default=120)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f'Cannot open video: {args.video}')

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames <= 0:
        raise SystemExit('Video has unknown frame count')

    step = max(1, n_frames // args.count)
    idx = 0
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0 and saved < args.count:
            out_path = out_dir / f'frame_{saved:06d}.jpg'
            cv2.imwrite(str(out_path), frame)
            saved += 1
        idx += 1

    cap.release()
    print(f'Saved {saved} frames to {out_dir}')


if __name__ == '__main__':
    main()
