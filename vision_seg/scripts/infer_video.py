import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--video', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--conf', type=float, default=0.25)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    model.predict(
        source=args.video,
        conf=args.conf,
        save=True,
        project=str(out_dir),
        name='pred',
        task='segment',
    )


if __name__ == '__main__':
    main()
