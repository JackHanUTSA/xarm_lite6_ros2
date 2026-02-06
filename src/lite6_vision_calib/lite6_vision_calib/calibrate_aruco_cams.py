import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class CamCalib:
    K: np.ndarray
    dist: np.ndarray
    rms: float


def open_cam(dev: str, width=1280, height=720, fps=15):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open camera {dev}')
    return cap


def capture_samples(dev: str, n=25, out_dir=None, show=False):
    out_dir = Path(out_dir) if out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    cap = open_cam(dev)
    try:
        samples = []
        i = 0
        last_save = 0.0
        while i < n:
            ok, frame = cap.read()
            if not ok:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            samples.append(gray)

            now = time.time()
            if out_dir and now - last_save > 0.2:
                cv2.imwrite(str(out_dir / f'{i:03d}.png'), gray)
                last_save = now

            i += 1
            if show:
                cv2.imshow(f'cap {dev}', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        return samples
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()


def calibrate_from_samples(samples, board, aruco_dict):
    aruco = cv2.aruco
    all_corners = []
    all_ids = []
    counter = []

    for img in samples:
        corners, ids, _rej = aruco.detectMarkers(img, aruco_dict)
        if ids is None or len(ids) == 0:
            continue
        all_corners.extend(corners)
        all_ids.extend(ids)
        counter.append(len(ids))

    if len(all_ids) < 10:
        raise RuntimeError('Not enough detections; make sure the board is visible and well-lit')

    imsize = samples[0].shape[::-1]
    camera_matrix_init = np.array([[imsize[0], 0, imsize[0]/2], [0, imsize[0], imsize[1]/2], [0, 0, 1]], dtype=np.float64)
    dist_init = np.zeros((5, 1), dtype=np.float64)

    flags = (cv2.CALIB_USE_INTRINSIC_GUESS)

    rms, K, dist, rvecs, tvecs = aruco.calibrateCameraAruco(
        corners=all_corners,
        ids=np.array(all_ids),
        counter=np.array(counter),
        board=board,
        imageSize=imsize,
        cameraMatrix=camera_matrix_init,
        distCoeffs=dist_init,
        flags=flags,
    )
    return CamCalib(K=K, dist=dist, rms=float(rms))


def save_yaml(path: Path, K, dist, rms: float):
    import yaml
    data = {
        'rms': rms,
        'camera_matrix': K.tolist(),
        'dist_coeffs': dist.reshape(-1).tolist(),
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def main():
    out_dir = Path(os.path.expanduser('~/ws_xarm/calibration'))
    out_dir.mkdir(parents=True, exist_ok=True)

    aruco = cv2.aruco
    dict_name = 'DICT_4X4_50'
    aruco_dict = aruco.getPredefinedDictionary(getattr(aruco, dict_name))

    # Must match printed board
    markers_x, markers_y = 2, 2
    marker_length_m = 0.060
    marker_sep_m = 0.015
    board = aruco.GridBoard_create(markers_x, markers_y, marker_length_m, marker_sep_m, aruco_dict)

    cams = [('/dev/video2', 'c920_left'), ('/dev/video0', 'm9_right')]

    for dev, name in cams:
        print(f'Capturing samples for {name} ({dev})...')
        samples = capture_samples(dev, n=35, out_dir=out_dir / f'samples_{name}', show=False)
        print('Calibrating...')
        calib = calibrate_from_samples(samples, board, aruco_dict)
        y = out_dir / f'{name}.yaml'
        save_yaml(y, calib.K, calib.dist, calib.rms)
        print(f'SAVED:{y} rms={calib.rms:.4f}')


if __name__ == '__main__':
    main()
