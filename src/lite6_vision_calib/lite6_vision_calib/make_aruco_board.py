import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm / 25.4 * dpi))


def render_gridboard(
    dict_name: str = 'DICT_4X4_50',
    markers_x: int = 2,
    markers_y: int = 2,
    marker_length_mm: float = 60.0,
    marker_separation_mm: float = 15.0,
    dpi: int = 300,
    margin_mm: float = 10.0,
) -> Tuple[np.ndarray, dict]:
    aruco = cv2.aruco
    aruco_dict = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
    board = aruco.GridBoard_create(
        markers_x,
        markers_y,
        marker_length_mm / 1000.0,
        marker_separation_mm / 1000.0,
        aruco_dict,
    )

    # Board size in pixels (approx)
    board_w_mm = markers_x * marker_length_mm + (markers_x - 1) * marker_separation_mm
    board_h_mm = markers_y * marker_length_mm + (markers_y - 1) * marker_separation_mm

    w_px = mm_to_px(board_w_mm + 2 * margin_mm, dpi)
    h_px = mm_to_px(board_h_mm + 2 * margin_mm, dpi)

    img = np.full((h_px, w_px), 255, dtype=np.uint8)

    # Draw board at margin
    draw_w = mm_to_px(board_w_mm, dpi)
    draw_h = mm_to_px(board_h_mm, dpi)
    board_img = np.full((draw_h, draw_w), 255, dtype=np.uint8)
    board.draw((draw_w, draw_h), board_img, marginSize=0, borderBits=1)

    mx = mm_to_px(margin_mm, dpi)
    my = mm_to_px(margin_mm, dpi)
    img[my:my+draw_h, mx:mx+draw_w] = board_img

    meta = {
        'dict': dict_name,
        'markers_x': markers_x,
        'markers_y': markers_y,
        'marker_length_mm': marker_length_mm,
        'marker_separation_mm': marker_separation_mm,
        'dpi': dpi,
        'margin_mm': margin_mm,
        'board_w_mm': board_w_mm,
        'board_h_mm': board_h_mm,
    }
    return img, meta


def main():
    out_dir = Path(os.path.expanduser('~/Pictures/aruco'))
    out_dir.mkdir(parents=True, exist_ok=True)

    img, meta = render_gridboard()
    png = out_dir / 'lite6_aruco_board_2x2_60mm.png'
    cv2.imwrite(str(png), img)

    # Also save a PDF via imagemagick-free route if available; otherwise just PNG.
    txt = out_dir / 'lite6_aruco_board_2x2_60mm.txt'
    txt.write_text(
        "Aruco GridBoard for calibration\n"
        f"meta={meta}\n\n"
        "Print at 100% scale. Measure a marker side to confirm 60mm.\n"
    )

    print(f"BOARD_PNG:{png}")
    print(f"META:{meta}")


if __name__ == '__main__':
    main()
