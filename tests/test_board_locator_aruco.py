"""Self-check for board_locator_aruco: synthetic markers, no camera needed."""
import sys
import numpy as np
import cv2

sys.path.insert(0, "/home/rifky/projects/github/UR-Chess-LLM-Arena")
from src.vision.board_locator_aruco import (
    CORNER_IDS,
    WARP_SIZE,
    locate_board,
    draw_detections,
    _DICT_ID,
)


def _make_synthetic_frame(size: int = 1200) -> np.ndarray:
    """Render 4 corner ArUco markers onto a white canvas at known positions."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(_DICT_ID)
    frame = np.ones((size, size, 3), dtype=np.uint8) * 200

    # Marker size and positions: corners of a centred 800x800 square.
    msz = 80
    pad = (size - 800) // 2
    positions = {
        40: (pad, pad),                          # top-left
        41: (pad + 800 - msz, pad),              # top-right
        42: (pad + 800 - msz, pad + 800 - msz), # bottom-right
        43: (pad, pad + 800 - msz),              # bottom-left
    }

    for mid, (col, row) in positions.items():
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, mid, msz)
        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        frame[row : row + msz, col : col + msz] = marker_bgr

    return frame


def test_full_detect():
    frame = _make_synthetic_frame()
    warped, detected = locate_board(frame, refine_subpix=False)

    assert set(detected.keys()) == set(CORNER_IDS), (
        f"Missing markers: {set(CORNER_IDS) - set(detected.keys())}"
    )
    assert warped is not None, "Homography warp failed"
    assert warped.shape == (WARP_SIZE, WARP_SIZE, 3), f"Bad warp shape: {warped.shape}"

    # Check draw_detections produces same-shaped image.
    vis = draw_detections(frame, detected)
    assert vis.shape == frame.shape


def test_partial_detect_returns_none():
    """Only 2 markers visible -> locate_board returns None warp."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(_DICT_ID)
    frame = np.ones((400, 400, 3), dtype=np.uint8) * 200
    msz = 60
    for mid in (40, 41):
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, mid, msz)
        frame[10:10+msz, 10+mid*2:10+mid*2+msz] = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)

    warped, detected = locate_board(frame, refine_subpix=False)
    assert warped is None, "Expected None for partial detection"


if __name__ == "__main__":
    test_full_detect()
    print("test_full_detect PASSED")
    test_partial_detect_returns_none()
    print("test_partial_detect_returns_none PASSED")
    print("ALL CHECKS PASSED")
