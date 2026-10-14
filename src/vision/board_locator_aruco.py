"""
Board localization via 4 outer-corner ArUco markers (IDs 40-43).

Marker layout (canonical board orientation, white side at bottom):
    40 --- 41
    |       |
    43 --- 42

Homography maps those 4 detected corners to a canonical 800x800 px image.
"""

from __future__ import annotations

import numpy as np
import cv2

# Dictionary must match the physical markers printed on the board.
_DICT_ID = cv2.aruco.DICT_4X4_50

# Expected corner marker IDs, in order: top-left, top-right, bottom-right, bottom-left.
CORNER_IDS = (40, 41, 42, 43)

# Output warp size (square).
WARP_SIZE = 800

# Canonical destination corners matching CORNER_IDS order (TL, TR, BR, BL).
_DST_CORNERS = np.array(
    [
        [0, 0],
        [WARP_SIZE - 1, 0],
        [WARP_SIZE - 1, WARP_SIZE - 1],
        [0, WARP_SIZE - 1],
    ],
    dtype=np.float32,
)


def _aruco_detector() -> cv2.aruco.ArucoDetector:
    """Build a cached-compatible ArucoDetector (cv2 >= 4.7 API)."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(_DICT_ID)
    params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(aruco_dict, params)


# Module-level singleton — avoids re-creating on every call.
_DETECTOR = _aruco_detector()


def _marker_center(corners_entry: np.ndarray) -> np.ndarray:
    """Return mean (cx, cy) of a single detected marker's 4 corners."""
    # corners_entry shape: (1, 4, 2)
    return corners_entry[0].mean(axis=0)


def locate_board(
    frame: np.ndarray,
    refine_subpix: bool = True,
) -> tuple[np.ndarray | None, dict[int, np.ndarray]]:
    """Detect corner ArUco markers and return a warped 800x800 board image.

    Parameters
    ----------
    frame:
        BGR image from the camera (must not be None, shape HxWx3).
    refine_subpix:
        If True, run cornerSubPix on each detected marker center for
        sub-pixel accuracy.

    Returns
    -------
    warped:
        800x800 BGR image of the board, or None when fewer than 4
        corner markers are visible.
    detected:
        Dict mapping marker_id -> center (x, y) in *frame* pixel coords.
        Always returned (may be partial).
    """
    assert frame is not None and frame.ndim == 3, "frame must be a valid BGR image"

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners_list, ids, _ = _DETECTOR.detectMarkers(gray)

    detected: dict[int, np.ndarray] = {}

    if ids is None or len(ids) == 0:
        return None, detected

    # Map id -> center.
    for marker_corners, marker_id in zip(corners_list, ids.flatten()):
        if marker_id in CORNER_IDS:
            center = _marker_center(marker_corners)
            detected[int(marker_id)] = center

    if len(detected) < 4:
        return None, detected

    # Sub-pixel refinement on gray image.
    if refine_subpix:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        pts = np.array(list(detected.values()), dtype=np.float32).reshape(-1, 1, 2)
        refined = cv2.cornerSubPix(gray, pts, (5, 5), (-1, -1), criteria)
        for i, mid in enumerate(detected):
            detected[mid] = refined[i, 0]

    # Build source-point array in CORNER_IDS order.
    src_corners = np.array(
        [detected[mid] for mid in CORNER_IDS],
        dtype=np.float32,
    )

    H, _ = cv2.findHomography(src_corners, _DST_CORNERS, cv2.RANSAC, 5.0)
    if H is None:
        return None, detected

    warped = cv2.warpPerspective(frame, H, (WARP_SIZE, WARP_SIZE))
    return warped, detected


def draw_detections(
    frame: np.ndarray,
    detected: dict[int, np.ndarray],
) -> np.ndarray:
    """Overlay marker IDs and center dots on *frame* (in-place safe copy).

    Returns a new BGR image; original frame is not modified.
    """
    assert frame is not None, "frame must not be None"
    vis = frame.copy()
    for mid, center in detected.items():
        cx, cy = int(center[0]), int(center[1])
        color = (0, 255, 0) if mid in CORNER_IDS else (255, 128, 0)
        cv2.circle(vis, (cx, cy), 8, color, -1)
        cv2.putText(
            vis,
            f"ID:{mid}",
            (cx + 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return vis
