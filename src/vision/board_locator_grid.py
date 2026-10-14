"""Board localization via outer contour detection (no ArUco corner markers).

Detects the outer black framing contour of an 8x8 chess grid, extracts the
four corner points, and warps the board region to a canonical 800x800 px image.
Each of the 64 cells occupies exactly 100x100 px in the output.

Pipeline:
    1. Grayscale + Canny edge detection.
    2. Find largest 4-vertex external contour (the board frame).
    3. Order corners TL→TR→BR→BL.
    4. Perspective warp to 800x800 px.
    5. Optional: verify 8x8 alternating pattern (dark/light cell count).
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Output canonical size; each of 8 cells = OUTPUT_SIZE // 8 px.
OUTPUT_SIZE: int = 800

# Canny thresholds tuned against assets/reference_board.jpg.
_CANNY_LOW: int = 50
_CANNY_HIGH: int = 150

# approxPolyDP epsilon factor.
_POLY_EPSILON: float = 0.02

# Minimum board contour area (px²) in input image.
_MIN_BOARD_AREA: float = 1_000.0


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Return pts ordered [TL, TR, BR, BL] from an unordered (4,2) float32 array."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).squeeze()
    rect[0] = pts[np.argmin(s)]     # TL: smallest x+y
    rect[2] = pts[np.argmax(s)]     # BR: largest  x+y
    rect[1] = pts[np.argmin(diff)]  # TR: smallest x-y
    rect[3] = pts[np.argmax(diff)]  # BL: largest  x-y
    return rect


def detect_board_corners(
    frame: np.ndarray,
    *,
    canny_low: int = _CANNY_LOW,
    canny_high: int = _CANNY_HIGH,
) -> Optional[np.ndarray]:
    """Detect the four outer corners of the chess board in *frame*.

    Args:
        frame:      BGR or grayscale image (H×W×3 or H×W).
        canny_low:  Lower Canny threshold.
        canny_high: Upper Canny threshold.

    Returns:
        Float32 (4,2) array of [TL, TR, BR, BL] pixel coordinates in *frame*,
        or None if detection failed.
    """
    if frame is None or frame.size == 0:
        logger.warning("detect_board_corners: empty frame")
        return None

    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, canny_low, canny_high)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logger.debug("detect_board_corners: no contours found")
        return None

    # Sort descending by area; take first 4-vertex contour large enough.
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    H, W = gray.shape[:2]
    min_full_board_area = 0.20 * H * W

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_full_board_area:
            break
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, _POLY_EPSILON * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            return _order_points(pts)

    # Fallback for boards with pieces: Morphological close connects alternating squares
    k_size = max(15, int(min(H, W) * 0.12))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    _, bin_dark = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    closed = cv2.morphologyEx(bin_dark, cv2.MORPH_CLOSE, kernel)

    closed_cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in sorted(closed_cnts, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(cnt) > min_full_board_area:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect).astype(np.float32)
            ordered_box = _order_points(box)
            # Refine edges against outer framing lines
            refined = _refine_grid_border(gray, ordered_box)
            return refined

    logger.warning("detect_board_corners: no 4-vertex contour found")
    return None


def _refine_grid_border(gray: np.ndarray, coarse_box: np.ndarray) -> np.ndarray:
    """Refine coarse bounding box to the exact outer boundary lines of the 8x8 grid."""
    xmin, xmax = int(coarse_box[:, 0].min()), int(coarse_box[:, 0].max())
    ymin, ymax = int(coarse_box[:, 1].min()), int(coarse_box[:, 1].max())
    bw, bh = xmax - xmin, ymax - ymin

    left_candidates, right_candidates = [], []
    for y in range(ymin + int(0.2 * bh), ymax - int(0.2 * bh), 10):
        row_left = gray[y, xmin:xmin + int(0.15 * bw)]
        idx_l = np.where(row_left < 80)[0]
        if len(idx_l) > 0:
            left_candidates.append(xmin + idx_l[0])
        row_right = gray[y, xmax - int(0.15 * bw):xmax]
        idx_r = np.where(row_right < 80)[0]
        if len(idx_r) > 0:
            right_candidates.append(xmax - int(0.15 * bw) + idx_r[-1])

    top_candidates, bottom_candidates = [], []
    for x in range(xmin + int(0.2 * bw), xmax - int(0.2 * bw), 10):
        col_top = gray[ymin:ymin + int(0.15 * bh), x]
        idx_t = np.where(col_top < 80)[0]
        if len(idx_t) > 0:
            top_candidates.append(ymin + idx_t[0])
        col_bottom = gray[ymax - int(0.15 * bh):ymax, x]
        idx_b = np.where(col_bottom < 80)[0]
        if len(idx_b) > 0:
            bottom_candidates.append(ymax - int(0.15 * bh) + idx_b[-1])

    x_left = float(np.median(left_candidates)) if left_candidates else float(xmin)
    x_right = float(np.median(right_candidates)) if right_candidates else float(xmax)
    y_top = float(np.median(top_candidates)) if top_candidates else float(ymin)
    y_bottom = float(np.median(bottom_candidates)) if bottom_candidates else float(ymax)

    return np.array([
        [x_left, y_top],
        [x_right, y_top],
        [x_right, y_bottom],
        [x_left, y_bottom]
    ], dtype=np.float32)


def warp_board(
    frame: np.ndarray,
    corners: np.ndarray,
    output_size: int = OUTPUT_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp *frame* so that *corners* map to a square canonical image.

    Args:
        frame:       Source BGR image.
        corners:     (4,2) float32 [TL, TR, BR, BL] in *frame* coordinates.
        output_size: Side length of the output square (default 800).

    Returns:
        (warped, M) where *warped* is a (output_size × output_size × 3) BGR
        image and *M* is the 3×3 perspective matrix.

    Raises:
        ValueError: If *frame* or *corners* are invalid.
    """
    if frame is None or frame.size == 0:
        raise ValueError("warp_board: frame is empty")
    if corners is None or corners.shape != (4, 2):
        raise ValueError(f"warp_board: corners must be (4,2), got {corners!r}")

    s = output_size
    dst = np.array([[0, 0], [s, 0], [s, s], [0, s]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(frame, M, (s, s))
    return warped, M


def localize_board(
    frame: np.ndarray,
    *,
    output_size: int = OUTPUT_SIZE,
    canny_low: int = _CANNY_LOW,
    canny_high: int = _CANNY_HIGH,
    verify_grid: bool = False,
) -> Optional[dict]:
    """Full board localization pipeline.

    Detects the board outer frame and returns the warped canonical image.

    Args:
        frame:       Input BGR camera frame.
        output_size: Canonical warp size (default 800).
        canny_low:   Lower Canny threshold.
        canny_high:  Upper Canny threshold.
        verify_grid: If True, check that warped image contains ~32 dark and
                     ~32 light cells; logs a warning if not.

    Returns:
        dict with keys:
            'warped'   – (output_size × output_size × 3) BGR canonical image.
            'corners'  – (4,2) float32 [TL,TR,BR,BL] in original frame coords.
            'matrix'   – 3×3 perspective transform matrix.
            'verified' – bool (only meaningful when verify_grid=True).
        Returns None if detection failed.
    """
    if frame is None or frame.size == 0:
        logger.warning("localize_board: empty frame")
        return None

    corners = detect_board_corners(frame, canny_low=canny_low, canny_high=canny_high)
    if corners is None:
        return None

    warped, M = warp_board(frame, corners, output_size=output_size)

    verified = False
    if verify_grid:
        verified = _verify_alternating_grid(warped, output_size)
        if not verified:
            logger.warning("localize_board: alternating grid verification failed")

    return {"warped": warped, "corners": corners, "matrix": M, "verified": verified}


def _verify_alternating_grid(warped: np.ndarray, output_size: int) -> bool:
    """Return True if warped image has roughly equal dark and light cells."""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    cell = output_size // 8
    dark = light = 0
    for row in range(8):
        for col in range(8):
            cx = col * cell + cell // 2
            cy = row * cell + cell // 2
            if gray[cy, cx] < 128:
                dark += 1
            else:
                light += 1
    # Allow ±8 tolerance (some boards have piece shadows).
    ok = abs(dark - light) <= 8
    logger.debug("Grid verify: dark=%d light=%d ok=%s", dark, light, ok)
    return ok


def cell_to_algebraic(col: int, row: int, flipped: bool = False) -> str:
    """Convert 0-indexed (col, row) in warped image to algebraic notation.

    Args:
        col:     Column index 0-7 (left to right in warped image).
        row:     Row index 0-7 (top to bottom in warped image).
        flipped: If True, board is oriented with Black at bottom (a1 at top-right).

    Returns:
        Algebraic square name e.g. 'a1', 'h8'.
    """
    if flipped:
        file_idx = 7 - col
        rank_idx = row
    else:
        file_idx = col
        rank_idx = 7 - row
    return chr(ord("a") + file_idx) + str(rank_idx + 1)


def pixel_to_square(
    x: float,
    y: float,
    output_size: int = OUTPUT_SIZE,
    flipped: bool = False,
) -> tuple[str, int, int]:
    """Map pixel (x, y) in warped image to algebraic square and cell indices.

    Returns:
        (algebraic, col, row) where col/row are 0-indexed grid coordinates.
    """
    cell = output_size / 8
    col = int(x // cell)
    row = int(y // cell)
    col = max(0, min(7, col))
    row = max(0, min(7, row))
    return cell_to_algebraic(col, row, flipped=flipped), col, row
