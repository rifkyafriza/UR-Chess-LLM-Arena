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
from typing import Any, Optional, Sequence

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


def _detect_folding_board_corners(
    contours: Sequence[Any],
    H: int,
    W: int,
    max_aspect_ratio: float = 1.4,
) -> Optional[np.ndarray]:
    """Detect folding chessboard where center crease splits border into two ~2:1 halves."""
    halves = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.005 * H * W or area > 0.15 * H * W:
            continue
        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        if rw <= 0 or rh <= 0:
            continue
        ar = max(rw, rh) / min(rw, rh)
        if 1.5 <= ar <= 2.4:
            halves.append((area, cnt, rect))

    for i in range(len(halves)):
        for j in range(i + 1, len(halves)):
            c1, c2 = halves[i][2][0], halves[j][2][0]
            dx = abs(c1[0] - c2[0])
            dy = abs(c1[1] - c2[1])
            if dx < 40 and 80 < dy < 200:
                merged = np.vstack([halves[i][1], halves[j][1]])
                rect_m = cv2.minAreaRect(merged)
                mw, mh = rect_m[1]
                if mw > 0 and mh > 0 and max(mw, mh) / min(mw, mh) <= max_aspect_ratio:
                    box = cv2.boxPoints(rect_m)
                    return _order_points(box.astype(np.float32))
    return None


def detect_board_corners(
    frame: np.ndarray,
    *,
    canny_low: int = _CANNY_LOW,
    canny_high: int = _CANNY_HIGH,
    min_area_ratio: float = 0.03,
    max_area_ratio: float = 0.45,
    max_aspect_ratio: float = 1.4,
) -> Optional[np.ndarray]:
    """Detect the four outer corners of the chess board in *frame*.

    Enforces strict aspect-ratio and area bounds so the detector does not
    falsely latch onto the full workbench/table edge or tiny noise contours.

    Args:
        frame:            BGR or grayscale image (H×W×3 or H×W).
        canny_low:        Lower Canny threshold.
        canny_high:       Upper Canny threshold.
        min_area_ratio:   Min contour area as fraction of frame area (default 0.03).
        max_area_ratio:   Max contour area as fraction of frame area (default 0.45).
        max_aspect_ratio: Max width/height ratio of bounding rect (default 1.4).

    Returns:
        Float32 (4,2) array of [TL, TR, BR, BL] pixel coordinates in *frame*,
        or None if detection failed.
    """
    if frame is None or frame.size == 0:
        logger.warning("detect_board_corners: empty frame")
        return None

    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)

    H, W = gray.shape[:2]
    frame_area = H * W
    min_area = min_area_ratio * frame_area
    max_area = max_area_ratio * frame_area

    # RETR_TREE catches inner framing contours when the outer border is broken.
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logger.debug("detect_board_corners: no contours found")
        return None

    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(cnt)
        if area > max_area:
            continue  # Whole table/workbench — skip.
        if area < min_area:
            break  # Remaining contours only get smaller.
        peri = cv2.arcLength(cnt, True)
        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        if rw <= 0 or rh <= 0:
            continue
        if max(rw, rh) / min(rw, rh) > max_aspect_ratio:
            continue  # Non-square (table edge, cable, etc.) — skip.

        # Try standard and slightly relaxed polygon approximation (for folding creases)
        for eps in (_POLY_EPSILON, 0.03, 0.035, 0.04):
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
                return _order_points(pts)

    # Secondary pass for folding chessboards (two 2:1 halves split by hinge crease)
    folding_corners = _detect_folding_board_corners(contours, H, W, max_aspect_ratio=max_aspect_ratio)
    if folding_corners is not None:
        return folding_corners

    # Fallback: morphological close to merge alternating dark squares.
    k_size = max(15, int(min(H, W) * 0.12))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    _, bin_dark = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    closed = cv2.morphologyEx(bin_dark, cv2.MORPH_CLOSE, kernel)

    closed_cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    morph_max_area = 0.80 * frame_area  # Looser bound for morphological blob (includes padding).
    for cnt in sorted(closed_cnts, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(cnt)
        if area < min_area or area > morph_max_area:
            continue
        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        if rw <= 0 or rh <= 0:
            continue
        if max(rw, rh) / min(rw, rh) > max_aspect_ratio:
            continue
        box = cv2.boxPoints(rect).astype(np.float32)
        ordered_box = _order_points(box)
        return _refine_grid_border(gray, ordered_box)

    logger.warning("detect_board_corners: no valid 4-vertex contour found")
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
    valid_corners = corners is not None

    if valid_corners and corners is not None:
        warped, M = warp_board(frame, corners, output_size=output_size)
        verified = False
        if verify_grid:
            verified = _verify_alternating_grid(warped, output_size)
            if not verified:
                logger.warning("localize_board: alternating grid verification failed")
        if not verify_grid or verified:
            return {"warped": warped, "corners": corners, "matrix": M, "verified": verified}

    # Marker-assisted fallback for boards with piece ArUco tags or non-ideal lighting/borders
    marker_res = _localize_from_markers(frame, output_size=output_size)
    if marker_res is not None:
        return marker_res

    if corners is not None:
        warped, M = warp_board(frame, corners, output_size=output_size)
        return {"warped": warped, "corners": corners, "matrix": M, "verified": False}

    return None


def _localize_from_markers(frame: np.ndarray, output_size: int = OUTPUT_SIZE) -> Optional[dict]:
    """Fallback: Estimate board perspective transform from piece ArUco markers."""
    try:
        from src.vision.piece_detector import _DETECTOR
    except ImportError:
        return None

    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners_list, ids, _ = _DETECTOR.detectMarkers(gray)
    if ids is None or len(ids) < 4:
        return None

    # Standard starting square mapping for piece IDs 1-32: (col 0-7, row 0-7)
    _START_SQUARES = {
        1: (0, 7), 2: (1, 7), 3: (2, 7), 4: (3, 7), 5: (4, 7), 6: (5, 7), 7: (6, 7), 8: (7, 7),
        9: (0, 6), 10: (1, 6), 11: (2, 6), 12: (3, 6), 13: (4, 6), 14: (5, 6), 15: (6, 6), 16: (7, 6),
        17: (0, 0), 18: (1, 0), 19: (2, 0), 20: (3, 0), 21: (4, 0), 22: (5, 0), 23: (6, 0), 24: (7, 0),
        25: (0, 1), 26: (1, 1), 27: (2, 1), 28: (3, 1), 29: (4, 1), 30: (5, 1), 31: (6, 1), 32: (7, 1),
    }

    cell = output_size / 8.0
    src_pts, dst_pts = [], []
    for marker_corners, mid in zip(corners_list, ids.flatten()):
        mid = int(mid)
        if mid not in _START_SQUARES:
            continue
        col, row = _START_SQUARES[mid]
        dst_x = (col + 0.5) * cell
        dst_y = (row + 0.5) * cell
        src_pts.append(marker_corners[0].mean(axis=0))
        dst_pts.append([dst_x, dst_y])

    if len(src_pts) < 4:
        return None

    M, _ = cv2.findHomography(
        np.array(src_pts, dtype=np.float32),
        np.array(dst_pts, dtype=np.float32),
        cv2.RANSAC,
    )
    if M is None:
        return None

    warped = cv2.warpPerspective(frame, M, (output_size, output_size))
    try:
        M_inv = np.linalg.inv(M)
        c_pts = np.array(
            [[0, 0], [output_size, 0], [output_size, output_size], [0, output_size]],
            dtype=np.float32,
        )
        corners_f = cv2.perspectiveTransform(c_pts.reshape(1, 4, 2), M_inv)[0]
    except Exception:
        corners_f = np.zeros((4, 2), dtype=np.float32)

    return {"warped": warped, "corners": corners_f, "matrix": M, "verified": True}


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
