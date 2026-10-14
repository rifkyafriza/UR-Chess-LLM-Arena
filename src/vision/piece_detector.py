"""ArUco piece detector for an 8x8 chess board.

Expects a canonical 800x800 px warped board image (output of either
board_locator_aruco.locate_board or board_locator_grid.localize_board).

Piece markers use IDs 1-32 from the DICT_4X4_50 dictionary.
Convention (customisable at call site via `white_ids` / `black_ids`):
    White pieces: IDs  1-16
    Black pieces: IDs 17-32

Square mapping (white-at-bottom, no flip):
    col 0-7  → files  a-h  (left → right)
    row 0-7  → ranks  8-1  (top  → bottom)
"""

from __future__ import annotations

import cv2
import numpy as np

from src.vision.board_locator_grid import pixel_to_square

# ── Constants ────────────────────────────────────────────────────────────────

WARP_SIZE: int = 800
PIECE_ID_MIN: int = 1
PIECE_ID_MAX: int = 32

_DICT_ID = cv2.aruco.DICT_4X4_50

# Module-level detector singleton.
_aruco_dict = cv2.aruco.getPredefinedDictionary(_DICT_ID)
_params = cv2.aruco.DetectorParameters()
_DETECTOR = cv2.aruco.ArucoDetector(_aruco_dict, _params)


# ── Core detection ────────────────────────────────────────────────────────────

def detect_pieces(
    warped: np.ndarray,
    *,
    flipped: bool = False,
    id_min: int = PIECE_ID_MIN,
    id_max: int = PIECE_ID_MAX,
) -> list[dict]:
    """Detect piece ArUco markers in a canonical warped board image.

    Parameters
    ----------
    warped:
        800x800 BGR image produced by a board locator.
    flipped:
        True when Black is at the bottom of the image (a1 at top-right).
    id_min, id_max:
        Inclusive ID range to accept as piece markers (default 1-32).

    Returns
    -------
    List of dicts, one per detected piece, each with:
        'id'      – int marker ID
        'square'  – algebraic square name, e.g. 'e4'
        'col'     – 0-indexed column (0=a, 7=h)
        'row'     – 0-indexed row (0=rank 8, 7=rank 1) in warped image
        'center'  – (x, y) float pixel centre in warped image
    """
    assert warped is not None and warped.ndim == 3, (
        "warped must be a valid BGR image"
    )
    assert warped.shape[0] == WARP_SIZE and warped.shape[1] == WARP_SIZE, (
        f"Expected {WARP_SIZE}x{WARP_SIZE}, got {warped.shape[:2]}"
    )

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    corners_list, ids, _ = _DETECTOR.detectMarkers(gray)

    pieces: list[dict] = []
    if ids is None or len(ids) == 0:
        return pieces

    for marker_corners, marker_id in zip(corners_list, ids.flatten()):
        mid = int(marker_id)
        if not (id_min <= mid <= id_max):
            continue
        # Center = mean of 4 corners; shape (1, 4, 2).
        cx, cy = marker_corners[0].mean(axis=0)
        square, col, row = pixel_to_square(cx, cy, output_size=WARP_SIZE, flipped=flipped)
        pieces.append(
            {
                "id": mid,
                "square": square,
                "col": col,
                "row": row,
                "center": (float(cx), float(cy)),
            }
        )

    return pieces


def detect_pieces_from_frame(
    frame: np.ndarray,
    matrix: np.ndarray,
    *,
    output_size: int = WARP_SIZE,
    flipped: bool = False,
    id_min: int = PIECE_ID_MIN,
    id_max: int = PIECE_ID_MAX,
) -> list[dict]:
    """Detect piece ArUco markers on the original full-resolution camera frame.

    Projects detected marker centers through perspective transform *matrix*
    to determine algebraic square coordinates. Preserves full resolution
    and avoids bilinear downsampling artifacts on dark squares.
    """
    assert frame is not None and frame.ndim == 3, "frame must be a valid BGR image"
    assert matrix is not None and matrix.shape == (3, 3), "matrix must be a 3x3 homography"

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners_list, ids, _ = _DETECTOR.detectMarkers(gray)

    pieces: list[dict] = []
    if ids is None or len(ids) == 0:
        return pieces

    for marker_corners, marker_id in zip(corners_list, ids.flatten()):
        mid = int(marker_id)
        if not (id_min <= mid <= id_max):
            continue
        cx_raw, cy_raw = marker_corners[0].mean(axis=0)
        pt = np.array([[[cx_raw, cy_raw]]], dtype=np.float32)
        wpt = cv2.perspectiveTransform(pt, matrix)[0][0]
        wx, wy = float(wpt[0]), float(wpt[1])
        square, col, row = pixel_to_square(wx, wy, output_size=output_size, flipped=flipped)
        pieces.append(
            {
                "id": mid,
                "square": square,
                "col": col,
                "row": row,
                "center": (wx, wy),
                "raw_center": (float(cx_raw), float(cy_raw)),
                "raw_corners": marker_corners[0].tolist(),
            }
        )

    return pieces


def build_board_state(
    pieces: list[dict],
    *,
    white_ids: range = range(1, 17),
    black_ids: range = range(17, 33),
) -> dict[str, str]:
    """Map detected pieces to a square→color dict.

    Parameters
    ----------
    pieces:
        Output of :func:`detect_pieces`.
    white_ids, black_ids:
        ID ranges for each side.

    Returns
    -------
    Dict mapping algebraic square → 'white' | 'black'.
    Multiple detections on the same square keep the last-seen one.
    """
    state: dict[str, str] = {}
    for p in pieces:
        if p["id"] in white_ids:
            state[p["square"]] = "white"
        elif p["id"] in black_ids:
            state[p["square"]] = "black"
    return state


# ── HUD overlay ──────────────────────────────────────────────────────────────

def draw_pieces(
    warped: np.ndarray,
    pieces: list[dict],
    *,
    white_ids: range = range(1, 17),
    black_ids: range = range(17, 33),
) -> np.ndarray:
    """Draw bounding dots + ID + square labels onto a copy of *warped*.

    White pieces → green circle; Black pieces → red circle.
    Returns a new BGR image.
    """
    assert warped is not None, "warped must not be None"
    vis = warped.copy()
    for p in pieces:
        cx, cy = int(p["center"][0]), int(p["center"][1])
        color = (0, 220, 0) if p["id"] in white_ids else (0, 0, 220)
        cv2.circle(vis, (cx, cy), 10, color, -1)
        label = f"{p['id']}:{p['square']}"
        cv2.putText(
            vis,
            label,
            (cx + 12, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return vis
