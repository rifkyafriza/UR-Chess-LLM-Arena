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

Standard starting position piece identity (ID_TO_PIECE_CHAR):
    1-8   : R N B Q K B N R  (White major pieces, rank 1 a-h)
    9-16  : P P P P P P P P  (White pawns, rank 2 a-h)
    17-24 : r n b q k b n r  (Black major pieces, rank 8 a-h)
    25-32 : p p p p p p p p  (Black pawns, rank 7 a-h)
"""

from __future__ import annotations

import threading
import time
from collections import deque

import cv2
import numpy as np

from src.vision.board_locator_grid import pixel_to_square

# ── Constants ────────────────────────────────────────────────────────────────

WARP_SIZE: int = 800
PIECE_ID_MIN: int = 1
PIECE_ID_MAX: int = 32

_DICT_ID = cv2.aruco.DICT_4X4_50

# Piece character by marker ID (1-indexed, IDs 1-32).
# White side: uppercase; Black side: lowercase (FEN convention).
ID_TO_PIECE_CHAR: dict[int, str] = {
    # White major pieces (rank 1, files a-h)
    1: "R", 2: "N", 3: "B", 4: "Q", 5: "K", 6: "B", 7: "N", 8: "R",
    # White pawns (rank 2, files a-h)
    9: "P", 10: "P", 11: "P", 12: "P", 13: "P", 14: "P", 15: "P", 16: "P",
    # Black major pieces (rank 8, files a-h)
    17: "r", 18: "n", 19: "b", 20: "q", 21: "k", 22: "b", 23: "n", 24: "r",
    # Black pawns (rank 7, files a-h)
    25: "p", 26: "p", 27: "p", 28: "p", 29: "p", 30: "p", 31: "p", 32: "p",
}

# ── Optimized detector singleton ──────────────────────────────────────────────

_aruco_dict = cv2.aruco.getPredefinedDictionary(_DICT_ID)

def _build_params() -> cv2.aruco.DetectorParameters:
    p = cv2.aruco.DetectorParameters()
    # Adaptive threshold: wider window range catches markers at varied sizes
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 53
    p.adaptiveThreshWinSizeStep = 4
    p.adaptiveThreshConstant = 7
    # Subpixel corner refinement for sub-pixel accuracy
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.cornerRefinementWinSize = 5
    p.cornerRefinementMaxIterations = 30
    p.cornerRefinementMinAccuracy = 0.1
    # Relax min perimeter to detect small markers at distance
    p.minMarkerPerimeterRate = 0.02
    p.maxMarkerPerimeterRate = 4.0
    # Perspective remove for border check
    p.perspectiveRemovePixelPerCell = 8
    p.perspectiveRemoveIgnoredMarginPerCell = 0.13
    # Error correction
    p.maxErroneousBitsInBorderRate = 0.35
    p.minOtsuStdDev = 5.0
    p.errorCorrectionRate = 0.6
    return p

_params = _build_params()
_DETECTOR = cv2.aruco.ArucoDetector(_aruco_dict, _params)

# CLAHE for contrast enhancement on low-light / uneven frames
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _enhance_gray(gray: np.ndarray) -> np.ndarray:
    """Apply CLAHE contrast enhancement to improve marker detection."""
    return _CLAHE.apply(gray)


def _detect_raw(gray: np.ndarray):
    """Two-pass detection: raw gray then CLAHE-enhanced; merge results."""
    corners1, ids1, _ = _DETECTOR.detectMarkers(gray)
    corners2, ids2, _ = _DETECTOR.detectMarkers(_enhance_gray(gray))

    if ids1 is None and ids2 is None:
        return [], None

    # Merge, deduplicate by marker id (prefer first-pass result)
    seen: set[int] = set()
    merged_corners: list = []
    merged_ids: list[int] = []

    for src_corners, src_ids in (
        (corners1, ids1),
        (corners2, ids2),
    ):
        if src_ids is None:
            continue
        for c, mid in zip(src_corners, src_ids.flatten()):
            mid = int(mid)
            if mid not in seen:
                seen.add(mid)
                merged_corners.append(c)
                merged_ids.append(mid)

    if not merged_ids:
        return [], None
    return merged_corners, np.array(merged_ids).reshape(-1, 1)


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
        'piece'   – piece char from ID_TO_PIECE_CHAR (e.g. 'K', 'p')
    """
    assert warped is not None and warped.ndim == 3, (
        "warped must be a valid BGR image"
    )
    assert warped.shape[0] == WARP_SIZE and warped.shape[1] == WARP_SIZE, (
        f"Expected {WARP_SIZE}x{WARP_SIZE}, got {warped.shape[:2]}"
    )

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    corners_list, ids = _detect_raw(gray)

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
                "piece": ID_TO_PIECE_CHAR.get(mid, "P" if mid <= 16 else "p"),
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
    corners_list, ids = _detect_raw(gray)

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
                "piece": ID_TO_PIECE_CHAR.get(mid, "P" if mid <= 16 else "p"),
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


def build_piece_type_board_state(pieces: list[dict]) -> dict[str, str]:
    """Map detected pieces to a square→piece_char dict (FEN convention).

    Uses ID_TO_PIECE_CHAR to preserve exact piece identity (King, Queen,
    Rook, Bishop, Knight, Pawn) rather than degrading everything to pawns.

    Parameters
    ----------
    pieces:
        Output of :func:`detect_pieces` or :func:`detect_pieces_from_frame`.

    Returns
    -------
    Dict mapping algebraic square → piece char, e.g.:
        {'e1': 'K', 'e8': 'k', 'b1': 'N', 'a7': 'p', ...}
    Multiple detections on the same square keep the last-seen one.
    """
    state: dict[str, str] = {}
    for p in pieces:
        mid = p["id"]
        piece_char = ID_TO_PIECE_CHAR.get(mid, "P" if mid <= 16 else "p")
        state[p["square"]] = piece_char
    return state


# ── Temporal Piece Tracker ────────────────────────────────────────────────────

class TemporalPieceTracker:
    """Stabilizes piece detections over time to handle lighting flicker/occlusion.

    Maintains a TTL-based memory of piece positions. A piece missing from the
    current frame is retained for up to `ttl` frames before being dropped.
    Ensures steady detection up to the full 32/32 piece count.

    Thread-safe via an internal lock.

    Parameters
    ----------
    ttl:
        Number of frames a piece is retained after it disappears. Default 8.
    max_pieces:
        Upper bound on tracked pieces (32 for standard chess). Default 32.
    """

    def __init__(self, ttl: int = 8, max_pieces: int = 32) -> None:
        assert ttl > 0, "ttl must be positive"
        assert max_pieces > 0, "max_pieces must be positive"
        self._ttl = ttl
        self._max_pieces = max_pieces
        # {marker_id: {"piece_dict": dict, "ttl_remaining": int}}
        self._cache: dict[int, dict] = {}
        self._lock = threading.Lock()

    def update(self, detected_pieces: list[dict]) -> list[dict]:
        """Merge new detections with cached state; return stabilized piece list.

        Parameters
        ----------
        detected_pieces:
            Raw output of detect_pieces / detect_pieces_from_frame for this frame.

        Returns
        -------
        Stabilized list of piece dicts. Cached pieces missing this frame have
        their TTL decremented; pieces at TTL=0 are evicted.
        """
        with self._lock:
            current_ids = {p["id"] for p in detected_pieces}

            # Refresh TTL for detected pieces; insert new detections.
            for p in detected_pieces:
                mid = p["id"]
                self._cache[mid] = {"piece_dict": p, "ttl_remaining": self._ttl}

            # Decrement TTL for absent pieces; evict expired ones. A ghost whose
            # square is now occupied by a freshly detected piece is dropped at
            # once (else it could overwrite the live piece in the board state).
            fresh_squares = {p["square"] for p in detected_pieces}
            evict = [
                mid
                for mid, entry in self._cache.items()
                if mid not in current_ids
            ]
            for mid in evict:
                if self._cache[mid]["piece_dict"]["square"] in fresh_squares:
                    self._cache[mid]["ttl_remaining"] = 0
            for mid in evict:
                self._cache[mid]["ttl_remaining"] -= 1
                if self._cache[mid]["ttl_remaining"] <= 0:
                    del self._cache[mid]

            # Enforce max_pieces: remove oldest cached entries if over limit.
            if len(self._cache) > self._max_pieces:
                # Sort by ttl_remaining ascending; drop lowest-priority entries
                sorted_ids = sorted(
                    self._cache, key=lambda m: self._cache[m]["ttl_remaining"]
                )
                for mid in sorted_ids[: len(self._cache) - self._max_pieces]:
                    del self._cache[mid]

            return [entry["piece_dict"] for entry in self._cache.values()]

    def reset(self) -> None:
        """Clear all cached piece state."""
        with self._lock:
            self._cache.clear()

    @property
    def tracked_count(self) -> int:
        """Current number of tracked pieces (including TTL-retained ones)."""
        with self._lock:
            return len(self._cache)


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
        piece_char = p.get("piece", "?")
        label = f"{piece_char}:{p['square']}"
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
