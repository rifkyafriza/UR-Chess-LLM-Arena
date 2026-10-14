"""
Mock camera image feeder for headless tests.

Generates synthetic BGR frames (800x800) with an 8x8 chessboard grid.
Each of the 64 squares can optionally carry an ArUco marker (DICT_4X4_50).
Squares are labeled a1..h8; ArUco ID = file*8 + rank (0-63).

Usage:
    cam = MockCamera(piece_squares=["e2", "e4"])  # squares that get a marker
    frame = cam.get_frame()          # np.ndarray (800,800,3) BGR
    ok, frame = cam.read()           # cv2.VideoCapture-compatible
"""

import numpy as np
import cv2

# Board layout
_FILES = "abcdefgh"
_RANKS = "12345678"
_BOARD_SIZE = 800
_SQUARE_PX = _BOARD_SIZE // 8   # 100 px per square
_MARKER_PX = int(_SQUARE_PX * 0.7)  # marker fits inside square

_ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)


def _square_to_rc(sq: str):
    """'e2' -> (row, col) in image coords (row 0 = rank 8 = top of image)."""
    f = _FILES.index(sq[0])      # col 0-7
    r = int(sq[1]) - 1           # rank 0-7 (0=rank1 bottom, 7=rank8 top)
    row = 7 - r                  # flip: rank 8 = row 0
    return row, f


def _square_id(sq: str) -> int:
    """'a1' -> 0, 'h8' -> 63."""
    f = _FILES.index(sq[0])
    r = int(sq[1]) - 1
    return r * 8 + f


def _draw_board(frame: np.ndarray):
    """Draw alternating light/dark squares."""
    for row in range(8):
        for col in range(8):
            light = (row + col) % 2 == 0
            color = (240, 217, 181) if light else (181, 136, 99)  # cream / brown
            y0, x0 = row * _SQUARE_PX, col * _SQUARE_PX
            frame[y0:y0 + _SQUARE_PX, x0:x0 + _SQUARE_PX] = color


def _draw_markers(frame: np.ndarray, piece_squares: list):
    """Stamp ArUco markers onto the given squares."""
    for sq in piece_squares:
        if len(sq) != 2 or sq[0] not in _FILES or sq[1] not in _RANKS:
            continue
        row, col = _square_to_rc(sq)
        marker_id = _square_id(sq)
        marker_img = cv2.aruco.generateImageMarker(_ARUCO_DICT, marker_id, _MARKER_PX)
        # Convert grayscale marker to BGR
        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        # Center marker in square
        y0 = row * _SQUARE_PX + (_SQUARE_PX - _MARKER_PX) // 2
        x0 = col * _SQUARE_PX + (_SQUARE_PX - _MARKER_PX) // 2
        frame[y0:y0 + _MARKER_PX, x0:x0 + _MARKER_PX] = marker_bgr


class MockCamera:
    """
    Synthetic camera feeder.

    Parameters
    ----------
    piece_squares : list[str]
        Chess squares (e.g. ["a1", "h8"]) that will contain ArUco markers.
        Marker ID = file_index * 8 + rank_index (0-63), using DICT_4X4_50.
    noise_sigma : float
        Gaussian noise stddev added per frame (0 = clean). Default 2.
    """

    def __init__(self, piece_squares: list | None = None, noise_sigma: float = 2.0):
        self.piece_squares = piece_squares or []
        self.noise_sigma = noise_sigma
        self._base_frame: np.ndarray | None = None
        self._build_base()

    def _build_base(self):
        frame = np.zeros((_BOARD_SIZE, _BOARD_SIZE, 3), dtype=np.uint8)
        _draw_board(frame)
        _draw_markers(frame, self.piece_squares)
        self._base_frame = frame

    def update_pieces(self, piece_squares: list):
        """Rebuild base frame with new piece layout."""
        self.piece_squares = piece_squares
        self._build_base()

    def get_frame(self) -> np.ndarray:
        """Return a fresh BGR frame (with noise)."""
        assert self._base_frame is not None
        frame = self._base_frame.copy()
        if self.noise_sigma > 0:
            noise = np.random.normal(0, self.noise_sigma, frame.shape).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return frame

    def read(self) -> tuple[bool, np.ndarray]:
        """cv2.VideoCapture-compatible read()."""
        return True, self.get_frame()

    @property
    def BOARD_SIZE(self) -> int:
        return _BOARD_SIZE

    @property
    def SQUARE_PX(self) -> int:
        return _SQUARE_PX

    @property
    def ARUCO_DICT(self):
        return _ARUCO_DICT
