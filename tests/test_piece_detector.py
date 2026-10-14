"""Self-check for piece_detector: synthetic piece markers on a blank 800x800 canvas."""
import sys
import numpy as np
import cv2

sys.path.insert(0, "/home/rifky/projects/github/UR-Chess-LLM-Arena")
from src.vision.piece_detector import (
    WARP_SIZE,
    detect_pieces,
    build_board_state,
    draw_pieces,
    _DICT_ID,
)


def _make_warped_with_pieces(marker_specs: dict[int, tuple[int, int]]) -> np.ndarray:
    """Render piece ArUco markers at given (col, row) grid positions on 800x800 canvas.

    marker_specs: {marker_id: (grid_col, grid_row)} — 0-indexed.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(_DICT_ID)
    canvas = np.ones((WARP_SIZE, WARP_SIZE, 3), dtype=np.uint8) * 200
    cell = WARP_SIZE // 8
    msz = 60  # marker pixel size (< cell = 100 px)
    offset = (cell - msz) // 2  # centre marker in cell

    for mid, (col, row) in marker_specs.items():
        px = col * cell + offset
        py = row * cell + offset
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, mid, msz)
        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        canvas[py : py + msz, px : px + msz] = marker_bgr

    return canvas


def test_detect_pieces_basic():
    """Place markers 1 (white, a8 = col0/row0) and 17 (black, h1 = col7/row7)."""
    specs = {1: (0, 0), 17: (7, 7)}
    warped = _make_warped_with_pieces(specs)

    pieces = detect_pieces(warped)

    assert len(pieces) == 2, f"Expected 2 pieces, got {len(pieces)}: {pieces}"

    by_id = {p["id"]: p for p in pieces}
    assert 1 in by_id and 17 in by_id, f"Wrong IDs: {list(by_id.keys())}"

    assert by_id[1]["square"] == "a8", f"ID1 square wrong: {by_id[1]['square']}"
    assert by_id[17]["square"] == "h1", f"ID17 square wrong: {by_id[17]['square']}"


def test_ids_outside_range_ignored():
    """Corner marker IDs (e.g. 40) must be ignored by detect_pieces."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(_DICT_ID)
    canvas = np.ones((WARP_SIZE, WARP_SIZE, 3), dtype=np.uint8) * 200
    msz = 60
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, 40, msz)
    canvas[50:50+msz, 50:50+msz] = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)

    pieces = detect_pieces(canvas)
    assert len(pieces) == 0, f"ID 40 should be filtered, got: {pieces}"


def test_build_board_state():
    specs = {1: (0, 0), 17: (7, 7), 5: (3, 3)}
    warped = _make_warped_with_pieces(specs)
    pieces = detect_pieces(warped)
    state = build_board_state(pieces)

    assert state.get("a8") == "white", f"a8: {state.get('a8')}"
    assert state.get("h1") == "black", f"h1: {state.get('h1')}"
    assert state.get("d5") == "white", f"d5: {state.get('d5')}"


def test_draw_pieces_returns_same_shape():
    specs = {2: (1, 1)}
    warped = _make_warped_with_pieces(specs)
    pieces = detect_pieces(warped)
    vis = draw_pieces(warped, pieces)
    assert vis.shape == warped.shape, f"Shape mismatch: {vis.shape} vs {warped.shape}"


if __name__ == "__main__":
    test_detect_pieces_basic()
    print("test_detect_pieces_basic PASSED")
    test_ids_outside_range_ignored()
    print("test_ids_outside_range_ignored PASSED")
    test_build_board_state()
    print("test_build_board_state PASSED")
    test_draw_pieces_returns_same_shape()
    print("test_draw_pieces_returns_same_shape PASSED")
    print("ALL CHECKS PASSED")
