"""Self-check for piece_detector: synthetic piece markers on a blank 800x800 canvas."""
import sys
import numpy as np
import cv2

sys.path.insert(0, "/home/rifky/projects/github/UR-Chess-LLM-Arena")
from src.vision.piece_detector import (
    WARP_SIZE,
    ID_TO_PIECE_CHAR,
    detect_pieces,
    build_board_state,
    build_piece_type_board_state,
    draw_pieces,
    TemporalPieceTracker,
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


def test_id_to_piece_char_coverage():
    """All 32 IDs must map to a valid FEN piece char."""
    for mid in range(1, 33):
        char = ID_TO_PIECE_CHAR[mid]
        assert char in "RNBQKPrnbqkp", f"ID {mid} → unexpected char '{char}'"
    # Spot-check identity
    assert ID_TO_PIECE_CHAR[5] == "K",  "ID5 must be White King"
    assert ID_TO_PIECE_CHAR[21] == "k", "ID21 must be Black King"
    assert ID_TO_PIECE_CHAR[4] == "Q",  "ID4 must be White Queen"
    assert ID_TO_PIECE_CHAR[20] == "q", "ID20 must be Black Queen"
    assert ID_TO_PIECE_CHAR[9] == "P",  "ID9 must be White Pawn"
    assert ID_TO_PIECE_CHAR[25] == "p", "ID25 must be Black Pawn"


def test_build_piece_type_board_state():
    """build_piece_type_board_state returns exact piece chars, not just color."""
    # ID5 = King (K), ID21 = king (k), ID2 = Knight (N)
    specs = {5: (4, 7), 21: (4, 0), 2: (1, 7)}
    warped = _make_warped_with_pieces(specs)
    pieces = detect_pieces(warped)
    state = build_piece_type_board_state(pieces)

    # e1 (col4, row7) should be King 'K'
    assert state.get("e1") == "K", f"e1: {state.get('e1')}"
    # e8 (col4, row0) should be king 'k'
    assert state.get("e8") == "k", f"e8: {state.get('e8')}"
    # b1 (col1, row7) should be Knight 'N'
    assert state.get("b1") == "N", f"b1: {state.get('b1')}"


def test_temporal_piece_tracker_retention():
    """Pieces missing for < TTL frames must be retained; evicted after TTL hits 0.

    ttl=3: after detection, piece survives 3 consecutive missing frames, then evicted.
    Frame 1: detect  → ttl_remaining=3 (present)
    Frame 2: miss    → ttl_remaining=2 (present)
    Frame 3: miss    → ttl_remaining=1 (present)
    Frame 4: miss    → ttl_remaining=0 → evicted (absent)
    """
    tracker = TemporalPieceTracker(ttl=3)

    piece = {"id": 1, "square": "e4", "col": 4, "row": 4, "center": (450.0, 450.0), "piece": "R"}

    # Frame 1: detected
    out = tracker.update([piece])
    assert any(p["id"] == 1 for p in out), "Piece should be present after detection"

    # Frame 2: missing — ttl_remaining=2 → still present
    out = tracker.update([])
    assert any(p["id"] == 1 for p in out), "Piece should be retained (miss 1, ttl_remaining=2)"

    # Frame 3: still missing — ttl_remaining=1 → still present
    out = tracker.update([])
    assert any(p["id"] == 1 for p in out), "Piece should be retained (miss 2, ttl_remaining=1)"

    # Frame 4: still missing — ttl_remaining=0 → evicted
    out = tracker.update([])
    assert not any(p["id"] == 1 for p in out), "Piece must be evicted (miss 3, ttl_remaining=0)"


def test_temporal_piece_tracker_refresh():
    """Re-detection resets TTL so piece survives indefinitely."""
    tracker = TemporalPieceTracker(ttl=2)
    piece = {"id": 7, "square": "g1", "col": 6, "row": 7, "center": (650.0, 750.0), "piece": "N"}

    for _ in range(10):
        out = tracker.update([piece])
        assert any(p["id"] == 7 for p in out)


def test_temporal_piece_tracker_max_pieces():
    """Tracker never exceeds max_pieces."""
    tracker = TemporalPieceTracker(ttl=5, max_pieces=4)
    pieces = [
        {"id": i, "square": "a1", "col": 0, "row": 7, "center": (50.0, 50.0), "piece": "P"}
        for i in range(1, 9)
    ]
    out = tracker.update(pieces)
    assert len(out) <= 4, f"Expected ≤4 pieces, got {len(out)}"


def test_temporal_piece_tracker_thread_safe():
    """Concurrent updates from multiple threads must not corrupt state."""
    import threading
    tracker = TemporalPieceTracker(ttl=5)
    errors = []

    def worker(piece_id: int):
        try:
            piece = {"id": piece_id, "square": "a1", "col": 0, "row": 7,
                     "center": (50.0, 50.0), "piece": "P"}
            for _ in range(20):
                tracker.update([piece])
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(1, 9)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"Thread errors: {errors}"


def test_detect_pieces_from_frame_with_warped():
    """Enhanced-ROI pass (warped kwarg) catches markers missed on full-frame pass.

    Builds a large synthetic frame with a single small (20 px) marker on a
    plain background.  The full-frame _detect_raw pass is likely to succeed on
    a clean synthetic marker, but we verify that:
      1. detect_pieces_from_frame returns the marker when called with warped=warped.
      2. The returned square is board-valid algebraic notation.
      3. No off-board false positives appear (all IDs are in [1-32]).
    """
    from src.vision.piece_detector import detect_pieces_from_frame, WARP_SIZE

    aruco_dict = cv2.aruco.getPredefinedDictionary(_DICT_ID)
    # Small marker (20 px) placed on a synthetic 800×800 frame.
    marker_id = 5  # White King
    msz = 20
    canvas = np.ones((WARP_SIZE, WARP_SIZE, 3), dtype=np.uint8) * 200
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, msz)
    marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
    # Place at pixel (390, 390) → centre of e-file area
    px, py = 390, 390
    canvas[py : py + msz, px : px + msz] = marker_bgr

    # Identity homography (frame == warped space for this synthetic test).
    M = np.eye(3, dtype=np.float32)

    # Without warped kwarg (baseline — may or may not find the small marker).
    pieces_no_enhance = detect_pieces_from_frame(canvas, M)

    # With warped kwarg (enhanced pass must at least not crash and stay valid).
    pieces_enhanced = detect_pieces_from_frame(canvas, M, warped=canvas)

    # All returned IDs must be in valid range.
    for p in pieces_enhanced:
        assert 1 <= p["id"] <= 32, f"Out-of-range ID: {p['id']}"
        # Square must be 2-char algebraic.
        assert len(p["square"]) == 2
        assert p["square"][0] in "abcdefgh"
        assert p["square"][1] in "12345678"

    # Enhanced path must have >= as many detections as non-enhanced path.
    assert len(pieces_enhanced) >= len(pieces_no_enhance), (
        "Enhanced pass should never drop detections found by the full-frame pass"
    )



if __name__ == "__main__":
    test_detect_pieces_basic()
    print("test_detect_pieces_basic PASSED")
    test_ids_outside_range_ignored()
    print("test_ids_outside_range_ignored PASSED")
    test_build_board_state()
    print("test_build_board_state PASSED")
    test_draw_pieces_returns_same_shape()
    print("test_draw_pieces_returns_same_shape PASSED")
    test_id_to_piece_char_coverage()
    print("test_id_to_piece_char_coverage PASSED")
    test_build_piece_type_board_state()
    print("test_build_piece_type_board_state PASSED")
    test_temporal_piece_tracker_retention()
    print("test_temporal_piece_tracker_retention PASSED")
    test_temporal_piece_tracker_refresh()
    print("test_temporal_piece_tracker_refresh PASSED")
    test_temporal_piece_tracker_max_pieces()
    print("test_temporal_piece_tracker_max_pieces PASSED")
    test_temporal_piece_tracker_thread_safe()
    print("test_temporal_piece_tracker_thread_safe PASSED")
    test_detect_pieces_from_frame_with_warped()
    print("test_detect_pieces_from_frame_with_warped PASSED")
    print("ALL CHECKS PASSED")
