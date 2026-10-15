"""Phase 8 field vision pipeline tests: board ROI, enhancement, small markers, FP rejection, tracker."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vision.board_locator_grid import detect_board_corners, localize_board
from src.vision.piece_detector import (
    TemporalPieceTracker,
    build_piece_type_board_state,
    detect_pieces_from_frame,
    enhance_roi,
    _DETECTOR,
)

H, W = 720, 1280
X0, Y0, S = 400, 120, 480
CELL = S // 8


def make_scene(on_board=None, off_board=None, marker_px=30, noise=False):
    """Synthetic overhead frame: table, framed checkerboard, ArUco markers."""
    f = np.full((H, W, 3), (120, 150, 170), np.uint8)
    cv2.rectangle(f, (X0 - 25, Y0 - 25), (X0 + S + 25, Y0 + S + 25), (40, 60, 90), -1)
    for r in range(8):
        for q in range(8):
            v = 230 if (r + q) % 2 == 0 else 60
            f[Y0 + r * CELL:Y0 + (r + 1) * CELL, X0 + q * CELL:X0 + (q + 1) * CELL] = v
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    def put(mid, px, py):
        m = cv2.cvtColor(cv2.aruco.generateImageMarker(d, mid, marker_px), cv2.COLOR_GRAY2BGR)
        f[py:py + marker_px, px:px + marker_px] = m

    off = (CELL - marker_px) // 2
    for mid, (q, r) in (on_board or {}).items():
        put(mid, X0 + q * CELL + off, Y0 + r * CELL + off)
    for mid, (px, py) in (off_board or {}).items():
        put(mid, px, py)
    if noise:
        rng = np.random.default_rng(0)
        f = np.clip(f.astype(np.int16) + rng.normal(0, 6, f.shape), 0, 255).astype(np.uint8)
    return f


def _pieces(frame):
    res = localize_board(frame)
    assert res is not None, "board must localize"
    return res, detect_pieces_from_frame(frame, res["matrix"], warped=res["warped"])


# 1. Board extraction -------------------------------------------------------

def test_board_corners_found_with_correct_geometry():
    c = detect_board_corners(make_scene())
    assert c is not None and c.shape == (4, 2)
    w = c[1][0] - c[0][0]
    h = c[2][1] - c[1][1]
    assert 0.9 < w / h < 1.1
    assert w < 0.5 * W, "must be the board, not the table"


def test_table_perimeter_rejected():
    """Frame whose only big rectangle is the table (>45% area) must not yield it."""
    f = np.full((H, W, 3), 30, np.uint8)
    cv2.rectangle(f, (40, 30), (W - 40, H - 30), (200, 200, 200), -1)
    c = detect_board_corners(f)
    if c is not None:
        area = cv2.contourArea(c.astype(np.float32))
        assert area <= 0.45 * H * W, f"table perimeter accepted (area={area})"


def test_elongated_rect_rejected_by_aspect_ratio():
    f = np.full((H, W, 3), 30, np.uint8)
    cv2.rectangle(f, (100, 250), (1100, 400), (220, 220, 220), -1)  # ~6.7:1
    c = detect_board_corners(f, min_area_ratio=0.03)
    if c is not None:
        w = np.linalg.norm(c[1] - c[0])
        h = np.linalg.norm(c[3] - c[0])
        assert max(w, h) / min(w, h) <= 1.4


def test_empty_frame_returns_none():
    assert detect_board_corners(np.zeros((0, 0, 3), np.uint8)) is None


# 2. ROI enhancement --------------------------------------------------------

def _wood(size=400):
    rng = np.random.default_rng(1)
    g = rng.integers(90, 140, (size, size), dtype=np.uint8)
    return cv2.cvtColor(cv2.GaussianBlur(g, (0, 0), 1.0), cv2.COLOR_GRAY2BGR)


def _quad_candidates(gray):
    edges = cv2.Canny(gray, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return len(cnts)


def test_enhance_roi_shape_and_gray():
    out = enhance_roi(make_scene()[Y0:Y0 + S, X0:X0 + S], 2.0)
    assert out.ndim == 2 and out.shape == (2 * S, 2 * S)


def test_enhance_roi_fast():
    roi = np.zeros((800, 800, 3), np.uint8)
    enhance_roi(roi)
    t = time.perf_counter()
    for _ in range(10):
        enhance_roi(roi)
    ms = (time.perf_counter() - t) / 10 * 1000
    assert ms < 15, f"enhance_roi {ms:.1f} ms/frame"


def test_unsharp_beats_laplacian_on_texture_noise():
    tex = _wood()
    raw = cv2.cvtColor(cv2.resize(tex, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY)
    lap = cv2.filter2D(raw, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32))
    uns = enhance_roi(tex, 2.0)
    n_raw, n_lap, n_uns = map(_quad_candidates, (raw, lap, uns))
    assert n_uns <= n_lap, f"unsharp ({n_uns}) noisier than laplacian ({n_lap})"
    assert n_uns <= 1.5 * n_raw + 5


def test_unsharp_increases_edge_contrast():
    roi = cv2.GaussianBlur(make_scene()[Y0:Y0 + S, X0:X0 + S], (0, 0), 2.0)
    raw = cv2.cvtColor(cv2.resize(roi, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY)
    uns = enhance_roi(roi, 2.0)
    assert cv2.Laplacian(uns, cv2.CV_64F).var() > cv2.Laplacian(raw, cv2.CV_64F).var()


# 3. Small marker detection -------------------------------------------------

def test_detector_params_tuned_for_small_markers():
    p = _DETECTOR.getDetectorParameters()
    assert p.minMarkerPerimeterRate <= 0.005
    assert p.adaptiveThreshConstant <= 5
    assert p.cornerRefinementMethod == cv2.aruco.CORNER_REFINE_SUBPIX


@pytest.mark.parametrize("px", [24, 30, 40])
def test_small_markers_0_and_17_style_ids(px):
    """IDs 1 (white rook) and 17 (black rook) at small pixel sizes."""
    _, pieces = _pieces(make_scene({1: (0, 0), 17: (7, 7)}, marker_px=px))
    by = {p["id"]: p["square"] for p in pieces}
    assert by.get(1) == "a8" and by.get(17) == "h1", by


def test_marker_id_0_is_not_a_piece():
    """ID 0 is outside the 1-32 piece range and must be dropped."""
    _, pieces = _pieces(make_scene({1: (0, 0), 0: (3, 3)}))
    assert 0 not in {p["id"] for p in pieces}


# 4. Off-board elimination --------------------------------------------------

def test_offboard_id22_eliminated():
    frame = make_scene({1: (0, 0), 17: (7, 7)}, off_board={22: (100, 100)})
    _, pieces = _pieces(frame)
    ids = {p["id"] for p in pieces}
    assert 22 not in ids, "off-board ID 22 leaked (clamped onto edge square)"
    assert ids == {1, 17}


def test_offboard_marker_far_right_eliminated():
    frame = make_scene({17: (7, 7)}, off_board={5: (1100, 600)})
    _, pieces = _pieces(frame)
    assert {p["id"] for p in pieces} == {17}


def test_offboard_does_not_pollute_board_state():
    frame = make_scene({1: (0, 0)}, off_board={22: (100, 100)})
    _, pieces = _pieces(frame)
    state = build_piece_type_board_state(pieces)
    assert state.get("a8") == "R" and len(state) == 1


# 5. Tracker + benchmark integration ----------------------------------------

def test_tracker_bridges_occlusion_and_expires():
    tr = TemporalPieceTracker(ttl=3)
    _, p = _pieces(make_scene({1: (0, 0), 17: (7, 7)}))
    assert {x["id"] for x in tr.update(p)} == {1, 17}
    hand = [x for x in p if x["id"] == 17]  # ID 1 occluded
    assert 1 in {x["id"] for x in tr.update(hand)}
    tr.update(hand)
    assert 1 not in {x["id"] for x in tr.update(hand)}, "ttl must expire"
    assert 17 in {x["id"] for x in tr.update(hand)}


def test_tracker_never_retains_offboard_id():
    tr = TemporalPieceTracker(ttl=5)
    _, p = _pieces(make_scene({1: (0, 0)}, off_board={22: (100, 100)}))
    assert 22 not in {x["id"] for x in tr.update(p)}


def test_sparse_benchmark_on_images_dir(tmp_path, capsys):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from scripts.benchmark_board import run_sparse_benchmark
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"f{i}.png"),
                    make_scene({1: (0, 0), 17: (7, 7)}, off_board={22: (100, 100)}))
    assert run_sparse_benchmark(str(tmp_path), is_images_dir=True) == 0
    out = capsys.readouterr().out
    assert "Tracking Continuity Rate    : 100.0%" in out
    assert "Active Board Localization   : 100.0%" in out
    assert "False Positive Elimination" in out


def test_sparse_benchmark_empty_dir_fails(tmp_path):
    from scripts.benchmark_board import run_sparse_benchmark
    assert run_sparse_benchmark(str(tmp_path), is_images_dir=True) == 1


def test_live_stream_uses_tracker():
    import src.vision.live_stream as ls
    assert isinstance(ls._tracker, TemporalPieceTracker)
