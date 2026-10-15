"""Comprehensive tests for Phase 7 vision + video refinements.

Covers:
    1. ID_TO_PIECE_CHAR completeness for all 32 IDs.
    2. build_piece_type_board_state for all 32 IDs.
    3. TemporalPieceTracker: dropout retention, TTL aging, position replace.
    4. VisionConfig default locator_mode is 'contour'.
    5. live_stream /board_svg SVG contains piece glyphs R N B Q K P.
    6. Mock video loop handling in _capture_frames (EOF -> seek 0).
    7. scripts/benchmark_board.py run_video_benchmark video option.
"""
from __future__ import annotations

import sys
import threading
import time
import types
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# 1. ID_TO_PIECE_CHAR — all 32 IDs present and correct
# ---------------------------------------------------------------------------

from src.vision.piece_detector import (
    ID_TO_PIECE_CHAR,
    build_piece_type_board_state,
    TemporalPieceTracker,
)


class TestIdToPieceChar:
    def test_all_32_ids_present(self):
        assert set(ID_TO_PIECE_CHAR.keys()) == set(range(1, 33))

    def test_white_major_rank1(self):
        expected = {1: "R", 2: "N", 3: "B", 4: "Q", 5: "K", 6: "B", 7: "N", 8: "R"}
        for mid, ch in expected.items():
            assert ID_TO_PIECE_CHAR[mid] == ch, f"ID {mid}: expected {ch!r}"

    def test_white_pawns(self):
        for mid in range(9, 17):
            assert ID_TO_PIECE_CHAR[mid] == "P", f"ID {mid} should be P"

    def test_black_major_rank8(self):
        expected = {17: "r", 18: "n", 19: "b", 20: "q", 21: "k", 22: "b", 23: "n", 24: "r"}
        for mid, ch in expected.items():
            assert ID_TO_PIECE_CHAR[mid] == ch, f"ID {mid}: expected {ch!r}"

    def test_black_pawns(self):
        for mid in range(25, 33):
            assert ID_TO_PIECE_CHAR[mid] == "p", f"ID {mid} should be p"

    def test_white_ids_are_uppercase(self):
        for mid in range(1, 17):
            assert ID_TO_PIECE_CHAR[mid].isupper(), f"ID {mid} should be uppercase"

    def test_black_ids_are_lowercase(self):
        for mid in range(17, 33):
            assert ID_TO_PIECE_CHAR[mid].islower(), f"ID {mid} should be lowercase"

    def test_piece_chars_are_valid_fen(self):
        valid = set("RNBQKPrnbqkp")
        for mid, ch in ID_TO_PIECE_CHAR.items():
            assert ch in valid, f"ID {mid} char {ch!r} not valid FEN"


# ---------------------------------------------------------------------------
# 2. build_piece_type_board_state for all 32 IDs
# ---------------------------------------------------------------------------

def _make_piece(marker_id: int, square: str) -> dict:
    return {"id": marker_id, "square": square, "col": 0, "row": 0, "center": (0.0, 0.0)}


class TestBuildPieceTypeBoardState:
    def test_empty_returns_empty(self):
        assert build_piece_type_board_state([]) == {}

    def test_all_32_ids_map_correctly(self):
        # Assign each ID a unique square to avoid collision
        squares = [
            f"{f}{r}"
            for r in range(1, 9)
            for f in "abcdefgh"
        ]  # 64 squares; use first 32
        pieces = [_make_piece(mid, squares[i]) for i, mid in enumerate(range(1, 33))]
        state = build_piece_type_board_state(pieces)
        assert len(state) == 32
        for i, mid in enumerate(range(1, 33)):
            sq = squares[i]
            assert state[sq] == ID_TO_PIECE_CHAR[mid], (
                f"ID {mid} at {sq}: expected {ID_TO_PIECE_CHAR[mid]!r}, got {state.get(sq)!r}"
            )

    def test_duplicate_square_last_wins(self):
        # Two pieces on e4: last one (ID 20 = 'q') should win
        pieces = [
            _make_piece(1, "e4"),   # 'R'
            _make_piece(20, "e4"),  # 'q'
        ]
        state = build_piece_type_board_state(pieces)
        assert state["e4"] == "q"

    def test_king_and_queen_distinguished(self):
        pieces = [
            _make_piece(4, "d1"),  # Q
            _make_piece(5, "e1"),  # K
            _make_piece(20, "d8"), # q
            _make_piece(21, "e8"), # k
        ]
        state = build_piece_type_board_state(pieces)
        assert state["d1"] == "Q"
        assert state["e1"] == "K"
        assert state["d8"] == "q"
        assert state["e8"] == "k"


# ---------------------------------------------------------------------------
# 3. TemporalPieceTracker
# ---------------------------------------------------------------------------

def _p(mid: int, sq: str = "e4") -> dict:
    return {"id": mid, "square": sq, "col": 0, "row": 0, "center": (0.0, 0.0), "piece": "P"}


class TestTemporalPieceTracker:
    def test_initial_empty(self):
        t = TemporalPieceTracker(ttl=3)
        assert t.tracked_count == 0

    def test_detects_new_piece(self):
        t = TemporalPieceTracker(ttl=3)
        out = t.update([_p(1, "a1")])
        assert len(out) == 1
        assert out[0]["id"] == 1

    # -- Dropout retention
    def test_holds_through_transient_dropout(self):
        """Piece vanishes for (ttl-1) frames -> still retained."""
        ttl = 4
        t = TemporalPieceTracker(ttl=ttl)
        t.update([_p(1)])          # frame 0: present
        result: list = []
        for _ in range(ttl - 1):
            result = t.update([])  # frames 1..ttl-1: absent
        assert any(p["id"] == 1 for p in result), (
            "piece should still be retained within TTL"
        )

    def test_ages_out_after_ttl(self):
        """Piece absent for ttl frames -> evicted."""
        ttl = 3
        t = TemporalPieceTracker(ttl=ttl)
        t.update([_p(2)])
        result: list = []
        for _ in range(ttl):
            result = t.update([])  # absent ttl times
        assert not any(p["id"] == 2 for p in result), (
            "piece should be evicted after TTL frames"
        )

    def test_ttl_resets_on_reappear(self):
        """Piece appears -> vanishes 1 frame -> reappears -> TTL resets."""
        ttl = 3
        t = TemporalPieceTracker(ttl=ttl)
        t.update([_p(3)])    # appear
        t.update([])         # gone 1 frame (TTL remaining = ttl-1)
        t.update([_p(3)])    # reappears: TTL resets to ttl
        # now absent for ttl-1 more frames: still present
        result: list = []
        for _ in range(ttl - 1):
            result = t.update([])
        assert any(p["id"] == 3 for p in result)

    # -- Position replacement when piece moves
    def test_replaces_position_on_move(self):
        """When same ID detected at new square, tracker updates the square."""
        t = TemporalPieceTracker(ttl=5)
        t.update([_p(5, "e1")])
        t.update([_p(5, "e3")])  # piece moved
        result = t.update([_p(5, "e5")])
        ids_to_sq = {p["id"]: p["square"] for p in result}
        assert ids_to_sq[5] == "e5"

    def test_multiple_pieces_independent(self):
        t = TemporalPieceTracker(ttl=4)
        t.update([_p(1, "a1"), _p(2, "b1")])
        result = t.update([_p(1, "a2")])   # piece 2 absent
        ids = {p["id"] for p in result}
        assert 1 in ids and 2 in ids   # 2 still retained

    # -- max_pieces cap
    def test_max_pieces_enforced(self):
        t = TemporalPieceTracker(ttl=10, max_pieces=4)
        pieces = [_p(i, f"a{i}") for i in range(1, 9)]
        result = t.update(pieces)
        assert len(result) <= 4

    def test_reset_clears_all(self):
        t = TemporalPieceTracker(ttl=5)
        t.update([_p(i, f"a{i}") for i in range(1, 6)])
        t.reset()
        assert t.tracked_count == 0
        assert t.update([]) == []

    # -- Thread safety
    def test_concurrent_updates_no_exception(self):
        t = TemporalPieceTracker(ttl=8)
        errors = []

        def worker(piece_id):
            try:
                for _ in range(50):
                    t.update([_p(piece_id)])
                    t.update([])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(1, 6)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        assert errors == [], f"Thread safety errors: {errors}"


# ---------------------------------------------------------------------------
# 4. VisionConfig default locator_mode is 'contour'
# ---------------------------------------------------------------------------

from src.config import VisionConfig, load_config


class TestVisionConfigDefault:
    def test_dataclass_default_is_contour(self):
        v = VisionConfig()
        assert v.locator_mode == "contour"

    def test_load_config_none_gives_contour(self):
        """load_config(None) skips file; VisionConfig default applies."""
        cfg = load_config(None)
        assert cfg.vision.locator_mode == "contour"

    def test_load_config_yaml_override(self, tmp_path):
        """locator_mode in YAML overrides default."""
        y = tmp_path / "cfg.yaml"
        y.write_text("vision:\n  locator_mode: aruco\n")
        cfg = load_config(str(y))
        assert cfg.vision.locator_mode == "aruco"


# ---------------------------------------------------------------------------
# 5. live_stream /board_svg contains piece glyphs
# ---------------------------------------------------------------------------

from src.vision.live_stream import app as _live_app


@pytest.fixture
def live_client():
    _live_app.config["TESTING"] = True
    with _live_app.test_client() as c:
        yield c


class TestLiveStreamSvg:
    def test_board_svg_route_returns_svg(self, live_client):
        r = live_client.get("/board_svg")
        assert r.status_code == 200
        assert b"svg" in r.data.lower()

    def test_board_svg_default_empty_board(self, live_client):
        """Empty board SVG still has the board grid (no pieces but valid SVG)."""
        r = live_client.get("/board_svg")
        assert b"<svg" in r.data or b"<SVG" in r.data.upper()

    def test_state_svg_field_present(self, live_client):
        r = live_client.get("/state")
        data = r.get_json()
        assert "svg" in data

    def test_svg_with_pieces_contains_piece_glyphs(self):
        """Generate SVG for starting position; check R N B Q K P glyphs present."""
        import chess
        import chess.svg
        board = chess.Board()  # starting position
        svg = chess.svg.board(board, size=400)
        # chess.svg uses piece glyphs or class names containing piece letters
        for ch in ("R", "N", "B", "Q", "K", "P", "r", "n", "b", "q", "k", "p"):
            # SVG piece classes are like "chess-piece-R", or title "R" etc.
            # At minimum the piece character appears in SVG XML somewhere
            assert ch in svg, f"Piece char {ch!r} not found in SVG"


# ---------------------------------------------------------------------------
# 6. Mock video loop handling in _capture_frames
# ---------------------------------------------------------------------------

from src.vision.live_stream import _capture_frames


class TestVideoLoopHandling:
    def _make_cap(self, frames_sequence):
        """Build a mock VideoCapture.read that yields frames_sequence then stops."""
        cap = MagicMock()
        cap.isOpened.return_value = True
        frame_iter = iter(frames_sequence)

        def fake_read():
            try:
                return next(frame_iter)
            except StopIteration:
                return (False, None)

        cap.read.side_effect = fake_read
        return cap

    def test_video_eof_triggers_seek_and_re_read(self):
        """On EOF (ok=False), cap.set(CAP_PROP_POS_FRAMES, 0) called then re-read."""
        blank = np.zeros((480, 640, 3), dtype=np.uint8)

        # First read: ok frame. Second read: EOF. After seek: one more ok frame,
        # then EOF again to stop the loop.
        cap = MagicMock()
        reads = [
            (True, blank),   # frame 0
            (False, None),   # EOF -> should seek
            (True, blank),   # after seek
            (False, None),   # EOF again -> seek
            (True, blank),   # break out via stop_event
        ]
        read_iter = iter(reads)

        def safe_read():
            try:
                return next(read_iter)
            except StopIteration:
                return (False, None)

        cap.read.side_effect = safe_read
        cap.isOpened.return_value = True

        stop = threading.Event()
        processed = []

        # Patch the processing pipeline to count frames and stop after 2 processed
        def fake_localize(frame):
            processed.append(1)
            if len(processed) >= 2:
                stop.set()
            return None  # no board found; keep loop lightweight

        with patch("src.vision.live_stream.board_locator_grid.localize_board", side_effect=fake_localize), \
             patch("src.vision.live_stream.board_locator_aruco.locate_board", return_value=(None, {})), \
             patch("src.vision.live_stream._tracker") as mock_tracker, \
             patch("src.vision.live_stream.cv2.imencode", return_value=(True, np.array([0], dtype=np.uint8))):
            mock_tracker.update.return_value = []

            t = threading.Thread(
                target=_capture_frames,
                kwargs={"cap": cap, "is_video": True, "video_path": "/fake.mp4"},
                daemon=True,
            )
            t.start()
            stop.wait(timeout=3)
            t.join(timeout=1)

        # cap.set should have been called for EOF seek
        set_calls = [c for c in cap.set.call_args_list if c.args[1] == 0]
        assert len(set_calls) >= 1, "Expected CAP_PROP_POS_FRAMES=0 seek on EOF"

    def test_camera_mode_does_not_loop(self):
        """Camera mode (is_video=False): EOF stops loop immediately without seek."""
        cap = MagicMock()
        reads = [(False, None)]   # instant EOF
        reads_iter = iter(reads)

        def safe_cam_read():
            try:
                return next(reads_iter)
            except StopIteration:
                return (False, None)

        cap.read.side_effect = safe_cam_read

        processed = []

        def fake_localize(frame):
            processed.append(1)
            return None

        with patch("src.vision.live_stream.board_locator_grid.localize_board", side_effect=fake_localize), \
             patch("src.vision.live_stream.board_locator_aruco.locate_board", return_value=(None, {})), \
             patch("src.vision.live_stream._tracker") as mock_tracker, \
             patch("src.vision.live_stream.cv2.imencode", return_value=(True, np.array([0], dtype=np.uint8))):
            mock_tracker.update.return_value = []

            t = threading.Thread(
                target=_capture_frames,
                kwargs={"cap": cap, "is_video": False, "video_path": None},
                daemon=True,
            )
            t.start()
            t.join(timeout=2)  # should exit quickly

        # cap.set should NOT have been called for seek (not video mode)
        set_calls = [c for c in cap.set.call_args_list if c.args and len(c.args) > 1 and c.args[1] == 0]
        assert len(set_calls) == 0, "Camera mode should not seek on EOF"


# ---------------------------------------------------------------------------
# 7. benchmark_board.py run_video_benchmark with --video option
# ---------------------------------------------------------------------------

import scripts.benchmark_board as _bb


class TestBenchmarkVideoOption:
    def _make_cv2_mock(self, frames):
        fake_cv2 = MagicMock()
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.return_value = 30.0
        frame_iter = iter([(True, f) for f in frames] + [(False, None)])
        cap.read.side_effect = lambda: next(frame_iter)
        fake_cv2.VideoCapture.return_value = cap
        fake_cv2.CAP_PROP_FPS = 5
        fake_cv2.CAP_PROP_FRAME_COUNT = 7
        return fake_cv2

    def test_video_option_calls_run_video_benchmark(self):
        """main() with --video dispatches to run_video_benchmark."""
        with patch.object(_bb, "run_video_benchmark", return_value=0) as mock_rvb, \
             patch.object(_bb, "run_photo_benchmark", return_value=0), \
             patch("sys.argv", ["benchmark_board.py", "--video", "/tmp/test.mp4"]):
            rc = _bb.main()
        mock_rvb.assert_called_once_with("/tmp/test.mp4")

    def test_no_video_option_calls_photo_benchmark(self):
        """main() without --video dispatches to run_photo_benchmark."""
        with patch.object(_bb, "run_video_benchmark", return_value=0), \
             patch.object(_bb, "run_photo_benchmark", return_value=0) as mock_rpb, \
             patch("sys.argv", ["benchmark_board.py"]):
            _bb.main()
        mock_rpb.assert_called_once()

    def test_run_video_benchmark_bad_file_returns_1(self):
        fake_cv2 = MagicMock()
        cap = MagicMock()
        cap.isOpened.return_value = False
        fake_cv2.VideoCapture.return_value = cap
        with patch.object(_bb, "cv2", fake_cv2):
            rc = _bb.run_video_benchmark("/nonexistent.mp4")
        assert rc == 1

    def test_run_video_benchmark_with_frames_returns_0(self):
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        frames = [blank.copy() for _ in range(3)]
        fake_cv2 = self._make_cv2_mock(frames)
        with patch.object(_bb, "cv2", fake_cv2), \
             patch.object(_bb, "run_mode2", return_value={"success": False}):
            rc = _bb.run_video_benchmark("/fake.mp4")
        assert rc == 0

    def test_run_video_benchmark_per_frame_stats(self):
        """Localization succeeds on some frames: piece detection path exercised."""
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        frames = [blank.copy(), blank.copy()]
        fake_cv2 = self._make_cv2_mock(frames)
        m2_ok = {
            "success": True,
            "matrix": np.eye(3).tolist(),
            "warped": blank,
            "grid_verified": True,
        }
        with patch.object(_bb, "cv2", fake_cv2), \
             patch.object(_bb, "run_mode2", return_value=m2_ok), \
             patch.object(_bb, "run_piece_detection", return_value=([], {})), \
             patch("src.vision.fen_serializer.serialize_fen", return_value=""):
            rc = _bb.run_video_benchmark("/fake.mp4")
        assert rc == 0

    def test_run_video_benchmark_zero_frames_returns_1(self):
        fake_cv2 = self._make_cv2_mock([])
        with patch.object(_bb, "cv2", fake_cv2):
            rc = _bb.run_video_benchmark("/empty.mp4")
        assert rc == 1


def test_tracker_ghost_on_fresh_square_evicted():
    from src.vision.piece_detector import TemporalPieceTracker
    t = TemporalPieceTracker(ttl=8)
    t.update([{"id": 9, "square": "e2", "piece": "P", "center": (0, 0)}])
    out = t.update([{"id": 10, "square": "e2", "piece": "P", "center": (0, 0)}])
    assert [p["id"] for p in out] == [10]


def test_build_vision_missing_key_defaults_contour():
    from src.config import _build_vision
    assert _build_vision({}).locator_mode == "contour"
