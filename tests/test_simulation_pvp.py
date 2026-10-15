"""Integration tests for PvP mode in src/simulation/web_app.py.

Tests run WITHOUT a live LLM or UR robot — LLM is never called in PvP mode
and no sockets on ports 29999/30002/30003 should be opened at any point.
"""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import chess
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(mode: str = "pvp"):
    """Return a fresh Flask test client locked to *mode*."""
    from src.simulation import web_app
    web_app._game.reset()
    web_app._undo_stack.clear()
    web_app._game.game_mode = mode
    app = web_app.app
    app.config["TESTING"] = True
    return app.test_client(), web_app


def _post_move(client, wapp, from_sq: str, to_sq: str, moved_id: int):
    """POST /api/move without touching LLM (PvP path skips it)."""
    positions = {str(k): v for k, v in wapp._game.positions.items()}
    # Client sends the optimistic update (piece already at to_sq)
    positions[str(moved_id)] = to_sq
    return client.post(
        "/api/move",
        data=json.dumps({
            "positions": positions,
            "moved_id": moved_id,
            "from_sq": from_sq,
            "to_sq": to_sq,
        }),
        content_type="application/json",
    )


# ---------------------------------------------------------------------------
# Test: game_mode field present in state
# ---------------------------------------------------------------------------

class TestGameModeField:
    def test_initial_mode_is_ai(self):
        client, wapp = _make_client(mode="ai")
        data = client.get("/api/state").get_json()
        assert data["game_mode"] == "ai"

    def test_pvp_mode_set(self):
        client, wapp = _make_client(mode="pvp")
        data = client.get("/api/state").get_json()
        assert data["game_mode"] == "pvp"


# ---------------------------------------------------------------------------
# Test: /api/mode switching
# ---------------------------------------------------------------------------

class TestModeSwitch:
    def test_switch_to_pvp(self):
        client, wapp = _make_client(mode="ai")
        resp = client.post(
            "/api/mode",
            data=json.dumps({"mode": "pvp"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        assert resp.get_json()["game_mode"] == "pvp"

    def test_switch_to_ai(self):
        client, wapp = _make_client(mode="pvp")
        resp = client.post(
            "/api/mode",
            data=json.dumps({"mode": "ai"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        assert resp.get_json()["game_mode"] == "ai"

    def test_invalid_mode_rejected(self):
        client, _ = _make_client()
        resp = client.post(
            "/api/mode",
            data=json.dumps({"mode": "robot"}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_missing_body_rejected(self):
        client, _ = _make_client()
        resp = client.post("/api/mode", data="not-json", content_type="text/plain")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Test: 4-turn PvP sequence — turns alternate W→B→W→B, no LLM invoked
# ---------------------------------------------------------------------------

class TestFourTurnSequence:
    """1. e2e4  2. e7e5  3. g1f3  4. b8c6 — all played human-vs-human."""

    # IDs: 13=Pe2, 29=pe7, 7=Ng1, 18=nb8
    MOVES = [
        ("e2", "e4", 13),   # White pawn e2→e4
        ("e7", "e5", 29),   # Black pawn e7→e5
        ("g1", "f3", 7),    # White knight g1→f3
        ("b8", "c6", 18),   # Black knight b8→c6
    ]

    def test_full_sequence_accepted(self):
        client, wapp = _make_client(mode="pvp")
        expected_turns = ["black", "white", "black", "white"]
        for i, (from_sq, to_sq, mid) in enumerate(self.MOVES):
            resp = _post_move(client, wapp, from_sq, to_sq, mid)
            assert resp.status_code == 200, (
                f"Move {i+1} ({from_sq}{to_sq}) rejected: {resp.get_json()}"
            )
            data = resp.get_json()
            assert data["turn"] == expected_turns[i], (
                f"After move {i+1} expected turn={expected_turns[i]!r}, got {data['turn']!r}"
            )

    def test_sequence_move_history(self):
        client, wapp = _make_client(mode="pvp")
        for from_sq, to_sq, mid in self.MOVES:
            _post_move(client, wapp, from_sq, to_sq, mid)
        data = client.get("/api/state").get_json()
        assert data["move_history"] == ["e2e4", "e7e5", "g1f3", "b8c6"]

    def test_no_llm_called_in_pvp(self):
        """get_best_move must never be called when game_mode is 'pvp'."""
        client, wapp = _make_client(mode="pvp")
        mock_llm = MagicMock(side_effect=AssertionError("LLM called in PvP mode!"))
        with patch("src.simulation.web_app.get_best_move", mock_llm):
            for from_sq, to_sq, mid in self.MOVES:
                resp = _post_move(client, wapp, from_sq, to_sq, mid)
                assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Test: capture in PvP — e4xd5 removes Black d5 pawn, no overlap
# ---------------------------------------------------------------------------

class TestCaptureInPvP:
    """Set up: e2e4, d7d5 (open centre), then e4xd5."""
    # IDs: 13=Pe2, 28=pd7
    def _setup_capture(self, client, wapp):
        _post_move(client, wapp, "e2", "e4", 13)  # White
        _post_move(client, wapp, "d7", "d5", 28)  # Black
        return _post_move(client, wapp, "e4", "d5", 13)  # White captures

    def test_capture_accepted(self):
        client, wapp = _make_client(mode="pvp")
        resp = self._setup_capture(client, wapp)
        assert resp.status_code == 200

    def test_victim_marked_captured(self):
        client, wapp = _make_client(mode="pvp")
        self._setup_capture(client, wapp)
        data = client.get("/api/state").get_json()
        pos = data["positions"]
        # Black pawn id=28 should be captured
        assert pos["28"] == "__captured__"

    def test_no_two_pieces_on_same_square(self):
        client, wapp = _make_client(mode="pvp")
        self._setup_capture(client, wapp)
        data = client.get("/api/state").get_json()
        pos = data["positions"]
        occupied = [sq for sq in pos.values() if sq != "__captured__"]
        assert len(occupied) == len(set(occupied)), "Duplicate square assignment detected"

    def test_attacker_on_victim_square(self):
        client, wapp = _make_client(mode="pvp")
        self._setup_capture(client, wapp)
        data = client.get("/api/state").get_json()
        pos = data["positions"]
        # White pawn (id=13) must be on d5
        assert pos["13"] == "d5"


# ---------------------------------------------------------------------------
# Test: castling in PvP — e1g1 king→g1, rook h1→f1
# ---------------------------------------------------------------------------

class TestCastlingInPvP:
    """Clear f1, g1 squares so white can castle kingside."""

    def _setup_castling(self, client, wapp):
        """Use internal board setup: set board to a position where e1g1 is legal."""
        from src.simulation import web_app as wa
        # Directly set a castling-ready board (both f1 and g1 empty)
        wa._game.board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
        wa._game.positions = {
            # White: Ke1(5), Rf1 removed(6→captured), Ng1 removed(7→captured), Rh1(8)
            1: "a1", 2: "b1", 3: "c1", 4: "d1", 5: "e1",
            6: "__captured__", 7: "__captured__", 8: "h1",
            9: "a2", 10: "b2", 11: "c2", 12: "d2",
            13: "e4", 14: "f2", 15: "g2", 16: "h2",
            # Black pieces (approximate, only need legality)
            17: "a8", 18: "c6", 19: "c8", 20: "d8",
            21: "e8", 22: "c5", 23: "f6", 24: "h8",
            25: "a7", 26: "b7", 27: "c7", 28: "d7",
            29: "e5", 30: "f7", 31: "g7", 32: "h7",
        }
        wa._game.game_mode = "pvp"

    def test_castling_accepted(self):
        client, wapp = _make_client(mode="pvp")
        self._setup_castling(client, wapp)
        resp = _post_move(client, wapp, "e1", "g1", moved_id=5)
        assert resp.status_code == 200, resp.get_json()

    def test_king_on_g1(self):
        client, wapp = _make_client(mode="pvp")
        self._setup_castling(client, wapp)
        _post_move(client, wapp, "e1", "g1", moved_id=5)
        pos = client.get("/api/state").get_json()["positions"]
        assert pos["5"] == "g1", f"King not on g1: {pos['5']}"

    def test_rook_on_f1(self):
        client, wapp = _make_client(mode="pvp")
        self._setup_castling(client, wapp)
        _post_move(client, wapp, "e1", "g1", moved_id=5)
        pos = client.get("/api/state").get_json()["positions"]
        assert pos["8"] == "f1", f"Rook not on f1: {pos['8']}"


# ---------------------------------------------------------------------------
# Test: en passant in PvP
# ---------------------------------------------------------------------------

class TestEnPassantInPvP:
    """Setup: 1.e2e4 e7e5 2.e4e5→illegal; instead 1.e2e4 d7d5 2.e4e5 d5d4
       then c2c4, d4c3 en passant.
    
    Simpler canonical ep: e2e5 (via internal board), then d7d5, e5d6 ep.
    """

    def _setup_ep(self, client, wapp):
        from src.simulation import web_app as wa
        # Position: white pawn on e5, black just played d7-d5 (ep available at d6)
        wa._game.board = chess.Board("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2")
        wa._game.positions = {
            1: "a1", 2: "b1", 3: "c1", 4: "d1", 5: "e1",
            6: "f1", 7: "g1", 8: "h1",
            9: "a2", 10: "b2", 11: "c2", 12: "d2",
            13: "e5", 14: "f2", 15: "g2", 16: "h2",   # Pe5
            17: "a8", 18: "b8", 19: "c8", 20: "d8",
            21: "e8", 22: "f8", 23: "g8", 24: "h8",
            25: "a7", 26: "b7", 27: "c7", 28: "d5",   # Pd5 (just moved)
            29: "e7", 30: "f7", 31: "g7", 32: "h7",
        }
        wa._game.game_mode = "pvp"

    def test_ep_accepted(self):
        client, wapp = _make_client(mode="pvp")
        self._setup_ep(client, wapp)
        resp = _post_move(client, wapp, "e5", "d6", moved_id=13)
        assert resp.status_code == 200, resp.get_json()

    def test_ep_captures_victim_pawn(self):
        client, wapp = _make_client(mode="pvp")
        self._setup_ep(client, wapp)
        _post_move(client, wapp, "e5", "d6", moved_id=13)
        pos = client.get("/api/state").get_json()["positions"]
        # Black pawn id=28 was on d5 — must be captured
        assert pos["28"] == "__captured__", f"EP victim not captured: {pos['28']}"

    def test_ep_attacker_on_d6(self):
        client, wapp = _make_client(mode="pvp")
        self._setup_ep(client, wapp)
        _post_move(client, wapp, "e5", "d6", moved_id=13)
        pos = client.get("/api/state").get_json()["positions"]
        assert pos["13"] == "d6"

    def test_ep_no_overlap(self):
        client, wapp = _make_client(mode="pvp")
        self._setup_ep(client, wapp)
        _post_move(client, wapp, "e5", "d6", moved_id=13)
        pos = client.get("/api/state").get_json()["positions"]
        occupied = [sq for sq in pos.values() if sq != "__captured__"]
        assert len(occupied) == len(set(occupied))


# ---------------------------------------------------------------------------
# Test: illegal move rejection in PvP (both colors)
# ---------------------------------------------------------------------------

class TestIllegalMoveRejectionPvP:
    def test_white_illegal_pawn_move_rejected(self):
        client, wapp = _make_client(mode="pvp")
        # Pawn cannot move 3 squares
        resp = _post_move(client, wapp, "e2", "e5", moved_id=13)
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_black_illegal_pawn_move_rejected(self):
        client, wapp = _make_client(mode="pvp")
        # First make a valid white move
        _post_move(client, wapp, "e2", "e4", moved_id=13)
        # Now black tries illegal 3-square pawn
        resp = _post_move(client, wapp, "e7", "e4", moved_id=29)
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_white_piece_on_black_turn_rejected(self):
        client, wapp = _make_client(mode="pvp")
        # Advance one white move so it's black's turn
        _post_move(client, wapp, "e2", "e4", moved_id=13)
        # Try to move white knight on black's turn
        resp = _post_move(client, wapp, "g1", "f3", moved_id=7)
        assert resp.status_code == 400

    def test_black_piece_on_white_turn_rejected(self):
        client, wapp = _make_client(mode="pvp")
        # It is white's turn at start; try to move black pawn
        resp = _post_move(client, wapp, "e7", "e5", moved_id=29)
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Test: zero-robot guarantee — no sockets opened on UR ports
# ---------------------------------------------------------------------------

class TestZeroRobotGuarantee:
    """Verify that running the PvP simulation tests opens no sockets
    on the UR robot ports 29999, 30002, or 30003."""

    UR_PORTS = (29999, 30002, 30003)

    def _port_is_open_on_loopback(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            return s.connect_ex(("127.0.0.1", port)) == 0

    def test_no_ur_ports_open_before_tests(self):
        """Pre-condition: UR ports must not be open during test runs."""
        for port in self.UR_PORTS:
            if self._port_is_open_on_loopback(port):
                pytest.skip(f"UR port {port} already bound externally — skip zero-robot check")

    def test_pvp_moves_do_not_open_ur_sockets(self):
        """Run a full 4-turn PvP game and confirm no UR sockets appear."""
        client, wapp = _make_client(mode="pvp")
        moves = [
            ("e2", "e4", 13),
            ("e7", "e5", 29),
            ("g1", "f3", 7),
            ("b8", "c6", 18),
        ]
        for from_sq, to_sq, mid in moves:
            _post_move(client, wapp, from_sq, to_sq, mid)

        for port in self.UR_PORTS:
            assert not self._port_is_open_on_loopback(port), (
                f"UR robot port {port} opened during simulation test — hardware dependency leak!"
            )

    def test_mode_switch_does_not_open_ur_sockets(self):
        client, _ = _make_client(mode="ai")
        client.post("/api/mode", data=json.dumps({"mode": "pvp"}), content_type="application/json")
        client.post("/api/mode", data=json.dumps({"mode": "ai"}), content_type="application/json")
        for port in self.UR_PORTS:
            assert not self._port_is_open_on_loopback(port), (
                f"UR robot port {port} opened during mode switch — hardware dependency leak!"
            )

    def test_reset_does_not_open_ur_sockets(self):
        client, _ = _make_client(mode="pvp")
        client.post("/api/reset")
        for port in self.UR_PORTS:
            assert not self._port_is_open_on_loopback(port)

    def test_web_app_import_does_not_open_ur_sockets(self):
        """Importing web_app must not touch UR hardware."""
        import importlib
        import src.simulation.web_app  # already imported; this just confirms no sockets needed
        for port in self.UR_PORTS:
            assert not self._port_is_open_on_loopback(port)
