"""Unit tests for src/simulation/web_app.py.

Tests run without a live LLM — llm_client.get_best_move is patched.
No browser or real HTTP needed; Flask test client used throughout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import chess
import pytest

# Ensure project root on path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_client():
    """Return a fresh Flask test client with a reset game."""
    from src.simulation import web_app
    web_app._game.reset()
    web_app._undo_stack.clear()
    app = web_app.app
    app.config["TESTING"] = True
    return app.test_client(), web_app


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestInitialState:
    def test_initial_positions_count(self):
        from src.simulation.web_app import _build_initial_positions
        pos = _build_initial_positions()
        assert len(pos) == 32

    def test_white_ids_range(self):
        from src.simulation.web_app import _build_initial_positions
        pos = _build_initial_positions()
        white = {k for k in pos if 1 <= k <= 16}
        assert len(white) == 16

    def test_black_ids_range(self):
        from src.simulation.web_app import _build_initial_positions
        pos = _build_initial_positions()
        black = {k for k in pos if 17 <= k <= 32}
        assert len(black) == 16

    def test_positions_to_board_state(self):
        from src.simulation.web_app import _build_initial_positions, _positions_to_board_state
        pos = _build_initial_positions()
        state = _positions_to_board_state(pos)
        # Standard start: 32 occupied squares
        assert len(state) == 32
        assert state.get("e1") == "K"   # White king
        assert state.get("e8") == "k"   # Black king

    def test_board_state_to_fen_initial(self):
        from src.simulation.web_app import (
            _build_initial_positions, _positions_to_board_state, _board_state_to_fen
        )
        pos = _build_initial_positions()
        state = _positions_to_board_state(pos)
        fen = _board_state_to_fen(state)
        # Should start with standard piece placement
        placement = fen.split(" ")[0]
        assert "RNBQKBNR" in placement  # white back rank
        assert "rnbqkbnr" in placement  # black back rank


class TestFlaskRoutes:
    def test_index_returns_200(self):
        client, _ = _make_client()
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"UR Chess" in resp.data

    def test_api_state(self):
        client, _ = _make_client()
        resp = client.get("/api/state")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "fen" in data
        assert "svg" in data
        assert "turn" in data
        assert data["turn"] == "white"

    def test_api_reset(self):
        client, wapp = _make_client()
        # Dirty the state first
        wapp._game.move_history.append("e2e4")
        resp = client.post("/api/reset")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["move_history"] == []
        assert data["turn"] == "white"

    def test_api_state_svg_present(self):
        client, _ = _make_client()
        data = client.get("/api/state").get_json()
        assert "<svg" in data["svg"]


class TestMoveEndpoint:
    def _post_move(self, client, wapp, from_sq, to_sq, moved_id,
                   llm_response_move="e7e5"):
        """Helper: post a move with LLM patched."""
        positions = {str(k): v for k, v in wapp._game.positions.items()}
        positions[str(moved_id)] = to_sq

        llm_move = chess.Move.from_uci(llm_response_move)
        with patch("src.simulation.web_app.get_best_move", return_value=llm_move):
            resp = client.post(
                "/api/move",
                data=json.dumps({
                    "positions": positions,
                    "moved_id": moved_id,
                    "from_sq": from_sq,
                    "to_sq": to_sq,
                }),
                content_type="application/json",
            )
        return resp

    def test_legal_move_accepted(self):
        client, wapp = _make_client()
        resp = self._post_move(client, wapp, "e2", "e4", moved_id=13)
        assert resp.status_code == 200
        data = resp.get_json()
        assert "e2e4" in data["move_history"]

    def test_llm_counter_move_applied(self):
        client, wapp = _make_client()
        resp = self._post_move(client, wapp, "e2", "e4", moved_id=13, llm_response_move="e7e5")
        data = resp.get_json()
        assert "e7e5" in data["move_history"]

    def test_illegal_move_rejected(self):
        client, wapp = _make_client()
        # e2->e5 is illegal for a pawn from start
        positions = {str(k): v for k, v in wapp._game.positions.items()}
        positions["13"] = "e5"
        resp = client.post(
            "/api/move",
            data=json.dumps({
                "positions": positions,
                "moved_id": 13,
                "from_sq": "e2",
                "to_sq": "e5",
            }),
            content_type="application/json",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_missing_squares_rejected(self):
        client, wapp = _make_client()
        resp = client.post(
            "/api/move",
            data=json.dumps({"positions": {}, "moved_id": 13}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_move_history_grows(self):
        client, wapp = _make_client()
        self._post_move(client, wapp, "e2", "e4", moved_id=13, llm_response_move="e7e5")
        data = client.get("/api/state").get_json()
        # white move + black LLM move = 2 entries
        assert len(data["move_history"]) == 2

    def test_turn_alternates(self):
        client, wapp = _make_client()
        resp = self._post_move(client, wapp, "e2", "e4", moved_id=13, llm_response_move="e7e5")
        data = resp.get_json()
        # After white + black move, it's white's turn again
        assert data["turn"] == "white"


class TestUndo:
    def test_undo_after_move(self):
        client, wapp = _make_client()
        # Make a move first
        positions = {str(k): v for k, v in wapp._game.positions.items()}
        positions["13"] = "e4"
        llm_move = chess.Move.from_uci("e7e5")
        with patch("src.simulation.web_app.get_best_move", return_value=llm_move):
            client.post("/api/move", data=json.dumps({
                "positions": positions, "moved_id": 13,
                "from_sq": "e2", "to_sq": "e4",
            }), content_type="application/json")

        resp = client.post("/api/undo")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["turn"] == "white"

    def test_undo_empty_stack(self):
        client, wapp = _make_client()
        resp = client.post("/api/undo")
        data = resp.get_json()
        assert "error" in data


class TestGameState:
    def test_game_state_to_dict_keys(self):
        from src.simulation.web_app import GameState
        gs = GameState()
        d = gs.to_dict()
        for key in ("fen", "svg", "move_history", "turn", "reasoning",
                    "llm_move", "is_game_over", "outcome", "positions"):
            assert key in d, f"missing key: {key}"

    def test_svg_contains_svg_tag(self):
        from src.simulation.web_app import GameState
        gs = GameState()
        assert "<svg" in gs.svg()

    def test_fen_valid(self):
        from src.simulation.web_app import GameState
        gs = GameState()
        board = chess.Board(gs.fen())
        assert board is not None
