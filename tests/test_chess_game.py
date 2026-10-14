"""Unit tests for chess_game.py CLI argument parsing and LLM integration."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import chess
import pytest

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chess_game
from src.strategy.llm_client import _MODEL


class TestParser:
    def test_default_model(self, monkeypatch):
        monkeypatch.delenv("CHESS_MODEL", raising=False)
        p = chess_game.build_parser()
        args = p.parse_args([])
        assert args.model == _MODEL

    def test_env_model_default(self, monkeypatch):
        monkeypatch.setenv("CHESS_MODEL", "custom-env-model")
        p = chess_game.build_parser()
        args = p.parse_args([])
        assert args.model == "custom-env-model"

    def test_cli_model_override(self):
        p = chess_game.build_parser()
        args = p.parse_args(["--model", "gpt-4o"])
        assert args.model == "gpt-4o"


class TestLlmMove:
    def test_llm_move_invokes_get_best_move(self):
        board = chess.Board()
        mock_move = chess.Move.from_uci("e2e4")
        with patch("chess_game.get_best_move", return_value=mock_move) as mock_gbm:
            uci = chess_game.llm_move(board, "custom-model")
            mock_gbm.assert_called_once_with(board, model="custom-model")
            assert uci == "e2e4"


class TestSanity:
    def test_no_placeholder_keys_in_file(self):
        src_path = ROOT / "chess_game.py"
        src = src_path.read_text(encoding="utf-8")
        assert "YOUR_API_KEY_HERE" not in src
        assert "Bearer" not in src
