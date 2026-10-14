"""Unit tests for src/strategy/llm_client.py — no network calls needed."""

import chess
import pytest
from unittest.mock import MagicMock, patch

from src.strategy.llm_client import (
    _board_from_input,
    _build_prompt,
    _extract_uci,
    _fallback_move,
    get_best_move,
)


# ---------- _board_from_input ----------

def test_board_from_board_passthrough():
    b = chess.Board()
    assert _board_from_input(b) is b


def test_board_from_fen():
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    b = _board_from_input(fen)
    assert isinstance(b, chess.Board)
    assert b.fen().startswith("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b")


# ---------- _build_prompt ----------

def test_build_prompt_contains_fen_and_legal_moves():
    b = chess.Board()
    prompt = _build_prompt(b)
    assert b.fen() in prompt
    # all legal moves present
    for m in b.legal_moves:
        assert m.uci() in prompt
    assert "UCI" in prompt


# ---------- _extract_uci ----------

@pytest.mark.parametrize("text,expected", [
    ("e2e4", "e2e4"),
    ("  e2e4  ", "e2e4"),
    ("The best move is e2e4.", "e2e4"),
    ("g1f3!", "g1f3"),
    ("e7e8q", "e7e8q"),       # promotion
    ("E2E4", "e2e4"),          # uppercase normalized
    ("garbage text xyz", ""),  # no valid UCI token → empty
])
def test_extract_uci(text, expected):
    assert _extract_uci(text) == expected


# ---------- _fallback_move ----------

def test_fallback_returns_legal_move():
    b = chess.Board()
    m = _fallback_move(b)
    assert m in b.legal_moves


def test_fallback_prefers_capture():
    # Set up a position where White can capture on d5
    b = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2")
    m = _fallback_move(b)
    assert b.is_capture(m)


# ---------- get_best_move (mocked) ----------

def _make_mock_client(content: str):
    """Build a minimal mock OpenAI client that returns `content`."""
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    client = MagicMock()
    client.chat.completions.create.return_value = resp
    return client


@patch("src.strategy.llm_client.OpenAI")
@patch("src.strategy.llm_client._load_api_key", return_value="fake-key")
def test_valid_llm_move_accepted(mock_key, mock_openai_cls):
    b = chess.Board()
    mock_openai_cls.return_value = _make_mock_client("e2e4")
    move = get_best_move(b)
    assert move == chess.Move.from_uci("e2e4")
    assert move in b.legal_moves


@patch("src.strategy.llm_client.OpenAI")
@patch("src.strategy.llm_client._load_api_key", return_value="fake-key")
def test_illegal_llm_move_triggers_fallback(mock_key, mock_openai_cls):
    b = chess.Board()
    mock_openai_cls.return_value = _make_mock_client("e1e9")  # illegal
    move = get_best_move(b)
    assert move in b.legal_moves


@patch("src.strategy.llm_client.OpenAI")
@patch("src.strategy.llm_client._load_api_key", return_value="fake-key")
def test_invalid_uci_string_triggers_fallback(mock_key, mock_openai_cls):
    b = chess.Board()
    mock_openai_cls.return_value = _make_mock_client("sorry, I cannot")
    move = get_best_move(b)
    assert move in b.legal_moves


@patch("src.strategy.llm_client.OpenAI")
@patch("src.strategy.llm_client._load_api_key", return_value="fake-key")
def test_api_error_triggers_fallback(mock_key, mock_openai_cls):
    client = MagicMock()
    client.chat.completions.create.side_effect = Exception("connection timeout")
    mock_openai_cls.return_value = client
    b = chess.Board()
    move = get_best_move(b)
    assert move in b.legal_moves


@patch("src.strategy.llm_client.OpenAI")
@patch("src.strategy.llm_client._load_api_key", return_value="fake-key")
def test_fen_string_input_accepted(mock_key, mock_openai_cls):
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    mock_openai_cls.return_value = _make_mock_client("e7e5")
    move = get_best_move(fen)
    b = chess.Board(fen)
    assert move in b.legal_moves


@patch("src.strategy.llm_client.OpenAI")
@patch("src.strategy.llm_client._load_api_key", return_value="fake-key")
def test_promotion_move_accepted(mock_key, mock_openai_cls):
    # White pawn on e7, can promote to queen
    b = chess.Board("8/4P3/8/8/8/8/8/4K2k w - - 0 1")
    mock_openai_cls.return_value = _make_mock_client("e7e8q")
    move = get_best_move(b)
    assert move in b.legal_moves
    assert move.promotion == chess.QUEEN
