"""LLM strategy client — connects to 9Router (OpenAI-compat) to pick chess moves."""

import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Union

import chess
from openai import OpenAI, APIError, APITimeoutError

logger = logging.getLogger(__name__)

_AUTH_JSON = Path.home() / ".hermes" / "auth.json"
_MODEL = "ag/claude-sonnet-4-6"
_BASE_URL = "http://127.0.0.1:20128/v1"
_TIMEOUT = 30  # seconds


def _load_api_key() -> str:
    """Read bearer token from env var or ~/.hermes/auth.json (custom:9router entry)."""
    env_key = os.environ.get("NINE_ROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key

    try:
        data = json.loads(_AUTH_JSON.read_text())
        entries = data["credential_pool"]["custom:9router"]
        if entries:
            # secret field name is typically 'api_key' or 'secret'
            entry = entries[0]
            for field in ("api_key", "secret", "key", "token"):
                if field in entry:
                    return entry[field]
    except Exception as exc:  # noqa: BLE001
        logger.debug("auth.json read failed: %s", exc)

    raise RuntimeError(
        "No API key found. Set NINE_ROUTER_API_KEY env or add 9router creds to ~/.hermes/auth.json"
    )


def _board_from_input(board_or_fen: Union[chess.Board, str]) -> chess.Board:
    if isinstance(board_or_fen, chess.Board):
        return board_or_fen
    return chess.Board(board_or_fen)


def _build_prompt(board: chess.Board) -> str:
    legal = [m.uci() for m in board.legal_moves]
    return (
        f"You are a chess engine. Current position (FEN): {board.fen()}\n"
        f"Legal moves (UCI): {', '.join(legal)}\n"
        "Reply with exactly ONE UCI move from the legal list. No explanation, no punctuation, "
        "just the move (e.g. e2e4)."
    )


def _extract_uci(text: str) -> str:
    """Pull first UCI-like token from LLM response."""
    # UCI move: letter + digit + letter + digit [+ optional promotion]
    match = re.search(r"\b([a-h][1-8][a-h][1-8][qrbnQRBN]?)\b", text.strip(), re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return ""


def _fallback_move(board: chess.Board) -> chess.Move:
    """Return a deterministic-ish safe legal move (avoid random where possible)."""
    moves = list(board.legal_moves)
    # prefer captures or checks for a vaguely reasonable fallback
    for m in moves:
        if board.is_capture(m):
            return m
    return moves[0]


def get_best_move(
    board_or_fen: Union[chess.Board, str],
    model: str = _MODEL,
    base_url: str = _BASE_URL,
    api_key: str | None = None,
    timeout: float = _TIMEOUT,
) -> chess.Move:
    """Return best legal chess.Move chosen by LLM, with fallback guardrail.

    Args:
        board_or_fen: python-chess Board or FEN string.
        model: model identifier forwarded to 9Router.
        base_url: OpenAI-compat endpoint URL.
        api_key: override bearer token (auto-loaded from auth.json if None).
        timeout: request timeout in seconds.

    Returns:
        A legal chess.Move.
    """
    board = _board_from_input(board_or_fen)

    if not board.legal_moves.count():
        raise ValueError("No legal moves — game is over")

    key = api_key or _load_api_key()
    client = OpenAI(api_key=key, base_url=base_url, timeout=timeout)

    prompt = _build_prompt(board)
    logger.debug("LLM prompt:\n%s", prompt)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=16,
            temperature=0,
        )
        raw = response.choices[0].message.content or ""
        logger.debug("LLM raw response: %r", raw)
    except (APIError, APITimeoutError, Exception) as exc:  # noqa: BLE001
        logger.warning("LLM call failed (%s), using fallback move", exc)
        return _fallback_move(board)

    uci_str = _extract_uci(raw)
    try:
        move = chess.Move.from_uci(uci_str)
        if move in board.legal_moves:
            logger.info("LLM chose move: %s", uci_str)
            return move
        logger.warning("LLM returned illegal move %r, using fallback", uci_str)
    except (ValueError, chess.InvalidMoveError) as exc:
        logger.warning("UCI parse error %r (%s), using fallback", uci_str, exc)

    return _fallback_move(board)
