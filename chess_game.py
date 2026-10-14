"""Terminal chess game with live HTML/SVG preview and LLM opponent support."""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
import webbrowser
from pathlib import Path

import chess
import chess.svg

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.strategy.llm_client import _MODEL, get_best_move


def llm_move(board: chess.Board, model_name: str) -> str:
    """Get best move from LLM strategy client and return UCI string."""
    move = get_best_move(board, model=model_name)
    return move.uci()


def render_gui(board: chess.Board, white_name: str, black_name: str) -> None:
    html = f"""
    <html>
    <head>
        <meta http-equiv="refresh" content="2">
        <style>
            body {{ display:flex; flex-direction:column; justify-content:center; align-items:center; background:#222; color:#fff; font-family:sans-serif; height:100vh; margin:0; }}
            .player {{ font-size: 24px; font-weight: bold; margin: 15px; padding: 10px 20px; background: #444; border-radius: 8px; }}
        </style>
    </head>
    <body>
        <div class="player">Black: {black_name}</div>
        {chess.svg.board(board, size=500)}
        <div class="player">White: {white_name}</div>
    </body>
    </html>
    """
    with open("ui_preview.html", "w", encoding="utf-8") as f:
        f.write(html)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for chess_game."""
    parser = argparse.ArgumentParser(description="AI Chess Arena (CLI Mode)")
    parser.add_argument(
        "--model",
        default=os.environ.get("CHESS_MODEL", _MODEL),
        help=f"LLM model name via 9Router (default: {os.environ.get('CHESS_MODEL', _MODEL)})",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    default_model = args.model

    print("=== AI Chess Arena ===")
    print("Select Game Mode:")
    print("1. Player vs Player")
    print("2. Player vs LLM")
    print("3. LLM vs LLM")

    try:
        mode_choice = input("Enter mode (1-3): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye.")
        return

    white_is_human = True
    black_is_human = True
    white_name = "Player 1"
    black_name = "Player 2"

    if mode_choice == "2":
        print("\nWho plays White?")
        print("1. Player")
        print("2. LLM")
        try:
            color_choice = input("Enter choice (1-2): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        if color_choice == "1":
            white_is_human = True
            black_is_human = False
            white_name = "Human"
            try:
                entered = input(f"Enter LLM model name for Black [{default_model}]: ").strip()
            except (EOFError, KeyboardInterrupt):
                entered = ""
            black_name = entered or default_model
        else:
            white_is_human = False
            black_is_human = True
            try:
                entered = input(f"Enter LLM model name for White [{default_model}]: ").strip()
            except (EOFError, KeyboardInterrupt):
                entered = ""
            white_name = entered or default_model
            black_name = "Human"

    elif mode_choice == "3":
        white_is_human = False
        black_is_human = False
        try:
            entered_w = input(f"Enter LLM model name for White [{default_model}]: ").strip()
            white_name = entered_w or default_model
            entered_b = input(f"Enter LLM model name for Black [{default_model}]: ").strip()
            black_name = entered_b or default_model
        except (EOFError, KeyboardInterrupt):
            white_name = default_model
            black_name = default_model

    board = chess.Board()
    render_gui(board, white_name, black_name)
    try:
        webbrowser.open("file://" + os.path.realpath("ui_preview.html"))
    except Exception:
        pass

    while not board.is_game_over():
        is_white_turn = (board.turn == chess.WHITE)
        current_is_human = white_is_human if is_white_turn else black_is_human
        current_name = white_name if is_white_turn else black_name

        turn_str = "White" if is_white_turn else "Black"
        print(f"\n{current_name}'s turn ({turn_str}).")

        if current_is_human:
            try:
                uci = input("Enter move (UCI format, e.g., e2e4): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGame aborted.")
                break
        else:
            print(f"{current_name} is thinking...")
            try:
                uci = llm_move(board, current_name)
                print(f"Move: {uci}")
            except Exception as e:
                print(f"Error: {e}. Forcing random move.")
                uci = random.choice(list(board.legal_moves)).uci()

        try:
            move = chess.Move.from_uci(uci)
            if move in board.legal_moves:
                board.push(move)
                render_gui(board, white_name, black_name)
            else:
                if not current_is_human:
                    print("Illegal move. Forcing random move to prevent freeze.")
                    board.push(random.choice(list(board.legal_moves)))
                    render_gui(board, white_name, black_name)
                else:
                    print("Illegal move. Try again.")
        except Exception:
            if current_is_human:
                print("Invalid format. Use UCI format like 'e2e4'. Try again.")
            else:
                print("Invalid LLM response. Forcing random move.")
                board.push(random.choice(list(board.legal_moves)))
                render_gui(board, white_name, black_name)

        if not current_is_human:
            time.sleep(1)

    print("\nGame Over:", board.result())


if __name__ == "__main__":
    main()
