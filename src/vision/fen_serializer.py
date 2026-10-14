"""FEN serializer and human-turn settle filter.

Converts a square→piece dict into a standard FEN position string and
maintains a stability filter that emits a settled state only after
15 consecutive identical frames.

Piece notation convention (caller must supply; not inferred here):
    board_state: dict[square, piece_char]
    square: algebraic, e.g. 'e4'
    piece_char: FEN single character — uppercase White, lowercase Black
                e.g. 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', ...

If the upstream detector only knows 'white'/'black' (no piece type), the
caller must map to a piece char before calling serialize_fen().
"""

from __future__ import annotations

from collections import deque
from typing import Optional

# ── Constants ─────────────────────────────────────────────────────────────────

FILES = "abcdefgh"
RANKS = "87654321"  # rank 8 → row index 0

SETTLE_FRAMES: int = 15  # consecutive identical frames required


# ── FEN helpers ───────────────────────────────────────────────────────────────

def _square_to_indices(square: str) -> tuple[int, int]:
    """Return (rank_idx, file_idx) for an algebraic square.

    rank_idx 0 = rank 8 (top of board), 7 = rank 1 (bottom).
    file_idx 0 = a-file, 7 = h-file.
    """
    assert len(square) == 2, f"Bad square: {square!r}"
    file_idx = FILES.index(square[0])
    rank_idx = RANKS.index(square[1])
    return rank_idx, file_idx


def serialize_fen(
    board_state: dict[str, str],
    *,
    active_color: str = "w",
    castling: str = "-",
    en_passant: str = "-",
    halfmove: int = 0,
    fullmove: int = 1,
) -> str:
    """Convert a square→piece_char mapping to a FEN string.

    Only the piece placement field is derived from board_state.
    The remaining FEN fields are passed through unchanged (defaults
    give a generic 'white to move, no rights' state).

    Parameters
    ----------
    board_state:
        dict mapping algebraic square (e.g. 'e4') to FEN piece char
        (e.g. 'P' white pawn, 'q' black queen).  Empty squares omitted.
    active_color:
        'w' or 'b'.
    castling:
        FEN castling availability string, e.g. 'KQkq' or '-'.
    en_passant:
        En-passant target square or '-'.
    halfmove:
        Halfmove clock (for 50-move rule).
    fullmove:
        Fullmove number.

    Returns
    -------
    Complete FEN string, e.g.
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    """
    # Build 8x8 grid: grid[rank_idx][file_idx] = piece_char or ''
    grid: list[list[str]] = [[""] * 8 for _ in range(8)]
    for square, piece in board_state.items():
        ri, fi = _square_to_indices(square)
        grid[ri][fi] = piece

    ranks_fen: list[str] = []
    for row in grid:
        segment = ""
        empty = 0
        for cell in row:
            if cell:
                if empty:
                    segment += str(empty)
                    empty = 0
                segment += cell
            else:
                empty += 1
        if empty:
            segment += str(empty)
        ranks_fen.append(segment)

    placement = "/".join(ranks_fen)
    return f"{placement} {active_color} {castling} {en_passant} {halfmove} {fullmove}"


# ── Stability filter ───────────────────────────────────────────────────────────

class SettleFilter:
    """Emit a settled board state only after SETTLE_FRAMES identical frames.

    Usage
    -----
    filt = SettleFilter()
    for frame in camera_loop():
        state = detect_board(frame)          # dict[square, piece_char]
        settled = filt.update(state)
        if settled is not None:
            fen = serialize_fen(settled)
            handle_settled_position(fen)

    A new candidate resets the counter even if it differs by only one piece.
    """

    def __init__(self, required_frames: int = SETTLE_FRAMES) -> None:
        assert required_frames >= 1, "required_frames must be positive"
        self._required = required_frames
        self._candidate: Optional[dict[str, str]] = None
        self._count: int = 0
        self._last_emitted: Optional[dict[str, str]] = None

    # ponytail: no hysteresis on re-settle; add when spurious re-triggers observed
    def update(self, board_state: dict[str, str]) -> Optional[dict[str, str]]:
        """Feed one frame's board state.

        Returns the stable board state dict when the settle threshold is
        reached for the first time with this candidate, else None.
        """
        if board_state == self._candidate:
            self._count += 1
            if self._count >= self._required and board_state != self._last_emitted:
                self._last_emitted = dict(board_state)
                return dict(board_state)
        else:
            self._candidate = dict(board_state)
            self._count = 1
        return None

    def reset(self) -> None:
        """Clear filter state (call between games or after a move is handled)."""
        self._candidate = None
        self._count = 0
        self._last_emitted = None

    @property
    def count(self) -> int:
        """Frames seen for the current candidate."""
        return self._count

    @property
    def stable(self) -> bool:
        """True when candidate has been stable for at least required_frames."""
        return self._count >= self._required
