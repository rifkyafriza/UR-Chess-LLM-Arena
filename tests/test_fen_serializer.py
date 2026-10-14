"""Self-check for fen_serializer: FEN generation and SettleFilter."""
import sys

sys.path.insert(0, "/home/rifky/projects/github/UR-Chess-LLM-Arena")
from src.vision.fen_serializer import serialize_fen, SettleFilter


# ── FEN serialization ─────────────────────────────────────────────────────────

def test_empty_board():
    fen = serialize_fen({})
    placement = fen.split()[0]
    assert placement == "8/8/8/8/8/8/8/8", f"Empty board wrong: {placement!r}"


def test_start_position():
    """Standard chess starting position FEN."""
    state = {}
    # Back rank White (rank 1 → rank_idx 7)
    for fi, piece in enumerate("RNBQKBNR"):
        state[f"{FILES[fi]}1"] = piece
    for fi in range(8):
        state[f"{FILES[fi]}2"] = "P"
    # Back rank Black (rank 8 → rank_idx 0)
    for fi, piece in enumerate("rnbqkbnr"):
        state[f"{FILES[fi]}8"] = piece
    for fi in range(8):
        state[f"{FILES[fi]}7"] = "p"

    fen = serialize_fen(state, castling="KQkq")
    placement = fen.split()[0]
    assert placement == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR", (
        f"Start position wrong:\n  got:      {placement}\n"
        f"  expected: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    )


def test_single_piece():
    fen = serialize_fen({"e4": "P"})
    placement = fen.split()[0]
    # rank8..rank5 empty, rank4 has P at e (index 4), rank3..rank1 empty
    assert "4P3" in placement, f"e4 pawn not encoded correctly: {placement}"


def test_fen_fields():
    fen = serialize_fen({}, active_color="b", castling="Kq", en_passant="e3", halfmove=3, fullmove=7)
    parts = fen.split()
    assert parts[1] == "b"
    assert parts[2] == "Kq"
    assert parts[3] == "e3"
    assert parts[4] == "3"
    assert parts[5] == "7"


# ── SettleFilter ───────────────────────────────────────────────────────────────

FILES = "abcdefgh"


def test_settle_emits_at_15():
    filt = SettleFilter(required_frames=15)
    state = {"e4": "P"}
    result = None
    for i in range(15):
        result = filt.update(state)
    assert result == state, f"Expected emission at frame 15, got {result}"
    assert filt.count == 15


def test_settle_no_emit_before_15():
    filt = SettleFilter(required_frames=15)
    state = {"e4": "P"}
    for i in range(14):
        r = filt.update(state)
        assert r is None, f"Premature emit at frame {i+1}: {r}"


def test_settle_reset_on_change():
    filt = SettleFilter(required_frames=15)
    state_a = {"e4": "P"}
    state_b = {"d4": "P"}
    for _ in range(14):
        filt.update(state_a)
    filt.update(state_b)  # change — resets counter
    assert filt.count == 1, f"Counter not reset: {filt.count}"


def test_settle_no_duplicate_emit():
    filt = SettleFilter(required_frames=3)
    state = {"e4": "P"}
    emitted = []
    for _ in range(10):
        r = filt.update(state)
        if r is not None:
            emitted.append(r)
    assert len(emitted) == 1, f"Emitted {len(emitted)} times, expected 1"


def test_settle_reset_clears():
    filt = SettleFilter(required_frames=3)
    state = {"e4": "P"}
    for _ in range(3):
        filt.update(state)
    filt.reset()
    assert filt.count == 0
    emitted = []
    for _ in range(3):
        r = filt.update(state)
        if r is not None:
            emitted.append(r)
    assert len(emitted) == 1, "Should emit again after reset"


if __name__ == "__main__":
    test_empty_board()
    print("test_empty_board PASSED")
    test_start_position()
    print("test_start_position PASSED")
    test_single_piece()
    print("test_single_piece PASSED")
    test_fen_fields()
    print("test_fen_fields PASSED")
    test_settle_emits_at_15()
    print("test_settle_emits_at_15 PASSED")
    test_settle_no_emit_before_15()
    print("test_settle_no_emit_before_15 PASSED")
    test_settle_reset_on_change()
    print("test_settle_reset_on_change PASSED")
    test_settle_no_duplicate_emit()
    print("test_settle_no_duplicate_emit PASSED")
    test_settle_reset_clears()
    print("test_settle_reset_clears PASSED")
    print("ALL CHECKS PASSED")
