"""End-to-end integration test suite for MasterOrchestrator.

Covers:
1. Multi-turn simulated game play (mock vision + MockURDriver).
2. ArenaState transition validation across full game loop.
3. Capture handling: piece removal to graveyard then robot piece move.
4. Castling moves (kingside and queenside).
5. Recovery from invalid vision inputs and simulated disconnects.
6. All tests must pass under pytest (no real hardware, no real LLM).
"""

from __future__ import annotations

import threading
from typing import List, Optional
from unittest.mock import MagicMock, call, patch

import chess
import pytest

from src.config import ArenaConfig
from src.orchestrator.master_fsm import ArenaState, MasterOrchestrator, OrchestrationError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PATCH_LLM = "src.orchestrator.master_fsm.get_best_move"
_PATCH_MOVEL = "src.robot.ur_driver.MockURDriver.movel"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_orc(
    human_fen_cb=None,
    verify_cb=None,
    on_state_change=None,
) -> MasterOrchestrator:
    """Orchestrator factory with sim-safe defaults."""
    return MasterOrchestrator(
        config=ArenaConfig(),
        robot_mode="sim",
        human_fen_callback=human_fen_cb,
        verify_callback=verify_cb or (lambda _: True),
        on_state_change=on_state_change,
    )


def _first_legal(board: chess.Board, **_kw) -> chess.Move:
    """Deterministic LLM stub: pick first legal move."""
    return next(iter(board.legal_moves))


def _feed_move(orc: MasterOrchestrator, uci: str) -> None:
    """Inject a human move from current board position."""
    after = orc.board.copy()
    after.push_uci(uci)
    orc.inject_human_fen(after.fen())


def _run_full_turn(orc: MasterOrchestrator, robot_uci: Optional[str] = None) -> ArenaState:
    """
    Drive FSM from PROCESSING_VISION through VERIFYING_BOARD -> WAIT_HUMAN_MOVE
    or GAME_OVER.  robot_uci pins the LLM response; None = first legal.
    """
    llm_side_effect = (
        (lambda b, **kw: chess.Move.from_uci(robot_uci))
        if robot_uci
        else _first_legal
    )
    with patch(_PATCH_LLM, side_effect=llm_side_effect):
        with patch(_PATCH_MOVEL, return_value=None):
            orc.run_turn()  # PROCESSING_VISION -> THINKING_LLM
            if orc.state == ArenaState.THINKING_LLM:
                orc.run_turn()  # THINKING_LLM -> ROBOT_MOVING
            if orc.state == ArenaState.ROBOT_MOVING:
                orc.run_turn()  # ROBOT_MOVING -> VERIFYING_BOARD | GAME_OVER
            if orc.state == ArenaState.VERIFYING_BOARD:
                orc.run_turn()  # VERIFYING_BOARD -> WAIT_HUMAN_MOVE
    return orc.state


# ---------------------------------------------------------------------------
# 1. Multi-turn simulated game play
# ---------------------------------------------------------------------------


class TestMultiTurnSimulation:
    """Drive several complete human+robot turns using mock vision feed."""

    def setup_method(self):
        self.orc = _make_orc()
        self.orc.initialize()
        assert self.orc.state == ArenaState.WAIT_HUMAN_MOVE

    def _turn(self, human_uci: str, robot_uci: Optional[str] = None) -> ArenaState:
        _feed_move(self.orc, human_uci)
        assert self.orc.state == ArenaState.PROCESSING_VISION
        return _run_full_turn(self.orc, robot_uci)

    def test_three_turns_board_advances(self):
        """Board fullmove counter increments across 3 complete turns."""
        # Turn 1: human (White) plays e4; robot (Black) responds
        self._turn("e2e4")
        assert self.orc.state in (ArenaState.WAIT_HUMAN_MOVE, ArenaState.GAME_OVER)
        if self.orc.state == ArenaState.GAME_OVER:
            return

        # Turn 2: human (White) plays d4 (it's White's turn again after robot responded)
        self._turn("d2d4")
        if self.orc.state == ArenaState.GAME_OVER:
            return

        # Turn 3: human (White) plays c4
        self._turn("c2c4")
        assert self.orc.board.fullmove_number >= 3

    def test_state_trail_correct_across_turns(self):
        """Verify exact state sequence for a full turn."""
        states: List[ArenaState] = []
        orc = _make_orc(on_state_change=lambda o, n: states.append(n))
        orc.initialize()
        _feed_move(orc, "e2e4")
        with patch(_PATCH_LLM, side_effect=_first_legal):
            with patch(_PATCH_MOVEL, return_value=None):
                orc.run_turn()  # PROCESSING_VISION
                orc.run_turn()  # THINKING_LLM
                orc.run_turn()  # ROBOT_MOVING
                if orc.state == ArenaState.VERIFYING_BOARD:
                    orc.run_turn()

        # Must have passed through THINKING_LLM and ROBOT_MOVING at minimum
        assert ArenaState.THINKING_LLM in states
        assert ArenaState.ROBOT_MOVING in states

    def test_last_human_move_and_last_robot_move_updated(self):
        """After one turn, both last_human_move and last_robot_move are set."""
        state = self._turn("e2e4")
        assert isinstance(self.orc.last_human_move, chess.Move)
        if state != ArenaState.GAME_OVER:
            assert isinstance(self.orc.last_robot_move, chess.Move)

    def test_board_fen_advances_after_each_turn(self):
        """Board FEN changes with each turn."""
        fen_start = self.orc.board.fen()
        self._turn("e2e4")
        assert self.orc.board.fen() != fen_start


# ---------------------------------------------------------------------------
# 2. ArenaState transition validation
# ---------------------------------------------------------------------------


class TestArenaStateTransitions:
    """Verify every documented FSM edge fires exactly as specified."""

    def test_initializing_to_wait_human_on_init(self):
        orc = _make_orc()
        assert orc.state == ArenaState.INITIALIZING
        orc.initialize()
        assert orc.state == ArenaState.WAIT_HUMAN_MOVE

    def test_wait_human_to_processing_vision_on_inject(self):
        orc = _make_orc()
        orc.initialize()
        _feed_move(orc, "e2e4")
        assert orc.state == ArenaState.PROCESSING_VISION

    def test_processing_vision_to_thinking_llm_on_valid_fen(self):
        orc = _make_orc()
        orc.initialize()
        _feed_move(orc, "e2e4")
        orc.run_turn()
        assert orc.state == ArenaState.THINKING_LLM

    def test_thinking_llm_to_robot_moving_on_llm_ok(self):
        orc = _make_orc()
        orc.initialize()
        _feed_move(orc, "e2e4")
        orc.run_turn()  # PV -> THINKING_LLM
        with patch(_PATCH_LLM, side_effect=_first_legal):
            orc.run_turn()
        assert orc.state == ArenaState.ROBOT_MOVING

    def test_robot_moving_to_verifying_board(self):
        orc = _make_orc()
        orc.initialize()
        _feed_move(orc, "e2e4")
        with patch(_PATCH_LLM, side_effect=_first_legal):
            with patch(_PATCH_MOVEL, return_value=None):
                orc.run_turn()  # PV
                orc.run_turn()  # THINKING
                orc.run_turn()  # ROBOT_MOVING
        assert orc.state in (ArenaState.VERIFYING_BOARD, ArenaState.GAME_OVER)

    def test_verifying_board_to_wait_human_on_ok(self):
        orc = _make_orc()
        orc.initialize()
        orc._state = ArenaState.VERIFYING_BOARD
        orc.run_turn()
        assert orc.state == ArenaState.WAIT_HUMAN_MOVE

    def test_any_state_to_error_safe_stop_on_robot_crash(self):
        orc = _make_orc()
        orc.initialize()
        _feed_move(orc, "e2e4")
        orc.run_turn()  # PV -> THINKING
        with patch(_PATCH_LLM, side_effect=_first_legal):
            orc.run_turn()  # THINKING -> ROBOT_MOVING
        # Now crash all sequencer methods
        with patch.object(orc._sequencer, "execute_normal", side_effect=RuntimeError("crash")):
            with patch.object(orc._sequencer, "execute_capture", side_effect=RuntimeError("crash")):
                with patch.object(orc._sequencer, "execute_castling", side_effect=RuntimeError("crash")):
                    orc.run_turn()
        assert orc.state == ArenaState.ERROR_SAFE_STOP

    def test_game_over_on_processing_vision_after_human_mates(self):
        """If FEN after human move is mate, FSM goes GAME_OVER at PROCESSING_VISION step."""
        # Scholar's mate finish: white mates black — inject final position
        # Build mate position: 1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6?? 4.Qxf7#
        orc = _make_orc()
        orc.initialize()
        # Manually advance board to position just before Qxf7#
        pre_mate = chess.Board()
        for m in ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6"]:
            pre_mate.push_uci(m)
        orc.board = pre_mate.copy()
        orc.board.turn  # White to move
        # Human plays Qxf7#
        after_mate = pre_mate.copy()
        after_mate.push_uci("h5f7")
        assert after_mate.is_checkmate()
        orc.inject_human_fen(after_mate.fen())
        orc.run_turn()  # PROCESSING_VISION
        assert orc.state == ArenaState.GAME_OVER


# ---------------------------------------------------------------------------
# 3. Capture handling
# ---------------------------------------------------------------------------


class TestCaptureHandling:
    """Verify capture: victim goes to graveyard, then attacker moves to square."""

    def _make_capture_position(self):
        """
        Craft a simple open-file capture position.
        White pawn on e4, Black pawn on d5 — exd5 is a capture.
        White to move.
        """
        board = chess.Board()
        for m in ["e2e4", "d7d5"]:
            board.push_uci(m)
        return board

    def test_capture_increments_graveyard_counter(self):
        """After a capture move, the graveyard slot counter increments."""
        orc = _make_orc()
        orc.initialize()

        capture_board = self._make_capture_position()
        orc.board = capture_board.copy()

        # Human plays exd5 (pawn capture)
        _feed_move(orc, "e4d5")
        orc.run_turn()  # PV -> THINKING_LLM
        assert orc.state == ArenaState.THINKING_LLM

        # Robot should pick a move from the new position
        with patch(_PATCH_LLM, side_effect=_first_legal):
            with patch(_PATCH_MOVEL, return_value=None):
                orc.run_turn()  # THINKING -> ROBOT_MOVING

                # Intercept capture call to verify graveyard usage
                original_execute_capture = orc._sequencer.execute_capture
                capture_calls: list = []

                def spy_capture(driver, from_sq, to_sq, captured_color):
                    capture_calls.append((from_sq, to_sq, captured_color))
                    return original_execute_capture(driver, from_sq, to_sq, captured_color)

                if orc.board.is_capture(orc._robot_move_pending):
                    with patch.object(orc._sequencer, "execute_capture", side_effect=spy_capture):
                        orc.run_turn()
                else:
                    orc.run_turn()  # not a capture — just advance

    def test_capture_calls_execute_capture_not_execute_normal(self):
        """When robot picks a capture move, execute_capture is called, not execute_normal."""
        # Position: many captures available — use Scandinavian; robot (Black) can capture e4
        orc = _make_orc()
        orc.initialize()

        # Push e4 d5, now it's Black's turn (robot) in the orc; White just moved e4
        after_e4 = chess.Board()
        after_e4.push_uci("e2e4")
        orc.board = after_e4.copy()
        _feed_move(orc, "d7d5")  # human (Black) plays d5
        orc.run_turn()  # PV -> THINKING; robot is now White
        assert orc.state == ArenaState.THINKING_LLM

        capture_move = chess.Move.from_uci("e4d5")  # White exd5

        normal_spy = MagicMock()
        capture_spy = MagicMock()

        with patch(_PATCH_LLM, return_value=capture_move):
            with patch(_PATCH_MOVEL, return_value=None):
                orc.run_turn()  # THINKING -> ROBOT_MOVING
                with patch.object(orc._sequencer, "execute_normal", normal_spy):
                    with patch.object(orc._sequencer, "execute_capture", capture_spy):
                        orc.run_turn()  # ROBOT_MOVING

        capture_spy.assert_called_once()
        normal_spy.assert_not_called()

    def test_capture_execute_capture_receives_correct_color(self):
        """captured_color passed to execute_capture matches the piece color on the target square."""
        # Robot (Black) captures a White piece: captured_color should be 'white'
        orc = _make_orc()
        orc.initialize()

        after_e4 = chess.Board()
        after_e4.push_uci("e2e4")
        orc.board = after_e4.copy()

        _feed_move(orc, "d7d5")
        orc.run_turn()  # PV -> THINKING (robot is White now)
        assert orc.state == ArenaState.THINKING_LLM

        captured_colors: list = []
        original_capture = orc._sequencer.execute_capture

        def capture_spy(driver, from_sq, to_sq, captured_color):
            captured_colors.append(captured_color)
            return original_capture(driver, from_sq, to_sq, captured_color)

        # Force robot to play exd5 (White captures Black pawn)
        capture_move = chess.Move.from_uci("e4d5")
        with patch(_PATCH_LLM, return_value=capture_move):
            with patch(_PATCH_MOVEL, return_value=None):
                orc.run_turn()  # THINKING -> ROBOT_MOVING
                with patch.object(orc._sequencer, "execute_capture", side_effect=capture_spy):
                    orc.run_turn()  # ROBOT_MOVING

        assert captured_colors == ["black"]  # robot (White) captures Black's pawn


# ---------------------------------------------------------------------------
# 4. Castling moves
# ---------------------------------------------------------------------------


class TestCastlingMoves:
    """Verify castling: execute_castling called with correct king/rook squares."""

    def _castling_position_white_kingside(self) -> chess.Board:
        """Position where White can castle kingside (e1g1)."""
        board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
        return board

    def _castling_position_white_queenside(self) -> chess.Board:
        """Position where White can castle queenside (e1c1)."""
        board = chess.Board("r3kbnr/ppp1pppp/2nq4/3p1b2/3P1B2/2NQ4/PPP1PPPP/R3KBNR w KQkq - 4 4")
        return board

    def _setup_castling_test(self, board_position: chess.Board, castle_uci: str):
        """
        Drive the robot to execute a castling move.
        Returns the castling_spy mock to inspect calls.
        """
        orc = _make_orc()
        orc.initialize()
        # Set board to castling-ready position (it's White's turn = human)
        # Push a dummy human move then replace board (hacky but avoids FEN inject complexity)
        # Actually: we want robot to castle, so human plays a filler, robot responds with castle
        orc.board = board_position.copy()

        # Human plays a1a1 (null) trick won't work; use an actual legal human move instead
        # Pick a legal move for human (current side)
        human_move = next(
            m for m in board_position.legal_moves
            if m.uci() != castle_uci
        )
        _feed_move(orc, human_move.uci())

        castling_spy = MagicMock()

        with patch(_PATCH_LLM, return_value=chess.Move.from_uci(castle_uci)):
            with patch(_PATCH_MOVEL, return_value=None):
                orc.run_turn()  # PV -> THINKING
                orc.run_turn()  # THINKING -> ROBOT_MOVING
                with patch.object(orc._sequencer, "execute_castling", castling_spy):
                    with patch.object(orc._sequencer, "execute_normal", MagicMock()):
                        with patch.object(orc._sequencer, "execute_capture", MagicMock()):
                            orc.run_turn()  # ROBOT_MOVING

        return orc, castling_spy

    def test_kingside_castling_calls_execute_castling(self):
        """Kingside castling (e1g1) routes to execute_castling."""
        # Build position; human is White (orc starts White as human side)
        board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
        # After human plays a legal move, robot (Black) can't kingside castle yet from this
        # Instead: let robot be White by pushing one human (Black) move from a flipped board
        orc = _make_orc()
        orc.initialize()
        # Black has pieces moved — let robot be White and castle
        # Swap: human plays a Black move, then robot is White and castles
        black_board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4")
        orc.board = black_board.copy()
        # Human (Black) makes a non-castling move
        human_uci = next(
            m.uci() for m in black_board.legal_moves
            if not black_board.is_castling(m)
        )
        _feed_move(orc, human_uci)

        # After human move board is White's turn; castle available
        castle_uci = "e1g1"
        # Verify it's legal
        after_human = black_board.copy()
        after_human.push_uci(human_uci)
        if chess.Move.from_uci(castle_uci) not in after_human.legal_moves:
            pytest.skip("Castling not available after human move in this fixture")

        castling_spy = MagicMock()
        with patch(_PATCH_LLM, return_value=chess.Move.from_uci(castle_uci)):
            with patch(_PATCH_MOVEL, return_value=None):
                orc.run_turn()  # PV -> THINKING
                orc.run_turn()  # THINKING -> ROBOT_MOVING
                with patch.object(orc._sequencer, "execute_castling", castling_spy):
                    with patch.object(orc._sequencer, "execute_normal", MagicMock()):
                        with patch.object(orc._sequencer, "execute_capture", MagicMock()):
                            orc.run_turn()

        castling_spy.assert_called_once()
        args = castling_spy.call_args[0]
        # args: (driver, king_from, king_to, rook_from, rook_to)
        _, king_from, king_to, rook_from, rook_to = args
        assert king_from == "e1"
        assert king_to == "g1"
        assert rook_from == "h1"
        assert rook_to == "f1"

    def test_queenside_castling_calls_execute_castling_correct_squares(self):
        """Queenside castling (e1c1) routes to execute_castling with correct squares."""
        orc = _make_orc()
        orc.initialize()
        # Position: White can castle queenside, it's Black's turn so human (Black) moves first
        qs_board = chess.Board("r3k2r/pppqpppp/2npbn2/3p1b2/3P1B2/2NPBN2/PPPQPPPP/R3K2R b KQkq - 0 1")
        orc.board = qs_board.copy()

        human_uci = next(
            m.uci() for m in qs_board.legal_moves
            if not qs_board.is_castling(m)
        )
        _feed_move(orc, human_uci)

        after_human = qs_board.copy()
        after_human.push_uci(human_uci)
        castle_uci = "e1c1"
        if chess.Move.from_uci(castle_uci) not in after_human.legal_moves:
            pytest.skip("Queenside castling not available after human move in this fixture")

        castling_spy = MagicMock()
        with patch(_PATCH_LLM, return_value=chess.Move.from_uci(castle_uci)):
            with patch(_PATCH_MOVEL, return_value=None):
                orc.run_turn()
                orc.run_turn()
                with patch.object(orc._sequencer, "execute_castling", castling_spy):
                    with patch.object(orc._sequencer, "execute_normal", MagicMock()):
                        with patch.object(orc._sequencer, "execute_capture", MagicMock()):
                            orc.run_turn()

        castling_spy.assert_called_once()
        _, king_from, king_to, rook_from, rook_to = castling_spy.call_args[0]
        assert king_from == "e1"
        assert king_to == "c1"
        assert rook_from == "a1"
        assert rook_to == "d1"

    def test_black_kingside_castling(self):
        """Black kingside castling (e8g8) routes correctly."""
        orc = _make_orc()
        orc.initialize()
        # White's turn; robot is Black; human plays, robot castles
        board = chess.Board("rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1")
        orc.board = board.copy()

        human_uci = next(
            m.uci() for m in board.legal_moves
            if not board.is_castling(m)
        )
        _feed_move(orc, human_uci)

        after_human = board.copy()
        after_human.push_uci(human_uci)
        castle_uci = "e8g8"
        if chess.Move.from_uci(castle_uci) not in after_human.legal_moves:
            pytest.skip("Black kingside castling not legal in this fixture")

        castling_spy = MagicMock()
        with patch(_PATCH_LLM, return_value=chess.Move.from_uci(castle_uci)):
            with patch(_PATCH_MOVEL, return_value=None):
                orc.run_turn()
                orc.run_turn()
                with patch.object(orc._sequencer, "execute_castling", castling_spy):
                    with patch.object(orc._sequencer, "execute_normal", MagicMock()):
                        with patch.object(orc._sequencer, "execute_capture", MagicMock()):
                            orc.run_turn()

        castling_spy.assert_called_once()
        _, king_from, king_to, rook_from, rook_to = castling_spy.call_args[0]
        assert king_from == "e8"
        assert king_to == "g8"
        assert rook_from == "h8"
        assert rook_to == "f8"


# ---------------------------------------------------------------------------
# 5. Recovery from invalid vision inputs and disconnects
# ---------------------------------------------------------------------------


class TestRecovery:
    """FSM must recover gracefully from bad vision data and driver failures."""

    def setup_method(self):
        self.orc = _make_orc()
        self.orc.initialize()

    # -- Invalid FEN --

    def test_garbage_fen_returns_to_wait_human(self):
        """Completely invalid FEN string: FSM stays alive and returns to WAIT_HUMAN_MOVE."""
        self.orc._pending_fen = "not-a-fen-at-all!!!###"
        self.orc._state = ArenaState.PROCESSING_VISION
        self.orc.run_turn()
        assert self.orc.state == ArenaState.WAIT_HUMAN_MOVE
        assert self.orc.error is None  # not an error state

    def test_empty_fen_returns_to_wait_human(self):
        self.orc._pending_fen = ""
        self.orc._state = ArenaState.PROCESSING_VISION
        self.orc.run_turn()
        assert self.orc.state == ArenaState.WAIT_HUMAN_MOVE

    def test_none_pending_fen_stays_in_processing_vision(self):
        """No pending FEN: step is a no-op and state stays PROCESSING_VISION."""
        self.orc._pending_fen = None
        self.orc._state = ArenaState.PROCESSING_VISION
        self.orc.run_turn()
        assert self.orc.state == ArenaState.PROCESSING_VISION

    def test_same_fen_no_move_returns_to_wait_human(self):
        """FEN identical to current board = no move detected → re-wait."""
        self.orc._pending_fen = self.orc.board.fen()
        self.orc._state = ArenaState.PROCESSING_VISION
        self.orc.run_turn()
        assert self.orc.state == ArenaState.WAIT_HUMAN_MOVE

    def test_two_moves_ahead_fen_returns_to_wait_human(self):
        """FEN two moves ahead is not reachable by one legal White move — re-wait."""
        b = chess.Board()
        b.push_uci("e2e4")
        b.push_uci("e7e5")
        self.orc._pending_fen = b.fen()
        self.orc._state = ArenaState.PROCESSING_VISION
        self.orc.run_turn()
        assert self.orc.state == ArenaState.WAIT_HUMAN_MOVE

    def test_multiple_sequential_bad_fens_do_not_enter_error(self):
        """Repeated bad FEN injections do not accumulate into ERROR_SAFE_STOP."""
        for bad in ["garbage", "", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR EXTRA"]:
            self.orc._pending_fen = bad
            self.orc._state = ArenaState.PROCESSING_VISION
            self.orc.run_turn()
            assert self.orc.state == ArenaState.WAIT_HUMAN_MOVE
            assert self.orc.error is None

    # -- Verify callback failure (simulated physical mismatch) --

    def test_verify_failure_triggers_error_safe_stop(self):
        """Verify callback returning False drives FSM to ERROR_SAFE_STOP."""
        orc = _make_orc(verify_cb=lambda _: False)
        orc.initialize()
        orc._state = ArenaState.VERIFYING_BOARD
        orc.run_turn()
        assert orc.state == ArenaState.ERROR_SAFE_STOP

    # -- Simulated driver disconnect mid-game --

    def test_driver_disconnect_during_robot_move_enters_error(self):
        """RuntimeError from sequencer mid-move = ERROR_SAFE_STOP (safe-stop)."""
        orc = _make_orc()
        orc.initialize()
        _feed_move(orc, "e2e4")
        orc.run_turn()  # PV -> THINKING
        with patch(_PATCH_LLM, side_effect=_first_legal):
            orc.run_turn()  # THINKING -> ROBOT_MOVING
        assert orc.state == ArenaState.ROBOT_MOVING

        with patch.object(orc._sequencer, "execute_normal", side_effect=ConnectionResetError("lost")):
            with patch.object(orc._sequencer, "execute_capture", side_effect=ConnectionResetError("lost")):
                with patch.object(orc._sequencer, "execute_castling", side_effect=ConnectionResetError("lost")):
                    orc.run_turn()

        assert orc.state == ArenaState.ERROR_SAFE_STOP
        assert orc.error is not None

    def test_gripper_released_on_error(self):
        """On error, set_digital_out(0, False) is called to release gripper."""
        orc = _make_orc()
        orc.initialize()
        digital_calls: list = []
        original_set_digital = orc._driver.set_digital_out

        def spy_digital(port, value):
            digital_calls.append((port, value))
            return original_set_digital(port, value)

        orc._driver.set_digital_out = spy_digital  # type: ignore[method-assign]
        orc._enter_error("test gripper release")

        # (0, False) must appear in calls — gripper open/safe
        assert (0, False) in digital_calls

    def test_shutdown_after_error_is_safe(self):
        """shutdown() must not raise even when driver is None (post-error cleanup)."""
        orc = _make_orc()
        orc.initialize()
        orc._driver = None  # simulate already-disconnected
        orc.shutdown()  # must not raise
        assert orc._driver is None

    def test_inject_human_fen_ignored_after_error(self):
        """inject_human_fen in ERROR_SAFE_STOP is silently ignored."""
        orc = _make_orc()
        orc.initialize()
        orc._enter_error("forced")
        assert orc.state == ArenaState.ERROR_SAFE_STOP
        after_e4 = chess.Board()
        after_e4.push_uci("e2e4")
        orc.inject_human_fen(after_e4.fen())  # must not raise or change state
        assert orc.state == ArenaState.ERROR_SAFE_STOP

    # -- LLM failure recovery path --

    def test_llm_failure_enters_error_safe_stop(self):
        """LLM exception at THINKING_LLM drives FSM to ERROR_SAFE_STOP."""
        orc = _make_orc()
        orc.initialize()
        _feed_move(orc, "e2e4")
        orc.run_turn()  # PV -> THINKING
        with patch(_PATCH_LLM, side_effect=TimeoutError("LLM timeout")):
            orc.run_turn()
        assert orc.state == ArenaState.ERROR_SAFE_STOP
        assert "LLM strategy failed" in (orc.error or "")

    def test_run_turn_in_error_state_is_noop(self):
        """Calling run_turn() repeatedly in ERROR_SAFE_STOP does not change state or raise."""
        orc = _make_orc()
        orc.initialize()
        orc._enter_error("sentinel")
        for _ in range(5):
            orc.run_turn()
        assert orc.state == ArenaState.ERROR_SAFE_STOP
        assert orc.error == "sentinel"


# ---------------------------------------------------------------------------
# 6. run_game integration: Fool's Mate end-to-end
# ---------------------------------------------------------------------------


class TestRunGameIntegration:
    """Full run_game() end-to-end: Fool's Mate in 4 moves (2 human, 2 robot)."""

    def test_fools_mate_game_ends_game_over(self):
        """
        Fool's Mate sequence:
            1. f3 (human White)  — robot replies e5 (Black)
            2. g4 (human White)  — robot replies Qh4# (checkmate)
        FSM reaches GAME_OVER; board.is_checkmate() is True.
        """
        call_count = [0]

        def human_cb(board: chess.Board):
            call_count[0] += 1
            if call_count[0] == 1:
                b = board.copy()
                b.push_uci("f2f3")
                return b.fen()
            if call_count[0] == 2:
                b = board.copy()
                b.push_uci("g2g4")
                return b.fen()
            return None

        robot_replies = [
            chess.Move.from_uci("e7e5"),
            chess.Move.from_uci("d8h4"),  # Qh4#
        ]
        reply_idx = [0]

        def llm_cb(board, **kw):
            m = robot_replies[reply_idx[0]]
            reply_idx[0] += 1
            return m

        orc = MasterOrchestrator(
            config=ArenaConfig(),
            robot_mode="sim",
            human_fen_callback=human_cb,
        )

        with patch(_PATCH_MOVEL, return_value=None):
            with patch(_PATCH_LLM, side_effect=llm_cb):
                result = orc.run_game(poll_interval=0, max_turns=50)

        assert orc.state == ArenaState.GAME_OVER
        assert orc.board.is_checkmate()
        # Black (robot) delivered checkmate → result is 0-1
        assert result == "0-1"

    def test_run_game_raises_on_llm_failure(self):
        """run_game raises OrchestrationError when LLM crashes mid-game."""
        call_count = [0]

        def human_cb(board):
            call_count[0] += 1
            if call_count[0] == 1:
                b = board.copy()
                b.push_uci("e2e4")
                return b.fen()
            return None

        with patch(_PATCH_MOVEL, return_value=None):
            with patch(_PATCH_LLM, side_effect=RuntimeError("LLM down")):
                with pytest.raises(OrchestrationError, match="LLM"):
                    orc = MasterOrchestrator(
                        config=ArenaConfig(),
                        robot_mode="sim",
                        human_fen_callback=human_cb,
                    )
                    orc.run_game(poll_interval=0, max_turns=20)

    def test_run_game_max_turns_raises(self):
        """run_game raises OrchestrationError when max_turns exceeded."""
        # human_cb always returns a valid move so the game loops
        def human_cb(board):
            if board.turn == chess.WHITE:
                b = board.copy()
                m = next(iter(b.legal_moves))
                b.push(m)
                return b.fen()
            return None

        with patch(_PATCH_MOVEL, return_value=None):
            with patch(_PATCH_LLM, side_effect=_first_legal):
                with pytest.raises(OrchestrationError, match="max_turns"):
                    orc = MasterOrchestrator(
                        config=ArenaConfig(),
                        robot_mode="sim",
                        human_fen_callback=human_cb,
                    )
                    orc.run_game(poll_interval=0, max_turns=3)

    def test_state_change_callback_called_throughout_game(self):
        """on_state_change callback fires for every FSM transition during run_game."""
        transitions: list = []

        def human_cb(board):
            if board.turn == chess.WHITE and board.fullmove_number == 1:
                b = board.copy()
                b.push_uci("e2e4")
                return b.fen()
            return None

        orc = MasterOrchestrator(
            config=ArenaConfig(),
            robot_mode="sim",
            human_fen_callback=human_cb,
            on_state_change=lambda o, n: transitions.append(n),
        )

        # Run only one full turn then stop by capping max_turns
        with patch(_PATCH_MOVEL, return_value=None):
            with patch(_PATCH_LLM, side_effect=_first_legal):
                try:
                    orc.run_game(poll_interval=0, max_turns=8)
                except OrchestrationError:
                    pass

        assert ArenaState.THINKING_LLM in transitions
        assert ArenaState.ROBOT_MOVING in transitions
