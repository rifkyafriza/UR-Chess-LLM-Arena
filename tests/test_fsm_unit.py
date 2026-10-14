"""Unit tests for MasterOrchestrator FSM.

All tests use MockURDriver (no real robot) and stub out:
  - human_fen_callback  (provides canned FEN after human move)
  - verify_callback     (always returns True)
  - LLM get_best_move   (patched with first legal move for determinism)
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import chess
import pytest

from src.config import ArenaConfig
from src.orchestrator.master_fsm import ArenaState, MasterOrchestrator, OrchestrationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_orchestrator(
    human_fen_cb=None,
    verify_cb=None,
    on_state_change=None,
) -> MasterOrchestrator:
    """Factory with safe defaults for unit tests."""
    return MasterOrchestrator(
        config=ArenaConfig(),
        robot_mode="sim",
        human_fen_callback=human_fen_cb,
        verify_callback=verify_cb or (lambda _: True),
        on_state_change=on_state_change,
    )


def _first_legal(board: chess.Board, **_kwargs) -> chess.Move:
    return next(iter(board.legal_moves))


# Patch target: get_best_move lives inside master_fsm module namespace
_PATCH_LLM = "src.orchestrator.master_fsm.get_best_move"


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInitialization:
    def test_initial_state_is_initializing(self):
        orc = _make_orchestrator()
        assert orc.state == ArenaState.INITIALIZING

    def test_initialize_transitions_to_wait_human(self):
        orc = _make_orchestrator()
        ok = orc.initialize()
        assert ok
        assert orc.state == ArenaState.WAIT_HUMAN_MOVE

    def test_initialize_idempotent_on_repeated_call(self):
        orc = _make_orchestrator()
        orc.initialize()
        # Second initialize is a no-op (state != INITIALIZING)
        prev_state = orc.state
        orc.initialize()
        # Still WAIT_HUMAN_MOVE, not INITIALIZING
        assert orc.state == ArenaState.WAIT_HUMAN_MOVE

    def test_board_starts_at_standard_position(self):
        orc = _make_orchestrator()
        orc.initialize()
        assert orc.board == chess.Board()

    def test_shutdown_after_init_ok(self):
        orc = _make_orchestrator()
        orc.initialize()
        orc.shutdown()
        assert orc._driver is None


# ---------------------------------------------------------------------------
# ArenaState enum
# ---------------------------------------------------------------------------

class TestArenaStateEnum:
    def test_all_required_states_present(self):
        required = {
            "INITIALIZING", "WAIT_HUMAN_MOVE", "PROCESSING_VISION",
            "THINKING_LLM", "ROBOT_MOVING", "VERIFYING_BOARD",
            "GAME_OVER", "ERROR_SAFE_STOP",
        }
        actual = {s.name for s in ArenaState}
        assert required <= actual


# ---------------------------------------------------------------------------
# Human move injection
# ---------------------------------------------------------------------------

class TestInjectHumanFen:
    def setup_method(self):
        self.orc = _make_orchestrator()
        self.orc.initialize()

    def test_inject_transitions_to_processing_vision(self):
        board = chess.Board()
        board.push_uci("e2e4")
        self.orc.inject_human_fen(board.fen())
        assert self.orc.state == ArenaState.PROCESSING_VISION

    def test_inject_ignored_in_wrong_state(self):
        self.orc._state = ArenaState.ROBOT_MOVING
        board = chess.Board()
        board.push_uci("e2e4")
        self.orc.inject_human_fen(board.fen())
        assert self.orc.state == ArenaState.ROBOT_MOVING  # unchanged

    def test_inject_bad_fen_goes_back_to_wait(self):
        self.orc._state = ArenaState.WAIT_HUMAN_MOVE
        self.orc._pending_fen = "not-a-valid-fen"
        self.orc._state = ArenaState.PROCESSING_VISION
        self.orc.run_turn()
        assert self.orc.state == ArenaState.WAIT_HUMAN_MOVE


# ---------------------------------------------------------------------------
# Vision processing step
# ---------------------------------------------------------------------------

class TestProcessingVision:
    def setup_method(self):
        self.orc = _make_orchestrator()
        self.orc.initialize()

    def _push_human(self, uci: str) -> None:
        """Simulate human playing uci move."""
        board = chess.Board()
        board.push_uci(uci)
        self.orc.inject_human_fen(board.fen())

    def test_valid_human_move_advances_to_thinking_llm(self):
        with patch(_PATCH_LLM, side_effect=_first_legal):
            self._push_human("e2e4")
            self.orc.run_turn()  # PROCESSING_VISION
            assert self.orc.state == ArenaState.THINKING_LLM

    def test_human_move_updates_board(self):
        with patch(_PATCH_LLM, side_effect=_first_legal):
            self._push_human("e2e4")
            self.orc.run_turn()
            assert self.orc.board.peek().uci() == "e2e4"

    def test_fen_with_no_move_diff_goes_back_to_wait(self):
        # Same FEN = no move detected
        self.orc._pending_fen = chess.Board().fen()
        self.orc._state = ArenaState.PROCESSING_VISION
        self.orc.run_turn()
        assert self.orc.state == ArenaState.WAIT_HUMAN_MOVE

    def test_fen_with_illegal_move_goes_back_to_wait(self):
        # Force a board where detected move is illegal for original board
        # Build a FEN from a different starting point to confuse diff
        alt = chess.Board()
        alt.push_uci("e2e4")
        alt.push_uci("e7e5")  # two moves in — not reachable from start by one legal White move
        self.orc._pending_fen = alt.fen()
        self.orc._state = ArenaState.PROCESSING_VISION
        self.orc.run_turn()
        assert self.orc.state == ArenaState.WAIT_HUMAN_MOVE


# ---------------------------------------------------------------------------
# LLM thinking step
# ---------------------------------------------------------------------------

class TestThinkingLLM:
    def setup_method(self):
        self.orc = _make_orchestrator()
        self.orc.initialize()

    def _advance_to_thinking(self, uci="e2e4"):
        board_after = chess.Board()
        board_after.push_uci(uci)
        self.orc.inject_human_fen(board_after.fen())
        self.orc.run_turn()  # PROCESSING_VISION
        assert self.orc.state == ArenaState.THINKING_LLM

    def test_llm_ok_transitions_to_robot_moving(self):
        with patch(_PATCH_LLM, side_effect=_first_legal):
            self._advance_to_thinking()
            self.orc.run_turn()  # THINKING_LLM
            assert self.orc.state == ArenaState.ROBOT_MOVING

    def test_llm_exception_enters_error(self):
        self._advance_to_thinking()
        with patch(_PATCH_LLM, side_effect=RuntimeError("LLM down")):
            self.orc.run_turn()  # THINKING_LLM
        assert self.orc.state == ArenaState.ERROR_SAFE_STOP
        assert "LLM strategy failed" in (self.orc.error or "")

    def test_robot_move_pending_set_after_thinking(self):
        with patch(_PATCH_LLM, side_effect=_first_legal):
            self._advance_to_thinking()
            self.orc.run_turn()
            assert hasattr(self.orc, "_robot_move_pending")
            assert isinstance(self.orc._robot_move_pending, chess.Move)


# ---------------------------------------------------------------------------
# Robot moving step (simulation — no real motion delays in these unit tests)
# ---------------------------------------------------------------------------

class TestRobotMoving:
    def setup_method(self):
        self.orc = _make_orchestrator()
        self.orc.initialize()
        # Patch MockURDriver.movel to be instant (no sleep)
        patcher = patch.object(
            self.orc._driver.__class__, "movel", return_value=None
        )
        self._movel_patcher = patcher
        self.mock_movel = patcher.start()

    def teardown_method(self):
        # Must stop patcher; otherwise MockURDriver.movel stays stubbed for later tests
        self._movel_patcher.stop()

    def _run_to_robot_moving(self, uci="e2e4"):
        board_after = chess.Board()
        board_after.push_uci(uci)
        self.orc.inject_human_fen(board_after.fen())
        with patch(_PATCH_LLM, side_effect=_first_legal):
            self.orc.run_turn()  # PROCESSING_VISION -> THINKING_LLM
            self.orc.run_turn()  # THINKING_LLM -> ROBOT_MOVING
        assert self.orc.state == ArenaState.ROBOT_MOVING

    def test_normal_move_transitions_to_verifying(self):
        self._run_to_robot_moving()
        self.orc.run_turn()  # ROBOT_MOVING
        assert self.orc.state in (ArenaState.VERIFYING_BOARD, ArenaState.GAME_OVER)

    def test_board_updated_after_robot_move(self):
        self._run_to_robot_moving()
        moves_before = self.orc.board.fullmove_number
        self.orc.run_turn()
        # After robot (Black) moves, fullmove increments
        assert self.orc.board.fullmove_number > moves_before or self.orc.board.is_game_over()

    def test_last_robot_move_recorded(self):
        self._run_to_robot_moving()
        self.orc.run_turn()
        if self.orc.state != ArenaState.GAME_OVER:
            assert isinstance(self.orc.last_robot_move, chess.Move)

    def test_robot_error_enters_error_state(self):
        self._run_to_robot_moving()
        with patch.object(self.orc._sequencer, "execute_normal", side_effect=RuntimeError("motor fault")):
            with patch.object(self.orc._sequencer, "execute_capture", side_effect=RuntimeError("motor fault")):
                with patch.object(self.orc._sequencer, "execute_castling", side_effect=RuntimeError("motor fault")):
                    self.orc.run_turn()
        assert self.orc.state == ArenaState.ERROR_SAFE_STOP


# ---------------------------------------------------------------------------
# Verifying board step
# ---------------------------------------------------------------------------

class TestVerifyingBoard:
    def setup_method(self):
        self.orc = _make_orchestrator()
        self.orc.initialize()

    def test_verify_ok_transitions_to_wait_human(self):
        self.orc._state = ArenaState.VERIFYING_BOARD
        self.orc.run_turn()
        assert self.orc.state == ArenaState.WAIT_HUMAN_MOVE

    def test_verify_fail_enters_error(self):
        orc = _make_orchestrator(verify_cb=lambda _: False)
        orc.initialize()
        orc._state = ArenaState.VERIFYING_BOARD
        orc.run_turn()
        assert orc.state == ArenaState.ERROR_SAFE_STOP
        assert "verification" in (orc.error or "").lower() or "match" in (orc.error or "").lower()


# ---------------------------------------------------------------------------
# State change callback
# ---------------------------------------------------------------------------

class TestStateChangeCallback:
    def test_callback_fired_on_each_transition(self):
        transitions = []
        orc = _make_orchestrator(on_state_change=lambda o, n: transitions.append((o, n)))
        orc.initialize()
        assert len(transitions) >= 1
        assert transitions[-1][1] == ArenaState.WAIT_HUMAN_MOVE

    def test_callback_exception_does_not_crash_fsm(self):
        def bad_cb(o, n):
            raise ValueError("oops")

        orc = _make_orchestrator(on_state_change=bad_cb)
        # Should not raise
        orc.initialize()
        assert orc.state == ArenaState.WAIT_HUMAN_MOVE


# ---------------------------------------------------------------------------
# Error state
# ---------------------------------------------------------------------------

class TestErrorState:
    def test_error_state_idempotent_on_run_turn(self):
        orc = _make_orchestrator()
        orc.initialize()
        orc._enter_error("test error")
        assert orc.state == ArenaState.ERROR_SAFE_STOP
        orc.run_turn()  # should be no-op
        assert orc.state == ArenaState.ERROR_SAFE_STOP

    def test_error_message_preserved(self):
        orc = _make_orchestrator()
        orc.initialize()
        orc._enter_error("something went wrong")
        assert orc.error == "something went wrong"


# ---------------------------------------------------------------------------
# Game over detection
# ---------------------------------------------------------------------------

class TestGameOver:
    def test_game_over_state_is_terminal(self):
        orc = _make_orchestrator()
        orc.initialize()
        orc._state = ArenaState.GAME_OVER
        orc.run_turn()  # should be no-op
        assert orc.state == ArenaState.GAME_OVER

    def test_checkmate_position_transitions_game_over(self):
        """Push a Fool's Mate sequence and verify FSM reaches GAME_OVER."""
        # Fool's mate: 1.f3 e5 2.g4 Qh4#
        # White (human) plays f2f3
        orc = _make_orchestrator()
        orc.initialize()

        moves = ["f2f3", "e7e5", "g2g4"]  # White moves; Black (robot) moves e7e5 g-pawn
        # We'll drive manually: alternating human/robot until mate

        with patch("src.robot.ur_driver.MockURDriver.movel", return_value=None):
            with patch(_PATCH_LLM) as mock_llm:
                # After human f2f3, robot plays e7e5
                after_f3 = chess.Board()
                after_f3.push_uci("f2f3")
                orc.inject_human_fen(after_f3.fen())
                mock_llm.return_value = chess.Move.from_uci("e7e5")
                orc.run_turn()  # PROCESSING_VISION -> THINKING_LLM
                orc.run_turn()  # THINKING_LLM -> ROBOT_MOVING
                orc.run_turn()  # ROBOT_MOVING -> VERIFYING_BOARD
                orc.run_turn()  # VERIFYING_BOARD -> WAIT_HUMAN_MOVE

                # Human plays g2g4
                cur = orc.board.copy()
                cur.push_uci("g2g4")
                orc.inject_human_fen(cur.fen())
                mock_llm.return_value = chess.Move.from_uci("d8h4")  # Qh4# (checkmate)
                orc.run_turn()  # PROCESSING_VISION
                orc.run_turn()  # THINKING_LLM
                orc.run_turn()  # ROBOT_MOVING — should transition to GAME_OVER

        assert orc.state == ArenaState.GAME_OVER


# ---------------------------------------------------------------------------
# detect_move static method
# ---------------------------------------------------------------------------

class TestDetectMove:
    def test_detects_simple_pawn_advance(self):
        before = chess.Board()
        after = chess.Board()
        after.push_uci("e2e4")
        move = MasterOrchestrator._detect_move(before, after)
        assert move == chess.Move.from_uci("e2e4")

    def test_detects_capture(self):
        board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 0 3")
        after = board.copy()
        cap_move = chess.Move.from_uci("c6d4")
        after.push(cap_move)
        move = MasterOrchestrator._detect_move(board, after)
        assert move == cap_move

    def test_returns_none_for_same_board(self):
        board = chess.Board()
        move = MasterOrchestrator._detect_move(board, board.copy())
        assert move is None

    def test_detects_castling_kingside(self):
        # Position where White can castle kingside
        board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
        after = board.copy()
        after.push_uci("e1g1")
        move = MasterOrchestrator._detect_move(board, after)
        assert move is not None
        assert move.uci() == "e1g1"


# ---------------------------------------------------------------------------
# run_game integration smoke test (fully mocked, no I/O)
# ---------------------------------------------------------------------------

class TestRunGameSmoke:
    def test_run_game_raises_on_error(self):
        """run_game re-raises OrchestrationError when FSM hits error."""
        orc = MasterOrchestrator(
            config=ArenaConfig(),
            robot_mode="sim",
            # Force error immediately: LLM fails, human_fen_cb returns a move
        )
        call_count = [0]

        def human_cb(board):
            call_count[0] += 1
            if call_count[0] == 1:
                after = board.copy()
                after.push_uci("e2e4")
                return after.fen()
            return None

        orc._human_fen_cb = human_cb

        with patch("src.robot.ur_driver.MockURDriver.movel", return_value=None):
            with patch(_PATCH_LLM, side_effect=RuntimeError("boom")):
                with pytest.raises(OrchestrationError):
                    orc.run_game(poll_interval=0, max_turns=20)
