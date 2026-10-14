"""Master Orchestrator FSM for UR Chess LLM Arena.

Ties together Vision, Strategy, and Robotics into a single game loop.

State machine:
    INITIALIZING       -> WAIT_HUMAN_MOVE (on successful init)
    WAIT_HUMAN_MOVE    -> PROCESSING_VISION (human move detected)
    PROCESSING_VISION  -> THINKING_LLM (FEN synced)
    THINKING_LLM       -> ROBOT_MOVING (move selected)
    ROBOT_MOVING       -> VERIFYING_BOARD (robot done)
    VERIFYING_BOARD    -> WAIT_HUMAN_MOVE (board OK, not game over)
                      -> GAME_OVER (checkmate / stalemate / draw)
    Any state          -> ERROR_SAFE_STOP (on unrecoverable error)
"""

from __future__ import annotations

import logging
import time
from enum import Enum, auto
from typing import Callable, Optional, Union

import chess

from src.config import ArenaConfig
from src.robot.coordinates import BoardCoordinates, WorkspaceSafetyEnvelope
from src.robot.graveyard import GraveyardManager, MoveSequencer
from src.robot.trajectory import TrajectoryGenerator, DEFAULT_PARK_POSE
from src.robot.ur_driver import MockURDriver, RealURDriver, create_robot_driver
from src.strategy.llm_client import get_best_move

log = logging.getLogger(__name__)

# Type alias for either driver
AnyDriver = Union[MockURDriver, RealURDriver]


class ArenaState(Enum):
    INITIALIZING = auto()
    WAIT_HUMAN_MOVE = auto()
    PROCESSING_VISION = auto()
    THINKING_LLM = auto()
    ROBOT_MOVING = auto()
    VERIFYING_BOARD = auto()
    GAME_OVER = auto()
    ERROR_SAFE_STOP = auto()


class OrchestrationError(Exception):
    """Raised when FSM hits an unrecoverable error."""


class MasterOrchestrator:
    """
    Top-level arena controller.

    In simulation mode, vision is replaced by a virtual FEN feed
    (inject_human_move calls human_fen_callback instead of live camera).

    Parameters
    ----------
    config          : ArenaConfig — centralized config.
    robot_mode      : 'sim' or 'real'.
    human_fen_callback : callable (board: chess.Board) -> str | None
                        Should return the new FEN after human move, or None
                        if no move detected yet.  Used in both sim and real
                        modes (real mode wraps vision into this contract).
    verify_callback : callable (board: chess.Board) -> bool
                      Return True if board physical state matches FEN.
                      Defaults to always-True (no physical verification in sim).
    on_state_change : optional callback (old: ArenaState, new: ArenaState) -> None
    """

    def __init__(
        self,
        config: ArenaConfig,
        robot_mode: str = "sim",
        human_fen_callback: Optional[Callable[[chess.Board], Optional[str]]] = None,
        verify_callback: Optional[Callable[[chess.Board], bool]] = None,
        on_state_change: Optional[Callable[[ArenaState, ArenaState], None]] = None,
    ) -> None:
        self.config = config
        self.robot_mode = robot_mode

        # Chess game state
        self.board = chess.Board()
        self.last_human_move: Optional[chess.Move] = None
        self.last_robot_move: Optional[chess.Move] = None

        # FSM state
        self._state = ArenaState.INITIALIZING
        self._error: Optional[str] = None

        # Callbacks
        self._human_fen_cb = human_fen_callback or (lambda _b: None)
        self._verify_cb = verify_callback or (lambda _b: True)
        self._on_state_change = on_state_change

        # Robot subsystem (initialized in _init_robot)
        self._driver: Optional[AnyDriver] = None
        self._sequencer: Optional[MoveSequencer] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> ArenaState:
        return self._state

    @property
    def error(self) -> Optional[str]:
        return self._error

    def initialize(self) -> bool:
        """
        Boot all subsystems.  Returns True on success.
        Transitions to WAIT_HUMAN_MOVE or ERROR_SAFE_STOP.
        """
        try:
            self._init_robot()
            self._transition(ArenaState.WAIT_HUMAN_MOVE)
            log.info("MasterOrchestrator initialized (robot_mode=%s)", self.robot_mode)
            return True
        except Exception as exc:
            self._enter_error(f"Initialization failed: {exc}")
            return False

    def run_turn(self) -> ArenaState:
        """
        Execute one FSM step from the current state.

        Returns the new state after the step.
        Raises OrchestrationError only when state is ERROR_SAFE_STOP.
        """
        try:
            if self._state == ArenaState.INITIALIZING:
                self.initialize()

            elif self._state == ArenaState.WAIT_HUMAN_MOVE:
                self._step_wait_human()

            elif self._state == ArenaState.PROCESSING_VISION:
                self._step_process_vision()

            elif self._state == ArenaState.THINKING_LLM:
                self._step_think_llm()

            elif self._state == ArenaState.ROBOT_MOVING:
                self._step_robot_move()

            elif self._state == ArenaState.VERIFYING_BOARD:
                self._step_verify()

            elif self._state in (ArenaState.GAME_OVER, ArenaState.ERROR_SAFE_STOP):
                pass  # terminal states — caller decides what to do

        except OrchestrationError:
            raise
        except Exception as exc:
            self._enter_error(f"Unhandled error in state {self._state.name}: {exc}")

        return self._state

    def run_game(
        self,
        poll_interval: float = 0.5,
        max_turns: int = 300,
    ) -> str:
        """
        Block until game ends.  Returns outcome string (e.g. '1-0', '1/2-1/2').

        Parameters
        ----------
        poll_interval : seconds to sleep between WAIT_HUMAN_MOVE polls.
        max_turns     : hard safety cap; raises OrchestrationError if exceeded.
        """
        if not self.initialize():
            raise OrchestrationError(self._error or "Init failed")

        turn = 0
        while self._state not in (ArenaState.GAME_OVER, ArenaState.ERROR_SAFE_STOP):
            if turn >= max_turns:
                self._enter_error(f"max_turns={max_turns} exceeded")
                break
            if self._state == ArenaState.WAIT_HUMAN_MOVE:
                time.sleep(poll_interval)
            self.run_turn()
            turn += 1

        if self._state == ArenaState.ERROR_SAFE_STOP:
            raise OrchestrationError(self._error or "Unknown error")

        return self.board.result()

    def inject_human_fen(self, fen: str) -> None:
        """
        Directly feed a new FEN (for testing / simulation without camera).
        Transition from WAIT_HUMAN_MOVE -> PROCESSING_VISION.
        """
        if self._state != ArenaState.WAIT_HUMAN_MOVE:
            log.warning("inject_human_fen called in state %s; ignored", self._state.name)
            return
        self._pending_fen = fen
        self._transition(ArenaState.PROCESSING_VISION)

    def shutdown(self) -> None:
        """Gracefully disconnect robot driver."""
        if self._driver:
            try:
                self._driver.disconnect()
            except Exception:
                log.exception("Driver disconnect error (ignored)")
            self._driver = None
        log.info("MasterOrchestrator shut down")

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_robot(self) -> None:
        cfg = self.config.robot
        env = WorkspaceSafetyEnvelope(
            x_min=cfg.safety_envelope.x_bounds[0],
            x_max=cfg.safety_envelope.x_bounds[1],
            y_min=cfg.safety_envelope.y_bounds[0],
            y_max=cfg.safety_envelope.y_bounds[1],
            z_min=cfg.safety_envelope.z_bounds[0],
            z_max=cfg.safety_envelope.z_bounds[1],
        )

        # Build coordinates
        bl = cfg.board_layout
        coords = BoardCoordinates(
            origin_xyz=(0.10, -0.45, 0.0),
            dx=bl.square_pitch_x,
            dy=bl.square_pitch_y,
            board_z=0.0,
            pick_z=bl.pick_height_z,
            transit_z=bl.transit_clearance_z,
            envelope=env,
        )

        # Graveyard — default positions left/right of board
        graveyard = GraveyardManager(
            origin_white=(-0.15, -0.45, bl.pick_height_z),
            origin_black=(0.55, -0.45, bl.pick_height_z),
            slot_spacing=0.055,
            slot_axis="x",
            envelope=env,
        )

        traj = TrajectoryGenerator(
            coords=coords,
            park_pose=list(DEFAULT_PARK_POSE),
        )

        self._sequencer = MoveSequencer(trajectory=traj, graveyard=graveyard)

        # Create and connect driver
        self._driver = create_robot_driver(
            mode=self.robot_mode,
            host=cfg.host,
            port_dash=cfg.ports.dashboard,
            port_script=cfg.ports.script,
            port_rt=cfg.ports.realtime,
        )
        self._driver.connect()

        # Pending FEN slot (used by inject_human_fen + _human_fen_cb)
        self._pending_fen: Optional[str] = None

    # ------------------------------------------------------------------
    # FSM steps
    # ------------------------------------------------------------------

    def _step_wait_human(self) -> None:
        """Poll for human move via callback."""
        new_fen = self._human_fen_cb(self.board)
        if new_fen is not None:
            self._pending_fen = new_fen
            self._transition(ArenaState.PROCESSING_VISION)

    def _step_process_vision(self) -> None:
        """Parse pending FEN, detect human move, update board."""
        fen = getattr(self, "_pending_fen", None)
        if fen is None:
            # Nothing to process yet; stay
            return

        try:
            new_board = chess.Board(fen)
        except ValueError as exc:
            log.warning("Invalid FEN from vision: %s (%s) — staying", fen, exc)
            self._pending_fen = None
            self._transition(ArenaState.WAIT_HUMAN_MOVE)
            return

        # Detect which move was played by diffing occupied squares
        human_move = self._detect_move(self.board, new_board)
        if human_move is None:
            log.warning("Could not determine human move from FEN diff — re-waiting")
            self._pending_fen = None
            self._transition(ArenaState.WAIT_HUMAN_MOVE)
            return

        if human_move not in self.board.legal_moves:
            log.warning("Detected human move %s is illegal — re-waiting", human_move)
            self._pending_fen = None
            self._transition(ArenaState.WAIT_HUMAN_MOVE)
            return

        self.board.push(human_move)
        self.last_human_move = human_move
        self._pending_fen = None
        log.info("Human move: %s  FEN: %s", human_move.uci(), self.board.fen())

        if self.board.is_game_over():
            self._transition(ArenaState.GAME_OVER)
        else:
            self._transition(ArenaState.THINKING_LLM)

    def _step_think_llm(self) -> None:
        """Query LLM for robot's best move."""
        cfg = self.config.strategy
        try:
            move = get_best_move(
                self.board,
                model=cfg.model,
                base_url=cfg.api_base,
                timeout=float(cfg.timeout),
            )
        except Exception as exc:
            self._enter_error(f"LLM strategy failed: {exc}")
            return

        self._robot_move_pending = move
        log.info("Robot move chosen: %s", move.uci())
        self._transition(ArenaState.ROBOT_MOVING)

    def _step_robot_move(self) -> None:
        """Execute robot physical trajectory for chosen move."""
        move: chess.Move = self._robot_move_pending  # type: ignore[attr-defined]
        board = self.board
        driver = self._driver
        seq = self._sequencer

        assert driver is not None and seq is not None

        from_sq = chess.square_name(move.from_square)
        to_sq = chess.square_name(move.to_square)

        # Not physically supported: en-passant victim sits off to_sq; promotion needs piece swap.
        if board.is_en_passant(move) or move.promotion:
            self._enter_error(f"Unsupported robot move {move.uci()} (en-passant/promotion)")
            return

        try:
            if board.is_castling(move):
                # Determine rook squares
                king_sq = move.from_square
                if board.is_kingside_castling(move):
                    rook_from = chess.square_name(
                        chess.square(7, chess.square_rank(king_sq))
                    )
                    rook_to = chess.square_name(
                        chess.square(5, chess.square_rank(king_sq))
                    )
                    king_to = chess.square_name(
                        chess.square(6, chess.square_rank(king_sq))
                    )
                else:
                    rook_from = chess.square_name(
                        chess.square(0, chess.square_rank(king_sq))
                    )
                    rook_to = chess.square_name(
                        chess.square(3, chess.square_rank(king_sq))
                    )
                    king_to = chess.square_name(
                        chess.square(2, chess.square_rank(king_sq))
                    )
                seq.execute_castling(driver, from_sq, king_to, rook_from, rook_to)

            elif board.is_capture(move):
                # Captured piece color = color of piece on to_sq (it's White's since robot plays Black)
                captured_color = "white" if board.turn == chess.BLACK else "black"
                seq.execute_capture(driver, from_sq, to_sq, captured_color)

            else:
                seq.execute_normal(driver, from_sq, to_sq)

        except Exception as exc:
            self._enter_error(f"Robot move execution failed: {exc}")
            return

        # Commit move to board state
        self.board.push(move)
        self.last_robot_move = move
        log.info("Robot executed: %s", move.uci())

        if self.board.is_game_over():
            self._transition(ArenaState.GAME_OVER)
        else:
            self._transition(ArenaState.VERIFYING_BOARD)

    def _step_verify(self) -> None:
        """Verify physical board matches FEN via verify_callback."""
        ok = self._verify_cb(self.board)
        if ok:
            self._transition(ArenaState.WAIT_HUMAN_MOVE)
        else:
            log.warning("Board verification failed — entering error state")
            self._enter_error("Board physical state does not match expected FEN")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _transition(self, new_state: ArenaState) -> None:
        old = self._state
        self._state = new_state
        log.debug("FSM: %s -> %s", old.name, new_state.name)
        if self._on_state_change:
            try:
                self._on_state_change(old, new_state)
            except Exception:
                log.exception("on_state_change callback error (ignored)")

    def _enter_error(self, reason: str) -> None:
        self._error = reason
        log.error("FSM ERROR: %s", reason)
        # Safe stop: release gripper if driver available
        if self._driver:
            try:
                stopl = getattr(self._driver, "stopl", None)
                if stopl:
                    stopl()
                self._driver.set_digital_out(0, False)  # release gripper
            except Exception:
                log.exception("Safe stop error (ignored)")
        self._transition(ArenaState.ERROR_SAFE_STOP)

    @staticmethod
    def _detect_move(
        before: chess.Board,
        after: chess.Board,
    ) -> Optional[chess.Move]:
        """
        Infer a chess.Move by diffing occupied squares between before and after.

        Works for normal moves and captures.  Castling and en-passant detection
        falls back to legal move search when simple diff is ambiguous.
        """
        before_map: dict[int, chess.Piece] = {
            sq: p for sq in chess.SQUARES
            if (p := before.piece_at(sq)) is not None
        }
        after_map: dict[int, chess.Piece] = {
            sq: p for sq in chess.SQUARES
            if (p := after.piece_at(sq)) is not None
        }

        disappeared = [sq for sq in before_map if sq not in after_map or after_map[sq] != before_map[sq]]
        appeared    = [sq for sq in after_map  if sq not in before_map or after_map[sq] != before_map[sq]]

        # Filter disappeared to squares that actually lost a piece (or changed)
        lost_sqs = [sq for sq in chess.SQUARES
                    if before.piece_at(sq) is not None and after.piece_at(sq) != before.piece_at(sq)]
        gained_sqs = [sq for sq in chess.SQUARES
                      if after.piece_at(sq) is not None and after.piece_at(sq) != before.piece_at(sq)]

        # Try every legal move and see which one produces the after position
        for move in before.legal_moves:
            test = before.copy()
            test.push(move)
            if test.board_fen() == after.board_fen():
                return move

        return None
