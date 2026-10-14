"""
Graveyard slot manager and move sequencer for captured chess pieces.

Layout
------
White pieces (captured from Black) → left  side of board, 16 slots.
Black pieces (captured from White) → right side of board, 16 slots.

Each slot is a fixed Cartesian pose [X, Y, Z, Rx, Ry, Rz] in the UR base frame.
Slots are filled in order; overflow raises GraveyardFullError.

Coordinates are computed from:
    - graveyard_origin_white : (X, Y, Z) of white graveyard slot 0  (meters)
    - graveyard_origin_black : (X, Y, Z) of black graveyard slot 0  (meters)
    - slot_spacing           : distance between consecutive slots     (meters)
    - slot_axis              : 'x' or 'y'  — axis along which slots are laid out

The graveyard poses are independent of BoardCoordinates; they live in UR base frame.

Units:
    Positions  : meters (m)
    Angles     : radians (rad)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Protocol, Tuple

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structural protocols (avoid circular imports)
# ---------------------------------------------------------------------------

from .coordinates import BoardCoordinates


# ---------------------------------------------------------------------------
# Structural protocols
# ---------------------------------------------------------------------------

class _DriverProto(Protocol):
    def movel(self, pose: List[float], a: float, v: float) -> None: ...
    def set_digital_out(self, port: int, value: bool) -> None: ...


class _EnvelopeProto(Protocol):
    def check(self, pose: List[float]) -> None: ...


class _TrajectoryProto(Protocol):
    coords: BoardCoordinates
    park_pose: List[float]
    vel_transit: float
    acc_transit: float
    vel_vert: float
    acc_vert: float
    vel_slow: float
    acc_slow: float
    def pick_and_place(self, driver: _DriverProto, from_sq: str, to_sq: str) -> None: ...

# Fixed downward-pointing tool orientation (mirrors coordinates.py convention)
_TOOL_RX: float = math.pi  # rad
_TOOL_RY: float = 0.0      # rad
_TOOL_RZ: float = 0.0      # rad

_MAX_PIECES: int = 16  # max captured pieces per side (all non-king pieces)


class GraveyardFullError(Exception):
    """Raised when a graveyard side has no remaining slots."""


# ---------------------------------------------------------------------------
# Graveyard manager
# ---------------------------------------------------------------------------

@dataclass
class GraveyardManager:
    """
    Tracks and allocates physical graveyard slots for captured pieces.

    Parameters
    ----------
    origin_white  : (X, Y, Z) meters — first white-side slot in UR base frame.
    origin_black  : (X, Y, Z) meters — first black-side slot in UR base frame.
    slot_spacing  : meters between consecutive slots (positive direction).
    slot_axis     : 'x' or 'y' — axis along which slots advance.
    envelope      : optional WorkspaceSafetyEnvelope for pose validation.
                    Pass None to skip envelope checks on graveyard poses.
    """

    origin_white: Tuple[float, float, float]   # meters, UR base frame
    origin_black: Tuple[float, float, float]   # meters, UR base frame
    slot_spacing: float                         # meters
    slot_axis: str = "y"                        # 'x' or 'y'
    envelope: _EnvelopeProto | None = None      # WorkspaceSafetyEnvelope | None

    # Internal slot counters (filled slots per side)
    _white_count: int = field(default=0, init=False, repr=False)
    _black_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.slot_axis not in ("x", "y"):
            raise ValueError(f"slot_axis must be 'x' or 'y', got {self.slot_axis!r}")
        if self.slot_spacing <= 0:
            raise ValueError(f"slot_spacing must be positive, got {self.slot_spacing}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def next_white_slot(self) -> List[float]:
        """
        Return the next available white-side graveyard pose and advance counter.
        Raises GraveyardFullError if all 16 slots are occupied.
        """
        return self._next_slot("white")

    def next_black_slot(self) -> List[float]:
        """
        Return the next available black-side graveyard pose and advance counter.
        Raises GraveyardFullError if all 16 slots are occupied.
        """
        return self._next_slot("black")

    def reset(self, side: str = "both") -> None:
        """
        Reset slot counters.  side: 'white', 'black', or 'both'.
        """
        if side in ("white", "both"):
            self._white_count = 0
        if side in ("black", "both"):
            self._black_count = 0
        if side not in ("white", "black", "both"):
            raise ValueError(f"side must be 'white', 'black', or 'both', got {side!r}")

    @property
    def white_slots_used(self) -> int:
        return self._white_count

    @property
    def black_slots_used(self) -> int:
        return self._black_count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_slot(self, side: str) -> List[float]:
        if side == "white":
            count = self._white_count
            origin = self.origin_white
        else:
            count = self._black_count
            origin = self.origin_black

        if count >= _MAX_PIECES:
            raise GraveyardFullError(
                f"{side} graveyard is full ({_MAX_PIECES} pieces)"
            )

        pose = self._slot_pose(origin, count)

        if self.envelope is not None:
            self.envelope.check(pose)

        if side == "white":
            self._white_count += 1
        else:
            self._black_count += 1

        log.debug("GraveyardManager %s slot %d -> %s", side, count, pose)
        return pose

    def _slot_pose(
        self,
        origin: Tuple[float, float, float],
        index: int,
    ) -> List[float]:
        """Compute (X, Y, Z, Rx, Ry, Rz) for graveyard slot at given index."""
        x, y, z = origin
        if self.slot_axis == "x":
            x += index * self.slot_spacing
        else:  # 'y'
            y += index * self.slot_spacing
        return [x, y, z, _TOOL_RX, _TOOL_RY, _TOOL_RZ]


# ---------------------------------------------------------------------------
# Move sequencer
# ---------------------------------------------------------------------------

@dataclass
class MoveSequencer:
    """
    High-level move sequencer: translates chess moves into robot action sequences.

    Supported move types
    --------------------
    normal   : one pick-and-place (from_sq → to_sq)
    capture  : two-stage
                 1. victim (to_sq) → graveyard
                 2. attacker (from_sq) → to_sq
    castling : two-stage
                 1. King move  (king_from → king_to)
                 2. Rook move  (rook_from → rook_to)

    Parameters
    ----------
    trajectory : TrajectoryGenerator — executes pick-and-place.
    graveyard  : GraveyardManager    — allocates capture slots.
    """

    trajectory: _TrajectoryProto  # TrajectoryGenerator
    graveyard: GraveyardManager

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute_normal(
        self,
        driver: _DriverProto,
        from_sq: str,
        to_sq: str,
    ) -> None:
        """
        Normal move: pick piece from from_sq, place on to_sq.

        Parameters
        ----------
        driver  : robot driver (RealURDriver or MockURDriver)
        from_sq : source square, e.g. 'e2'
        to_sq   : destination square, e.g. 'e4'
        """
        log.info("MoveSequencer normal %s -> %s", from_sq, to_sq)
        self.trajectory.pick_and_place(driver, from_sq, to_sq)

    def execute_capture(
        self,
        driver: _DriverProto,
        from_sq: str,
        to_sq: str,
        captured_color: str,
    ) -> None:
        """
        Capture move (2-stage):
            Stage 1: move victim from to_sq to graveyard.
            Stage 2: move attacker from from_sq to to_sq.

        Parameters
        ----------
        driver          : robot driver
        from_sq         : attacker's source square
        to_sq           : destination square (also victim's current square)
        captured_color  : color of the captured piece — 'white' or 'black'
                          (determines which graveyard side receives the piece)
        """
        if captured_color not in ("white", "black"):
            raise ValueError(
                f"captured_color must be 'white' or 'black', got {captured_color!r}"
            )

        # Allocate graveyard slot for captured piece
        if captured_color == "white":
            grave_pose = self.graveyard.next_white_slot()
        else:
            grave_pose = self.graveyard.next_black_slot()

        log.info(
            "MoveSequencer capture %s -> graveyard, %s -> %s",
            to_sq, from_sq, to_sq,
        )

        # Stage 1: victim → graveyard (place at fixed pose, not a board square)
        self._pick_to_pose(driver, to_sq, grave_pose)

        # Stage 2: attacker → destination
        self.trajectory.pick_and_place(driver, from_sq, to_sq)

    def execute_castling(
        self,
        driver: _DriverProto,
        king_from: str,
        king_to: str,
        rook_from: str,
        rook_to: str,
    ) -> None:
        """
        Castling move (2-stage):
            Stage 1: King from_sq → to_sq.
            Stage 2: Rook from_sq → to_sq.

        Parameters
        ----------
        driver     : robot driver
        king_from  : King's source square (e.g. 'e1')
        king_to    : King's destination   (e.g. 'g1' for kingside)
        rook_from  : Rook's source square (e.g. 'h1')
        rook_to    : Rook's destination   (e.g. 'f1')
        """
        log.info(
            "MoveSequencer castling K:%s->%s R:%s->%s",
            king_from, king_to, rook_from, rook_to,
        )
        # Stage 1: King
        self.trajectory.pick_and_place(driver, king_from, king_to)
        # Stage 2: Rook
        self.trajectory.pick_and_place(driver, rook_from, rook_to)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick_to_pose(
        self,
        driver: _DriverProto,
        from_sq: str,
        to_pose: List[float],
    ) -> None:
        """
        Pick piece from from_sq and place at an arbitrary Cartesian pose.

        Used for capture: victim → graveyard pose (not a board square).
        Re-uses TrajectoryGenerator internals via its coords attribute.
        """
        tg = self.trajectory
        coords = tg.coords

        approach_src = coords.approach_pose(from_sq)
        pick_src     = coords.pick_pose(from_sq)

        # Graveyard approach: same X,Y as to_pose but at transit_z
        grave_approach = [
            to_pose[0],
            to_pose[1],
            coords.transit_z,
            to_pose[3],
            to_pose[4],
            to_pose[5],
        ]
        coords.envelope.check(grave_approach)
        coords.envelope.check(to_pose)  # validate slot pose BEFORE any motion

        log.debug("_pick_to_pose %s -> graveyard pose %s", from_sq, to_pose)

        # Approach source
        driver.movel(approach_src, a=tg.acc_transit, v=tg.vel_transit)
        # Descend to piece
        driver.movel(pick_src, a=tg.acc_slow, v=tg.vel_slow)
        # Grasp
        driver.set_digital_out(0, True)
        # Ascend
        driver.movel(approach_src, a=tg.acc_vert, v=tg.vel_vert)
        # Transit to graveyard approach
        driver.movel(grave_approach, a=tg.acc_transit, v=tg.vel_transit)
        # Descend to graveyard slot
        driver.movel(to_pose, a=tg.acc_slow, v=tg.vel_slow)
        # Release
        driver.set_digital_out(0, False)
        # Retract to graveyard approach height
        driver.movel(grave_approach, a=tg.acc_vert, v=tg.vel_vert)
        # Park
        driver.movel(list(tg.park_pose), a=tg.acc_transit, v=tg.vel_transit)
