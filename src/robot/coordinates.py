"""
Board-to-robot frame coordinate mapping and workspace safety envelope.

Units:
    Positions  : meters (m)
    Angles     : radians (rad)
    Files (a-h): columns, index 0-7
    Ranks (1-8): rows,    index 0-7

Algebraic notation: file letter (a-h) + rank digit (1-8), e.g. 'e4'.

Fixed tool orientation: pointing straight down.
    Rx = pi, Ry = 0, Rz = 0  (rotation vector, UR convention)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

# Fixed downward-pointing tool orientation (rotation vector, radians)
_TOOL_RX: float = math.pi   # rad — TCP Z-axis antiparallel to base Z
_TOOL_RY: float = 0.0       # rad
_TOOL_RZ: float = 0.0       # rad

TRANSIT_Z_MIN: float = 0.080  # meters — hard floor for transit moves


class SafetyViolationError(Exception):
    """Raised when a computed pose exceeds workspace safety bounds."""


# ---------------------------------------------------------------------------
# Workspace safety envelope
# ---------------------------------------------------------------------------

@dataclass
class WorkspaceSafetyEnvelope:
    """
    Axis-aligned bounding box in UR base frame (meters).

    Call check(pose) before sending any pose to hardware.
    Raises SafetyViolationError on violation.
    """

    x_min: float = -0.80   # meters
    x_max: float =  0.80   # meters
    y_min: float = -0.80   # meters
    y_max: float =  0.80   # meters
    z_min: float = -0.20   # meters — floor (prevent driving into table)
    z_max: float =  0.90   # meters — ceiling

    def check(self, pose: List[float]) -> None:
        """
        Validate a 6-element TCP pose [X, Y, Z, Rx, Ry, Rz] (meters/radians).

        Raises SafetyViolationError if any Cartesian coordinate is out of bounds.
        """
        if len(pose) < 3:
            raise ValueError(f"pose must have >= 3 elements, got {len(pose)}")
        x, y, z = pose[0], pose[1], pose[2]
        if not (self.x_min <= x <= self.x_max):
            raise SafetyViolationError(
                f"X={x:.4f} m violates envelope [{self.x_min}, {self.x_max}]"
            )
        if not (self.y_min <= y <= self.y_max):
            raise SafetyViolationError(
                f"Y={y:.4f} m violates envelope [{self.y_min}, {self.y_max}]"
            )
        if not (self.z_min <= z <= self.z_max):
            raise SafetyViolationError(
                f"Z={z:.4f} m violates envelope [{self.z_min}, {self.z_max}]"
            )


# ---------------------------------------------------------------------------
# Board coordinate mapper
# ---------------------------------------------------------------------------

@dataclass
class BoardCoordinates:
    """
    Maps chess algebraic squares ('a1'..'h8') to UR TCP poses.

    Parameters
    ----------
    origin_xyz : (X, Y, Z) of board square a1, meters, UR base frame.
    dx         : spacing between files (a→b), meters (positive = +X direction).
    dy         : spacing between ranks (1→2), meters (positive = +Y direction).
    board_z    : Z level of board surface, meters.
    pick_z     : Z level where TCP touches the piece for pick/place, meters.
                 Must be >= board_z.
    transit_z  : Z clearance for transit moves, meters. Clamped to >= TRANSIT_Z_MIN.
    envelope   : WorkspaceSafetyEnvelope instance (default envelope used if None).
    """

    origin_xyz: Tuple[float, float, float]  # (X, Y, Z) of a1, meters
    dx: float                                # file spacing, meters
    dy: float                                # rank spacing, meters
    board_z: float                           # board surface Z, meters
    pick_z: float                            # pick/place Z, meters
    transit_z: float = 0.120                 # transit clearance Z, meters
    envelope: WorkspaceSafetyEnvelope = field(
        default_factory=WorkspaceSafetyEnvelope
    )

    def __post_init__(self) -> None:
        if self.transit_z < TRANSIT_Z_MIN:
            raise ValueError(
                f"transit_z={self.transit_z:.4f} m < minimum {TRANSIT_Z_MIN:.4f} m"
            )
        if self.pick_z < self.board_z:
            raise ValueError(
                f"pick_z={self.pick_z:.4f} m < board_z={self.board_z:.4f} m"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def square_to_xy(self, square: str) -> Tuple[float, float]:
        """
        Return (X, Y) of square center in UR base frame (meters).

        Parameters
        ----------
        square : algebraic notation, e.g. 'e4'.
        """
        file_idx, rank_idx = self._parse_square(square)
        x = self.origin_xyz[0] + file_idx * self.dx  # meters
        y = self.origin_xyz[1] + rank_idx * self.dy  # meters
        return x, y

    def approach_pose(self, square: str) -> List[float]:
        """
        TCP pose above the square at transit_z (safe transit height).

        Returns [X, Y, Z, Rx, Ry, Rz] in meters / radians.
        Validated against envelope before return.
        """
        x, y = self.square_to_xy(square)
        if self.transit_z < TRANSIT_Z_MIN:  # fields are mutable; re-check at call time
            raise SafetyViolationError(
                f"transit_z={self.transit_z:.4f} m < minimum {TRANSIT_Z_MIN:.4f} m"
            )
        pose = [x, y, self.transit_z, _TOOL_RX, _TOOL_RY, _TOOL_RZ]
        self._check(pose)
        return pose

    def pick_pose(self, square: str) -> List[float]:
        """
        TCP pose at pick/place height for the given square.

        Returns [X, Y, Z, Rx, Ry, Rz] in meters / radians.
        Validated against envelope before return.
        """
        x, y = self.square_to_xy(square)
        pose = [x, y, self.pick_z, _TOOL_RX, _TOOL_RY, _TOOL_RZ]
        self._check(pose)
        return pose

    def transit_pose(self, square: str) -> List[float]:
        """Alias for approach_pose — same height, semantic clarity for moves."""
        return self.approach_pose(square)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check(self, pose: List[float]) -> None:
        """Envelope check plus board-surface floor (re-checked at call time; fields are mutable)."""
        self.envelope.check(pose)
        if pose[2] < self.board_z:
            raise SafetyViolationError(
                f"Z={pose[2]:.4f} m below board surface {self.board_z:.4f} m"
            )

    @staticmethod
    def _parse_square(square: str) -> Tuple[int, int]:
        """
        Parse algebraic square string to (file_index, rank_index), both 0-based.

        'a1' -> (0, 0), 'h8' -> (7, 7).
        Raises ValueError on invalid input.
        """
        if len(square) != 2:
            raise ValueError(f"Invalid square: {square!r}")
        file_char = square[0].lower()
        rank_char = square[1]
        if file_char not in 'abcdefgh':
            raise ValueError(f"Invalid file in square: {square!r}")
        if rank_char not in '12345678':
            raise ValueError(f"Invalid rank in square: {square!r}")
        file_idx = ord(file_char) - ord('a')  # 0..7
        rank_idx = int(rank_char) - 1          # 0..7
        return file_idx, rank_idx
