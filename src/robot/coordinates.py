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

Feature Plane mode (UR Polyscope 3-point feature):
    The 3-point Feature defines a local coordinate frame whose origin is a1.
    Square positions are computed in that local frame then transformed to the
    UR base frame via a 3-D rigid-body rotation (Rodrigues formula) + translation,
    matching the math inside URScript pose_trans().

    Local frame:
        x_local = file_idx * pitch_x       (a→h direction)
        y_local = rank_idx * pitch_y       (1→8 direction)
        z_local = 0  (pick surface, or transit_clearance_z for approach)

    The feature plane pose [X, Y, Z, Rx, Ry, Rz] encodes the origin (a1 centre)
    and the rotation vector (axis * angle) of the board frame in the robot base frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Fixed downward-pointing tool orientation (rotation vector, radians)
_TOOL_RX: float = math.pi   # rad — TCP Z-axis antiparallel to base Z
_TOOL_RY: float = 0.0       # rad
_TOOL_RZ: float = 0.0       # rad

TRANSIT_Z_MIN: float = 0.080  # meters — hard floor for transit moves


# ---------------------------------------------------------------------------
# Rodrigues rotation helper
# ---------------------------------------------------------------------------

def _rotvec_to_matrix(rotvec: Tuple[float, ...]) -> List[List[float]]:
    """
    Convert a rotation vector (axis * angle, radians) to a 3x3 rotation matrix.

    Uses the Rodrigues formula, identical to the rotation embedded in
    URScript's pose_trans() internal math.

    Parameters
    ----------
    rotvec : (rx, ry, rz) — rotation vector, radians.

    Returns
    -------
    R : 3x3 row-major rotation matrix as list-of-lists.
    """
    rx, ry, rz = rotvec
    angle = math.sqrt(rx * rx + ry * ry + rz * rz)  # rad — rotation magnitude

    if angle < 1e-10:
        # Identity for near-zero rotation
        return [[1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]]

    # Unit axis
    kx, ky, kz = rx / angle, ry / angle, rz / angle

    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c  # 1 - cos(angle)

    # Rodrigues: R = I*cos + (1-cos)*k*k^T + sin*[k]x
    return [
        [t*kx*kx + c,     t*kx*ky - s*kz,  t*kx*kz + s*ky],
        [t*kx*ky + s*kz,  t*ky*ky + c,     t*ky*kz - s*kx],
        [t*kx*kz - s*ky,  t*ky*kz + s*kx,  t*kz*kz + c   ],
    ]


def _apply_rotation(R: List[List[float]], v: Tuple[float, ...]) -> Tuple[float, float, float]:
    """Multiply 3x3 matrix R by column vector v → (x, y, z)."""
    x = R[0][0]*v[0] + R[0][1]*v[1] + R[0][2]*v[2]
    y = R[1][0]*v[0] + R[1][1]*v[1] + R[1][2]*v[2]
    z = R[2][0]*v[0] + R[2][1]*v[1] + R[2][2]*v[2]
    return x, y, z


# ---------------------------------------------------------------------------
# Feature Plane configuration
# ---------------------------------------------------------------------------

@dataclass
class FeaturePlaneConfig:
    """
    UR Polyscope 3-point Feature Plane definition.

    The feature plane pose describes the local board coordinate frame expressed
    in the UR robot base frame, following the URScript convention:
        pose = [X, Y, Z, Rx, Ry, Rz]
        (X,Y,Z) : origin of the board frame in base frame, meters
        (Rx,Ry,Rz) : rotation vector (axis * angle), radians

    The local board frame has:
        +X along the a→h file direction
        +Y along the 1→8 rank direction
        +Z perpendicular to the board surface (outward normal)

    Parameters
    ----------
    pose : [X, Y, Z, Rx, Ry, Rz] — feature plane pose, meters and radians.
    pitch_x : square centre-to-centre distance along files (a→h), meters.
    pitch_y : square centre-to-centre distance along ranks (1→8), meters.
    pick_height_z : local Z offset when touching a piece top, meters.
                    Positive = away from board surface.
    transit_clearance_z : local Z offset for transit moves, meters.
                          Must be >= TRANSIT_Z_MIN when transformed to base frame.
    """

    pose: List[float]                   # [X, Y, Z, Rx, Ry, Rz] in base frame
    pitch_x: float = 0.055             # m — file spacing
    pitch_y: float = 0.055             # m — rank spacing
    pick_height_z: float = 0.010       # m — local Z at piece top
    transit_clearance_z: float = 0.120 # m — local Z for transit


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

    Two operating modes:

    **Base-frame mode** (default, ``feature_plane=None``):
        Square positions are computed directly in the UR base frame using a
        fixed origin (a1 XY) plus uniform dx/dy offsets.

    **Feature Plane mode** (``feature_plane`` given):
        Square positions are first expressed in the board's local coordinate
        frame, then transformed to the UR base frame via the Rodrigues rigid-body
        formula matching URScript ``pose_trans()``.  dx/dy/origin_xyz are ignored
        in this mode; spacing comes from ``feature_plane.pitch_x / pitch_y``.

    Parameters
    ----------
    origin_xyz : (X, Y, Z) of board square a1, meters, UR base frame.
                 Ignored when feature_plane is set.
    dx         : spacing between files (a→b), meters (positive = +X direction).
                 Ignored when feature_plane is set.
    dy         : spacing between ranks (1→2), meters (positive = +Y direction).
                 Ignored when feature_plane is set.
    board_z    : Z level of board surface, meters.
    pick_z     : Z level where TCP touches the piece for pick/place, meters.
                 Must be >= board_z.  Ignored when feature_plane is set
                 (pick height comes from feature_plane.pick_height_z instead).
    transit_z  : Z clearance for transit moves, meters. Clamped to >= TRANSIT_Z_MIN.
                 Ignored when feature_plane is set.
    envelope   : WorkspaceSafetyEnvelope instance (default envelope used if None).
    feature_plane : Optional FeaturePlaneConfig enabling Feature Plane mode.
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
    feature_plane: Optional[FeaturePlaneConfig] = None

    def __post_init__(self) -> None:
        if self.feature_plane is None:
            # Base-frame mode: validate flat-board geometry
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

        In Feature Plane mode the full 3-D transform is applied; only X and Y
        of the resulting base-frame position are returned.

        Parameters
        ----------
        square : algebraic notation, e.g. 'e4'.
        """
        if self.feature_plane is not None:
            pos = self._fp_local_to_base(square, z_local=0.0)
            return pos[0], pos[1]
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
        if self.feature_plane is not None:
            return self._fp_pose(square, z_local=self.feature_plane.transit_clearance_z)
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
        if self.feature_plane is not None:
            return self._fp_pose(square, z_local=self.feature_plane.pick_height_z)
        x, y = self.square_to_xy(square)
        pose = [x, y, self.pick_z, _TOOL_RX, _TOOL_RY, _TOOL_RZ]
        self._check(pose)
        return pose

    def transit_pose(self, square: str) -> List[float]:
        """Alias for approach_pose — same height, semantic clarity for moves."""
        return self.approach_pose(square)

    # ------------------------------------------------------------------
    # Feature Plane internal helpers
    # ------------------------------------------------------------------

    def _fp_local_to_base(
        self, square: str, z_local: float
    ) -> Tuple[float, float, float]:
        """
        Transform a square's local-frame position to the UR base frame.

        Steps (mirrors URScript pose_trans logic):
            1. Compute local position: (file*pitch_x, rank*pitch_y, z_local).
            2. Rotate into base frame using Rodrigues formula on feature plane
               rotation vector.
            3. Translate by feature plane origin.

        Parameters
        ----------
        square  : algebraic square, e.g. 'e4'.
        z_local : Z coordinate in the board local frame, meters.

        Returns
        -------
        (X, Y, Z) in UR base frame, meters.
        """
        assert self.feature_plane is not None  # guard; caller checks
        fp = self.feature_plane
        file_idx, rank_idx = self._parse_square(square)

        # Local board frame position, meters
        x_local = file_idx * fp.pitch_x
        y_local = rank_idx * fp.pitch_y

        # Rodrigues rotation: board local → robot base
        rotvec = (fp.pose[3], fp.pose[4], fp.pose[5])  # rad — axis*angle
        R = _rotvec_to_matrix(rotvec)
        dx_base, dy_base, dz_base = _apply_rotation(R, (x_local, y_local, z_local))

        # Translate by feature plane origin
        x_base = fp.pose[0] + dx_base  # meters
        y_base = fp.pose[1] + dy_base  # meters
        z_base = fp.pose[2] + dz_base  # meters
        return x_base, y_base, z_base

    def _fp_pose(self, square: str, z_local: float) -> List[float]:
        """
        Build a full 6-element TCP pose using Feature Plane transform.

        Tool orientation is inherited from the feature plane rotation vector
        so the TCP stays perpendicular to the board surface.

        Parameters
        ----------
        square  : algebraic square.
        z_local : board-local Z, meters.
        """
        assert self.feature_plane is not None
        fp = self.feature_plane
        x_base, y_base, z_base = self._fp_local_to_base(square, z_local)
        # TCP orientation matches the board frame orientation (TCP Z ⊥ board)
        pose = [x_base, y_base, z_base, fp.pose[3], fp.pose[4], fp.pose[5]]
        self._check(pose)
        # Transit safety: Z in base frame must meet TRANSIT_Z_MIN for approach moves
        if z_local >= fp.transit_clearance_z - 1e-9 and z_base < TRANSIT_Z_MIN:
            raise SafetyViolationError(
                f"Feature Plane transit Z={z_base:.4f} m < minimum {TRANSIT_Z_MIN:.4f} m"
            )
        return pose

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
