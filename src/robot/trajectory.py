"""
Trajectory waypoint generator for chess pick-and-place operations.

Units:
    Positions  : meters (m)
    Angles     : radians (rad)
    Velocity   : m/s
    Accel      : m/s^2

Gripper control uses controller digital output 0:
    True  = vacuum/grip ON  (grasp piece)
    False = vacuum/grip OFF (release piece)

Park pose: safe home position above and behind the board,
           clear of all play squares.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Protocol, Sequence

from .coordinates import BoardCoordinates

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gripper digital output port
# ---------------------------------------------------------------------------
GRIPPER_DO_PORT: int = 0  # controller digital output index

# ---------------------------------------------------------------------------
# Default motion parameters
# ---------------------------------------------------------------------------
_VEL_TRANSIT: float = 0.30   # m/s — horizontal transit speed
_ACC_TRANSIT: float = 1.20   # m/s^2
_VEL_VERT: float = 0.10      # m/s — vertical pick/place speed
_ACC_VERT: float = 0.60      # m/s^2
_VEL_SLOW: float = 0.05      # m/s — final descent / ascent near piece
_ACC_SLOW: float = 0.30      # m/s^2

# Default park pose [X, Y, Z, Rx, Ry, Rz] in UR base frame (meters, radians).
# Tuned so the robot is well clear of the chess board while idle.
DEFAULT_PARK_POSE: List[float] = [0.0, -0.60, 0.35, 3.14159, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Driver protocol (structural, avoids circular import with ur_driver)
# ---------------------------------------------------------------------------

class _DriverProto(Protocol):
    def movel(self, pose: List[float], a: float, v: float) -> None: ...
    def set_digital_out(self, port: int, value: bool) -> None: ...


# ---------------------------------------------------------------------------
# Waypoint dataclass
# ---------------------------------------------------------------------------

@dataclass
class Waypoint:
    """A single commanded robot pose with motion parameters."""

    pose: List[float]          # [X, Y, Z, Rx, Ry, Rz]  meters / radians
    vel: float                 # m/s
    acc: float                 # m/s^2
    label: str = ""            # human-readable step description


# ---------------------------------------------------------------------------
# Trajectory generator
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryGenerator:
    """
    Generate collision-free pick-and-place waypoint sequences.

    Motion profile (per pick-and-place):
        1. Move to approach pose above source (transit_z height)
        2. Descend to pick_z
        3. Grasp (set_digital_out(0, True))
        4. Ascend back to transit_z
        5. Transit horizontally to approach pose above destination
        6. Descend to pick_z
        7. Release (set_digital_out(0, False))
        8. Retract to safe park pose

    All poses validated by BoardCoordinates.envelope before motion.

    Parameters
    ----------
    coords     : BoardCoordinates — maps squares to poses, enforces envelope.
    park_pose  : safe idle pose after each sequence (default: DEFAULT_PARK_POSE).
    vel_transit: horizontal transit velocity  m/s
    acc_transit: horizontal transit accel     m/s^2
    vel_vert   : vertical raise/lower velocity m/s
    acc_vert   : vertical raise/lower accel   m/s^2
    vel_slow   : slow final descent/ascent    m/s
    acc_slow   : slow final descent/ascent    m/s^2
    """

    coords: BoardCoordinates
    park_pose: List[float] = field(default_factory=lambda: list(DEFAULT_PARK_POSE))
    vel_transit: float = _VEL_TRANSIT
    acc_transit: float = _ACC_TRANSIT
    vel_vert: float = _VEL_VERT
    acc_vert: float = _ACC_VERT
    vel_slow: float = _VEL_SLOW
    acc_slow: float = _ACC_SLOW

    def __post_init__(self) -> None:
        # Validate park pose against envelope at construction time.
        self.coords.envelope.check(self.park_pose)

    # ------------------------------------------------------------------
    # Public high-level API
    # ------------------------------------------------------------------

    def pick_and_place(
        self,
        driver: _DriverProto,
        from_sq: str,
        to_sq: str,
    ) -> None:
        """
        Execute a single pick-and-place: lift piece from from_sq, place on to_sq.

        Parameters
        ----------
        driver  : robot driver (RealURDriver or MockURDriver)
        from_sq : source algebraic square, e.g. 'e2'
        to_sq   : destination algebraic square, e.g. 'e4'
        """
        waypoints = self.plan_pick_and_place(from_sq, to_sq)
        self._execute(driver, waypoints, from_sq, to_sq)

    def plan_pick_and_place(
        self,
        from_sq: str,
        to_sq: str,
    ) -> List[Waypoint]:
        """
        Return the ordered list of Waypoints for a pick-and-place.
        Does not communicate with hardware.
        """
        approach_src = self.coords.approach_pose(from_sq)  # transit height above source
        pick_src     = self.coords.pick_pose(from_sq)       # contact height at source
        approach_dst = self.coords.approach_pose(to_sq)     # transit height above destination
        pick_dst     = self.coords.pick_pose(to_sq)         # contact height at destination

        return [
            Waypoint(approach_src, self.vel_transit, self.acc_transit,
                     f"approach_src:{from_sq}"),
            Waypoint(pick_src,     self.vel_slow,    self.acc_slow,
                     f"pick:{from_sq}"),
            # Grasp is a digital-out command, not a Waypoint; emitted in _execute.
            Waypoint(approach_src, self.vel_vert,    self.acc_vert,
                     f"ascend_src:{from_sq}"),
            Waypoint(approach_dst, self.vel_transit, self.acc_transit,
                     f"transit_to:{to_sq}"),
            Waypoint(pick_dst,     self.vel_slow,    self.acc_slow,
                     f"place:{to_sq}"),
            # Release is a digital-out command; emitted in _execute.
            Waypoint(approach_dst, self.vel_vert,    self.acc_vert,
                     f"retract_dst:{to_sq}"),
            Waypoint(list(self.park_pose), self.vel_transit, self.acc_transit,
                     "park"),
        ]

    # ------------------------------------------------------------------
    # Internal execution engine
    # ------------------------------------------------------------------

    def _execute(
        self,
        driver: _DriverProto,
        waypoints: List[Waypoint],
        from_sq: str,
        to_sq: str,
    ) -> None:
        """
        Walk through waypoints and fire gripper commands at the right steps.

        Gripper events:
            After  pick pose    → grasp  (DO 0 = True)
            After  place pose   → release (DO 0 = False)
        """
        PICK_LABEL_PREFIX  = "pick:"
        PLACE_LABEL_PREFIX = "place:"

        try:
            for wp in waypoints:
                log.debug("TrajectoryGenerator movel label=%r pose=%s", wp.label, wp.pose)
                driver.movel(wp.pose, a=wp.acc, v=wp.vel)

                if wp.label.startswith(PICK_LABEL_PREFIX):
                    log.debug("Gripper GRASP (DO%d=True)", GRIPPER_DO_PORT)
                    driver.set_digital_out(GRIPPER_DO_PORT, True)

                elif wp.label.startswith(PLACE_LABEL_PREFIX):
                    log.debug("Gripper RELEASE (DO%d=False)", GRIPPER_DO_PORT)
                    driver.set_digital_out(GRIPPER_DO_PORT, False)
        except Exception:
            # Fail safe: halt motion; keep vacuum state so a held piece is not dropped.
            log.exception("Trajectory aborted at %s->%s; sending stopl", from_sq, to_sq)
            stop = getattr(driver, "stopl", None)
            if stop:
                try:
                    stop()
                except Exception:
                    log.exception("stopl failed")
            raise
