"""
Universal Robots (UR) Chess Robotics Package.

Provides drivers, coordinate transforms, trajectory generators, and graveyard
sequencing for automated chess pick-and-place operations.
"""

from .coordinates import (
    BoardCoordinates,
    SafetyViolationError,
    WorkspaceSafetyEnvelope,
)
from .graveyard import GraveyardManager, MoveSequencer
from .trajectory import TrajectoryGenerator, Waypoint
from .ur_driver import (
    DashboardClient,
    MockURDriver,
    RealURDriver,
    create_robot_driver,
)

__all__ = [
    "BoardCoordinates",
    "DashboardClient",
    "GraveyardManager",
    "MockURDriver",
    "MoveSequencer",
    "RealURDriver",
    "SafetyViolationError",
    "TrajectoryGenerator",
    "Waypoint",
    "WorkspaceSafetyEnvelope",
    "create_robot_driver",
]
