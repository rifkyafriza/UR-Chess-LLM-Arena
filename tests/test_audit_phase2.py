"""Regression tests for Phase 2 audit fixes (reviewer)."""
import pytest
from src.robot.coordinates import BoardCoordinates, SafetyViolationError
from src.robot.trajectory import TrajectoryGenerator
from src.robot.ur_driver import MockURDriver, _check_pose_limits


def _bc():
    return BoardCoordinates((0.0, 0.2, 0.1), 0.05, 0.05, board_z=0.1, pick_z=0.1)


def test_mutated_transit_z_rejected():
    c = _bc(); c.transit_z = 0.01
    with pytest.raises(SafetyViolationError):
        c.approach_pose("a1")


def test_pick_below_board_rejected():
    c = _bc(); c.pick_z = 0.0
    with pytest.raises(SafetyViolationError):
        c.pick_pose("a1")


def test_driver_rejects_short_pose():
    with pytest.raises(ValueError):
        _check_pose_limits([0.0, 0.0, 0.1])


def test_trajectory_stops_on_failure():
    class D(MockURDriver):
        stopped = False
        def movel(self, pose, a=0, v=0): raise RuntimeError("boom")
        def stopl(self, a=10.0): D.stopped = True
    with pytest.raises(RuntimeError):
        TrajectoryGenerator(_bc()).pick_and_place(D(), "a1", "b2")
    assert D.stopped
