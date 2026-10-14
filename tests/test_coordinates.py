"""
Unit tests for src/robot/coordinates.py.

Coverage:
- BoardCoordinates._parse_square: valid / invalid inputs
- BoardCoordinates.square_to_xy: corner squares, interior squares
- BoardCoordinates.approach_pose / pick_pose: Z values, orientation vector
- BoardCoordinates.__post_init__: reject bad transit_z, pick_z
- WorkspaceSafetyEnvelope.check: pass inside, raise on each axis violation
- SafetyViolationError raised through BoardCoordinates when pose out of bounds
"""

import math
import pytest

from src.robot.coordinates import (
    BoardCoordinates,
    SafetyViolationError,
    TRANSIT_Z_MIN,
    WorkspaceSafetyEnvelope,
    _TOOL_RX,
    _TOOL_RY,
    _TOOL_RZ,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ORIGIN = (0.30, -0.45, 0.010)   # a1 in robot base frame (meters)
DX = 0.050   # meters per file
DY = 0.050   # meters per rank
BOARD_Z = 0.010   # meters — board surface
PICK_Z = 0.005    # meters — just below surface (simulates contact)
TRANSIT_Z = 0.120  # meters


@pytest.fixture
def bc() -> BoardCoordinates:
    return BoardCoordinates(
        origin_xyz=ORIGIN,
        dx=DX,
        dy=DY,
        board_z=BOARD_Z,
        pick_z=BOARD_Z,       # touch surface
        transit_z=TRANSIT_Z,
    )


# ---------------------------------------------------------------------------
# _parse_square
# ---------------------------------------------------------------------------

class TestParseSquare:
    def test_a1(self):
        assert BoardCoordinates._parse_square("a1") == (0, 0)

    def test_h8(self):
        assert BoardCoordinates._parse_square("h8") == (7, 7)

    def test_e4(self):
        assert BoardCoordinates._parse_square("e4") == (4, 3)

    def test_b2(self):
        assert BoardCoordinates._parse_square("b2") == (1, 1)

    def test_uppercase(self):
        assert BoardCoordinates._parse_square("E4") == (4, 3)

    def test_invalid_too_long(self):
        with pytest.raises(ValueError):
            BoardCoordinates._parse_square("e44")

    def test_invalid_file(self):
        with pytest.raises(ValueError):
            BoardCoordinates._parse_square("z4")

    def test_invalid_rank(self):
        with pytest.raises(ValueError):
            BoardCoordinates._parse_square("e9")

    def test_invalid_rank_zero(self):
        with pytest.raises(ValueError):
            BoardCoordinates._parse_square("e0")


# ---------------------------------------------------------------------------
# square_to_xy
# ---------------------------------------------------------------------------

class TestSquareToXY:
    def test_a1_is_origin(self, bc):
        x, y = bc.square_to_xy("a1")
        assert x == pytest.approx(ORIGIN[0])
        assert y == pytest.approx(ORIGIN[1])

    def test_b1_one_file_right(self, bc):
        x, y = bc.square_to_xy("b1")
        assert x == pytest.approx(ORIGIN[0] + DX)
        assert y == pytest.approx(ORIGIN[1])

    def test_a2_one_rank_up(self, bc):
        x, y = bc.square_to_xy("a2")
        assert x == pytest.approx(ORIGIN[0])
        assert y == pytest.approx(ORIGIN[1] + DY)

    def test_h8_corner(self, bc):
        x, y = bc.square_to_xy("h8")
        assert x == pytest.approx(ORIGIN[0] + 7 * DX)
        assert y == pytest.approx(ORIGIN[1] + 7 * DY)

    def test_e4_center(self, bc):
        x, y = bc.square_to_xy("e4")
        assert x == pytest.approx(ORIGIN[0] + 4 * DX)
        assert y == pytest.approx(ORIGIN[1] + 3 * DY)


# ---------------------------------------------------------------------------
# approach_pose
# ---------------------------------------------------------------------------

class TestApproachPose:
    def test_z_is_transit(self, bc):
        pose = bc.approach_pose("e4")
        assert pose[2] == pytest.approx(TRANSIT_Z)

    def test_orientation_downward(self, bc):
        pose = bc.approach_pose("e4")
        assert pose[3] == pytest.approx(_TOOL_RX)
        assert pose[4] == pytest.approx(_TOOL_RY)
        assert pose[5] == pytest.approx(_TOOL_RZ)

    def test_pose_length(self, bc):
        pose = bc.approach_pose("a1")
        assert len(pose) == 6

    def test_transit_alias(self, bc):
        assert bc.transit_pose("d5") == bc.approach_pose("d5")


# ---------------------------------------------------------------------------
# pick_pose
# ---------------------------------------------------------------------------

class TestPickPose:
    def test_z_is_pick_z(self, bc):
        pose = bc.pick_pose("e4")
        assert pose[2] == pytest.approx(BOARD_Z)

    def test_xy_matches_approach(self, bc):
        approach = bc.approach_pose("e4")
        pick = bc.pick_pose("e4")
        assert pick[0] == pytest.approx(approach[0])
        assert pick[1] == pytest.approx(approach[1])


# ---------------------------------------------------------------------------
# __post_init__ validation
# ---------------------------------------------------------------------------

class TestPostInit:
    def test_transit_z_below_minimum_raises(self):
        with pytest.raises(ValueError, match="transit_z"):
            BoardCoordinates(
                origin_xyz=ORIGIN, dx=DX, dy=DY,
                board_z=BOARD_Z, pick_z=BOARD_Z,
                transit_z=TRANSIT_Z_MIN - 0.001,  # too low
            )

    def test_transit_z_exactly_minimum_ok(self):
        bc = BoardCoordinates(
            origin_xyz=ORIGIN, dx=DX, dy=DY,
            board_z=BOARD_Z, pick_z=BOARD_Z,
            transit_z=TRANSIT_Z_MIN,
        )
        assert bc.transit_z == pytest.approx(TRANSIT_Z_MIN)

    def test_pick_z_below_board_z_raises(self):
        with pytest.raises(ValueError, match="pick_z"):
            BoardCoordinates(
                origin_xyz=ORIGIN, dx=DX, dy=DY,
                board_z=0.010, pick_z=0.005,   # pick below board
                transit_z=TRANSIT_Z,
            )


# ---------------------------------------------------------------------------
# WorkspaceSafetyEnvelope
# ---------------------------------------------------------------------------

class TestWorkspaceSafetyEnvelope:
    @pytest.fixture
    def env(self):
        return WorkspaceSafetyEnvelope(
            x_min=0.0, x_max=0.5,
            y_min=-0.5, y_max=0.0,
            z_min=0.0, z_max=0.5,
        )

    def test_valid_pose_no_raise(self, env):
        env.check([0.25, -0.25, 0.25, math.pi, 0.0, 0.0])

    def test_x_below_min_raises(self, env):
        with pytest.raises(SafetyViolationError, match="X="):
            env.check([-0.01, -0.25, 0.25, 0.0, 0.0, 0.0])

    def test_x_above_max_raises(self, env):
        with pytest.raises(SafetyViolationError, match="X="):
            env.check([0.51, -0.25, 0.25, 0.0, 0.0, 0.0])

    def test_y_below_min_raises(self, env):
        with pytest.raises(SafetyViolationError, match="Y="):
            env.check([0.25, -0.51, 0.25, 0.0, 0.0, 0.0])

    def test_y_above_max_raises(self, env):
        with pytest.raises(SafetyViolationError, match="Y="):
            env.check([0.25, 0.01, 0.25, 0.0, 0.0, 0.0])

    def test_z_below_min_raises(self, env):
        with pytest.raises(SafetyViolationError, match="Z="):
            env.check([0.25, -0.25, -0.01, 0.0, 0.0, 0.0])

    def test_z_above_max_raises(self, env):
        with pytest.raises(SafetyViolationError, match="Z="):
            env.check([0.25, -0.25, 0.51, 0.0, 0.0, 0.0])

    def test_boundary_exactly_at_min_ok(self, env):
        env.check([0.0, -0.5, 0.0, 0.0, 0.0, 0.0])

    def test_boundary_exactly_at_max_ok(self, env):
        env.check([0.5, 0.0, 0.5, 0.0, 0.0, 0.0])

    def test_too_short_pose_raises_value_error(self, env):
        with pytest.raises(ValueError):
            env.check([0.1, 0.1])


# ---------------------------------------------------------------------------
# Safety envelope integrated with BoardCoordinates
# ---------------------------------------------------------------------------

class TestEnvelopeIntegration:
    def test_out_of_bounds_origin_raises_on_approach(self):
        # Place board far outside default envelope
        bc = BoardCoordinates(
            origin_xyz=(2.0, 0.0, 0.010),  # X=2.0 m — way outside ±0.80 m
            dx=DX, dy=DY,
            board_z=0.010, pick_z=0.010,
            transit_z=TRANSIT_Z,
        )
        with pytest.raises(SafetyViolationError):
            bc.approach_pose("a1")

    def test_out_of_bounds_origin_raises_on_pick(self):
        bc = BoardCoordinates(
            origin_xyz=(0.0, 2.0, 0.010),  # Y=2.0 m
            dx=DX, dy=DY,
            board_z=0.010, pick_z=0.010,
            transit_z=TRANSIT_Z,
        )
        with pytest.raises(SafetyViolationError):
            bc.pick_pose("a1")

    def test_valid_board_passes_all_squares(self):
        bc = BoardCoordinates(
            origin_xyz=ORIGIN,
            dx=DX, dy=DY,
            board_z=BOARD_Z, pick_z=BOARD_Z,
            transit_z=TRANSIT_Z,
        )
        files = 'abcdefgh'
        ranks = '12345678'
        for f in files:
            for r in ranks:
                square = f + r
                bc.approach_pose(square)   # must not raise
                bc.pick_pose(square)       # must not raise
