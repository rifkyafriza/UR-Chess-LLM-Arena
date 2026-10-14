"""
Unit tests for Feature Plane support in src/robot/coordinates.py
and the calibration math in scripts/calibrate_feature.py.

Test cases use exact analytic results computed by hand so the test
can catch any regression in the Rodrigues / matrix-to-rotvec pipeline.

Coverage:
- _rotvec_to_matrix: identity, 90-deg rotations, 180-deg edge case.
- _rotation_matrix_to_rotvec: round-trip identity.
- compute_feature_plane_pose: axis-aligned board, tilted board.
- FeaturePlaneConfig construction.
- BoardCoordinates in Feature Plane mode:
    - square_to_xy, approach_pose, pick_pose for axis-aligned board.
    - envelope safety check still applied.
    - collinear calibration points rejected.
- BoardCoordinates base-frame mode still works (regression guard).
"""

from __future__ import annotations

import math
import pytest

from src.robot.coordinates import (
    BoardCoordinates,
    FeaturePlaneConfig,
    SafetyViolationError,
    TRANSIT_Z_MIN,
    WorkspaceSafetyEnvelope,
    _rotvec_to_matrix,
    _apply_rotation,
)
from scripts.calibrate_feature import (
    compute_feature_plane_pose,
    _rotation_matrix_to_rotvec,
    _vec_cross,
    _vec_norm,
    _vec_normalize,
    _vec_sub,
)


# ---------------------------------------------------------------------------
# Rodrigues helpers
# ---------------------------------------------------------------------------

class TestRotvecToMatrix:
    def test_identity_zero_angle(self):
        """Zero rotation vector → identity matrix."""
        R = _rotvec_to_matrix((0.0, 0.0, 0.0))
        assert R[0][0] == pytest.approx(1.0)
        assert R[1][1] == pytest.approx(1.0)
        assert R[2][2] == pytest.approx(1.0)
        assert R[0][1] == pytest.approx(0.0)
        assert R[1][0] == pytest.approx(0.0)

    def test_90_deg_around_z(self):
        """
        90-degree rotation about Z: (0,0,pi/2).
        Expected: x→y, y→-x, z→z.
        """
        R = _rotvec_to_matrix((0.0, 0.0, math.pi / 2))
        # R * [1,0,0] = [0,1,0]
        assert R[0][0] == pytest.approx(0.0, abs=1e-9)
        assert R[1][0] == pytest.approx(1.0, abs=1e-9)
        assert R[2][0] == pytest.approx(0.0, abs=1e-9)
        # R * [0,1,0] = [-1,0,0]
        assert R[0][1] == pytest.approx(-1.0, abs=1e-9)
        assert R[1][1] == pytest.approx(0.0, abs=1e-9)

    def test_90_deg_around_x(self):
        """
        90-degree rotation about X: (pi/2, 0, 0).
        Expected: y→z, z→-y.
        """
        R = _rotvec_to_matrix((math.pi / 2, 0.0, 0.0))
        # R * [0,1,0] = [0,0,1]
        assert R[0][1] == pytest.approx(0.0, abs=1e-9)
        assert R[1][1] == pytest.approx(0.0, abs=1e-9)
        assert R[2][1] == pytest.approx(1.0, abs=1e-9)

    def test_180_deg_around_y(self):
        """
        180-degree rotation about Y: (0, pi, 0).
        Expected: x→-x, z→-z, y unchanged.
        """
        R = _rotvec_to_matrix((0.0, math.pi, 0.0))
        assert R[0][0] == pytest.approx(-1.0, abs=1e-9)
        assert R[1][1] == pytest.approx(1.0, abs=1e-9)
        assert R[2][2] == pytest.approx(-1.0, abs=1e-9)

    def test_result_is_orthogonal(self):
        """R * R^T = I for arbitrary rotation."""
        rv = (0.3, -0.5, 0.7)
        R = _rotvec_to_matrix(rv)
        # R * R^T
        for i in range(3):
            for j in range(3):
                dot = sum(R[i][k] * R[j][k] for k in range(3))
                expected = 1.0 if i == j else 0.0
                assert dot == pytest.approx(expected, abs=1e-9)

    def test_unit_vector_preserved_on_rotation_axis(self):
        """Rotation about Z leaves Z invariant."""
        angle = 1.23  # rad — arbitrary
        R = _rotvec_to_matrix((0.0, 0.0, angle))
        # R * [0,0,1] = [0,0,1]
        result = _apply_rotation(R, (0.0, 0.0, 1.0))
        assert result[0] == pytest.approx(0.0, abs=1e-9)
        assert result[1] == pytest.approx(0.0, abs=1e-9)
        assert result[2] == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Round-trip: rotvec → matrix → rotvec
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def _round_trip(self, rotvec):
        R = _rotvec_to_matrix(rotvec)
        rv2 = _rotation_matrix_to_rotvec(R)
        # Verify R matches re-constructed matrix
        R2 = _rotvec_to_matrix(tuple(rv2))
        for i in range(3):
            for j in range(3):
                assert R[i][j] == pytest.approx(R2[i][j], abs=1e-9)

    def test_round_trip_identity(self):
        self._round_trip((0.0, 0.0, 0.0))

    def test_round_trip_z_rotation(self):
        self._round_trip((0.0, 0.0, math.pi / 4))

    def test_round_trip_arbitrary(self):
        self._round_trip((0.3, -0.5, 0.7))

    def test_round_trip_near_pi(self):
        """Near-180-degree rotation is the numerically tricky edge case."""
        axis = _vec_normalize([1.0, 1.0, 0.0])
        angle = math.pi - 1e-5  # rad — almost 180 deg
        rv = tuple(a * angle for a in axis)
        # Near-pi branch has limited float precision; accept 1e-4 tolerance
        R = _rotvec_to_matrix(rv)
        rv2 = _rotation_matrix_to_rotvec(R)
        R2 = _rotvec_to_matrix(tuple(rv2))
        for i in range(3):
            for j in range(3):
                assert R[i][j] == pytest.approx(R2[i][j], abs=1e-4)


# ---------------------------------------------------------------------------
# compute_feature_plane_pose
# ---------------------------------------------------------------------------

class TestComputeFeaturePlanePose:
    def test_axis_aligned_board_flat(self):
        """
        Board lying flat on the table (Z=const), aligned with robot X and Y.
        p_a1 = (0.3, -0.4, 0.01)
        p_h1 = (0.3 + 7*0.055, -0.4, 0.01)  → pure +X direction
        p_a8 = (0.3, -0.4 + 7*0.055, 0.01)  → pure +Y direction

        Expected origin = p_a1.
        Expected rotation: board local X aligns with base X,
                           board local Y aligns with base Y → identity rotation.
        """
        pitch = 0.055  # m
        p_a1 = [0.3, -0.4, 0.01]
        p_h1 = [0.3 + 7 * pitch, -0.4, 0.01]
        p_a8 = [0.3, -0.4 + 7 * pitch, 0.01]
        pose = compute_feature_plane_pose(p_a1, p_h1, p_a8)

        # Origin = a1
        assert pose[0] == pytest.approx(p_a1[0], abs=1e-9)
        assert pose[1] == pytest.approx(p_a1[1], abs=1e-9)
        assert pose[2] == pytest.approx(p_a1[2], abs=1e-9)

        # For identity-ish rotation, rotvec magnitude should be near 0 or pi
        # (board X||base X, board Y||base Y, board Z||base Z → identity)
        rotvec = pose[3:]
        R = _rotvec_to_matrix(tuple(rotvec))
        # R * [1,0,0] ≈ [1,0,0]
        x_base = _apply_rotation(R, (1.0, 0.0, 0.0))
        assert x_base[0] == pytest.approx(1.0, abs=1e-9)
        assert x_base[1] == pytest.approx(0.0, abs=1e-9)
        assert x_base[2] == pytest.approx(0.0, abs=1e-9)

    def test_board_rotated_90_deg_around_z(self):
        """
        Board rotated 90 deg CCW around base Z:
            a→h direction maps to base -Y (not +X).
        p_h1 is at (0.3, -0.4 - 7*pitch, 0.01).
        p_a8 is at (0.3 + 7*pitch, -0.4, 0.01).

        Board local X = base -Y, board local Y = base +X.
        """
        pitch = 0.055
        p_a1 = [0.3, -0.4, 0.01]
        p_h1 = [0.3, -0.4 - 7 * pitch, 0.01]   # a→h goes in base -Y
        p_a8 = [0.3 + 7 * pitch, -0.4, 0.01]   # 1→8 goes in base +X
        pose = compute_feature_plane_pose(p_a1, p_h1, p_a8)
        rotvec = pose[3:]
        R = _rotvec_to_matrix(tuple(rotvec))
        # Board local X (first column) → base -Y
        x_base = _apply_rotation(R, (1.0, 0.0, 0.0))
        assert x_base[0] == pytest.approx(0.0, abs=1e-9)
        assert x_base[1] == pytest.approx(-1.0, abs=1e-9)
        assert x_base[2] == pytest.approx(0.0, abs=1e-9)

    def test_collinear_points_raise(self):
        """Three collinear points cannot define a plane."""
        p_a1 = [0.3, 0.3, 0.01]
        p_h1 = [0.6, 0.3, 0.01]   # along same X line
        p_a8 = [0.9, 0.3, 0.01]   # collinear
        with pytest.raises(ValueError):
            compute_feature_plane_pose(p_a1, p_h1, p_a8)

    def test_tilted_board_z_normal_orthogonal(self):
        """
        Board tilted 15 deg about base X:
            p_a1 at origin, p_h1 along +X, p_a8 tilted into YZ plane.
        The computed board normal (Z-axis) must be perpendicular to X and Y.
        """
        tilt = math.radians(15)  # rad
        pitch = 0.055
        p_a1 = [0.0, 0.0, 0.0]
        p_h1 = [7 * pitch, 0.0, 0.0]
        p_a8 = [0.0, 7 * pitch * math.cos(tilt), 7 * pitch * math.sin(tilt)]
        pose = compute_feature_plane_pose(p_a1, p_h1, p_a8)
        rotvec = pose[3:]
        R = _rotvec_to_matrix(tuple(rotvec))
        # Board normal is Z column of R (third column)
        z_col = [R[0][2], R[1][2], R[2][2]]
        # Must be unit length
        assert _vec_norm(z_col) == pytest.approx(1.0, abs=1e-9)
        # Must be orthogonal to x column
        x_col = [R[0][0], R[1][0], R[2][0]]
        assert sum(z_col[i]*x_col[i] for i in range(3)) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# FeaturePlaneConfig
# ---------------------------------------------------------------------------

class TestFeaturePlaneConfig:
    def test_defaults(self):
        fp = FeaturePlaneConfig(pose=[0.3, -0.4, 0.01, 0.0, 0.0, 0.0])
        assert fp.pitch_x == pytest.approx(0.055)
        assert fp.pitch_y == pytest.approx(0.055)
        assert fp.pick_height_z == pytest.approx(0.010)
        assert fp.transit_clearance_z == pytest.approx(0.120)

    def test_custom_pitch(self):
        fp = FeaturePlaneConfig(
            pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            pitch_x=0.060,
            pitch_y=0.060,
        )
        assert fp.pitch_x == pytest.approx(0.060)


# ---------------------------------------------------------------------------
# BoardCoordinates — Feature Plane mode
# ---------------------------------------------------------------------------

# Axis-aligned board, identity orientation, flat on table.
_FP_ORIGIN = [0.30, -0.40, 0.10]   # meters — a1 in base frame
_FP_POSE   = _FP_ORIGIN + [0.0, 0.0, 0.0]  # identity rotation
_PITCH     = 0.055   # m


@pytest.fixture
def fp_bc():
    """BoardCoordinates in Feature Plane mode, axis-aligned identity board."""
    fp = FeaturePlaneConfig(
        pose=_FP_POSE,
        pitch_x=_PITCH,
        pitch_y=_PITCH,
        pick_height_z=0.010,
        transit_clearance_z=0.120,
    )
    return BoardCoordinates(
        origin_xyz=(0.0, 0.0, 0.0),  # unused in FP mode
        dx=0.0,                       # unused in FP mode
        dy=0.0,                       # unused in FP mode
        board_z=_FP_ORIGIN[2],       # board surface Z in base frame
        pick_z=_FP_ORIGIN[2],        # unused in FP mode but validates
        transit_z=0.120,              # unused in FP mode
        feature_plane=fp,
    )


class TestBoardCoordinatesFeaturePlane:
    def test_a1_xy_is_origin(self, fp_bc):
        """a1 maps to the feature plane origin (identity rotation)."""
        x, y = fp_bc.square_to_xy("a1")
        assert x == pytest.approx(_FP_ORIGIN[0], abs=1e-9)
        assert y == pytest.approx(_FP_ORIGIN[1], abs=1e-9)

    def test_b1_one_file(self, fp_bc):
        """b1 is one pitch_x east of a1 in base X."""
        x, y = fp_bc.square_to_xy("b1")
        assert x == pytest.approx(_FP_ORIGIN[0] + _PITCH, abs=1e-9)
        assert y == pytest.approx(_FP_ORIGIN[1], abs=1e-9)

    def test_a2_one_rank(self, fp_bc):
        """a2 is one pitch_y north of a1 in base Y."""
        x, y = fp_bc.square_to_xy("a2")
        assert x == pytest.approx(_FP_ORIGIN[0], abs=1e-9)
        assert y == pytest.approx(_FP_ORIGIN[1] + _PITCH, abs=1e-9)

    def test_h8_corner(self, fp_bc):
        x, y = fp_bc.square_to_xy("h8")
        assert x == pytest.approx(_FP_ORIGIN[0] + 7 * _PITCH, abs=1e-9)
        assert y == pytest.approx(_FP_ORIGIN[1] + 7 * _PITCH, abs=1e-9)

    def test_approach_pose_length(self, fp_bc):
        pose = fp_bc.approach_pose("a1")
        assert len(pose) == 6

    def test_approach_pose_z_is_transit(self, fp_bc):
        """For identity rotation, transit Z in base = origin_Z + transit_clearance_z."""
        pose = fp_bc.approach_pose("a1")
        expected_z = _FP_ORIGIN[2] + 0.120  # local transit + board Z origin
        assert pose[2] == pytest.approx(expected_z, abs=1e-9)

    def test_pick_pose_z_is_pick_height(self, fp_bc):
        """For identity rotation, pick Z = origin_Z + pick_height_z."""
        pose = fp_bc.pick_pose("a1")
        expected_z = _FP_ORIGIN[2] + 0.010
        assert pose[2] == pytest.approx(expected_z, abs=1e-9)

    def test_transit_alias(self, fp_bc):
        assert fp_bc.transit_pose("e4") == fp_bc.approach_pose("e4")

    def test_orientation_from_feature_plane(self, fp_bc):
        """Pose orientation = feature plane rotation vector (identity → [0,0,0])."""
        pose = fp_bc.approach_pose("d4")
        assert pose[3] == pytest.approx(0.0, abs=1e-9)
        assert pose[4] == pytest.approx(0.0, abs=1e-9)
        assert pose[5] == pytest.approx(0.0, abs=1e-9)

    def test_envelope_violation_raises(self):
        """Board placed outside envelope → SafetyViolationError."""
        fp = FeaturePlaneConfig(
            pose=[2.0, 0.0, 0.10, 0.0, 0.0, 0.0],  # X=2.0 outside ±0.80
        )
        bc = BoardCoordinates(
            origin_xyz=(0.0, 0.0, 0.0),
            dx=0.0, dy=0.0,
            board_z=0.10, pick_z=0.10,
            feature_plane=fp,
        )
        with pytest.raises(SafetyViolationError):
            bc.approach_pose("a1")

    def test_all_squares_pass_for_valid_board(self, fp_bc):
        """All 64 squares must not raise for a well-placed board."""
        for f in "abcdefgh":
            for r in "12345678":
                fp_bc.approach_pose(f + r)
                fp_bc.pick_pose(f + r)

    def test_fp_mode_no_post_init_transit_check(self):
        """In FP mode, transit_z < TRANSIT_Z_MIN must NOT raise on construction."""
        fp = FeaturePlaneConfig(
            pose=[0.3, -0.4, 0.10, 0.0, 0.0, 0.0],
            transit_clearance_z=0.120,
        )
        # transit_z=0.0 would fail in base-frame mode, must succeed in FP mode
        bc = BoardCoordinates(
            origin_xyz=(0.0, 0.0, 0.0),
            dx=0.0, dy=0.0,
            board_z=0.10, pick_z=0.10,
            transit_z=0.0,      # deliberately invalid for base-frame mode
            feature_plane=fp,
        )
        # Construction must succeed; FP mode ignores transit_z
        assert bc.feature_plane is not None


# ---------------------------------------------------------------------------
# Regression guard: base-frame mode still works
# ---------------------------------------------------------------------------

class TestBaseModeRegression:
    """Ensure Feature Plane additions did not break the original behaviour."""

    def test_base_mode_square_to_xy(self):
        bc = BoardCoordinates(
            origin_xyz=(0.30, -0.45, 0.010),
            dx=0.050, dy=0.050,
            board_z=0.010, pick_z=0.010,
            transit_z=0.120,
        )
        x, y = bc.square_to_xy("h8")
        assert x == pytest.approx(0.30 + 7 * 0.050, abs=1e-9)
        assert y == pytest.approx(-0.45 + 7 * 0.050, abs=1e-9)

    def test_base_mode_transit_z_below_min_raises(self):
        with pytest.raises(ValueError, match="transit_z"):
            BoardCoordinates(
                origin_xyz=(0.30, -0.45, 0.010),
                dx=0.050, dy=0.050,
                board_z=0.010, pick_z=0.010,
                transit_z=TRANSIT_Z_MIN - 0.001,
            )

    def test_base_mode_pick_below_board_raises(self):
        with pytest.raises(ValueError, match="pick_z"):
            BoardCoordinates(
                origin_xyz=(0.30, -0.45, 0.010),
                dx=0.050, dy=0.050,
                board_z=0.010, pick_z=0.005,
                transit_z=0.120,
            )
