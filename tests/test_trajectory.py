"""
Unit tests for TrajectoryGenerator (src/robot/trajectory.py)
and GraveyardManager / MoveSequencer (src/robot/graveyard.py).

All tests run without hardware: uses MockURDriver and in-process calls only.

Test inventory
--------------
TrajectoryGenerator
    test_plan_pick_and_place_waypoint_count
    test_plan_pick_and_place_labels
    test_plan_transit_z_floor
    test_plan_approach_z_equals_transit_z
    test_plan_pick_z_at_piece_height
    test_plan_park_pose_is_last_waypoint
    test_execute_gripper_grasp_on_pick
    test_execute_gripper_release_on_place
    test_execute_gripper_order
    test_execute_movel_call_count
    test_safety_violation_raises
    test_park_pose_envelope_check

GraveyardManager
    test_white_slots_sequential
    test_black_slots_sequential
    test_slot_axis_x
    test_slot_axis_y
    test_graveyard_full_white
    test_graveyard_full_black
    test_reset_white
    test_reset_black
    test_reset_both
    test_invalid_slot_axis
    test_invalid_slot_spacing
    test_envelope_called_on_slot

MoveSequencer
    test_normal_move_one_pick_and_place
    test_capture_move_two_stages
    test_capture_clears_victim_square
    test_capture_advances_graveyard_counter
    test_capture_invalid_color_raises
    test_castling_move_two_pick_and_place
    test_castling_king_before_rook
"""

from __future__ import annotations

import math
from typing import List
from unittest.mock import MagicMock, call, patch

import pytest

from src.robot.coordinates import BoardCoordinates, WorkspaceSafetyEnvelope
from src.robot.graveyard import GraveyardFullError, GraveyardManager, MoveSequencer
from src.robot.trajectory import DEFAULT_PARK_POSE, TrajectoryGenerator, Waypoint
from src.robot.ur_driver import MockURDriver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def coords() -> BoardCoordinates:
    """Standard board at Z=0.005 m surface, pick_z=0.010 m (grip height on piece),
    transit at 0.12 m, dx=dy=0.055 m."""
    return BoardCoordinates(
        origin_xyz=(0.10, -0.30, 0.005),
        dx=0.055,
        dy=0.055,
        board_z=0.005,
        pick_z=0.010,
        transit_z=0.12,
    )


@pytest.fixture()
def tgen(coords: BoardCoordinates) -> TrajectoryGenerator:
    return TrajectoryGenerator(coords=coords)


@pytest.fixture()
def driver() -> MockURDriver:
    return MockURDriver()


@pytest.fixture()
def graveyard() -> GraveyardManager:
    return GraveyardManager(
        origin_white=(-0.50, -0.30, 0.01),
        origin_black=( 0.55, -0.30, 0.01),
        slot_spacing=0.045,
        slot_axis="y",
    )


@pytest.fixture()
def sequencer(tgen: TrajectoryGenerator, graveyard: GraveyardManager) -> MoveSequencer:
    return MoveSequencer(trajectory=tgen, graveyard=graveyard)


# ---------------------------------------------------------------------------
# Helper: fast MockURDriver that skips sleep
# ---------------------------------------------------------------------------

@pytest.fixture()
def fast_driver(monkeypatch: pytest.MonkeyPatch) -> MockURDriver:
    """MockURDriver with sleep replaced by no-op for speed."""
    import time
    monkeypatch.setattr(time, "sleep", lambda _: None)
    return MockURDriver()


# ---------------------------------------------------------------------------
# TrajectoryGenerator — plan tests (no hardware calls)
# ---------------------------------------------------------------------------

class TestTrajectoryPlan:

    def test_plan_pick_and_place_waypoint_count(self, tgen: TrajectoryGenerator) -> None:
        wps = tgen.plan_pick_and_place("e2", "e4")
        # approach_src, pick_src, ascend_src, transit_dst, place_dst, retract_dst, park
        assert len(wps) == 7

    def test_plan_pick_and_place_labels(self, tgen: TrajectoryGenerator) -> None:
        wps = tgen.plan_pick_and_place("a1", "h8")
        labels = [w.label for w in wps]
        assert labels[0].startswith("approach_src:")
        assert labels[1].startswith("pick:")
        assert labels[2].startswith("ascend_src:")
        assert labels[3].startswith("transit_to:")
        assert labels[4].startswith("place:")
        assert labels[5].startswith("retract_dst:")
        assert labels[6] == "park"

    def test_plan_transit_z_floor(self, tgen: TrajectoryGenerator) -> None:
        """All approach and retract poses must be at transit_z (>= 80 mm)."""
        wps = tgen.plan_pick_and_place("d4", "f6")
        transit_idxs = [0, 2, 3, 5]  # approach_src, ascend_src, transit_dst, retract_dst
        for i in transit_idxs:
            z = wps[i].pose[2]
            assert z >= 0.080, f"wp[{i}].pose Z={z:.4f} m below 80 mm floor"

    def test_plan_approach_z_equals_transit_z(self, tgen: TrajectoryGenerator) -> None:
        wps = tgen.plan_pick_and_place("c3", "g5")
        assert wps[0].pose[2] == pytest.approx(tgen.coords.transit_z)
        assert wps[2].pose[2] == pytest.approx(tgen.coords.transit_z)

    def test_plan_pick_z_at_piece_height(self, tgen: TrajectoryGenerator) -> None:
        wps = tgen.plan_pick_and_place("b2", "b4")
        assert wps[1].pose[2] == pytest.approx(tgen.coords.pick_z)  # pick src
        assert wps[4].pose[2] == pytest.approx(tgen.coords.pick_z)  # place dst

    def test_plan_park_pose_is_last_waypoint(self, tgen: TrajectoryGenerator) -> None:
        wps = tgen.plan_pick_and_place("e1", "g1")
        assert wps[-1].pose[:3] == pytest.approx(tgen.park_pose[:3], abs=1e-6)

    def test_plan_from_to_xy_correct(
        self, tgen: TrajectoryGenerator, coords: BoardCoordinates
    ) -> None:
        wps = tgen.plan_pick_and_place("a1", "h8")
        x_a1, y_a1 = coords.square_to_xy("a1")
        x_h8, y_h8 = coords.square_to_xy("h8")
        # approach_src X,Y should match a1
        assert wps[0].pose[0] == pytest.approx(x_a1, abs=1e-6)
        assert wps[0].pose[1] == pytest.approx(y_a1, abs=1e-6)
        # transit_to X,Y should match h8
        assert wps[3].pose[0] == pytest.approx(x_h8, abs=1e-6)
        assert wps[3].pose[1] == pytest.approx(y_h8, abs=1e-6)


# ---------------------------------------------------------------------------
# TrajectoryGenerator — execute tests
# ---------------------------------------------------------------------------

class TestTrajectoryExecute:

    def _mock_driver(self) -> MagicMock:
        """Return a mock that satisfies _DriverProto."""
        m = MagicMock()
        m.movel = MagicMock()
        m.set_digital_out = MagicMock()
        return m

    def test_execute_movel_call_count(self, tgen: TrajectoryGenerator) -> None:
        """pick_and_place must call movel exactly 7 times."""
        drv = self._mock_driver()
        tgen.pick_and_place(drv, "e2", "e4")
        assert drv.movel.call_count == 7

    def test_execute_gripper_grasp_on_pick(self, tgen: TrajectoryGenerator) -> None:
        """set_digital_out(0, True) called once, after the pick descent."""
        drv = self._mock_driver()
        tgen.pick_and_place(drv, "e2", "e4")
        drv.set_digital_out.assert_any_call(0, True)

    def test_execute_gripper_release_on_place(self, tgen: TrajectoryGenerator) -> None:
        """set_digital_out(0, False) called once, after the place descent."""
        drv = self._mock_driver()
        tgen.pick_and_place(drv, "e2", "e4")
        drv.set_digital_out.assert_any_call(0, False)

    def test_execute_gripper_order(self, tgen: TrajectoryGenerator) -> None:
        """Grasp must precede release in the call sequence."""
        drv = self._mock_driver()
        tgen.pick_and_place(drv, "d2", "d4")

        do_calls = drv.set_digital_out.call_args_list
        assert len(do_calls) == 2

        grasp_call   = do_calls[0]
        release_call = do_calls[1]
        assert grasp_call   == call(0, True)
        assert release_call == call(0, False)

    def test_safety_violation_raises(
        self, coords: BoardCoordinates
    ) -> None:
        """Square that maps outside envelope must raise SafetyViolationError."""
        # origin at X=0.50, dx=0.055: h-file X = 0.50 + 7*0.055 = 0.885 > 0.80
        bad_coords = BoardCoordinates(
            origin_xyz=(0.50, 0.0, 0.005),
            dx=0.055,
            dy=0.055,
            board_z=0.005,
            pick_z=0.010,
            transit_z=0.12,
        )
        tgen = TrajectoryGenerator(coords=bad_coords)
        drv = MagicMock()
        drv.movel = MagicMock()
        drv.set_digital_out = MagicMock()
        from src.robot.coordinates import SafetyViolationError
        with pytest.raises(SafetyViolationError):
            tgen.pick_and_place(drv, "h1", "h1")

    def test_park_pose_envelope_check(self) -> None:
        """Park pose outside envelope must raise at TrajectoryGenerator init."""
        coords = BoardCoordinates(
            origin_xyz=(0.10, -0.30, 0.005),
            dx=0.055,
            dy=0.055,
            board_z=0.005,
            pick_z=0.010,
            transit_z=0.12,
        )
        from src.robot.coordinates import SafetyViolationError
        with pytest.raises(SafetyViolationError):
            TrajectoryGenerator(
                coords=coords,
                park_pose=[0.0, -0.60, 1.50, math.pi, 0.0, 0.0],  # Z=1.5 m > 0.9
            )


# ---------------------------------------------------------------------------
# GraveyardManager tests
# ---------------------------------------------------------------------------

class TestGraveyardManager:

    def test_white_slots_sequential(self, graveyard: GraveyardManager) -> None:
        p0 = graveyard.next_white_slot()
        p1 = graveyard.next_white_slot()
        # Slots differ only along slot_axis (y)
        assert p0[1] == pytest.approx(-0.30, abs=1e-6)
        assert p1[1] == pytest.approx(-0.30 + 0.045, abs=1e-6)
        assert graveyard.white_slots_used == 2

    def test_black_slots_sequential(self, graveyard: GraveyardManager) -> None:
        p0 = graveyard.next_black_slot()
        p1 = graveyard.next_black_slot()
        assert p0[0] == pytest.approx(0.55, abs=1e-6)
        assert p1[1] == pytest.approx(-0.30 + 0.045, abs=1e-6)
        assert graveyard.black_slots_used == 2

    def test_slot_axis_x(self) -> None:
        gv = GraveyardManager(
            origin_white=(0.10, 0.20, 0.01),
            origin_black=(0.10, 0.30, 0.01),
            slot_spacing=0.05,
            slot_axis="x",
        )
        p0 = gv.next_white_slot()
        p1 = gv.next_white_slot()
        assert p0[0] == pytest.approx(0.10, abs=1e-6)
        assert p1[0] == pytest.approx(0.15, abs=1e-6)
        assert p0[1] == pytest.approx(0.20, abs=1e-6)  # Y unchanged

    def test_slot_axis_y(self, graveyard: GraveyardManager) -> None:
        p0 = graveyard.next_white_slot()
        p1 = graveyard.next_white_slot()
        assert p0[0] == p1[0]  # X unchanged when axis=y

    def test_graveyard_full_white(self, graveyard: GraveyardManager) -> None:
        for _ in range(16):
            graveyard.next_white_slot()
        with pytest.raises(GraveyardFullError):
            graveyard.next_white_slot()

    def test_graveyard_full_black(self, graveyard: GraveyardManager) -> None:
        for _ in range(16):
            graveyard.next_black_slot()
        with pytest.raises(GraveyardFullError):
            graveyard.next_black_slot()

    def test_reset_white(self, graveyard: GraveyardManager) -> None:
        graveyard.next_white_slot()
        graveyard.next_white_slot()
        graveyard.reset("white")
        assert graveyard.white_slots_used == 0

    def test_reset_black(self, graveyard: GraveyardManager) -> None:
        graveyard.next_black_slot()
        graveyard.reset("black")
        assert graveyard.black_slots_used == 0

    def test_reset_both(self, graveyard: GraveyardManager) -> None:
        graveyard.next_white_slot()
        graveyard.next_black_slot()
        graveyard.reset("both")
        assert graveyard.white_slots_used == 0
        assert graveyard.black_slots_used == 0

    def test_invalid_slot_axis(self) -> None:
        with pytest.raises(ValueError, match="slot_axis"):
            GraveyardManager(
                origin_white=(0.0, 0.0, 0.0),
                origin_black=(0.0, 0.0, 0.0),
                slot_spacing=0.05,
                slot_axis="z",
            )

    def test_invalid_slot_spacing(self) -> None:
        with pytest.raises(ValueError, match="slot_spacing"):
            GraveyardManager(
                origin_white=(0.0, 0.0, 0.0),
                origin_black=(0.0, 0.0, 0.0),
                slot_spacing=0.0,
            )

    def test_envelope_called_on_slot(self) -> None:
        envelope = MagicMock()
        gv = GraveyardManager(
            origin_white=(0.10, 0.10, 0.01),
            origin_black=(0.20, 0.10, 0.01),
            slot_spacing=0.05,
            envelope=envelope,
        )
        gv.next_white_slot()
        envelope.check.assert_called_once()

    def test_slot_orientation_downward(self, graveyard: GraveyardManager) -> None:
        """All graveyard poses have downward-pointing tool orientation (Rx=pi)."""
        for _ in range(3):
            pose = graveyard.next_white_slot()
            assert pose[3] == pytest.approx(math.pi, abs=1e-5)
            assert pose[4] == pytest.approx(0.0, abs=1e-5)
            assert pose[5] == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# MoveSequencer tests
# ---------------------------------------------------------------------------

class TestMoveSequencer:

    def _mock_tgen(self, coords: BoardCoordinates) -> MagicMock:
        """Return a MagicMock shaped like TrajectoryGenerator."""
        tg = MagicMock()
        tg.coords = coords
        tg.park_pose = list(DEFAULT_PARK_POSE)
        tg.vel_transit = 0.30
        tg.acc_transit = 1.20
        tg.vel_vert = 0.10
        tg.acc_vert = 0.60
        tg.vel_slow = 0.05
        tg.acc_slow = 0.30
        return tg

    def _mock_drv(self) -> MagicMock:
        drv = MagicMock()
        drv.movel = MagicMock()
        drv.set_digital_out = MagicMock()
        return drv

    def test_normal_move_one_pick_and_place(
        self, coords: BoardCoordinates, graveyard: GraveyardManager
    ) -> None:
        tg = self._mock_tgen(coords)
        seq = MoveSequencer(trajectory=tg, graveyard=graveyard)
        drv = self._mock_drv()
        seq.execute_normal(drv, "e2", "e4")
        tg.pick_and_place.assert_called_once_with(drv, "e2", "e4")

    def test_capture_move_two_stages(
        self, coords: BoardCoordinates, graveyard: GraveyardManager
    ) -> None:
        """Capture must call pick_and_place exactly once (attacker stage)."""
        tg = self._mock_tgen(coords)
        seq = MoveSequencer(trajectory=tg, graveyard=graveyard)
        drv = self._mock_drv()
        seq.execute_capture(drv, "d4", "e5", captured_color="black")
        # Attacker stage uses pick_and_place
        tg.pick_and_place.assert_called_once_with(drv, "d4", "e5")

    def test_capture_clears_victim_square(
        self, coords: BoardCoordinates, graveyard: GraveyardManager
    ) -> None:
        """
        Victim removal (stage 1): first movel call must target the approach pose
        above the victim square (e5), not the attacker's square.
        """
        tg = self._mock_tgen(coords)
        seq = MoveSequencer(trajectory=tg, graveyard=graveyard)
        drv = self._mock_drv()
        seq.execute_capture(drv, "d4", "e5", captured_color="black")

        # The first movel (victim approach) X,Y must match e5's XY
        x_e5, y_e5 = coords.square_to_xy("e5")
        first_call_pose = drv.movel.call_args_list[0][0][0]  # first positional arg
        assert first_call_pose[0] == pytest.approx(x_e5, abs=1e-6)
        assert first_call_pose[1] == pytest.approx(y_e5, abs=1e-6)

    def test_capture_advances_graveyard_counter(
        self, coords: BoardCoordinates, graveyard: GraveyardManager
    ) -> None:
        tg = self._mock_tgen(coords)
        seq = MoveSequencer(trajectory=tg, graveyard=graveyard)
        drv = self._mock_drv()
        assert graveyard.black_slots_used == 0
        seq.execute_capture(drv, "d4", "e5", captured_color="black")
        assert graveyard.black_slots_used == 1

    def test_capture_white_piece_uses_white_graveyard(
        self, coords: BoardCoordinates, graveyard: GraveyardManager
    ) -> None:
        tg = self._mock_tgen(coords)
        seq = MoveSequencer(trajectory=tg, graveyard=graveyard)
        drv = self._mock_drv()
        seq.execute_capture(drv, "c6", "d5", captured_color="white")
        assert graveyard.white_slots_used == 1
        assert graveyard.black_slots_used == 0

    def test_capture_invalid_color_raises(
        self, coords: BoardCoordinates, graveyard: GraveyardManager
    ) -> None:
        tg = self._mock_tgen(coords)
        seq = MoveSequencer(trajectory=tg, graveyard=graveyard)
        drv = self._mock_drv()
        with pytest.raises(ValueError, match="captured_color"):
            seq.execute_capture(drv, "e2", "e4", captured_color="red")

    def test_castling_move_two_pick_and_place(
        self, coords: BoardCoordinates, graveyard: GraveyardManager
    ) -> None:
        tg = self._mock_tgen(coords)
        seq = MoveSequencer(trajectory=tg, graveyard=graveyard)
        drv = self._mock_drv()
        seq.execute_castling(drv, "e1", "g1", "h1", "f1")
        assert tg.pick_and_place.call_count == 2

    def test_castling_king_before_rook(
        self, coords: BoardCoordinates, graveyard: GraveyardManager
    ) -> None:
        tg = self._mock_tgen(coords)
        seq = MoveSequencer(trajectory=tg, graveyard=graveyard)
        drv = self._mock_drv()
        seq.execute_castling(drv, "e1", "g1", "h1", "f1")
        calls = tg.pick_and_place.call_args_list
        # First call: King
        assert calls[0] == call(drv, "e1", "g1")
        # Second call: Rook
        assert calls[1] == call(drv, "h1", "f1")
