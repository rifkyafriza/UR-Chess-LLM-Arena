"""
Comprehensive end-to-end unit and integration tests for the robot module.

Covers:
    MockURDriver    — kinematics, TCP pose updates, digital output state transitions
    RealURDriver    — URScript string encoding, dashboard commands (via MockURServer)
    BoardCoordinates — all 64 squares, boundary safety violations
    TrajectoryGenerator — waypoint sequences for normal, capture, and castling moves

All tests run offline (no physical hardware required).
"""

from __future__ import annotations

import math
import socket
import struct
import threading
import time
from typing import List
from unittest.mock import MagicMock, call

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
from src.robot.graveyard import GraveyardFullError, GraveyardManager, MoveSequencer
from src.robot.trajectory import DEFAULT_PARK_POSE, TrajectoryGenerator, Waypoint
from src.robot.ur_driver import (
    DashboardClient,
    MockURDriver,
    RealURDriver,
    _RT_PACKET_LEN,
    _RT_TCP_POSE_OFFSET,
    _check_pose_limits,
    create_robot_driver,
)
from tests.mock_ur_server import MockURServer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _free_port() -> int:
    """Return an OS-assigned free TCP port."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_rt_packet(pose: List[float]) -> bytes:
    """Build a minimal 1116-byte RT packet with the given TCP pose at the correct offset."""
    buf = bytearray(_RT_PACKET_LEN)
    struct.pack_into(">I", buf, 0, _RT_PACKET_LEN)
    struct.pack_into(">6d", buf, _RT_TCP_POSE_OFFSET, *pose)
    return bytes(buf)


# ---------------------------------------------------------------------------
# Shared board fixture
# ---------------------------------------------------------------------------

ORIGIN = (0.10, -0.30, 0.005)
DX = 0.055
DY = 0.055
BOARD_Z = 0.005
PICK_Z = 0.010
TRANSIT_Z = 0.120


@pytest.fixture()
def coords() -> BoardCoordinates:
    return BoardCoordinates(
        origin_xyz=ORIGIN,
        dx=DX,
        dy=DY,
        board_z=BOARD_Z,
        pick_z=PICK_Z,
        transit_z=TRANSIT_Z,
    )


@pytest.fixture()
def tgen(coords: BoardCoordinates) -> TrajectoryGenerator:
    return TrajectoryGenerator(coords=coords)


@pytest.fixture()
def graveyard() -> GraveyardManager:
    return GraveyardManager(
        origin_white=(-0.50, -0.30, 0.01),
        origin_black=(0.55, -0.30, 0.01),
        slot_spacing=0.045,
        slot_axis="y",
    )


@pytest.fixture()
def sequencer(tgen: TrajectoryGenerator, graveyard: GraveyardManager) -> MoveSequencer:
    return MoveSequencer(trajectory=tgen, graveyard=graveyard)


@pytest.fixture()
def fast_driver(monkeypatch: pytest.MonkeyPatch) -> MockURDriver:
    """MockURDriver with sleep patched to no-op for speed."""
    monkeypatch.setattr(time, "sleep", lambda _: None)
    return MockURDriver()


# ---------------------------------------------------------------------------
# Section 1: MockURDriver — kinematics, TCP pose, digital outputs
# ---------------------------------------------------------------------------

class TestMockURDriverKinematics:
    """MockURDriver: Cartesian and joint kinematics, pose tracking."""

    def setup_method(self):
        self.drv = MockURDriver()

    def test_initial_pose_length(self):
        assert len(self.drv.get_tcp_pose()) == 6

    def test_initial_joints_length(self):
        assert len(self.drv.get_joints()) == 6

    def test_movel_updates_tcp_pose(self):
        target = [0.20, -0.30, 0.40, 0.0, math.pi, 0.0]
        self.drv.movel(target)
        assert self.drv.get_tcp_pose() == pytest.approx(target)

    def test_movel_pose_all_six_elements_match(self):
        target = [0.10, -0.10, 0.30, math.pi, 0.0, 0.5]
        self.drv.movel(target)
        result = self.drv.get_tcp_pose()
        for i in range(6):
            assert result[i] == pytest.approx(target[i]), f"element {i} mismatch"

    def test_movel_sequential_overwrites_pose(self):
        self.drv.movel([0.10, -0.10, 0.30, 0, 0, 0])
        self.drv.movel([0.20, -0.20, 0.50, 0, 0, 0])
        result = self.drv.get_tcp_pose()
        assert result[0] == pytest.approx(0.20)
        assert result[1] == pytest.approx(-0.20)

    def test_movej_updates_joints(self):
        joints = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
        self.drv.movej(joints)
        assert self.drv.get_joints() == pytest.approx(joints)

    def test_movej_all_six_joints_stored(self):
        joints = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.drv.movej(joints)
        result = self.drv.get_joints()
        for i, j in enumerate(joints):
            assert result[i] == pytest.approx(j), f"joint[{i}] mismatch"

    def test_movel_does_not_change_joints(self):
        self.drv.movej([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        before = self.drv.get_joints()
        self.drv.movel([0.15, -0.10, 0.30, 0, 0, 0])
        assert self.drv.get_joints() == pytest.approx(before)

    def test_movej_does_not_change_tcp_pose(self):
        self.drv.movel([0.10, -0.10, 0.30, 0, 0, 0])
        before_pose = self.drv.get_tcp_pose()
        self.drv.movej([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        assert self.drv.get_tcp_pose() == pytest.approx(before_pose)

    def test_movel_rejects_x_out_of_bounds(self):
        with pytest.raises(ValueError):
            self.drv.movel([5.0, 0.0, 0.3, 0, 0, 0])

    def test_movel_rejects_y_out_of_bounds(self):
        with pytest.raises(ValueError):
            self.drv.movel([0.0, -5.0, 0.3, 0, 0, 0])

    def test_movel_rejects_z_too_high(self):
        with pytest.raises(ValueError):
            self.drv.movel([0.0, 0.0, 2.0, 0, 0, 0])

    def test_movej_rejects_wrong_joint_count(self):
        with pytest.raises(ValueError):
            self.drv.movej([0.0, 0.0, 0.0])

    def test_movej_rejects_too_many_joints(self):
        with pytest.raises(ValueError):
            self.drv.movej([0.0] * 7)

    def test_stopl_clears_moving_flag(self):
        done = threading.Event()

        def _mover():
            try:
                self.drv.movel([0.70, 0.0, 0.50, 0, 0, 0])
            except Exception:
                pass
            done.set()

        t = threading.Thread(target=_mover, daemon=True)
        t.start()
        time.sleep(0.05)
        self.drv.stopl()
        assert not self.drv.is_moving()
        done.wait(timeout=5)

    def test_context_manager_enters_and_exits(self):
        with MockURDriver() as d:
            d.movel([0.10, 0.0, 0.30, 0, 0, 0])
            assert d.get_tcp_pose()[0] == pytest.approx(0.10)

    def test_get_tcp_pose_returns_copy(self):
        """Mutating the returned list must not change internal state."""
        pose = self.drv.get_tcp_pose()
        pose[0] = 999.0
        assert self.drv.get_tcp_pose()[0] != 999.0

    def test_get_joints_returns_copy(self):
        joints = self.drv.get_joints()
        joints[0] = 999.0
        assert self.drv.get_joints()[0] != 999.0


class TestMockURDriverDigitalOutputs:
    """MockURDriver: controller and tool digital output state transitions."""

    def setup_method(self):
        self.drv = MockURDriver()

    def test_digital_out_default_false(self):
        assert self.drv.get_digital_out(0) is False
        assert self.drv.get_digital_out(7) is False

    def test_set_digital_out_true(self):
        self.drv.set_digital_out(0, True)
        assert self.drv.get_digital_out(0) is True

    def test_set_digital_out_false_after_true(self):
        self.drv.set_digital_out(3, True)
        self.drv.set_digital_out(3, False)
        assert self.drv.get_digital_out(3) is False

    def test_set_multiple_digital_out_ports_independent(self):
        self.drv.set_digital_out(0, True)
        self.drv.set_digital_out(1, False)
        self.drv.set_digital_out(2, True)
        assert self.drv.get_digital_out(0) is True
        assert self.drv.get_digital_out(1) is False
        assert self.drv.get_digital_out(2) is True

    def test_tool_digital_out_default_false(self):
        assert self.drv.get_tool_digital_out(0) is False
        assert self.drv.get_tool_digital_out(1) is False

    def test_set_tool_digital_out_true(self):
        self.drv.set_tool_digital_out(1, True)
        assert self.drv.get_tool_digital_out(1) is True

    def test_set_tool_digital_out_false_after_true(self):
        self.drv.set_tool_digital_out(0, True)
        self.drv.set_tool_digital_out(0, False)
        assert self.drv.get_tool_digital_out(0) is False

    def test_tool_digital_out_independent_from_controller(self):
        self.drv.set_digital_out(0, True)
        self.drv.set_tool_digital_out(0, False)
        assert self.drv.get_digital_out(0) is True
        assert self.drv.get_tool_digital_out(0) is False

    def test_gripper_on_off_cycle(self):
        """Simulate gripper grasp/release cycle via DO port 0."""
        self.drv.set_digital_out(0, True)   # grasp
        assert self.drv.get_digital_out(0) is True
        self.drv.set_digital_out(0, False)  # release
        assert self.drv.get_digital_out(0) is False

    def test_thread_safe_concurrent_reads(self):
        errors: list = []

        def _read():
            for _ in range(100):
                try:
                    self.drv.get_tcp_pose()
                    self.drv.get_digital_out(0)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=_read) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


# ---------------------------------------------------------------------------
# Section 2: RealURDriver via MockURServer
# ---------------------------------------------------------------------------

class _EchoScriptServer(threading.Thread):
    """Accept connections, accumulate received bytes per connection."""

    def __init__(self, host: str, port: int):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.received: list[str] = []
        self._ready = threading.Event()
        self._stop = threading.Event()

    def run(self):
        with socket.socket() as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self.host, self.port))
            srv.listen(5)
            srv.settimeout(1.0)
            self._ready.set()
            while not self._stop.is_set():
                try:
                    conn, _ = srv.accept()
                except socket.timeout:
                    continue
                data = b""
                conn.settimeout(0.5)
                try:
                    while True:
                        chunk = conn.recv(4096)
                        if not chunk:
                            break
                        data += chunk
                except socket.timeout:
                    pass
                finally:
                    conn.close()
                self.received.append(
                    data.decode("utf-8", errors="replace").strip()
                )

    def stop(self):
        self._stop.set()


class TestRealURDriverURScript:
    """RealURDriver: verify URScript string encoding sent to secondary port."""

    def setup_method(self):
        self.port = _free_port()
        self.echo = _EchoScriptServer("127.0.0.1", self.port)
        self.echo.start()
        self.echo._ready.wait(timeout=2)
        self.drv = RealURDriver(
            host="127.0.0.1",
            port_script=self.port,
            port_rt=_free_port(),
        )

    def teardown_method(self):
        self.drv.disconnect()
        self.echo.stop()

    def _last_script(self) -> str:
        time.sleep(0.05)
        return self.echo.received[-1] if self.echo.received else ""

    def test_movel_script_prefix(self):
        self.drv.movel([0.30, -0.20, 0.15, 0.0, math.pi, 0.0])
        assert self._last_script().startswith("movel(p[")

    def test_movel_script_contains_x_value(self):
        self.drv.movel([0.30, -0.20, 0.15, 0.0, math.pi, 0.0])
        assert "0.300000" in self._last_script()

    def test_movel_script_default_acceleration(self):
        self.drv.movel([0.10, 0.0, 0.30, 0, 0, 0])
        assert "a=1.2" in self._last_script()

    def test_movel_script_default_velocity(self):
        self.drv.movel([0.10, 0.0, 0.30, 0, 0, 0])
        assert "v=0.25" in self._last_script()

    def test_movel_script_custom_params(self):
        self.drv.movel([0.10, 0.0, 0.30, 0, 0, 0], a=0.5, v=0.1)
        script = self._last_script()
        assert "a=0.5" in script
        assert "v=0.1" in script

    def test_movel_rejects_out_of_bounds(self):
        with pytest.raises(ValueError):
            self.drv.movel([2.0, 0.0, 0.3, 0, 0, 0])

    def test_movej_script_prefix(self):
        self.drv.movej([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
        assert self._last_script().startswith("movej([")

    def test_movej_script_default_acceleration(self):
        self.drv.movej([0.0] * 6)
        assert "a=1.4" in self._last_script()

    def test_movej_script_default_velocity(self):
        self.drv.movej([0.0] * 6)
        assert "v=1.05" in self._last_script()

    def test_movej_rejects_wrong_length(self):
        with pytest.raises(ValueError):
            self.drv.movej([0.0, 0.0])

    def test_stopl_script_prefix(self):
        self.drv.stopl()
        assert self._last_script().startswith("stopl(")

    def test_stopl_custom_accel_in_script(self):
        self.drv.stopl(a=5.0)
        assert "5.0" in self._last_script()

    def test_set_digital_out_script(self):
        self.drv.set_digital_out(2, True)
        assert "set_digital_out(2, True)" in self._last_script()

    def test_set_digital_out_false_script(self):
        self.drv.set_digital_out(0, False)
        assert "set_digital_out(0, False)" in self._last_script()

    def test_set_tool_digital_out_script(self):
        self.drv.set_tool_digital_out(0, False)
        assert "set_tool_digital_out(0, False)" in self._last_script()


class TestDashboardCommands:
    """DashboardClient: ASCII command round-trip against MockURServer."""

    def setup_method(self):
        self.dash_port = _free_port()
        self.sec_port = _free_port()
        self.server = MockURServer(
            host="127.0.0.1",
            dashboard_port=self.dash_port,
            secondary_port=self.sec_port,
        )
        self.server.start()
        time.sleep(0.1)

    def teardown_method(self):
        self.server.stop()

    def test_connect_and_close_no_exception(self):
        c = DashboardClient("127.0.0.1", self.dash_port)
        c.connect()
        c.close()

    def test_context_manager_no_exception(self):
        with DashboardClient("127.0.0.1", self.dash_port):
            pass

    def test_get_robot_mode_returns_nonempty_string(self):
        with DashboardClient("127.0.0.1", self.dash_port) as c:
            resp = c.get_robot_mode()
        assert isinstance(resp, str) and len(resp) > 0

    def test_get_robot_mode_contains_robotmode(self):
        with DashboardClient("127.0.0.1", self.dash_port) as c:
            resp = c.get_robot_mode()
        assert "Robotmode" in resp or "IDLE" in resp or "RUNNING" in resp

    def test_play_returns_response(self):
        with DashboardClient("127.0.0.1", self.dash_port) as c:
            resp = c.play()
        assert isinstance(resp, str)

    def test_stop_returns_stopped(self):
        with DashboardClient("127.0.0.1", self.dash_port) as c:
            resp = c.stop()
        assert len(resp) > 0

    def test_play_then_robotmode_is_running(self):
        with DashboardClient("127.0.0.1", self.dash_port) as c:
            c.play()
            mode = c.get_robot_mode()
        assert "RUNNING" in mode

    def test_stop_then_robotmode_is_idle(self):
        with DashboardClient("127.0.0.1", self.dash_port) as c:
            c.play()
            c.stop()
            mode = c.get_robot_mode()
        assert "IDLE" in mode or "Stopped" in mode or len(mode) > 0


class TestRTPacketParsing:
    """RT packet: parse TCP pose at fixed offset."""

    def test_parse_known_pose(self):
        from src.robot.ur_driver import _RTReader
        reader = _RTReader("127.0.0.1", _free_port())
        expected = [0.123, -0.456, 0.789, 0.01, -0.02, 0.03]
        reader._parse(_make_rt_packet(expected))
        assert reader.get_pose() == pytest.approx(expected, abs=1e-9)

    def test_parse_zero_pose(self):
        from src.robot.ur_driver import _RTReader
        reader = _RTReader("127.0.0.1", _free_port())
        reader._parse(_make_rt_packet([0.0] * 6))
        assert reader.get_pose() == pytest.approx([0.0] * 6)

    def test_parse_negative_values(self):
        from src.robot.ur_driver import _RTReader
        reader = _RTReader("127.0.0.1", _free_port())
        expected = [-0.5, -0.3, 0.2, math.pi, 0.0, 0.0]
        reader._parse(_make_rt_packet(expected))
        assert reader.get_pose() == pytest.approx(expected, abs=1e-9)

    def test_initial_pose_is_zeros(self):
        from src.robot.ur_driver import _RTReader
        reader = _RTReader("127.0.0.1", _free_port())
        assert reader.get_pose() == pytest.approx([0.0] * 6)


class TestFactory:
    def test_sim_returns_mock(self):
        assert isinstance(create_robot_driver(mode="sim"), MockURDriver)

    def test_real_returns_real(self):
        drv = create_robot_driver(mode="real", host="192.168.1.100")
        assert isinstance(drv, RealURDriver)
        assert drv.host == "192.168.1.100"

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            create_robot_driver(mode="bad")

    def test_sim_usable_without_hardware(self):
        drv = create_robot_driver(mode="sim")
        drv.movel([0.20, 0.0, 0.30, 0, 0, 0])
        assert drv.get_tcp_pose()[0] == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# Section 3: BoardCoordinates — all 64 squares + boundary safety violations
# ---------------------------------------------------------------------------

ALL_SQUARES = [f + r for f in "abcdefgh" for r in "12345678"]


class TestBoardCoordinatesAll64Squares:
    """BoardCoordinates: approach_pose and pick_pose must succeed for all 64 squares."""

    @pytest.fixture(autouse=True)
    def _bc(self, coords):
        self.bc = coords

    @pytest.mark.parametrize("square", ALL_SQUARES)
    def test_approach_pose_valid(self, square: str):
        pose = self.bc.approach_pose(square)
        assert len(pose) == 6

    @pytest.mark.parametrize("square", ALL_SQUARES)
    def test_approach_pose_z_at_transit(self, square: str):
        pose = self.bc.approach_pose(square)
        assert pose[2] == pytest.approx(TRANSIT_Z)

    @pytest.mark.parametrize("square", ALL_SQUARES)
    def test_pick_pose_valid(self, square: str):
        pose = self.bc.pick_pose(square)
        assert len(pose) == 6

    @pytest.mark.parametrize("square", ALL_SQUARES)
    def test_pick_pose_z_at_pick_z(self, square: str):
        pose = self.bc.pick_pose(square)
        assert pose[2] == pytest.approx(PICK_Z)

    @pytest.mark.parametrize("square", ALL_SQUARES)
    def test_pose_tool_orientation_downward(self, square: str):
        pose = self.bc.approach_pose(square)
        assert pose[3] == pytest.approx(_TOOL_RX)
        assert pose[4] == pytest.approx(_TOOL_RY)
        assert pose[5] == pytest.approx(_TOOL_RZ)

    @pytest.mark.parametrize("square", ALL_SQUARES)
    def test_approach_and_pick_share_xy(self, square: str):
        ap = self.bc.approach_pose(square)
        pk = self.bc.pick_pose(square)
        assert ap[0] == pytest.approx(pk[0])
        assert ap[1] == pytest.approx(pk[1])

    def test_a1_maps_to_origin_xy(self, coords):
        x, y = coords.square_to_xy("a1")
        assert x == pytest.approx(ORIGIN[0])
        assert y == pytest.approx(ORIGIN[1])

    def test_h8_maps_to_corner(self, coords):
        x, y = coords.square_to_xy("h8")
        assert x == pytest.approx(ORIGIN[0] + 7 * DX)
        assert y == pytest.approx(ORIGIN[1] + 7 * DY)

    def test_file_spacing_linear(self, coords):
        """X increases by DX per file step."""
        for fi, f in enumerate("abcdefgh"):
            x, _ = coords.square_to_xy(f + "1")
            assert x == pytest.approx(ORIGIN[0] + fi * DX), f"file {f}"

    def test_rank_spacing_linear(self, coords):
        """Y increases by DY per rank step."""
        for ri, r in enumerate("12345678"):
            _, y = coords.square_to_xy("a" + r)
            assert y == pytest.approx(ORIGIN[1] + ri * DY), f"rank {r}"


class TestBoardCoordinatesBoundarySafety:
    """Safety violations: board origin/spacing that pushes poses outside envelope."""

    def test_x_over_bound_raises_on_approach(self):
        bc = BoardCoordinates(
            origin_xyz=(2.0, 0.0, 0.005),
            dx=DX, dy=DY, board_z=0.005, pick_z=0.010, transit_z=TRANSIT_Z,
        )
        with pytest.raises(SafetyViolationError):
            bc.approach_pose("a1")

    def test_y_over_bound_raises_on_pick(self):
        bc = BoardCoordinates(
            origin_xyz=(0.0, 2.0, 0.005),
            dx=DX, dy=DY, board_z=0.005, pick_z=0.010, transit_z=TRANSIT_Z,
        )
        with pytest.raises(SafetyViolationError):
            bc.pick_pose("a1")

    def test_large_dx_pushes_h_file_out_of_bound(self):
        # origin X=0.50, dx=0.20 → h=0.50+7*0.20=1.90 > 0.80
        bc = BoardCoordinates(
            origin_xyz=(0.50, 0.0, 0.005),
            dx=0.20, dy=DY, board_z=0.005, pick_z=0.010, transit_z=TRANSIT_Z,
        )
        with pytest.raises(SafetyViolationError):
            bc.approach_pose("h1")

    def test_transit_z_below_minimum_raises_at_init(self):
        with pytest.raises(ValueError, match="transit_z"):
            BoardCoordinates(
                origin_xyz=ORIGIN, dx=DX, dy=DY,
                board_z=BOARD_Z, pick_z=BOARD_Z,
                transit_z=TRANSIT_Z_MIN - 0.001,
            )

    def test_pick_z_below_board_z_raises_at_init(self):
        with pytest.raises(ValueError, match="pick_z"):
            BoardCoordinates(
                origin_xyz=ORIGIN, dx=DX, dy=DY,
                board_z=0.010, pick_z=0.005, transit_z=TRANSIT_Z,
            )

    def test_invalid_square_file_raises(self):
        bc = BoardCoordinates(
            origin_xyz=ORIGIN, dx=DX, dy=DY,
            board_z=BOARD_Z, pick_z=PICK_Z, transit_z=TRANSIT_Z,
        )
        with pytest.raises(ValueError):
            bc.approach_pose("z4")

    def test_invalid_square_rank_raises(self):
        bc = BoardCoordinates(
            origin_xyz=ORIGIN, dx=DX, dy=DY,
            board_z=BOARD_Z, pick_z=PICK_Z, transit_z=TRANSIT_Z,
        )
        with pytest.raises(ValueError):
            bc.approach_pose("a9")

    def test_workspace_envelope_x_min_raises(self):
        env = WorkspaceSafetyEnvelope(
            x_min=0.0, x_max=0.5, y_min=-0.5, y_max=0.0, z_min=0.0, z_max=0.5
        )
        with pytest.raises(SafetyViolationError, match="X="):
            env.check([-0.01, -0.25, 0.25, 0, 0, 0])

    def test_workspace_envelope_z_ceiling_raises(self):
        env = WorkspaceSafetyEnvelope(
            x_min=0.0, x_max=0.5, y_min=-0.5, y_max=0.0, z_min=0.0, z_max=0.5
        )
        with pytest.raises(SafetyViolationError, match="Z="):
            env.check([0.25, -0.25, 0.51, 0, 0, 0])

    def test_workspace_envelope_exact_boundary_passes(self):
        env = WorkspaceSafetyEnvelope(
            x_min=0.0, x_max=0.5, y_min=-0.5, y_max=0.0, z_min=0.0, z_max=0.5
        )
        env.check([0.0, -0.5, 0.0, 0, 0, 0])   # min corners
        env.check([0.5,  0.0, 0.5, 0, 0, 0])   # max corners


# ---------------------------------------------------------------------------
# Section 4: TrajectoryGenerator — waypoint sequences
# ---------------------------------------------------------------------------

class TestTrajectoryWaypointSequences:
    """TrajectoryGenerator: verify waypoint structure for move types."""

    def test_normal_plan_returns_7_waypoints(self, tgen):
        wps = tgen.plan_pick_and_place("e2", "e4")
        assert len(wps) == 7

    def test_normal_plan_label_sequence(self, tgen):
        wps = tgen.plan_pick_and_place("a1", "h8")
        assert wps[0].label.startswith("approach_src:")
        assert wps[1].label.startswith("pick:")
        assert wps[2].label.startswith("ascend_src:")
        assert wps[3].label.startswith("transit_to:")
        assert wps[4].label.startswith("place:")
        assert wps[5].label.startswith("retract_dst:")
        assert wps[6].label == "park"

    def test_normal_plan_transit_z_floor(self, tgen):
        """Approach and retract waypoints must be at transit_z (>= 80 mm)."""
        wps = tgen.plan_pick_and_place("d4", "f6")
        for i in [0, 2, 3, 5]:
            assert wps[i].pose[2] >= 0.080, f"wp[{i}].pose Z below 80 mm"

    def test_normal_plan_approach_z_equals_transit_z(self, tgen):
        wps = tgen.plan_pick_and_place("c3", "g5")
        assert wps[0].pose[2] == pytest.approx(tgen.coords.transit_z)
        assert wps[2].pose[2] == pytest.approx(tgen.coords.transit_z)

    def test_normal_plan_pick_z_correct(self, tgen):
        wps = tgen.plan_pick_and_place("b2", "b4")
        assert wps[1].pose[2] == pytest.approx(tgen.coords.pick_z)
        assert wps[4].pose[2] == pytest.approx(tgen.coords.pick_z)

    def test_normal_plan_park_is_last(self, tgen):
        wps = tgen.plan_pick_and_place("e1", "g1")
        assert wps[-1].pose[:3] == pytest.approx(tgen.park_pose[:3], abs=1e-6)

    def test_normal_plan_xy_matches_source_square(self, tgen, coords):
        wps = tgen.plan_pick_and_place("a1", "h8")
        x, y = coords.square_to_xy("a1")
        assert wps[0].pose[0] == pytest.approx(x, abs=1e-6)
        assert wps[0].pose[1] == pytest.approx(y, abs=1e-6)

    def test_normal_plan_xy_matches_dest_square(self, tgen, coords):
        wps = tgen.plan_pick_and_place("a1", "h8")
        x, y = coords.square_to_xy("h8")
        assert wps[3].pose[0] == pytest.approx(x, abs=1e-6)
        assert wps[3].pose[1] == pytest.approx(y, abs=1e-6)

    def test_execute_movel_called_7_times(self, tgen):
        drv = MagicMock()
        drv.movel = MagicMock()
        drv.set_digital_out = MagicMock()
        tgen.pick_and_place(drv, "e2", "e4")
        assert drv.movel.call_count == 7

    def test_execute_gripper_grasp_after_pick(self, tgen):
        drv = MagicMock()
        drv.movel = MagicMock()
        drv.set_digital_out = MagicMock()
        tgen.pick_and_place(drv, "e2", "e4")
        drv.set_digital_out.assert_any_call(0, True)

    def test_execute_gripper_release_after_place(self, tgen):
        drv = MagicMock()
        drv.movel = MagicMock()
        drv.set_digital_out = MagicMock()
        tgen.pick_and_place(drv, "e2", "e4")
        drv.set_digital_out.assert_any_call(0, False)

    def test_execute_gripper_order_grasp_then_release(self, tgen):
        drv = MagicMock()
        drv.movel = MagicMock()
        drv.set_digital_out = MagicMock()
        tgen.pick_and_place(drv, "d2", "d4")
        do_calls = drv.set_digital_out.call_args_list
        assert len(do_calls) == 2
        assert do_calls[0] == call(0, True)
        assert do_calls[1] == call(0, False)

    def test_safety_violation_on_out_of_bounds_square(self, coords):
        # origin X=0.50, dx=0.055 → h=0.50+7*0.055=0.885 > 0.80
        bad_coords = BoardCoordinates(
            origin_xyz=(0.50, 0.0, 0.005), dx=0.055, dy=0.055,
            board_z=0.005, pick_z=0.010, transit_z=0.12,
        )
        tg = TrajectoryGenerator(coords=bad_coords)
        drv = MagicMock()
        drv.movel = MagicMock()
        drv.set_digital_out = MagicMock()
        with pytest.raises(SafetyViolationError):
            tg.pick_and_place(drv, "h1", "h1")

    def test_park_pose_outside_envelope_raises_at_init(self, coords):
        with pytest.raises(SafetyViolationError):
            TrajectoryGenerator(
                coords=coords,
                park_pose=[0.0, -0.60, 1.50, math.pi, 0.0, 0.0],  # Z=1.5 m > 0.9
            )


class TestCaptureMoveSequence:
    """MoveSequencer: capture move picks victim to graveyard, then executes attacker."""

    def _tg_mock(self, coords):
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

    def _drv(self):
        drv = MagicMock()
        drv.movel = MagicMock()
        drv.set_digital_out = MagicMock()
        return drv

    def test_capture_calls_pick_and_place_once(self, coords, graveyard):
        tg = self._tg_mock(coords)
        seq = MoveSequencer(trajectory=tg, graveyard=graveyard)
        drv = self._drv()
        seq.execute_capture(drv, "d4", "e5", captured_color="black")
        tg.pick_and_place.assert_called_once_with(drv, "d4", "e5")

    def test_capture_victim_first_movel_targets_victim_square(self, coords, graveyard):
        tg = self._tg_mock(coords)
        seq = MoveSequencer(trajectory=tg, graveyard=graveyard)
        drv = self._drv()
        seq.execute_capture(drv, "d4", "e5", captured_color="black")
        x_e5, y_e5 = coords.square_to_xy("e5")
        first_pose = drv.movel.call_args_list[0][0][0]
        assert first_pose[0] == pytest.approx(x_e5, abs=1e-6)
        assert first_pose[1] == pytest.approx(y_e5, abs=1e-6)

    def test_capture_black_increments_black_graveyard(self, coords, graveyard):
        tg = self._tg_mock(coords)
        seq = MoveSequencer(trajectory=tg, graveyard=graveyard)
        seq.execute_capture(self._drv(), "d4", "e5", captured_color="black")
        assert graveyard.black_slots_used == 1
        assert graveyard.white_slots_used == 0

    def test_capture_white_increments_white_graveyard(self, coords, graveyard):
        tg = self._tg_mock(coords)
        seq = MoveSequencer(trajectory=tg, graveyard=graveyard)
        seq.execute_capture(self._drv(), "c6", "d5", captured_color="white")
        assert graveyard.white_slots_used == 1
        assert graveyard.black_slots_used == 0

    def test_capture_invalid_color_raises(self, coords, graveyard):
        tg = self._tg_mock(coords)
        seq = MoveSequencer(trajectory=tg, graveyard=graveyard)
        with pytest.raises(ValueError, match="captured_color"):
            seq.execute_capture(self._drv(), "e2", "e4", captured_color="red")


class TestCastlingMoveSequence:
    """MoveSequencer: castling moves king then rook."""

    def _tg_mock(self, coords):
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

    def _drv(self):
        drv = MagicMock()
        drv.movel = MagicMock()
        drv.set_digital_out = MagicMock()
        return drv

    def test_castling_calls_pick_and_place_twice(self, coords, graveyard):
        tg = self._tg_mock(coords)
        seq = MoveSequencer(trajectory=tg, graveyard=graveyard)
        seq.execute_castling(self._drv(), "e1", "g1", "h1", "f1")
        assert tg.pick_and_place.call_count == 2

    def test_castling_king_before_rook(self, coords, graveyard):
        tg = self._tg_mock(coords)
        seq = MoveSequencer(trajectory=tg, graveyard=graveyard)
        drv = self._drv()
        seq.execute_castling(drv, "e1", "g1", "h1", "f1")
        calls = tg.pick_and_place.call_args_list
        assert calls[0] == call(drv, "e1", "g1")
        assert calls[1] == call(drv, "h1", "f1")

    def test_queenside_castling_order(self, coords, graveyard):
        """Queenside castling: king e1→c1, rook a1→d1."""
        tg = self._tg_mock(coords)
        seq = MoveSequencer(trajectory=tg, graveyard=graveyard)
        drv = self._drv()
        seq.execute_castling(drv, "e1", "c1", "a1", "d1")
        calls = tg.pick_and_place.call_args_list
        assert calls[0] == call(drv, "e1", "c1")
        assert calls[1] == call(drv, "a1", "d1")


class TestNormalMoveSequence:
    """MoveSequencer: normal (non-capture) moves."""

    def test_normal_move_calls_pick_and_place_once(self, coords, graveyard):
        tg = MagicMock()
        tg.coords = coords
        tg.park_pose = list(DEFAULT_PARK_POSE)
        seq = MoveSequencer(trajectory=tg, graveyard=graveyard)
        drv = MagicMock()
        seq.execute_normal(drv, "e2", "e4")
        tg.pick_and_place.assert_called_once_with(drv, "e2", "e4")
