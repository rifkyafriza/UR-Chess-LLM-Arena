"""
Unit tests for src/robot/ur_driver.py.

Tests cover:
    - MockURDriver: pose tracking, joint tracking, digital outputs, safety limits
    - DashboardClient: ASCII command round-trip against MockURServer
    - RealURDriver: URScript formatting, safety guard, telemetry parsing
    - Factory: create_robot_driver() mode selection

All tests run without hardware (MockURServer for socket tests).
"""

import math
import socket
import struct
import threading
import time
import pytest

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


def _make_rt_packet(pose: list[float]) -> bytes:
    """Build a minimal 1116-byte RT packet with the given TCP pose."""
    buf = bytearray(_RT_PACKET_LEN)
    struct.pack_into(">I", buf, 0, _RT_PACKET_LEN)
    struct.pack_into(">6d", buf, _RT_TCP_POSE_OFFSET, *pose)
    return bytes(buf)


# ---------------------------------------------------------------------------
# _check_pose_limits
# ---------------------------------------------------------------------------

class TestSafetyLimits:
    def test_valid_pose_passes(self):
        _check_pose_limits([0.3, -0.2, 0.4, 0.0, 3.14, 0.0])  # no raise

    def test_x_over_limit_raises(self):
        with pytest.raises(ValueError, match="X="):
            _check_pose_limits([1.0, 0.0, 0.3, 0, 0, 0])

    def test_y_under_limit_raises(self):
        with pytest.raises(ValueError, match="Y="):
            _check_pose_limits([0.0, -1.0, 0.3, 0, 0, 0])

    def test_z_under_floor_raises(self):
        with pytest.raises(ValueError, match="Z="):
            _check_pose_limits([0.0, 0.0, -0.5, 0, 0, 0])

    def test_z_over_ceiling_raises(self):
        with pytest.raises(ValueError, match="Z="):
            _check_pose_limits([0.0, 0.0, 1.5, 0, 0, 0])


# ---------------------------------------------------------------------------
# MockURDriver
# ---------------------------------------------------------------------------

class TestMockURDriver:
    def setup_method(self):
        self.drv = MockURDriver()

    def test_initial_pose_is_list_of_six(self):
        pose = self.drv.get_tcp_pose()
        assert len(pose) == 6

    def test_movel_updates_pose(self):
        target = [0.2, -0.3, 0.4, 0.0, 3.14, 0.0]
        self.drv.movel(target)
        assert self.drv.get_tcp_pose() == pytest.approx(target)

    def test_movej_updates_joints(self):
        joints = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
        self.drv.movej(joints)
        assert self.drv.get_joints() == pytest.approx(joints)

    def test_movel_rejects_unsafe_pose(self):
        with pytest.raises(ValueError):
            self.drv.movel([5.0, 0.0, 0.3, 0, 0, 0])

    def test_movej_rejects_wrong_length(self):
        with pytest.raises(ValueError):
            self.drv.movej([0.0, 0.0, 0.0])

    def test_digital_out_tracking(self):
        self.drv.set_digital_out(0, True)
        assert self.drv.get_digital_out(0) is True
        self.drv.set_digital_out(0, False)
        assert self.drv.get_digital_out(0) is False

    def test_tool_digital_out_tracking(self):
        self.drv.set_tool_digital_out(1, True)
        assert self.drv.get_tool_digital_out(1) is True

    def test_stopl_clears_moving_flag(self):
        # Start a move in background, stop it immediately
        done = threading.Event()

        def _mover():
            # far move → long sim travel time
            try:
                self.drv.movel([0.7, 0.0, 0.5, 0, 0, 0])
            except Exception:
                pass
            done.set()

        t = threading.Thread(target=_mover, daemon=True)
        t.start()
        time.sleep(0.05)  # let move start
        self.drv.stopl()
        # is_moving should be False immediately after stopl
        assert not self.drv.is_moving()
        done.wait(timeout=5)

    def test_context_manager(self):
        with MockURDriver() as d:
            d.movel([0.1, 0.0, 0.3, 0, 0, 0])
            assert d.get_tcp_pose()[0] == pytest.approx(0.1)

    def test_thread_safe_concurrent_reads(self):
        """Two threads reading pose simultaneously must not raise."""
        errors = []

        def _read():
            for _ in range(50):
                try:
                    self.drv.get_tcp_pose()
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=_read) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


# ---------------------------------------------------------------------------
# DashboardClient via MockURServer
# ---------------------------------------------------------------------------

class TestDashboardClient:
    def setup_method(self):
        self.dash_port = _free_port()
        self.sec_port = _free_port()
        self.server = MockURServer(
            host="127.0.0.1",
            dashboard_port=self.dash_port,
            secondary_port=self.sec_port,
        )
        self.server.start()
        time.sleep(0.1)  # give server threads time to bind

    def teardown_method(self):
        self.server.stop()

    def test_connect_and_close(self):
        c = DashboardClient("127.0.0.1", self.dash_port)
        c.connect()
        c.close()  # no exception

    def test_get_robot_mode_returns_string(self):
        with DashboardClient("127.0.0.1", self.dash_port) as c:
            resp = c.get_robot_mode()
        assert isinstance(resp, str)
        assert len(resp) > 0

    def test_play_returns_response(self):
        with DashboardClient("127.0.0.1", self.dash_port) as c:
            resp = c.play()
        assert isinstance(resp, str)

    def test_stop_returns_response(self):
        with DashboardClient("127.0.0.1", self.dash_port) as c:
            resp = c.stop()
        assert "Stopped" in resp or len(resp) > 0


# ---------------------------------------------------------------------------
# RealURDriver – script formatting (uses a minimal echo server)
# ---------------------------------------------------------------------------

class _EchoScriptServer(threading.Thread):
    """Listens on a port, stores every received line, closes connection."""

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
                self.received.append(data.decode("utf-8", errors="replace").strip())

    def stop(self):
        self._stop.set()


class TestRealURDriverScripts:
    def setup_method(self):
        self.port = _free_port()
        self.echo = _EchoScriptServer("127.0.0.1", self.port)
        self.echo.start()
        self.echo._ready.wait(timeout=2)
        self.drv = RealURDriver(
            host="127.0.0.1",
            port_script=self.port,
            port_rt=_free_port(),  # RT reader will fail to connect but that's fine
        )

    def teardown_method(self):
        self.drv.disconnect()
        self.echo.stop()

    def _last_script(self) -> str:
        time.sleep(0.05)  # give echo server time to receive
        return self.echo.received[-1] if self.echo.received else ""

    def test_movel_sends_correct_urscript(self):
        pose = [0.3, -0.2, 0.15, 0.0, 3.14159, 0.0]
        self.drv.movel(pose)
        script = self._last_script()
        assert script.startswith("movel(p[")
        assert "0.300000" in script
        assert f"a=1.2" in script
        assert f"v=0.25" in script

    def test_movel_custom_params(self):
        pose = [0.1, 0.0, 0.3, 0, 0, 0]
        self.drv.movel(pose, a=0.5, v=0.1)
        script = self._last_script()
        assert "a=0.5" in script
        assert "v=0.1" in script

    def test_movel_rejects_unsafe(self):
        with pytest.raises(ValueError):
            self.drv.movel([2.0, 0.0, 0.3, 0, 0, 0])

    def test_movej_sends_correct_urscript(self):
        joints = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
        self.drv.movej(joints)
        script = self._last_script()
        assert script.startswith("movej([")
        assert "a=1.4" in script
        assert "v=1.05" in script

    def test_movej_rejects_wrong_length(self):
        with pytest.raises(ValueError):
            self.drv.movej([0.0, 0.0])

    def test_stopl_sends_stopl(self):
        self.drv.stopl()
        script = self._last_script()
        assert script.startswith("stopl(")

    def test_stopl_custom_accel(self):
        self.drv.stopl(a=5.0)
        script = self._last_script()
        assert "5.0" in script

    def test_set_digital_out(self):
        self.drv.set_digital_out(2, True)
        script = self._last_script()
        assert "set_digital_out(2, True)" in script

    def test_set_tool_digital_out(self):
        self.drv.set_tool_digital_out(0, False)
        script = self._last_script()
        assert "set_tool_digital_out(0, False)" in script


# ---------------------------------------------------------------------------
# RT telemetry packet parsing (unit-level, no socket)
# ---------------------------------------------------------------------------

class TestRTPacketParsing:
    def test_parse_known_pose(self):
        """_RTReader._parse must extract pose from a crafted packet."""
        from src.robot.ur_driver import _RTReader

        reader = _RTReader("127.0.0.1", _free_port())  # port not used here
        expected = [0.123, -0.456, 0.789, 0.01, -0.02, 0.03]
        pkt = _make_rt_packet(expected)
        reader._parse(pkt)
        result = reader.get_pose()
        assert result == pytest.approx(expected, abs=1e-9)

    def test_parse_zero_pose(self):
        from src.robot.ur_driver import _RTReader

        reader = _RTReader("127.0.0.1", _free_port())
        pkt = _make_rt_packet([0.0] * 6)
        reader._parse(pkt)
        assert reader.get_pose() == pytest.approx([0.0] * 6)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestFactory:
    def test_sim_returns_mock(self):
        drv = create_robot_driver(mode="sim")
        assert isinstance(drv, MockURDriver)

    def test_real_returns_real(self):
        drv = create_robot_driver(mode="real", host="192.168.1.100")
        assert isinstance(drv, RealURDriver)
        assert drv.host == "192.168.1.100"

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            create_robot_driver(mode="invalid")

    def test_sim_usable_without_hardware(self):
        drv = create_robot_driver(mode="sim")
        drv.movel([0.2, 0.0, 0.3, 0, 0, 0])
        assert drv.get_tcp_pose()[0] == pytest.approx(0.2)
