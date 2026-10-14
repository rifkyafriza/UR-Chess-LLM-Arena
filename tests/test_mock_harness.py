"""
Automated tests for mock_camera and mock_ur_server harnesses.
Run: python -m pytest tests/test_mock_harness.py -v
"""

import socket
import struct
import time
import numpy as np
import cv2
import pytest

from tests.mock_camera import MockCamera, _BOARD_SIZE, _SQUARE_PX, _ARUCO_DICT
from tests.mock_ur_server import MockURServer


# ============================================================
# MockCamera tests
# ============================================================

class TestMockCamera:

    def test_frame_shape(self):
        cam = MockCamera()
        ok, frame = cam.read()
        assert ok
        assert frame.shape == (_BOARD_SIZE, _BOARD_SIZE, 3)
        assert frame.dtype == np.uint8

    def test_no_noise_deterministic(self):
        cam = MockCamera(noise_sigma=0)
        f1 = cam.get_frame()
        f2 = cam.get_frame()
        assert np.array_equal(f1, f2), "Zero-noise frames must be identical"

    def test_noise_adds_variation(self):
        cam = MockCamera(noise_sigma=5)
        f1 = cam.get_frame()
        f2 = cam.get_frame()
        assert not np.array_equal(f1, f2), "Noisy frames should differ"

    def test_aruco_markers_detected(self):
        squares = ["a1", "e4", "h8"]
        cam = MockCamera(piece_squares=squares, noise_sigma=0)
        frame = cam.get_frame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(_ARUCO_DICT, params)
        corners, ids, _ = detector.detectMarkers(gray)
        assert ids is not None, "No ArUco markers detected in frame"
        detected_ids = set(ids.flatten().tolist())
        # a1 = 0*8+0=0, e4 = 3*8+4=28, h8 = 7*8+7=63
        assert 0 in detected_ids, "Marker for a1 (id=0) not found"
        assert 28 in detected_ids, "Marker for e4 (id=28) not found"
        assert 63 in detected_ids, "Marker for h8 (id=63) not found"

    def test_update_pieces_rebuilds_frame(self):
        cam = MockCamera(piece_squares=["a1"], noise_sigma=0)
        frame_before = cam.get_frame()
        cam.update_pieces(["h8"])
        frame_after = cam.get_frame()
        assert not np.array_equal(frame_before, frame_after), \
            "Frame must change after update_pieces"

    def test_invalid_square_ignored(self):
        # Should not raise; bad squares are skipped
        cam = MockCamera(piece_squares=["z9", "a0", "e4"], noise_sigma=0)
        ok, frame = cam.read()
        assert ok
        assert frame is not None


# ============================================================
# MockURServer — Dashboard protocol tests
# ============================================================

class TestMockURServerDashboard:

    @pytest.fixture
    def server(self):
        s = MockURServer(dashboard_port=39999, secondary_port=39002)
        s.start()
        time.sleep(0.1)   # let sockets bind
        yield s
        s.stop()

    def _connect(self, port: int) -> socket.socket:
        sock = socket.create_connection(("127.0.0.1", port), timeout=5)
        return sock

    def _recv_line(self, sock: socket.socket) -> str:
        buf = b""
        while b"\n" not in buf:
            buf += sock.recv(256)
        return buf.decode().strip()

    def test_dashboard_banner(self, server):
        with self._connect(server.dashboard_port) as sock:
            line = self._recv_line(sock)
            assert "Universal Robots" in line

    def test_robotmode_idle(self, server):
        with self._connect(server.dashboard_port) as sock:
            self._recv_line(sock)   # banner
            sock.sendall(b"robotmode\n")
            resp = self._recv_line(sock)
            assert "IDLE" in resp.upper()

    def test_play_pause_stop(self, server):
        with self._connect(server.dashboard_port) as sock:
            self._recv_line(sock)   # banner
            for cmd, expected in [
                (b"play\n", "Starting"),
                (b"pause\n", "Pausing"),
                (b"stop\n", "Stopped"),
            ]:
                sock.sendall(cmd)
                resp = self._recv_line(sock)
                assert expected.lower() in resp.lower(), \
                    f"Expected '{expected}' in response to {cmd!r}, got {resp!r}"

    def test_unknown_command(self, server):
        with self._connect(server.dashboard_port) as sock:
            self._recv_line(sock)
            sock.sendall(b"boguscommand\n")
            resp = self._recv_line(sock)
            assert "unknown" in resp.lower()

    def test_quit_disconnects(self, server):
        with self._connect(server.dashboard_port) as sock:
            self._recv_line(sock)
            sock.sendall(b"quit\n")
            resp = self._recv_line(sock)
            assert "disconnected" in resp.lower()


# ============================================================
# MockURServer — Secondary interface stream tests
# ============================================================

class TestMockURServerSecondary:

    @pytest.fixture
    def server(self):
        s = MockURServer(dashboard_port=49999, secondary_port=49002)
        s.start()
        time.sleep(0.1)
        yield s
        s.stop()

    def test_packet_length(self, server):
        with socket.create_connection(("127.0.0.1", server.secondary_port), timeout=5) as sock:
            sock.settimeout(2)
            data = b""
            while len(data) < 1060:
                data += sock.recv(4096)
            pkt_len = struct.unpack_from(">I", data, 0)[0]
            assert pkt_len == 1060, f"Expected packet length 1060, got {pkt_len}"

    def test_message_type(self, server):
        with socket.create_connection(("127.0.0.1", server.secondary_port), timeout=5) as sock:
            sock.settimeout(2)
            data = b""
            while len(data) < 1060:
                data += sock.recv(4096)
            msg_type = data[4]
            assert msg_type == 16, f"Expected msg type 16, got {msg_type}"

    def test_joint_angles_zero_at_start(self, server):
        with socket.create_connection(("127.0.0.1", server.secondary_port), timeout=5) as sock:
            sock.settimeout(2)
            data = b""
            while len(data) < 1060:
                data += sock.recv(4096)
            joints = [struct.unpack_from(">d", data, 5 + i * 8)[0] for i in range(6)]
            assert all(abs(j) < 0.01 for j in joints), \
                f"Expected all joints near 0 at start, got {joints}"

    def test_joint_motion_toward_target(self, server):
        """After move_joints(), joint values should approach target over time."""
        target = [0.5, -0.3, 0.1, 0.0, 0.2, -0.1]
        server.move_joints(target)
        time.sleep(0.5)   # let tick loop advance joints
        state_joints = server._state.snapshot()
        # At least one joint should have moved from 0
        assert any(abs(j) > 0.01 for j in state_joints), \
            f"Joints did not move after move_joints(); got {state_joints}"

    def test_multiple_clients_simultaneous(self, server):
        """Two secondary clients must each receive a valid stream."""
        import concurrent.futures

        def read_one():
            with socket.create_connection(("127.0.0.1", server.secondary_port), timeout=5) as sock:
                sock.settimeout(2)
                data = b""
                while len(data) < 1060:
                    data += sock.recv(4096)
                return struct.unpack_from(">I", data, 0)[0]

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            futures = [ex.submit(read_one) for _ in range(2)]
            results = [f.result() for f in futures]
        assert all(r == 1060 for r in results), f"Bad packet lengths: {results}"


# ============================================================
# Integration: camera + server together
# ============================================================

class TestIntegration:

    def test_camera_server_independent(self):
        """MockCamera and MockURServer run concurrently without interference."""
        cam = MockCamera(piece_squares=["d4", "e5"], noise_sigma=0)
        server = MockURServer(dashboard_port=59999, secondary_port=59002)
        server.start()
        time.sleep(0.05)
        try:
            ok, frame = cam.read()
            assert ok and frame.shape == (_BOARD_SIZE, _BOARD_SIZE, 3)
            assert server.robot_mode == "IDLE"
        finally:
            server.stop()
