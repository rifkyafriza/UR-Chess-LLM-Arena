"""
Mock UR robot socket server for headless tests.

Simulates the UR Dashboard Server (port 29999) and the UR Secondary Interface
(port 30002) well enough for integration tests that don't need physical hardware.

Dashboard (port 29999):
    Text-based command/response protocol.
    Supported commands: robotmode, play, pause, stop, quit
    Responds with realistic UR Dashboard strings.

Secondary Interface (port 30002):
    Binary TCP stream: every ~8 ms sends a 1060-byte packet whose first 4
    bytes are packet length (big-endian int32) and byte[4] is message type 16
    (ROBOT_STATE). The joint angles advance linearly toward a target so motion
    commands feel realistic in time-based tests.

Usage:
    server = MockURServer(dashboard_port=29999, secondary_port=30002)
    server.start()
    # ... run tests ...
    server.stop()

    # Or as context manager:
    with MockURServer() as s:
        sock = socket.create_connection(("127.0.0.1", s.dashboard_port))
        ...
"""

import socket
import struct
import threading
import time
import logging

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal state shared between server threads
# ---------------------------------------------------------------------------

class _RobotState:
    """Mutable robot state; thread-safe via lock."""

    _MODES = ("POWER_OFF", "BOOTING", "ROBOT_OFF", "ROBOT_ON",
              "IDLE", "BACKDRIVE", "RUNNING")

    def __init__(self):
        self._lock = threading.Lock()
        self.mode = "IDLE"
        # 6 joints in radians
        self.joints = [0.0] * 6
        self.target = [0.0] * 6
        self.speed = 0.05  # rad per tick (8 ms)
        self.running = False

    def set_target(self, joints: list[float]):
        with self._lock:
            self.target = list(joints)
            self.mode = "RUNNING"
            self.running = True

    def tick(self):
        """Advance joints toward target."""
        with self._lock:
            reached = True
            for i in range(6):
                delta = self.target[i] - self.joints[i]
                if abs(delta) < self.speed:
                    self.joints[i] = self.target[i]
                else:
                    self.joints[i] += self.speed * (1 if delta > 0 else -1)
                    reached = False
            if reached and self.running:
                self.mode = "IDLE"
                self.running = False

    def snapshot(self) -> list[float]:
        with self._lock:
            return list(self.joints)

    def get_mode(self) -> str:
        with self._lock:
            return self.mode

    def set_mode(self, mode: str):
        with self._lock:
            self.mode = mode


# ---------------------------------------------------------------------------
# Dashboard Server (port 29999)
# ---------------------------------------------------------------------------

def _handle_dashboard(conn: socket.socket, state: _RobotState):
    conn.sendall(b"Connected: Universal Robots Dashboard Server\n")
    try:
        buf = b""
        while True:
            chunk = conn.recv(256)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                cmd = line.strip().decode("utf-8", errors="replace").lower()
                if cmd == "robotmode":
                    mode = state.get_mode()
                    conn.sendall(f"Robotmode: {mode}\n".encode())
                elif cmd == "play":
                    state.set_mode("RUNNING")
                    conn.sendall(b"Starting program\n")
                elif cmd == "pause":
                    state.set_mode("PAUSED")
                    conn.sendall(b"Pausing program\n")
                elif cmd == "stop":
                    state.set_mode("IDLE")
                    conn.sendall(b"Stopped\n")
                elif cmd in ("quit", "exit", "disconnect"):
                    conn.sendall(b"Disconnected\n")
                    return
                else:
                    conn.sendall(f"Unknown command: {cmd}\n".encode())
    except (ConnectionResetError, BrokenPipeError, OSError):
        pass
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Secondary Interface Stream (port 30002)
# ---------------------------------------------------------------------------

_PACKET_LEN = 1060        # fixed size; real UR varies but 1060 is common
_MSG_TYPE_ROBOT_STATE = 16
_TICK_INTERVAL = 0.008    # 8 ms = 125 Hz


def _build_robot_state_packet(joints: list[float]) -> bytes:
    """
    Build a minimal robot state packet.
    Layout:
        Bytes 0-3:  packet length (big-endian int32) = 1060
        Byte  4:    message type = 16
        Bytes 5-52: 6x joint angles (float64, big-endian)
        Bytes 53-1059: zeros (unused fields)
    """
    buf = bytearray(_PACKET_LEN)
    struct.pack_into(">I", buf, 0, _PACKET_LEN)
    buf[4] = _MSG_TYPE_ROBOT_STATE
    for i, angle in enumerate(joints):
        struct.pack_into(">d", buf, 5 + i * 8, angle)
    return bytes(buf)


def _handle_secondary(conn: socket.socket, state: _RobotState, stop_evt: threading.Event):
    try:
        while not stop_evt.is_set():
            joints = state.snapshot()
            pkt = _build_robot_state_packet(joints)
            try:
                conn.sendall(pkt)
            except (BrokenPipeError, ConnectionResetError, OSError):
                break
            time.sleep(_TICK_INTERVAL)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Main server class
# ---------------------------------------------------------------------------

class MockURServer:
    """
    Headless mock UR robot server.

    Parameters
    ----------
    host : str
        Bind address (default "127.0.0.1").
    dashboard_port : int
        Dashboard port (default 29999).
    secondary_port : int
        Secondary interface port (default 30002).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        dashboard_port: int = 29999,
        secondary_port: int = 30002,
    ):
        self.host = host
        self.dashboard_port = dashboard_port
        self.secondary_port = secondary_port
        self._state = _RobotState()
        self._stop_evt = threading.Event()
        self._threads: list[threading.Thread] = []

    # -- public API ----------------------------------------------------------

    def start(self):
        """Start dashboard and secondary listener threads."""
        self._stop_evt.clear()
        self._threads.append(threading.Thread(
            target=self._serve_dashboard, daemon=True, name="mock-ur-dashboard"))
        self._threads.append(threading.Thread(
            target=self._serve_secondary, daemon=True, name="mock-ur-secondary"))
        self._threads.append(threading.Thread(
            target=self._tick_loop, daemon=True, name="mock-ur-tick"))
        for t in self._threads:
            t.start()
        log.info("MockURServer started on %s:%d / %d",
                 self.host, self.dashboard_port, self.secondary_port)

    def stop(self):
        """Signal all threads to stop and join."""
        self._stop_evt.set()
        for t in self._threads:
            t.join(timeout=2)
        self._threads.clear()
        log.info("MockURServer stopped")

    def move_joints(self, joints: list[float]):
        """Command the mock robot to move to joint angles (radians)."""
        if len(joints) != 6:
            raise ValueError(f"Expected 6 joint angles, got {len(joints)}")
        self._state.set_target(joints)

    @property
    def robot_mode(self) -> str:
        return self._state.get_mode()

    # -- context manager support --------------------------------------------

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    # -- internal listener loops --------------------------------------------

    def _serve_dashboard(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self.host, self.dashboard_port))
            srv.listen(5)
            srv.settimeout(1.0)
            while not self._stop_evt.is_set():
                try:
                    conn, addr = srv.accept()
                    log.debug("Dashboard client: %s", addr)
                    t = threading.Thread(
                        target=_handle_dashboard,
                        args=(conn, self._state),
                        daemon=True,
                    )
                    t.start()
                except socket.timeout:
                    continue

    def _serve_secondary(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self.host, self.secondary_port))
            srv.listen(5)
            srv.settimeout(1.0)
            while not self._stop_evt.is_set():
                try:
                    conn, addr = srv.accept()
                    log.debug("Secondary client: %s", addr)
                    t = threading.Thread(
                        target=_handle_secondary,
                        args=(conn, self._state, self._stop_evt),
                        daemon=True,
                    )
                    t.start()
                except socket.timeout:
                    continue

    def _tick_loop(self):
        while not self._stop_evt.is_set():
            self._state.tick()
            time.sleep(_TICK_INTERVAL)
