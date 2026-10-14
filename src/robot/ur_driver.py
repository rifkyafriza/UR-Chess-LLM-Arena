"""
UR Robot driver: DashboardClient, RealURDriver, MockURDriver, and factory.

Ports:
    29999  Dashboard (ASCII text)
    30002  Secondary interface (URScript send)
    30003  Real-time interface (binary telemetry, 125 Hz)

Units:
    Positions : meters (m)
    Angles    : radians (rad)
    Speed     : m/s or rad/s
    Accel     : m/s^2 or rad/s^2
"""

from __future__ import annotations

import socket
import struct
import threading
import time
import logging
from typing import List

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safety bounding box (UR base frame, meters)
# ---------------------------------------------------------------------------
# Reject any pose outside these limits before sending to hardware.
_SAFE_X = (-0.80, 0.80)   # meters
_SAFE_Y = (-0.80, 0.80)   # meters
_SAFE_Z = (-0.20, 0.90)   # meters
_TRANSIT_Z = 0.080         # minimum Z clearance during transit (meters)


def _check_pose_limits(pose: List[float]) -> None:
    """Raise ValueError if pose violates hard workspace safety limits."""
    if len(pose) != 6:
        raise ValueError(f"Pose must have 6 elements, got {len(pose)}")
    x, y, z = pose[0], pose[1], pose[2]
    if not (_SAFE_X[0] <= x <= _SAFE_X[1]):
        raise ValueError(f"Pose X={x:.4f} m outside safe range {_SAFE_X}")
    if not (_SAFE_Y[0] <= y <= _SAFE_Y[1]):
        raise ValueError(f"Pose Y={y:.4f} m outside safe range {_SAFE_Y}")
    if not (_SAFE_Z[0] <= z <= _SAFE_Z[1]):
        raise ValueError(f"Pose Z={z:.4f} m outside safe range {_SAFE_Z}")


# ---------------------------------------------------------------------------
# Dashboard Client (port 29999)
# ---------------------------------------------------------------------------

class DashboardClient:
    """
    ASCII text command channel to UR Dashboard Server on port 29999.

    The server sends a banner on connect, then responds line-by-line to
    each command terminated with '\\n'.
    """

    _TIMEOUT = 5.0  # seconds

    def __init__(self, host: str = "127.0.0.1", port: int = 29999):
        self.host = host
        self.port = port
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()

    # -- connection ----------------------------------------------------------

    def connect(self) -> None:
        """Open TCP connection and consume the welcome banner."""
        with self._lock:
            self._sock = socket.create_connection((self.host, self.port),
                                                  timeout=self._TIMEOUT)
            self._sock.settimeout(self._TIMEOUT)
            # Consume banner line
            self._sock.recv(256)

    def close(self) -> None:
        with self._lock:
            if self._sock:
                try:
                    self._sock.sendall(b"quit\n")
                except OSError:
                    pass
                self._sock.close()
                self._sock = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.close()

    # -- internal send/recv --------------------------------------------------

    def _cmd(self, command: str) -> str:
        """Send command, return response line (strips trailing whitespace)."""
        with self._lock:
            if not self._sock:
                raise RuntimeError("DashboardClient not connected")
            self._sock.sendall((command + "\n").encode())
            return self._sock.recv(1024).decode("utf-8", errors="replace").strip()

    # -- public API ----------------------------------------------------------

    def power_on(self) -> str:
        return self._cmd("power on")

    def power_off(self) -> str:
        return self._cmd("power off")

    def brake_release(self) -> str:
        return self._cmd("brake release")

    def play(self) -> str:
        return self._cmd("play")

    def stop(self) -> str:
        return self._cmd("stop")

    def close_popup(self) -> str:
        return self._cmd("close popup")

    def get_robot_mode(self) -> str:
        """Return robot mode string, e.g. 'Robotmode: RUNNING'."""
        return self._cmd("robotmode")

    def get_safety_mode(self) -> str:
        """Return safety mode string, e.g. 'Safetymode: NORMAL'."""
        return self._cmd("safetymode")


# ---------------------------------------------------------------------------
# Real-time telemetry reader (port 30003, 125 Hz binary stream)
# ---------------------------------------------------------------------------

# UR Real-time Interface packet layout (1116 bytes total):
#   Offset 0:   int32 BE  – message length
#   Offset 4:   double BE – time (s)
#   ... many fields ...
#   Offset 588: 6x double BE – actual Cartesian TCP pose [X Y Z Rx Ry Rz] (m/rad)
_RT_PACKET_LEN = 1116
_RT_TCP_POSE_OFFSET = 588  # bytes from start of packet
_RT_POSE_COUNT = 6


class _RTReader(threading.Thread):
    """
    Background thread: continuously reads 1116-byte packets from port 30003
    and exposes the latest TCP Cartesian pose [X, Y, Z, Rx, Ry, Rz] (m, rad).
    """

    def __init__(self, host: str, port: int):
        super().__init__(daemon=True, name="ur-rt-reader")
        self.host = host
        self.port = port
        self._lock = threading.Lock()
        self._pose: List[float] = [0.0] * 6
        self._stop_evt = threading.Event()
        self._connected = threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            try:
                with socket.create_connection((self.host, self.port),
                                              timeout=5.0) as sock:
                    sock.settimeout(1.0)
                    self._connected.set()
                    buf = b""
                    while not self._stop_evt.is_set():
                        try:
                            chunk = sock.recv(4096)
                        except socket.timeout:
                            continue
                        if not chunk:
                            break
                        buf += chunk
                        while len(buf) >= _RT_PACKET_LEN:
                            pkt = buf[:_RT_PACKET_LEN]
                            buf = buf[_RT_PACKET_LEN:]
                            self._parse(pkt)
            except (OSError, ConnectionRefusedError) as exc:
                log.warning("RTReader connection lost: %s – retry in 1s", exc)
                self._connected.clear()
                time.sleep(1.0)

    def _parse(self, pkt: bytes) -> None:
        """Unpack TCP pose from packet at fixed offset."""
        pose = list(struct.unpack_from(">6d", pkt, _RT_TCP_POSE_OFFSET))
        with self._lock:
            self._pose = pose

    def get_pose(self) -> List[float]:
        """Return latest [X, Y, Z, Rx, Ry, Rz] (meters, radians)."""
        with self._lock:
            return list(self._pose)

    def stop(self):
        self._stop_evt.set()


# ---------------------------------------------------------------------------
# RealURDriver – live hardware backend
# ---------------------------------------------------------------------------

class RealURDriver:
    """
    Live UR robot driver.

    Sends URScript over the secondary interface (port 30002) and reads
    Cartesian TCP pose from the real-time interface (port 30003).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port_script: int = 30002,
        port_rt: int = 30003,
    ):
        self.host = host
        self.port_script = port_script
        self.port_rt = port_rt
        self._rt_reader: _RTReader | None = None
        self._script_lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------------

    def connect(self) -> None:
        """Start RT telemetry reader thread."""
        self._rt_reader = _RTReader(self.host, self.port_rt)
        self._rt_reader.start()

    def disconnect(self) -> None:
        if self._rt_reader:
            self._rt_reader.stop()
            self._rt_reader.join(timeout=3)
            self._rt_reader = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()

    # -- internal script sender ---------------------------------------------

    def _send_script(self, script: str) -> None:
        """Open a short-lived TCP connection and send a URScript program."""
        with self._script_lock:
            payload = script.strip() + "\n"
            with socket.create_connection(
                (self.host, self.port_script), timeout=5.0
            ) as sock:
                sock.sendall(payload.encode())

    # -- motion commands ----------------------------------------------------

    def movel(
        self,
        pose: List[float],
        a: float = 1.2,   # acceleration m/s^2
        v: float = 0.25,  # velocity m/s
    ) -> None:
        """
        Linear Cartesian move (movel).

        Parameters
        ----------
        pose : [X, Y, Z, Rx, Ry, Rz]  meters / radians
        a    : acceleration  m/s^2
        v    : velocity      m/s
        """
        _check_pose_limits(pose)
        p = ", ".join(f"{x:.6f}" for x in pose)
        self._send_script(f"movel(p[{p}], a={a}, v={v})")
        self._wait_arrival(pose)

    def _wait_arrival(self, pose: List[float], tol: float = 0.002, timeout: float = 30.0) -> None:
        """
        Block until TCP is within tol (m) of pose. A new script on port 30002 preempts
        the running one, so back-to-back movel calls would skip waypoints without this.
        No-op when RT telemetry is not connected (e.g. unit tests against a script-only server).
        """
        rd = self._rt_reader
        if rd is None or not rd._connected.is_set():
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            cur = rd.get_pose()
            if sum((c - t) ** 2 for c, t in zip(cur[:3], pose[:3])) ** 0.5 <= tol:
                return
            time.sleep(0.02)
        self.stopl()
        raise TimeoutError(f"TCP did not reach {pose[:3]} within {timeout}s")

    def movej(
        self,
        joints: List[float],
        a: float = 1.4,   # acceleration rad/s^2
        v: float = 1.05,  # velocity rad/s
    ) -> None:
        """
        Joint-space move (movej).

        Parameters
        ----------
        joints : [q1..q6]  radians
        a      : acceleration  rad/s^2
        v      : velocity      rad/s
        """
        if len(joints) != 6:
            raise ValueError(f"Expected 6 joints, got {len(joints)}")
        j = ", ".join(f"{x:.6f}" for x in joints)
        self._send_script(f"movej([{j}], a={a}, v={v})")

    def stopl(self, a: float = 10.0) -> None:
        """Emergency linear deceleration stop. a: decel m/s^2."""
        self._send_script(f"stopl({a})")

    def set_digital_out(self, port: int, value: bool) -> None:
        """Set controller digital output. port 0-7, value True/False."""
        self._send_script(f"set_digital_out({port}, {'True' if value else 'False'})")

    def set_tool_digital_out(self, port: int, value: bool) -> None:
        """Set tool digital output (end-effector). port 0-1."""
        self._send_script(
            f"set_tool_digital_out({port}, {'True' if value else 'False'})"
        )

    # -- telemetry -----------------------------------------------------------

    def get_tcp_pose(self) -> List[float]:
        """Return latest actual TCP pose [X, Y, Z, Rx, Ry, Rz] (m, rad)."""
        if not self._rt_reader:
            raise RuntimeError("Not connected – call connect() first")
        return self._rt_reader.get_pose()


# ---------------------------------------------------------------------------
# MockURDriver – zero-install in-memory simulator
# ---------------------------------------------------------------------------

class MockURDriver:
    """
    In-memory mock UR driver for offline testing.

    - Thread-safe state tracking.
    - Simulates motion delay (proportional to Euclidean distance).
    - Digital outputs stored in a dict.
    """

    # Simulation parameters
    _SIM_SPEED = 0.10     # m/s Cartesian travel speed
    _SIM_JOINT_SPEED = 1.0  # rad/s joint travel speed

    def __init__(self):
        self._lock = threading.Lock()
        # Initial TCP pose [X, Y, Z, Rx, Ry, Rz] (meters, radians)
        self._pose: List[float] = [0.0, -0.5, 0.5, 0.0, 3.14159, 0.0]
        self._joints: List[float] = [0.0] * 6
        self._digital_out: dict[int, bool] = {}
        self._tool_digital_out: dict[int, bool] = {}
        self._moving = threading.Event()

    # -- connection stubs (no-ops for mock) ----------------------------------

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    # -- motion commands ----------------------------------------------------

    def movel(
        self,
        pose: List[float],
        a: float = 1.2,
        v: float = 0.25,
    ) -> None:
        """Simulate linear move; blocks until simulated travel complete."""
        _check_pose_limits(pose)
        with self._lock:
            current = list(self._pose[:3])
            target = list(pose[:3])
        dist = sum((t - c) ** 2 for t, c in zip(target, current)) ** 0.5  # meters
        travel_time = dist / self._SIM_SPEED  # seconds (ignore accel ramp)
        self._moving.set()
        time.sleep(min(travel_time, 2.0))  # cap at 2 s for test ergonomics
        with self._lock:
            self._pose = list(pose)
        self._moving.clear()
        log.debug("MockURDriver movel -> %s", pose)

    def movej(
        self,
        joints: List[float],
        a: float = 1.4,
        v: float = 1.05,
    ) -> None:
        """Simulate joint move; blocks until simulated travel complete."""
        if len(joints) != 6:
            raise ValueError(f"Expected 6 joints, got {len(joints)}")
        with self._lock:
            current = list(self._joints)
        dist = sum((t - c) ** 2 for t, c in zip(joints, current)) ** 0.5  # rad
        travel_time = dist / self._SIM_JOINT_SPEED
        self._moving.set()
        time.sleep(min(travel_time, 2.0))
        with self._lock:
            self._joints = list(joints)
        self._moving.clear()
        log.debug("MockURDriver movej -> %s", joints)

    def stopl(self, a: float = 10.0) -> None:
        """Simulate emergency stop (clears moving flag)."""
        self._moving.clear()
        log.debug("MockURDriver stopl")

    def set_digital_out(self, port: int, value: bool) -> None:
        with self._lock:
            self._digital_out[port] = value
        log.debug("MockURDriver digital_out[%d] = %s", port, value)

    def set_tool_digital_out(self, port: int, value: bool) -> None:
        with self._lock:
            self._tool_digital_out[port] = value
        log.debug("MockURDriver tool_digital_out[%d] = %s", port, value)

    # -- telemetry -----------------------------------------------------------

    def get_tcp_pose(self) -> List[float]:
        """Return current simulated TCP pose [X, Y, Z, Rx, Ry, Rz]."""
        with self._lock:
            return list(self._pose)

    def get_joints(self) -> List[float]:
        """Return current simulated joint angles [q1..q6] radians."""
        with self._lock:
            return list(self._joints)

    def is_moving(self) -> bool:
        return self._moving.is_set()

    def get_digital_out(self, port: int) -> bool:
        with self._lock:
            return self._digital_out.get(port, False)

    def get_tool_digital_out(self, port: int) -> bool:
        with self._lock:
            return self._tool_digital_out.get(port, False)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_robot_driver(
    mode: str = "sim",
    host: str = "127.0.0.1",
    port_dash: int = 29999,
    port_script: int = 30002,
    port_rt: int = 30003,
) -> MockURDriver | RealURDriver:
    """
    Factory: return a robot driver instance.

    Parameters
    ----------
    mode        : 'sim' for MockURDriver, 'real' for RealURDriver
    host        : robot IP (ignored for 'sim')
    port_dash   : Dashboard port (used to build DashboardClient separately)
    port_script : URScript send port
    port_rt     : Real-time telemetry port

    Returns
    -------
    MockURDriver or RealURDriver (both expose the same motion/telemetry API)
    """
    if mode == "sim":
        return MockURDriver()
    if mode == "real":
        return RealURDriver(host=host, port_script=port_script, port_rt=port_rt)
    raise ValueError(f"Unknown mode '{mode}'. Use 'sim' or 'real'.")
