"""
calibrate_feature.py — Interactive UR Feature Plane calibration helper.

Usage:
    python scripts/calibrate_feature.py [--sim] [--config PATH]

Workflow:
    1. Guide the operator to touch a1, h1, a8 in Freedrive mode.
    2. Compute board local frame: origin (a1), X-axis (a→h), Y-axis (1→8).
    3. Derive UR axis-angle rotation vector [Rx, Ry, Rz] from the frame.
    4. Write feature_plane_pose and use_ur_feature into arena_config.yaml.

In --sim mode the operator enters XYZ coordinates manually (no robot
connection required).

Units throughout: meters (position), radians (orientation).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Allow running from repo root without package install
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    import yaml as _yaml  # PyYAML
    _YAML_AVAILABLE = True
except ImportError:
    _yaml = None  # type: ignore[assignment]
    _YAML_AVAILABLE = False

from src.robot.coordinates import _rotvec_to_matrix  # reuse validated helper


# ---------------------------------------------------------------------------
# Linear algebra helpers (stdlib only)
# ---------------------------------------------------------------------------

def _vec_sub(a: List[float], b: List[float]) -> List[float]:
    """a - b, element-wise."""
    return [a[i] - b[i] for i in range(3)]


def _vec_norm(v: List[float]) -> float:
    """Euclidean norm of 3-vector."""
    return math.sqrt(sum(x * x for x in v))


def _vec_normalize(v: List[float]) -> List[float]:
    """Return unit vector; raises ValueError if near-zero."""
    n = _vec_norm(v)
    if n < 1e-9:
        raise ValueError(f"Cannot normalize near-zero vector {v}")
    return [x / n for x in v]


def _vec_cross(a: List[float], b: List[float]) -> List[float]:
    """Cross product a × b."""
    return [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ]


def _vec_dot(a: List[float], b: List[float]) -> float:
    """Dot product."""
    return sum(a[i]*b[i] for i in range(3))


def _rotation_matrix_to_rotvec(R: List[List[float]]) -> List[float]:
    """
    Convert 3x3 rotation matrix to UR rotation vector (axis * angle), radians.

    Inverse of the Rodrigues formula used in _rotvec_to_matrix().
    Returns [Rx, Ry, Rz] where the magnitude encodes the rotation angle.
    """
    # Angle from trace: cos(angle) = (tr(R) - 1) / 2
    trace = R[0][0] + R[1][1] + R[2][2]
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = max(-1.0, min(1.0, cos_angle))  # clamp for numerical safety
    angle = math.acos(cos_angle)  # rad in [0, pi]

    if angle < 1e-10:
        # Identity rotation — axis undefined, return zero vector
        return [0.0, 0.0, 0.0]

    if math.pi - angle < 1e-6:
        # Angle near pi: use diagonal of (R + I)/2 to recover axis
        diag = [(R[i][i] + 1.0) / 2.0 for i in range(3)]
        diag = [max(0.0, d) for d in diag]  # clamp numerical noise
        mx = max(diag)
        idx = diag.index(mx)
        axis = [0.0, 0.0, 0.0]
        axis[idx] = math.sqrt(diag[idx])
        for j in range(3):
            if j != idx:
                axis[j] = R[idx][j] / (2.0 * axis[idx])
        axis = _vec_normalize(axis)
        return [axis[i] * angle for i in range(3)]

    # General case: axis from skew-symmetric part
    s = 1.0 / (2.0 * math.sin(angle))
    axis = [
        (R[2][1] - R[1][2]) * s,
        (R[0][2] - R[2][0]) * s,
        (R[1][0] - R[0][1]) * s,
    ]
    return [axis[i] * angle for i in range(3)]


# ---------------------------------------------------------------------------
# Frame computation
# ---------------------------------------------------------------------------

def compute_feature_plane_pose(
    p_a1: List[float],
    p_h1: List[float],
    p_a8: List[float],
) -> List[float]:
    """
    Compute the UR Feature Plane pose [X, Y, Z, Rx, Ry, Rz] from three
    calibration points.

    Geometry:
        p_a1 → origin of the board local frame (a1 square centre).
        p_h1 → defines the file (X) axis direction: a→h.
        p_a8 → defines the rank (Y) axis direction: 1→8.

    The board Z (normal) is computed as X × Y_approx, then Y is re-orthogonalised
    to ensure a proper right-handed frame.

    Parameters
    ----------
    p_a1, p_h1, p_a8 : [X, Y, Z] in UR base frame, meters.

    Returns
    -------
    [X, Y, Z, Rx, Ry, Rz] — Feature Plane pose, meters and radians.

    Raises
    ------
    ValueError if points are collinear (degenerate board geometry).
    """
    # Local X axis: a1 → h1, normalized
    x_axis = _vec_normalize(_vec_sub(p_h1, p_a1))

    # Approximate Y axis: a1 → a8
    y_approx = _vec_sub(p_a8, p_a1)

    # Board normal Z: X × Y_approx (right-hand rule)
    z_axis = _vec_normalize(_vec_cross(x_axis, y_approx))

    # Re-orthogonalised Y: Z × X
    y_axis = _vec_normalize(_vec_cross(z_axis, x_axis))

    # Build rotation matrix R (columns are local X, Y, Z expressed in base frame)
    # Column-major: R[:,0] = x_axis, R[:,1] = y_axis, R[:,2] = z_axis
    R = [
        [x_axis[0], y_axis[0], z_axis[0]],
        [x_axis[1], y_axis[1], z_axis[1]],
        [x_axis[2], y_axis[2], z_axis[2]],
    ]

    rotvec = _rotation_matrix_to_rotvec(R)

    return [p_a1[0], p_a1[1], p_a1[2], rotvec[0], rotvec[1], rotvec[2]]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _prompt_xyz(label: str) -> List[float]:
    """
    Ask the operator to enter X Y Z for a calibration point.
    Returns [X, Y, Z] in meters.
    """
    print(f"\n  Touch the TCP to {label} then enter the TCP pose reported by Polyscope.")
    while True:
        raw = input(f"  {label} [X Y Z in meters, space-separated]: ").strip()
        parts = raw.split()
        if len(parts) < 3:
            print("  ERROR: need exactly 3 values (X Y Z). Try again.")
            continue
        try:
            coords = [float(p) for p in parts[:3]]
            return coords
        except ValueError:
            print("  ERROR: non-numeric value. Try again.")


def _read_tcp_from_robot(host: str, port: int = 29999) -> Optional[List[float]]:
    """
    Read current TCP pose from the UR Dashboard Server (port 29999).
    Returns [X, Y, Z, Rx, Ry, Rz] or None on failure.
    """
    import socket
    try:
        with socket.create_connection((host, port), timeout=5.0) as s:
            # Dashboard handshake
            _ = s.recv(256)
            s.sendall(b"get tcp\n")
            resp = s.recv(256).decode("ascii", errors="replace").strip()
            # Response: "TCP pose: p[X,Y,Z,Rx,Ry,Rz]"
            start = resp.find("[")
            end = resp.find("]")
            if start < 0 or end < 0:
                return None
            values = [float(x) for x in resp[start+1:end].split(",")]
            return values if len(values) >= 3 else None
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: could not read TCP from robot ({exc})")
        return None


def _update_yaml(config_path: Path, pose: List[float]) -> None:
    """
    Write feature_plane_pose and use_ur_feature: true into the YAML config.
    Preserves all other keys.
    """
    if not _YAML_AVAILABLE:
        print("WARNING: PyYAML not installed. Cannot save to YAML.")
        print(f"  Manual entry: feature_plane_pose: {pose}")
        return

    assert _yaml is not None
    if config_path.exists():
        raw = _yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    else:
        raw = {}

    robot_section = raw.setdefault("robot", {})
    robot_section["use_ur_feature"] = True
    # Round to 6 decimal places to keep YAML readable
    robot_section["feature_plane_pose"] = [round(v, 6) for v in pose]

    config_path.write_text(
        _yaml.dump(raw, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    print(f"\n  Saved to {config_path}")


def _print_pose_summary(pose: List[float]) -> None:
    """Print computed pose in a human-readable format."""
    labels = ["X", "Y", "Z", "Rx", "Ry", "Rz"]
    units  = ["m", "m", "m", "rad", "rad", "rad"]
    print("\n  --- Computed Feature Plane Pose ---")
    for lbl, unit, val in zip(labels, units, pose):
        print(f"    {lbl:2s} = {val:+.6f} {unit}")
    angle_deg = math.degrees(math.sqrt(sum(v*v for v in pose[3:])))
    print(f"    rotation magnitude = {angle_deg:.3f} deg")
    print("  -----------------------------------")


# ---------------------------------------------------------------------------
# Main calibration routine
# ---------------------------------------------------------------------------

def run_calibration(
    sim: bool,
    config_path: Path,
    robot_host: Optional[str] = None,
) -> List[float]:
    """
    Run the interactive calibration workflow.

    Parameters
    ----------
    sim         : True → manual entry mode (no robot connection).
    config_path : YAML config file to update.
    robot_host  : UR robot IP (used in real mode for TCP readback).

    Returns
    -------
    Computed feature plane pose [X, Y, Z, Rx, Ry, Rz].
    """
    print("=" * 60)
    print("  UR Feature Plane Calibration")
    print("=" * 60)
    print()
    print("  You will be asked to position the TCP on three squares:")
    print("    a1 — board origin (White's queen-rook corner)")
    print("    h1 — defines the file (X) axis direction")
    print("    a8 — defines the rank (Y) axis direction")
    print()
    if sim:
        print("  MODE: Simulation — enter XYZ coordinates manually.")
    else:
        print("  MODE: Real robot — enable Freedrive, touch each square,")
        print("        then enter the TCP pose shown in Polyscope.")
    print()
    input("  Press ENTER to begin...")

    calibration_points = []
    labels = ["a1 (origin)", "h1 (file axis end)", "a8 (rank axis end)"]

    for label in labels:
        if not sim and robot_host:
            print(f"\n  Move TCP to {label} in Freedrive.")
            input("  Press ENTER when TCP is in position...")
            pose = _read_tcp_from_robot(robot_host)
            if pose is not None:
                print(f"  Auto-read TCP: X={pose[0]:.4f} Y={pose[1]:.4f} Z={pose[2]:.4f} m")
                confirm = input("  Use this position? [Y/n]: ").strip().lower()
                if confirm not in ("n", "no"):
                    calibration_points.append(pose[:3])
                    continue
        # Fall back to manual entry
        calibration_points.append(_prompt_xyz(label))

    p_a1, p_h1, p_a8 = calibration_points

    # Sanity check: points must not be collinear
    v1 = _vec_sub(p_h1, p_a1)
    v2 = _vec_sub(p_a8, p_a1)
    cross = _vec_cross(v1, v2)
    if _vec_norm(cross) < 1e-4:
        raise ValueError(
            "Calibration points are nearly collinear — the board plane cannot be "
            "determined. Check that a1, h1, a8 are not along the same line."
        )

    pose = compute_feature_plane_pose(p_a1, p_h1, p_a8)

    _print_pose_summary(pose)

    print()
    save = input("  Save to config? [Y/n]: ").strip().lower()
    if save not in ("n", "no"):
        _update_yaml(config_path, pose)
    else:
        print("  Not saved. Copy the pose above into arena_config.yaml manually.")

    return pose


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="UR Feature Plane calibration helper (3-point method).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Simulation mode: enter XYZ manually, no robot connection.",
    )
    parser.add_argument(
        "--config",
        default=str(_REPO_ROOT / "config" / "arena_config.yaml"),
        help="Path to arena_config.yaml (default: config/arena_config.yaml).",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="UR robot IP address for automatic TCP readback (real mode only).",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    try:
        run_calibration(
            sim=args.sim,
            config_path=config_path,
            robot_host=args.host,
        )
    except KeyboardInterrupt:
        print("\n\n  Calibration aborted by user.")
        sys.exit(1)
    except ValueError as exc:
        print(f"\n  ERROR: {exc}")
        sys.exit(2)


if __name__ == "__main__":
    main()
