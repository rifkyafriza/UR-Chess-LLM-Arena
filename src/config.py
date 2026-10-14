"""
Centralized configuration loader for UR Chess LLM Arena.

Usage::
    from src.config import load_config
    cfg = load_config()           # uses config/arena_config.yaml
    cfg = load_config("my.yaml")  # explicit path
    cfg = load_config(None)       # safe defaults when file absent
"""

from __future__ import annotations

import dataclasses
import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import yaml as _yaml  # PyYAML
    _YAML_AVAILABLE = True
except ImportError:  # pragma: no cover
    _yaml = None  # type: ignore[assignment]
    _YAML_AVAILABLE = False

# Default YAML path relative to repo root (two levels up from src/)
_DEFAULT_CONFIG = Path(__file__).parent.parent / "config" / "arena_config.yaml"


# ---------------------------------------------------------------------------
# Sub-section dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PortsConfig:
    dashboard: int = 29999
    script: int = 30002
    realtime: int = 30003


@dataclass
class BoardLayoutConfig:
    square_pitch_x: float = 0.055        # m — file spacing
    square_pitch_y: float = 0.055        # m — rank spacing
    pick_height_z: float = 0.010         # m — grip height
    transit_clearance_z: float = 0.120   # m — transit floor


@dataclass
class SafetyEnvelopeConfig:
    x_bounds: Tuple[float, float] = (-0.80, 0.80)
    y_bounds: Tuple[float, float] = (-0.80, 0.80)
    z_bounds: Tuple[float, float] = (-0.20, 0.90)


@dataclass
class SpeedsConfig:
    move_velocity: float = 0.25      # m/s
    move_acceleration: float = 0.50  # m/s²
    joint_velocity: float = 1.05     # rad/s
    joint_acceleration: float = 1.40 # rad/s²


@dataclass
class RobotConfig:
    mode: str = "sim"                                          # "sim" | "real"
    host: str = "192.168.1.100"
    ports: PortsConfig = field(default_factory=PortsConfig)
    use_ur_feature: bool = False
    feature_plane_pose: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 3.14159, 0.0, 0.0]
    )
    board_layout: BoardLayoutConfig = field(default_factory=BoardLayoutConfig)
    safety_envelope: SafetyEnvelopeConfig = field(default_factory=SafetyEnvelopeConfig)
    speeds: SpeedsConfig = field(default_factory=SpeedsConfig)


@dataclass
class VisionConfig:
    camera_index: int = 0
    locator_mode: str = "aruco"          # "contour" | "aruco"
    aruco_dict: str = "DICT_4X4_50"
    resolution: Tuple[int, int] = (1280, 720)
    stream_port: int = 5050


@dataclass
class StrategyConfig:
    api_base: str = "http://127.0.0.1:20128/v1"
    model: str = "ag/claude-sonnet-4-6"
    timeout: int = 30                    # seconds
    fallback_engine: str = "stockfish"


@dataclass
class SimulationConfig:
    host: str = "127.0.0.1"
    port: int = 5060


@dataclass
class ArenaConfig:
    robot: RobotConfig = field(default_factory=RobotConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Recursive dataclass → plain dict (no YAML dep)."""
        return dataclasses.asdict(self)

    def to_yaml(self) -> str:
        """Serialize to YAML string. Requires PyYAML."""
        if not _YAML_AVAILABLE:
            raise ImportError("PyYAML not installed; run: pip install pyyaml")
        assert _yaml is not None
        return _yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tuple_pair(raw, dtype=float) -> tuple:
    """Convert a 2-element list/tuple from YAML into a typed tuple."""
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return (dtype(raw[0]), dtype(raw[1]))
    raise ValueError(f"Expected 2-element sequence, got {raw!r}")


def _build_robot(d: dict) -> RobotConfig:
    ports_d = d.get("ports", {})
    bl_d = d.get("board_layout", {})
    se_d = d.get("safety_envelope", {})
    sp_d = d.get("speeds", {})

    return RobotConfig(
        mode=str(d.get("mode", "sim")),
        host=str(d.get("host", "192.168.1.100")),
        ports=PortsConfig(
            dashboard=int(ports_d.get("dashboard", 29999)),
            script=int(ports_d.get("script", 30002)),
            realtime=int(ports_d.get("realtime", 30003)),
        ),
        use_ur_feature=bool(d.get("use_ur_feature", False)),
        feature_plane_pose=list(d.get("feature_plane_pose",
                                      [0.0, 0.0, 0.0, 3.14159, 0.0, 0.0])),
        board_layout=BoardLayoutConfig(
            square_pitch_x=float(bl_d.get("square_pitch_x", 0.055)),
            square_pitch_y=float(bl_d.get("square_pitch_y", 0.055)),
            pick_height_z=float(bl_d.get("pick_height_z", 0.010)),
            transit_clearance_z=float(bl_d.get("transit_clearance_z", 0.120)),
        ),
        safety_envelope=SafetyEnvelopeConfig(
            x_bounds=_tuple_pair(se_d.get("x_bounds", [-0.80, 0.80])),
            y_bounds=_tuple_pair(se_d.get("y_bounds", [-0.80, 0.80])),
            z_bounds=_tuple_pair(se_d.get("z_bounds", [-0.20, 0.90])),
        ),
        speeds=SpeedsConfig(
            move_velocity=float(sp_d.get("move_velocity", 0.25)),
            move_acceleration=float(sp_d.get("move_acceleration", 0.50)),
            joint_velocity=float(sp_d.get("joint_velocity", 1.05)),
            joint_acceleration=float(sp_d.get("joint_acceleration", 1.40)),
        ),
    )


def _build_vision(d: dict) -> VisionConfig:
    res = d.get("resolution", [1280, 720])
    return VisionConfig(
        camera_index=int(d.get("camera_index", 0)),
        locator_mode=str(d.get("locator_mode", "aruco")),
        aruco_dict=str(d.get("aruco_dict", "DICT_4X4_50")),
        resolution=_tuple_pair(res, int),
        stream_port=int(d.get("stream_port", 5050)),
    )


def _build_strategy(d: dict) -> StrategyConfig:
    return StrategyConfig(
        api_base=str(d.get("api_base", "http://127.0.0.1:20128/v1")),
        model=str(d.get("model", "ag/claude-sonnet-4-6")),
        timeout=int(d.get("timeout", 30)),
        fallback_engine=str(d.get("fallback_engine", "stockfish")),
    )


def _build_simulation(d: dict) -> SimulationConfig:
    return SimulationConfig(
        host=str(d.get("host", "127.0.0.1")),
        port=int(d.get("port", 5060)),
    )


def _parse(raw: dict) -> ArenaConfig:
    return ArenaConfig(
        robot=_build_robot(raw.get("robot", {})),
        vision=_build_vision(raw.get("vision", {})),
        strategy=_build_strategy(raw.get("strategy", {})),
        simulation=_build_simulation(raw.get("simulation", {})),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(path: Optional[str] = None) -> ArenaConfig:
    """
    Load ArenaConfig from a YAML file.

    Resolution order:
    1. ``path`` argument (if given).
    2. ``ARENA_CONFIG`` environment variable.
    3. ``config/arena_config.yaml`` relative to the repo root.

    If the resolved file does not exist (or PyYAML is absent), returns a
    safe-default ArenaConfig with no exception raised.
    """
    resolved: Optional[Path] = None

    if path is not None:
        resolved = Path(path)
    elif "ARENA_CONFIG" in os.environ:
        resolved = Path(os.environ["ARENA_CONFIG"])
    else:
        resolved = _DEFAULT_CONFIG

    if resolved is None or not resolved.exists():
        return ArenaConfig()

    if not _YAML_AVAILABLE:
        return ArenaConfig()

    try:
        assert _yaml is not None
        raw = _yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
        return _parse(raw)
    except Exception:  # noqa: BLE001 — corrupt YAML → safe defaults
        return ArenaConfig()
