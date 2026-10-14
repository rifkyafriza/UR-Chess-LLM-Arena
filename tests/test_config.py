"""Unit tests for src/config.py."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from src.config import (
    ArenaConfig,
    BoardLayoutConfig,
    PortsConfig,
    RobotConfig,
    SafetyEnvelopeConfig,
    SimulationConfig,
    SpeedsConfig,
    StrategyConfig,
    VisionConfig,
    load_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "test_config.yaml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Default / missing file
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_arena_config_defaults(self):
        cfg = ArenaConfig()
        assert cfg.robot.mode == "sim"
        assert cfg.robot.host == "192.168.1.100"
        assert cfg.robot.ports.dashboard == 29999
        assert cfg.robot.ports.script == 30002
        assert cfg.robot.ports.realtime == 30003
        assert cfg.robot.use_ur_feature is False
        assert len(cfg.robot.feature_plane_pose) == 6

    def test_board_layout_defaults(self):
        bl = BoardLayoutConfig()
        assert bl.square_pitch_x == pytest.approx(0.055)
        assert bl.square_pitch_y == pytest.approx(0.055)
        assert bl.pick_height_z == pytest.approx(0.010)
        assert bl.transit_clearance_z == pytest.approx(0.120)

    def test_safety_envelope_defaults(self):
        se = SafetyEnvelopeConfig()
        assert se.x_bounds == (-0.80, 0.80)
        assert se.y_bounds == (-0.80, 0.80)
        assert se.z_bounds == (-0.20, 0.90)

    def test_speeds_defaults(self):
        s = SpeedsConfig()
        assert s.move_velocity == pytest.approx(0.25)
        assert s.move_acceleration == pytest.approx(0.50)
        assert s.joint_velocity == pytest.approx(1.05)
        assert s.joint_acceleration == pytest.approx(1.40)

    def test_vision_defaults(self):
        v = VisionConfig()
        assert v.camera_index == 0
        assert v.locator_mode == "aruco"
        assert v.aruco_dict == "DICT_4X4_50"
        assert v.resolution == (1280, 720)
        assert v.stream_port == 5050

    def test_strategy_defaults(self):
        s = StrategyConfig()
        assert "127.0.0.1:20128" in s.api_base
        assert s.timeout == 30
        assert s.fallback_engine == "stockfish"

    def test_simulation_defaults(self):
        s = SimulationConfig()
        assert s.host == "127.0.0.1"
        assert s.port == 5060


# ---------------------------------------------------------------------------
# load_config — missing file → safe defaults
# ---------------------------------------------------------------------------

class TestLoadConfigMissingFile:
    def test_nonexistent_path_returns_defaults(self, tmp_path):
        cfg = load_config(str(tmp_path / "does_not_exist.yaml"))
        assert isinstance(cfg, ArenaConfig)
        assert cfg.robot.mode == "sim"

    def test_none_path_falls_back(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ARENA_CONFIG", str(tmp_path / "missing.yaml"))
        cfg = load_config()
        assert isinstance(cfg, ArenaConfig)


# ---------------------------------------------------------------------------
# load_config — full YAML roundtrip
# ---------------------------------------------------------------------------

FULL_YAML = """\
robot:
  mode: real
  host: 10.0.0.5
  ports:
    dashboard: 29999
    script: 30002
    realtime: 30003
  use_ur_feature: true
  feature_plane_pose: [0.1, 0.2, 0.3, 3.14, 0.0, 0.0]
  board_layout:
    square_pitch_x: 0.060
    square_pitch_y: 0.060
    pick_height_z: 0.015
    transit_clearance_z: 0.130
  safety_envelope:
    x_bounds: [-0.70, 0.70]
    y_bounds: [-0.70, 0.70]
    z_bounds: [-0.10, 0.80]
  speeds:
    move_velocity: 0.30
    move_acceleration: 0.60
    joint_velocity: 1.20
    joint_acceleration: 1.50

vision:
  camera_index: 1
  locator_mode: contour
  aruco_dict: DICT_6X6_250
  resolution: [640, 480]
  stream_port: 5051

strategy:
  api_base: http://192.168.1.1:8080/v1
  model: gpt-4o
  timeout: 60
  fallback_engine: lc0

simulation:
  host: 0.0.0.0
  port: 5061
"""


class TestLoadConfigFullYaml:
    def test_robot_section(self, tmp_path):
        p = _write_yaml(tmp_path, FULL_YAML)
        cfg = load_config(str(p))
        assert cfg.robot.mode == "real"
        assert cfg.robot.host == "10.0.0.5"
        assert cfg.robot.use_ur_feature is True
        assert cfg.robot.feature_plane_pose == pytest.approx([0.1, 0.2, 0.3, 3.14, 0.0, 0.0])

    def test_ports(self, tmp_path):
        p = _write_yaml(tmp_path, FULL_YAML)
        cfg = load_config(str(p))
        assert cfg.robot.ports.dashboard == 29999
        assert cfg.robot.ports.script == 30002
        assert cfg.robot.ports.realtime == 30003

    def test_board_layout(self, tmp_path):
        p = _write_yaml(tmp_path, FULL_YAML)
        bl = load_config(str(p)).robot.board_layout
        assert bl.square_pitch_x == pytest.approx(0.060)
        assert bl.pick_height_z == pytest.approx(0.015)
        assert bl.transit_clearance_z == pytest.approx(0.130)

    def test_safety_envelope(self, tmp_path):
        p = _write_yaml(tmp_path, FULL_YAML)
        se = load_config(str(p)).robot.safety_envelope
        assert se.x_bounds == pytest.approx((-0.70, 0.70))
        assert se.z_bounds == pytest.approx((-0.10, 0.80))

    def test_speeds(self, tmp_path):
        p = _write_yaml(tmp_path, FULL_YAML)
        sp = load_config(str(p)).robot.speeds
        assert sp.move_velocity == pytest.approx(0.30)
        assert sp.joint_acceleration == pytest.approx(1.50)

    def test_vision(self, tmp_path):
        p = _write_yaml(tmp_path, FULL_YAML)
        v = load_config(str(p)).vision
        assert v.camera_index == 1
        assert v.locator_mode == "contour"
        assert v.aruco_dict == "DICT_6X6_250"
        assert v.resolution == (640, 480)
        assert v.stream_port == 5051

    def test_strategy(self, tmp_path):
        p = _write_yaml(tmp_path, FULL_YAML)
        s = load_config(str(p)).strategy
        assert s.api_base == "http://192.168.1.1:8080/v1"
        assert s.model == "gpt-4o"
        assert s.timeout == 60
        assert s.fallback_engine == "lc0"

    def test_simulation(self, tmp_path):
        p = _write_yaml(tmp_path, FULL_YAML)
        sim = load_config(str(p)).simulation
        assert sim.host == "0.0.0.0"
        assert sim.port == 5061


# ---------------------------------------------------------------------------
# ARENA_CONFIG env var override
# ---------------------------------------------------------------------------

class TestEnvVar:
    def test_env_var_picked_up(self, tmp_path, monkeypatch):
        p = _write_yaml(tmp_path, "robot:\n  mode: real\n")
        monkeypatch.setenv("ARENA_CONFIG", str(p))
        cfg = load_config()   # no explicit path
        assert cfg.robot.mode == "real"

    def test_explicit_path_overrides_env(self, tmp_path, monkeypatch):
        env_yaml = _write_yaml(tmp_path, "robot:\n  mode: real\n")
        explicit_yaml = tmp_path / "explicit.yaml"
        explicit_yaml.write_text("robot:\n  mode: sim\n", encoding="utf-8")
        monkeypatch.setenv("ARENA_CONFIG", str(env_yaml))
        cfg = load_config(str(explicit_yaml))
        assert cfg.robot.mode == "sim"


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

class TestExport:
    def test_to_dict_structure(self):
        cfg = ArenaConfig()
        d = cfg.to_dict()
        assert "robot" in d
        assert "vision" in d
        assert "strategy" in d
        assert "simulation" in d
        assert "mode" in d["robot"]
        assert "ports" in d["robot"]

    def test_to_dict_values_match(self):
        cfg = ArenaConfig()
        d = cfg.to_dict()
        assert d["robot"]["mode"] == "sim"
        assert d["vision"]["stream_port"] == 5050
        assert d["simulation"]["port"] == 5060

    def test_to_yaml_roundtrip(self, tmp_path):
        cfg = ArenaConfig()
        yaml_str = cfg.to_yaml()
        assert "robot" in yaml_str
        assert "vision" in yaml_str

    def test_to_yaml_custom_values(self, tmp_path):
        cfg = load_config(str(_write_yaml(tmp_path, "robot:\n  mode: real\n")))
        assert "real" in cfg.to_yaml()


# ---------------------------------------------------------------------------
# Partial YAML — missing sections fall back to defaults
# ---------------------------------------------------------------------------

class TestPartialYaml:
    def test_only_robot_section(self, tmp_path):
        p = _write_yaml(tmp_path, "robot:\n  mode: real\n")
        cfg = load_config(str(p))
        assert cfg.robot.mode == "real"
        # Other sections get defaults
        assert cfg.vision.stream_port == 5050
        assert cfg.simulation.port == 5060

    def test_only_vision_section(self, tmp_path):
        p = _write_yaml(tmp_path, "vision:\n  camera_index: 2\n")
        cfg = load_config(str(p))
        assert cfg.vision.camera_index == 2
        assert cfg.robot.mode == "sim"   # default

    def test_corrupt_yaml_returns_defaults(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("{{{{invalid yaml!!!", encoding="utf-8")
        cfg = load_config(str(p))
        assert isinstance(cfg, ArenaConfig)
        assert cfg.robot.mode == "sim"


# ---------------------------------------------------------------------------
# Ship-on-disk config/arena_config.yaml is loadable
# ---------------------------------------------------------------------------

class TestShippedConfig:
    def test_shipped_yaml_loads(self):
        """The committed config/arena_config.yaml must parse without error."""
        cfg = load_config()
        assert isinstance(cfg, ArenaConfig)
        assert cfg.robot.mode in ("sim", "real")
        assert cfg.vision.stream_port == 5050
        assert cfg.simulation.port == 5060


def test_unsafe_values_rejected(tmp_path):
    import pytest
    from src.config import load_config
    for body in ("robot: {mode: banana}", "robot: {board_layout: {transit_clearance_z: 0.01}}",
                 "robot: {board_layout: {square_pitch_x: -1}}", "robot: {ports: {script: 99999}}",
                 "robot: {safety_envelope: {x_bounds: [5, -5]}}",
                 "robot: {board_layout: {pick_height_z: .nan}}", "robot: {feature_plane_pose: [1, 2]}"):
        f = tmp_path / "c.yaml"
        f.write_text(body)
        with pytest.raises(ValueError):
            load_config(str(f))
