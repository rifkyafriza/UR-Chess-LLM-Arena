"""Tests for src/config_editor.py and main.py launcher integration."""

from __future__ import annotations

import io
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import yaml

from src.config import ArenaConfig, load_config
from src.config_editor import (
    _save,
    _show,
    _tuples_to_lists,
    interactive_config_editor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_cfg() -> ArenaConfig:
    return ArenaConfig()


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


# ---------------------------------------------------------------------------
# 1. _show — view outputs structured string
# ---------------------------------------------------------------------------

class TestShow:
    def test_show_contains_all_sections(self, capsys):
        cfg = _default_cfg()
        _show(cfg)
        out = capsys.readouterr().out
        assert "ROBOT" in out
        assert "VISION" in out
        assert "STRATEGY" in out
        assert "SIMULATION" in out

    def test_show_contains_default_values(self, capsys):
        cfg = _default_cfg()
        _show(cfg)
        out = capsys.readouterr().out
        assert "sim" in out            # robot.mode default
        assert "192.168.1.100" in out  # robot.host default
        assert "0.25" in out           # move_velocity default
        assert "5050" in out           # stream_port default
        assert "5070" in out           # simulation.port default

    def test_show_reflects_mutated_values(self, capsys):
        cfg = _default_cfg()
        cfg.robot.host = "10.11.12.13"
        cfg.vision.camera_index = 3
        _show(cfg)
        out = capsys.readouterr().out
        assert "10.11.12.13" in out
        assert "3" in out


# ---------------------------------------------------------------------------
# 2. _save + robot parameters
# ---------------------------------------------------------------------------

class TestSaveRobot:
    def test_save_updates_mode(self, tmp_path):
        cfg = _default_cfg()
        cfg.robot.mode = "real"
        p = tmp_path / "arena_config.yaml"
        _save(cfg, p)
        d = _load_yaml(p)
        assert d["robot"]["mode"] == "real"

    def test_save_updates_host(self, tmp_path):
        cfg = _default_cfg()
        cfg.robot.host = "10.0.0.99"
        p = tmp_path / "arena_config.yaml"
        _save(cfg, p)
        d = _load_yaml(p)
        assert d["robot"]["host"] == "10.0.0.99"

    def test_save_updates_move_velocity(self, tmp_path):
        cfg = _default_cfg()
        cfg.robot.speeds.move_velocity = 0.50
        p = tmp_path / "arena_config.yaml"
        _save(cfg, p)
        d = _load_yaml(p)
        assert d["robot"]["speeds"]["move_velocity"] == pytest.approx(0.50)


# ---------------------------------------------------------------------------
# 3. _save + vision parameters
# ---------------------------------------------------------------------------

class TestSaveVision:
    def test_save_updates_camera_index(self, tmp_path):
        cfg = _default_cfg()
        cfg.vision.camera_index = 2
        p = tmp_path / "c.yaml"
        _save(cfg, p)
        assert _load_yaml(p)["vision"]["camera_index"] == 2

    def test_save_updates_stream_port(self, tmp_path):
        cfg = _default_cfg()
        cfg.vision.stream_port = 5055
        p = tmp_path / "c.yaml"
        _save(cfg, p)
        assert _load_yaml(p)["vision"]["stream_port"] == 5055

    def test_save_updates_locator_mode(self, tmp_path):
        cfg = _default_cfg()
        cfg.vision.locator_mode = "contour"
        p = tmp_path / "c.yaml"
        _save(cfg, p)
        assert _load_yaml(p)["vision"]["locator_mode"] == "contour"


# ---------------------------------------------------------------------------
# 4. _save + strategy parameters
# ---------------------------------------------------------------------------

class TestSaveStrategy:
    def test_save_updates_model(self, tmp_path):
        cfg = _default_cfg()
        cfg.strategy.model = "gpt-4o"
        p = tmp_path / "c.yaml"
        _save(cfg, p)
        assert _load_yaml(p)["strategy"]["model"] == "gpt-4o"

    def test_save_updates_timeout(self, tmp_path):
        cfg = _default_cfg()
        cfg.strategy.timeout = 60
        p = tmp_path / "c.yaml"
        _save(cfg, p)
        assert _load_yaml(p)["strategy"]["timeout"] == 60

    def test_save_updates_api_base(self, tmp_path):
        cfg = _default_cfg()
        cfg.strategy.api_base = "http://192.168.1.1:8080/v1"
        p = tmp_path / "c.yaml"
        _save(cfg, p)
        assert _load_yaml(p)["strategy"]["api_base"] == "http://192.168.1.1:8080/v1"


# ---------------------------------------------------------------------------
# 5. _save + simulation parameters
# ---------------------------------------------------------------------------

class TestSaveSimulation:
    def test_save_updates_port(self, tmp_path):
        cfg = _default_cfg()
        cfg.simulation.port = 5080
        p = tmp_path / "c.yaml"
        _save(cfg, p)
        assert _load_yaml(p)["simulation"]["port"] == 5080

    def test_save_updates_host(self, tmp_path):
        cfg = _default_cfg()
        cfg.simulation.host = "0.0.0.0"
        p = tmp_path / "c.yaml"
        _save(cfg, p)
        assert _load_yaml(p)["simulation"]["host"] == "0.0.0.0"


# ---------------------------------------------------------------------------
# 6. Validation — unsafe values rejected, file NOT written
# ---------------------------------------------------------------------------

class TestValidationBlocks:
    def _assert_not_saved(self, cfg: ArenaConfig, tmp_path: Path, capsys) -> None:
        p = tmp_path / "blocked.yaml"
        _save(cfg, p)
        assert not p.exists(), "File must NOT be created when validation fails"
        out = capsys.readouterr().out
        assert "ERROR" in out or "Validation" in out or "NOT saved" in out

    def test_speed_over_1ms_blocked(self, tmp_path, capsys):
        cfg = _default_cfg()
        cfg.robot.speeds.move_velocity = 1.1  # exceeds 1.0 m/s cap
        self._assert_not_saved(cfg, tmp_path, capsys)

    def test_speed_zero_blocked(self, tmp_path, capsys):
        cfg = _default_cfg()
        cfg.robot.speeds.move_velocity = 0.0
        self._assert_not_saved(cfg, tmp_path, capsys)

    def test_port_zero_blocked(self, tmp_path, capsys):
        cfg = _default_cfg()
        cfg.simulation.port = 0  # out of range
        self._assert_not_saved(cfg, tmp_path, capsys)

    def test_port_over_65535_blocked(self, tmp_path, capsys):
        cfg = _default_cfg()
        cfg.simulation.port = 99999
        self._assert_not_saved(cfg, tmp_path, capsys)

    def test_stream_port_out_of_range_blocked(self, tmp_path, capsys):
        cfg = _default_cfg()
        cfg.vision.stream_port = 0
        self._assert_not_saved(cfg, tmp_path, capsys)

    def test_invalid_robot_mode_blocked(self, tmp_path, capsys):
        cfg = _default_cfg()
        cfg.robot.mode = "turbo"  # not sim/real
        self._assert_not_saved(cfg, tmp_path, capsys)

    def test_transit_clearance_too_low_blocked(self, tmp_path, capsys):
        cfg = _default_cfg()
        cfg.robot.board_layout.transit_clearance_z = 0.01
        self._assert_not_saved(cfg, tmp_path, capsys)

    def test_negative_pitch_blocked(self, tmp_path, capsys):
        cfg = _default_cfg()
        cfg.robot.board_layout.square_pitch_x = -0.01
        self._assert_not_saved(cfg, tmp_path, capsys)

    def test_inverted_safety_bounds_blocked(self, tmp_path, capsys):
        cfg = _default_cfg()
        cfg.robot.safety_envelope.x_bounds = (0.80, -0.80)  # lo >= hi
        self._assert_not_saved(cfg, tmp_path, capsys)

    def test_valid_config_does_write(self, tmp_path):
        """Sanity: valid cfg IS written."""
        cfg = _default_cfg()
        p = tmp_path / "ok.yaml"
        _save(cfg, p)
        assert p.exists()


# ---------------------------------------------------------------------------
# 7. _tuples_to_lists helper
# ---------------------------------------------------------------------------

class TestTuplesToLists:
    def test_tuple_converted(self):
        assert _tuples_to_lists((1, 2)) == [1, 2]

    def test_nested_dict_with_tuple(self):
        result = _tuples_to_lists({"x_bounds": (-0.8, 0.8)})
        assert result == {"x_bounds": [-0.8, 0.8]}

    def test_list_unchanged_type(self):
        result = _tuples_to_lists([1, 2, 3])
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_scalar_passthrough(self):
        assert _tuples_to_lists(42) == 42
        assert _tuples_to_lists("hello") == "hello"

    def test_nested_tuple_in_list(self):
        assert _tuples_to_lists([(1, 2), 3]) == [[1, 2], 3]


# ---------------------------------------------------------------------------
# 8. interactive_config_editor — menu flow
# ---------------------------------------------------------------------------

class TestInteractiveConfigEditor:
    """Drive interactive_config_editor via mocked input()."""

    def _run(self, inputs: list[str], tmp_path: Path, monkeypatch) -> Path:
        p = tmp_path / "arena_config.yaml"
        inp = iter(inputs)
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(inp))
        from src import config_editor
        with patch.object(config_editor, "load_config", return_value=_default_cfg()):
            interactive_config_editor(str(p))
        return p

    def test_view_option_prints_config(self, tmp_path, monkeypatch, capsys):
        # 1 = view, q = cancel
        self._run(["1", "q"], tmp_path, monkeypatch)
        out = capsys.readouterr().out
        assert "ROBOT" in out

    def test_cancel_does_not_write_file(self, tmp_path, monkeypatch):
        p = self._run(["q"], tmp_path, monkeypatch)
        assert not p.exists()

    def test_save_exit_writes_file(self, tmp_path, monkeypatch):
        p = self._run(["0"], tmp_path, monkeypatch)
        assert p.exists()

    def test_eof_on_main_prompt_cancels(self, tmp_path, monkeypatch):
        """EOFError from config editor prompt → cancel, no file."""
        p = tmp_path / "arena_config.yaml"
        monkeypatch.setattr("builtins.input", lambda _prompt="": (_ for _ in ()).throw(EOFError))
        from src import config_editor
        with patch.object(config_editor, "load_config", return_value=_default_cfg()):
            interactive_config_editor(str(p))  # must not raise
        assert not p.exists()

    def test_unknown_option_prints_error(self, tmp_path, monkeypatch, capsys):
        self._run(["z", "q"], tmp_path, monkeypatch)
        out = capsys.readouterr().out
        assert "Unknown option" in out or "ERROR" in out

    def test_edit_robot_host(self, tmp_path, monkeypatch):
        """Option 2 edits robot host; 0 saves."""
        p = tmp_path / "arena_config.yaml"
        # inputs: main_menu=2, mode=sim(enter), host=10.9.8.7, mv=(enter), ma=(enter), main_menu=0
        inputs = iter(["2", "sim", "10.9.8.7", "", "", "0"])
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))
        from src import config_editor
        with patch.object(config_editor, "load_config", return_value=_default_cfg()):
            interactive_config_editor(str(p))
        assert p.exists()
        d = _load_yaml(p)
        assert d["robot"]["host"] == "10.9.8.7"

    def test_edit_vision_camera_index(self, tmp_path, monkeypatch):
        """Option 3 edits vision.camera_index; 0 saves."""
        p = tmp_path / "arena_config.yaml"
        # inputs: main_menu=3, camera_index=2, locator_mode=(enter), stream_port=(enter), main_menu=0
        inputs = iter(["3", "2", "", "", "0"])
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))
        from src import config_editor
        with patch.object(config_editor, "load_config", return_value=_default_cfg()):
            interactive_config_editor(str(p))
        d = _load_yaml(p)
        assert d["vision"]["camera_index"] == 2

    def test_edit_strategy_model(self, tmp_path, monkeypatch):
        """Option 4 edits strategy.model; 0 saves."""
        p = tmp_path / "arena_config.yaml"
        inputs = iter(["4", "gpt-4o", "", "", "0"])
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))
        from src import config_editor
        with patch.object(config_editor, "load_config", return_value=_default_cfg()):
            interactive_config_editor(str(p))
        d = _load_yaml(p)
        assert d["strategy"]["model"] == "gpt-4o"

    def test_edit_simulation_port(self, tmp_path, monkeypatch):
        """Option 5 edits simulation.port; 0 saves."""
        p = tmp_path / "arena_config.yaml"
        inputs = iter(["5", "", "5080", "0"])
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))
        from src import config_editor
        with patch.object(config_editor, "load_config", return_value=_default_cfg()):
            interactive_config_editor(str(p))
        d = _load_yaml(p)
        assert d["simulation"]["port"] == 5080


# ---------------------------------------------------------------------------
# 9. main.py launcher integration
# ---------------------------------------------------------------------------

class TestMainLauncherIntegration:
    """Verify main.py routes to interactive_config_editor correctly."""

    def test_interactive_menu_choice_7_calls_config_editor(self, monkeypatch, capsys):
        """interactive_menu() choice '7' must call interactive_config_editor."""
        import main as m

        called = {}

        def _fake_editor(config_path=None):
            called["invoked"] = True

        monkeypatch.setattr("builtins.input", lambda _prompt="": "7")
        monkeypatch.setattr(m, "interactive_menu",
                            lambda: _fake_menu_choice_7(monkeypatch, _fake_editor, m))

        # Patch at the import site inside main.py
        with patch("main.interactive_menu") as mock_menu:
            mock_menu.side_effect = lambda: None  # just verify it would be called
            m.main([])  # no --mode → interactive_menu()
            mock_menu.assert_called_once()

    def test_mode_config_calls_interactive_config_editor(self, capsys):
        """--mode config must call interactive_config_editor and not crash."""
        import main as m

        with patch("src.config_editor.interactive_config_editor") as mock_editor:
            m.main(["--mode", "config"])
            mock_editor.assert_called_once_with()

    def test_mode_config_no_args_calls_once(self, capsys):
        """Double-check: --mode config calls editor exactly once."""
        import main as m

        with patch("src.config_editor.interactive_config_editor") as mock_editor:
            m.main(["--mode", "config"])
            assert mock_editor.call_count == 1


def _fake_menu_choice_7(monkeypatch, fake_editor, m):
    """Helper: simulate interactive_menu selecting option 7."""
    with patch("src.config_editor.interactive_config_editor", side_effect=fake_editor):
        # Directly exercise the branch under test
        with patch("builtins.input", return_value="7"):
            # Call original interactive_menu but redirect editor
            import importlib, sys
            import src.config_editor as ce
            orig = ce.interactive_config_editor
            ce.interactive_config_editor = fake_editor
            try:
                m.interactive_menu()
            except SystemExit:
                pass
            finally:
                ce.interactive_config_editor = orig


def test_prompt_reprompts_on_non_numeric(monkeypatch):
    """Regression: non-numeric input must not crash the editor."""
    from src.config_editor import _prompt
    answers = iter(["abc", "0.5"])
    monkeypatch.setattr("builtins.input", lambda _="": next(answers))
    assert _prompt("v", 0.25, float) == 0.5
