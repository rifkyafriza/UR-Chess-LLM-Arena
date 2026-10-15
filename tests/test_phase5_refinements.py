"""Phase 5 refinements test suite.

Coverage:
  1. Benchmark script execution (subprocess, no ModuleNotFoundError)
  2. main.py interactive menu option '6' and non-interactive arg parsing
  3. Default port 5070 migration (no 5060 as default anywhere)
  4. Live stream dual-pane: /state returns svg+fen, /board_svg returns SVG
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import re
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_main_module(module_name: str = "main_mod"):
    """Load main.py with heavy src.* stubs to avoid real server imports."""
    spec = importlib.util.spec_from_file_location(module_name, REPO / "main.py")
    mod = importlib.util.module_from_spec(spec)
    stub_names = (
        "src.simulation.web_app",
        "src.vision.live_stream",
        "src.arena.master_orchestrator",
        "src.config",
    )
    saved = {k: sys.modules.get(k) for k in stub_names}
    for name in stub_names:
        sys.modules.setdefault(name, types.ModuleType(name))
    try:
        spec.loader.exec_module(mod)
    finally:
        # Restore originals so other tests can import the real modules
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    return mod


# ---------------------------------------------------------------------------
# 1. Benchmark script
# ---------------------------------------------------------------------------

class TestBenchmarkScript(unittest.TestCase):
    """benchmark_board.py must not raise ModuleNotFoundError regardless of cwd."""

    def _run(self, cwd: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(REPO / "scripts" / "benchmark_board.py")],
            capture_output=True,
            text=True,
            cwd=str(cwd),
        )

    def test_exit_code_from_repo_root(self):
        result = self._run(REPO)
        self.assertNotIn("ModuleNotFoundError", result.stderr,
                         msg=f"ModuleNotFoundError in stderr:\n{result.stderr}")
        self.assertEqual(result.returncode, 0,
                         msg=f"Non-zero exit:\nstdout={result.stdout}\nstderr={result.stderr}")

    def test_no_module_not_found_from_outside_repo(self):
        """Script self-inserts repo root; ModuleNotFoundError must not appear even from /tmp."""
        result = self._run(Path("/tmp"))
        self.assertNotIn("ModuleNotFoundError", result.stderr,
                         msg=f"ModuleNotFoundError in stderr:\n{result.stderr}")
        # May fail with asset read error (relative path) but NOT import error
        self.assertNotIn("ModuleNotFoundError", result.stdout)

    def test_no_import_error_from_repo_root(self):
        result = self._run(REPO)
        self.assertNotIn("ImportError", result.stderr)


# ---------------------------------------------------------------------------
# 2. main.py interactive menu option '6'
# ---------------------------------------------------------------------------

class TestInteractiveMenuOption6(unittest.TestCase):

    def test_option6_sim_calls_run_arena(self):
        mod = _load_main_module("main_menu_sim")
        with patch.object(mod, "run_arena") as mock_arena, \
             patch("builtins.input", side_effect=["6", "1"]):
            try:
                mod.interactive_menu()
            except SystemExit:
                pass
        mock_arena.assert_called_once()
        self.assertEqual(mock_arena.call_args[0][0], "sim")

    def test_option6_real_calls_run_arena(self):
        mod = _load_main_module("main_menu_real")
        with patch.object(mod, "run_arena") as mock_arena, \
             patch("builtins.input", side_effect=["6", "2"]):
            try:
                mod.interactive_menu()
            except SystemExit:
                pass
        mock_arena.assert_called_once()
        self.assertEqual(mock_arena.call_args[0][0], "real")

    def test_option6_unexpected_input_defaults_to_sim(self):
        """Any robot_choice other than '2' → sim."""
        mod = _load_main_module("main_menu_default")
        with patch.object(mod, "run_arena") as mock_arena, \
             patch("builtins.input", side_effect=["6", "X"]):
            try:
                mod.interactive_menu()
            except SystemExit:
                pass
        mock_arena.assert_called_once()
        self.assertEqual(mock_arena.call_args[0][0], "sim")


# ---------------------------------------------------------------------------
# 3. Non-interactive arg parsing
# ---------------------------------------------------------------------------

class TestArgParserArenaMode(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.parser = _load_main_module("main_parser").build_parser()

    def test_mode_arena_robot_sim(self):
        args = self.parser.parse_args(["--mode", "arena", "--robot-mode", "sim"])
        self.assertEqual(args.mode, "arena")
        self.assertEqual(args.robot_mode, "sim")

    def test_mode_arena_robot_real(self):
        args = self.parser.parse_args(["--mode", "arena", "--robot-mode", "real"])
        self.assertEqual(args.mode, "arena")
        self.assertEqual(args.robot_mode, "real")

    def test_mode_arena_default_robot_mode(self):
        """If --robot-mode omitted, value must be 'sim' or None (resolved to sim in main)."""
        args = self.parser.parse_args(["--mode", "arena"])
        self.assertIn(args.robot_mode, ("sim", None))


# ---------------------------------------------------------------------------
# 4. Default port 5070 — config layer
# ---------------------------------------------------------------------------

class TestDefaultPort5070(unittest.TestCase):

    def test_arena_config_simulation_port_default(self):
        # Fresh import — do not rely on any stub
        from src.config import ArenaConfig  # noqa: PLC0415
        cfg = ArenaConfig()
        self.assertEqual(cfg.simulation.port, 5070,
                         f"ArenaConfig().simulation.port expected 5070, got {cfg.simulation.port}")

    def test_arena_config_yaml_contains_5070(self):
        yaml_path = REPO / "config" / "arena_config.yaml"
        self.assertTrue(yaml_path.exists(), f"Missing {yaml_path}")
        text = yaml_path.read_text()
        self.assertIn("5070", text)

    def test_arena_config_yaml_no_5060_as_port(self):
        yaml_path = REPO / "config" / "arena_config.yaml"
        text = yaml_path.read_text()
        bad = re.search(r"(?:port|sim_port)\s*:\s*5060\b", text)
        self.assertIsNone(bad, f"5060 found as a port default in {yaml_path}")

    def test_main_defaults_sim_port_5070(self):
        mod = _load_main_module("main_port_check")
        self.assertEqual(mod._DEFAULTS["sim_port"], 5070)

    def test_web_app_run_default_port_5070(self):
        from src.simulation import web_app  # noqa: PLC0415
        self.assertTrue(hasattr(web_app, "run"),
                        "src.simulation.web_app has no 'run' function")
        sig = inspect.signature(web_app.run)
        port_param = sig.parameters.get("port")
        self.assertIsNotNone(port_param, "web_app.run() has no 'port' parameter")
        self.assertEqual(port_param.default, 5070)

    def test_port_5060_not_in_config_py(self):
        config_text = (REPO / "src" / "config.py").read_text()
        bad = re.search(r"\b5060\b", config_text)
        self.assertIsNone(bad, "Port 5060 found in src/config.py — must not be default")

    def test_port_5060_not_in_arena_config_yaml(self):
        yaml_path = REPO / "config" / "arena_config.yaml"
        text = yaml_path.read_text()
        bad = re.search(r"\b5060\b", text)
        self.assertIsNone(bad, f"Port 5060 found in {yaml_path}")


# ---------------------------------------------------------------------------
# 5. Live stream dual-pane endpoints
# ---------------------------------------------------------------------------

class TestLiveStreamDualPane(unittest.TestCase):
    """/state must include svg+fen keys; /board_svg must return valid SVG."""

    @classmethod
    def setUpClass(cls):
        # Remove any stub that may have been left over, then import the real module
        sys.modules.pop("src.vision.live_stream", None)
        from src.vision import live_stream  # noqa: PLC0415
        cls.client = live_stream.app.test_client()

    def test_state_has_svg_key(self):
        rv = self.client.get("/state")
        self.assertEqual(rv.status_code, 200)
        data = rv.get_json()
        self.assertIn("svg", data, f"/state JSON missing 'svg': {data}")

    def test_state_has_fen_key(self):
        rv = self.client.get("/state")
        data = rv.get_json()
        self.assertIn("fen", data, f"/state JSON missing 'fen': {data}")

    def test_state_returns_json(self):
        rv = self.client.get("/state")
        self.assertIn("application/json", rv.content_type)

    def test_board_svg_endpoint_exists(self):
        rv = self.client.get("/board_svg")
        self.assertNotEqual(rv.status_code, 404,
                            "/board_svg endpoint not registered (404)")

    def test_board_svg_returns_svg_body(self):
        rv = self.client.get("/board_svg")
        self.assertEqual(rv.status_code, 200)
        body = rv.data.decode("utf-8", errors="replace")
        self.assertIn("<svg", body,
                      f"/board_svg body does not contain <svg: {body[:200]}")

    def test_board_svg_content_type_is_svg(self):
        rv = self.client.get("/board_svg")
        self.assertIn("svg", rv.content_type,
                      f"Expected svg mimetype, got: {rv.content_type}")


if __name__ == "__main__":
    unittest.main()
