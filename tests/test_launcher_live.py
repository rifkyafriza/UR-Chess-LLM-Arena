"""Unit tests for run_live() in main.py."""

from __future__ import annotations

from unittest.mock import patch, MagicMock
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_run_live():
    """Import run_live fresh each test (avoids module-level import side effects)."""
    import importlib
    import main as m
    importlib.reload(m)
    return m.run_live


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRunLive:
    """run_live(port, camera) must call live_stream.start with correct args."""

    def test_calls_start_with_default_args(self):
        """run_live passes camera_index=0 and port=5050 to start()."""
        mock_start = MagicMock()
        mock_module = MagicMock()
        mock_module.start = mock_start

        import sys
        import importlib

        with patch.dict("sys.modules", {"src.vision.live_stream": mock_module}):
            import main
            importlib.reload(main)
            with patch.object(main, "_port_busy", return_value=False):
                main.run_live(port=5050, camera=0)

        mock_start.assert_called_once_with(camera_index=0, port=5050)

    def test_calls_start_custom_camera_and_port(self):
        """run_live forwards custom camera index and port to start()."""
        mock_start = MagicMock()
        mock_module = MagicMock()
        mock_module.start = mock_start

        import sys
        import importlib

        with patch.dict("sys.modules", {"src.vision.live_stream": mock_module}):
            import main
            importlib.reload(main)
            with patch.object(main, "_port_busy", return_value=False):
                main.run_live(port=6060, camera=2)

        mock_start.assert_called_once_with(camera_index=2, port=6060)

    def test_exits_when_port_busy(self):
        """run_live exits with error when port already in use."""
        import main
        with patch.object(main, "_port_busy", return_value=True):
            with pytest.raises(SystemExit) as exc_info:
                main.run_live(port=5050, camera=0)
        assert exc_info.value.code != 0 or "[ERROR]" in str(exc_info.value.code)

    def test_no_vision_app_reference(self):
        """run_live source must not contain 'vision_app' (regression guard)."""
        import inspect
        import main
        source = inspect.getsource(main.run_live)
        assert "vision_app" not in source, "vision_app NameError regression detected in run_live()"

    def test_interactive_menu_option2_passes_defaults(self):
        """Menu option [2] calls run_live with live_port=5050, camera=0."""
        import main
        import importlib
        importlib.reload(main)

        with (
            patch("builtins.input", return_value="2"),
            patch.object(main, "run_live") as mock_run_live,
        ):
            try:
                main.interactive_menu()
            except SystemExit:
                pass

        mock_run_live.assert_called_once_with(
            main._DEFAULTS["live_port"],
            main._DEFAULTS["camera"],
        )
